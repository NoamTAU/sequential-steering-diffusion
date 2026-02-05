"""
Performs sequential forward-backward experiments (U-turns) on a single starting image
to generate trajectories of images and their embeddings.

Features:
- Robust Continuation: Automatically detects the last U-turn and continues.
- Incremental Saving: Saves .npz after every step to prevent data loss.
- Slurm Array Support: targeted execution via --trajectory_idx.
"""

import argparse
import os
import sys
import glob
import numpy as np
import torch as th
from PIL import Image
from torchvision.transforms import Normalize, Resize

# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import clip
from guided_diffusion import logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def perform_single_uturn(model, diffusion, start_tensor, noise_step, device):
    """
    Takes a single image tensor and performs one forward-backward U-turn.
    Manually implements the loop to avoid AttributeError on missing library methods.
    """
    with th.no_grad():
        start_tensor = start_tensor.to(device)
        
        # 1. Forward Process: Add noise
        t_batch = th.tensor([noise_step] * start_tensor.shape[0], device=device)
        noisy_image = diffusion.q_sample(start_tensor, t_batch)
        
        # 2. Backward Process: Denoise step-by-step
        # This manual loop replaces diffusion.p_sample_loop_forw_back
        img = noisy_image
        indices = list(range(noise_step))[::-1]
        
        for i in indices:
            t = th.tensor([i] * img.shape[0], device=device)
            out = diffusion.p_sample(
                model=model,
                x=img,
                t=t,
                clip_denoised=True,
                model_kwargs={}
            )
            img = out["sample"]
            
        return img

def get_clip_patch_embeddings(visual_model, images, clip_normalize, clip_resize):
    """
    Extracts 7x7 patch embeddings from a batch of image tensors.
    """
    with th.no_grad():
        resized_images = clip_resize(images)
        normalized_images = clip_normalize(resized_images).half()
        
        x = normalized_images
        x = visual_model.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        
        class_embedding = visual_model.class_embedding.to(x.dtype)
        x = th.cat([class_embedding + th.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        
        x = x + visual_model.positional_embedding.to(x.dtype)
        x = visual_model.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual_model.transformer(x)
        x = x.permute(1, 0, 2)
        
        return x[:, 1:, :]

def run_single_trajectory(args, model, diffusion, clip_model, device,
                          start_tensor, trajectory_dir):
    """
    Runs or continues a single sequence of U-turns and saves the results.
    """
    clip_normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    clip_resize = Resize([224, 224])

    # --- Smart Detection of Last Step ---
    # Robustly find all existing images (jpeg or png)
    jpeg_paths = glob.glob(os.path.join(trajectory_dir, "uturn_*.jpeg"))
    png_paths = glob.glob(os.path.join(trajectory_dir, "uturn_*.png"))
    existing_images = sorted(jpeg_paths + png_paths)
    
    if not existing_images:
        # --- This is a brand new trajectory ---
        logger.log("No existing trajectory found. Starting from scratch.")
        start_step = 0
        current_image_tensor = start_tensor
        all_embeddings = [] 

        # Save the initial state (step 0)
        logger.log("Saving initial state (U-turn 0)...")
        img_to_save = ((current_image_tensor[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img_to_save).save(os.path.join(trajectory_dir, "uturn_000.jpeg"))
        
        embedding = get_clip_patch_embeddings(clip_model.visual, current_image_tensor, clip_normalize, clip_resize)
        all_embeddings.append(embedding.cpu().numpy())
    else:
        # --- This is a continuation ---
        # Find the file with the highest index
        max_idx = -1
        last_image_path = ""
        for p in existing_images:
            try:
                # robustly parse filename "uturn_123.jpeg"
                idx = int(os.path.splitext(os.path.basename(p))[0].split('_')[-1])
                if idx > max_idx:
                    max_idx = idx
                    last_image_path = p
            except ValueError:
                continue
        
        start_step = max_idx
        logger.log(f"Found existing trajectory. Last step was {start_step}. Continuing from here.")
        
        # Load the last image to continue from
        last_image_pil = Image.open(last_image_path).convert("RGB")
        
        # Ensure it's 256x256 (crucial for U-Net stability)
        diffusion_resize = Resize([args.image_size, args.image_size], interpolation=Image.BICUBIC)
        last_image_pil = diffusion_resize(last_image_pil)
        
        current_image_tensor = th.tensor(np.array(last_image_pil)).float() / 127.5 - 1
        current_image_tensor = current_image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Load previous embeddings
        npz_path = os.path.join(trajectory_dir, "trajectory_data.npz")
        if os.path.exists(npz_path):
            # Load only up to the start_step (inclusive) to ensure sync
            all_embeddings = list(np.load(npz_path)['embeddings'][:start_step + 1])
            logger.log(f"Loaded {len(all_embeddings)} previous embeddings.")
        else:
            logger.log("Warning: .npz file not found. History lost, but continuing image generation.")
            all_embeddings = []

    # --- Sequential U-turn Loop ---
    for i in range(args.num_uturns):
        uturn_idx = start_step + 1 + i
        logger.log(f"Performing U-turn {uturn_idx}...")
        
        current_image_tensor = perform_single_uturn(model, diffusion, current_image_tensor, args.noise_step, device)

        img_to_save = ((current_image_tensor[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img_to_save).save(os.path.join(trajectory_dir, f"uturn_{uturn_idx:03d}.jpeg"))
        
        embedding = get_clip_patch_embeddings(clip_model.visual, current_image_tensor, clip_normalize, clip_resize)
        all_embeddings.append(embedding.cpu().numpy())

        # --- INCREMENTAL SAVING ---
        # Save after every step to prevent data loss on timeout
        npz_path = os.path.join(trajectory_dir, "trajectory_data.npz")
        np.savez(
            npz_path,
            embeddings=np.concatenate(all_embeddings, axis=0),
            start_image_path=args.start_image_path
        )

def main():
    args = create_argparser().parse_args()
    
    # Fix argparse string->int conversion if passed
    if args.trajectory_idx is not None:
        args.trajectory_idx = int(args.trajectory_idx)

    # --- Setup Models and Device ---
    device = "cuda" if th.cuda.is_available() else "cpu"
    logger.log(f"Using device: {device}")

    logger.log("Creating diffusion model...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model.to(device)
    if args.use_fp16: model.convert_to_fp16()
    model.eval()

    logger.log("Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # --- Prepare Starting Image (Global Resize Fix) ---
    logger.log(f"Loading starting image from: {args.start_image_path}")
    start_image_pil = Image.open(args.start_image_path).convert("RGB")
    
    # Resize to 256x256 to prevent U-Net crashes on arbitrary image sizes
    diffusion_resize = Resize([args.image_size, args.image_size], interpolation=Image.BICUBIC)
    start_image_pil = diffusion_resize(start_image_pil)
    
    start_image_tensor = th.tensor(np.array(start_image_pil)).float() / 127.5 - 1
    start_image_tensor = start_image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    base_image_name = os.path.splitext(os.path.basename(args.start_image_path))[0]
    output_base = os.path.join(args.output_dir, base_image_name, f"noise_step_{args.noise_step}")
    os.makedirs(output_base, exist_ok=True)
    logger.configure(dir=output_base)

    # --- Execution Loop ---
    if args.trajectory_idx is not None:
        trajectories_to_run = [args.trajectory_idx]
        logger.log(f"--- Running for single Trajectory index: {args.trajectory_idx} ---")
    else:
        trajectories_to_run = range(args.num_trajectories)
        logger.log(f"--- Starting New Experiment for {args.num_trajectories} trajectories ---")

    for traj_idx in trajectories_to_run:
        logger.log(f"--- Processing Trajectory index {traj_idx} ---")
        trajectory_dir = os.path.join(output_base, f"trajectory_{traj_idx:03d}")
        os.makedirs(trajectory_dir, exist_ok=True)
        
        run_single_trajectory(args, model, diffusion, clip_model, device,
                              start_tensor=start_image_tensor,
                              trajectory_dir=trajectory_dir)

def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        start_image_path="", 
        output_dir="/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns",
        num_uturns=100,      
        noise_step=150,
        num_trajectories=50,
        trajectory_idx=None,
        clip_denoised=True,
        use_ddim=False,
        model_path="/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/models/256x256_diffusion_uncond.pt",
        
        # --- CORRECT CONFIGURATION FOR 1000 STEP MODEL ---
        image_size=256,
        class_cond=False,
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        attention_resolutions="32,16,8",
        channel_mult="",
        use_checkpoint=False,
        use_fp16=True,
        num_head_channels=64,
        resblock_updown=True,
        use_new_attention_order=False,
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()