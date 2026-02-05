"""
Performs sequential forward-backward experiments (U-turns) on a single starting image
to generate trajectories of images and their embeddings.
"""

import argparse
import os
import sys
import datetime
import pickle

# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import clip
import numpy as np
import torch as th
from PIL import Image
from torchvision.transforms import Normalize, Resize
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import imageio # For creating GIFs


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
    """
    with th.no_grad():
        # Ensure input tensor is on the correct device
        start_tensor = start_tensor.to(device)
        
        # Forward process: Add noise
        t_reverse = diffusion._scale_timesteps(th.tensor([noise_step])).to(device)
        noisy_image = diffusion.q_sample(start_tensor, t_reverse)
        
        # Backward process: Denoise the image
        model_kwargs = {}
        sample_fn = diffusion.p_sample_loop_forw_back
        
        denoised_image = sample_fn(
            model,
            (1, 3, model.image_size, model.image_size), # Batch size is always 1
            step_reverse=noise_step,
            noise=noisy_image,
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )
        return denoised_image

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
        
        # Return only the patch tokens (shape: [batch_size, 49, embed_dim])
        return x[:, 1:, :]

def run_single_trajectory(args, model, diffusion, clip_model, device,
                          start_tensor, trajectory_dir):
    """
    Runs or continues a single sequence of U-turns and saves the results.
    It automatically detects the last step and continues from there.
    """
    clip_normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    clip_resize = Resize([224, 224])

    # --- CORRECTED: Smart Detection of Last Step ---
    # Robustly find all existing images (jpeg or png)
    existing_images = sorted(glob.glob(os.path.join(trajectory_dir, "uturn_*.[jp][pn]g")))
    
    if not existing_images:
        # --- This is a brand new trajectory ---
        logger.log("No existing trajectory found. Starting from scratch.")
        start_step = 0
        current_image_tensor = start_tensor
        previous_embeddings = []

        # Save the initial state (step 0)
        logger.log("Saving initial state (U-turn 0)...")
        img_to_save = ((current_image_tensor[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img_to_save).save(os.path.join(trajectory_dir, "uturn_000.jpeg"))
        
        embedding = get_clip_patch_embeddings(clip_model.visual, current_image_tensor, clip_normalize, clip_resize)
        previous_embeddings.append(embedding.cpu().numpy())
    else:
        # --- This is a continuation ---
        last_image_path = existing_images[-1]
        start_step = int(os.path.splitext(os.path.basename(last_image_path))[0].split('_')[-1])
        
        logger.log(f"Found existing trajectory. Last step was {start_step}. Continuing from here.")
        
        # Load the last image to continue from
        last_image_pil = Image.open(last_image_path).convert("RGB")
        current_image_tensor = th.tensor(np.array(last_image_pil)).float() / 127.5 - 1
        current_image_tensor = current_image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Load previous embeddings from the .npz file
        npz_path = os.path.join(trajectory_dir, "trajectory_data.npz")
        previous_embeddings = []
        if os.path.exists(npz_path):
            # Load only up to the steps we have images for, to be safe
            previous_embeddings = list(np.load(npz_path)['embeddings'][:start_step + 1])
            logger.log(f"Loaded {len(previous_embeddings)} previous embeddings.")
        else:
            logger.log("Warning: .npz file not found. It will be recreated.")

    new_embeddings_list = []

    # --- Sequential U-turn Loop (now starts from the correct index) ---
    for i in range(args.num_uturns):
        uturn_idx = start_step + 1 + i
        logger.log(f"Performing U-turn {uturn_idx}...")
        
        current_image_tensor = perform_single_uturn(model, diffusion, current_image_tensor, args.noise_step, device)

        img_to_save = ((current_image_tensor[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img_to_save).save(os.path.join(trajectory_dir, f"uturn_{uturn_idx:03d}.jpeg"))
        
        embedding = get_clip_patch_embeddings(clip_model.visual, current_image_tensor, clip_normalize, clip_resize)
        new_embeddings_list.append(embedding.cpu().numpy())

    # --- Save Trajectory Data ---
    all_embeddings = np.concatenate(previous_embeddings + new_embeddings_list, axis=0)
        
    npz_path = os.path.join(trajectory_dir, "trajectory_data.npz")
    np.savez(
        npz_path,
        embeddings=all_embeddings,
        start_image_path=args.start_image_path
    )
    logger.log(f"Saved/updated trajectory data with {len(all_embeddings)} steps to {npz_path}")
    


def main():
    args = create_argparser().parse_args()

    # --- 1. Setup Models and Device ---
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
    
    # --- 2. LOGIC FORK: New Experiment vs. Continuation ---
    if args.continue_from:
        # --- CONTINUATION MODE ---
        trajectory_dir = args.continue_from
        logger.log(f"--- Continuing Trajectory from: {trajectory_dir} ---")
        if not os.path.isdir(trajectory_dir):
            raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

        # Load previous embeddings and find the start step
        npz_path = os.path.join(trajectory_dir, "trajectory_data.npz")
        previous_data = np.load(npz_path)
        previous_embeddings = previous_data['embeddings']
        start_step = len(previous_embeddings) - 1
        
        # Find and load the last image
        last_image_path = os.path.join(trajectory_dir, f"uturn_{start_step:03d}.jpeg")
        start_image_pil = Image.open(last_image_path).convert("RGB")
        
        logger.log(f"Found {start_step + 1} existing steps. Starting from U-turn {start_step + 1}.")
        
        # Convert PIL image to the required tensor format
        start_image_tensor = th.tensor(np.array(start_image_pil)).float() / 127.5 - 1
        start_image_tensor = start_image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Run a single trajectory to extend the data
        run_single_trajectory(args, model, diffusion, clip_model, device,
                              start_tensor=start_image_tensor,
                              trajectory_dir=trajectory_dir,
                              start_step=start_step,
                              previous_embeddings=previous_embeddings)

    else:
        # --- NEW EXPERIMENT MODE (as before) ---
        logger.log(f"--- Starting New Experiment for image: {args.start_image_path} ---")
        diffusion_resize = Resize([args.image_size, args.image_size], interpolation=Image.BICUBIC)
        start_image_pil = diffusion_resize(Image.open(args.start_image_path).convert("RGB"))
        start_image_tensor = th.tensor(np.array(start_image_pil)).float() / 127.5 - 1
        start_image_tensor = start_image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        base_image_name = os.path.splitext(os.path.basename(args.start_image_path))[0]
        output_base = os.path.join(args.output_dir, base_image_name, f"noise_step_{args.noise_step}")
        os.makedirs(output_base, exist_ok=True)
        logger.configure(dir=output_base)
        
        for traj_idx in range(args.num_trajectories):
            logger.log(f"--- Starting Trajectory {traj_idx + 1}/{args.num_trajectories} ---")
            trajectory_dir = os.path.join(output_base, f"trajectory_{traj_idx:03d}")
            os.makedirs(trajectory_dir, exist_ok=True)
            run_single_trajectory(args, model, diffusion, clip_model, device,
                                  start_tensor=start_image_tensor,
                                  trajectory_dir=trajectory_dir)


def main():
    args = create_argparser().parse_args()
    
    # --- THIS IS THE FIX ---
    # argparse reads command-line args as strings. We must convert it to an integer.
    if args.trajectory_idx is not None:
        args.trajectory_idx = int(args.trajectory_idx)
    # --- END OF FIX ---

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
    
    # Prepare the starting image tensor once
    diffusion_resize = Resize([args.image_size, args.image_size], interpolation=Image.BICUBIC)
    start_image_pil = diffusion_resize(Image.open(args.start_image_path).convert("RGB"))
    start_image_tensor = th.tensor(np.array(start_image_pil)).float() / 127.5 - 1
    start_image_tensor = start_image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    base_image_name = os.path.splitext(os.path.basename(args.start_image_path))[0]
    output_base = os.path.join(args.output_dir, base_image_name, f"noise_step_{args.noise_step}")
    os.makedirs(output_base, exist_ok=True)
    logger.configure(dir=output_base)

    # This logic is now correct because args.trajectory_idx is an integer
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
        num_uturns=100,      # This is the number of *new* u-turns to add
        noise_step=150,
        num_trajectories=5,
        trajectory_idx=None, # NEW: Argument to target a single trajectory

        # continue_from=None, # NEW: The key argument for continuation
        
        clip_denoised=True,
        use_ddim=False,
        model_path="/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/models/256x256_diffusion_uncond.pt", # MODIFIED: Path to the downloaded model
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






# def main():
#     args = create_argparser().parse_args()

#     # --- 1. Setup Models and Device ---
#     device = "cuda" if th.cuda.is_available() else "cpu"
#     logger.log(f"Using device: {device}")

#     logger.log("Creating diffusion model...")
#     model, diffusion = create_model_and_diffusion(
#         **args_to_dict(args, model_and_diffusion_defaults().keys())
#     )
#     model.load_state_dict(th.load(args.model_path, map_location="cpu"))
#     model.to(device)
#     if args.use_fp16:
#         model.convert_to_fp16()
#     model.eval()

#     logger.log("Loading CLIP model...")
#     clip_model, _ = clip.load("ViT-B/32", device=device)
#     clip_model.eval()
    
#     clip_normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
#     clip_resize = Resize([224, 224])

#     # --- 2. Prepare Starting Image ---
#     logger.log(f"Loading starting image from: {args.start_image_path}")
#     start_image_pil = Image.open(args.start_image_path).convert("RGB")

#     # --- THIS IS THE FIX ---
#     # Create a resize transform for the diffusion model's input size
#     diffusion_resize = Resize([args.image_size, args.image_size], interpolation=Image.BICUBIC)
#     # Apply the resize to ensure the image is exactly the size the model expects
#     start_image_pil = diffusion_resize(start_image_pil)
#     # --- END OF FIX ---

#     # Preprocess the now-resized image to a tensor in the [-1, 1] range
#     start_image_tensor = th.tensor(np.array(start_image_pil)).float() / 127.5 - 1
#     start_image_tensor = start_image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

#     # --- 3. Setup Output Directory ---
#     base_image_name = os.path.splitext(os.path.basename(args.start_image_path))[0]
#     output_base = os.path.join(args.output_dir, base_image_name, f"noise_step_{args.noise_step}")
#     os.makedirs(output_base, exist_ok=True)
#     logger.configure(dir=output_base)
#     logger.log(f"Saving results to: {output_base}")
    
#     # --- 4. Main Experiment Loop ---
#     # (The rest of the main function is correct and remains the same)
#     for traj_idx in range(args.num_trajectories):
#         logger.log(f"--- Starting Trajectory {traj_idx + 1}/{args.num_trajectories} ---")
        
#         trajectory_dir = os.path.join(output_base, f"trajectory_{traj_idx:03d}")
#         os.makedirs(trajectory_dir, exist_ok=True)

#         current_image_tensor = start_image_tensor
#         trajectory_embeddings = []

#         # --- U-turn Step 0 (The Original Image) ---
#         # Save image
#         img_to_save = ((current_image_tensor[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
#         Image.fromarray(img_to_save).save(os.path.join(trajectory_dir, "uturn_000.png"))
        
#         # Get and store embedding
#         embedding = get_clip_patch_embeddings(clip_model.visual, current_image_tensor, clip_normalize, clip_resize)
#         trajectory_embeddings.append(embedding.cpu().numpy())
        
#         # --- Sequential U-turn Loop ---
#         for uturn_idx in range(1, args.num_uturns + 1):
#             logger.log(f"Performing U-turn {uturn_idx}/{args.num_uturns}...")
            
#             # Get the next image in the sequence
#             current_image_tensor = perform_single_uturn(
#                 model, diffusion, current_image_tensor, args.noise_step, device
#             )

#             # Save the new image
#             img_to_save = ((current_image_tensor[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
#             Image.fromarray(img_to_save).save(os.path.join(trajectory_dir, f"uturn_{uturn_idx:03d}.png"))
            
#             # Get and store the new embedding
#             embedding = get_clip_patch_embeddings(clip_model.visual, current_image_tensor, clip_normalize, clip_resize)
#             trajectory_embeddings.append(embedding.cpu().numpy())

#         # --- Save Trajectory Data ---
#         # Concatenate embeddings into a single array for this trajectory
#         final_embeddings_array = np.concatenate(trajectory_embeddings, axis=0) # Shape: [N+1, 49, 768]
        
#         npz_path = os.path.join(trajectory_dir, "trajectory_data.npz")
#         np.savez(
#             npz_path,
#             embeddings=final_embeddings_array,
#             uturn_steps=np.arange(args.num_uturns + 1),
#             start_image_path=args.start_image_path
#         )
#         logger.log(f"Saved trajectory data to {npz_path}")

# def create_argparser():
#     defaults = model_and_diffusion_defaults()
#     defaults.update(dict(
#         # --- New Experiment Control Arguments ---
#         start_image_path="", # REQUIRED: Path to the single starting image
#         output_dir="results/sequential_uturns",
#         num_uturns=10,       # N: Number of sequential U-turns
#         noise_step=150,      # t: The noise level for each U-turn
#         num_trajectories=5, # How many different random paths to generate
        
#         # --- Model Configuration (same as before) ---

#         clip_denoised=True,
#         use_ddim=False,
#         model_path="/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/models/256x256_diffusion_uncond.pt", # MODIFIED: Path to the downloaded model
#         image_size=256,
#         class_cond=False,
#         learn_sigma=True,
#         diffusion_steps=1000,
#         noise_schedule="linear",
#         timestep_respacing="",
#         use_kl=False,
#         predict_xstart=False,
#         rescale_timesteps=True,
#         rescale_learned_sigmas=True,
#         num_channels=256,
#         num_res_blocks=2,
#         num_heads=4,
#         num_heads_upsample=-1,
#         use_scale_shift_norm=True,
#         dropout=0.0,
#         attention_resolutions="32,16,8",
#         channel_mult="",
#         use_checkpoint=False,
#         use_fp16=True,
#         num_head_channels=64,
#         resblock_updown=True,
#         use_new_attention_order=False,
#     ))
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser

# if __name__ == "__main__":
#     main()



