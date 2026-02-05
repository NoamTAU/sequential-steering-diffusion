"""
Performs UNSTEERED sequential U-turns to probe the natural manifold geometry.
Saves full classifier logits at every step to map probability flow.
"""
import argparse
import os
import sys
import numpy as np
import torch as th
from PIL import Image
from torchvision.transforms import Normalize, Resize

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from guided_diffusion import logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.torch_classifiers import load_classifier

def perform_single_uturn(model, diffusion, start_tensor, noise_step, device):
    with th.no_grad():
        # 1. Forward (Add Noise)
        t_batch = th.tensor([noise_step] * start_tensor.shape[0], device=device)
        noisy_image = diffusion.q_sample(start_tensor, t_batch)
        
        # 2. Backward (Denoise)
        img = noisy_image
        indices = list(range(noise_step))[::-1]
        for i in indices:
            t = th.tensor([i] * start_tensor.shape[0], device=device)
            out = diffusion.p_sample(
                model=model,
                x=img,
                t=t,
                clip_denoised=True
            )
            img = out["sample"]
        return img

def run_probe(args, model, diffusion, classifier, classifier_preprocess, device, start_tensor, trajectory_dir):
    current_image = start_tensor
    
    # Store FULL logits for every step (Shape: [Steps, 1000])
    # This allows us to calculate force/potential for ANY class later.
    all_logits = [] 
    
    logger.log(f"Starting Probe. Steps: {args.num_steps}. Noise: {args.noise_step}")

    with th.no_grad():
        for step in range(args.num_steps + 1): # +1 to include step 0
            # 1. Evaluate Classifier
            logits = classifier(classifier_preprocess(current_image))
            all_logits.append(logits.cpu().numpy())
            
            # Save Image (Optional, maybe skip every few steps to save space)
            if step % 5 == 0:
                img_save = ((current_image[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
                Image.fromarray(img_save).save(os.path.join(trajectory_dir, f"step_{step:03d}.jpeg"))

            if step < args.num_steps:
                # 2. Perform Random U-Turn (No Selection, just take the result)
                # We essentially accept the first proposal every time.
                current_image = perform_single_uturn(model, diffusion, current_image, args.noise_step, device)
                
                # Periodic Logging
                if step % 10 == 0:
                    logger.log(f"Step {step}/{args.num_steps} complete.")

    # Save Data
    # Concatenate to shape (N_steps, 1000)
    full_logits_array = np.concatenate(all_logits, axis=0)
    np.savez_compressed(os.path.join(trajectory_dir, "manifold_logits.npz"), logits=full_logits_array)
    logger.log("Saved full logit trajectory.")

def main():
    args = create_argparser().parse_args()
    device = "cuda" if th.cuda.is_available() else "cpu"
    
    # Load Models
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model.to(device); model.eval()
    if args.use_fp16: model.convert_to_fp16()

    classifier, classifier_preprocess, _ = load_classifier(args.classifier_name)
    classifier.to(device); classifier.eval()
    if args.classifier_use_fp16: classifier.convert_to_fp16()

    # Load Image
    diffusion_resize = Resize([args.image_size, args.image_size], Image.BICUBIC)
    start_pil = diffusion_resize(Image.open(args.start_image_path).convert("RGB"))
    start_tensor = th.tensor(np.array(start_pil)).float() / 127.5 - 1
    start_tensor = start_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # Output Dir
    base_name = os.path.splitext(os.path.basename(args.start_image_path))[0]
    out_dir = os.path.join(args.output_dir, base_name, f"probe_noise_{args.noise_step}", f"traj_{args.trajectory_idx:03d}")
    os.makedirs(out_dir, exist_ok=True)
    logger.configure(dir=out_dir)

    run_probe(args, model, diffusion, classifier, classifier_preprocess, device, start_tensor, out_dir)

def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        start_image_path="", 
        output_dir="results/manifold_probe", 
        num_steps=200,      # Long trajectory to see where it drifts
        noise_step=100,     # The "Temperature" of the probe
        trajectory_idx=0,   # ID for this random realization
        
        classifier_name="convnext_base",
        clip_denoised=True,
        use_ddim=False,
        model_path="/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/models/256x256_diffusion_uncond.pt",
        image_size=256,
        class_cond=False,
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="250", # Default to match your steering experiments
        use_fp16=True,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        attention_resolutions="32,16,8",
        channel_mult="",
        use_checkpoint=False,
        num_head_channels=64,
        resblock_updown=True,
        use_new_attention_order=False,
        classifier_use_fp16=False
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()