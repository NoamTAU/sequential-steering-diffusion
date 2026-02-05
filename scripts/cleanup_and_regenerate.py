"""
Cleans up trajectory directories by standardizing all images to JPEG,
removing duplicates, and regenerating the .npz file with correct embeddings.
This version includes targeted noise level selection and robust, deterministic cleaning logic.
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

# Helper function from your generation script
def get_clip_patch_embeddings(visual_model, images, clip_normalize, clip_resize):
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

def main():
    parser = argparse.ArgumentParser(description="Clean and regenerate trajectory data.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory of the results")
    parser.add_argument("--dry_run", action='store_true', help="Show what would be done without actually changing files.")
    parser.add_argument(
        "--noise_steps", nargs="+", type=int, required=True,
        help="A list of specific noise steps to process, e.g., --noise_steps 25 50 75"
    )
    args = parser.parse_args()

    print("Setting up models...")
    device = "cuda" if th.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    clip_normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    clip_resize = Resize([224, 224])
    print(f"Using device: {device}")

    all_trajectory_dirs = []
    image_dirs = glob.glob(os.path.join(args.base_dir, "*"))
    for image_dir in image_dirs:
        for noise in args.noise_steps:
            noise_dir = os.path.join(image_dir, f"noise_step_{noise}")
            if os.path.isdir(noise_dir):
                all_trajectory_dirs.extend(glob.glob(os.path.join(noise_dir, "trajectory_*")))
    all_trajectory_dirs.sort()
    print(f"Found {len(all_trajectory_dirs)} total trajectory directories for the specified noise levels: {args.noise_steps}.")

    if args.dry_run:
        print("\n--- DRY RUN MODE: No files will be changed. ---")

    for i, traj_dir in enumerate(all_trajectory_dirs):
        print(f"\n--- Processing directory {i+1}/{len(all_trajectory_dirs)}: {traj_dir} ---")
        
        # --- THIS IS THE CORRECTED FILE FINDING LOGIC ---
        jpeg_paths = glob.glob(os.path.join(traj_dir, "uturn_*.jpeg"))
        png_paths = glob.glob(os.path.join(traj_dir, "uturn_*.png"))
        all_image_paths = jpeg_paths + png_paths
        
        grouped_by_step = {}
        for path in all_image_paths:
            try:
                step = int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
                if step not in grouped_by_step: grouped_by_step[step] = []
                grouped_by_step[step].append(path)
            except ValueError: continue
            
        final_image_paths_to_process = []

        if not grouped_by_step:
            print("  -> No images found. Skipping.")
            continue

        print("  Cleaning and standardizing images...")
        for step in sorted(grouped_by_step.keys()):
            paths = grouped_by_step[step]
            
            latest_file = max(paths, key=os.path.getmtime)
            if len(paths) > 1:
                print(f"    Step {step}: Found duplicates. Keeping newest: {os.path.basename(latest_file)}")
                for path_to_delete in paths:
                    if path_to_delete != latest_file:
                        if not args.dry_run: os.remove(path_to_delete)

            final_jpeg_path = latest_file
            if latest_file.endswith(".png"):
                final_jpeg_path = latest_file.replace(".png", ".jpeg")
                if not args.dry_run:
                    Image.open(latest_file).save(final_jpeg_path, "jpeg")
                    os.remove(latest_file)
            
            final_image_paths_to_process.append(final_jpeg_path)

        print(f"  Cleanup complete. Found {len(final_image_paths_to_process)} unique images to process.")

        print("  Regenerating embeddings...")
        regenerated_embeddings = []
        for img_path in final_image_paths_to_process:
            path_to_open = img_path
            if args.dry_run and not os.path.exists(img_path):
                path_to_open = img_path.replace(".jpeg", ".png")
            
            try:
                img_pil = Image.open(path_to_open).convert("RGB")
                img_tensor = th.tensor(np.array(img_pil)).float() / 127.5 - 1
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                embedding = get_clip_patch_embeddings(clip_model.visual, img_tensor, clip_normalize, clip_resize)
                regenerated_embeddings.append(embedding.cpu().numpy())
            except FileNotFoundError:
                print(f"    -> Warning: Could not open {path_to_open}. Skipping this step.")
                continue

        if not regenerated_embeddings:
            print("  -> No embeddings were generated. Skipping .npz save.")
            continue
            
        final_embeddings_array = np.concatenate(regenerated_embeddings, axis=0)
        npz_path = os.path.join(traj_dir, "trajectory_data.npz")
        print(f"  Saving new .npz file with {final_embeddings_array.shape[0]} embeddings.")
        if not args.dry_run:
            try:
                old_data = np.load(npz_path)
                start_img_path = str(old_data['start_image_path'])
            except (FileNotFoundError, KeyError):
                start_img_path = "unknown"
            np.savez(
                npz_path,
                embeddings=final_embeddings_array,
                start_image_path=start_img_path
            )
        print("  Done.")
    
    print("\n--- Cleanup and regeneration complete. ---")

if __name__ == "__main__":
    main()