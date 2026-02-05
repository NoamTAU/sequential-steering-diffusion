"""
Calculates classifier activation differences for ALL images found in the data directory.
Hierarchical structure: Image -> Noise Level -> Trajectories
Features:
- Robust to corrupted image files.
- Correctly aligns trajectories (Start vs Final).
- Performs Whitening (Mean Subtraction) using provided statistics.
"""
import argparse
import os
import sys
import pickle
import copy
import glob
import numpy as np
import torch as th
import re
from PIL import Image, UnidentifiedImageError

# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import add_dict_to_argparser
from guided_diffusion.torch_classifiers import load_classifier

def get_max_uturns(traj_dir):
    files = glob.glob(os.path.join(traj_dir, "uturn_*.jpeg"))
    if not files: 
        files = glob.glob(os.path.join(traj_dir, "uturn_*.png"))
    if not files: return 0
    nums = [int(re.search(r'uturn_(\d+)', f).group(1)) for f in files]
    return max(nums)

def load_batch_for_step(trajectory_dirs, step_index, device):
    """
    Loads images for a specific step. Skips corrupted files.
    Returns:
        batch: Tensor batch of valid images
        valid_indices: List of indices corresponding to the input 'trajectory_dirs' list.
    """
    tensors = []
    valid_indices = []
    
    for i, d in enumerate(trajectory_dirs):
        path = os.path.join(d, f"uturn_{step_index:03d}.jpeg")
        if not os.path.exists(path):
            path = os.path.join(d, f"uturn_{step_index:03d}.png")
        
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((256, 256), Image.BICUBIC)
                arr = np.array(img).astype(np.float32) / 127.5 - 1.0
                tensors.append(th.from_numpy(arr).permute(2, 0, 1))
                valid_indices.append(i)
            except (UnidentifiedImageError, OSError, ValueError) as e:
                # logger.log(f"Warning: Corrupted image at {path}. Skipping.")
                continue
            
    if not tensors: return None, []
    batch = th.stack(tensors).to(device)
    return batch, valid_indices

def process_single_image_noise(image_name, noise_step, args, classifier, preprocess, activations, activations_mean, activations_var, do_whitening):
    """
    Runs analysis for a specific image AND specific noise level.
    """
    logger.log(f"--- Processing: {image_name} | Noise: {noise_step} ---")
    
    # Construct output directory for this specific image and noise level
    out_dir = os.path.join(args.output_base, args.classifier_name, image_name, f"noise_{noise_step}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Find trajectories
    base_search_path = os.path.join(args.data_dir, image_name, f"noise_step_{noise_step}", "trajectory_*")
    trajectory_dirs = sorted(glob.glob(base_search_path))
    
    if not trajectory_dirs:
        logger.log(f"No trajectories found for {image_name} at noise {noise_step}. Skipping.")
        return

    max_steps = get_max_uturns(trajectory_dirs[0])
    logger.log(f"Found {len(trajectory_dirs)} trajectories. Max U-turns: {max_steps}")

    # --- Step 0 (Reference) ---
    device = next(classifier.parameters()).device
    batch_0, valid_indices_0 = load_batch_for_step(trajectory_dirs, 0, device)
    
    if batch_0 is None: 
        logger.log("Error: Could not load any Step 0 images.")
        return

    # Filter our directory list to keep only trajectories that successfully loaded Step 0
    valid_trajectory_dirs = [trajectory_dirs[i] for i in valid_indices_0]
    
    with th.no_grad():
        _ = classifier(preprocess(batch_0))
    
    # --- Whitening Helper ---
    def whiten_act(aa):
        if not do_whitening: return aa
        for key in activations_mean.keys():
            if key in aa:
                # Normalize: (x - mean) / sqrt(var + eps)
                aa[key] = (aa[key] - activations_mean[key]) / th.sqrt(activations_var[key] + 1e-8)
        return aa

    # Apply whitening to Step 0 activations
    activations_0 = copy.deepcopy(whiten_act(activations))
    cosine_sim = th.nn.CosineSimilarity(dim=1, eps=1e-8)
    results_by_step = {}

    # --- Loop over U-turns ---
    for k in range(1, max_steps + 1):
        # if k % 10 == 0: logger.log(f"  Step {k}/{max_steps}...")
        
        # Load batch for step k using ONLY the trajectories valid at step 0
        batch_k, valid_indices_k = load_batch_for_step(valid_trajectory_dirs, k, device)
        
        if batch_k is None: continue 

        with th.no_grad():
            _ = classifier(preprocess(batch_k))
        
        # Apply whitening to Step k activations
        activations_k = copy.deepcopy(whiten_act(activations))
        
        step_results = {}
        for key in activations_k.keys():
            # Align Reference (Step 0) with Current Batch (Step k)
            # valid_indices_k tells us which of the 'valid_trajectory_dirs' succeeded this time.
            act_0 = activations_0[key][valid_indices_k].flatten(start_dim=1)
            act_k = activations_k[key].flatten(start_dim=1)
            
            # Cosine Similarity
            cos_val = cosine_sim(act_0, act_k)
            step_results[key] = {"cosine": cos_val.cpu().numpy()}
            
        results_by_step[k] = step_results

    # Save
    outfile = os.path.join(out_dir, "sequential_activations_v2.pk")
    with open(outfile, "wb") as f:
        pickle.dump(results_by_step, f)
    logger.log(f"Saved: {outfile}")

def main():
    args = create_argparser().parse_args()
    
    # We use a direct logger config instead of dist_util to avoid MPI overhead for single process
    logger.configure(dir=args.output_base)
    device = "cuda" if th.cuda.is_available() else "cpu"

    logger.log(f"Loading classifier {args.classifier_name}...")
    classifier, preprocess, module_names = load_classifier(args.classifier_name)
    classifier.to(device)
    if args.classifier_use_fp16: classifier.convert_to_fp16()
    classifier.eval()

    # --- Setup Hooks ---
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    hooks = []
    for layer_name in module_names:
        try:
            layer = dict([*classifier.named_modules()])[layer_name]
            hooks.append(layer.register_forward_hook(get_activation(layer_name)))
        except KeyError: pass

    # --- Load Whitening Stats ---
    # Determine where to look for stats: explicit arg or default output base
    stat_base = args.stats_dir if args.stats_dir else args.output_base
    stat_file = os.path.join(stat_base, args.classifier_name, f"act_stat_{args.classifier_name}.pk")
    
    activations_mean = {}
    activations_var = {}
    do_whitening = False
    
    if os.path.exists(stat_file):
        logger.log(f"Loading stats from: {stat_file}")
        with open(stat_file, "rb") as f:
            act_stat = pickle.load(f)
        activations_mean = act_stat["activations_mean"]
        activations_var = act_stat["activations_var"]
        # Ensure stats are on the correct device
        for key in activations_mean.keys():
            activations_mean[key] = th.tensor(activations_mean[key]).to(device)
            activations_var[key] = th.tensor(activations_var[key]).to(device)
        do_whitening = True
    else:
        logger.log(f"WARNING: Activation stats not found at {stat_file}")
        logger.log("Whitening will be SKIPPED. Results will be incorrect (cosine similarity will not drop to 0).")

    # --- Discovery Loop ---
    logger.log("Discovering images...")
    
    all_items = sorted(os.listdir(args.data_dir))
    image_names = [d for d in all_items if os.path.isdir(os.path.join(args.data_dir, d))]
    
    logger.log(f"Found {len(image_names)} potential image folders.")

    for img_name in image_names:
        img_path = os.path.join(args.data_dir, img_name)
        
        # Find ALL noise folders (e.g. noise_step_25, noise_step_50...)
        noise_folders = sorted(glob.glob(os.path.join(img_path, "noise_step_*")))
        
        if not noise_folders: continue
            
        for n_folder in noise_folders:
            try:
                noise_val = int(os.path.basename(n_folder).split('_')[-1])
                
                process_single_image_noise(
                    img_name, 
                    noise_val, 
                    args, 
                    classifier, 
                    preprocess, 
                    activations, 
                    activations_mean, 
                    activations_var, 
                    do_whitening
                )
            except ValueError:
                continue

    for h in hooks: h.remove()
    logger.log("All images processed.")

def create_argparser():
    defaults = dict(
        classifier_name="convnext_base",
        classifier_use_fp16=False,
        data_dir="/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns",
        # Default Output root
        output_base=os.path.join(os.getcwd(), "sequential_analysis_results"),
        # NEW: Optional separate path for stats
        stats_dir=None
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()