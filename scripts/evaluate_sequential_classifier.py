"""
Calculates classifier activation differences (L2 and Cosine) for sequential U-turn data.
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
from PIL import Image

# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import add_dict_to_argparser
from guided_diffusion.torch_classifiers import load_classifier

def get_max_uturns(traj_dir):
    files = glob.glob(os.path.join(traj_dir, "uturn_*.jpeg"))
    if not files: return 0
    nums = [int(re.search(r'uturn_(\d+)', f).group(1)) for f in files]
    return max(nums)

def load_batch_for_step(trajectory_dirs, step_index, device):
    tensors = []
    
    for d in trajectory_dirs:
        path = os.path.join(d, f"uturn_{step_index:03d}.jpeg")
        if not os.path.exists(path):
            path = os.path.join(d, f"uturn_{step_index:03d}.png")
        
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            img = img.resize((256, 256), Image.BICUBIC)
            arr = np.array(img).astype(np.float32) / 127.5 - 1.0
            tensors.append(th.from_numpy(arr).permute(2, 0, 1))
            
    if not tensors:
        return None, 0
        
    batch = th.stack(tensors).to(device)
    return batch, len(tensors)

def main():
    args = create_argparser().parse_args()
    
    out_dir = os.path.join(args.output_base, args.classifier_name, args.image_name, f"noise_{args.noise_step}")
    os.makedirs(out_dir, exist_ok=True)
    dist_util.setup_dist()
    logger.configure(dir=out_dir)

    logger.log(f"Loading classifier {args.classifier_name}...")
    classifier, preprocess, module_names = load_classifier(args.classifier_name)
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    hooks = []
    for layer_name in module_names:
        try:
            layer = dict([*classifier.named_modules()])[layer_name]
            hook = layer.register_forward_hook(get_activation(layer_name))
            hooks.append(hook)
        except KeyError:
            logger.log(f"Warning: Layer {layer_name} not found.")

    # Whitening setup
    stat_file = os.path.join(args.output_base, args.classifier_name, f"act_stat_{args.classifier_name}.pk")
    activations_mean = {}
    activations_var = {}
    do_whitening = False
    
    if os.path.exists(stat_file):
        logger.log(f"Loading activation stats from {stat_file}")
        with open(stat_file, "rb") as f:
            act_stat = pickle.load(f)
        activations_mean = act_stat["activations_mean"]
        activations_var = act_stat["activations_var"]
        for key in activations_mean.keys():
            activations_mean[key] = th.tensor(activations_mean[key]).to(dist_util.dev())
            activations_var[key] = th.tensor(activations_var[key]).to(dist_util.dev())
        do_whitening = True

    def whiten_act(aa):
        if not do_whitening: return aa
        for key in activations_mean.keys():
            if key in aa:
                aa[key] = (aa[key] - activations_mean[key]) / th.sqrt(activations_var[key] + 1e-8)
        return aa

    base_search_path = os.path.join(args.data_dir, args.image_name, f"noise_step_{args.noise_step}", "trajectory_*")
    trajectory_dirs = sorted(glob.glob(base_search_path))
    
    if not trajectory_dirs:
        logger.log("No trajectory directories found.")
        return

    max_steps = get_max_uturns(trajectory_dirs[0])
    logger.log(f"Found {len(trajectory_dirs)} trajectories with up to {max_steps} U-turns.")

    # Process Step 0 (Reference)
    logger.log("Processing Reference (Step 0)...")
    batch_0, count = load_batch_for_step(trajectory_dirs, 0, dist_util.dev())
    
    if batch_0 is None:
        logger.log("Error: Could not load Step 0 images.")
        return

    with th.no_grad():
        _ = classifier(preprocess(batch_0))
    activations_0 = copy.deepcopy(whiten_act(activations))
    
    # Init Cosine Similarity object
    cosine_sim = th.nn.CosineSimilarity(dim=1, eps=1e-8)

    results_by_step = {}

    for k in range(1, max_steps + 1):
        logger.log(f"Processing Step {k}/{max_steps}...")
        
        batch_k, count = load_batch_for_step(trajectory_dirs, k, dist_util.dev())
        if batch_k is None: continue 

        with th.no_grad():
            _ = classifier(preprocess(batch_k))
        activations_k = copy.deepcopy(whiten_act(activations))
        
        step_results = {}
        
        for key in activations_k.keys():
            act_0 = activations_0[key][:count].flatten(start_dim=1)
            act_k = activations_k[key].flatten(start_dim=1)
            
            # 1. Normalized L2
            diff_sq = th.linalg.norm(act_k - act_0, dim=1) ** 2
            norm_0 = th.linalg.norm(act_0, dim=1)
            norm_k = th.linalg.norm(act_k, dim=1)
            normalized_diff = diff_sq / (norm_0 * norm_k + 1e-8)
            
            # 2. Cosine Similarity
            cos_val = cosine_sim(act_0, act_k)
            
            # Save both metrics
            step_results[key] = {
                "l2_norm": normalized_diff.cpu().numpy(),
                "cosine": cos_val.cpu().numpy()
            }
            
        results_by_step[k] = step_results

    outfile = os.path.join(out_dir, "sequential_activations_v2.pk") # Saved as v2 to distinguish
    logger.log(f"Saving results to {outfile}")
    with open(outfile, "wb") as f:
        pickle.dump(results_by_step, f)

    for hook in hooks: hook.remove()
    logger.log("Done.")

def create_argparser():
    defaults = dict(
        classifier_name="convnext_base",
        classifier_use_fp16=False,
        data_dir="/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns",
        image_name="ILSVRC2012_val_00000729",
        noise_step=75,
        output_base=os.path.join(os.getcwd(), "sequential_analysis_results"),
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()