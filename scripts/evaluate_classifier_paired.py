import argparse
import os
import sys
import pickle
import copy
import glob
import time
import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image
from torchvision.transforms import ToTensor

# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import add_dict_to_argparser
from guided_diffusion.torch_classifiers import load_classifier

# --- NEW: Custom Data Loader for your specific folder structure ---
def yield_paired_data(base_path, time_step, batch_size, device):
    """
    Looks into .../visual_examples/t_{time_step}/
    Finds matching start_X.png and final_X.png
    Yields batches of (start_tensor, final_tensor) in [-1, 1] range.
    """
    # Construct path: e.g. results/.../visual_examples/t_0125
    folder_name = f"t_{time_step:04d}"
    target_dir = os.path.join(base_path, "visual_examples", folder_name)
    
    if not os.path.exists(target_dir):
        logger.log(f"Warning: Directory not found: {target_dir}")
        return

    # Find all start images
    start_files = sorted(glob.glob(os.path.join(target_dir, "start_*.png")))
    
    pairs = []
    for s_path in start_files:
        # Construct the expected final path
        # e.g. start_000.png -> final_000.png
        f_path = s_path.replace("start_", "final_")
        
        if os.path.exists(f_path):
            pairs.append((s_path, f_path))
    
    logger.log(f"Found {len(pairs)} pairs in {folder_name}")

    # Helper to convert PIL -> Tensor [-1, 1]
    def process_img(path):
        img = Image.open(path).convert("RGB")
        # Resize to 256x256 if needed, though they should already be correct
        img = img.resize((256, 256), Image.BICUBIC)
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return th.from_numpy(arr).permute(2, 0, 1) # HWC -> CHW

    # Yield batches
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i+batch_size]
        
        start_tensors = []
        final_tensors = []
        
        for sp, fp in batch_pairs:
            start_tensors.append(process_img(sp))
            final_tensors.append(process_img(fp))
            
        batch_start = th.stack(start_tensors).to(device)
        batch_final = th.stack(final_tensors).to(device)
        
        yield batch_start, batch_final

def main():
    args = create_argparser().parse_args()
    
    # Create specific output dir for this classifier
    out_dir = os.path.join(args.output_base, args.classifier_name)
    dist_util.setup_dist()
    logger.configure(dir=out_dir)

    logger.log(f"Loading classifier {args.classifier_name}...")
    classifier, preprocess, module_names = load_classifier(args.classifier_name)
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def class_eval(x):
        with th.no_grad():
            # x is in [-1, 1]. preprocess handles internal normalization for the specific classifier model.
            logits = classifier(preprocess(x))
            return logits

    # --- Statistics Loading Logic ---
    # NOTE: This part expects you to have already run a script to generate "act_stat_...pk"
    # If you haven't, you might want to comment out the whitening part temporarily.
    stat_file = os.path.join(out_dir, f"act_stat_{args.classifier_name}.pk")
    activations_mean = {}
    activations_var = {}
    
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
    else:
        logger.log("Warning: Activation stats file not found. Skipping whitening.")
        do_whitening = False

    def whiten_act(aa):
        if not do_whitening: return aa
        for key in activations_mean.keys():
            if key in aa:
                aa[key] = (aa[key] - activations_mean[key]) / th.sqrt(activations_var[key] + 1e-8)
        return aa

    # --- Hooking Logic ---
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
            logger.log(f"Warning: Layer {layer_name} not found in model.")

    # --- Main Loop over Time Series ---
    for time_step in args.time_series:
        logger.log(f"--- Processing time step {time_step} ---")
        
        data_loader = yield_paired_data(
            args.results_dir, 
            time_step, 
            args.batch_size, 
            dist_util.dev()
        )

        dict_list = []
        all_logits_start = []
        all_logits_sample = []
        
        total_samples = 0
        time_start = time.time()

        for batch_start, batch_sample in data_loader:
            # 1. Run classifier on Start (Clean) images
            class_eval_start = class_eval(batch_start)
            # Deepcopy essential because 'activations' dict is overwritten in next forward pass
            activations_start = copy.deepcopy(whiten_act(activations))

            # 2. Run classifier on Sample (Noisy/Denoised) images
            class_eval_sample = class_eval(batch_sample)
            activations_sample = copy.deepcopy(whiten_act(activations))

            # 3. Compute differences
            diff_activations = {}
            cosine_sim = th.nn.CosineSimilarity(dim=1, eps=1e-8)
            
            for key in activations_start.keys():
                diff_activations[key] = {}
                # Flatten spatial dims: (B, C, H, W) -> (B, C*H*W)
                act_s = activations_sample[key].flatten(start_dim=1)
                act_st = activations_start[key].flatten(start_dim=1)
                
                # L2 Squared
                diff_activations[key]["L2"] = th.linalg.norm(act_s - act_st, dim=1) ** 2
                
                # Normalized L2
                norm_s = th.linalg.norm(act_s, dim=1)
                norm_st = th.linalg.norm(act_st, dim=1)
                diff_activations[key]["L2_normalized"] = diff_activations[key]["L2"] / (norm_s * norm_st + 1e-8)
                
                # Cosine Similarity
                diff_activations[key]["cosine"] = cosine_sim(act_s, act_st)

            dict_list.append(diff_activations)
            
            all_logits_start.append(class_eval_start.cpu().numpy())
            all_logits_sample.append(class_eval_sample.cpu().numpy())

            total_samples += batch_start.shape[0]
            logger.log(f"Processed {total_samples} samples...")

        if total_samples == 0:
            logger.log(f"No samples found for time step {time_step}. Skipping.")
            continue

        # --- Aggregate Results ---
        logger.log(f"Aggregating results for t={time_step}...")
        dictionary_act = {}
        
        # Structure: dict[layer][metric] = numpy array of shape (N,)
        first_batch_dict = dict_list[0]
        for layer_key in first_batch_dict.keys():
            dictionary_act[layer_key] = {}
            for metric_key in first_batch_dict[layer_key].keys():
                # Concatenate this specific metric across all batches
                metric_values = [d[layer_key][metric_key] for d in dict_list]
                dictionary_act[layer_key][metric_key] = th.cat(metric_values, dim=0).cpu().numpy()

        all_logits_start = np.concatenate(all_logits_start, axis=0)
        all_logits_sample = np.concatenate(all_logits_sample, axis=0)

        # --- Save ---
        outfile_act = os.path.join(out_dir, f"act_diff_{args.classifier_name}_t{time_step}.pk")
        logger.log(f"Saving activations to {outfile_act}")
        with open(outfile_act, "wb") as handle:
            pickle.dump(dictionary_act, handle)

        outfile_logits = os.path.join(out_dir, f"logits_{args.classifier_name}_t{time_step}.pk")
        logger.log(f"Saving logits to {outfile_logits}")
        with open(outfile_logits, "wb") as handle:
            pickle.dump({
                "logits_start": all_logits_start,
                "logits_sample": all_logits_sample,
            }, handle)

    # Cleanup
    for hook in hooks:
        hook.remove()
    
    logger.log("Analysis complete!")

def create_argparser():
    defaults = dict(
        classifier_name="convnext_base",
        classifier_use_fp16=False,
        batch_size=32,
        image_size=256,
        # Path to the SPECIFIC timestamped results folder
        results_dir="/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/scripts/results/forw_back_uncond/2025-12-18-14-12-37-321303",
        # Output directory for the statistics
        output_base=os.path.join(os.getcwd(), "classifier_statistics"),
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(
        "--time_series",
        nargs="+",
        type=int,
        default=[25, 50, 75, 100, 125, 150, 175, 200, 225, 249],
        help="Time steps to evaluate. Pass like: --time_series 25 50 100",
    )
    return parser

if __name__ == "__main__":
    main()