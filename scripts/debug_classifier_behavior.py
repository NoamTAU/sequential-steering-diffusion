"""
Debug script to inspect classifier predictions and input statistics
on specific sequential U-turn images.
"""
import argparse
import os
import sys
import glob
import torch as th
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from guided_diffusion import dist_util, logger
from guided_diffusion.torch_classifiers import load_classifier

# Function to load ImageNet labels for readability
# (If you don't have this file, it will print indices)
def load_imagenet_labels():
    try:
        # Standard location on many systems, or download a simple json/txt
        # Using a simple placeholder dict logic or attempting to load
        import json
        import urllib.request
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url) as url:
            labels = json.loads(url.read().decode())
        return labels
    except Exception as e:
        print(f"Could not load ImageNet class names: {e}. Using indices.")
        return None

def get_image_tensor(path, device):
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0 # [-1, 1] range
    tensor = th.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_name", type=str, default="convnext_base")
    # Point this to a SPECIFIC trajectory folder that you want to debug
    parser.add_argument("--traj_dir", type=str, required=True, 
                        help="Path to a specific trajectory folder, e.g., .../trajectory_000")
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 1, 5, 10, 25, 50, 100])
    args = parser.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Classifier
    print(f"Loading classifier: {args.classifier_name}...")
    classifier, preprocess, _ = load_classifier(args.classifier_name)
    classifier.to(device)
    classifier.eval()

    labels_map = load_imagenet_labels()

    # 2. Loop through requested steps
    print("\n--- DEBUGGING CLASSIFIER PREDICTIONS ---")
    print(f"{'Step':<5} | {'Min':<6} | {'Max':<6} | {'Top-1 Class':<30} | {'Prob':<6} | {'Status'}")
    print("-" * 80)

    # Store for visualization
    debug_images = []
    debug_titles = []

    # Get Reference (Step 0) Class
    # Try JPEG then PNG
    path_0 = os.path.join(args.traj_dir, "uturn_000.jpeg")
    if not os.path.exists(path_0): path_0 = os.path.join(args.traj_dir, "uturn_000.png")
    
    tensor_0, _ = get_image_tensor(path_0, device)
    with th.no_grad():
        logits_0 = classifier(preprocess(tensor_0))
        probs_0 = th.nn.functional.softmax(logits_0, dim=1)
        top1_prob_0, top1_idx_0 = th.topk(probs_0, 1)
        ref_class_idx = top1_idx_0.item()

    for step in args.steps:
        # Load Image
        filename = f"uturn_{step:03d}.jpeg"
        path = os.path.join(args.traj_dir, filename)
        if not os.path.exists(path):
            filename = f"uturn_{step:03d}.png"
            path = os.path.join(args.traj_dir, filename)
        
        if not os.path.exists(path):
            print(f"{step:<5} | File not found")
            continue

        # Prepare Tensor
        raw_tensor, pil_img = get_image_tensor(path, device)
        
        # --- CRITICAL CHECK: What is going into the classifier? ---
        # The preprocess function usually normalizes. We check inputs to preprocess
        # and outputs of preprocess.
        
        with th.no_grad():
            processed_input = preprocess(raw_tensor)
            input_min = processed_input.min().item()
            input_max = processed_input.max().item()

            logits = classifier(processed_input)
            probs = th.nn.functional.softmax(logits, dim=1)
            
            topk_probs, topk_indices = th.topk(probs, 3)

        # Get readable label
        top1_idx = topk_indices[0][0].item()
        top1_prob = topk_probs[0][0].item()
        
        class_name = str(top1_idx)
        if labels_map:
            class_name = labels_map[top1_idx][:28] # Truncate for print

        # Check if it matches original
        status = "MATCH" if top1_idx == ref_class_idx else "DRIFT"
        if step == 0: status = "REF"

        print(f"{step:<5} | {input_min:<6.2f} | {input_max:<6.2f} | {class_name:<30} | {top1_prob:<6.2f} | {status}")

        debug_images.append(pil_img)
        debug_titles.append(f"T={step}\n{class_name}\n({top1_prob:.2f})")

    # 3. Create Visualization
    print("\nSaving visual debug summary...")
    cols = 5
    rows = (len(debug_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows))
    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(debug_images, debug_titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')
    
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    save_path = os.path.join(args.traj_dir, "debug_classifier_predictions.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()