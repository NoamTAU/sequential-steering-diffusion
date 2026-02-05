"""
Visualizes features of a pre-trained CNN via Activation Maximization.
Includes Regularization (Jitter, Smoothing) for high-quality images.
"""
import argparse
import os
import sys
import torch as th
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from guided_diffusion import dist_util, logger
from guided_diffusion.torch_classifiers import load_classifier

# --- HELPER: Random Jitter ---
def jitter(img, max_jitter=8):
    """
    Randomly shifts the image. This encourages the feature to be robust 
    to translation and reduces grid artifacts.
    """
    ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
    return th.roll(img, shifts=(ox, oy), dims=(2, 3))

# --- CORE VISUALIZATION FUNCTION ---
def visualize_layer(classifier, preprocess, layer_name, num_channels=4, steps=200, lr=0.05):
    device = dist_util.dev()
    
    # 1. Start with Gray noise (better than random white noise)
    # We initialize in the space [-1, 1] (or whatever the model expects BEFORE normalization)
    img_tensor = (th.randn(1, 3, 224, 224, device=device) * 0.01).requires_grad_(True)
    
    optimizer = th.optim.Adam([img_tensor], lr=lr, weight_decay=1e-4)
    
    # Gaussian Blur for smoothing (Standard trick for better visuals)
    blur = transforms.GaussianBlur(kernel_size=3, sigma=0.5)

    # Hook to capture activation
    activations = {}
    def hook(module, input, output):
        activations['act'] = output
    
    try:
        layer_module = dict([*classifier.named_modules()])[layer_name]
        handle = layer_module.register_forward_hook(hook)
    except KeyError:
        print(f"Layer {layer_name} not found.")
        return []
    
    print(f"Dreaming for layer: {layer_name}...")
    
    generated_images = []
    
    # Generate multiple examples (different random seeds/channels)
    for ch in range(num_channels):
        # Reset image for each channel example
        img_tensor.data.normal_(0, 0.01)
        optimizer = th.optim.Adam([img_tensor], lr=lr)

        for i in range(steps):
            optimizer.zero_grad()
            
            # A. Apply Jitter (Regularization)
            jittered_img = jitter(img_tensor, max_jitter=4)
            
            # B. Preprocess (Normalization)
            # IMPORTANT: We must use the classifier's specific preprocess function
            # This handles mean/std subtraction expected by ConvNeXt
            input_to_model = preprocess(jittered_img)
            
            # C. Forward Pass
            classifier(input_to_model)
            
            # D. Loss: Maximize activation
            act = activations['act']
            
            # Option 1: Maximize specific channel (if looking for specific features)
            # loss = -act[:, ch % act.shape[1], ...].mean() 
            
            # Option 2: Maximize the "energy" of the whole layer (shows general texture)
            # This is better for understanding the layer's general style
            loss = -act.pow(2).mean()

            loss.backward()
            optimizer.step()
            
            # E. Periodic Smoothing (Regularization)
            # This kills high-frequency static and encourages shapes
            if i % 10 == 0:
                with th.no_grad():
                    img_tensor.copy_(blur(img_tensor))
            
            # F. Clamp to valid range (assuming [-1, 1] is the base range before norm)
            with th.no_grad():
                img_tensor.clamp_(-1, 1)

        # Post-process for display
        # Convert [-1, 1] tensor back to [0, 1] for plotting
        img_vis = img_tensor.detach().squeeze().permute(1, 2, 0).cpu().numpy()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
        generated_images.append(img_vis)

    handle.remove()
    return generated_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_name", type=str, default="convnext_base")
    # Note: Using features.6 instead of classifier for visualizable features
    parser.add_argument("--layers", type=str, default="features.0,features.2,features.4,features.6")
    parser.add_argument("--output_dir", type=str, default="feature_visualizations")
    args = parser.parse_args()
    
    dist_util.setup_dist()
    
    print(f"Loading {args.classifier_name}...")
    classifier, preprocess, _ = load_classifier(args.classifier_name)
    classifier.to(dist_util.dev())
    classifier.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    layers = args.layers.split(',')
    
    # Setup plot
    cols = 4
    fig, axes = plt.subplots(len(layers), cols, figsize=(3 * cols, 3 * len(layers)))
    # Handle single layer case
    if len(layers) == 1: axes = axes[None, :] 
    
    plt.subplots_adjust(hspace=0.4)
    
    for i, layer in enumerate(layers):
        # Skip 'classifier' or 'head' usually, as they are 1D vectors and don't make good images
        if "classifier" in layer or "head" in layer:
            print(f"Skipping visualization for {layer} (Usually 1D vector, not visualizable)")
            continue

        images = visualize_layer(classifier, preprocess, layer, num_channels=cols)
        
        if not images: continue

        axes[i, 0].set_ylabel(layer, fontsize=12, fontweight='bold')
        
        for j, img in enumerate(images):
            axes[i, j].imshow(img)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            
            # Save individual image
            plt.imsave(os.path.join(args.output_dir, f"{layer}_sample_{j}.png"), img)

    plt.suptitle(f"Feature Visualization ({args.classifier_name})", fontsize=16)
    save_path = os.path.join(args.output_dir, "summary_plot.png")
    plt.savefig(save_path)
    print(f"Summary saved to {save_path}")

if __name__ == "__main__":
    main()