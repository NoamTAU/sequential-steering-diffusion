import argparse
import os
import sys
import numpy as np
import torch as th
from PIL import Image
from torchvision.transforms import Normalize, Resize
from tqdm.auto import tqdm

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

# --- IMAGE NET CLASS RANGES ---
DOG_INDICES = list(range(151, 269)) 
CAT_INDICES = list(range(281, 286))

def get_meta_score_full(classifier, preprocess, batch_images, penalty_weight):
    """
    Returns scores for the ENTIRE batch to calculate force statistics.
    Score = LogSumExp(Cats) - weight * LogSumExp(Dogs)
    """
    with th.no_grad():
        # Keep input dtype aligned with classifier to avoid FP16/FP32 mismatch.
        clf_dtype = next(classifier.parameters()).dtype
        processed = preprocess(batch_images)
        processed = processed.to(clf_dtype)
        logits = classifier(processed)
        
        # Calculate Log-Probability of the Meta-Classes
        cat_score = th.logsumexp(logits[:, CAT_INDICES], dim=1)
        dog_score = th.logsumexp(logits[:, DOG_INDICES], dim=1)
        
        # Score for every image in the batch
        scores = cat_score - (penalty_weight * dog_score)
        
        # Find best for steering
        best_idx = th.argmax(scores).item()
        best_score = scores[best_idx].item()
        
        # Calculate probs for logging
        probs = th.nn.functional.softmax(logits, dim=1)
        total_cat_prob = probs[best_idx, CAT_INDICES].sum().item()
        total_dog_prob = probs[best_idx, DOG_INDICES].sum().item()
        
        return scores, best_idx, best_score, total_cat_prob, total_dog_prob, cat_score, dog_score

def run_steering_trajectory(args, model, diffusion, classifier, classifier_preprocess, device, start_tensor, trajectory_dir):
    
    current_image = start_tensor 
    BATCH_SIZE = args.batch_size 
    MAX_RETRIES = args.max_retries

    # History
    probs_dog = []
    probs_cat = []
    scores_log = []
    cat_scores_log = []
    dog_scores_log = []
    attempts_log = [] 
    force_history = [] # [mean, std] per batch (combined score)
    force_history_cat = [] # [mean, std] per batch (cat score delta)
    force_history_dog = [] # [mean, std] per batch (dog score delta)

    # --- Step 0 ---
    scores_0, best_idx, best_score, p_cat, p_dog, cat_score0, dog_score0 = get_meta_score_full(
        classifier, classifier_preprocess, current_image, args.penalty
    )
    
    current_score = best_score # Initialize current score
    current_cat = cat_score0[0].item()
    current_dog = dog_score0[0].item()
    current_cat_prob = p_cat
    current_dog_prob = p_dog
    
    probs_dog.append(p_dog)
    probs_cat.append(p_cat)
    scores_log.append(current_score)
    cat_scores_log.append(current_cat)
    dog_scores_log.append(current_dog)
    attempts_log.append(0)

    img_save = ((current_image[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img_save).save(os.path.join(trajectory_dir, "step_000.jpeg"))
    
    # Save Initial Data
    np.savez(
        os.path.join(trajectory_dir, "steering_data.npz"),
        probs_dog=np.array(probs_dog),
        probs_cat=np.array(probs_cat),
        scores=np.array(scores_log),
        cat_scores=np.array(cat_scores_log),
        dog_scores=np.array(dog_scores_log),
        attempts=np.array(attempts_log)
    )
    
    logger.log(f"Starting Meta-Steering. Noise: {args.noise_step}. Penalty: {args.penalty}")

    with th.no_grad():
        progress_bar = tqdm(range(1, args.num_steps + 1), desc="Steering Progress", unit="step")
        for step in progress_bar:
            
            step_accepted = False
            retries = 0
            
            while not step_accepted and retries < MAX_RETRIES:
                th.cuda.empty_cache()
                
                # 1. Forward
                batch_input = current_image.repeat(BATCH_SIZE, 1, 1, 1)
                t_batch = th.tensor([args.noise_step] * BATCH_SIZE, device=device)
                noisy_batch = diffusion.q_sample(batch_input, t_batch)
                
                # 2. Backward
                sample_img = noisy_batch
                indices = list(range(args.noise_step))[::-1]
                for i in indices:
                    t = th.tensor([i] * BATCH_SIZE, device=device)
                    out = diffusion.p_sample(model=model, x=sample_img, t=t, clip_denoised=True)
                    sample_img = out["sample"]
                
                # 3. Evaluate Meta-Proposals (Get Full Batch Scores)
                batch_scores, best_idx, best_score, best_p_cat, best_p_dog, cat_scores, dog_scores = get_meta_score_full(
                    classifier, classifier_preprocess, sample_img, args.penalty
                )
                
                # --- FORCE CALCULATION ---
                # Force = New Scores - Current Score
                deltas = batch_scores - current_score
                mean_force = deltas.mean().item()
                std_force = deltas.std().item()
                
                # We log every batch attempt to see the force landscape over time
                force_history.append([mean_force, std_force])

                # --- META FORCE (CAT/DOG) ---
                cat_deltas = cat_scores - current_cat
                dog_deltas = dog_scores - current_dog
                force_history_cat.append([cat_deltas.mean().item(), cat_deltas.std().item()])
                force_history_dog.append([dog_deltas.mean().item(), dog_deltas.std().item()])
                
                # 4. Acceptance Logic
                score_ok = best_score > current_score
                target_ok = (best_p_cat >= current_cat_prob + args.target_prob_epsilon) if args.require_target_increase else True
                accept_ok = score_ok and target_ok

                if accept_ok:
                    step_accepted = True
                    current_image = sample_img[best_idx].unsqueeze(0)
                    
                    score_diff = best_score - current_score
                    current_score = best_score
                    current_cat = cat_scores[best_idx].item()
                    current_dog = dog_scores[best_idx].item()
                    current_cat_prob = best_p_cat
                    current_dog_prob = best_p_dog
                    
                    probs_dog.append(best_p_dog)
                    probs_cat.append(best_p_cat)
                    scores_log.append(current_score)
                    cat_scores_log.append(current_cat)
                    dog_scores_log.append(current_dog)
                    attempts_log.append((retries + 1) * BATCH_SIZE)

                    status = "ACCEPTED"
                    logger.log(
                        f"Step {step}: {status} ({retries+1}). "
                        f"Score {best_score:.2f} (Force: {mean_force:.3f}) | "
                        f"Cat P: {best_p_cat:.4f} | Dog P: {best_p_dog:.4f} | "
                        f"Cat Score: {current_cat:.2f} | Dog Score: {current_dog:.2f}"
                    )
                    progress_bar.set_postfix({
                        "CatP": f"{best_p_cat:.3f}",
                        "DogP": f"{best_p_dog:.3f}",
                        "Score": f"{current_score:.2f}",
                        "Status": status,
                    })
                    
                    img_save = ((current_image[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
                    Image.fromarray(img_save).save(os.path.join(trajectory_dir, f"step_{step:03d}.jpeg"))

                    np.savez(
                        os.path.join(trajectory_dir, "steering_data.npz"),
                        probs_dog=np.array(probs_dog),
                        probs_cat=np.array(probs_cat),
                        scores=np.array(scores_log),
                        cat_scores=np.array(cat_scores_log),
                        dog_scores=np.array(dog_scores_log),
                        attempts=np.array(attempts_log)
                    )
                    
                    # Save Force Data Incrementally
                    np.savez(
                        os.path.join(trajectory_dir, "force_stats.npz"),
                        force=np.array(force_history),
                        force_cat=np.array(force_history_cat),
                        force_dog=np.array(force_history_dog),
                    )
                else:
                    retries += 1

                # If we exhausted retries without acceptance, handle fail behavior
                if (not step_accepted) and (retries >= MAX_RETRIES):
                    # Stop early if we've reached target probability threshold
                    if (args.target_prob_stop is not None) and (args.target_prob_stop >= 0) and (current_cat_prob >= args.target_prob_stop):
                        logger.log(
                            f"Step {step}: STOP (cat_prob {current_cat_prob:.4f} >= {args.target_prob_stop:.4f})"
                        )
                        return

                    if args.fail_behavior == "skip":
                        # Keep current image; log a flat step
                        attempts_log.append(retries * BATCH_SIZE)
                        probs_dog.append(current_dog_prob)
                        probs_cat.append(current_cat_prob)
                        scores_log.append(current_score)
                        cat_scores_log.append(current_cat)
                        dog_scores_log.append(current_dog)
                        status = "SKIP"
                        logger.log(
                            f"Step {step}: {status} after {retries} batches. "
                            f"Score {current_score:.2f} | Cat P: {current_cat_prob:.4f} | Dog P: {current_dog_prob:.4f}"
                        )
                        progress_bar.set_postfix({
                            "CatP": f"{current_cat_prob:.3f}",
                            "DogP": f"{current_dog_prob:.3f}",
                            "Score": f"{current_score:.2f}",
                            "Status": status,
                        })

                        img_save = ((current_image[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
                        Image.fromarray(img_save).save(os.path.join(trajectory_dir, f"step_{step:03d}.jpeg"))

                        np.savez(
                            os.path.join(trajectory_dir, "steering_data.npz"),
                            probs_dog=np.array(probs_dog),
                            probs_cat=np.array(probs_cat),
                            scores=np.array(scores_log),
                            cat_scores=np.array(cat_scores_log),
                            dog_scores=np.array(dog_scores_log),
                            attempts=np.array(attempts_log)
                        )
                        np.savez(
                            os.path.join(trajectory_dir, "force_stats.npz"),
                            force=np.array(force_history),
                            force_cat=np.array(force_history_cat),
                            force_dog=np.array(force_history_dog),
                        )
                        step_accepted = True
                    elif args.fail_behavior == "force":
                        # Force accept best candidate (may violate monotonic target prob)
                        step_accepted = True
                        current_image = sample_img[best_idx].unsqueeze(0)
                        score_diff = best_score - current_score
                        current_score = best_score
                        current_cat = cat_scores[best_idx].item()
                        current_dog = dog_scores[best_idx].item()
                        current_cat_prob = best_p_cat
                        current_dog_prob = best_p_dog

                        probs_dog.append(best_p_dog)
                        probs_cat.append(best_p_cat)
                        scores_log.append(current_score)
                        cat_scores_log.append(current_cat)
                        dog_scores_log.append(current_dog)
                        attempts_log.append(retries * BATCH_SIZE)

                        status = "FORCED"
                        logger.log(
                            f"Step {step}: {status} ({retries}). "
                            f"Score {best_score:.2f} (Force: {mean_force:.3f}) | "
                            f"Cat P: {best_p_cat:.4f} | Dog P: {best_p_dog:.4f}"
                        )
                        progress_bar.set_postfix({
                            "CatP": f"{best_p_cat:.3f}",
                            "DogP": f"{best_p_dog:.3f}",
                            "Score": f"{current_score:.2f}",
                            "Status": status,
                        })

                        img_save = ((current_image[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
                        Image.fromarray(img_save).save(os.path.join(trajectory_dir, f"step_{step:03d}.jpeg"))

                        np.savez(
                            os.path.join(trajectory_dir, "steering_data.npz"),
                            probs_dog=np.array(probs_dog),
                            probs_cat=np.array(probs_cat),
                            scores=np.array(scores_log),
                            cat_scores=np.array(cat_scores_log),
                            dog_scores=np.array(dog_scores_log),
                            attempts=np.array(attempts_log)
                        )
                        np.savez(
                            os.path.join(trajectory_dir, "force_stats.npz"),
                            force=np.array(force_history),
                            force_cat=np.array(force_history_cat),
                            force_dog=np.array(force_history_dog),
                        )
                    else:
                        logger.log(f"Step {step}: STOP (no acceptable candidate after {retries} batches)")
                        return

def main():
    args = create_argparser().parse_args()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    def log_cuda_mem(prefix):
        if device.type != "cuda":
            return
        free, total = th.cuda.mem_get_info()
        print(f"{prefix} CUDA free: {free/1e9:.2f} GB / {total/1e9:.2f} GB")
    
    # Load Models
    if device.type == "cuda":
        th.cuda.set_device(0)
        log_cuda_mem("Before model init")

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    if args.use_fp16:
        # Convert on CPU before moving to GPU to reduce peak memory.
        if hasattr(model, "convert_to_fp16"):
            model.convert_to_fp16()
        else:
            model.half()
    try:
        model.to(device); model.eval()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == "cuda":
            log_cuda_mem("OOM during model.to")
        raise
    if device.type == "cuda":
        log_cuda_mem("After model.to")

    classifier, classifier_preprocess, _ = load_classifier(args.classifier_name)
    if args.classifier_use_fp16:
        # ConvNeXt doesn't implement convert_to_fp16(), so fall back to .half().
        if hasattr(classifier, "convert_to_fp16"):
            classifier.convert_to_fp16()
        else:
            classifier.half()
    try:
        classifier.to(device); classifier.eval()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == "cuda":
            log_cuda_mem("OOM during classifier.to")
        raise

    # Load Image
    diffusion_resize = Resize([args.image_size, args.image_size], Image.BICUBIC)
    start_pil = diffusion_resize(Image.open(args.start_image_path).convert("RGB"))
    start_tensor = th.tensor(np.array(start_pil)).float() / 127.5 - 1
    start_tensor = start_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # Output Dir
    base_name = os.path.splitext(os.path.basename(args.start_image_path))[0]
    out_dir = os.path.join(args.output_dir, base_name, f"meta_cat", f"noise_{args.noise_step}_p{args.penalty}")
    os.makedirs(out_dir, exist_ok=True)
    logger.configure(dir=out_dir)

    run_steering_trajectory(
        args, model, diffusion, classifier, classifier_preprocess, device, start_tensor, out_dir
    )

def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        start_image_path="", 
        output_dir="results/steering_experiments_meta", 
        num_steps=50,       
        noise_step=100,      
        penalty=1.0,        
        batch_size=16,
        max_retries=10,
        classifier_name="convnext_base",
        clip_denoised=True,
        use_ddim=False,
        model_path="/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/models/256x256_diffusion_uncond.pt",
        image_size=256,
        class_cond=False,
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="", 
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
        classifier_use_fp16=False,
        require_target_increase=False,
        target_prob_epsilon=1e-4,
        target_prob_stop=-1.0,
        fail_behavior="force",  # force | skip | stop
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
