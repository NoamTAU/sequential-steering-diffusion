import argparse
import os
import sys
import numpy as np
import torch as th
from PIL import Image
from torchvision.transforms import Normalize, Resize
from tqdm.auto import tqdm  # <--- NEW IMPORT

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

def get_best_proposal(classifier, preprocess, batch_images, orig_idx, target_idx, penalty_weight):
    with th.no_grad():
        clf_dtype = next(classifier.parameters()).dtype
        processed = preprocess(batch_images).to(clf_dtype)
        logits = classifier(processed)
        
        target_logits = logits[:, target_idx]
        orig_logits = logits[:, orig_idx]
        
        # We want Target High AND Original Low
        scores = target_logits - (penalty_weight * orig_logits)
        
        best_idx = th.argmax(scores).item()
        best_score = scores[best_idx].item()
        
        probs = th.nn.functional.softmax(logits, dim=1)
        best_prob_target = probs[best_idx, target_idx].item()
        best_prob_orig = probs[best_idx, orig_idx].item()
        
        return scores, best_idx, best_score, best_prob_target, best_prob_orig, target_logits, orig_logits


def select_auto_target_dog(classifier, preprocess, start_tensor, orig_idx, topk=5):
    with th.no_grad():
        clf_dtype = next(classifier.parameters()).dtype
        logits = classifier(preprocess(start_tensor).to(clf_dtype))
    dog_scores = logits[0, DOG_INDICES]
    order = th.argsort(dog_scores, descending=True).tolist()
    chosen = None
    ranked = []
    for i in order:
        idx = DOG_INDICES[i]
        score = dog_scores[i].item()
        ranked.append((idx, score))
        if chosen is None and idx != orig_idx:
            chosen = idx
    if chosen is None and ranked:
        chosen = ranked[0][0]
    return chosen, ranked[:topk]

def run_steering_trajectory(args, model, diffusion, classifier, classifier_preprocess, device, start_tensor, trajectory_dir):
    
    current_image = start_tensor 
    BATCH_SIZE = args.batch_size 
    MAX_RETRIES = args.max_retries

    # History
    probs_orig = []
    probs_target = []
    attempts_log = [] 
    force_history = []
    force_history_target = []
    force_history_orig = []
    
    # --- Step 0 ---
    with th.no_grad():
        clf_dtype = next(classifier.parameters()).dtype
        logits = classifier(classifier_preprocess(current_image).to(clf_dtype))
        probs = th.nn.functional.softmax(logits, dim=1)
        
        target_logit = logits[0, args.target_class_idx]
        orig_logit = logits[0, args.orig_class_idx]
        current_score = (target_logit - args.penalty * orig_logit).item()
        current_target = target_logit.item()
        current_orig = orig_logit.item()
        
        p_o = probs[0, args.orig_class_idx].item()
        p_t = probs[0, args.target_class_idx].item()
        current_target_prob = p_t
        current_orig_prob = p_o
    
    probs_orig.append(p_o)
    probs_target.append(p_t)
    attempts_log.append(0)

    img_save = ((current_image[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img_save).save(os.path.join(trajectory_dir, "step_000.jpeg"))
    
    np.savez(
        os.path.join(trajectory_dir, "steering_data.npz"),
        probs_orig=np.array(probs_orig),
        probs_target=np.array(probs_target),
        attempts=np.array(attempts_log)
    )
    
    logger.log(f"Starting Steering. Batch: {BATCH_SIZE}. Retries: {MAX_RETRIES}. Noise: {args.noise_step}. Penalty: {args.penalty}")

    with th.no_grad():
        # --- NEW: Outer Progress Bar for Steering Steps ---
        progress_bar = tqdm(range(1, args.num_steps + 1), desc="Steering Progress", unit="step")
        
        for step in progress_bar:
            
            step_accepted = False
            retries = 0
            
            # --- RETRY LOOP ---
            while not step_accepted and retries < MAX_RETRIES:
                th.cuda.empty_cache()
                
                # 1. Forward
                batch_input = current_image.repeat(BATCH_SIZE, 1, 1, 1)
                t_batch = th.tensor([args.noise_step] * BATCH_SIZE, device=device)
                noisy_batch = diffusion.q_sample(batch_input, t_batch)
                
                # 2. Backward (U-Turn)
                sample_img = noisy_batch
                indices = list(range(args.noise_step))[::-1]
                
                # --- NEW: Inner Progress Bar for Diffusion ---
                # leave=False means it disappears after the U-turn is done, keeping logs clean
                # We update the description to show which retry attempt we are on
                for i in tqdm(indices, desc=f"Denoising (Try {retries+1})", leave=False, unit="t"):
                    t = th.tensor([i] * BATCH_SIZE, device=device)
                    out = diffusion.p_sample(
                        model=model,
                        x=sample_img,
                        t=t,
                        clip_denoised=True
                    )
                    sample_img = out["sample"]

                # 3. Evaluate Proposals
                scores, best_idx, best_score, best_prob_t, best_prob_o, target_logits, orig_logits = get_best_proposal(
                    classifier, classifier_preprocess, sample_img, 
                    args.orig_class_idx, args.target_class_idx, args.penalty
                )
                
                # Calculate Force Statistics
                # Change in score for the whole batch relative to current
            
                deltas = scores - current_score
                
                # CORRECT: Use PyTorch methods and extract the value
                mean_force = deltas.mean().item()
                std_force = deltas.std().item()
                
                # Log these to a list 'force_history'
                force_history.append([mean_force, std_force])

                # Per-class force stats
                target_deltas = target_logits - current_target
                orig_deltas = orig_logits - current_orig
                force_history_target.append([target_deltas.mean().item(), target_deltas.std().item()])
                force_history_orig.append([orig_deltas.mean().item(), orig_deltas.std().item()])

                
                # 4. Check Acceptance Criteria
                score_ok = best_score > current_score
                target_ok = (best_prob_t >= current_target_prob + args.target_prob_epsilon) if args.require_target_increase else True
                accept_ok = score_ok and target_ok

                if accept_ok:
                    step_accepted = True
                    
                    # Update State
                    current_image = sample_img[best_idx].unsqueeze(0)
                    
                    score_diff = best_score - current_score
                    current_score = best_score
                    current_target = target_logits[best_idx].item()
                    current_orig = orig_logits[best_idx].item()
                    current_target_prob = best_prob_t
                    current_orig_prob = best_prob_o
                    
                    probs_orig.append(best_prob_o)
                    probs_target.append(best_prob_t)
                    
                    total_proposals = (retries + 1) * BATCH_SIZE
                    attempts_log.append(total_proposals)

                    status = "ACCEPTED"
                    
                    # Log to file/terminal
                    logger.log(f"Step {step}: {status} after {retries+1} batches. Score {best_score:.2f} (Delta {score_diff:.2e}) | Target P: {best_prob_t:.4f}")
                    
                    # Update Progress Bar Description with latest stats
                    progress_bar.set_postfix({"Target P": f"{best_prob_t:.4f}", "Status": status})
                    
                    # Save Image
                    img_save = ((current_image[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
                    Image.fromarray(img_save).save(os.path.join(trajectory_dir, f"step_{step:03d}.jpeg"))

                    np.savez(
                        os.path.join(trajectory_dir, "steering_data.npz"),
                        probs_orig=np.array(probs_orig),
                        probs_target=np.array(probs_target),
                        attempts=np.array(attempts_log)
                    )
                    
                    # Save force history
                    np.savez(
                        os.path.join(trajectory_dir, "force_stats.npz"),
                        force=np.array(force_history),
                        force_target=np.array(force_history_target),
                        force_orig=np.array(force_history_orig),
                    )
                else:
                    retries += 1

                # If we exhausted retries without acceptance, handle fail behavior
                if (not step_accepted) and (retries >= MAX_RETRIES):
                    # Stop early if we've reached target probability threshold
                    if (args.target_prob_stop is not None) and (args.target_prob_stop >= 0) and (current_target_prob >= args.target_prob_stop):
                        logger.log(
                            f"Step {step}: STOP (target_prob {current_target_prob:.4f} >= {args.target_prob_stop:.4f})"
                        )
                        return

                    if args.fail_behavior == "skip":
                        # Keep current image; log a flat step
                        total_proposals = retries * BATCH_SIZE
                        attempts_log.append(total_proposals)
                        probs_orig.append(current_orig_prob)
                        probs_target.append(current_target_prob)
                        status = "SKIP"
                        logger.log(
                            f"Step {step}: {status} after {retries} batches. Score {current_score:.2f} | Target P: {current_target_prob:.4f}"
                        )

                        # Save image (unchanged) to preserve step count
                        img_save = ((current_image[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
                        Image.fromarray(img_save).save(os.path.join(trajectory_dir, f"step_{step:03d}.jpeg"))

                        np.savez(
                            os.path.join(trajectory_dir, "steering_data.npz"),
                            probs_orig=np.array(probs_orig),
                            probs_target=np.array(probs_target),
                            attempts=np.array(attempts_log)
                        )
                        np.savez(
                            os.path.join(trajectory_dir, "force_stats.npz"),
                            force=np.array(force_history),
                            force_target=np.array(force_history_target),
                            force_orig=np.array(force_history_orig),
                        )
                        step_accepted = True
                    elif args.fail_behavior == "force":
                        # Force accept best candidate (may violate monotonic target prob)
                        step_accepted = True
                        current_image = sample_img[best_idx].unsqueeze(0)
                        score_diff = best_score - current_score
                        current_score = best_score
                        current_target = target_logits[best_idx].item()
                        current_orig = orig_logits[best_idx].item()
                        current_target_prob = best_prob_t
                        current_orig_prob = best_prob_o

                        probs_orig.append(best_prob_o)
                        probs_target.append(best_prob_t)
                        attempts_log.append(retries * BATCH_SIZE)

                        status = "FORCED"
                        logger.log(
                            f"Step {step}: {status} after {retries} batches. Score {best_score:.2f} (Delta {score_diff:.2e}) | Target P: {best_prob_t:.4f}"
                        )

                        img_save = ((current_image[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
                        Image.fromarray(img_save).save(os.path.join(trajectory_dir, f"step_{step:03d}.jpeg"))

                        np.savez(
                            os.path.join(trajectory_dir, "steering_data.npz"),
                            probs_orig=np.array(probs_orig),
                            probs_target=np.array(probs_target),
                            attempts=np.array(attempts_log)
                        )
                        np.savez(
                            os.path.join(trajectory_dir, "force_stats.npz"),
                            force=np.array(force_history),
                            force_target=np.array(force_history_target),
                            force_orig=np.array(force_history_orig),
                        )
                    else:
                        # stop
                        logger.log(f"Step {step}: STOP (no acceptable candidate after {retries} batches)")
                        return

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
    if args.classifier_use_fp16:
        if hasattr(classifier, "convert_to_fp16"):
            classifier.convert_to_fp16()
        else:
            classifier.half()

    # Load Image
    diffusion_resize = Resize([args.image_size, args.image_size], Image.BICUBIC)
    start_pil = diffusion_resize(Image.open(args.start_image_path).convert("RGB"))
    start_tensor = th.tensor(np.array(start_pil)).float() / 127.5 - 1
    start_tensor = start_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    auto_info = None
    auto_message = None
    if args.auto_target_dog:
        if args.orig_class_idx not in DOG_INDICES:
            print(f"[auto-target] orig_class_idx {args.orig_class_idx} not in DOG_INDICES. Still selecting top dog class.")
        chosen_idx, topk = select_auto_target_dog(
            classifier, classifier_preprocess, start_tensor, args.orig_class_idx, topk=args.auto_target_topk
        )
        if chosen_idx is not None:
            msg = f"[auto-target] Selected target_class_idx={chosen_idx} (orig={args.orig_class_idx})"
            print(msg)
            args.target_class_idx = int(chosen_idx)
            auto_info = {"orig_class_idx": int(args.orig_class_idx), "target_class_idx": int(chosen_idx), "topk": topk}
            topk_str = ", ".join([f"{idx}:{score:.3f}" for idx, score in auto_info["topk"]])
            auto_message = f"[auto-target] chosen={auto_info['target_class_idx']} topk={topk_str}"
            print(auto_message)
        else:
            print("[auto-target] Failed to select a target class; using provided target_class_idx.")

    # Output Dir
    base_name = os.path.splitext(os.path.basename(args.start_image_path))[0]
    target_dir = f"target_{args.target_class_idx}"
    if args.auto_target_dog:
        target_dir = f"target_auto_{args.target_class_idx}"
    out_dir = os.path.join(args.output_dir, base_name, target_dir, f"noise_{args.noise_step}_p{args.penalty}_b{args.batch_size}_r{args.max_retries}")
    os.makedirs(out_dir, exist_ok=True)
    logger.configure(dir=out_dir)
    if auto_info is not None:
        try:
            import json
            with open(os.path.join(out_dir, "auto_target.json"), "w") as f:
                json.dump(auto_info, f, indent=2)
        except Exception:
            pass
    if auto_message is not None:
        logger.log(auto_message)

    run_steering_trajectory(
        args, model, diffusion, classifier, classifier_preprocess, device, start_tensor, out_dir
    )

def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        start_image_path="", 
        output_dir="results/steering_experiments_v4", 
        num_steps=50,       
        noise_step=100,      
        orig_class_idx=0,   
        target_class_idx=0, 
        penalty=1.0,        
        
        # --- NEW ARGUMENTS ---
        batch_size=16,
        max_retries=10,
        # ---------------------
        
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
        auto_target_dog=False,
        auto_target_topk=5,
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

# import argparse
# import os
# import sys
# import numpy as np
# import torch as th
# from PIL import Image
# from torchvision.transforms import Normalize, Resize

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

# import clip
# from guided_diffusion import logger
# from guided_diffusion.script_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict,
# )
# from guided_diffusion.torch_classifiers import load_classifier

# def get_best_proposal(classifier, preprocess, batch_images, target_idx):
#     """
#     Evaluates a batch of images and returns the index and score of the one 
#     that maximizes the TARGET CLASS LOGIT.
#     """
#     with th.no_grad():
#         # Preprocess batch
#         # batch_images is [B, 3, 256, 256] in range [-1, 1]
#         logits = classifier(preprocess(batch_images))
        
#         # Extract target logits
#         target_logits = logits[:, target_idx]
        
#         best_idx = th.argmax(target_logits).item()
#         best_score = target_logits[best_idx].item()
        
#         # Also get prob for logging
#         probs = th.nn.functional.softmax(logits, dim=1)
#         best_prob = probs[best_idx, target_idx].item()
        
#         return best_idx, best_score, best_prob

# def run_steering_trajectory(args, model, diffusion, classifier, classifier_preprocess, device, start_tensor, trajectory_dir):
    
#     clip_normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
#     clip_resize = Resize([224, 224])
    
#     current_image = start_tensor 
#     BATCH_SIZE = 16 

#     # Lists to store history
#     probs_orig = []
#     probs_target = []
#     attempts_log = [] # In batch mode, attempts is always 1 (since we pick best of batch)

#     # --- Step 0 ---
#     # Calculate initial stats
#     with th.no_grad():
#         logits = classifier(classifier_preprocess(current_image))
#         probs = th.nn.functional.softmax(logits, dim=1)
#         p_o = probs[0, args.orig_class_idx].item()
#         p_t = probs[0, args.target_class_idx].item()
    
#     probs_orig.append(p_o)
#     probs_target.append(p_t)
#     attempts_log.append(0)

#     img_save = ((current_image[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
#     Image.fromarray(img_save).save(os.path.join(trajectory_dir, "step_000.jpeg"))
    
#     # Save Initial Data
#     np.savez(
#         os.path.join(trajectory_dir, "steering_data.npz"),
#         probs_orig=np.array(probs_orig),
#         probs_target=np.array(probs_target),
#         attempts=np.array(attempts_log)
#     )
    
#     logger.log(f"Starting Steering. Batch Size: {BATCH_SIZE}. Noise: {args.noise_step}")

#     with th.no_grad():
#         for step in range(1, args.num_steps + 1):
#             th.cuda.empty_cache()
            
#             # 1. Create BATCH copies
#             batch_input = current_image.repeat(BATCH_SIZE, 1, 1, 1)
            
#             # 2. Add Noise (Forward)
#             t_batch = th.tensor([args.noise_step] * BATCH_SIZE, device=device)
#             noisy_batch = diffusion.q_sample(batch_input, t_batch)
            
#             # 3. Denoise (Backward - U-Turn)
#             sample_img = noisy_batch
#             indices = list(range(args.noise_step))[::-1]
            
#             for i in indices:
#                 t = th.tensor([i] * BATCH_SIZE, device=device)
#                 out = diffusion.p_sample(
#                     model=model,
#                     x=sample_img,
#                     t=t,
#                     clip_denoised=True
#                 )
#                 sample_img = out["sample"]
                
#             # 4. Evaluate Proposals
#             best_idx, best_score, best_prob = get_best_proposal(
#                 classifier, classifier_preprocess, sample_img, args.target_class_idx
#             )
            
#             # Get stats for the chosen image
#             # We need to re-run classifier or extract from the batch used in get_best_proposal
#             # get_best_proposal returns best_prob (target), but we also want orig prob.
#             # Let's just re-run eval on the single chosen image to be clean/easy.
            
#             # 5. Update State
#             current_image = sample_img[best_idx].unsqueeze(0)
            
#             # Recalculate full stats for the winner
#             logits = classifier(classifier_preprocess(current_image))
#             probs = th.nn.functional.softmax(logits, dim=1)
#             p_o = probs[0, args.orig_class_idx].item()
#             p_t = probs[0, args.target_class_idx].item()
            
#             probs_orig.append(p_o)
#             probs_target.append(p_t)
#             attempts_log.append(1) # Batch mode = 1 attempt (consisting of 16 candidates)

#             logger.log(f"Step {step}: Selected proposal {best_idx}. Target P: {p_t:.4f}")
            
#             # Save Image
#             img_save = ((current_image[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
#             Image.fromarray(img_save).save(os.path.join(trajectory_dir, f"step_{step:03d}.jpeg"))

#             # --- SAVE DATA ---
#             np.savez(
#                 os.path.join(trajectory_dir, "steering_data.npz"),
#                 probs_orig=np.array(probs_orig),
#                 probs_target=np.array(probs_target),
#                 attempts=np.array(attempts_log)
#             )


# def main():
#     args = create_argparser().parse_args()
#     device = "cuda" if th.cuda.is_available() else "cpu"
    
#     # Load Models
#     model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
#     model.load_state_dict(th.load(args.model_path, map_location="cpu"))
#     model.to(device); model.eval()
    
#     if args.use_fp16: model.convert_to_fp16()

#     classifier, classifier_preprocess, _ = load_classifier(args.classifier_name)
#     classifier.to(device); classifier.eval()
#     if args.classifier_use_fp16: classifier.convert_to_fp16()

#     # Load Image
#     diffusion_resize = Resize([args.image_size, args.image_size], Image.BICUBIC)
#     start_pil = diffusion_resize(Image.open(args.start_image_path).convert("RGB"))
#     start_tensor = th.tensor(np.array(start_pil)).float() / 127.5 - 1
#     start_tensor = start_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

#     # Output Dir
#     base_name = os.path.splitext(os.path.basename(args.start_image_path))[0]
#     out_dir = os.path.join(args.output_dir, base_name, f"target_{args.target_class_idx}", f"noise_{args.noise_step}")
#     os.makedirs(out_dir, exist_ok=True)
#     logger.configure(dir=out_dir)

#     run_steering_trajectory(
#         args, model, diffusion, classifier, classifier_preprocess, device, start_tensor, out_dir
#     )

# def create_argparser():
#     defaults = model_and_diffusion_defaults()
#     defaults.update(dict(
#         start_image_path="", 
#         output_dir="results/steering_experiments_v2",
#         num_steps=50,       
#         noise_step=100,      
#         orig_class_idx=0,   
#         target_class_idx=0, 
#         classifier_name="convnext_base",
#         clip_denoised=True,
#         use_ddim=False,
#         model_path="/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/models/256x256_diffusion_uncond.pt",
#         image_size=256,
#         class_cond=False,
#         learn_sigma=True,
#         diffusion_steps=1000,
#         noise_schedule="linear",
#         timestep_respacing="", 
#         use_fp16=True,
#         num_channels=256,
#         num_res_blocks=2,
#         num_heads=4,
#         num_heads_upsample=-1,
#         use_scale_shift_norm=True,
#         dropout=0.0,
#         attention_resolutions="32,16,8",
#         channel_mult="",
#         use_checkpoint=False,
#         num_head_channels=64,
#         resblock_updown=True,
#         use_new_attention_order=False,
#         classifier_use_fp16=False
#     ))
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser

# if __name__ == "__main__":
#     main()


# # """
# # Performs STEERED sequential U-turns.
# # It repeatedly proposes a U-turn and only accepts it if the classifier probability
# # for a target class increases relative to the original class.
# # """

# # import argparse
# # import os
# # import sys
# # import glob
# # import numpy as np
# # import torch as th
# # from PIL import Image
# # from torchvision.transforms import Normalize, Resize
# # import torch.nn.functional as F

# # # Add project root
# # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# # sys.path.insert(0, project_root)

# # import clip
# # from guided_diffusion import logger
# # from guided_diffusion.script_util import (
# #     model_and_diffusion_defaults,
# #     create_model_and_diffusion,
# #     add_dict_to_argparser,
# #     args_to_dict,
# # )
# # from guided_diffusion.torch_classifiers import load_classifier

# # # --- HELPER: Classifier Scoring ---
# # def get_classifier_score(classifier, preprocess, image_tensor, original_idx, target_idx):
# #     """
# #     Returns a score: Prob(Target) - Prob(Original).
# #     We want to MAXIMIZE this.
# #     """
# #     with th.no_grad():
# #         # image_tensor is [1, 3, 256, 256] in range [-1, 1]
# #         logits = classifier(preprocess(image_tensor))
# #         probs = F.softmax(logits, dim=1)
        
# #         prob_orig = probs[0, original_idx].item()
# #         prob_target = probs[0, target_idx].item()
        
# #         # Score: We want high target prob, low original prob
# #         return prob_target - prob_orig, prob_orig, prob_target

# # # --- HELPER: Diffusion U-Turn (Manual Loop Fix) ---
# # def perform_single_uturn(model, diffusion, start_tensor, noise_step, device):
# #     with th.no_grad():
# #         start_tensor = start_tensor.to(device)
        
# #         # 1. Forward Process: Add noise
# #         t_batch = th.tensor([noise_step] * start_tensor.shape[0], device=device)
# #         noisy_image = diffusion.q_sample(start_tensor, t_batch)
        
# #         # 2. Backward Process: Denoise step-by-step
# #         # Manual loop to avoid AttributeError on SpacedDiffusion
# #         img = noisy_image
# #         indices = list(range(noise_step))[::-1]
        
# #         for i in indices:
# #             t = th.tensor([i] * img.shape[0], device=device)
# #             out = diffusion.p_sample(
# #                 model=model,
# #                 x=img,
# #                 t=t,
# #                 clip_denoised=True,
# #                 model_kwargs={}
# #             )
# #             img = out["sample"]
            
# #         return img

# # def get_clip_patch_embeddings(visual_model, images, clip_normalize, clip_resize):
# #     with th.no_grad():
# #         resized_images = clip_resize(images)
# #         normalized_images = clip_normalize(resized_images).half()
# #         x = normalized_images
# #         x = visual_model.conv1(x)
# #         x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
# #         class_embedding = visual_model.class_embedding.to(x.dtype)
# #         x = th.cat([class_embedding + th.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
# #         x = x + visual_model.positional_embedding.to(x.dtype)
# #         x = visual_model.ln_pre(x)
# #         x = x.permute(1, 0, 2)
# #         x = visual_model.transformer(x)
# #         x = x.permute(1, 0, 2)
# #         return x[:, 1:, :]

# # # --- MAIN LOGIC ---
# # def run_steering_trajectory(args, model, diffusion, classifier, classifier_preprocess, clip_model, device, start_tensor, trajectory_dir):
    
# #     clip_normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
# #     clip_resize = Resize([224, 224])

# #     current_image_tensor = start_tensor
    
# #     # Tracking Data
# #     embeddings = []
# #     scores = []
# #     probs_orig = []
# #     probs_target = []
# #     attempts_log = [] # How many tries did it take to accept?

# #     # --- Step 0: Initial State ---
# #     score, p_o, p_t = get_classifier_score(classifier, classifier_preprocess, current_image_tensor, args.orig_class_idx, args.target_class_idx)
    
# #     logger.log(f"Step 0: Orig P={p_o:.4f}, Target P={p_t:.4f}, Score={score:.4f}")
    
# #     # Save Image 0
# #     img_save = ((current_image_tensor[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
# #     Image.fromarray(img_save).save(os.path.join(trajectory_dir, "step_000.jpeg"))
    
# #     # Save Stats 0
# #     embedding = get_clip_patch_embeddings(clip_model.visual, current_image_tensor, clip_normalize, clip_resize)
# #     embeddings.append(embedding.cpu().numpy())
# #     scores.append(score)
# #     probs_orig.append(p_o)
# #     probs_target.append(p_t)
# #     attempts_log.append(0)

# #     # --- Steering Loop ---
# #     for i in range(1, args.num_steps + 1):
# #         accepted = False
# #         attempts = 0
        
# #         while not accepted:
# #             attempts += 1
            
# #             # 1. Propose New Image
# #             proposal_tensor = perform_single_uturn(model, diffusion, current_image_tensor, args.noise_step, device)
            
# #             # 2. Evaluate
# #             new_score, new_p_o, new_p_t = get_classifier_score(classifier, classifier_preprocess, proposal_tensor, args.orig_class_idx, args.target_class_idx)
            
# #             # 3. Accept/Reject (Greedy Hill Climbing)
# #             # Accept if score improves (gets larger)
# #             if new_score > score:
# #                 accepted = True
# #                 current_image_tensor = proposal_tensor
# #                 score = new_score
# #                 p_o = new_p_o
# #                 p_t = new_p_t
# #                 logger.log(f"Step {i}: ACCEPTED after {attempts} tries. Score: {score:.4f} (Orig: {p_o:.2f}, Target: {p_t:.2f})")
# #             else:
# #                 # Optional: break if too many attempts to avoid infinite loops
# #                 if attempts > 50:
# #                     logger.log(f"Step {i}: Stuck after 50 tries. Forcing acceptance of last attempt to move on.")
# #                     accepted = True
# #                     current_image_tensor = proposal_tensor
# #                     score = new_score
# #                     p_o = new_p_o
# #                     p_t = new_p_t

# #         # --- Save Step ---
# #         img_save = ((current_image_tensor[0] + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).cpu().numpy()
# #         Image.fromarray(img_save).save(os.path.join(trajectory_dir, f"step_{i:03d}.jpeg"))
        
# #         embedding = get_clip_patch_embeddings(clip_model.visual, current_image_tensor, clip_normalize, clip_resize)
# #         embeddings.append(embedding.cpu().numpy())
# #         scores.append(score)
# #         probs_orig.append(p_o)
# #         probs_target.append(p_t)
# #         attempts_log.append(attempts)

# #         # Incrementally save data
# #         np.savez(
# #             os.path.join(trajectory_dir, "steering_data.npz"),
# #             embeddings=np.concatenate(embeddings, axis=0),
# #             scores=np.array(scores),
# #             probs_orig=np.array(probs_orig),
# #             probs_target=np.array(probs_target),
# #             attempts=np.array(attempts_log)
# #         )

# # def main():
# #     args = create_argparser().parse_args()
    
# #     device = "cuda" if th.cuda.is_available() else "cpu"
    
# #     # 1. Load Diffusion
# #     logger.log("Loading Diffusion Model...")
# #     model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
# #     model.load_state_dict(th.load(args.model_path, map_location="cpu"))
# #     model.to(device)
# #     if args.use_fp16: model.convert_to_fp16()
# #     model.eval()

# #     # 2. Load Classifier
# #     logger.log(f"Loading Classifier {args.classifier_name}...")
# #     classifier, classifier_preprocess, _ = load_classifier(args.classifier_name)
# #     classifier.to(device)
# #     classifier.eval()

# #     # 3. Load CLIP
# #     logger.log("Loading CLIP...")
# #     clip_model, _ = clip.load("ViT-B/32", device=device)
# #     clip_model.eval()

# #     # 4. Prepare Start Image
# #     logger.log(f"Loading Image: {args.start_image_path}")
# #     diffusion_resize = Resize([args.image_size, args.image_size], interpolation=Image.BICUBIC)
# #     start_pil = diffusion_resize(Image.open(args.start_image_path).convert("RGB"))
# #     start_tensor = th.tensor(np.array(start_pil)).float() / 127.5 - 1
# #     start_tensor = start_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

# #     # 5. Output Dir
# #     base_name = os.path.splitext(os.path.basename(args.start_image_path))[0]
# #     out_dir = os.path.join(args.output_dir, base_name, f"target_{args.target_class_idx}", f"noise_{args.noise_step}")
# #     os.makedirs(out_dir, exist_ok=True)
# #     logger.configure(dir=out_dir)

# #     # 6. Run Experiment
# #     run_steering_trajectory(
# #         args, model, diffusion, classifier, classifier_preprocess, clip_model, device,
# #         start_tensor, out_dir
# #     )

# # def create_argparser():
# #     defaults = model_and_diffusion_defaults()
# #     defaults.update(dict(
# #         start_image_path="", 
# #         output_dir="results/steering_experiments",
# #         num_steps=100,      # How many ACCEPTED steps to take
# #         noise_step=100,     # Noise level per U-turn
        
# #         # Steering Targets
# #         orig_class_idx=0,   # e.g. 250 (Husky)
# #         target_class_idx=0, # e.g. 281 (Tabby Cat)
# #         classifier_name="convnext_base",

# #         # Model Config (Absolute path + 1000 step config)
# #         clip_denoised=True,
# #         use_ddim=False,
# #         model_path="/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/models/256x256_diffusion_uncond.pt",
# #         image_size=256,
# #         class_cond=False,
# #         learn_sigma=True,
# #         diffusion_steps=1000,
# #         noise_schedule="linear",
# #         timestep_respacing="", # 1000 steps (matches sequential experiment)
# #         use_fp16=True,
# #         num_channels=256,
# #         num_res_blocks=2,
# #         num_heads=4,
# #         num_heads_upsample=-1,
# #         use_scale_shift_norm=True,
# #         dropout=0.0,
# #         attention_resolutions="32,16,8",
# #         channel_mult="",
# #         use_checkpoint=False,
# #         num_head_channels=64,
# #         resblock_updown=True,
# #         use_new_attention_order=False,
# #     ))
# #     parser = argparse.ArgumentParser()
# #     add_dict_to_argparser(parser, defaults)
# #     return parser

# # if __name__ == "__main__":
# #     main()
