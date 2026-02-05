# """
# Generate a large batch of image samples from a model and save them as a large
# numpy array. This can be used to produce samples for FID evaluation.
# """

# import argparse
# import os
# import sys

# # Add at the top of your file
# from torchvision.transforms import Resize 
# # Add the project's root directory to the Python path
# # This allows the script to find the 'guided_diffusion' module
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

# import clip

# import numpy as np
# import torch as th
# import torch.distributed as dist

# from guided_diffusion import dist_util, logger

# from guided_diffusion.script_util import (
#     NUM_CLASSES,
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict,
# )
# from guided_diffusion.image_datasets import load_data, _list_images_per_classes, _list_image_files_recursively
# import datetime
# import pickle
# from torchvision.transforms import Normalize # <-- ADD THIS LINE
# from torchvision.transforms import Resize      # <-- You added this one before, keep it


# # from torchvision.utils import save_image
# from PIL import Image

# # from more_itertools import ilen
# import time

# # In your main() function
# def main():
#     args = create_argparser().parse_args()

#     dist_util.setup_dist()
#     logger.configure(dir=args.output)

#     logger.log("creating model and diffusion...")
#     model, diffusion = create_model_and_diffusion(
#         **args_to_dict(args, model_and_diffusion_defaults().keys())
#     )
#     model.load_state_dict(
#         dist_util.load_state_dict(args.model_path, map_location="cpu")
#     )
#     model.to(dist_util.dev())
#     if args.use_fp16:
#         model.convert_to_fp16()
#     model.eval()
    
#     logger.log("loading CLIP model...")
#     clip_model, _ = clip.load("ViT-B/32", device=dist_util.dev())
#     clip_model.eval()

#     # The loop now correctly handles the data loader
#     for step_reverse in args.time_series:
#         logger.log(f"Sampling for step_reverse = {step_reverse}")
        
#         # --- THE FIX IS HERE ---
#         # Create a new, fresh data loader for each time step.
#         logger.log("creating data loader...")
#         data_start = load_data(
#             data_dir=args.data_dir,
#             batch_size=args.batch_size,
#             image_size=args.image_size,
#             deterministic=True,
#             class_cond=False, # From our previous fix for the unconditional model
#         )
#         # --- END OF FIX ---

#         sample_loop(
#             model,
#             diffusion,
#             data_start,
#             args,
#             step_reverse=step_reverse,
#             clip_model=clip_model,
#         )



# def sample_loop(model, diffusion, data, args, step_reverse, clip_model):
#     clip_normalize = Normalize(
#         (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
#     )
#     clip_resize = Resize([224, 224])

#     # --- HELPER FUNCTION TO GET PATCH EMBEDDINGS (This part is correct) ---
#     def get_clip_patch_embeddings(visual_model, images):
#         x = images.half()
#         x = visual_model.conv1(x)
#         x = x.reshape(x.shape[0], x.shape[1], -1)
#         x = x.permute(0, 2, 1)
#         class_embedding = visual_model.class_embedding.to(x.dtype)
#         x = th.cat([class_embedding + th.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
#         x = x + visual_model.positional_embedding.to(x.dtype)
#         x = visual_model.ln_pre(x)
#         x = x.permute(1, 0, 2)
#         x = visual_model.transformer(x)
#         x = x.permute(1, 0, 2)
#         return x
#     # --- END OF HELPER FUNCTION ---

#     all_start_embeddings = []
#     all_final_embeddings = []
    
#     num_generated = 0
#     while num_generated < args.num_samples:
#         try:
#             batch_start, _ = next(data)
#         except StopIteration:
#             logger.log("Data loader exhausted.")
#             break

#         current_batch_size = batch_start.shape[0]
#         if current_batch_size == 0: continue

#         batch_start = batch_start.to(dist_util.dev())

#         # Forward process: Add noise
#         t_reverse = diffusion._scale_timesteps(th.tensor([step_reverse])).to(dist_util.dev())
#         batch_noisy = diffusion.q_sample(batch_start, t_reverse)
        
#         # --- THIS BLOCK WAS MISSING ---
#         # Backward process: Denoise the image to create the 'sample' variable
#         model_kwargs = {}
#         sample_fn = (
#             diffusion.p_sample_loop_forw_back if not args.use_ddim else diffusion.ddim_sample_loop_forw_back
#         )
#         sample = sample_fn(
#             model,
#             (current_batch_size, 3, args.image_size, args.image_size),
#             step_reverse=step_reverse,
#             noise=batch_noisy,
#             clip_denoised=args.clip_denoised,
#             model_kwargs=model_kwargs,
#         )
#         # --- END OF MISSING BLOCK ---

#         # --- CLIP EMBEDDING EXTRACTION ---
#         # Now the 'sample' variable exists and can be used here
#         with th.no_grad():
#             start_resized = clip_resize(batch_start)
#             final_resized = clip_resize(sample)
            
#             start_features_full = get_clip_patch_embeddings(clip_model.visual, clip_normalize(start_resized))
#             start_patch_embeddings = start_features_full[:, 1:, :].cpu().numpy()

#             final_features_full = get_clip_patch_embeddings(clip_model.visual, clip_normalize(final_resized))
#             final_patch_embeddings = final_features_full[:, 1:, :].cpu().numpy()
            
#             all_start_embeddings.append(start_patch_embeddings)
#             all_final_embeddings.append(final_patch_embeddings)

#         # ... (rest of the function for saving visuals, logging, etc. is fine) ...
#         num_generated += current_batch_size * dist.get_world_size()
#         logger.log(f"processed {num_generated}/{args.num_samples} samples for step_reverse={step_reverse}")

#     # Code for saving embeddings remains the same...
#     if dist.get_rank() == 0 and all_start_embeddings:
#         # ... (concatenation and saving logic) ...
    
#         dist.barrier()
#         logger.log(f"sampling for step_reverse = {step_reverse} complete")




# def sample_and_save(model, diffusion, data_start, num_samples, output_images, args):
#     logger.log("sampling...")
#     # all_images = []
#     # all_labels = []
#     # all_start_images = []
#     # all_noisy_images = []
#     generated_samples = 0
#     # batch_samples = []
#     g_forw = th.Generator(device=dist_util.dev())
#     g_forw.manual_seed(args.seed_trajectory) if args.seed_trajectory is not None else g_forw.seed()
#     g_back = th.Generator()
#     g_back.manual_seed(args.seed_trajectory) if args.seed_trajectory is not None else g_back.seed()
        
#     time_start = time.time()
#     while generated_samples < num_samples:
#         batch_start, extra = next(data_start)
#         # logger.log(f"batch loaded: {batch_start.shape}")

#         labels_start = extra["y"].to(dist_util.dev())
#         batch_start = batch_start.to(dist_util.dev())
#         img_names = extra["img_name"]
#         # Sample noisy images from the diffusion process at time t_reverse given by the step_reverse argument
#         t_reverse = diffusion._scale_timesteps(th.tensor([args.step_reverse])).to(
#             dist_util.dev()
#         )
#         # t_reverse = t_reverse.to(dist_util.dev())
#         # noise = th.randn_like(batch_start, device=dist_util.dev(), generator=g)
#         noise = th.randn(batch_start.shape, device=dist_util.dev(), generator=g_forw)
#         batch_noisy = (
#             diffusion.q_sample(batch_start, t_reverse, noise=noise)
#             if args.step_reverse < int(args.timestep_respacing)
#             else noise
#         )
#         logger.log("completed forward diffusion...")

#         model_kwargs = {}
#         if args.class_cond:
#             classes = labels_start  # Condition the diffusion on the labels of the original images
#             # classes = th.randint(
#             #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
#             # )
#             model_kwargs["y"] = classes

#         sample_fn = (
#             diffusion.p_sample_loop_forw_back
#             if not args.use_ddim
#             else diffusion.ddim_sample_loop_forw_back
#         )
#         sample = sample_fn(
#             model,
#             (len(batch_start), 3, args.image_size, args.image_size),
#             step_reverse=args.step_reverse,  # step when to reverse the diffusion process
#             noise=batch_noisy,
#             clip_denoised=args.clip_denoised,
#             model_kwargs=model_kwargs,
#             generator=g_back,
#         )
#         logger.log("completed backward diffusion...")
#         sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
#         sample = sample.permute(0, 2, 3, 1)
#         sample = sample.contiguous()

#         # Save the images
#         real_t_reverse = (
#             int(t_reverse.item())
#             if diffusion.rescale_timesteps
#             else int(t_reverse.item() * (1000.0 / float(args.timestep_respacing)))
#         )
#         for ii in range(len(sample)):
#             name = (
#                 img_names[ii].split(".")[0]
#                 + "_t"
#                 + "{:04d}".format(real_t_reverse)
#                 + ".JPEG"
#             )
#             img = Image.fromarray(np.array(sample[ii].cpu()).astype(np.uint8))
#             img.save(os.path.join(output_images, name))
#             # save_image(sample[ii], os.path.join(output_images, name))
#             # save_image(sample[ii], os.path.join(output_images, name), normalize=True, range=(-1, 1))

#         sample_size = th.tensor(len(sample)).to(dist_util.dev())
#         dist.all_reduce(sample_size, op=dist.ReduceOp.SUM)

#         # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
#         # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
#         # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

#         # if args.class_cond:
#         #     gathered_labels = [
#         #         th.zeros_like(classes) for _ in range(dist.get_world_size())
#         #     ]
#         #     dist.all_gather(gathered_labels, classes)
#         #     all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

#         # # Save the start images
#         # batch_start = ((batch_start + 1) * 127.5).clamp(0, 255).to(th.uint8)
#         # batch_start = batch_start.permute(0, 2, 3, 1)
#         # batch_start = batch_start.contiguous()
#         # gathered_start_samples = [th.zeros_like(batch_start) for _ in range(dist.get_world_size())]
#         # dist.all_gather(gathered_start_samples, batch_start)  # gather not supported with NCCL
#         # all_start_images.extend([sample.cpu().numpy() for sample in gathered_start_samples])
#         # Save the noised images
#         # batch_noisy = ((batch_noisy + 1) * 127.5).clamp(0, 255).to(th.uint8)
#         # batch_noisy = batch_noisy.permute(0, 2, 3, 1)
#         # batch_noisy = batch_noisy.contiguous()
#         # gathered_noisy_samples = [th.zeros_like(batch_noisy) for _ in range(dist.get_world_size())]
#         # dist.all_gather(gathered_noisy_samples, batch_noisy)  # gather not supported with NCCL
#         # all_noisy_images.extend([sample.cpu().numpy() for sample in gathered_noisy_samples])

#         sample_size = th.tensor(len(sample)).to(dist_util.dev())
#         dist.all_reduce(sample_size, op=dist.ReduceOp.SUM)
#         generated_samples += sample_size.item()
#         logger.log(
#             f"created {generated_samples} samples in {time.time() - time_start:.1f} seconds"
#         )

#     # arr = np.concatenate(all_images, axis=0)
#     # arr = arr[: num_samples]
#     # if args.class_cond:
#     #     label_arr = np.concatenate(all_labels, axis=0)
#     #     label_arr = label_arr[: num_samples]
#     # arr_start = np.concatenate(all_start_images, axis=0)
#     # arr_start = arr_start[: num_samples]
#     # arr_noisy = np.concatenate(all_noisy_images, axis=0)
#     # arr_noisy = arr_noisy[: num_samples]
#     if dist.get_rank() == 0:
#         # Save the arguments of the run
#         date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
#         out_args = os.path.join(
#             args.output,
#             f"t_{args.step_reverse}_{args.timestep_respacing}_args_{date_time}.pk",
#         )
#         logger.log(f"saving args to {out_args}")
#         with open(out_args, "wb") as handle:
#             pickle.dump(args, handle)

#         # Save the time it took to generate the samples
#         out_time = os.path.join(
#             args.output, f"t_{args.step_reverse}_{args.timestep_respacing}_timing.txt"
#         )
#         with open(out_time, "a") as f:
#             f.write(f"{generated_samples} \t {time.time() - time_start:.3f}\n")

#         # # Save the data of the run
#         # shape_str = "x".join([str(x) for x in arr.shape])
#         # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
#         # logger.log(f"saving to {out_path}")
#         # if args.class_cond:
#         #     np.savez(out_path, arr, label_arr, arr_start, arr_noisy)
#         # else:
#         #     np.savez(out_path, arr, arr_start, arr_noisy)

#     dist.barrier()
#     logger.log("sampling complete")


# # def create_argparser():
# #     defaults = dict(
# #         clip_denoised=True,
# #         num_samples=10000,
# #         batch_size=16,
# #         use_ddim=False,
# #         model_path="",
# #     )
# #     defaults.update(model_and_diffusion_defaults())
# #     defaults.update(
# #         dict(
# #             # step_reverse=10,
# #             data_dir="datasets/ILSVRC2012/validation",
# #             output=os.path.join(
# #                 os.getcwd(), "results", "diffused_ILSVRC2012_validation"
# #             ),
# #             num_per_class=10,
# #             num_classes=10,
# #             all_data=False, # if True, use all data in the dataset and ignore num_per_class and num_classes
# #             seed_trajectory=0,
# #         )
# #     )
# #     parser = argparse.ArgumentParser()
# #     add_dict_to_argparser(parser, defaults)
# #     parser.add_argument(
# #         "--time_series",
# #         nargs="+",
# #         type=int,
# #         default=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
# #         help="Time steps for the step reverse. Pass like: --time_series 25 50 100",
# #     )

# #     # assert parser.parse_args().step_reverse >= 0, "step_reverse must be positive"
# #     # assert parser.parse_args().step_reverse <= int(
# #     #     parser.parse_args().timestep_respacing
# #     # ), "step_reverse must be smaller than or equal to timestep_respacing"

# #     # if parser.parse_args().seed_trajectory is not None:
# #     #     # modify output directory to include seed_trajectory
# #     #     parser.parse_args().output = parser.parse_args().output + f"-seed_{parser.parse_args().seed_trajectory}"

# #     return parser

# # In your create_argparser() function
# def create_argparser():
#     # MODIFIED: These defaults are for the unconditional 256x256 ImageNet model
#     defaults = dict(
#         clip_denoised=True,
#         num_samples=10000,
#         batch_size=16,
#         use_ddim=False,
#         model_path="/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/models/256x256_diffusion_uncond.pt", # MODIFIED: Path to the downloaded model
        
#         # --- Architecture settings for this specific model ---
#         image_size=256,         # MODIFIED
#         class_cond=False,       # MODIFIED: This is the key change!
#         learn_sigma=True,
#         diffusion_steps=1000,
#         noise_schedule="linear",
#         timestep_respacing="",
#         use_kl=False,
#         predict_xstart=False,
#         rescale_timesteps=True,
#         rescale_learned_sigmas=True,
#         num_channels=256,       # MODIFIED
#         num_res_blocks=2,       # MODIFIED
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
#     )

#     # This part remains mostly the same, just update the data_dir suggestion
#     defaults.update(dict(
#         data_dir = '/work/pcsl/Noam/diffusion_datasets', # MODIFIED: Point to your 256x256 ImageNet data
#         output  =  os.path.join(os.getcwd(),
#              'results',
#              'forw_back_uncond', # A new folder for these experiments
#              datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")),
#     ))
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     parser.add_argument(
#         "--time_series",
#         nargs="+",
#         type=int,
#         default=[100, 200, 300, 400, 500], # Adjusted for 1000 steps
#         help="Time steps for the step reverse. Pass like: --time_series 100 200 500",
#     )
#     return parser


# if __name__ == "__main__":
#     main()











"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
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
import torch.distributed as dist
from PIL import Image
from torchvision.transforms import Normalize, Resize

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.image_datasets import load_data


def main():
    args = create_argparser().parse_args()

    # No more dist_util.setup_dist()
    logger.configure(dir=args.output)

    logger.log("creating model and diffusion...")
    device = "cuda" if th.cuda.is_available() else "cpu"
    logger.log(f"Using device: {device}")
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    logger.log("loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    for step_reverse in args.time_series:
        logger.log(f"--- Starting sampling for step_reverse = {step_reverse} ---")
        
        logger.log("creating data loader...")
        data_start = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            deterministic=True,
            class_cond=False,
        )

        # Pass the device to sample_loop
        sample_loop(
            model, diffusion, data_start, args, step_reverse, clip_model, device
        )




def sample_loop(model, diffusion, data, args, step_reverse, clip_model, device):
    clip_normalize = Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    clip_resize = Resize([224, 224])

    def get_clip_patch_embeddings(visual_model, images):
        x = images.half()
        x = visual_model.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        class_embedding = visual_model.class_embedding.to(x.dtype)
        x = th.cat([class_embedding + th.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + visual_model.positional_embedding.to(x.dtype)
        x = visual_model.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual_model.transformer(x)
        x = x.permute(1, 0, 2)
        return x

    all_start_embeddings = []
    all_final_embeddings = []
    
    num_generated = 0
    # first_batch_saved = False
    
    while num_generated < args.num_samples:
        try:
            batch_start, _ = next(data)
        except StopIteration:
            logger.log("Data loader exhausted.")
            break

        current_batch_size = batch_start.shape[0]
        if current_batch_size == 0: continue

        # CORRECTED: Use the simple 'device' variable
        batch_start = batch_start.to(device)

        # CORRECTED: Use the simple 'device' variable
        t_reverse = diffusion._scale_timesteps(th.tensor([step_reverse])).to(device)
        batch_noisy = diffusion.q_sample(batch_start, t_reverse)
        
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop_forw_back if not args.use_ddim else diffusion.ddim_sample_loop_forw_back
        )
        sample = sample_fn(
            model,
            (current_batch_size, 3, args.image_size, args.image_size),
            step_reverse=step_reverse,
            noise=batch_noisy,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        with th.no_grad():
            start_resized = clip_resize(batch_start)
            final_resized = clip_resize(sample)
            
            start_features_full = get_clip_patch_embeddings(clip_model.visual, clip_normalize(start_resized))
            start_patch_embeddings = start_features_full[:, 1:, :].cpu().numpy()

            final_features_full = get_clip_patch_embeddings(clip_model.visual, clip_normalize(final_resized))
            final_patch_embeddings = final_features_full[:, 1:, :].cpu().numpy()
            
            all_start_embeddings.append(start_patch_embeddings)
            all_final_embeddings.append(final_patch_embeddings)



                # --- CORRECTED VISUAL SAVING (for EVERY batch) ---
        visual_dir = os.path.join(logger.get_dir(), "visual_examples", f"t_{step_reverse:04d}")
        os.makedirs(visual_dir, exist_ok=True)
        
        start_images_uint8 = ((batch_start + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for i, img_np in enumerate(start_images_uint8):
            # --- THE FIX IS HERE ---
            # Use a global index (num_generated + i) for a unique filename.
            file_index = num_generated + i
            Image.fromarray(img_np).save(os.path.join(visual_dir, f"start_{file_index:03d}.png"))

        final_images_uint8 = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for i, img_np in enumerate(final_images_uint8):
            # --- AND HERE ---
            file_index = num_generated + i
            Image.fromarray(img_np).save(os.path.join(visual_dir, f"final_{file_index:03d}.png"))
        
        logger.log(f"Saved {current_batch_size} visual examples to {visual_dir} (indices {num_generated} to {num_generated + current_batch_size - 1})")
        # REMOVED: The line `first_batch_saved = True` is no longer needed.
        num_generated += current_batch_size
        logger.log(f"processed {num_generated}/{args.num_samples} samples for step_reverse={step_reverse}")


    # CORRECTED: The distributed check is removed.
    if all_start_embeddings:
        start_embeddings_arr = np.concatenate(all_start_embeddings, axis=0)[:args.num_samples]
        final_embeddings_arr = np.concatenate(all_final_embeddings, axis=0)[:args.num_samples]

        out_path = os.path.join(logger.get_dir(), f"embeddings_step_reverse_{step_reverse}.npz")
        logger.log(f"saving {len(start_embeddings_arr)} embeddings to {out_path}")
        np.savez(out_path, start_embeddings=start_embeddings_arr, final_embeddings=final_embeddings_arr)
    
    logger.log(f"sampling for step_reverse = {step_reverse} complete")



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=4, # Default to a small number for testing
        batch_size=2,   # Default to a small number for testing
        use_ddim=False,
        model_path="/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/models/256x256_diffusion_uncond.pt", # MODIFIED: Path to the downloaded model
        image_size=256,
        class_cond=False,
        learn_sigma=True,
        diffusion_steps=250,
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
    )
    defaults.update(dict(
        data_dir = '/work/pcsl/Noam/diffusion_datasets/selected_images',
        output  =  os.path.join(os.getcwd(),
             'results',
             'forw_back_uncond',
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")),
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(
        "--time_series",
        nargs="+",
        type=int,
        default =  [25, 50, 75, 100, 125, 150, 175, 200, 225, 249],
        help="Time steps for the step reverse. Pass like: --time_series 100 200 500",
    )
    return parser

if __name__ == "__main__":
    main()