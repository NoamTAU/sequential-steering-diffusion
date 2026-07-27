#!/usr/bin/env python3
"""Generate independent samples from the paper's unconditional image model."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = create_argparser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    if args.use_fp16:
        model.convert_to_fp16()
    model.to(device).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with torch.inference_mode():
        while written < args.num_samples:
            batch_size = min(args.batch_size, args.num_samples - written)
            samples = diffusion.p_sample_loop(
                model,
                (batch_size, 3, args.image_size, args.image_size),
                clip_denoised=True,
                model_kwargs={},
                device=device,
                progress=False,
            )
            arrays = (
                ((samples + 1) * 127.5)
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            for batch_index, array in enumerate(arrays):
                sample_index = written + batch_index
                Image.fromarray(array).save(
                    output_dir
                    / f"direct_seed_{args.seed:04d}_sample_{sample_index:03d}.jpeg",
                    quality=95,
                )
            written += batch_size
            print(f"seed={args.seed} generated={written}/{args.num_samples}", flush=True)


def create_argparser() -> argparse.ArgumentParser:
    defaults = model_and_diffusion_defaults()
    defaults.update(
        dict(
            output_dir="",
            num_samples=50,
            batch_size=10,
            seed=43,
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
            clip_denoised=True,
            use_ddim=False,
        )
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
