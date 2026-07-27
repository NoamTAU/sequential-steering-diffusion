#!/usr/bin/env python3
"""Classifier-energy Metropolis-Hastings steering with U-turn proposals."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from guided_diffusion import logger
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from guided_diffusion.torch_classifiers import load_classifier


DOG_INDICES = list(range(151, 269))
CAT_INDICES = list(range(281, 286))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classify(
    classifier,
    preprocess,
    images: torch.Tensor,
    target_mode: str,
    orig_class_idx: int,
    target_class_idx: int,
) -> dict[str, torch.Tensor]:
    classifier_dtype = next(classifier.parameters()).dtype
    logits = classifier(preprocess(images).to(classifier_dtype)).float()
    probabilities = logits.softmax(dim=1)
    dog_probability = probabilities[:, DOG_INDICES].sum(dim=1)
    cat_probability = probabilities[:, CAT_INDICES].sum(dim=1)
    orig_probability = probabilities[:, orig_class_idx]
    target_class_probability = probabilities[:, target_class_idx]
    if target_mode == "cat":
        target_probability = cat_probability
        source_probability = dog_probability
    elif target_mode == "dog_class":
        target_probability = target_class_probability
        source_probability = orig_probability
    else:
        raise ValueError(f"Unknown target mode: {target_mode}")
    return {
        "target": target_probability,
        "source": source_probability,
        "dog": dog_probability,
        "cat": cat_probability,
        "orig": orig_probability,
        "target_class": target_class_probability,
        "top1": probabilities.argmax(dim=1),
        "max_probability": probabilities.max(dim=1).values,
    }


def single_uturn(model, diffusion, image, noise_step: int) -> torch.Tensor:
    device = image.device
    timestep = torch.full((len(image),), noise_step, device=device, dtype=torch.long)
    sample = diffusion.q_sample(image, timestep)
    for index in range(noise_step - 1, -1, -1):
        timestep = torch.full((len(image),), index, device=device, dtype=torch.long)
        sample = diffusion.p_sample(
            model=model,
            x=sample,
            t=timestep,
            clip_denoised=True,
            model_kwargs={},
        )["sample"]
    return sample


def save_image(image: torch.Tensor, path: Path) -> None:
    array = (
        ((image[0] + 1) * 127.5)
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(array).save(path)


def scalar(metrics: dict[str, torch.Tensor], key: str) -> float:
    return float(metrics[key][0].item())


def run_chain(
    args,
    model,
    diffusion,
    classifier,
    preprocess,
    start_image: torch.Tensor,
    output_dir: Path,
) -> None:
    current = start_image
    current_metrics = classify(
        classifier,
        preprocess,
        current,
        args.target_mode,
        args.orig_class_idx,
        args.target_class_idx,
    )
    history = {
        "target_probability": [scalar(current_metrics, "target")],
        "source_probability": [scalar(current_metrics, "source")],
        "dog_probability": [scalar(current_metrics, "dog")],
        "cat_probability": [scalar(current_metrics, "cat")],
        "orig_probability": [scalar(current_metrics, "orig")],
        "target_class_probability": [scalar(current_metrics, "target_class")],
        "max_probability": [scalar(current_metrics, "max_probability")],
        "top1": [int(current_metrics["top1"][0].item())],
        "accepted": [True],
        "log_acceptance_ratio": [0.0],
        "proposal_target_probability": [scalar(current_metrics, "target")],
        "proposal_source_probability": [scalar(current_metrics, "source")],
    }
    save_image(current, output_dir / "step_000.jpeg")

    with torch.inference_mode():
        for step in range(1, args.num_steps + 1):
            proposal = single_uturn(model, diffusion, current, args.noise_step)
            proposal_metrics = classify(
                classifier,
                preprocess,
                proposal,
                args.target_mode,
                args.orig_class_idx,
                args.target_class_idx,
            )
            current_target = max(scalar(current_metrics, "target"), args.energy_floor)
            proposal_target = max(
                scalar(proposal_metrics, "target"), args.energy_floor
            )
            log_ratio = args.energy_lambda * (
                math.log(proposal_target) - math.log(current_target)
            )
            log_acceptance = min(0.0, log_ratio)
            accepted = math.log(max(random.random(), 1e-12)) < log_acceptance
            if accepted:
                current = proposal
                current_metrics = proposal_metrics

            history["target_probability"].append(scalar(current_metrics, "target"))
            history["source_probability"].append(scalar(current_metrics, "source"))
            history["dog_probability"].append(scalar(current_metrics, "dog"))
            history["cat_probability"].append(scalar(current_metrics, "cat"))
            history["orig_probability"].append(scalar(current_metrics, "orig"))
            history["target_class_probability"].append(
                scalar(current_metrics, "target_class")
            )
            history["max_probability"].append(
                scalar(current_metrics, "max_probability")
            )
            history["top1"].append(int(current_metrics["top1"][0].item()))
            history["accepted"].append(accepted)
            history["log_acceptance_ratio"].append(log_ratio)
            history["proposal_target_probability"].append(proposal_target)
            history["proposal_source_probability"].append(
                scalar(proposal_metrics, "source")
            )
            save_image(current, output_dir / f"step_{step:03d}.jpeg")
            np.savez(
                output_dir / "mh_data.npz",
                **{key: np.asarray(values) for key, values in history.items()},
            )
            logger.log(
                f"step={step} accepted={int(accepted)} target="
                f"{history['target_probability'][-1]:.5f} source="
                f"{history['source_probability'][-1]:.5f} log_ratio={log_ratio:.3f}"
            )


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

    classifier, preprocess, _ = load_classifier(args.classifier_name)
    classifier.to(device).eval()

    resize = Resize([args.image_size, args.image_size], Image.BICUBIC)
    start_pil = resize(Image.open(args.start_image_path).convert("RGB"))
    start = torch.tensor(np.asarray(start_pil).copy()).float().div(127.5).sub(1)
    start = start.permute(2, 0, 1).unsqueeze(0).to(device)

    initial = classify(
        classifier,
        preprocess,
        start,
        args.target_mode,
        args.orig_class_idx,
        args.target_class_idx,
    )
    if int(initial["top1"][0].item()) != args.orig_class_idx:
        raise RuntimeError(
            f"Start-image top1 is {int(initial['top1'][0])}, expected "
            f"{args.orig_class_idx}"
        )

    image_name = Path(args.start_image_path).stem
    run_name = (
        f"mode_{args.target_mode}_rho_{args.noise_step:03d}_"
        f"lambda_{args.energy_lambda:g}_rep_{args.repeat_index:02d}"
    )
    output_dir = Path(args.output_dir) / image_name / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=str(output_dir))
    config = {
        "start_image_path": args.start_image_path,
        "image_name": image_name,
        "target_mode": args.target_mode,
        "orig_class_idx": args.orig_class_idx,
        "target_class_idx": args.target_class_idx,
        "noise_step": args.noise_step,
        "rho": args.noise_step / 1000,
        "energy_definition": "H(x) = -lambda * log(p_target(x) + floor)",
        "energy_lambda": args.energy_lambda,
        "energy_floor": args.energy_floor,
        "num_steps": args.num_steps,
        "repeat_index": args.repeat_index,
        "seed": args.seed,
        "initial_target_probability": scalar(initial, "target"),
        "initial_source_probability": scalar(initial, "source"),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    run_chain(
        args,
        model,
        diffusion,
        classifier,
        preprocess,
        start,
        output_dir,
    )


def create_argparser() -> argparse.ArgumentParser:
    defaults = model_and_diffusion_defaults()
    defaults.update(
        dict(
            start_image_path="",
            output_dir="/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx/mh_steering",
            target_mode="cat",
            orig_class_idx=183,
            target_class_idx=189,
            noise_step=200,
            energy_lambda=4.0,
            energy_floor=1e-8,
            num_steps=50,
            repeat_index=0,
            seed=43,
            classifier_name="convnext_base",
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
