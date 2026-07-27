#!/usr/bin/env python3
"""Compute one-step CLIP patch distance for the single-U-turn manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def image_tensor(path: str) -> torch.Tensor:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            image = transform_functional.resize(
                image,
                [256, 256],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            return transform_functional.pil_to_tensor(image).float().div(255.0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        return {
            "index": torch.tensor(index, dtype=torch.int64),
            "start": self.image_tensor(row["start_path"]),
            "end": self.image_tensor(row["image_path"]),
        }


def patch_embeddings(visual_model, images: torch.Tensor) -> torch.Tensor:
    images = transform_functional.resize(
        images,
        [224, 224],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    mean = torch.tensor(
        (0.48145466, 0.4578275, 0.40821073),
        device=images.device,
        dtype=images.dtype,
    )[None, :, None, None]
    std = torch.tensor(
        (0.26862954, 0.26130258, 0.27577711),
        device=images.device,
        dtype=images.dtype,
    )[None, :, None, None]
    x = ((images - mean) / std).to(visual_model.conv1.weight.dtype)
    x = visual_model.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    class_embedding = visual_model.class_embedding.to(x.dtype)
    class_tokens = class_embedding + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([class_tokens, x], dim=1)
    x = x + visual_model.positional_embedding.to(x.dtype)
    x = visual_model.ln_pre(x)
    x = visual_model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    return x[:, 1:, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    dataset = PairDataset(frame)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    distances = np.empty(len(frame), dtype=np.float32)
    with torch.inference_mode():
        for batch in loader:
            indices = batch["index"].numpy()
            start = patch_embeddings(
                model.visual, batch["start"].to(device, non_blocking=True)
            ).flatten(1).float()
            end = patch_embeddings(
                model.visual, batch["end"].to(device, non_blocking=True)
            ).flatten(1).float()
            start = torch.nn.functional.normalize(start, dim=1)
            end = torch.nn.functional.normalize(end, dim=1)
            distances[indices] = (1 - (start * end).sum(dim=1)).cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, distances)
    print(
        f"Wrote {len(distances)} CLIP distances to {args.output}; "
        f"range=({distances.min():.6f}, {distances.max():.6f})"
    )


if __name__ == "__main__":
    main()
