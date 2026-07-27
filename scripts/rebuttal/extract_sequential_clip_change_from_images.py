#!/usr/bin/env python3
"""Compute sequential CLIP paths directly from every saved trajectory image."""

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


def step_path(trajectory_dir: Path, step: int) -> Path:
    for suffix in (".jpeg", ".png", ".jpg"):
        path = trajectory_dir / f"uturn_{step:03d}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image for step {step} in {trajectory_dir}")


class TrajectoryDataset(Dataset):
    def __init__(self, trajectory_dirs: list[Path], terminal_step: int):
        self.trajectory_dirs = trajectory_dirs
        self.terminal_step = terminal_step

    def __len__(self) -> int:
        return len(self.trajectory_dirs)

    @staticmethod
    def image_tensor(path: Path) -> torch.Tensor:
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
        trajectory_dir = self.trajectory_dirs[index]
        images = [
            self.image_tensor(step_path(trajectory_dir, step))
            for step in range(self.terminal_step + 1)
        ]
        return {
            "index": torch.tensor(index, dtype=torch.int64),
            "images": torch.stack(images),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    groups = list(frame.groupby("npz_path", sort=False))
    trajectory_dirs = [Path(npz_path).parent for npz_path, _ in groups]
    terminal_step = int(frame["step"].max())
    dataset = TrajectoryDataset(trajectory_dirs, terminal_step)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=1 if args.workers > 0 else None,
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

    net_output = np.full(len(frame), np.nan, dtype=np.float32)
    cumulative_output = np.full(len(frame), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for batch in loader:
            trajectory_index = int(batch["index"].item())
            images = batch["images"].squeeze(0).to(device, non_blocking=True)
            vectors = patch_embeddings(model.visual, images).flatten(1).float()
            vectors = torch.nn.functional.normalize(vectors, dim=1)
            net = (1 - vectors @ vectors[0]).clamp_min(0)
            increments = (1 - (vectors[1:] * vectors[:-1]).sum(dim=1)).clamp_min(0)
            cumulative = torch.cat(
                [
                    torch.zeros(1, device=device),
                    torch.cumsum(increments, dim=0),
                ]
            )
            net = net.cpu().numpy()
            cumulative = cumulative.cpu().numpy()
            _, group = groups[trajectory_index]
            row_indices = group.index.to_numpy(dtype=int)
            steps = group["step"].to_numpy(dtype=int)
            net_output[row_indices] = net[steps]
            cumulative_output[row_indices] = cumulative[steps]

    if not np.isfinite(net_output).all() or not np.isfinite(cumulative_output).all():
        raise RuntimeError("Not all manifest rows received a CLIP path value")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "clip_net_distance_from_images.npy", net_output)
    np.save(
        args.output_dir / "clip_cumulative_path_from_images.npy",
        cumulative_output,
    )
    print(
        f"Wrote {len(frame)} sequential image-derived CLIP rows from "
        f"{len(groups)} trajectories to {args.output_dir}; terminal mean path "
        f"{frame.assign(path=cumulative_output).query('step == @terminal_step')['path'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
