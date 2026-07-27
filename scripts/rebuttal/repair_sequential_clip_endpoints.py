#!/usr/bin/env python3
"""Reconstruct CLIP metrics missing from legacy sequential trajectory caches."""

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


class MissingEndpointDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def image_tensor(path: str | Path) -> torch.Tensor:
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
        previous_path = Path(row["image_path"]).with_name(
            f"uturn_{int(row['step']) - 1:03d}.jpeg"
        )
        if not previous_path.exists():
            raise FileNotFoundError(previous_path)
        return {
            "index": torch.tensor(index, dtype=torch.int64),
            "previous": self.image_tensor(previous_path),
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


def normalized_vectors(embeddings: np.ndarray) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float32).reshape(len(embeddings), -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def find_missing(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for npz_path, group in frame.groupby("npz_path", sort=False):
        with np.load(npz_path, allow_pickle=True) as data:
            length = len(data["embeddings"])
        missing = group[group["step"] >= length]
        rows.extend(missing.to_dict("records"))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    missing = find_missing(frame)
    if missing.empty:
        raise RuntimeError("No missing sequential CLIP endpoints found")

    dataset = MissingEndpointDataset(missing)
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
    previous = np.empty((len(missing), 49 * 768), dtype=np.float32)
    end = np.empty_like(previous)
    with torch.inference_mode():
        for batch in loader:
            indices = batch["index"].numpy()
            previous_batch = patch_embeddings(
                model.visual, batch["previous"].to(device, non_blocking=True)
            ).flatten(1).float()
            end_batch = patch_embeddings(
                model.visual, batch["end"].to(device, non_blocking=True)
            ).flatten(1).float()
            previous[indices] = torch.nn.functional.normalize(
                previous_batch, dim=1
            ).cpu().numpy()
            end[indices] = torch.nn.functional.normalize(
                end_batch, dim=1
            ).cpu().numpy()

    rows = []
    for index, row in missing.reset_index(drop=True).iterrows():
        with np.load(row["npz_path"], allow_pickle=True) as data:
            vectors = normalized_vectors(data["embeddings"])
        expected_step = len(vectors)
        if int(row["step"]) != expected_step:
            raise RuntimeError(
                f"{row['npz_path']}: missing step {row['step']} is not the next "
                f"embedding after length {expected_step}"
            )
        increments = np.maximum(
            0, 1 - np.sum(vectors[1:] * vectors[:-1], axis=1)
        )
        compatibility = max(0.0, 1 - float(np.dot(vectors[-1], previous[index])))
        endpoint_increment = max(
            0.0, 1 - float(np.dot(vectors[-1], end[index]))
        )
        rows.append(
            {
                "record_id": int(row["record_id"]),
                "clip_net_distance": max(
                    0.0, 1 - float(np.dot(vectors[0], end[index]))
                ),
                "clip_cumulative_path": float(increments.sum())
                + endpoint_increment,
                "previous_jpeg_compatibility_distance": compatibility,
                "endpoint_increment": endpoint_increment,
            }
        )

    output = pd.DataFrame(rows).sort_values("record_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(
        f"Wrote {len(output)} endpoint repairs to {args.output}; "
        "previous-JPEG compatibility distance "
        f"mean={output['previous_jpeg_compatibility_distance'].mean():.6f}, "
        f"max={output['previous_jpeg_compatibility_distance'].max():.6f}"
    )


if __name__ == "__main__":
    main()
