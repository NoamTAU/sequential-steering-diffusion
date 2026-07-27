#!/usr/bin/env python3
"""Extract FID Inception features and no-reference classifier diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_fid.inception import InceptionV3
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Base_Weights, convnext_base
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, convnext_transform, include_start: bool):
        self.frame = frame.reset_index(drop=True)
        self.convnext_transform = convnext_transform
        self.include_start = include_start

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def resized_tensor(image: Image.Image) -> torch.Tensor:
        image = transform_functional.resize(
            image,
            [256, 256],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return transform_functional.pil_to_tensor(image).float().div_(255)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as handle:
            image = handle.convert("RGB")
            convnext = self.convnext_transform(image)
            image_256 = self.resized_tensor(image)
            inception = image_256

        pixel_rmse = torch.tensor(float("nan"), dtype=torch.float32)
        pixel_l1 = torch.tensor(float("nan"), dtype=torch.float32)
        if self.include_start:
            with Image.open(row["start_path"]) as handle:
                start = self.resized_tensor(handle.convert("RGB"))
            difference = image_256 - start
            pixel_rmse = difference.square().mean().sqrt()
            pixel_l1 = difference.abs().mean()

        return {
            "index": torch.tensor(index, dtype=torch.int64),
            "inception": inception,
            "convnext": convnext,
            "pixel_rmse": pixel_rmse,
            "pixel_l1": pixel_l1,
        }


def open_array(path: Path, shape: tuple[int, ...], dtype, resume: bool):
    if resume and path.exists():
        array = np.lib.format.open_memmap(path, mode="r+")
        if array.shape != shape:
            raise RuntimeError(f"{path}: expected shape {shape}, found {array.shape}")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--skip-convnext", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest, keep_default_na=False)
    include_start = bool(frame["start_path"].astype(bool).any())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    start_index = 0
    if args.resume and progress_path.exists():
        start_index = int(json.loads(progress_path.read_text())["rows_completed"])

    n_rows = len(frame)
    inception_features = open_array(
        args.output_dir / "inception_features.npy",
        (n_rows, 2048),
        np.float32,
        args.resume,
    )
    pixel_rmse = open_array(
        args.output_dir / "pixel_rmse.npy", (n_rows,), np.float32, args.resume
    )
    pixel_l1 = open_array(
        args.output_dir / "pixel_l1.npy", (n_rows,), np.float32, args.resume
    )
    max_probability = top1 = entropy = None
    if not args.skip_convnext:
        max_probability = open_array(
            args.output_dir / "convnext_max_probability.npy",
            (n_rows,),
            np.float32,
            args.resume,
        )
        entropy = open_array(
            args.output_dir / "convnext_entropy.npy",
            (n_rows,),
            np.float32,
            args.resume,
        )
        top1 = open_array(
            args.output_dir / "convnext_top1.npy",
            (n_rows,),
            np.int16,
            args.resume,
        )

    weights = ConvNeXt_Base_Weights.DEFAULT
    dataset = ManifestDataset(frame, weights.transforms(), include_start)
    subset = torch.utils.data.Subset(dataset, range(start_index, n_rows))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block]).to(device).eval()
    convnext = None
    if not args.skip_convnext:
        convnext = convnext_base(weights=weights).to(device).eval()

    rows_completed = start_index
    with torch.inference_mode():
        for batch in loader:
            indices = batch["index"].numpy()
            fid_input = batch["inception"].to(device, non_blocking=True)
            features = inception(fid_input)[0]
            if features.shape[-2:] != (1, 1):
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1).float().cpu().numpy()
            inception_features[indices] = features
            pixel_rmse[indices] = batch["pixel_rmse"].numpy()
            pixel_l1[indices] = batch["pixel_l1"].numpy()

            if convnext is not None:
                conv_input = batch["convnext"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = convnext(conv_input).float()
                probabilities = logits.softmax(dim=1)
                max_values, top_indices = probabilities.max(dim=1)
                sample_entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
                    dim=1
                )
                max_probability[indices] = max_values.cpu().numpy()
                entropy[indices] = sample_entropy.cpu().numpy()
                top1[indices] = top_indices.cpu().numpy().astype(np.int16)

            rows_completed = int(indices[-1]) + 1
            if rows_completed % (args.batch_size * 10) < args.batch_size:
                progress_path.write_text(
                    json.dumps({"rows_completed": rows_completed}) + "\n"
                )
                inception_features.flush()

    progress_path.write_text(
        json.dumps({"rows_completed": n_rows, "complete": True}) + "\n"
    )
    metadata = {
        "manifest": str(args.manifest),
        "rows": n_rows,
        "device": str(device),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "inception_features": 2048,
        "convnext": not args.skip_convnext,
        "pixel_distances": include_start,
    }
    (args.output_dir / "extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
