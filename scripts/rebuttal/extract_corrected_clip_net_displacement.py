#!/usr/bin/env python3
"""Extract correctly normalized CLIP-patch net displacement from cached images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional


MODEL_NAME = "ViT-B/32"
SEED = 44
BOOTSTRAP_RESAMPLES = 500


def image_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image = transform_functional.resize(
            image,
            [256, 256],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return transform_functional.pil_to_tensor(image).float().div_(255.0)


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        return {
            "index": torch.tensor(index, dtype=torch.int64),
            "start": image_tensor(row["start_path"]),
            "end": image_tensor(row["image_path"]),
        }


class ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "index": torch.tensor(index, dtype=torch.int64),
            "image": image_tensor(self.frame.iloc[index]["image_path"]),
        }


def patch_embeddings(
    visual_model: torch.nn.Module, images: torch.Tensor
) -> torch.Tensor:
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
    return torch.nn.functional.normalize(x[:, 1:, :].flatten(1).float(), dim=1)


def loader(dataset: Dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def load_model(device: torch.device, download_root: Path):
    model, _ = clip.load(
        MODEL_NAME,
        device=device,
        download_root=str(download_root),
    )
    return model.eval()


def extract_images(
    visual_model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    output = None
    with torch.inference_mode():
        for batch_index, batch in enumerate(data_loader):
            indices = batch["index"].numpy()
            vectors = patch_embeddings(
                visual_model,
                batch["image"].to(device, non_blocking=True),
            ).cpu().numpy()
            if output is None:
                output = np.empty((len(data_loader.dataset), vectors.shape[1]))
            output[indices] = vectors
            if (batch_index + 1) % 50 == 0:
                print(f"embedded {indices[-1] + 1}/{len(data_loader.dataset)}")
    if output is None or not np.isfinite(output).all():
        raise RuntimeError("Image embedding extraction was incomplete")
    return output.astype(np.float32, copy=False)


def run_pairs(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.manifest, keep_default_na=False)
    required = {"image_path", "start_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{args.manifest} is missing columns: {missing}")
    if not frame["start_path"].astype(bool).all():
        raise ValueError(f"{args.manifest} contains empty start paths")

    device = torch.device(args.device)
    model = load_model(device, args.download_root)
    distances = np.empty(len(frame), dtype=np.float32)
    data_loader = loader(PairDataset(frame), args.batch_size, args.workers)
    with torch.inference_mode():
        for batch_index, batch in enumerate(data_loader):
            indices = batch["index"].numpy()
            start = patch_embeddings(
                model.visual,
                batch["start"].to(device, non_blocking=True),
            )
            end = patch_embeddings(
                model.visual,
                batch["end"].to(device, non_blocking=True),
            )
            distances[indices] = (
                1.0 - (start * end).sum(dim=1)
            ).clamp_min(0).cpu().numpy()
            if (batch_index + 1) % 50 == 0:
                print(f"scored {indices[-1] + 1}/{len(frame)} pairs")
    if not np.isfinite(distances).all():
        raise RuntimeError("Pair extraction produced non-finite values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, distances)
    metadata = {
        "manifest": str(args.manifest),
        "rows": len(frame),
        "model": MODEL_NAME,
        "image_range_before_clip_normalization": "[0, 1]",
        "embedding": "flattened final CLIP ViT-B/32 patch-token states",
        "distance": "max(0, 1 - cosine(start, output))",
        "minimum": float(distances.min()),
        "mean": float(distances.mean()),
        "maximum": float(distances.max()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


def run_reference(args: argparse.Namespace) -> None:
    starts = pd.read_csv(args.start_manifest, keep_default_na=False)
    direct = pd.read_csv(args.direct_manifest, keep_default_na=False)
    for path, frame in (
        (args.start_manifest, starts),
        (args.direct_manifest, direct),
    ):
        if "image_path" not in frame:
            raise ValueError(f"{path} is missing image_path")

    device = torch.device(args.device)
    model = load_model(device, args.download_root)
    start_vectors = extract_images(
        model.visual,
        loader(ImageDataset(starts), args.batch_size, args.workers),
        device,
    )
    direct_vectors = extract_images(
        model.visual,
        loader(ImageDataset(direct), args.batch_size, args.workers),
        device,
    )
    distances = np.maximum(0.0, 1.0 - start_vectors @ direct_vectors.T)
    estimate = float(distances.mean())

    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for draw in range(BOOTSTRAP_RESAMPLES):
        start_indices = rng.integers(0, len(starts), size=len(starts))
        direct_indices = rng.integers(0, len(direct), size=len(direct))
        bootstrap[draw] = distances[np.ix_(start_indices, direct_indices)].mean()
    low, high = np.quantile(bootstrap, (0.025, 0.975))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "direct_clip_pair_distances.npy", distances)
    pair_table = pd.DataFrame(
        {
            "start_image_id": np.repeat(
                starts["image_id"].astype(str).to_numpy(), len(direct)
            ),
            "direct_image_id": np.tile(
                direct["image_id"].astype(str).to_numpy(), len(starts)
            ),
            "clip_net_distance": distances.reshape(-1),
        }
    )
    pair_table.to_csv(
        args.output_dir / "direct_clip_pair_distances.csv", index=False
    )
    summary = pd.DataFrame(
        [
            {
                "starts": len(starts),
                "direct_samples": len(direct),
                "pairs": distances.size,
                "direct_clip_net_distance": estimate,
                "direct_clip_net_distance_ci_low": float(low),
                "direct_clip_net_distance_ci_high": float(high),
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "seed": SEED,
            }
        ]
    )
    summary.to_csv(
        args.output_dir / "direct_clip_reference_summary.csv", index=False
    )
    metadata = {
        "start_manifest": str(args.start_manifest),
        "direct_manifest": str(args.direct_manifest),
        "model": MODEL_NAME,
        "image_range_before_clip_normalization": "[0, 1]",
        "embedding": "flattened final CLIP ViT-B/32 patch-token states",
        "distance": "max(0, 1 - cosine(start, independent direct sample))",
        "normalization_estimate": estimate,
        "bootstrap_ci": [float(low), float(high)],
    }
    (args.output_dir / "direct_clip_reference_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    pair_parser = subparsers.add_parser("pairs")
    pair_parser.add_argument("--manifest", type=Path, required=True)
    pair_parser.add_argument("--output", type=Path, required=True)
    pair_parser.add_argument("--batch-size", type=int, default=96)
    pair_parser.add_argument("--workers", type=int, default=8)
    pair_parser.add_argument("--device", default="cuda")
    pair_parser.add_argument(
        "--download-root", type=Path, default=Path.home() / ".cache" / "clip"
    )
    pair_parser.set_defaults(handler=run_pairs)

    reference_parser = subparsers.add_parser("reference")
    reference_parser.add_argument("--start-manifest", type=Path, required=True)
    reference_parser.add_argument("--direct-manifest", type=Path, required=True)
    reference_parser.add_argument("--output-dir", type=Path, required=True)
    reference_parser.add_argument("--batch-size", type=int, default=96)
    reference_parser.add_argument("--workers", type=int, default=8)
    reference_parser.add_argument("--device", default="cuda")
    reference_parser.add_argument(
        "--download-root", type=Path, default=Path.home() / ".cache" / "clip"
    )
    reference_parser.set_defaults(handler=run_reference)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
