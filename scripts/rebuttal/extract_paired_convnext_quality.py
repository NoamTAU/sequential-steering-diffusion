#!/usr/bin/env python3
"""Extract paired ConvNeXt semantic-preservation metrics from a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Base_Weights, convnext_base


class ImageDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with Image.open(self.paths[index]) as handle:
            image = self.transform(handle.convert("RGB"))
        return {
            "index": torch.tensor(index, dtype=torch.int64),
            "image": image,
        }


def model_outputs(model, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = model.features(images)
    features = model.avgpool(features)
    features = model.classifier[0](features)
    features = model.classifier[1](features)
    logits = model.classifier[2](features)
    return features.float(), logits.float()


def infer(
    model,
    paths: list[str],
    transform,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = ImageDataset(paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    features = np.empty((len(paths), 1024), dtype=np.float32)
    logits = np.empty((len(paths), 1000), dtype=np.float32)
    with torch.inference_mode():
        for batch in loader:
            indices = batch["index"].numpy()
            inputs = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                batch_features, batch_logits = model_outputs(model, inputs)
            features[indices] = batch_features.cpu().numpy()
            logits[indices] = batch_logits.cpu().numpy()
    return features, logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest, keep_default_na=False)
    starts = frame[["image_id", "start_path"]].drop_duplicates("image_id")
    if not starts["start_path"].astype(bool).all():
        raise RuntimeError("Every row must have a starting image")
    image_ids = starts["image_id"].tolist()
    image_to_start_index = {image_id: index for index, image_id in enumerate(image_ids)}

    weights = ConvNeXt_Base_Weights.DEFAULT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = convnext_base(weights=weights).to(device).eval()
    start_features, start_logits = infer(
        model,
        starts["start_path"].tolist(),
        weights.transforms(),
        device,
        args.batch_size,
        args.workers,
    )
    output_features, output_logits = infer(
        model,
        frame["image_path"].tolist(),
        weights.transforms(),
        device,
        args.batch_size,
        args.workers,
    )

    start_indices = np.asarray(
        [image_to_start_index[image_id] for image_id in frame["image_id"]]
    )
    matched_start_features = start_features[start_indices]
    normalized_start = matched_start_features / np.maximum(
        np.linalg.norm(matched_start_features, axis=1, keepdims=True), 1e-12
    )
    normalized_output = output_features / np.maximum(
        np.linalg.norm(output_features, axis=1, keepdims=True), 1e-12
    )
    feature_distance = np.maximum(
        0, 1 - np.sum(normalized_start * normalized_output, axis=1)
    )

    start_classes = start_logits.argmax(axis=1)
    matched_start_classes = start_classes[start_indices]
    output_probabilities = torch.from_numpy(output_logits).softmax(dim=1).numpy()
    row_indices = np.arange(len(frame))
    start_class_probability = output_probabilities[
        row_indices, matched_start_classes
    ]
    retained = output_logits.argmax(axis=1) == matched_start_classes

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output_dir / "convnext_feature_cosine_distance.npy",
        feature_distance.astype(np.float32),
    )
    np.save(
        args.output_dir / "convnext_start_class_probability.npy",
        start_class_probability.astype(np.float32),
    )
    np.save(
        args.output_dir / "convnext_start_class_retained.npy",
        retained.astype(np.float32),
    )
    np.save(
        args.output_dir / "convnext_start_class.npy",
        matched_start_classes.astype(np.int16),
    )
    print(
        f"Wrote {len(frame)} paired ConvNeXt rows to {args.output_dir}; "
        f"mean feature distance={feature_distance.mean():.4f}, "
        f"retention={retained.mean():.3f}"
    )


if __name__ == "__main__":
    main()
