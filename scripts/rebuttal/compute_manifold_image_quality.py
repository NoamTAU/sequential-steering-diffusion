#!/usr/bin/env python3
"""Compute coverage-independent image realism metrics in Inception space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def pairwise_squared_distance(
    queries: torch.Tensor,
    reference: torch.Tensor,
    reference_norm: torch.Tensor,
) -> torch.Tensor:
    query_norm = queries.square().sum(dim=1, keepdim=True)
    distances = query_norm + reference_norm[None, :] - 2 * queries @ reference.T
    return distances.clamp_min_(0)


def real_manifold_radii(
    reference: torch.Tensor,
    k: int,
    chunk_size: int,
) -> torch.Tensor:
    reference_norm = reference.square().sum(dim=1)
    radii = torch.empty(len(reference), dtype=torch.float32, device=reference.device)
    for start in range(0, len(reference), chunk_size):
        stop = min(start + chunk_size, len(reference))
        distances = pairwise_squared_distance(
            reference[start:stop], reference, reference_norm
        )
        # The nearest entry is the query itself; position k is its kth neighbor.
        radii[start:stop] = torch.topk(
            distances, k=k + 1, dim=1, largest=False
        ).values[:, -1]
        if start % (chunk_size * 10) == 0:
            print(f"reference radii: {stop:,}/{len(reference):,}", flush=True)
    return radii


def evaluate_queries(
    features: np.ndarray,
    reference: torch.Tensor,
    reference_norm: torch.Tensor,
    radii_squared: torch.Tensor,
    k: int,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    precision = np.empty(len(features), dtype=np.float32)
    density = np.empty(len(features), dtype=np.float32)
    nearest_distance = np.empty(len(features), dtype=np.float32)
    realism = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), chunk_size):
        stop = min(start + chunk_size, len(features))
        queries = torch.as_tensor(
            np.asarray(features[start:stop], dtype=np.float32),
            device=reference.device,
        )
        distances = pairwise_squared_distance(queries, reference, reference_norm)
        inside = distances <= radii_squared[None, :]
        precision[start:stop] = inside.any(dim=1).float().cpu().numpy()
        density[start:stop] = inside.sum(dim=1).float().div(k).cpu().numpy()
        nearest_distance[start:stop] = (
            distances.min(dim=1).values.sqrt().cpu().numpy()
        )
        score = (
            radii_squared[None, :]
            .div(distances.clamp_min(1e-12))
            .max(dim=1)
            .values.sqrt()
        )
        realism[start:stop] = score.cpu().numpy()
        if start % (chunk_size * 10) == 0:
            print(f"queries: {stop:,}/{len(features):,}", flush=True)
    return {
        f"manifold_precision_k{k}.npy": precision,
        f"manifold_density_k{k}.npy": density,
        "manifold_nearest_distance.npy": nearest_distance,
        f"manifold_realism_score_k{k}.npy": realism,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("FEATURES", "OUTPUT_DIR"),
        required=True,
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--radii-cache", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    reference_array = np.load(args.reference_features, mmap_mode="r")
    reference = torch.as_tensor(
        np.asarray(reference_array, dtype=np.float32), device=device
    )
    reference_norm = reference.square().sum(dim=1)
    if args.radii_cache.exists():
        radii_squared = torch.as_tensor(
            np.load(args.radii_cache), dtype=torch.float32, device=device
        )
        if len(radii_squared) != len(reference):
            raise RuntimeError(
                f"{args.radii_cache}: expected {len(reference)} radii, "
                f"found {len(radii_squared)}"
            )
    else:
        radii_squared = real_manifold_radii(
            reference, args.k, args.chunk_size
        )
        args.radii_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.radii_cache, radii_squared.cpu().numpy())

    metadata = {
        "reference_features": str(args.reference_features),
        "reference_rows": len(reference),
        "k": args.k,
        "chunk_size": args.chunk_size,
        "device": str(device),
        "definition": (
            "Improved precision/density using kth-neighbor radii of the real "
            "ImageNet Inception-feature manifold"
        ),
    }
    for feature_path_string, output_dir_string in args.dataset:
        feature_path = Path(feature_path_string)
        output_dir = Path(output_dir_string)
        features = np.load(feature_path, mmap_mode="r")
        outputs = evaluate_queries(
            features,
            reference,
            reference_norm,
            radii_squared,
            args.k,
            args.chunk_size,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, values in outputs.items():
            np.save(output_dir / filename, values)
        dataset_metadata = {
            **metadata,
            "features": str(feature_path),
            "rows": len(features),
            "precision": float(outputs[f"manifold_precision_k{args.k}.npy"].mean()),
            "density": float(outputs[f"manifold_density_k{args.k}.npy"].mean()),
            "nearest_distance": float(
                outputs["manifold_nearest_distance.npy"].mean()
            ),
        }
        (output_dir / "manifold_quality_metadata.json").write_text(
            json.dumps(dataset_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(dataset_metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
