#!/usr/bin/env python3
"""Score audited image-rebuttal manifests with no-reference IQA models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODELS = ("musiq", "topiq_nr")
MANIFEST_COLUMNS = (
    "dataset",
    "source_record_id",
    "image_id",
    "noise_step",
    "rho",
    "trajectory_id",
    "step",
    "image_path",
    "shared20",
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def normalize_uturn_manifest(
    path: Path, dataset: str, shared_image_ids: set[str]
) -> pd.DataFrame:
    frame = read_csv(path).rename(columns={"record_id": "source_record_id"})
    required = {
        "source_record_id",
        "image_id",
        "noise_step",
        "rho",
        "trajectory_id",
        "step",
        "image_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame["dataset"] = dataset
    frame["shared20"] = frame["image_id"].astype(str).isin(shared_image_ids)
    return frame[list(MANIFEST_COLUMNS)]


def normalize_baseline_manifest(path: Path, dataset: str) -> pd.DataFrame:
    frame = read_csv(path)
    required = {"image_id", "image_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame = frame.copy()
    frame["dataset"] = dataset
    frame["source_record_id"] = np.arange(len(frame), dtype=int)
    for column in ("noise_step", "rho", "trajectory_id", "step"):
        frame[column] = ""
    frame["shared20"] = dataset == "original_start"
    return frame[list(MANIFEST_COLUMNS)]


def build_manifest(args: argparse.Namespace) -> None:
    shared = read_csv(args.shared_start_manifest)
    if not {"image_id", "image_path"} <= set(shared.columns):
        raise ValueError("Shared-start manifest must contain image_id and image_path")
    shared_ids = set(shared["image_id"].astype(str))
    frames = [
        normalize_uturn_manifest(args.single_manifest, "single", shared_ids),
        normalize_uturn_manifest(args.sequential_manifest, "sequential", shared_ids),
        normalize_baseline_manifest(args.direct_manifest, "direct_diffusion"),
        normalize_baseline_manifest(args.shared_start_manifest, "original_start"),
    ]
    manifest = pd.concat(frames, ignore_index=True)
    manifest.insert(0, "global_id", np.arange(len(manifest), dtype=int))

    missing_paths = [
        path
        for path in manifest["image_path"].astype(str)
        if not Path(path).is_file()
    ]
    if missing_paths:
        preview = "\n".join(missing_paths[:10])
        raise FileNotFoundError(
            f"{len(missing_paths)} manifest image paths are missing:\n{preview}"
        )
    if not manifest["global_id"].is_unique:
        raise RuntimeError("global_id values are not unique")
    if not manifest["image_path"].is_unique:
        duplicates = manifest.loc[
            manifest["image_path"].duplicated(keep=False),
            ["dataset", "image_path"],
        ]
        raise RuntimeError(f"Image paths are duplicated:\n{duplicates.head(20)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    metadata = {
        "rows": int(len(manifest)),
        "datasets": {
            str(key): int(value)
            for key, value in manifest["dataset"].value_counts().sort_index().items()
        },
        "single_shared20_rows": int(
            (
                manifest["dataset"].eq("single")
                & manifest["shared20"].astype(bool)
            ).sum()
        ),
        "unique_image_paths": int(manifest["image_path"].nunique()),
        "single_manifest": str(args.single_manifest),
        "sequential_manifest": str(args.sequential_manifest),
        "direct_manifest": str(args.direct_manifest),
        "shared_start_manifest": str(args.shared_start_manifest),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def save_scores(path: Path, frame: pd.DataFrame, model: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame[["global_id", model]].to_csv(temporary, index=False)
    os.replace(temporary, path)


def score_shard(args: argparse.Namespace) -> None:
    try:
        import pyiqa
        import torch
    except ImportError as error:
        raise RuntimeError("score-shard requires pyiqa and torch") from error

    if args.model not in pyiqa.list_models():
        raise ValueError(f"Unknown PyIQA model: {args.model}")
    manifest = read_csv(args.manifest)
    expected_ids = manifest.loc[
        manifest["global_id"].astype(int) % args.num_shards == args.shard_id,
        "global_id",
    ].astype(int)
    shard = pd.DataFrame({"global_id": expected_ids.to_numpy()})
    output = (
        args.output_dir
        / f"{args.model}_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        prior = read_csv(output)
        if not prior["global_id"].astype(int).equals(shard["global_id"]):
            raise RuntimeError(f"{output} does not match the current manifest shard")
        shard[args.model] = pd.to_numeric(prior[args.model], errors="coerce")
    else:
        shard[args.model] = np.nan

    paths = manifest.set_index("global_id")["image_path"].astype(str)
    incomplete = shard[args.model].isna()
    print(
        f"{args.model} shard {args.shard_id}/{args.num_shards}: "
        f"{int(incomplete.sum())}/{len(shard)} remaining",
        flush=True,
    )
    if not incomplete.any():
        return

    metric = pyiqa.create_metric(args.model, device=args.device)
    completed_since_save = 0
    with torch.inference_mode():
        for row_index in shard.index[incomplete]:
            global_id = int(shard.at[row_index, "global_id"])
            value = float(
                metric(paths.loc[global_id]).detach().cpu().reshape(-1)[0]
            )
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"{args.model} produced {value} for global_id={global_id}"
                )
            shard.at[row_index, args.model] = value
            completed_since_save += 1
            if completed_since_save >= args.checkpoint_every:
                save_scores(output, shard, args.model)
                completed_since_save = 0
                done = int(shard[args.model].notna().sum())
                print(f"{args.model}: {done}/{len(shard)}", flush=True)
    save_scores(output, shard, args.model)
    print(f"Wrote {output}", flush=True)


def merge_scores(args: argparse.Namespace) -> None:
    manifest = read_csv(args.manifest)
    expected_ids = set(manifest["global_id"].astype(int))
    merged = manifest.copy()
    for model in args.models:
        parts = []
        for shard_id in range(args.num_shards):
            path = (
                args.scores_dir
                / f"{model}_shard_{shard_id:02d}_of_{args.num_shards:02d}.csv"
            )
            part = read_csv(path)
            if part[model].isna().any():
                raise RuntimeError(f"{path} contains incomplete scores")
            parts.append(part[["global_id", model]])
        scores = pd.concat(parts, ignore_index=True)
        if scores["global_id"].duplicated().any():
            raise RuntimeError(f"{model} has duplicate global_id values")
        if set(scores["global_id"].astype(int)) != expected_ids:
            raise RuntimeError(f"{model} global_id coverage does not match manifest")
        merged = merged.merge(scores, on="global_id", how="left", validate="1:1")
    for model in args.models:
        if not np.isfinite(merged[model].to_numpy(dtype=float)).all():
            raise RuntimeError(f"{model} merged scores contain non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    metadata = {
        "rows": int(len(merged)),
        "models": list(args.models),
        "num_shards": int(args.num_shards),
        "all_scores_finite": True,
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest")
    build.add_argument("--single-manifest", type=Path, required=True)
    build.add_argument("--sequential-manifest", type=Path, required=True)
    build.add_argument("--direct-manifest", type=Path, required=True)
    build.add_argument("--shared-start-manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(function=build_manifest)

    score = subparsers.add_parser("score-shard")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--model", choices=DEFAULT_MODELS, required=True)
    score.add_argument("--shard-id", type=int, required=True)
    score.add_argument("--num-shards", type=int, required=True)
    score.add_argument("--device", default="cuda")
    score.add_argument("--checkpoint-every", type=int, default=100)
    score.set_defaults(function=score_shard)

    merge = subparsers.add_parser("merge-scores")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--scores-dir", type=Path, required=True)
    merge.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    merge.add_argument("--num-shards", type=int, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(function=merge_scores)
    return root


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "shard_id") and not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    args.function(args)


if __name__ == "__main__":
    main()
