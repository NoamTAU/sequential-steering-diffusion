#!/usr/bin/env python3
"""Build balanced manifests for cached image-quality analyses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SINGLE_NOISES = (25, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999)
SEQUENTIAL_NOISES = (100, 200, 400, 600, 800, 999)
SEQUENTIAL_STEPS = (1, 2, 5, 10, 20, 40, 50, 60, 80, 100)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def trajectory_index(path: Path) -> int:
    return int(path.name.rsplit("_", 1)[1])


def step_path(trajectory_dir: Path, step: int) -> Path | None:
    for suffix in (".jpeg", ".png", ".jpg"):
        path = trajectory_dir / f"uturn_{step:03d}{suffix}"
        if path.exists():
            return path
    return None


def complete_trajectories(noise_dir: Path, terminal_step: int) -> list[Path]:
    paths = []
    for trajectory_dir in noise_dir.glob("trajectory_*"):
        if not trajectory_dir.is_dir():
            continue
        if not (trajectory_dir / "trajectory_data.npz").exists():
            continue
        if step_path(trajectory_dir, terminal_step) is None:
            continue
        paths.append(trajectory_dir)
    return sorted(paths, key=trajectory_index)


def read_image_names(path: Path) -> list[str]:
    return [Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()]


def single_common_images(root: Path) -> list[str]:
    image_sets = []
    for noise in SINGLE_NOISES:
        image_sets.append(
            {
                image_dir.name
                for image_dir in root.glob("*")
                if complete_trajectories(image_dir / f"noise_step_{noise}", 1)
            }
        )
    return sorted(set.intersection(*image_sets))


def build_single(root: Path, trajectories_per_image: int) -> list[dict]:
    rows = []
    common_images = single_common_images(root)
    for image_name in common_images:
        for noise in SINGLE_NOISES:
            trajectories = complete_trajectories(
                root / image_name / f"noise_step_{noise}", 1
            )
            if len(trajectories) < trajectories_per_image:
                raise RuntimeError(
                    f"{image_name}, noise {noise}: found {len(trajectories)} complete "
                    f"pairs, need {trajectories_per_image}"
                )
            for trajectory_dir in trajectories[:trajectories_per_image]:
                trajectory_id = trajectory_index(trajectory_dir)
                output_path = step_path(trajectory_dir, 1)
                start_path = step_path(trajectory_dir, 0)
                if output_path is None or start_path is None:
                    raise RuntimeError(f"Missing single-U-turn image in {trajectory_dir}")
                rows.append(
                    {
                        "dataset": "single",
                        "image_id": image_name,
                        "noise_step": noise,
                        "rho": noise / 1000,
                        "trajectory_id": trajectory_id,
                        "step": 1,
                        "image_path": str(output_path),
                        "start_path": str(start_path),
                        "npz_path": str(trajectory_dir / "trajectory_data.npz"),
                    }
                )
    return rows


def build_sequential(
    root: Path, image_names: list[str], trajectories_per_image: int
) -> list[dict]:
    rows = []
    for image_name in image_names:
        for noise in SEQUENTIAL_NOISES:
            trajectories = complete_trajectories(
                root / image_name / f"noise_step_{noise}", 100
            )
            if len(trajectories) < trajectories_per_image:
                raise RuntimeError(
                    f"{image_name}, noise {noise}: found {len(trajectories)} complete "
                    f"trajectories, need {trajectories_per_image}"
                )
            for trajectory_dir in trajectories[:trajectories_per_image]:
                trajectory_id = trajectory_index(trajectory_dir)
                start_path = step_path(trajectory_dir, 0)
                if start_path is None:
                    raise RuntimeError(f"Missing initial image in {trajectory_dir}")
                for step in SEQUENTIAL_STEPS:
                    output_path = step_path(trajectory_dir, step)
                    if output_path is None:
                        raise RuntimeError(
                            f"Missing U-turn step {step} in {trajectory_dir}"
                        )
                    rows.append(
                        {
                            "dataset": "sequential",
                            "image_id": image_name,
                            "noise_step": noise,
                            "rho": noise / 1000,
                            "trajectory_id": trajectory_id,
                            "step": step,
                            "image_path": str(output_path),
                            "start_path": str(start_path),
                            "npz_path": str(trajectory_dir / "trajectory_data.npz"),
                        }
                    )
    return rows


def build_reference(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rows.append(
            {
                "dataset": "reference",
                "image_id": path.stem,
                "noise_step": "",
                "rho": "",
                "trajectory_id": "",
                "step": "",
                "image_path": str(path),
                "start_path": "",
                "npz_path": "",
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "dataset",
        "image_id",
        "noise_step",
        "rho",
        "trajectory_id",
        "step",
        "image_path",
        "start_path",
        "npz_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record_id, row in enumerate(rows):
            writer.writerow({"record_id": record_id, **row})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/work/pcsl/Noam/sequential_diffusion"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/work/pcsl/Noam/diffusion_datasets"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--single-trajectories-per-image", type=int, default=40)
    parser.add_argument("--sequential-trajectories-per-image", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    single = build_single(
        args.root / "results/single_uturn_baseline",
        args.single_trajectories_per_image,
    )
    sequential = build_sequential(
        args.root / "results/sequential_uturns",
        read_image_names(args.root / "metadata/high_noise_image_list.txt"),
        args.sequential_trajectories_per_image,
    )
    reference = build_reference(args.dataset_root / "all_images")

    manifests = {
        "reference": reference,
        "single": single,
        "sequential": sequential,
    }
    for name, rows in manifests.items():
        write_manifest(args.output_dir / f"{name}_manifest.csv", rows)

    metadata = {
        "single_noises": SINGLE_NOISES,
        "sequential_noises": SEQUENTIAL_NOISES,
        "sequential_steps": SEQUENTIAL_STEPS,
        "single_trajectories_per_image": args.single_trajectories_per_image,
        "sequential_trajectories_per_image": args.sequential_trajectories_per_image,
        "rows": {name: len(rows) for name, rows in manifests.items()},
        "single_images": len({row["image_id"] for row in single}),
        "sequential_images": len({row["image_id"] for row in sequential}),
    }
    (args.output_dir / "manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
