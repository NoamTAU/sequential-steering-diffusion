#!/usr/bin/env python3
"""Inventory image-only U-turn caches needed for the NeurIPS rebuttal."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


NOISE_RE = re.compile(r"noise_step_(\d+)$")
RUN_NOISE_RE = re.compile(r"(?:^|_)noise_(\d+)(?:_|$)")


def summarize_trajectory_root(root: Path, terminal_step: int) -> list[dict]:
    by_noise: dict[int, dict] = defaultdict(
        lambda: {
            "images": set(),
            "trajectories": 0,
            "complete": 0,
            "per_image": Counter(),
        }
    )
    for npz_path in root.glob("*/*/trajectory_*/trajectory_data.npz"):
        trajectory_dir = npz_path.parent
        noise_dir = trajectory_dir.parent
        image_dir = noise_dir.parent
        match = NOISE_RE.fullmatch(noise_dir.name)
        if match is None:
            continue
        noise = int(match.group(1))
        row = by_noise[noise]
        row["images"].add(image_dir.name)
        row["trajectories"] += 1
        row["per_image"][image_dir.name] += 1
        if (trajectory_dir / f"uturn_{terminal_step:03d}.jpeg").exists():
            row["complete"] += 1

    rows = []
    for noise, values in sorted(by_noise.items()):
        per_image = list(values["per_image"].values())
        rows.append(
            {
                "noise_step": noise,
                "rho": noise / 1000,
                "images": len(values["images"]),
                "trajectories": values["trajectories"],
                "complete": values["complete"],
                "trajectories_per_image_min": min(per_image),
                "trajectories_per_image_max": max(per_image),
            }
        )
    return rows


def summarize_steering_root(root: Path) -> dict:
    runs = list(root.glob("*/*/*/steering_data.npz"))
    images = {path.parents[2].name for path in runs}
    noises = Counter()
    for path in runs:
        match = RUN_NOISE_RE.search(path.parent.name)
        if match:
            noises[int(match.group(1))] += 1
    return {
        "root": root.name,
        "images": len(images),
        "runs": len(runs),
        "noise_steps": dict(sorted(noises.items())),
    }


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, dict):
                value = ", ".join(f"{key}:{count}" for key, count in value.items())
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/work/pcsl/Noam/sequential_diffusion"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    results = args.root / "results"
    metadata = args.root / "metadata"
    sequential = summarize_trajectory_root(results / "sequential_uturns", 100)
    single = summarize_trajectory_root(results / "single_uturn_baseline", 1)
    steering = [
        summarize_steering_root(results / "steering_meta_v2_multi"),
        summarize_steering_root(results / "steering_dog2dog_v1_multi"),
    ]
    high_noise_images = [
        line
        for line in (metadata / "high_noise_image_list.txt").read_text().splitlines()
        if line.strip()
    ]
    strict_dogs = [
        line
        for line in (metadata / "dog_image_list_strict_100.txt").read_text().splitlines()
        if line.strip()
    ]

    payload = {
        "root": str(args.root),
        "high_noise_matched_images": len(high_noise_images),
        "strict_dog_images": len(strict_dogs),
        "sequential_uturns": sequential,
        "single_uturn_baseline": single,
        "steering": steering,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "image_cache_inventory.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    lines = [
        "# Image-only rebuttal cache inventory",
        "",
        f"- Matched sequential image set: **{len(high_noise_images)} images**.",
        f"- Strict steering image set: **{len(strict_dogs)} dog images**.",
        "",
        "## Sequential U-turns",
        "",
    ]
    lines += markdown_table(
        sequential,
        [
            ("noise_step", "noise step"),
            ("rho", "rho"),
            ("images", "images"),
            ("trajectories", "trajectories"),
            ("complete", "complete at n=100"),
            ("trajectories_per_image_min", "min/image"),
            ("trajectories_per_image_max", "max/image"),
        ],
    )
    lines += ["", "## Single U-turns", ""]
    lines += markdown_table(
        single,
        [
            ("noise_step", "noise step"),
            ("rho", "rho"),
            ("images", "images"),
            ("trajectories", "pairs"),
            ("complete", "complete pairs"),
            ("trajectories_per_image_min", "min/image"),
            ("trajectories_per_image_max", "max/image"),
        ],
    )
    lines += ["", "## Cached steering", ""]
    lines += markdown_table(
        steering,
        [
            ("root", "experiment"),
            ("images", "images"),
            ("runs", "runs"),
            ("noise_steps", "noise: runs"),
        ],
    )
    (args.output_dir / "image_cache_inventory.md").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
