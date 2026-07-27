#!/usr/bin/env python3
"""Build the focused MH image-steering pilot manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image-path",
        default="/work/pcsl/Noam/diffusion_datasets/all_images/ILSVRC2012_val_00038116.JPEG",
    )
    parser.add_argument("--orig-class-idx", type=int, default=183)
    parser.add_argument("--target-class-idx", type=int, default=189)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument(
        "--target-modes", nargs="+", default=["dog_class", "cat"]
    )
    parser.add_argument(
        "--noise-steps", nargs="+", type=int, default=[100, 200, 400]
    )
    parser.add_argument(
        "--energy-lambdas", nargs="+", type=float, default=[0.0, 1.0, 4.0]
    )
    args = parser.parse_args()

    rows = []
    repeat_indices = range(args.repeat_start, args.repeat_start + args.repeats)
    for target_mode in args.target_modes:
        for noise_step in args.noise_steps:
            for energy_lambda in args.energy_lambdas:
                for repeat_index in repeat_indices:
                    rows.append(
                        {
                            "image_path": args.image_path,
                            "target_mode": target_mode,
                            "orig_class_idx": args.orig_class_idx,
                            "target_class_idx": args.target_class_idx,
                            "noise_step": noise_step,
                            "energy_lambda": energy_lambda,
                            "repeat_index": repeat_index,
                            "seed": 43 + 1000 * repeat_index + noise_step,
                            "num_steps": args.num_steps,
                        }
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
