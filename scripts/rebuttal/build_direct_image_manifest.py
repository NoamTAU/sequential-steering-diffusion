#!/usr/bin/env python3
"""Build a feature-extraction manifest for direct diffusion samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=200)
    args = parser.parse_args()

    image_paths = sorted(args.image_dir.glob("direct_*.jpeg"))
    if len(image_paths) != args.expected:
        raise RuntimeError(
            f"Expected {args.expected} direct samples, found {len(image_paths)}"
        )
    frame = pd.DataFrame(
        {
            "image_id": [path.stem for path in image_paths],
            "image_path": [str(path) for path in image_paths],
            "start_path": [""] * len(image_paths),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
