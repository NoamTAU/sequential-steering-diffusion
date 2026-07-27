#!/usr/bin/env python3
"""Build the exact shared starting-image reference for paired quality analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-manifest", type=Path, required=True)
    parser.add_argument("--sequential-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    single = pd.read_csv(args.single_manifest, keep_default_na=False)
    sequential = pd.read_csv(args.sequential_manifest, keep_default_na=False)
    shared_ids = sorted(
        set(single["image_id"].unique()) & set(sequential["image_id"].unique())
    )
    rows = []
    for image_id in shared_ids:
        candidates = sequential[sequential["image_id"].eq(image_id)]
        rows.append(
            {
                "image_id": image_id,
                "image_path": candidates.iloc[0]["start_path"],
                "start_path": "",
            }
        )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    metadata = {
        "images": len(output),
        "single_images": int(single["image_id"].nunique()),
        "sequential_images": int(sequential["image_id"].nunique()),
        "definition": "intersection of single and sequential image_id values",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(output)} shared starting images to {args.output}")


if __name__ == "__main__":
    main()
