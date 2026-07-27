#!/usr/bin/env python3
"""Audit full no-reference IQA scores and render source-image contact sheets."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def contact_sheet(
    frame: pd.DataFrame,
    output: Path,
    *,
    columns: int,
    title: str,
    tile_size: int = 224,
) -> None:
    label_height = 82
    title_height = 54
    gap = 12
    rows = int(np.ceil(len(frame) / columns))
    width = columns * tile_size + (columns - 1) * gap
    height = title_height + rows * (tile_size + label_height + gap) - gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (width // 2, 26),
        title,
        anchor="mm",
        fill="#202020",
        font=font(20),
    )
    for index, row in enumerate(frame.itertuples(index=False)):
        grid_row, grid_column = divmod(index, columns)
        x = grid_column * (tile_size + gap)
        y = title_height + grid_row * (tile_size + label_height + gap)
        with Image.open(row.image_path) as handle:
            image = handle.convert("RGB")
            image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_size, tile_size), "#F1F1F1")
        tile.paste(
            image,
            ((tile_size - image.width) // 2, (tile_size - image.height) // 2),
        )
        canvas.paste(tile, (x, y))
        label = (
            f"{row.audit_group} | {row.audit_rank}\n"
            f"MUSIQ {row.musiq:.1f}; TOPIQ-NR {row.topiq_nr:.2f}\n"
            f"{row.image_id}"
        )
        draw.multiline_text(
            (x + tile_size // 2, y + tile_size + label_height // 2),
            "\n".join(textwrap.fill(line, 31) for line in label.splitlines()),
            anchor="mm",
            align="center",
            spacing=3,
            fill="#202020",
            font=font(12),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def add_consensus(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    z_columns = []
    for column in ("musiq", "topiq_nr"):
        values = frame[column].to_numpy(dtype=float)
        z_column = f"{column}_z"
        frame[z_column] = (values - values.mean()) / values.std(ddof=0)
        z_columns.append(z_column)
    frame["learned_iqa_consensus"] = frame[z_columns].mean(axis=1)
    frame["model_disagreement"] = frame["musiq_z"] - frame["topiq_nr_z"]
    return frame


def quantile_examples(group: pd.DataFrame, audit_group: str) -> pd.DataFrame:
    ordered = group.sort_values("learned_iqa_consensus").reset_index(drop=True)
    indices = [
        int(round((len(ordered) - 1) * quantile))
        for quantile in (0.1, 0.5, 0.9)
    ]
    selected = ordered.iloc[indices].copy()
    selected["audit_group"] = audit_group
    selected["audit_rank"] = ("low score", "median score", "high score")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--pilot-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full = read_csv(args.scores)
    pilot = read_csv(args.pilot_scores)
    required = {
        "global_id",
        "dataset",
        "image_id",
        "image_path",
        "noise_step",
        "rho",
        "step",
        "musiq",
        "topiq_nr",
    }
    missing = sorted(required - set(full.columns))
    if missing:
        raise ValueError(f"Full score table is missing columns: {missing}")
    if full["global_id"].duplicated().any():
        raise RuntimeError("Duplicate global_id values")
    if full["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image paths")
    if not np.isfinite(full[["musiq", "topiq_nr"]].to_numpy(float)).all():
        raise RuntimeError("Non-finite IQA values")

    overlap = pilot[pilot["image_path"].isin(set(full["image_path"]))].merge(
        full[["image_path", "musiq", "topiq_nr"]],
        on="image_path",
        suffixes=("_pilot", "_full"),
        validate="1:1",
    )
    overlap_differences = {}
    for model in ("musiq", "topiq_nr"):
        overlap_differences[model] = float(
            np.abs(overlap[f"{model}_pilot"] - overlap[f"{model}_full"]).max()
        )

    for column in ("noise_step", "rho", "step"):
        full[column] = pd.to_numeric(full[column], errors="coerce")
    full = add_consensus(full)
    representative_groups = (
        ("Original starts", full["dataset"].eq("original_start")),
        ("Direct diffusion", full["dataset"].eq("direct_diffusion")),
        (
            "Single rho=0.1",
            full["dataset"].eq("single") & full["rho"].eq(0.1),
        ),
        (
            "Single rho=0.999",
            full["dataset"].eq("single") & full["rho"].eq(0.999),
        ),
        (
            "Sequential rho=0.2, step=100",
            full["dataset"].eq("sequential")
            & full["rho"].eq(0.2)
            & full["step"].eq(100),
        ),
        (
            "Sequential rho=0.6, step=100",
            full["dataset"].eq("sequential")
            & full["rho"].eq(0.6)
            & full["step"].eq(100),
        ),
    )
    representatives = pd.concat(
        [
            quantile_examples(full[mask], label)
            for label, mask in representative_groups
        ],
        ignore_index=True,
    )
    representatives.to_csv(
        args.output_dir / "iqa_representative_images.csv", index=False
    )
    contact_sheet(
        representatives,
        args.output_dir / "iqa_representative_contact_sheet.png",
        columns=3,
        title="IQA audit: low, median, and high score within each cohort",
    )

    uturns = full[full["dataset"].isin({"single", "sequential"})]
    musiq_preferred = uturns.nlargest(4, "model_disagreement").copy()
    musiq_preferred["audit_group"] = "MUSIQ scores higher"
    musiq_preferred["audit_rank"] = [
        f"disagreement rank {index}" for index in range(1, 5)
    ]
    topiq_preferred = uturns.nsmallest(4, "model_disagreement").copy()
    topiq_preferred["audit_group"] = "TOPIQ-NR scores higher"
    topiq_preferred["audit_rank"] = [
        f"disagreement rank {index}" for index in range(1, 5)
    ]
    disagreements = pd.concat(
        [musiq_preferred, topiq_preferred], ignore_index=True
    )
    disagreements.to_csv(
        args.output_dir / "iqa_metric_disagreements.csv", index=False
    )
    contact_sheet(
        disagreements,
        args.output_dir / "iqa_metric_disagreement_contact_sheet.png",
        columns=4,
        title="IQA audit: largest MUSIQ versus TOPIQ-NR disagreements",
    )

    correlation = float(
        full[["musiq", "topiq_nr"]].corr(method="spearman").iloc[0, 1]
    )
    lines = [
        "# No-reference IQA integrity audit",
        "",
        "- Overall status: **PASS**",
        f"- Full rows: `{len(full)}`",
        f"- Unique global IDs: `{full['global_id'].nunique()}`",
        f"- Unique image paths: `{full['image_path'].nunique()}`",
        "- All MUSIQ and TOPIQ-NR values are finite.",
        f"- Pilot-overlap images: `{len(overlap)}`",
        (
            "- Maximum pilot/full score difference: "
            f"MUSIQ `{overlap_differences['musiq']:.3g}`, "
            f"TOPIQ-NR `{overlap_differences['topiq_nr']:.3g}`."
        ),
        f"- Full-table MUSIQ/TOPIQ-NR Spearman correlation: `{correlation:.3f}`.",
        "",
        "The representative sheet samples the 10th, 50th, and 90th percentile "
        "of the two-model consensus within each named cohort. The disagreement "
        "sheet surfaces the largest model-specific ranking differences for "
        "manual inspection.",
    ]
    (args.output_dir / "iqa_integrity_audit.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(f"Wrote IQA audit to {args.output_dir}")


if __name__ == "__main__":
    main()
