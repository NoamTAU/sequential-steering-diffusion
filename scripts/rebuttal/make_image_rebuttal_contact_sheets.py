#!/usr/bin/env python3
"""Build contact sheets from the exact images used in rebuttal analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


TILE = 178
LABEL_HEIGHT = 30
GAP = 8
BACKGROUND = "#F7F7F7"
TEXT = "#222222"


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def tile(path: str | Path, label: str) -> Image.Image:
    canvas = Image.new("RGB", (TILE, TILE + LABEL_HEIGHT), BACKGROUND)
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image.thumbnail((TILE, TILE), Image.Resampling.LANCZOS)
        x = (TILE - image.width) // 2
        y = (TILE - image.height) // 2
        canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (TILE // 2, TILE + LABEL_HEIGHT // 2),
        label,
        anchor="mm",
        fill=TEXT,
        font=font(13),
    )
    return canvas


def sheet(
    rows: list[list[tuple[str | Path, str]]],
    row_labels: list[str],
    title: str,
) -> Image.Image:
    row_label_width = 150
    title_height = 42
    columns = max(len(row) for row in rows)
    width = row_label_width + columns * (TILE + GAP) - GAP
    height = title_height + len(rows) * (TILE + LABEL_HEIGHT + GAP) - GAP
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 20), title, anchor="mm", fill=TEXT, font=font(19))
    for row_index, (items, row_label) in enumerate(zip(rows, row_labels)):
        y = title_height + row_index * (TILE + LABEL_HEIGHT + GAP)
        draw.text(
            (row_label_width - 10, y + TILE // 2),
            row_label,
            anchor="rm",
            fill=TEXT,
            font=font(14),
        )
        for column_index, (path, label) in enumerate(items):
            x = row_label_width + column_index * (TILE + GAP)
            canvas.paste(tile(path, label), (x, y))
    return canvas


def single_sheet(frame: pd.DataFrame, output_dir: Path) -> None:
    selected_rhos = (0.025, 0.1, 0.4, 0.8, 0.999)
    image_ids = sorted(frame["image_id"].unique())[:3]
    rows = []
    for image_id in image_ids:
        image = frame[frame["image_id"].eq(image_id)]
        start_path = image.iloc[0]["start_path"]
        items: list[tuple[str, str]] = [(start_path, "start")]
        for rho in selected_rhos:
            row = image[
                np.isclose(image["rho"], rho) & image["trajectory_id"].eq(0)
            ].iloc[0]
            items.append((row["image_path"], rf"rho={rho:g}"))
        rows.append(items)
    contact = sheet(rows, image_ids, "Single U-turn examples used in quality analysis")
    contact.save(output_dir / "single_uturn_source_image_audit.png")


def sequential_sheets(frame: pd.DataFrame, output_dir: Path) -> None:
    selected_rhos = (0.1, 0.4, 0.8, 0.999)
    selected_steps = (1, 10, 50, 100)
    image_ids = sorted(frame["image_id"].unique())[:2]
    for image_index, image_id in enumerate(image_ids, start=1):
        image = frame[
            frame["image_id"].eq(image_id) & frame["trajectory_id"].eq(0)
        ]
        rows = []
        for rho in selected_rhos:
            rho_frame = image[np.isclose(image["rho"], rho)]
            items: list[tuple[str, str]] = [
                (rho_frame.iloc[0]["start_path"], "start")
            ]
            for step in selected_steps:
                row = rho_frame[rho_frame["step"].eq(step)].iloc[0]
                items.append((row["image_path"], f"step {step}"))
            rows.append(items)
        contact = sheet(
            rows,
            [f"rho={rho:g}" for rho in selected_rhos],
            f"Sequential U-turn example {image_index}: {image_id}",
        )
        contact.save(
            output_dir / f"sequential_uturn_source_image_audit_{image_index}.png"
        )


def direct_sheet(frame: pd.DataFrame, output_dir: Path) -> None:
    rng = np.random.default_rng(44)
    selected = frame.iloc[rng.choice(len(frame), size=20, replace=False)]
    rows = []
    labels = []
    for row_index in range(4):
        subset = selected.iloc[row_index * 5 : (row_index + 1) * 5]
        rows.append(
            [
                (row["image_path"], str(row["image_id"]).replace("direct_", ""))
                for _, row in subset.iterrows()
            ]
        )
        labels.append(f"sample set {row_index + 1}")
    contact = sheet(rows, labels, "Independent direct-diffusion baseline samples")
    contact.save(output_dir / "direct_diffusion_source_image_audit.png")


def representative_runs(run_dirs: list[Path], count: int = 5) -> list[tuple[Path, float]]:
    endpoints = []
    for run_dir in run_dirs:
        with np.load(run_dir / "mh_data.npz") as data:
            endpoints.append((run_dir, float(data["target_probability"][-1])))
    endpoints.sort(key=lambda item: item[1])
    indices = np.linspace(0, len(endpoints) - 1, count).round().astype(int)
    return [endpoints[index] for index in indices]


def mh_sheet(root: Path, output_dir: Path) -> None:
    rows = []
    row_labels = []
    for energy_lambda in (0, 4):
        run_dirs = sorted(
            (root / "mh_steering").glob(
                f"*/mode_dog_class_rho_400_lambda_{energy_lambda}_rep_*"
            )
        )
        selected = representative_runs(run_dirs)
        items: list[tuple[str | Path, str]] = [
            (run_dirs[0] / "step_000.jpeg", "start")
        ]
        for run_dir, probability in selected:
            items.append(
                (run_dir / "step_050.jpeg", f"target p={probability:.3f}")
            )
        rows.append(items)
        row_labels.append("H=0" if energy_lambda == 0 else "lambda=4")
    contact = sheet(
        rows,
        row_labels,
        "Classifier-energy MH-rule steering at rho=0.4",
    )
    contact.save(output_dir / "mh_steering_source_image_audit.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.root / "image_quality/manifests"
    single = pd.read_csv(manifest_dir / "single_manifest.csv")
    sequential = pd.read_csv(manifest_dir / "sequential_manifest.csv")
    direct = pd.read_csv(
        args.root / "direct_diffusion/direct_manifest.csv", keep_default_na=False
    )
    single_sheet(single, args.output_dir)
    sequential_sheets(sequential, args.output_dir)
    direct_sheet(direct, args.output_dir)
    mh_sheet(args.root, args.output_dir)
    print(f"wrote contact sheets to {args.output_dir}")


if __name__ == "__main__":
    main()
