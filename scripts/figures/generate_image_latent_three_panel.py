#!/usr/bin/env python
"""Generate a text-figure-style three-panel ConvNeXt latent relaxation plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path("results/sequential_latents/data")
DEFAULT_OUTPUT_ROOT = Path("results/sequential_latents/figures")
DEFAULT_NOISE_STEPS = [100, 400, 800]


def configure_matplotlib(font_size: int) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size + 6,
            "axes.labelsize": font_size + 5,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.7,
        }
    )


def load_layer_names(data_root: Path) -> list[str]:
    metadata_path = data_root / "metadata.json"
    if not metadata_path.exists():
        return []
    with metadata_path.open("r") as handle:
        metadata = json.load(handle)
    return list(metadata.get("all_layer_names", []))


def title_for_rho(rho: float) -> str:
    return rf"$\rho \approx {rho:.1f}$"


def make_three_panel(args: argparse.Namespace) -> tuple[Path, Path]:
    configure_matplotlib(args.font_size)
    data_path = args.data_root / "layer_curve_summary.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing source table: {data_path}")

    df = pd.read_csv(data_path)
    if args.exclude_classifier:
        df = df[df["layer"] != "classifier"].copy()

    layer_names = load_layer_names(args.data_root)
    if args.exclude_classifier:
        layer_names = [name for name in layer_names if name != "classifier"]
    if not layer_names:
        layer_names = (
            df[["layer", "layer_idx"]]
            .drop_duplicates()
            .sort_values("layer_idx")["layer"]
            .tolist()
        )
    layer_positions = {layer: idx for idx, layer in enumerate(layer_names)}
    df["plot_layer_idx"] = df["layer"].map(layer_positions)
    df = df[df["plot_layer_idx"].notna()].copy()

    noise_steps = args.noise_steps
    missing = [noise for noise in noise_steps if noise not in set(df["noise_step"].unique())]
    if missing:
        raise ValueError(f"Requested noise steps are missing from {data_path}: {missing}")

    cmap = plt.cm.rainbow_r
    norm = plt.Normalize(vmin=0, vmax=max(len(layer_names) - 1, 1))

    fig, axes = plt.subplots(1, len(noise_steps), figsize=(12.8, 3.25), sharey=True)
    if len(noise_steps) == 1:
        axes = [axes]

    for ax, noise in zip(axes, noise_steps):
        sub_noise = df[df["noise_step"] == noise].copy()
        rho = float(sub_noise["rho"].iloc[0])
        for layer in layer_names:
            sub = sub_noise[sub_noise["layer"] == layer].sort_values("x")
            if sub.empty:
                continue
            step_pos = sub["x"].to_numpy(dtype=float) * args.num_uturns
            integer_mask = np.isclose(step_pos, np.round(step_pos), atol=1e-8)
            sub = sub[integer_mask].copy()
            if sub.empty:
                continue
            layer_idx = layer_positions[layer]
            n_steps = sub["x"].to_numpy(dtype=float) * args.num_uturns
            if args.x_mode == "cumulative_noise":
                x_vals = rho * n_steps
            else:
                x_vals = n_steps
            ax.plot(
                x_vals,
                sub["mean_cosine"],
                color=cmap(norm(layer_idx)),
                marker="o",
                markevery=max(1, len(sub) // 6),
                markersize=3.4,
                linewidth=1.8,
                alpha=0.95,
            )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(title_for_rho(rho), pad=8)
        ax.set_ylim(-0.05, 1.05)
        if args.x_mode == "cumulative_noise":
            ax.set_xlabel(r"Cumulative noise  $\rho \cdot n$")
        else:
            ax.set_xlabel(r"U-turn step  $n$")
    axes[0].set_ylabel(r"correlation $C_\ell(n)$")

    scalar = ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=axes, fraction=0.030, pad=0.025)
    tick_candidates = np.array([0, 10, 20, 30, len(layer_names) - 1], dtype=int)
    ticks = sorted({int(t) for t in tick_candidates if 0 <= int(t) < len(layer_names)})
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([rf"$\ell={tick}$" for tick in ticks])

    suffix = "no_classifier" if args.exclude_classifier else "with_classifier"
    x_suffix = "cumnoise" if args.x_mode == "cumulative_noise" else "uturn_step"
    out_base = args.output_root / f"image_latent_layer_inversion_three_panel_{x_suffix}_{suffix}"
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    args.output_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--noise-steps", nargs="+", type=int, default=DEFAULT_NOISE_STEPS)
    parser.add_argument("--num-uturns", type=int, default=100)
    parser.add_argument(
        "--x-mode",
        choices=("uturn_step", "cumulative_noise"),
        default="uturn_step",
    )
    parser.add_argument("--font-size", type=int, default=13)
    parser.add_argument(
        "--exclude-classifier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude the ConvNeXt classifier/head layer from the colorbar depth ordering.",
    )
    return parser


def main() -> None:
    args = create_argparser().parse_args()
    pdf_path, png_path = make_three_panel(args)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
