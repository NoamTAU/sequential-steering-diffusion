#!/usr/bin/env python
"""Generate image-diffusion appendix figures from tracked result tables.

The quantitative panels use the same sequential latent CSVs as the main image
figures. The qualitative montage uses a small tracked subset of raw sequential
U-turn frames copied from the Kuma run tree.
"""

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
from PIL import Image


DEFAULT_DATA_ROOT = Path("results/sequential_latents/data")
DEFAULT_RAW_ROOT = Path("results/sequential_latents/raw/sequential_montage")
DEFAULT_OUTPUT_ROOT = Path("results/sequential_latents/figures")
DEFAULT_IMAGE_NAME = "ILSVRC2012_val_00000487"
DEFAULT_TRAJECTORY = "trajectory_000"
DEFAULT_MONTAGE_NOISES = [100, 400, 800]
DEFAULT_MONTAGE_STEPS = [0, 1, 2, 5, 10, 25, 50, 100]


def configure_matplotlib(font_size: int) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size + 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.7,
        }
    )


def load_metadata(data_root: Path) -> dict:
    metadata_path = data_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    with metadata_path.open("r") as handle:
        return json.load(handle)


def layer_names_without_classifier(metadata: dict, df: pd.DataFrame | None = None) -> list[str]:
    names = [name for name in metadata.get("all_layer_names", []) if name != "classifier"]
    if names:
        return names
    if df is None:
        return []
    return (
        df[df["layer"] != "classifier"][["layer", "layer_idx"]]
        .drop_duplicates()
        .sort_values("layer_idx")["layer"]
        .tolist()
    )


def crossing_rho(rhos: pd.Series | np.ndarray, values: pd.Series | np.ndarray) -> float:
    x = np.asarray(rhos, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    for idx in range(len(y) - 1):
        y0, y1 = y[idx], y[idx + 1]
        if y0 == 0:
            return float(x[idx])
        if np.sign(y0) != np.sign(y1):
            x0, x1 = x[idx], x[idx + 1]
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return float("nan")


def save(fig: plt.Figure, path: Path, *, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_sequential_montage(args: argparse.Namespace) -> Path:
    row_labels = {
        100: "very low",
        400: "moderate",
        800: "above transition",
        999: "extreme",
    }
    noises = args.montage_noises
    steps = args.montage_steps

    fig, axes = plt.subplots(
        len(noises),
        len(steps),
        figsize=(1.56 * len(steps), 1.64 * len(noises)),
        squeeze=False,
    )
    for row_idx, noise in enumerate(noises):
        for col_idx, step in enumerate(steps):
            path = (
                args.raw_root
                / args.image_name
                / f"noise_step_{noise}"
                / args.trajectory
                / f"uturn_{step:03d}.jpeg"
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing montage frame: {path}")
            img = Image.open(path).convert("RGB")
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.45)
                spine.set_color("0.82")
            if row_idx == 0:
                ax.set_title(rf"$n={step}$", pad=5)
            if col_idx == 0:
                ax.set_ylabel(
                    rf"$\rho={noise / args.noise_tmax:.1f}$" + "\n" + row_labels.get(noise, ""),
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=34,
                )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.91, bottom=0.025, wspace=0.025, hspace=0.09)
    out = args.output_root / "image_sequential_uturn_montage_matched_data.png"
    save(fig, out, dpi=260)
    return out


def plot_layer_persistence_grid(args: argparse.Namespace, metadata: dict) -> tuple[Path, Path]:
    df = pd.read_csv(args.data_root / "layer_curve_summary.csv")
    layer_names = layer_names_without_classifier(metadata, df)
    df = df[df["layer"].isin(layer_names)].copy()
    positions = {layer: idx for idx, layer in enumerate(layer_names)}
    df["plot_layer_idx"] = df["layer"].map(positions)

    noises = sorted(df["noise_step"].unique())
    cmap = plt.cm.rainbow_r
    norm = plt.Normalize(vmin=0, vmax=max(len(layer_names) - 1, 1))

    ncols = 4
    nrows = int(np.ceil(len(noises) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.05 * ncols, 2.75 * nrows), sharex=True, sharey=True)
    axes_arr = np.asarray(axes).reshape(nrows, ncols)

    for ax in axes_arr.ravel():
        ax.set_visible(False)

    for ax, noise in zip(axes_arr.ravel(), noises):
        ax.set_visible(True)
        sub_noise = df[df["noise_step"] == noise]
        rho = float(sub_noise["rho"].iloc[0])
        n_images = int(sub_noise["n_images"].max())
        panel_curves: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]] = []
        for layer in layer_names:
            sub = sub_noise[sub_noise["layer"] == layer].sort_values("x")
            if sub.empty:
                continue
            x_vals = sub["x"].to_numpy(dtype=float) * args.num_uturns
            y_vals = sub["mean_cosine"].to_numpy(dtype=float)
            color = cmap(norm(positions[layer]))
            panel_curves.append((x_vals, y_vals, color))
            ax.plot(x_vals, y_vals, color=color, linewidth=1.35, alpha=0.96)
        ax.axhline(0.0, color="0.25", linestyle="--", linewidth=0.65)
        ax.set_title(rf"$\rho={rho:.3g}$, $N_\mathrm{{img}}={n_images}$", pad=5)
        ax.set_xlim(0, args.num_uturns)
        ax.set_ylim(-0.055, 1.03)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0.0, 0.5, 1.0])
        if noise >= args.inset_noise_threshold and panel_curves:
            inset = ax.inset_axes([0.42, 0.48, 0.54, 0.45])
            for x_vals, y_vals, color in panel_curves:
                inset.plot(x_vals, y_vals, color=color, linewidth=1.05, alpha=0.96)
            inset.axhline(0.0, color="0.25", linestyle="--", linewidth=0.5)
            inset.set_xlim(args.inset_xmin, args.num_uturns)
            inset.set_ylim(args.inset_ymin, args.inset_ymax)
            inset.set_xticks([50, 100])
            inset.set_yticks([args.inset_ymin, args.inset_ymax])
            inset.tick_params(labelsize=max(args.font_size - 4, 6), pad=1)
            inset.grid(True, alpha=0.22, linewidth=0.5)

    for ax in axes_arr[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(r"U-turn step $n$")
    for ax in axes_arr[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(r"correlation $C_\ell(n)$")

    scalar = ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    hidden_axes = [ax for ax in axes_arr.ravel() if not ax.get_visible()]
    if hidden_axes:
        hidden_axes[0].set_visible(True)
        hidden_axes[0].axis("off")
        cax = hidden_axes[0].inset_axes([0.43, 0.04, 0.13, 0.88])
        cbar = fig.colorbar(scalar, cax=cax)
    else:
        cbar = fig.colorbar(
            scalar,
            ax=[ax for ax in axes_arr.ravel() if ax.get_visible()],
            fraction=0.030,
            pad=0.018,
        )
    tick_candidates = np.array([0, 10, 20, 30, len(layer_names) - 1], dtype=int)
    ticks = sorted({int(t) for t in tick_candidates if 0 <= int(t) < len(layer_names)})
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([rf"$\ell={tick}$" for tick in ticks])
    cbar.set_label("ConvNeXt feature depth")

    pdf = args.output_root / "latent_cosine_noise_grid_appendix_no_classifier.pdf"
    png = args.output_root / "latent_cosine_noise_grid_appendix_no_classifier.png"
    save(fig, pdf)

    # Recreate for PNG because save() closes the figure.
    plot_layer_persistence_grid_png(args, metadata, png)
    return pdf, png


def plot_layer_persistence_grid_png(args: argparse.Namespace, metadata: dict, out: Path) -> None:
    df = pd.read_csv(args.data_root / "layer_curve_summary.csv")
    layer_names = layer_names_without_classifier(metadata, df)
    df = df[df["layer"].isin(layer_names)].copy()
    positions = {layer: idx for idx, layer in enumerate(layer_names)}
    noises = sorted(df["noise_step"].unique())
    cmap = plt.cm.rainbow_r
    norm = plt.Normalize(vmin=0, vmax=max(len(layer_names) - 1, 1))
    ncols = 4
    nrows = int(np.ceil(len(noises) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.05 * ncols, 2.75 * nrows), sharex=True, sharey=True)
    axes_arr = np.asarray(axes).reshape(nrows, ncols)
    for ax in axes_arr.ravel():
        ax.set_visible(False)
    for ax, noise in zip(axes_arr.ravel(), noises):
        ax.set_visible(True)
        sub_noise = df[df["noise_step"] == noise]
        rho = float(sub_noise["rho"].iloc[0])
        n_images = int(sub_noise["n_images"].max())
        panel_curves = []
        for layer in layer_names:
            sub = sub_noise[sub_noise["layer"] == layer].sort_values("x")
            if sub.empty:
                continue
            x_vals = sub["x"].to_numpy(dtype=float) * args.num_uturns
            y_vals = sub["mean_cosine"].to_numpy(dtype=float)
            color = cmap(norm(positions[layer]))
            panel_curves.append((x_vals, y_vals, color))
            ax.plot(x_vals, y_vals, color=color, linewidth=1.35, alpha=0.96)
        ax.axhline(0.0, color="0.25", linestyle="--", linewidth=0.65)
        ax.set_title(rf"$\rho={rho:.3g}$, $N_\mathrm{{img}}={n_images}$", pad=5)
        ax.set_xlim(0, args.num_uturns)
        ax.set_ylim(-0.055, 1.03)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0.0, 0.5, 1.0])
        if noise >= args.inset_noise_threshold and panel_curves:
            inset = ax.inset_axes([0.42, 0.48, 0.54, 0.45])
            for x_vals, y_vals, color in panel_curves:
                inset.plot(x_vals, y_vals, color=color, linewidth=1.05, alpha=0.96)
            inset.axhline(0.0, color="0.25", linestyle="--", linewidth=0.5)
            inset.set_xlim(args.inset_xmin, args.num_uturns)
            inset.set_ylim(args.inset_ymin, args.inset_ymax)
            inset.set_xticks([50, 100])
            inset.set_yticks([args.inset_ymin, args.inset_ymax])
            inset.tick_params(labelsize=max(args.font_size - 4, 6), pad=1)
            inset.grid(True, alpha=0.22, linewidth=0.5)
    for ax in axes_arr[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(r"U-turn step $n$")
    for ax in axes_arr[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(r"correlation $C_\ell(n)$")
    scalar = ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    hidden_axes = [ax for ax in axes_arr.ravel() if not ax.get_visible()]
    if hidden_axes:
        hidden_axes[0].set_visible(True)
        hidden_axes[0].axis("off")
        cax = hidden_axes[0].inset_axes([0.43, 0.04, 0.13, 0.88])
        cbar = fig.colorbar(scalar, cax=cax)
    else:
        cbar = fig.colorbar(
            scalar,
            ax=[ax for ax in axes_arr.ravel() if ax.get_visible()],
            fraction=0.030,
            pad=0.018,
        )
    tick_candidates = np.array([0, 10, 20, 30, len(layer_names) - 1], dtype=int)
    ticks = sorted({int(t) for t in tick_candidates if 0 <= int(t) < len(layer_names)})
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([rf"$\ell={tick}$" for tick in ticks])
    cbar.set_label("ConvNeXt feature depth")
    save(fig, out, dpi=220)


def plot_auc_summary(args: argparse.Namespace, metadata: dict) -> Path:
    layer_auc = pd.read_csv(args.data_root / "image_layer_auc.csv")
    layer_names = layer_names_without_classifier(metadata, layer_auc)
    low_layers = layer_names[: args.low_layer_count]
    high_layers = layer_names[-args.high_layer_count :]
    rows = []
    for (noise, rho, image_name), sub in layer_auc.groupby(["noise_step", "rho", "image_name"]):
        low = sub[sub["layer"].isin(low_layers)]["auc"]
        high = sub[sub["layer"].isin(high_layers)]["auc"]
        if len(low) != len(low_layers) or len(high) != len(high_layers):
            continue
        rows.append(
            {
                "noise_step": noise,
                "rho": rho,
                "image_name": image_name,
                "low_auc": float(low.mean()),
                "high_auc": float(high.mean()),
                "gap": float(high.mean() - low.mean()),
            }
        )
    image_df = pd.DataFrame(rows)
    summary = (
        image_df.groupby(["noise_step", "rho"], as_index=False)
        .agg(
            n_images=("image_name", "size"),
            low_auc_mean=("low_auc", "mean"),
            low_auc_sem=("low_auc", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0),
            high_auc_mean=("high_auc", "mean"),
            high_auc_sem=("high_auc", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0),
            gap_mean=("gap", "mean"),
            gap_sem=("gap", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0),
        )
        .sort_values("noise_step")
    )
    nonzero = summary[summary["noise_step"] > 0]
    transition = crossing_rho(nonzero["rho"], nonzero["gap_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharex=True)
    low_color = "#2f6fbb"
    high_color = "#c83e4d"
    axes[0].plot(summary["rho"], summary["low_auc_mean"], marker="o", color=low_color, linewidth=2.0, label="early features")
    axes[0].plot(summary["rho"], summary["high_auc_mean"], marker="o", color=high_color, linewidth=2.0, label="late features")
    axes[0].fill_between(summary["rho"], summary["low_auc_mean"] - summary["low_auc_sem"], summary["low_auc_mean"] + summary["low_auc_sem"], color=low_color, alpha=0.15)
    axes[0].fill_between(summary["rho"], summary["high_auc_mean"] - summary["high_auc_sem"], summary["high_auc_mean"] + summary["high_auc_sem"], color=high_color, alpha=0.15)
    axes[0].set_ylabel(r"AUC of $C_\ell(n)$")
    axes[0].set_xlabel(r"Noise fraction $\rho$")
    axes[0].set_title("Integrated persistence")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(summary["rho"], summary["gap_mean"], marker="o", color="black", linewidth=2.0)
    axes[1].fill_between(summary["rho"], summary["gap_mean"] - summary["gap_sem"], summary["gap_mean"] + summary["gap_sem"], color="black", alpha=0.14)
    axes[1].axhline(0.0, color="0.45", linestyle="--", linewidth=1.0)
    if np.isfinite(transition):
        axes[1].axvline(transition, color="#6a3d9a", linestyle=":", linewidth=1.6)
        axes[1].text(
            transition + 0.018,
            0.86,
            rf"$\rho_\ast\approx{transition:.2f}$",
            transform=axes[1].get_xaxis_transform(),
            color="#4d2d73",
            ha="left",
            va="top",
        )
    axes[1].set_ylabel("late AUC minus early AUC")
    axes[1].set_xlabel(r"Noise fraction $\rho$")
    axes[1].set_title("Ordering gap")
    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
    fig.tight_layout()
    out = args.output_root / "latent_auc_vs_noise_appendix_no_classifier.pdf"
    save(fig, out)
    return out


def plot_single_uturn_noise_sweep(args: argparse.Namespace, metadata: dict) -> Path:
    df = pd.read_csv(args.data_root / "step1_layer_summary.csv")
    layer_names = layer_names_without_classifier(metadata, df)
    df = df[df["layer"].isin(layer_names)].copy()
    positions = {layer: idx for idx, layer in enumerate(layer_names)}

    gap = pd.read_csv(args.data_root / "variant_step1_gap_summary.csv")
    gap = gap[(gap["variant"] == "without_classifier") & (gap["noise_step"] > 0)]
    transition = crossing_rho(gap["rho"], gap["gap_mean"])

    cmap = plt.cm.rainbow_r
    norm = plt.Normalize(vmin=0, vmax=max(len(layer_names) - 1, 1))
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.55))
    for layer in layer_names:
        sub = df[df["layer"] == layer].sort_values("rho")
        if sub.empty:
            continue
        color = cmap(norm(positions[layer]))
        ax.plot(sub["rho"], sub["mean_cosine"], marker="o", markersize=3.2, linewidth=1.65, color=color)
    if np.isfinite(transition):
        ax.axvline(transition, color="black", linestyle="--", linewidth=1.3)
        ax.text(
            transition + 0.018,
            0.94,
            rf"single-step transition" + "\n" + rf"$\rho_\ast\approx{transition:.2f}$",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=args.font_size - 1,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.035, 1.04)
    ax.set_xlabel(r"Noise fraction $\rho$")
    ax.set_ylabel(r"single-step correlation $C_\ell(1)$")
    ax.set_title("Single U-turn from the sequential trajectories")
    scalar = ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=ax, fraction=0.052, pad=0.025)
    tick_candidates = np.array([0, 10, 20, 30, len(layer_names) - 1], dtype=int)
    ticks = sorted({int(t) for t in tick_candidates if 0 <= int(t) < len(layer_names)})
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([rf"$\ell={tick}$" for tick in ticks])
    cbar.set_label("ConvNeXt feature depth")
    fig.tight_layout()
    out = args.output_root / "latent_single_uturn_noise_sweep_no_classifier.pdf"
    save(fig, out)
    return out


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-name", default=DEFAULT_IMAGE_NAME)
    parser.add_argument("--trajectory", default=DEFAULT_TRAJECTORY)
    parser.add_argument("--montage-noises", nargs="+", type=int, default=DEFAULT_MONTAGE_NOISES)
    parser.add_argument("--montage-steps", nargs="+", type=int, default=DEFAULT_MONTAGE_STEPS)
    parser.add_argument("--noise-tmax", type=int, default=1000)
    parser.add_argument("--num-uturns", type=int, default=100)
    parser.add_argument("--low-layer-count", type=int, default=3)
    parser.add_argument("--high-layer-count", type=int, default=3)
    parser.add_argument("--inset-noise-threshold", type=int, default=600)
    parser.add_argument("--inset-xmin", type=float, default=5.0)
    parser.add_argument("--inset-ymin", type=float, default=0.0)
    parser.add_argument("--inset-ymax", type=float, default=0.10)
    parser.add_argument("--font-size", type=int, default=10)
    return parser


def main() -> None:
    args = create_argparser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(args.font_size)
    metadata = load_metadata(args.data_root)
    plot_sequential_montage(args)
    plot_layer_persistence_grid(args, metadata)
    plot_auc_summary(args, metadata)
    plot_single_uturn_noise_sweep(args, metadata)


if __name__ == "__main__":
    main()
