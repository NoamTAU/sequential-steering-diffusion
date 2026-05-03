#!/usr/bin/env python
"""Generate sequential U-turn latent figures and compact source tables.

This script replaces the late latent-analysis cells in
``notebooks/plot_generation_sequential.ipynb`` for the theory/ergodicity
figures. It reads the ConvNeXt sequential activation pickles produced on Kuma
and writes tracked PDFs plus CSV/JSON data under ``results/sequential_latents``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_ROOT = Path(
    "/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/scripts/sequential_analysis_results"
)
DEFAULT_IMAGE_LIST = Path(
    "/work/pcsl/Noam/sequential_diffusion/metadata/high_noise_image_list.txt"
)
DEFAULT_NOISE_STEPS = [0, 100, 200, 400, 600, 800, 999]


def layer_sort_key(name: str) -> tuple[int, int | str]:
    if name == "classifier" or name.endswith("head"):
        return (2, 999)
    match = re.search(r"features\.(\d+)", name)
    if match:
        return (1, int(match.group(1)))
    return (0, name)


def is_classifier_layer(name: str) -> bool:
    return name == "classifier" or name.endswith("head")


def sem_or_nan(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def read_image_names(path: Path) -> list[str]:
    image_names: list[str] = []
    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                image_names.append(Path(line).stem)
    return image_names


def load_image_layer_curves(
    results_file: Path,
    *,
    min_traj_samples: int = 1,
    include_t0: bool = True,
) -> tuple[dict[str, dict[str, list[float] | list[int]]], list[str], int]:
    with results_file.open("rb") as handle:
        results_by_step = pickle.load(handle)

    steps = sorted(results_by_step.keys())
    if not steps:
        return {}, [], 0

    max_step = max(steps)
    layer_names = set()
    for step in steps:
        for layer, metrics in results_by_step[step].items():
            if isinstance(metrics, dict) and "cosine" in metrics:
                layer_names.add(layer)
    sorted_layers = sorted(layer_names, key=layer_sort_key)

    data: dict[str, dict[str, list[float] | list[int]]] = {}
    for layer in sorted_layers:
        xs: list[float] = []
        ys: list[float] = []
        sems: list[float] = []
        ns: list[int] = []
        for step in steps:
            metrics = results_by_step[step].get(layer)
            if not metrics or "cosine" not in metrics:
                continue
            vals = np.asarray(metrics["cosine"], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) < min_traj_samples:
                continue
            xs.append(step / float(max_step))
            ys.append(float(np.mean(vals)))
            sems.append(sem_or_nan(vals))
            ns.append(int(len(vals)))
        if include_t0:
            xs = [0.0] + xs
            ys = [1.0] + ys
            sems = [0.0] + sems
            ns = [ns[0] if ns else 0] + ns
        data[layer] = {"x": xs, "y": ys, "sem": sems, "n": ns}
    return data, sorted_layers, max_step


def build_identity_curves(layer_names: list[str]) -> dict[str, dict[str, list[float] | list[int]]]:
    return {
        layer: {"x": [0.0, 1.0], "y": [1.0, 1.0], "sem": [0.0, 0.0], "n": [0, 0]}
        for layer in layer_names
    }


def interpolate_curve(x_vals: list[float], y_vals: list[float], x_grid: np.ndarray) -> np.ndarray:
    xs = np.asarray(x_vals, dtype=float)
    ys = np.asarray(y_vals, dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    xs_unique, unique_idx = np.unique(xs, return_index=True)
    ys_unique = ys[unique_idx]
    return np.interp(x_grid, xs_unique, ys_unique)


def first_crossing_time(x_grid: np.ndarray, curve: np.ndarray, threshold: float = 0.5) -> float:
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(curve, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    if y[0] <= threshold:
        return float(x[0])
    for idx in range(len(y) - 1):
        y0, y1 = y[idx], y[idx + 1]
        if y0 >= threshold and y1 <= threshold:
            x0, x1 = x[idx], x[idx + 1]
            if y1 == y0:
                return float(x0)
            return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))
    return float("nan")


def crossing_rho(rhos: pd.Series | np.ndarray, gaps: pd.Series | np.ndarray) -> float:
    x = np.asarray(rhos, dtype=float)
    y = np.asarray(gaps, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if np.any(y == 0):
        return float(x[np.where(y == 0)[0][0]])
    signs = np.sign(y)
    for idx in range(len(y) - 1):
        if signs[idx] == 0:
            return float(x[idx])
        if signs[idx] != signs[idx + 1]:
            x0, x1 = x[idx], x[idx + 1]
            y0, y1 = y[idx], y[idx + 1]
            if y1 == y0:
                return float(x0)
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return float("nan")


def select_group_layers(
    layer_names: list[str],
    *,
    low_count: int,
    high_count: int,
    exclude_classifier: bool,
) -> tuple[list[str], list[str], list[str]]:
    filtered = [
        layer for layer in layer_names if not (exclude_classifier and is_classifier_layer(layer))
    ]
    if len(filtered) < low_count + high_count:
        raise ValueError(f"Not enough layers after filtering: {len(filtered)}")
    return filtered, filtered[:low_count], filtered[-high_count:]


def configure_matplotlib(font_size: int) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
        }
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def load_dataset(args: argparse.Namespace) -> tuple[
    pd.DataFrame,
    dict[int, dict[str, dict[str, dict[str, list[float] | list[int]]]]],
    list[str],
]:
    image_names = read_image_names(args.image_list)
    status_rows = []
    available_files: dict[tuple[str, int], Path] = {}

    for noise in args.noise_steps:
        for image_name in image_names:
            if noise == 0 and args.allow_analytic_zero_noise:
                status_rows.append(
                    {
                        "noise_step": noise,
                        "rho": noise / float(args.noise_tmax),
                        "image_name": image_name,
                        "exists": True,
                        "source": "analytic",
                    }
                )
                continue
            result_file = (
                args.analysis_root
                / args.classifier_name
                / image_name
                / f"noise_{noise}"
                / "sequential_activations_v2.pk"
            )
            exists = result_file.exists()
            status_rows.append(
                {
                    "noise_step": noise,
                    "rho": noise / float(args.noise_tmax),
                    "image_name": image_name,
                    "exists": exists,
                    "source": "file",
                }
            )
            if exists:
                available_files[(image_name, noise)] = result_file

    if not available_files:
        raise FileNotFoundError("No nonzero sequential activation files were found.")

    status_df = pd.DataFrame(status_rows)
    all_layer_names: set[str] = set()
    per_noise_per_image: dict[int, dict[str, dict[str, dict[str, list[float] | list[int]]]]] = {}

    for (image_name, noise), path in sorted(available_files.items()):
        curves, layer_names, _ = load_image_layer_curves(
            path,
            min_traj_samples=args.min_traj_samples,
            include_t0=True,
        )
        if curves:
            per_noise_per_image.setdefault(noise, {})[image_name] = curves
            all_layer_names.update(layer_names)

    sorted_layers = sorted(all_layer_names, key=layer_sort_key)
    if not sorted_layers:
        raise ValueError("Could not infer ConvNeXt layers from activation files.")

    if 0 in args.noise_steps and args.allow_analytic_zero_noise:
        zero_images = status_df[
            (status_df["noise_step"] == 0) & (status_df["exists"])
        ]["image_name"].tolist()
        per_noise_per_image[0] = {
            image_name: build_identity_curves(sorted_layers) for image_name in zero_images
        }

    return status_df, per_noise_per_image, sorted_layers


def build_tables(
    args: argparse.Namespace,
    status_df: pd.DataFrame,
    per_noise_per_image: dict[int, dict[str, dict[str, dict[str, list[float] | list[int]]]]],
    all_layer_names: list[str],
    x_grid: np.ndarray,
) -> dict[str, pd.DataFrame]:
    layer_curve_rows = []
    image_layer_auc_rows = []
    step1_rows = []
    group_curve_rows = []
    image_group_curve_rows = []
    image_group_auc_rows = []

    low_layers = all_layer_names[: args.low_layer_count]
    high_layers = all_layer_names[-args.high_layer_count :]

    for noise in sorted(per_noise_per_image):
        image_layer_curves: dict[str, list[np.ndarray]] = {layer: [] for layer in all_layer_names}
        group_curves_by_image = {"low": [], "high": []}

        for image_name, curves in sorted(per_noise_per_image[noise].items()):
            layer_auc_values = []
            layer_interps: dict[str, np.ndarray] = {}
            for layer_idx, layer in enumerate(all_layer_names):
                layer_data = curves.get(layer)
                if not layer_data or len(layer_data["x"]) < 2:
                    continue
                interp = interpolate_curve(layer_data["x"], layer_data["y"], x_grid)
                layer_interps[layer] = interp
                image_layer_curves[layer].append(interp)
                auc = float(np.trapz(interp, x_grid))
                layer_auc_values.append((layer, layer_idx, auc))
                if len(layer_data["y"]) > args.single_uturn_step:
                    step1_rows.append(
                        {
                            "noise_step": noise,
                            "rho": noise / float(args.noise_tmax),
                            "image_name": image_name,
                            "layer": layer,
                            "layer_idx": layer_idx,
                            "cosine_step1": float(layer_data["y"][args.single_uturn_step]),
                        }
                    )

            for layer, layer_idx, auc in layer_auc_values:
                image_layer_auc_rows.append(
                    {
                        "noise_step": noise,
                        "rho": noise / float(args.noise_tmax),
                        "image_name": image_name,
                        "layer": layer,
                        "layer_idx": layer_idx,
                        "auc": auc,
                    }
                )

            for group, selected_layers in (("low", low_layers), ("high", high_layers)):
                samples = [layer_interps[layer] for layer in selected_layers if layer in layer_interps]
                if len(samples) != len(selected_layers):
                    continue
                curve = np.mean(np.vstack(samples), axis=0)
                group_curves_by_image[group].append(curve)
                auc = float(np.trapz(curve, x_grid))
                image_group_auc_rows.append(
                    {
                        "noise_step": noise,
                        "rho": noise / float(args.noise_tmax),
                        "image_name": image_name,
                        "group": group,
                        "auc": auc,
                        "half_life": first_crossing_time(
                            x_grid, curve, threshold=args.half_life_threshold
                        ),
                    }
                )
                for x, y in zip(x_grid, curve):
                    image_group_curve_rows.append(
                        {
                            "noise_step": noise,
                            "rho": noise / float(args.noise_tmax),
                            "image_name": image_name,
                            "group": group,
                            "x": float(x),
                            "cosine": float(y),
                        }
                    )

        for layer_idx, layer in enumerate(all_layer_names):
            samples = image_layer_curves[layer]
            if not samples:
                continue
            arr = np.vstack(samples)
            mean = np.mean(arr, axis=0)
            sem = (
                np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
                if arr.shape[0] > 1
                else np.zeros(arr.shape[1])
            )
            for x, y, y_sem in zip(x_grid, mean, sem):
                layer_curve_rows.append(
                    {
                        "noise_step": noise,
                        "rho": noise / float(args.noise_tmax),
                        "layer": layer,
                        "layer_idx": layer_idx,
                        "x": float(x),
                        "mean_cosine": float(y),
                        "sem_cosine": float(y_sem),
                        "n_images": int(arr.shape[0]),
                    }
                )

        for group, samples in group_curves_by_image.items():
            if not samples:
                continue
            arr = np.vstack(samples)
            mean = np.mean(arr, axis=0)
            sd = (
                np.std(arr, axis=0, ddof=1)
                if arr.shape[0] > 1
                else np.zeros(arr.shape[1])
            )
            sem = sd / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
            for x, y, y_sd, y_sem in zip(x_grid, mean, sd, sem):
                group_curve_rows.append(
                    {
                        "noise_step": noise,
                        "rho": noise / float(args.noise_tmax),
                        "group": group,
                        "x": float(x),
                        "mean_cosine": float(y),
                        "sd_cosine": float(y_sd),
                        "sem_cosine": float(y_sem),
                        "n_images": int(arr.shape[0]),
                    }
                )

    status_summary_df = (
        status_df.groupby(["noise_step", "rho"], as_index=False)
        .agg(n_images=("exists", "sum"), requested_images=("exists", "size"))
        .sort_values("noise_step")
    )
    image_layer_auc_df = pd.DataFrame(image_layer_auc_rows)
    step1_df = pd.DataFrame(step1_rows)
    image_group_auc_df = pd.DataFrame(image_group_auc_rows)

    regime_rows = []
    for (noise, rho, image_name), sub in image_layer_auc_df.groupby(
        ["noise_step", "rho", "image_name"]
    ):
        sub = sub.sort_values("layer_idx")
        if len(sub) < args.low_layer_count + args.high_layer_count:
            continue
        low_auc = float(sub.head(args.low_layer_count)["auc"].mean())
        high_auc = float(sub.tail(args.high_layer_count)["auc"].mean())
        regime_rows.append(
            {
                "noise_step": noise,
                "rho": rho,
                "image_name": image_name,
                "low_auc_mean": low_auc,
                "high_auc_mean": high_auc,
                "gap_high_minus_low": high_auc - low_auc,
            }
        )
    image_regime_auc_df = pd.DataFrame(regime_rows)
    regime_auc_summary_df = (
        image_regime_auc_df.groupby(["noise_step", "rho"], as_index=False)
        .agg(
            n_images=("image_name", "size"),
            low_auc_mean=("low_auc_mean", "mean"),
            low_auc_sem=("low_auc_mean", sem_or_nan),
            high_auc_mean=("high_auc_mean", "mean"),
            high_auc_sem=("high_auc_mean", sem_or_nan),
            gap_high_minus_low_mean=("gap_high_minus_low", "mean"),
            gap_high_minus_low_sem=("gap_high_minus_low", sem_or_nan),
        )
        .sort_values("noise_step")
    )

    step1_layer_summary_df = (
        step1_df.groupby(["noise_step", "rho", "layer", "layer_idx"], as_index=False)
        .agg(
            mean_cosine=("cosine_step1", "mean"),
            sem_cosine=("cosine_step1", sem_or_nan),
            n_images=("cosine_step1", "size"),
        )
        .sort_values(["noise_step", "layer_idx"])
    )

    variant_tables = build_variant_tables(args, per_noise_per_image, all_layer_names, x_grid, step1_df)

    return {
        "status": status_df,
        "status_summary": status_summary_df,
        "layer_curve_summary": pd.DataFrame(layer_curve_rows),
        "image_layer_auc": image_layer_auc_df,
        "image_regime_auc": image_regime_auc_df,
        "regime_auc_summary": regime_auc_summary_df,
        "step1_layer_values": step1_df,
        "step1_layer_summary": step1_layer_summary_df,
        "group_curve_summary": pd.DataFrame(group_curve_rows),
        "image_group_curves": pd.DataFrame(image_group_curve_rows),
        "image_group_auc": image_group_auc_df,
        **variant_tables,
    }


def build_variant_tables(
    args: argparse.Namespace,
    per_noise_per_image: dict[int, dict[str, dict[str, dict[str, list[float] | list[int]]]]],
    all_layer_names: list[str],
    x_grid: np.ndarray,
    step1_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    variant_auc_rows = []
    variant_gap_rows = []
    variant_half_life_rows = []
    variant_curve_rows = []

    variants = [("with_classifier", False), ("without_classifier", True)]
    for variant, exclude_classifier in variants:
        _, low_layers, high_layers = select_group_layers(
            all_layer_names,
            low_count=args.low_layer_count,
            high_count=args.high_layer_count,
            exclude_classifier=exclude_classifier,
        )
        for noise in sorted(per_noise_per_image):
            low_curves = []
            high_curves = []
            image_rows = []
            for image_name, curves in sorted(per_noise_per_image[noise].items()):
                low_samples = []
                high_samples = []
                for layer in low_layers:
                    layer_data = curves.get(layer)
                    if not layer_data or len(layer_data["x"]) < 2:
                        continue
                    low_samples.append(interpolate_curve(layer_data["x"], layer_data["y"], x_grid))
                for layer in high_layers:
                    layer_data = curves.get(layer)
                    if not layer_data or len(layer_data["x"]) < 2:
                        continue
                    high_samples.append(interpolate_curve(layer_data["x"], layer_data["y"], x_grid))
                if len(low_samples) != len(low_layers) or len(high_samples) != len(high_layers):
                    continue
                low_curve = np.mean(np.vstack(low_samples), axis=0)
                high_curve = np.mean(np.vstack(high_samples), axis=0)
                low_curves.append(low_curve)
                high_curves.append(high_curve)
                image_rows.append(
                    {
                        "variant": variant,
                        "exclude_classifier": exclude_classifier,
                        "noise_step": noise,
                        "rho": noise / float(args.noise_tmax),
                        "image_name": image_name,
                        "low_auc": float(np.trapz(low_curve, x_grid)),
                        "high_auc": float(np.trapz(high_curve, x_grid)),
                        "auc_gap": float(np.trapz(high_curve - low_curve, x_grid)),
                        "low_step1": float(low_curve[1]),
                        "high_step1": float(high_curve[1]),
                        "step1_gap": float(high_curve[1] - low_curve[1]),
                        "low_half_life": first_crossing_time(
                            x_grid, low_curve, threshold=args.half_life_threshold
                        ),
                        "high_half_life": first_crossing_time(
                            x_grid, high_curve, threshold=args.half_life_threshold
                        ),
                    }
                )
            if not low_curves or not high_curves:
                continue
            low_arr = np.vstack(low_curves)
            high_arr = np.vstack(high_curves)
            low_mean = low_arr.mean(axis=0)
            high_mean = high_arr.mean(axis=0)
            image_df = pd.DataFrame(image_rows)
            variant_auc_rows.append(
                {
                    "variant": variant,
                    "exclude_classifier": exclude_classifier,
                    "noise_step": noise,
                    "rho": noise / float(args.noise_tmax),
                    "n_images": low_arr.shape[0],
                    "low_auc_curvefirst": float(np.trapz(low_mean, x_grid)),
                    "high_auc_curvefirst": float(np.trapz(high_mean, x_grid)),
                    "auc_gap_curvefirst": float(np.trapz(high_mean - low_mean, x_grid)),
                    "low_auc_imagefirst_mean": float(image_df["low_auc"].mean()),
                    "high_auc_imagefirst_mean": float(image_df["high_auc"].mean()),
                    "auc_gap_imagefirst_mean": float(image_df["auc_gap"].mean()),
                    "auc_gap_imagefirst_sem": sem_or_nan(image_df["auc_gap"]),
                }
            )
            variant_gap_rows.append(
                {
                    "variant": variant,
                    "exclude_classifier": exclude_classifier,
                    "noise_step": noise,
                    "rho": noise / float(args.noise_tmax),
                    "n_images": len(image_df),
                    "gap_mean": float(image_df["step1_gap"].mean()),
                    "gap_sd": float(image_df["step1_gap"].std(ddof=1))
                    if len(image_df) > 1
                    else 0.0,
                    "gap_sem": sem_or_nan(image_df["step1_gap"]),
                }
            )
            finite_half = image_df[
                np.isfinite(image_df["low_half_life"]) & np.isfinite(image_df["high_half_life"])
            ].copy()
            if not finite_half.empty:
                finite_half["half_life_gap"] = (
                    finite_half["high_half_life"] - finite_half["low_half_life"]
                )
                variant_half_life_rows.append(
                    {
                        "variant": variant,
                        "exclude_classifier": exclude_classifier,
                        "noise_step": noise,
                        "rho": noise / float(args.noise_tmax),
                        "n_images": len(finite_half),
                        "half_life_gap_mean": float(finite_half["half_life_gap"].mean()),
                        "half_life_gap_sd": float(finite_half["half_life_gap"].std(ddof=1))
                        if len(finite_half) > 1
                        else 0.0,
                        "half_life_gap_sem": sem_or_nan(finite_half["half_life_gap"]),
                    }
                )
            for x, low_y, high_y in zip(x_grid, low_mean, high_mean):
                variant_curve_rows.append(
                    {
                        "variant": variant,
                        "exclude_classifier": exclude_classifier,
                        "noise_step": noise,
                        "rho": noise / float(args.noise_tmax),
                        "x": float(x),
                        "low_mean": float(low_y),
                        "high_mean": float(high_y),
                        "gap_mean": float(high_y - low_y),
                    }
                )

    variant_step1_rows = []
    for variant, exclude_classifier in variants:
        filtered_layers, _, _ = select_group_layers(
            all_layer_names,
            low_count=args.low_layer_count,
            high_count=args.high_layer_count,
            exclude_classifier=exclude_classifier,
        )
        sub = step1_df[step1_df["layer"].isin(filtered_layers)].copy()
        layer_positions = {layer: idx for idx, layer in enumerate(filtered_layers)}
        sub["layer_pos"] = sub["layer"].map(layer_positions)
        grouped = (
            sub.groupby(["noise_step", "rho", "layer", "layer_pos"], as_index=False)
            .agg(mean_cosine=("cosine_step1", "mean"), n_images=("cosine_step1", "size"))
            .sort_values(["noise_step", "layer_pos"])
        )
        grouped["variant"] = variant
        grouped["exclude_classifier"] = exclude_classifier
        variant_step1_rows.append(grouped)

    variant_step1_df = (
        pd.concat(variant_step1_rows, ignore_index=True)
        if variant_step1_rows
        else pd.DataFrame()
    )

    return {
        "variant_auc_summary": pd.DataFrame(variant_auc_rows).sort_values(
            ["variant", "noise_step"]
        ),
        "variant_step1_gap_summary": pd.DataFrame(variant_gap_rows).sort_values(
            ["variant", "noise_step"]
        ),
        "variant_half_life_summary": pd.DataFrame(variant_half_life_rows).sort_values(
            ["variant", "noise_step"]
        ),
        "variant_curve_summary": pd.DataFrame(variant_curve_rows),
        "variant_step1_layer_summary": variant_step1_df,
    }


def plot_multi_noise_grid(tables: dict[str, pd.DataFrame], all_layer_names: list[str], out: Path) -> None:
    df = tables["layer_curve_summary"]
    noises = sorted(df["noise_step"].unique())
    colors = plt.cm.rainbow(np.linspace(1, 0, len(all_layer_names)))
    color_map = {layer: colors[idx] for idx, layer in enumerate(all_layer_names)}

    fig, axes = plt.subplots(1, len(noises), figsize=(3.1 * len(noises), 3.35), sharey=True)
    if len(noises) == 1:
        axes = [axes]
    for ax, noise in zip(axes, noises):
        sub_noise = df[df["noise_step"] == noise]
        n_images = int(sub_noise["n_images"].max())
        rho = float(sub_noise["rho"].iloc[0])
        for layer in all_layer_names:
            sub = sub_noise[sub_noise["layer"] == layer].sort_values("x")
            if sub.empty:
                continue
            ax.plot(sub["x"], sub["mean_cosine"], color=color_map[layer], linewidth=1.6)
        ax.set_title(rf"$\rho={rho:.2f}$ | n={n_images}")
        ax.set_xlabel(r"Normalized U-turns ($t/t_{\max}$)")
        ax.set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Cosine similarity")
    handles = [plt.Line2D([0], [0], color=color_map[layer], lw=2) for layer in all_layer_names]
    labels = [rf"$\ell={idx}$" for idx, _ in enumerate(all_layer_names)]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(6, max(2, len(labels))),
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, out)


def plot_regime_auc(tables: dict[str, pd.DataFrame], args: argparse.Namespace, out: Path) -> None:
    df = tables["regime_auc_summary"].sort_values("noise_step")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=True)
    axes[0].plot(
        df["rho"],
        df["low_auc_mean"],
        marker="o",
        color="#2f6fbb",
        linewidth=2,
        label=f"Low-level ({args.low_layer_count})",
    )
    axes[0].plot(
        df["rho"],
        df["high_auc_mean"],
        marker="o",
        color="#c83e4d",
        linewidth=2,
        label=f"High-level ({args.high_layer_count})",
    )
    axes[0].fill_between(
        df["rho"],
        df["low_auc_mean"] - df["low_auc_sem"].fillna(0),
        df["low_auc_mean"] + df["low_auc_sem"].fillna(0),
        color="#2f6fbb",
        alpha=0.15,
    )
    axes[0].fill_between(
        df["rho"],
        df["high_auc_mean"] - df["high_auc_sem"].fillna(0),
        df["high_auc_mean"] + df["high_auc_sem"].fillna(0),
        color="#c83e4d",
        alpha=0.15,
    )
    axes[0].set_xlabel(r"Noise fraction $\rho$")
    axes[0].set_ylabel("AUC of cosine survival")
    axes[0].set_title("A. Image-averaged layer-group survival")
    axes[0].legend(frameon=False, loc="best")

    axes[1].plot(
        df["rho"],
        df["gap_high_minus_low_mean"],
        marker="o",
        color="black",
        linewidth=2,
    )
    axes[1].fill_between(
        df["rho"],
        df["gap_high_minus_low_mean"] - df["gap_high_minus_low_sem"].fillna(0),
        df["gap_high_minus_low_mean"] + df["gap_high_minus_low_sem"].fillna(0),
        color="black",
        alpha=0.15,
    )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel(r"Noise fraction $\rho$")
    axes[1].set_ylabel("High-level AUC minus low-level AUC")
    axes[1].set_title("B. Image-averaged regime gap")
    fig.tight_layout()
    save_figure(fig, out)


def plot_single_uturn(tables: dict[str, pd.DataFrame], all_layer_names: list[str], args: argparse.Namespace, out: Path) -> None:
    df = tables["step1_layer_summary"]
    noises = sorted(df["noise_step"].unique())
    colors = plt.cm.rainbow(np.linspace(1, 0, len(all_layer_names)))
    color_map = {layer: colors[idx] for idx, layer in enumerate(all_layer_names)}
    palette = plt.cm.viridis(np.linspace(0.05, 0.95, len(noises)))

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.2), gridspec_kw={"width_ratios": [1.2, 1.2]})
    for color, noise in zip(palette, noises):
        sub = df[df["noise_step"] == noise].sort_values("layer_idx")
        axes[0].plot(
            sub["layer_idx"],
            sub["mean_cosine"],
            marker="o",
            linewidth=2,
            color=color,
            label=rf"$\rho={noise / float(args.noise_tmax):.2f}$",
        )
        axes[0].fill_between(
            sub["layer_idx"],
            sub["mean_cosine"] - sub["sem_cosine"].fillna(0),
            sub["mean_cosine"] + sub["sem_cosine"].fillna(0),
            color=color,
            alpha=0.12,
        )
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel(f"Cosine similarity after U-turn {args.single_uturn_step}")
    axes[0].set_title("A. Single-U-turn latent profile")
    axes[0].legend(frameon=False, loc="best")

    for layer in all_layer_names:
        sub = df[df["layer"] == layer].sort_values("rho")
        axes[1].plot(
            sub["rho"],
            sub["mean_cosine"],
            marker="o",
            linewidth=2,
            color=color_map[layer],
        )
        axes[1].fill_between(
            sub["rho"],
            sub["mean_cosine"] - sub["sem_cosine"].fillna(0),
            sub["mean_cosine"] + sub["sem_cosine"].fillna(0),
            color=color_map[layer],
            alpha=0.10,
        )
    axes[1].set_xlabel(r"Noise fraction $\rho$")
    axes[1].set_ylabel(f"Cosine similarity after U-turn {args.single_uturn_step}")
    axes[1].set_title("B. Single-U-turn all-latent noise sweep")
    handles = [plt.Line2D([0], [0], color=color_map[layer], lw=2) for layer in all_layer_names]
    labels = [rf"$\ell={idx}$" for idx, _ in enumerate(all_layer_names)]
    axes[1].legend(handles, labels, frameon=False, loc="best", ncol=2)
    fig.tight_layout()
    save_figure(fig, out)


def plot_step1_variants(tables: dict[str, pd.DataFrame], all_layer_names: list[str], args: argparse.Namespace, out: Path) -> None:
    variants = [("with_classifier", False), ("without_classifier", True)]
    fig, axes = plt.subplots(1, len(variants), figsize=(6.2 * len(variants), 3.8), sharey=True)
    if len(variants) == 1:
        axes = [axes]
    summary = tables["variant_step1_layer_summary"]

    for ax, (variant, exclude_classifier) in zip(axes, variants):
        filtered_layers, _, _ = select_group_layers(
            all_layer_names,
            low_count=args.low_layer_count,
            high_count=args.high_layer_count,
            exclude_classifier=exclude_classifier,
        )
        positions = {layer: idx for idx, layer in enumerate(filtered_layers)}
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=0, vmax=max(len(filtered_layers) - 1, 1))
        sub_variant = summary[summary["variant"] == variant]
        for layer in filtered_layers:
            sub = sub_variant[sub_variant["layer"] == layer].sort_values("rho")
            if sub.empty:
                continue
            ax.plot(
                sub["rho"],
                sub["mean_cosine"],
                marker="o",
                linewidth=1.8,
                color=cmap(norm(positions[layer])),
            )
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Layer depth")
        cbar.set_ticks([0, max(len(filtered_layers) - 1, 1)])
        cbar.set_ticklabels(["low", "high"])
        title = "including classifier" if not exclude_classifier else "excluding classifier"
        ax.set_title(title)
        ax.set_xlabel(r"Noise fraction $\rho$")
    axes[0].set_ylabel("Cosine after one U-turn")
    fig.suptitle("Sequential step-1 all-layer noise sweep", y=1.02)
    fig.tight_layout()
    save_figure(fig, out)


def plot_relaxation_variants(tables: dict[str, pd.DataFrame], args: argparse.Namespace, out: Path) -> None:
    auc_df = tables["variant_auc_summary"]
    half_df = tables["variant_half_life_summary"]
    variants = [("with_classifier", "-", "#7a1f5c"), ("without_classifier", "--", "#1f6f78")]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), sharex=True)
    for variant, linestyle, color in variants:
        auc_sub = auc_df[auc_df["variant"] == variant].sort_values("noise_step")
        axes[0].plot(
            auc_sub["rho"],
            auc_sub["auc_gap_curvefirst"],
            marker="o",
            linewidth=2,
            linestyle=linestyle,
            color=color,
            label=variant.replace("_", " "),
        )
        axes[1].plot(
            auc_sub["rho"],
            auc_sub["auc_gap_imagefirst_mean"],
            marker="o",
            linewidth=2,
            linestyle=linestyle,
            color=color,
            label=variant.replace("_", " "),
        )
        axes[1].fill_between(
            auc_sub["rho"],
            auc_sub["auc_gap_imagefirst_mean"] - auc_sub["auc_gap_imagefirst_sem"].fillna(0),
            auc_sub["auc_gap_imagefirst_mean"] + auc_sub["auc_gap_imagefirst_sem"].fillna(0),
            color=color,
            alpha=0.12,
        )
        half_sub = half_df[half_df["variant"] == variant].sort_values("noise_step")
        axes[2].plot(
            half_sub["rho"],
            half_sub["half_life_gap_mean"],
            marker="o",
            linewidth=2,
            linestyle=linestyle,
            color=color,
            label=variant.replace("_", " "),
        )
        axes[2].fill_between(
            half_sub["rho"],
            half_sub["half_life_gap_mean"] - half_sub["half_life_gap_sem"].fillna(0),
            half_sub["half_life_gap_mean"] + half_sub["half_life_gap_sem"].fillna(0),
            color=color,
            alpha=0.12,
        )

    titles = [
        "A. Integrate image-averaged curves",
        "B. Average per-image AUCs",
        "C. Half-life ordering",
    ]
    ylabels = [
        "High AUC minus low AUC",
        "Mean per-image AUC gap",
        rf"High $\tau_{{{args.half_life_threshold:.1f}}}$ minus low $\tau_{{{args.half_life_threshold:.1f}}}$",
    ]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(r"Noise fraction $\rho$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, out)


def write_metadata(args: argparse.Namespace, tables: dict[str, pd.DataFrame], all_layer_names: list[str], path: Path) -> None:
    regime = tables["regime_auc_summary"]
    variant_auc = tables["variant_auc_summary"]
    metadata = {
        "analysis_root": str(args.analysis_root),
        "classifier_name": args.classifier_name,
        "image_list": str(args.image_list),
        "noise_steps": args.noise_steps,
        "noise_tmax": args.noise_tmax,
        "low_layer_count": args.low_layer_count,
        "high_layer_count": args.high_layer_count,
        "all_layer_names": all_layer_names,
        "observed_auc_gap_crossing_rho": crossing_rho(
            regime[regime["noise_step"] > 0]["rho"],
            regime[regime["noise_step"] > 0]["gap_high_minus_low_mean"],
        ),
        "variant_auc_gap_crossing_rho": {
            variant: crossing_rho(
                sub[sub["noise_step"] > 0]["rho"],
                sub[sub["noise_step"] > 0]["auc_gap_imagefirst_mean"],
            )
            for variant, sub in variant_auc.groupby("variant")
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {path}")


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--classifier-name", default="convnext_base")
    parser.add_argument("--image-list", type=Path, default=DEFAULT_IMAGE_LIST)
    parser.add_argument("--noise-steps", nargs="+", type=int, default=DEFAULT_NOISE_STEPS)
    parser.add_argument("--noise-tmax", type=int, default=1000)
    parser.add_argument("--output-root", type=Path, default=Path("results/sequential_latents"))
    parser.add_argument("--low-layer-count", type=int, default=3)
    parser.add_argument("--high-layer-count", type=int, default=3)
    parser.add_argument("--auc-grid-size", type=int, default=201)
    parser.add_argument("--single-uturn-step", type=int, default=1)
    parser.add_argument("--half-life-threshold", type=float, default=0.5)
    parser.add_argument("--min-traj-samples", type=int, default=1)
    parser.add_argument("--font-size", type=int, default=10)
    parser.add_argument("--allow-analytic-zero-noise", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = create_argparser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_root / "figures"
    data_dir = args.output_root / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib(args.font_size)
    x_grid = np.linspace(0.0, 1.0, args.auc_grid_size)

    print("Loading sequential latent dataset...")
    status_df, per_noise_per_image, all_layer_names = load_dataset(args)
    print(f"Loaded noises: {sorted(per_noise_per_image)}")
    print(f"Loaded layers: {len(all_layer_names)}")

    tables = build_tables(args, status_df, per_noise_per_image, all_layer_names, x_grid)

    for name, df in tables.items():
        write_csv(df, data_dir / f"{name}.csv")
    write_metadata(args, tables, all_layer_names, data_dir / "metadata.json")

    tag = args.image_list.stem
    plot_multi_noise_grid(
        tables,
        all_layer_names,
        figures_dir / f"latent_cosine_noise_grid_avg_{tag}.pdf",
    )
    plot_regime_auc(
        tables,
        args,
        figures_dir / f"latent_regime_auc_vs_noise_avg_{tag}.pdf",
    )
    plot_single_uturn(
        tables,
        all_layer_names,
        args,
        figures_dir / f"latent_single_uturn_vs_noise_{tag}.pdf",
    )
    plot_step1_variants(
        tables,
        all_layer_names,
        args,
        figures_dir / f"latent_step1_all_layers_variants_{tag}.pdf",
    )
    plot_relaxation_variants(
        tables,
        args,
        figures_dir / f"latent_relaxation_ordering_variants_{tag}.pdf",
    )


if __name__ == "__main__":
    main()
