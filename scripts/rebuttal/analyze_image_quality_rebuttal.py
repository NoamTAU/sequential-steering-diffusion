#!/usr/bin/env python3
"""Aggregate image-quality metrics and generate rebuttal figures/tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


PAPER_CMAP = LinearSegmentedColormap.from_list(
    "paper_noise", ["#D96B6B", "#E8A27E", "#91AFC9", "#3B6FB6"]
)
SEED = 42


def load_feature_frame(manifest: Path, feature_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifest)
    frame["pixel_rmse"] = np.load(feature_dir / "pixel_rmse.npy", mmap_mode="r")
    frame["pixel_l1"] = np.load(feature_dir / "pixel_l1.npy", mmap_mode="r")
    frame["convnext_max_probability"] = np.load(
        feature_dir / "convnext_max_probability.npy", mmap_mode="r"
    )
    frame["convnext_entropy"] = np.load(
        feature_dir / "convnext_entropy.npy", mmap_mode="r"
    )
    frame["convnext_class_perplexity"] = np.exp(
        np.asarray(frame["convnext_entropy"], dtype=np.float64)
    )
    optional_arrays = {
        "convnext_feature_cosine_distance": (
            "convnext_feature_cosine_distance.npy"
        ),
        "convnext_start_class_probability": (
            "convnext_start_class_probability.npy"
        ),
        "convnext_start_class_retained": "convnext_start_class_retained.npy",
        "manifold_precision_k3": "manifold_precision_k3.npy",
        "manifold_density_k3": "manifold_density_k3.npy",
        "manifold_nearest_distance": "manifold_nearest_distance.npy",
        "manifold_realism_score_k3": "manifold_realism_score_k3.npy",
    }
    for column, filename in optional_arrays.items():
        path = feature_dir / filename
        frame[column] = (
            np.load(path, mmap_mode="r") if path.exists() else np.nan
        )
    return frame


def add_clip_path_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    net = np.full(len(frame), np.nan, dtype=np.float32)
    cumulative = np.full(len(frame), np.nan, dtype=np.float32)
    for npz_path, indices in frame.groupby("npz_path", sort=False).groups.items():
        data = np.load(npz_path, allow_pickle=True)
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        vectors = embeddings.reshape(len(embeddings), -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        net_distances = np.maximum(0, 1 - vectors @ vectors[0])
        increments = np.maximum(0, 1 - np.sum(vectors[1:] * vectors[:-1], axis=1))
        cumulative_distances = np.concatenate(
            [np.zeros(1, dtype=np.float32), np.cumsum(increments, dtype=np.float32)]
        )
        row_indices = np.asarray(list(indices), dtype=int)
        steps = frame.loc[row_indices, "step"].to_numpy(dtype=int)
        valid = steps < len(net_distances)
        net[row_indices[valid]] = net_distances[steps[valid]]
        cumulative[row_indices[valid]] = cumulative_distances[steps[valid]]
    frame["clip_net_distance"] = net
    frame["clip_cumulative_path"] = cumulative
    return frame


def apply_clip_metric_overrides(
    frame: pd.DataFrame, override_path: Path
) -> pd.DataFrame:
    if not override_path.exists():
        return frame
    overrides = pd.read_csv(override_path).set_index("record_id")
    if not overrides.index.is_unique:
        raise RuntimeError(f"{override_path}: record_id values are not unique")
    frame = frame.copy()
    selected = frame["record_id"].isin(overrides.index)
    for column in ("clip_net_distance", "clip_cumulative_path"):
        frame.loc[selected, column] = (
            frame.loc[selected, "record_id"].map(overrides[column]).to_numpy()
        )
    return frame


def image_mean_ci(
    frame: pd.DataFrame, column: str, draws: int = 500, seed: int = 44
) -> tuple[float, float, float]:
    values = frame.groupby("image_id")[column].mean().dropna().to_numpy()
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )


def mmd_polynomial(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dimension = x.shape[1]
    k_xx = (x @ x.T / dimension + 1).pow(3)
    k_yy = (y @ y.T / dimension + 1).pow(3)
    k_xy = (x @ y.T / dimension + 1).pow(3)
    size = x.shape[0]
    xx = (k_xx.sum() - k_xx.diagonal().sum()) / (size * (size - 1))
    yy = (k_yy.sum() - k_yy.diagonal().sum()) / (size * (size - 1))
    return xx + yy - 2 * k_xy.mean()


def kid_statistics(
    features: np.ndarray,
    reference: torch.Tensor,
    device: torch.device,
    seed: int,
    subsets: int = 100,
    subset_size: int = 100,
) -> tuple[float, float, float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    values = []
    features_tensor = torch.as_tensor(
        np.asarray(features, dtype=np.float32), device=device
    )
    size = min(subset_size, len(features_tensor), len(reference))
    for _ in range(subsets):
        x_indices = torch.randperm(
            len(features_tensor), generator=generator, device=device
        )[:size]
        y_indices = torch.randperm(
            len(reference), generator=generator, device=device
        )[:size]
        values.append(mmd_polynomial(features_tensor[x_indices], reference[y_indices]))
    values = torch.stack(values).cpu().numpy()
    return (
        float(values.mean()),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def metric_row(
    group: pd.DataFrame,
    features: np.ndarray,
    reference: torch.Tensor,
    device: torch.device,
    seed: int,
) -> dict:
    row = {"samples": len(group), "images": group["image_id"].nunique()}
    for column in (
        "pixel_rmse",
        "pixel_l1",
        "clip_net_distance",
        "clip_cumulative_path",
        "convnext_max_probability",
        "convnext_entropy",
        "convnext_class_perplexity",
        "convnext_feature_cosine_distance",
        "convnext_start_class_probability",
        "convnext_start_class_retained",
        "manifold_precision_k3",
        "manifold_density_k3",
        "manifold_nearest_distance",
        "manifold_realism_score_k3",
    ):
        mean, low, high = image_mean_ci(group, column, seed=seed + 2)
        row[column] = mean
        row[f"{column}_ci_low"] = low
        row[f"{column}_ci_high"] = high
    kid, kid_low, kid_high = kid_statistics(
        features[group.index.to_numpy()],
        reference,
        device,
        seed=seed,
    )
    row.update({"kid": kid, "kid_ci_low": kid_low, "kid_ci_high": kid_high})
    return row


def summarize_single(
    frame: pd.DataFrame,
    features: np.ndarray,
    reference: torch.Tensor,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    binned_rows = []
    for noise, group in frame.groupby("noise_step", sort=True):
        row = {
            "noise_step": int(noise),
            "rho": float(group["rho"].iloc[0]),
            **metric_row(group, features, reference, device, SEED + int(noise)),
        }
        rows.append(row)

        quantiles = pd.qcut(group["pixel_rmse"], q=4, labels=False, duplicates="drop")
        for bin_index in sorted(quantiles.unique()):
            bin_group = group.loc[quantiles == bin_index]
            bin_row = {
                "noise_step": int(noise),
                "rho": float(group["rho"].iloc[0]),
                "change_bin": int(bin_index),
                **metric_row(
                    bin_group,
                    features,
                    reference,
                    device,
                    SEED + int(noise) * 10 + int(bin_index),
                ),
            }
            binned_rows.append(bin_row)
    return pd.DataFrame(rows), pd.DataFrame(binned_rows)


def summarize_sequential(
    frame: pd.DataFrame,
    features: np.ndarray,
    reference: torch.Tensor,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for (noise, step), group in frame.groupby(["noise_step", "step"], sort=True):
        rows.append(
            {
                "noise_step": int(noise),
                "rho": float(group["rho"].iloc[0]),
                "step": int(step),
                **metric_row(
                    group,
                    features,
                    reference,
                    device,
                    SEED + int(noise) * 100 + int(step),
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_direct(
    frame: pd.DataFrame,
    features: np.ndarray,
    reference: torch.Tensor,
    device: torch.device,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "noise_step": 0,
                "rho": np.nan,
                **metric_row(frame, features, reference, device, SEED + 9999),
            }
        ]
    )


def reference_statistics(reference: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    reference = reference.double()
    mean = reference.mean(dim=0)
    covariance = torch.zeros(
        (reference.shape[1], reference.shape[1]),
        dtype=torch.float64,
        device=reference.device,
    )
    chunk_size = 5000
    for start in range(0, len(reference), chunk_size):
        centered = reference[start : start + chunk_size] - mean
        covariance.add_(centered.T @ centered)
    covariance.div_(len(reference) - 1)
    return mean.cpu().double().numpy(), covariance.cpu().double().numpy()


def balanced_sample_indices(group: pd.DataFrame, total: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image_ids = sorted(group["image_id"].unique())
    base, remainder = divmod(total, len(image_ids))
    selected = []
    for index, image_id in enumerate(image_ids):
        candidates = group.index[group["image_id"] == image_id].to_numpy()
        count = base + (index < remainder)
        selected.extend(rng.choice(candidates, size=count, replace=False))
    return np.asarray(selected, dtype=int)


def empirical_fid_terms(
    sample: np.ndarray,
    reference_mean: torch.Tensor,
    reference_covariance: torch.Tensor,
) -> dict[str, float]:
    sample_tensor = torch.as_tensor(
        sample, dtype=torch.float64, device=reference_mean.device
    )
    sample_mean = sample_tensor.mean(dim=0)
    centered = sample_tensor - sample_mean

    # The 2048-D empirical covariance is singular for N=200. Its nonzero
    # product eigenvalues equal those of this N x N Gram matrix, yielding
    # the standard empirical FID without an unstable 2048-D matrix square root.
    product_gram = (
        centered
        @ reference_covariance
        @ centered.T
        / (len(sample_tensor) - 1)
    )
    product_gram = (product_gram + product_gram.T) / 2
    product_eigenvalues = torch.linalg.eigvalsh(product_gram).clamp_min(0)
    sample_trace = centered.square().sum() / (len(sample_tensor) - 1)
    reference_trace = torch.trace(reference_covariance)
    mean_term = (sample_mean - reference_mean).square().sum()
    covariance_term = (
        sample_trace
        + reference_trace
        - 2 * product_eigenvalues.sqrt().sum()
    )

    sample_gram = centered @ centered.T / (len(sample_tensor) - 1)
    sample_gram = (sample_gram + sample_gram.T) / 2
    sample_eigenvalues = torch.linalg.eigvalsh(sample_gram).clamp_min(0)
    effective_rank = sample_eigenvalues.sum().square() / (
        sample_eigenvalues.square().sum().clamp_min(1e-20)
    )
    return {
        "fid": float((mean_term + covariance_term).cpu()),
        "mean_term": float(mean_term.cpu()),
        "covariance_term": float(covariance_term.cpu()),
        "sample_trace": float(sample_trace.cpu()),
        "effective_rank": float(effective_rank.cpu()),
    }


def add_matched_fid(
    summary: pd.DataFrame,
    frame: pd.DataFrame,
    features: np.ndarray,
    reference_mean: np.ndarray,
    reference_covariance: np.ndarray,
    dataset: str,
) -> pd.DataFrame:
    summary = summary.copy()
    summary["fid_matched_200"] = np.nan
    summary["fid_mean_term"] = np.nan
    summary["fid_covariance_term"] = np.nan
    summary["fid_sample_trace"] = np.nan
    summary["fid_effective_rank"] = np.nan
    if dataset == "single":
        targets = [(row.noise_step, None) for row in summary.itertuples()]
    else:
        targets = [
            (row.noise_step, row.step)
            for row in summary[["noise_step", "step"]]
            .drop_duplicates()
            .itertuples(index=False)
        ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_mean_tensor = torch.as_tensor(
        reference_mean, dtype=torch.float64, device=device
    )
    reference_covariance_tensor = torch.as_tensor(
        reference_covariance, dtype=torch.float64, device=device
    )
    for noise, step in targets:
        mask = frame["noise_step"].eq(noise)
        summary_mask = summary["noise_step"].eq(noise)
        if step is not None:
            mask &= frame["step"].eq(step)
            summary_mask &= summary["step"].eq(step)
        group = frame.loc[mask]
        indices = balanced_sample_indices(
            group, total=200, seed=SEED + int(noise) + int(step or 0)
        )
        sample = np.asarray(features[indices], dtype=np.float64)
        terms = empirical_fid_terms(
            sample, reference_mean_tensor, reference_covariance_tensor
        )
        summary.loc[summary_mask, "fid_matched_200"] = terms["fid"]
        summary.loc[summary_mask, "fid_mean_term"] = terms["mean_term"]
        summary.loc[summary_mask, "fid_covariance_term"] = terms[
            "covariance_term"
        ]
        summary.loc[summary_mask, "fid_sample_trace"] = terms["sample_trace"]
        summary.loc[summary_mask, "fid_effective_rank"] = terms[
            "effective_rank"
        ]
    return summary


def add_source_reference_metrics(
    summary: pd.DataFrame,
    frame: pd.DataFrame,
    features: np.ndarray,
    source_features: np.ndarray,
    dataset: str,
) -> pd.DataFrame:
    summary = summary.copy()
    columns = (
        "source_kid",
        "source_kid_ci_low",
        "source_kid_ci_high",
        "source_fid_200",
        "source_fid_mean_term",
        "source_fid_covariance_term",
        "source_fid_effective_rank",
    )
    for column in columns:
        summary[column] = np.nan

    repeats, remainder = divmod(200, len(source_features))
    if remainder:
        raise RuntimeError(
            f"Cannot balance 200 reference rows over {len(source_features)} starts"
        )
    reference_array = np.repeat(
        np.asarray(source_features, dtype=np.float32), repeats, axis=0
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = torch.as_tensor(reference_array, device=device)
    reference_mean, reference_covariance = reference_statistics(reference)
    reference_mean_tensor = torch.as_tensor(
        reference_mean, dtype=torch.float64, device=device
    )
    reference_covariance_tensor = torch.as_tensor(
        reference_covariance, dtype=torch.float64, device=device
    )

    if dataset == "single":
        targets = [(row.noise_step, None) for row in summary.itertuples()]
    else:
        targets = [
            (row.noise_step, row.step)
            for row in summary[["noise_step", "step"]]
            .drop_duplicates()
            .itertuples(index=False)
        ]
    for noise, step in targets:
        mask = frame["noise_step"].eq(noise)
        summary_mask = summary["noise_step"].eq(noise)
        if step is not None:
            mask &= frame["step"].eq(step)
            summary_mask &= summary["step"].eq(step)
        group = frame.loc[mask]
        seed = SEED + int(noise) + int(step or 0)
        indices = balanced_sample_indices(group, total=200, seed=seed)
        sample = np.asarray(features[indices], dtype=np.float64)
        kid, kid_low, kid_high = kid_statistics(
            sample, reference, device, seed=seed
        )
        terms = empirical_fid_terms(
            sample, reference_mean_tensor, reference_covariance_tensor
        )
        summary.loc[summary_mask, "source_kid"] = kid
        summary.loc[summary_mask, "source_kid_ci_low"] = kid_low
        summary.loc[summary_mask, "source_kid_ci_high"] = kid_high
        summary.loc[summary_mask, "source_fid_200"] = terms["fid"]
        summary.loc[summary_mask, "source_fid_mean_term"] = terms["mean_term"]
        summary.loc[summary_mask, "source_fid_covariance_term"] = terms[
            "covariance_term"
        ]
        summary.loc[summary_mask, "source_fid_effective_rank"] = terms[
            "effective_rank"
        ]
    return summary


def style_axes() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.framealpha": 0.82,
        }
    )


def add_noise_colorbar(fig, axes, minimum: float, maximum: float) -> None:
    scalar = plt.cm.ScalarMappable(norm=Normalize(minimum, maximum), cmap=PAPER_CMAP)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label(r"noise fraction $\rho$")


def save_figure(fig, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_single(
    single: pd.DataFrame, direct: pd.DataFrame | None, output_dir: Path
) -> None:
    style_axes()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.15), layout="constrained")
    norm = Normalize(single["rho"].min(), single["rho"].max())
    single = single.sort_values("rho")
    axes[0].plot(
        single["pixel_rmse"],
        single["kid"] * 1000,
        color="#9A9A9A",
        lw=1,
        zorder=1,
    )
    axes[1].plot(
        single["pixel_rmse"],
        single["fid_matched_200"],
        color="#9A9A9A",
        lw=1,
        zorder=1,
    )
    axes[2].plot(
        single["pixel_rmse"],
        single["convnext_max_probability"],
        color="#9A9A9A",
        lw=1,
        zorder=1,
    )
    for row in single.itertuples():
        color = PAPER_CMAP(norm(row.rho))
        axes[0].errorbar(
            row.pixel_rmse,
            row.kid * 1000,
            xerr=[
                [row.pixel_rmse - row.pixel_rmse_ci_low],
                [row.pixel_rmse_ci_high - row.pixel_rmse],
            ],
            yerr=[
                [(row.kid - row.kid_ci_low) * 1000],
                [(row.kid_ci_high - row.kid) * 1000],
            ],
            fmt="o",
            ms=5,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=1.5,
            zorder=2,
        )
        axes[1].scatter(
            row.pixel_rmse,
            row.fid_matched_200,
            s=25,
            color=color,
            zorder=2,
        )
        axes[2].errorbar(
            row.pixel_rmse,
            row.convnext_max_probability,
            xerr=[
                [row.pixel_rmse - row.pixel_rmse_ci_low],
                [row.pixel_rmse_ci_high - row.pixel_rmse],
            ],
            yerr=[
                [
                    row.convnext_max_probability
                    - row.convnext_max_probability_ci_low
                ],
                [
                    row.convnext_max_probability_ci_high
                    - row.convnext_max_probability
                ],
            ],
            fmt="o",
            ms=5,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=1.5,
            zorder=2,
        )
    baseline_kid = (
        direct["kid"].iloc[0]
        if direct is not None
        else single.loc[single["noise_step"] == 999, "kid"].iloc[0]
    )
    baseline_fid = (
        direct["fid_matched_200"].iloc[0]
        if direct is not None
        else single.loc[single["noise_step"] == 999, "fid_matched_200"].iloc[0]
    )
    baseline_label = "direct diffusion" if direct is not None else r"$\rho=0.999$"
    axes[0].axhline(
        baseline_kid * 1000,
        color="black",
        ls="--",
        lw=1,
        label=baseline_label,
    )
    axes[0].legend(fontsize=8)
    axes[0].set(
        xlabel="realized pixel RMS change",
        ylabel=r"KID to ImageNet ($\times 10^3$; $\downarrow$)",
        title="Global kernel distance",
    )
    axes[1].set(
        xlabel="realized pixel RMS change",
        ylabel=r"FID-200 to ImageNet ($\downarrow$)",
        title="Global Fréchet distance",
    )
    axes[1].axhline(
        baseline_fid,
        color="black",
        ls="--",
        lw=1,
    )
    axes[2].set(
        xlabel="realized pixel RMS change",
        ylabel=r"ConvNeXt max probability ($\uparrow$)",
        title="Per-image semantic confidence",
    )
    add_noise_colorbar(fig, axes, single["rho"].min(), single["rho"].max())
    fig.suptitle(
        "Single U-turn diagnostics versus realized image change", fontsize=11
    )
    save_figure(fig, output_dir, "single_uturn_quality_vs_change")


def plot_sequential(
    sequential: pd.DataFrame, direct_summary: pd.DataFrame | None, output_dir: Path
) -> None:
    style_axes()
    norm = Normalize(sequential["rho"].min(), sequential["rho"].max())
    direct = (
        direct_summary["kid"].iloc[0] * 1000
        if direct_summary is not None
        else sequential.loc[sequential["noise_step"] == 999, "kid"].median() * 1000
    )
    direct_fid = (
        direct_summary["fid_matched_200"].iloc[0]
        if direct_summary is not None
        else sequential.loc[
            sequential["noise_step"] == 999, "fid_matched_200"
        ].median()
    )
    baseline_label = (
        "direct diffusion" if direct_summary is not None else r"$\rho=0.999$"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.15), layout="constrained")
    for rho, group in sequential.groupby("rho"):
        group = group.sort_values("step")
        color = PAPER_CMAP(norm(rho))
        axes[0].plot(
            group["clip_cumulative_path"],
            group["kid"] * 1000,
            marker="o",
            ms=3,
            lw=1.6,
            color=color,
        )
        axes[0].fill_between(
            group["clip_cumulative_path"],
            group["kid_ci_low"] * 1000,
            group["kid_ci_high"] * 1000,
            color=color,
            alpha=0.11,
            linewidth=0,
        )
        axes[1].plot(
            group["clip_cumulative_path"],
            group["fid_matched_200"],
            marker="o",
            ms=3,
            lw=1.6,
            color=color,
        )
        axes[2].plot(
            group["clip_cumulative_path"],
            group["convnext_max_probability"],
            marker="o",
            ms=3,
            lw=1.6,
            color=color,
        )
        axes[2].fill_between(
            group["clip_cumulative_path"],
            group["convnext_max_probability_ci_low"],
            group["convnext_max_probability_ci_high"],
            color=color,
            alpha=0.11,
            linewidth=0,
        )
    axes[0].axhline(
        direct, color="black", ls="--", lw=1, label=baseline_label
    )
    axes[0].legend(fontsize=8)
    axes[0].set(
        xlabel="cumulative CLIP path",
        ylabel=r"KID ($\times 10^3$; $\downarrow$)",
        title="Global kernel distance",
    )
    axes[1].set(
        xlabel="cumulative CLIP path",
        ylabel=r"FID-200 ($\downarrow$)",
        title="Global Fréchet distance",
    )
    axes[1].axhline(direct_fid, color="black", ls="--", lw=1)
    axes[2].set(
        xlabel="cumulative CLIP path",
        ylabel=r"Max class probability ($\uparrow$)",
        title="Per-image semantic confidence",
    )
    add_noise_colorbar(fig, axes, sequential["rho"].min(), sequential["rho"].max())
    fig.suptitle(
        "Sequential U-turn diagnostics versus accumulated change", fontsize=11
    )
    save_figure(fig, output_dir, "sequential_uturn_quality_vs_accumulated_change")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.15), layout="constrained")
    for rho, group in sequential.groupby("rho"):
        group = group.sort_values("step")
        color = PAPER_CMAP(norm(rho))
        axes[0].plot(
            group["step"], group["kid"] * 1000, marker="o", ms=3, lw=1.6, color=color
        )
        axes[1].plot(
            group["step"],
            group["fid_matched_200"],
            marker="o",
            ms=3,
            lw=1.6,
            color=color,
        )
        axes[2].plot(
            group["step"],
            group["convnext_max_probability"],
            marker="o",
            ms=3,
            lw=1.6,
            color=color,
        )
    axes[0].axhline(direct, color="black", ls="--", lw=1)
    axes[0].set(
        xlabel="U-turn step",
        ylabel=r"KID ($\times 10^3$; $\downarrow$)",
        title="Global kernel distance",
    )
    axes[1].set(
        xlabel="U-turn step",
        ylabel=r"FID-200 ($\downarrow$)",
        title="Global Fréchet distance",
    )
    axes[1].axhline(direct_fid, color="black", ls="--", lw=1)
    axes[2].set(
        xlabel="U-turn step",
        ylabel=r"Max class probability ($\uparrow$)",
        title="Per-image semantic confidence",
    )
    add_noise_colorbar(fig, axes, sequential["rho"].min(), sequential["rho"].max())
    fig.suptitle("Accumulation of learned-denoiser error", fontsize=11)
    save_figure(fig, output_dir, "sequential_uturn_quality_vs_step")


def plot_single_sequential_comparison(
    single: pd.DataFrame,
    sequential: pd.DataFrame,
    direct_summary: pd.DataFrame | None,
    output_dir: Path,
    source_referenced: bool = False,
) -> None:
    if not np.isfinite(single["clip_cumulative_path"]).any():
        return
    kid_column = "source_kid" if source_referenced else "kid"
    kid_low_column = (
        "source_kid_ci_low" if source_referenced else "kid_ci_low"
    )
    kid_high_column = (
        "source_kid_ci_high" if source_referenced else "kid_ci_high"
    )
    fid_column = "source_fid_200" if source_referenced else "fid_matched_200"
    if kid_column not in single or fid_column not in single:
        return
    style_axes()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25), layout="constrained")
    norm = Normalize(
        min(single["rho"].min(), sequential["rho"].min()),
        max(single["rho"].max(), sequential["rho"].max()),
    )

    for rho, group in sequential.groupby("rho"):
        group = group.sort_values("clip_cumulative_path")
        color = PAPER_CMAP(norm(rho))
        axes[0].plot(
            group["clip_cumulative_path"],
            group[kid_column] * 1000,
            color=color,
            marker="o",
            ms=2.8,
            lw=1.35,
            alpha=0.9,
        )
        axes[1].plot(
            group["clip_cumulative_path"],
            group[fid_column],
            color=color,
            marker="o",
            ms=2.8,
            lw=1.35,
            alpha=0.9,
        )

    single = single.sort_values("clip_cumulative_path")
    axes[0].plot(
        single["clip_cumulative_path"],
        single[kid_column] * 1000,
        color="#595959",
        ls="--",
        lw=1.2,
        zorder=3,
    )
    axes[1].plot(
        single["clip_cumulative_path"],
        single[fid_column],
        color="#595959",
        ls="--",
        lw=1.2,
        zorder=3,
    )
    for row in single.itertuples():
        color = PAPER_CMAP(norm(row.rho))
        axes[0].errorbar(
            row.clip_cumulative_path,
            getattr(row, kid_column) * 1000,
            yerr=[
                [
                    (
                        getattr(row, kid_column)
                        - getattr(row, kid_low_column)
                    )
                    * 1000
                ],
                [
                    (
                        getattr(row, kid_high_column)
                        - getattr(row, kid_column)
                    )
                    * 1000
                ],
            ],
            fmt="D",
            ms=5.3,
            color=color,
            markeredgecolor="#333333",
            markeredgewidth=0.5,
            ecolor=color,
            elinewidth=0.7,
            capsize=1.5,
            zorder=4,
        )
        axes[1].scatter(
            row.clip_cumulative_path,
            getattr(row, fid_column),
            marker="D",
            s=30,
            color=color,
            edgecolor="#333333",
            linewidth=0.5,
            zorder=4,
        )

    if not source_referenced:
        baseline_kid = (
            direct_summary["kid"].iloc[0] * 1000
            if direct_summary is not None
            else sequential.loc[
                sequential["noise_step"] == 999, "kid"
            ].median()
            * 1000
        )
        baseline_fid = (
            direct_summary["fid_matched_200"].iloc[0]
            if direct_summary is not None
            else sequential.loc[
                sequential["noise_step"] == 999, "fid_matched_200"
            ].median()
        )
        for axis, baseline in zip(axes, (baseline_kid, baseline_fid)):
            axis.axhline(baseline, color="black", ls=":", lw=1.2)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("accumulated CLIP-patch change")
    if source_referenced:
        axes[0].set_ylabel(
            r"source-set KID ($\times 10^3$; $\downarrow$)"
        )
        axes[1].set_ylabel(r"source-set FID-200 ($\downarrow$)")
        axes[0].set_title("Drift from the 20 starting images")
        axes[1].set_title("Drift from the 20 starting images")
    else:
        axes[0].set_ylabel(
            r"KID to ImageNet ($\times 10^3$; $\downarrow$)"
        )
        axes[1].set_ylabel(r"FID-200 to ImageNet ($\downarrow$)")
        axes[0].set_title("Global kernel distance")
        axes[1].set_title("Global Fréchet distance")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#595959",
            ls="--",
            marker="D",
            markerfacecolor="white",
            label="single U-turn",
        ),
        Line2D(
            [0],
            [0],
            color="#595959",
            ls="-",
            marker="o",
            markerfacecolor="white",
            label="sequential U-turns",
        ),
    ]
    if not source_referenced:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                ls=":",
                label="direct diffusion",
            )
        )
    axes[0].legend(
        handles=legend_handles,
        fontsize=7.5,
    )
    add_noise_colorbar(
        fig,
        axes,
        min(single["rho"].min(), sequential["rho"].min()),
        max(single["rho"].max(), sequential["rho"].max()),
    )
    title = (
        "Single and sequential U-turn drift from their starting distribution"
        if source_referenced
        else "Global ImageNet fit versus accumulated change (coverage-sensitive)"
    )
    fig.suptitle(title, fontsize=11)
    stem = (
        "single_vs_sequential_quality_accumulated_change"
        if source_referenced
        else "single_vs_sequential_global_imagenet_accumulated_change"
    )
    save_figure(
        fig, output_dir, stem
    )


def plot_paired_quality_comparison(
    single: pd.DataFrame, sequential: pd.DataFrame, output_dir: Path
) -> None:
    metrics = (
        "convnext_feature_cosine_distance",
        "convnext_start_class_probability",
        "convnext_start_class_retained",
    )
    if any(column not in single for column in metrics):
        return
    if not np.isfinite(single[list(metrics)].to_numpy()).all():
        return

    style_axes()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2), layout="constrained")
    norm = Normalize(
        min(single["rho"].min(), sequential["rho"].min()),
        max(single["rho"].max(), sequential["rho"].max()),
    )
    for rho, group in sequential.groupby("rho"):
        group = group.sort_values("clip_cumulative_path")
        color = PAPER_CMAP(norm(rho))
        for axis, metric in zip(axes, metrics):
            axis.fill_between(
                group["clip_cumulative_path"],
                group[f"{metric}_ci_low"],
                group[f"{metric}_ci_high"],
                color=color,
                alpha=0.09,
                linewidth=0,
            )
            axis.plot(
                group["clip_cumulative_path"],
                group[metric],
                color=color,
                marker="o",
                ms=2.8,
                lw=1.35,
                alpha=0.9,
            )

    single = single.sort_values("clip_cumulative_path")
    for axis, metric in zip(axes, metrics):
        axis.plot(
            single["clip_cumulative_path"],
            single[metric],
            color="#595959",
            ls="--",
            lw=1.2,
            zorder=3,
        )
        for row in single.itertuples():
            value = getattr(row, metric)
            low = getattr(row, f"{metric}_ci_low")
            high = getattr(row, f"{metric}_ci_high")
            axis.errorbar(
                row.clip_cumulative_path,
                value,
                yerr=[[value - low], [high - value]],
                fmt="D",
                ms=5.3,
                color=PAPER_CMAP(norm(row.rho)),
                markeredgecolor="#333333",
                markeredgewidth=0.5,
                ecolor=PAPER_CMAP(norm(row.rho)),
                elinewidth=0.7,
                capsize=1.5,
                zorder=4,
            )
        axis.set_xscale("log")
        axis.set_xlabel("accumulated CLIP-patch change")

    axes[0].set(
        ylabel=r"ConvNeXt feature distance ($\downarrow$)",
        title="Perceptual-semantic change",
    )
    axes[1].set(
        ylabel=r"starting-class probability ($\uparrow$)",
        title="Original semantics",
        ylim=(0, 1),
    )
    axes[2].set(
        ylabel=r"top-1 class retained ($\uparrow$)",
        title="Class retention",
        ylim=(0, 1),
    )
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#595959",
                ls="--",
                marker="D",
                markerfacecolor="white",
                label="single U-turn",
            ),
            Line2D(
                [0],
                [0],
                color="#595959",
                ls="-",
                marker="o",
                markerfacecolor="white",
                label="sequential U-turns",
            ),
        ],
        fontsize=7.5,
    )
    add_noise_colorbar(
        fig,
        axes,
        min(single["rho"].min(), sequential["rho"].min()),
        max(single["rho"].max(), sequential["rho"].max()),
    )
    fig.suptitle(
        "Paired deterioration from the exact starting images", fontsize=11
    )
    save_figure(
        fig, output_dir, "single_vs_sequential_paired_quality_accumulated_change"
    )


def plot_manifold_quality_comparison(
    single: pd.DataFrame,
    sequential: pd.DataFrame,
    direct_summary: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    metrics = (
        "manifold_precision_k3",
        "manifold_density_k3",
        "manifold_nearest_distance",
        "convnext_class_perplexity",
    )
    if direct_summary is None or any(column not in single for column in metrics):
        return
    if not np.isfinite(single[list(metrics)].to_numpy()).all():
        return
    if not np.isfinite(direct_summary[list(metrics)].to_numpy()).all():
        return

    style_axes()
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.15), layout="constrained")
    norm = Normalize(
        min(single["rho"].min(), sequential["rho"].min()),
        max(single["rho"].max(), sequential["rho"].max()),
    )
    for rho, group in sequential.groupby("rho"):
        group = group.sort_values("clip_cumulative_path")
        color = PAPER_CMAP(norm(rho))
        for axis, metric in zip(axes, metrics):
            axis.fill_between(
                group["clip_cumulative_path"],
                group[f"{metric}_ci_low"],
                group[f"{metric}_ci_high"],
                color=color,
                alpha=0.09,
                linewidth=0,
            )
            axis.plot(
                group["clip_cumulative_path"],
                group[metric],
                color=color,
                marker="o",
                ms=2.8,
                lw=1.35,
                alpha=0.9,
            )

    single = single.sort_values("clip_cumulative_path")
    for axis, metric in zip(axes, metrics):
        axis.plot(
            single["clip_cumulative_path"],
            single[metric],
            color="#595959",
            ls="--",
            lw=1.2,
            zorder=3,
        )
        for row in single.itertuples():
            value = getattr(row, metric)
            low = getattr(row, f"{metric}_ci_low")
            high = getattr(row, f"{metric}_ci_high")
            axis.errorbar(
                row.clip_cumulative_path,
                value,
                yerr=[[value - low], [high - value]],
                fmt="D",
                ms=5.3,
                color=PAPER_CMAP(norm(row.rho)),
                markeredgecolor="#333333",
                markeredgewidth=0.5,
                ecolor=PAPER_CMAP(norm(row.rho)),
                elinewidth=0.7,
                capsize=1.5,
                zorder=4,
            )
        axis.axhspan(
            direct_summary[f"{metric}_ci_low"].iloc[0],
            direct_summary[f"{metric}_ci_high"].iloc[0],
            color="black",
            alpha=0.055,
            linewidth=0,
        )
        axis.axhline(
            direct_summary[metric].iloc[0],
            color="black",
            ls=":",
            lw=1.2,
        )
        axis.set_xscale("log")
        axis.set_xlabel("accumulated CLIP-patch change")

    axes[0].set(
        ylabel=r"manifold precision ($\uparrow$)",
        title="Real-manifold membership",
        ylim=(0, 1.035),
    )
    axes[1].set(
        ylabel=r"manifold density ($\uparrow$)",
        title="Local real-manifold support",
    )
    axes[2].set(
        ylabel=r"nearest-real feature distance ($\downarrow$)",
        title="Distance to real manifold",
    )
    axes[3].set(
        ylabel=r"ConvNeXt class perplexity ($\downarrow$)",
        title="Classifier uncertainty",
    )
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#595959",
                ls="--",
                marker="D",
                markerfacecolor="white",
                label="single U-turn",
            ),
            Line2D(
                [0],
                [0],
                color="#595959",
                ls="-",
                marker="o",
                markerfacecolor="white",
                label="sequential U-turns",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                ls=":",
                label="direct diffusion",
            ),
        ],
        fontsize=7.5,
    )
    add_noise_colorbar(
        fig,
        axes,
        min(single["rho"].min(), sequential["rho"].min()),
        max(single["rho"].max(), sequential["rho"].max()),
    )
    fig.suptitle("Per-image fidelity versus accumulated change", fontsize=10.5)
    save_figure(
        fig,
        output_dir,
        "single_vs_sequential_per_image_quality_accumulated_change",
    )


def plot_combined(
    single: pd.DataFrame,
    sequential: pd.DataFrame,
    direct_summary: pd.DataFrame | None,
    steering_json: Path | None,
    mh_steering_summary_csv: Path | None,
    output_dir: Path,
) -> None:
    if direct_summary is None:
        return
    style_axes()
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.2), layout="constrained")
    norm = Normalize(
        min(single["rho"].min(), sequential["rho"].min()),
        max(single["rho"].max(), sequential["rho"].max()),
    )
    metrics = (
        "manifold_precision_k3",
        "convnext_class_perplexity",
        "convnext_start_class_retained",
    )
    for rho, group in sequential.groupby("rho"):
        group = group.sort_values("clip_cumulative_path")
        color = PAPER_CMAP(norm(rho))
        for axis, metric in zip(axes[:3], metrics):
            axis.fill_between(
                group["clip_cumulative_path"],
                group[f"{metric}_ci_low"],
                group[f"{metric}_ci_high"],
                color=color,
                alpha=0.08,
                linewidth=0,
            )
            axis.plot(
                group["clip_cumulative_path"],
                group[metric],
                color=color,
                lw=1.35,
                marker="o",
                ms=2.6,
            )
    single = single.sort_values("clip_cumulative_path")
    for axis, metric in zip(axes[:3], metrics):
        axis.plot(
            single["clip_cumulative_path"],
            single[metric],
            color="#595959",
            ls="--",
            lw=1.15,
        )
        for row in single.itertuples():
            axis.scatter(
                row.clip_cumulative_path,
                getattr(row, metric),
                color=PAPER_CMAP(norm(row.rho)),
                marker="D",
                s=26,
                edgecolor="#333333",
                linewidth=0.45,
                zorder=4,
            )
        axis.set_xscale("log")
        axis.set_xlabel("accumulated CLIP-patch change")

    for axis, metric in zip(axes[:2], metrics[:2]):
        axis.axhspan(
            direct_summary[f"{metric}_ci_low"].iloc[0],
            direct_summary[f"{metric}_ci_high"].iloc[0],
            color="black",
            alpha=0.055,
            linewidth=0,
        )
        axis.axhline(
            direct_summary[metric].iloc[0], color="black", ls=":", lw=1.2
        )
    axes[0].set(
        ylabel=r"manifold precision ($\uparrow$)",
        title="Per-image fidelity",
        ylim=(0, 1.035),
    )
    axes[1].set(
        ylabel=r"class perplexity ($\downarrow$)",
        title="Classifier uncertainty",
    )
    axes[2].set(
        ylabel=r"starting class retained ($\uparrow$)",
        title="Semantic preservation",
        ylim=(0, 1),
    )
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#595959",
                ls="--",
                marker="D",
                markerfacecolor="white",
                label="single U-turn",
            ),
            Line2D(
                [0],
                [0],
                color="#595959",
                ls="-",
                marker="o",
                markerfacecolor="white",
                label="sequential U-turns",
            ),
            Line2D([0], [0], color="black", ls=":", label="direct diffusion"),
        ],
        fontsize=7.1,
    )

    plotted_steering = False
    if mh_steering_summary_csv is not None and mh_steering_summary_csv.exists():
        mh = pd.read_csv(mh_steering_summary_csv)
        focus = mh[
            (mh["target_mode"] == "dog_class")
            & np.isclose(mh["rho"], 0.4)
            & mh["energy_lambda"].isin([0.0, 4.0])
            & (mh["runs"] >= 12)
        ].sort_values("energy_lambda")
        if len(focus) == 2:
            values = focus["final_target_probability"].to_numpy()
            low = focus["final_target_probability_ci_low"].to_numpy()
            high = focus["final_target_probability_ci_high"].to_numpy()
            axes[3].bar(
                [0, 1],
                values,
                color=["#8A8A8A", "#D96B6B"],
                width=0.58,
                zorder=2,
            )
            axes[3].errorbar(
                [0, 1],
                values,
                yerr=[values - low, high - values],
                fmt="none",
                color="black",
                capsize=3,
                lw=1.0,
                zorder=3,
            )
            for index, value in enumerate(values):
                axes[3].text(
                    index,
                    high[index] + 0.018,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            axes[3].set_xticks(
                [0, 1], [r"$H=0$", r"$H=-4\log p_{\rm target}$"]
            )
            axes[3].set(
                ylabel=r"final target-class probability ($\uparrow$)",
                title=r"Exact MH steering ($\rho=0.4$)",
                ylim=(0, max(0.5, float(high.max()) * 1.16)),
            )
            plotted_steering = True
    if not plotted_steering and steering_json is not None and steering_json.exists():
        steering = json.loads(steering_json.read_text())
        modes = ["dog_to_dog", "dog_to_cat"]
        values = [steering[mode]["crossing_rate"] for mode in modes]
        axes[3].bar(
            ["dog to dog", "dog to cat"],
            values,
            color=["#3B6FB6", "#D96B6B"],
            width=0.62,
        )
        axes[3].set(
            ylim=(0, 1),
            ylabel=r"target exceeds source ($\uparrow$)",
            title="Cached steering",
        )

    fig.suptitle(
        "Image U-turn fidelity, preservation, and energy steering", fontsize=10.5
    )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=PAPER_CMAP)
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar,
        ax=axes[:3],
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label(r"noise fraction $\rho$")
    save_figure(fig, output_dir, "image_rebuttal_combined_summary")


def write_markdown(
    single: pd.DataFrame,
    sequential: pd.DataFrame,
    direct_summary: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    terminal = sequential[sequential["step"] == 100].copy()
    lines = [
        "# Image quality rebuttal results",
        "",
        "The primary fidelity metrics are computed per generated image against a "
        "fixed 50,000-image ImageNet reference and therefore do not reward generated-"
        "set diversity. Manifold precision is the fraction inside the real-data "
        "feature manifold; density measures local real-data support; nearest-real "
        "distance is the Inception-feature distance to the closest real image. "
        "ConvNeXt class perplexity is the mean exponentiated predictive entropy "
        "(lower means a more class-recognizable image); it is not model likelihood.",
        "",
        "Intervals on per-image observables are 500-resample bootstraps over starting "
        "images. Global KID/FID use Inception-v3 features against all 50,000 ImageNet "
        "validation images and are reported only as coverage-sensitive distribution-"
        "fit diagnostics. FID-200 uses 200 balanced generated samples.",
        "",
    ]
    if direct_summary is not None:
        row = direct_summary.iloc[0]
        lines += [
            "## Direct diffusion baseline",
            "",
            "| samples | manifold precision | density | nearest-real distance | "
            "class perplexity | max class prob. | KID x1e3 | FID-200 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {int(row.samples)} | {row.manifold_precision_k3:.3f} | "
            f"{row.manifold_density_k3:.3f} | "
            f"{row.manifold_nearest_distance:.2f} | "
            f"{row.convnext_class_perplexity:.1f} | "
            f"{row.convnext_max_probability:.3f} | "
            f"{1000 * row.kid:.2f} | {row.fid_matched_200:.2f} |",
            "",
        ]
    lines += [
        "## Single U-turn",
        "",
        "| rho | accumulated change | precision | density | nearest-real | "
        "class perplexity | start class retained | source FID |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in single.itertuples():
        source_fid = (
            f"{row.source_fid_200:.1f}"
            if hasattr(row, "source_fid_200") and np.isfinite(row.source_fid_200)
            else "-"
        )
        lines.append(
            f"| {row.rho:.3f} | {row.clip_cumulative_path:.3f} | "
            f"{row.manifold_precision_k3:.3f} | {row.manifold_density_k3:.3f} | "
            f"{row.manifold_nearest_distance:.2f} | "
            f"{row.convnext_class_perplexity:.1f} | "
            f"{row.convnext_start_class_retained:.3f} | {source_fid} |"
        )
    lines += [
        "",
        "## Sequential U-turn terminal state",
        "",
        "| rho | accumulated change | precision | density | nearest-real | "
        "class perplexity | start class retained | source FID |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in terminal.itertuples():
        source_fid = (
            f"{row.source_fid_200:.1f}"
            if hasattr(row, "source_fid_200") and np.isfinite(row.source_fid_200)
            else "-"
        )
        lines.append(
            f"| {row.rho:.3f} | {row.clip_cumulative_path:.3f} | "
            f"{row.manifold_precision_k3:.3f} | {row.manifold_density_k3:.3f} | "
            f"{row.manifold_nearest_distance:.2f} | "
            f"{row.convnext_class_perplexity:.1f} | "
            f"{row.convnext_start_class_retained:.3f} | {source_fid} |"
        )
    lines += [
        "",
        "## Learned-denoiser drift by noise level",
        "",
        "| rho | precision change / 100 U-turns | class-perplexity change / 100 | "
        "source-FID change / 100 |",
        "|---:|---:|---:|---:|",
    ]
    for rho, group in sequential.groupby("rho"):
        precision_slope = (
            np.polyfit(group["step"], group["manifold_precision_k3"], 1)[0] * 100
        )
        perplexity_slope = (
            np.polyfit(group["step"], group["convnext_class_perplexity"], 1)[0]
            * 100
        )
        source_fid_slope = (
            np.polyfit(group["step"], group["source_fid_200"], 1)[0] * 100
            if "source_fid_200" in group
            else np.nan
        )
        lines.append(
            f"| {rho:.3f} | {precision_slope:+.3f} | "
            f"{perplexity_slope:+.2f} | {source_fid_slope:+.1f} |"
        )
    (output_dir / "image_quality_key_results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steering-summary-json", type=Path)
    parser.add_argument("--mh-steering-summary-csv", type=Path)
    parser.add_argument("--direct-manifest", type=Path)
    parser.add_argument("--direct-feature-dir", type=Path)
    parser.add_argument("--compute-fid", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    single_frame = load_feature_frame(
        args.manifest_dir / "single_manifest.csv", args.feature_dir / "single"
    )
    sequential_frame = load_feature_frame(
        args.manifest_dir / "sequential_manifest.csv",
        args.feature_dir / "sequential",
    )
    single_clip_path = args.feature_dir / "single/clip_net_distance.npy"
    if single_clip_path.exists():
        single_clip_distance = np.load(single_clip_path, mmap_mode="r")
        if len(single_clip_distance) != len(single_frame):
            raise RuntimeError(
                f"{single_clip_path}: expected {len(single_frame)} rows, "
                f"found {len(single_clip_distance)}"
            )
        single_frame["clip_net_distance"] = single_clip_distance
        single_frame["clip_cumulative_path"] = single_clip_distance
    else:
        single_frame["clip_net_distance"] = np.nan
        single_frame["clip_cumulative_path"] = np.nan
    sequential_clip_dir = args.feature_dir / "sequential"
    sequential_net_path = (
        sequential_clip_dir / "clip_net_distance_from_images.npy"
    )
    sequential_cumulative_path = (
        sequential_clip_dir / "clip_cumulative_path_from_images.npy"
    )
    if sequential_net_path.exists() and sequential_cumulative_path.exists():
        sequential_frame["clip_net_distance"] = np.load(
            sequential_net_path, mmap_mode="r"
        )
        sequential_frame["clip_cumulative_path"] = np.load(
            sequential_cumulative_path, mmap_mode="r"
        )
    else:
        sequential_frame = add_clip_path_metrics(sequential_frame)
        sequential_frame = apply_clip_metric_overrides(
            sequential_frame,
            args.feature_dir / "sequential/clip_metric_overrides.csv",
        )
    single_frame.to_csv(args.output_dir / "single_sample_metrics.csv", index=False)
    sequential_frame.to_csv(
        args.output_dir / "sequential_sample_metrics.csv", index=False
    )

    single_features = np.load(
        args.feature_dir / "single/inception_features.npy", mmap_mode="r"
    )
    sequential_features = np.load(
        args.feature_dir / "sequential/inception_features.npy", mmap_mode="r"
    )
    direct_frame = direct_features = None
    if (
        args.direct_manifest is not None
        and args.direct_feature_dir is not None
        and args.direct_manifest.exists()
        and args.direct_feature_dir.exists()
    ):
        direct_frame = load_feature_frame(
            args.direct_manifest, args.direct_feature_dir
        )
        direct_frame["noise_step"] = 0
        direct_frame["rho"] = np.nan
        direct_frame["clip_net_distance"] = np.nan
        direct_frame["clip_cumulative_path"] = np.nan
        direct_features = np.load(
            args.direct_feature_dir / "inception_features.npy", mmap_mode="r"
        )
    reference_array = np.load(
        args.feature_dir / "reference/inception_features.npy", mmap_mode="r"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = torch.tensor(
        np.asarray(reference_array, dtype=np.float32), device=device
    )

    single, single_bins = summarize_single(
        single_frame, single_features, reference, device
    )
    sequential = summarize_sequential(
        sequential_frame, sequential_features, reference, device
    )
    direct_summary = (
        summarize_direct(direct_frame, direct_features, reference, device)
        if direct_frame is not None and direct_features is not None
        else None
    )
    shared_single = shared_sequential = None
    shared_single_frame = shared_sequential_frame = None
    source_features = None
    shared_manifest_path = args.manifest_dir / "shared_start_manifest.csv"
    shared_feature_path = args.feature_dir / "shared_start/inception_features.npy"
    if shared_manifest_path.exists() and shared_feature_path.exists():
        shared_starts = pd.read_csv(
            shared_manifest_path, keep_default_na=False
        )
        shared_ids = set(shared_starts["image_id"])
        shared_single_frame = single_frame[
            single_frame["image_id"].isin(shared_ids)
        ].copy()
        shared_sequential_frame = sequential_frame[
            sequential_frame["image_id"].isin(shared_ids)
        ].copy()
        shared_single, _ = summarize_single(
            shared_single_frame, single_features, reference, device
        )
        shared_sequential = summarize_sequential(
            shared_sequential_frame, sequential_features, reference, device
        )
        source_features = np.load(shared_feature_path, mmap_mode="r")
        if len(source_features) != len(shared_starts):
            raise RuntimeError(
                f"{shared_feature_path}: expected {len(shared_starts)} rows, "
                f"found {len(source_features)}"
            )

    if args.compute_fid:
        reference_mean, reference_covariance = reference_statistics(reference)
        single = add_matched_fid(
            single,
            single_frame,
            single_features,
            reference_mean,
            reference_covariance,
            "single",
        )
        sequential = add_matched_fid(
            sequential,
            sequential_frame,
            sequential_features,
            reference_mean,
            reference_covariance,
            "sequential",
        )
        if direct_summary is not None:
            direct_summary = add_matched_fid(
                direct_summary,
                direct_frame,
                direct_features,
                reference_mean,
                reference_covariance,
                "single",
            )
        if shared_single is not None and shared_sequential is not None:
            shared_single = add_matched_fid(
                shared_single,
                shared_single_frame,
                single_features,
                reference_mean,
                reference_covariance,
                "single",
            )
            shared_sequential = add_matched_fid(
                shared_sequential,
                shared_sequential_frame,
                sequential_features,
                reference_mean,
                reference_covariance,
                "sequential",
            )
            shared_single = add_source_reference_metrics(
                shared_single,
                shared_single_frame,
                single_features,
                source_features,
                "single",
            )
            shared_sequential = add_source_reference_metrics(
                shared_sequential,
                shared_sequential_frame,
                sequential_features,
                source_features,
                "sequential",
            )

    single.to_csv(args.output_dir / "single_uturn_quality_summary.csv", index=False)
    single_bins.to_csv(
        args.output_dir / "single_uturn_quality_change_bins.csv", index=False
    )
    sequential.to_csv(
        args.output_dir / "sequential_uturn_quality_summary.csv", index=False
    )
    if direct_summary is not None:
        direct_summary.to_csv(
            args.output_dir / "direct_diffusion_quality_summary.csv", index=False
        )
    if shared_single is not None and shared_sequential is not None:
        shared_single.to_csv(
            args.output_dir / "single_uturn_shared20_quality_summary.csv",
            index=False,
        )
        shared_sequential.to_csv(
            args.output_dir / "sequential_uturn_shared20_quality_summary.csv",
            index=False,
        )
    plot_single(single, direct_summary, args.output_dir)
    plot_sequential(sequential, direct_summary, args.output_dir)
    plot_single_sequential_comparison(
        shared_single if shared_single is not None else single,
        shared_sequential if shared_sequential is not None else sequential,
        direct_summary,
        args.output_dir,
    )
    if shared_single is not None and shared_sequential is not None:
        plot_single_sequential_comparison(
            shared_single,
            shared_sequential,
            None,
            args.output_dir,
            source_referenced=True,
        )
        plot_paired_quality_comparison(
            shared_single, shared_sequential, args.output_dir
        )
        plot_manifold_quality_comparison(
            shared_single,
            shared_sequential,
            direct_summary,
            args.output_dir,
        )
    plot_combined(
        shared_single if shared_single is not None else single,
        shared_sequential if shared_sequential is not None else sequential,
        direct_summary,
        args.steering_summary_json,
        args.mh_steering_summary_csv,
        args.output_dir,
    )
    write_markdown(
        shared_single if shared_single is not None else single,
        shared_sequential if shared_sequential is not None else sequential,
        direct_summary,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
