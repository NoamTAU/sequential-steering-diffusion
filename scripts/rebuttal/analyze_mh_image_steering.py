#!/usr/bin/env python3
"""Summarize classifier-energy MH image-steering runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAMBDA_COLORS = {0.0: "#777777", 1.0: "#6E93BF", 4.0: "#D96B6B"}
MODE_LABELS = {"dog_class": "dog to dog", "cat": "dog to cat"}


def bootstrap_mean_ci(
    values: np.ndarray, seed: int = 44, draws: int = 500
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def first_crossing(target: np.ndarray, source: np.ndarray) -> float:
    indices = np.flatnonzero(target >= source)
    return float(indices[0]) if len(indices) else float("nan")


def collect(root: Path) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows = []
    trajectories = {}
    for data_path in sorted(root.glob("*/mh_data.npz")):
        run_dir = data_path.parent
        config = json.loads((run_dir / "config.json").read_text())
        data = np.load(data_path)
        target = np.asarray(data["target_probability"], dtype=float)
        source = np.asarray(data["source_probability"], dtype=float)
        accepted = np.asarray(data["accepted"], dtype=bool)
        max_probability = np.asarray(data["max_probability"], dtype=float)
        row = {
            **config,
            "run_dir": str(run_dir),
            "steps_completed": len(target) - 1,
            "acceptance_rate": accepted[1:].mean(),
            "initial_target_probability": target[0],
            "final_target_probability": target[-1],
            "max_target_probability": target.max(),
            "final_source_probability": source[-1],
            "crossed": bool(np.any(target >= source)),
            "first_crossing": first_crossing(target, source),
            "final_target_ge_0_5": bool(target[-1] >= 0.5),
            "final_max_probability": max_probability[-1],
        }
        rows.append(row)
        trajectories[str(run_dir)] = {
            "target": target,
            "source": source,
            "accepted": accepted,
        }
    return pd.DataFrame(rows), trajectories


def grouped_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "acceptance_rate",
        "initial_target_probability",
        "final_target_probability",
        "max_target_probability",
        "final_source_probability",
        "crossed",
        "first_crossing",
        "final_target_ge_0_5",
        "final_max_probability",
    ]
    rows = []
    for keys, group in frame.groupby(
        ["target_mode", "rho", "energy_lambda"], sort=True
    ):
        row = {
            "target_mode": keys[0],
            "rho": keys[1],
            "energy_lambda": keys[2],
            "runs": len(group),
        }
        for column in columns:
            values = group[column].astype(float)
            row[column] = values.mean()
            row[f"{column}_sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
            low, high = bootstrap_mean_ci(
                values.to_numpy(), seed=44 + int(100 * keys[1]) + int(keys[2])
            )
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def padded(values: np.ndarray, length: int) -> np.ndarray:
    return np.pad(values, (0, length - len(values)), mode="edge")


def plot_trajectories(
    frame: pd.DataFrame, trajectories: dict[str, dict], output_dir: Path
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    rhos = sorted(frame["rho"].unique())
    fig, axes = plt.subplots(2, len(rhos), figsize=(10, 5.4), sharex=True, sharey=True)
    for row_index, mode in enumerate(("dog_class", "cat")):
        for column_index, rho in enumerate(rhos):
            axis = axes[row_index, column_index]
            subset = frame[
                frame["target_mode"].eq(mode) & frame["rho"].eq(rho)
            ]
            for energy_lambda, group in subset.groupby("energy_lambda"):
                arrays = [
                    trajectories[run_dir]["target"]
                    for run_dir in group["run_dir"].tolist()
                ]
                length = max(map(len, arrays))
                values = np.stack([padded(array, length) for array in arrays])
                mean = values.mean(axis=0)
                low = values.min(axis=0)
                high = values.max(axis=0)
                steps = np.arange(length)
                color = LAMBDA_COLORS.get(float(energy_lambda), "#333333")
                axis.plot(
                    steps,
                    mean,
                    color=color,
                    lw=1.8,
                    label=rf"$\lambda={energy_lambda:g}$",
                )
                axis.fill_between(steps, low, high, color=color, alpha=0.14)
            axis.set_title(rf"{MODE_LABELS[mode]}, $\rho={rho:.1f}$")
            if row_index == 1:
                axis.set_xlabel("U-turn step")
            if column_index == 0:
                axis.set_ylabel(r"target probability ($\uparrow$)")
            axis.set_ylim(0, 1)
    axes[0, 0].legend(frameon=True, framealpha=0.82, fontsize=8)
    fig.suptitle("Classifier-energy Metropolis-Hastings steering", y=1.01)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"mh_image_steering_trajectories.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_trajectory_summary(
    frame: pd.DataFrame, trajectories: dict[str, dict], output_dir: Path
) -> None:
    rows = []
    for keys, group in frame.groupby(
        ["target_mode", "rho", "energy_lambda"], sort=True
    ):
        target_arrays = [
            trajectories[run_dir]["target"] for run_dir in group["run_dir"].tolist()
        ]
        source_arrays = [
            trajectories[run_dir]["source"] for run_dir in group["run_dir"].tolist()
        ]
        length = max(map(len, target_arrays))
        values = np.stack([padded(array, length) for array in target_arrays])
        source_values = np.stack(
            [padded(array, length) for array in source_arrays]
        )
        instantaneous_crossing = (values >= source_values).astype(float)
        crossing = np.maximum.accumulate(values >= source_values, axis=1).astype(
            float
        )
        rng = np.random.default_rng(
            44 + int(round(100 * keys[1])) + int(keys[2])
        )
        bootstrap_indices = rng.integers(
            0, len(crossing), size=(500, len(crossing))
        )
        bootstrap_crossing = crossing[bootstrap_indices].mean(axis=1)
        crossing_low = np.quantile(bootstrap_crossing, 0.025, axis=0)
        crossing_high = np.quantile(bootstrap_crossing, 0.975, axis=0)
        bootstrap_instantaneous = instantaneous_crossing[
            bootstrap_indices
        ].mean(axis=1)
        instantaneous_low = np.quantile(
            bootstrap_instantaneous, 0.025, axis=0
        )
        instantaneous_high = np.quantile(
            bootstrap_instantaneous, 0.975, axis=0
        )
        for step in range(length):
            target_column = values[:, step]
            source_column = source_values[:, step]
            rows.append(
                {
                    "target_mode": keys[0],
                    "rho": keys[1],
                    "energy_lambda": keys[2],
                    "step": step,
                    "mean_target_probability": target_column.mean(),
                    "q025_target_probability": np.quantile(
                        target_column, 0.025
                    ),
                    "q975_target_probability": np.quantile(
                        target_column, 0.975
                    ),
                    "min_target_probability": target_column.min(),
                    "max_target_probability": target_column.max(),
                    "mean_source_probability": source_column.mean(),
                    "q025_source_probability": np.quantile(
                        source_column, 0.025
                    ),
                    "q975_source_probability": np.quantile(
                        source_column, 0.975
                    ),
                    "min_source_probability": source_column.min(),
                    "max_source_probability": source_column.max(),
                    "instantaneous_crossing_rate": (
                        instantaneous_crossing[:, step].mean()
                    ),
                    "instantaneous_crossing_ci_low": instantaneous_low[step],
                    "instantaneous_crossing_ci_high": instantaneous_high[step],
                    "cumulative_crossing_rate": crossing[:, step].mean(),
                    "cumulative_crossing_ci_low": crossing_low[step],
                    "cumulative_crossing_ci_high": crossing_high[step],
                    "runs": len(target_column),
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / "mh_image_steering_trajectories.csv", index=False
    )


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.7, 3))
    markers = {"dog_class": "o", "cat": "s"}
    for mode in ("dog_class", "cat"):
        for energy_lambda in sorted(summary["energy_lambda"].unique()):
            group = summary[
                summary["target_mode"].eq(mode)
                & summary["energy_lambda"].eq(energy_lambda)
            ].sort_values("rho")
            label = f"{MODE_LABELS[mode]}, lambda={energy_lambda:g}"
            style = dict(
                marker=markers[mode],
                color=LAMBDA_COLORS.get(float(energy_lambda), "#333333"),
                lw=1.4,
                ms=4,
                label=label,
            )
            axes[0].plot(group["rho"], group["final_target_probability"], **style)
            axes[1].plot(group["rho"], group["acceptance_rate"], **style)
            axes[2].plot(group["rho"], group["crossed"], **style)
    axes[0].set(
        xlabel=r"$\rho$", ylabel=r"final target probability ($\uparrow$)"
    )
    axes[1].set(xlabel=r"$\rho$", ylabel="acceptance rate", ylim=(0, 1))
    axes[2].set(
        xlabel=r"$\rho$", ylabel=r"target exceeds source ($\uparrow$)", ylim=(0, 1)
    )
    axes[0].legend(frameon=True, framealpha=0.82, fontsize=6.8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"mh_image_steering_summary.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_focused_exact_mh(
    frame: pd.DataFrame, trajectories: dict[str, dict], output_dir: Path
) -> None:
    focused = frame[
        frame["target_mode"].eq("dog_class")
        & frame["rho"].eq(0.4)
        & frame["energy_lambda"].isin([0.0, 4.0])
    ].copy()
    if focused.groupby("energy_lambda").size().min() < 5:
        return

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), layout="constrained")
    for energy_lambda in (0.0, 4.0):
        group = focused[focused["energy_lambda"].eq(energy_lambda)]
        arrays = [
            trajectories[run_dir]["target"] for run_dir in group["run_dir"].tolist()
        ]
        length = max(map(len, arrays))
        values = np.stack([padded(array, length) for array in arrays])
        mean = values.mean(axis=0)
        low = np.quantile(values, 0.025, axis=0)
        high = np.quantile(values, 0.975, axis=0)
        steps = np.arange(length)
        color = LAMBDA_COLORS[energy_lambda]
        axes[0].plot(
            steps,
            mean,
            color=color,
            lw=1.8,
            label=rf"$\lambda={energy_lambda:g}$",
        )
        axes[0].fill_between(steps, low, high, color=color, alpha=0.14, linewidth=0)

        endpoints = group["final_target_probability"].to_numpy(dtype=float)
        rng = np.random.default_rng(42 + int(energy_lambda))
        x = np.full(len(endpoints), energy_lambda) + rng.uniform(
            -0.12, 0.12, size=len(endpoints)
        )
        axes[1].scatter(x, endpoints, color=color, alpha=0.72, s=18)
        ci_low, ci_high = bootstrap_mean_ci(endpoints)
        axes[1].errorbar(
            energy_lambda,
            endpoints.mean(),
            yerr=[
                [endpoints.mean() - ci_low],
                [ci_high - endpoints.mean()],
            ],
            fmt="D",
            color="black",
            markerfacecolor=color,
            ms=5,
            capsize=2,
            zorder=4,
        )

    axes[0].set(
        xlabel="U-turn step",
        ylabel=r"target-class probability ($\uparrow$)",
        title=r"Trajectory at $\rho=0.4$",
        ylim=(0, 0.65),
    )
    axes[0].legend(frameon=True, framealpha=0.82, fontsize=8)
    axes[1].set(
        xticks=[0, 4],
        xticklabels=[r"$H=0$", r"$H=-4\log p_{\rm target}$"],
        ylabel=r"final target-class probability ($\uparrow$)",
        title="Independent-chain endpoints",
        ylim=(0, 0.65),
    )
    fig.suptitle("Classifier-energy Metropolis-Hastings steering", fontsize=11)
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"mh_image_steering_focused.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Classifier-energy MH image steering",
        "",
        r"The target is \(P_H(x) \propto P(x)p_{\rm target}(x)^\lambda\), "
        r"equivalently \(H(x)=-\lambda\log p_{\rm target}(x)\). "
        "Each row averages independent chains; \\(\\lambda=0\\) is the unsteered control.",
        "",
        "| target | rho | lambda | runs | target prob. start -> final | acceptance | "
        "target > source | max class prob. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {MODE_LABELS[row.target_mode]} | {row.rho:.1f} | "
            f"{row.energy_lambda:g} | {row.runs} | "
            f"{row.initial_target_probability:.4f} -> "
            f"{row.final_target_probability:.4f} | "
            f"{row.acceptance_rate:.3f} | {row.crossed:.3f} | "
            f"{row.final_max_probability:.3f} |"
        )
    focused = summary[
        summary["target_mode"].eq("dog_class")
        & summary["rho"].eq(0.4)
        & summary["energy_lambda"].isin([0.0, 4.0])
    ].sort_values("energy_lambda")
    if len(focused) == 2:
        control, steered = list(focused.itertuples())
        lines += [
            "",
            "## Focused MH-rule result",
            "",
            f"At rho=0.4, the unsteered endpoint target probability was "
            f"{control.final_target_probability:.3f} "
            f"[{control.final_target_probability_ci_low:.3f}, "
            f"{control.final_target_probability_ci_high:.3f}], versus "
            f"{steered.final_target_probability:.3f} "
            f"[{steered.final_target_probability_ci_low:.3f}, "
            f"{steered.final_target_probability_ci_high:.3f}] for lambda=4 "
            f"(n={int(steered.runs)} independent chains per condition).",
        ]
    (output_dir / "mh_image_steering_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, trajectories = collect(args.root)
    if frame.empty:
        raise RuntimeError(f"No completed MH runs found under {args.root}")
    summary = grouped_summary(frame)
    frame.to_csv(args.output_dir / "mh_image_steering_runs.csv", index=False)
    summary.to_csv(args.output_dir / "mh_image_steering_summary.csv", index=False)
    write_trajectory_summary(frame, trajectories, args.output_dir)
    write_markdown(summary, args.output_dir)
    plot_trajectories(frame, trajectories, args.output_dir)
    plot_summary(summary, args.output_dir)
    plot_focused_exact_mh(frame, trajectories, args.output_dir)


if __name__ == "__main__":
    main()
