#!/usr/bin/env python3
"""Summarize the cached greedy dog-to-dog and dog-to-cat steering runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "dog_to_dog": "#3B6FB6",
    "dog_to_cat": "#D96B6B",
}
LABELS = {
    "dog_to_dog": "dog to dog",
    "dog_to_cat": "dog to cat",
}
RUN_ID_RE = re.compile(r"_run(\d+)_")
TARGET_DIR_RE = re.compile(r"target(?:_auto)?_(\d+)")


def bootstrap_image_mean(
    values: pd.Series, image_ids: pd.Series, seed: int = 44, draws: int = 500
) -> tuple[float, float, float]:
    frame = pd.DataFrame(
        {
            "value": np.asarray(values),
            "image_id": np.asarray(image_ids),
        }
    ).dropna()
    by_image = frame.groupby("image_id")["value"].mean()
    rng = np.random.default_rng(seed)
    samples = rng.choice(by_image.to_numpy(), (draws, len(by_image)), replace=True)
    means = samples.mean(axis=1)
    return (
        float(by_image.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def read_active_images(path: Path) -> set[str]:
    return {Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()}


def run_id(name: str) -> int:
    match = RUN_ID_RE.search(name)
    return int(match.group(1)) if match else -1


def selected_dog_classes(run_dir: str) -> tuple[int, int]:
    path = Path(run_dir)
    start_info = json.loads((path / "start_image_info.json").read_text())
    orig_idx = int(start_info["top1_class_idx"])
    auto_target = path / "auto_target.json"
    if auto_target.exists():
        target_idx = int(json.loads(auto_target.read_text())["target_class_idx"])
    else:
        matches = [
            TARGET_DIR_RE.fullmatch(part)
            for part in path.parts
            if TARGET_DIR_RE.fullmatch(part)
        ]
        if not matches:
            raise RuntimeError(f"Cannot recover target class from {path}")
        target_idx = int(matches[-1].group(1))
    return orig_idx, target_idx


def saved_seed(run_dir: str) -> int:
    path = Path(run_dir) / "start_image_info.json"
    return int(json.loads(path.read_text())["seed"])


def clean_summary(
    path: Path,
    active: set[str],
    mode: str,
    preferred_seeds: dict[tuple[str, int], int] | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[
        frame["image_name"].isin(active)
        & frame["repeat_index"].between(0, 3, inclusive="both")
    ].copy()
    frame["run_id"] = frame["run_dir_name"].map(run_id)
    frame["seed"] = frame["run_dir"].map(saved_seed)
    if preferred_seeds is None:
        frame = frame.sort_values("run_id").drop_duplicates(
            ["image_name", "repeat_index"], keep="last"
        )
    else:
        selected = []
        for (image_name, repeat_index), group in frame.groupby(
            ["image_name", "repeat_index"], sort=False
        ):
            expected_seed = preferred_seeds[(image_name, int(repeat_index))]
            matches = group[group["seed"].eq(expected_seed)]
            if matches.empty:
                raise RuntimeError(
                    f"{mode}: no seed-matched run for {(image_name, repeat_index)} "
                    f"with seed {expected_seed}"
                )
            selected.append(matches.sort_values("run_id").iloc[-1])
        frame = pd.DataFrame(selected)
    expected = len(active) * 4
    if len(frame) != expected:
        raise RuntimeError(f"{mode}: expected {expected} clean runs, found {len(frame)}")
    if mode == "dog_to_dog":
        invalid = []
        for row in frame.itertuples():
            orig_idx, target_idx = selected_dog_classes(row.run_dir)
            if orig_idx == target_idx:
                invalid.append((row.image_name, row.repeat_index, orig_idx))
        if invalid:
            raise RuntimeError(
                "dog_to_dog: selected runs contain identical source and target "
                f"classes: {invalid[:5]}"
            )
    frame["mode"] = mode
    return frame


def load_probability_trajectory(
    run_dir: str, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(Path(run_dir) / "steering_data.npz", allow_pickle=True)
    if mode == "dog_to_cat":
        source = np.asarray(data["probs_dog"], dtype=float)
        target = np.asarray(data["probs_cat"], dtype=float)
    else:
        source = np.asarray(data["probs_orig"], dtype=float)
        target = np.asarray(data["probs_target"], dtype=float)
    attempts = np.asarray(data["attempts"], dtype=float)
    return source, target, attempts


def pad_last(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) >= length:
        return values[:length]
    return np.pad(values, (0, length - len(values)), mode="edge")


def trajectory_statistics(frame: pd.DataFrame, max_step: int = 50) -> dict:
    target_rows = []
    source_rows = []
    instantaneous_crossing_rows = []
    crossing_rows = []
    run_rows = []
    for row in frame.itertuples():
        source_raw, target_raw, attempts = load_probability_trajectory(
            row.run_dir, row.mode
        )
        first_crossing = np.flatnonzero(target_raw >= source_raw)
        first_target_half = np.flatnonzero(target_raw >= 0.5)
        recorded_steps = max(0, len(target_raw) - 1)
        attempted = attempts[1 : recorded_steps + 1]
        accepted_steps = int(np.sum(np.diff(target_raw) > 1e-6))
        run_rows.append(
            {
                "mode": row.mode,
                "image_name": row.image_name,
                "repeat_index": int(row.repeat_index),
                "recorded_steps": recorded_steps,
                "accepted_steps": accepted_steps,
                "skipped_steps": recorded_steps - accepted_steps,
                "total_proposals": float(attempted.sum()),
                "proposals_per_recorded_step": (
                    float(attempted.sum() / recorded_steps)
                    if recorded_steps
                    else np.nan
                ),
                "crossed": float(len(first_crossing) > 0),
                "first_crossing_step": (
                    float(first_crossing[0]) if len(first_crossing) else np.nan
                ),
                "restricted_first_crossing_step": (
                    float(first_crossing[0])
                    if len(first_crossing)
                    else float(max_step + 1)
                ),
                "final_target_side": float(target_raw[-1] >= source_raw[-1]),
                "final_target_probability_ge_0_5": float(target_raw[-1] >= 0.5),
                "first_target_probability_ge_0_5_step": (
                    float(first_target_half[0])
                    if len(first_target_half)
                    else float(max_step + 1)
                ),
                "final_target_probability": float(target_raw[-1]),
                "final_source_probability": float(source_raw[-1]),
            }
        )
        target = pad_last(target_raw, max_step + 1)
        source = pad_last(source_raw, max_step + 1)
        target_rows.append(target)
        source_rows.append(source)
        instantaneous_crossing_rows.append((target >= source).astype(float))
        crossing = np.maximum.accumulate(target >= source).astype(float)
        crossing_rows.append(crossing)

    target = np.stack(target_rows)
    source = np.stack(source_rows)
    instantaneous_crossing = np.stack(instantaneous_crossing_rows)
    crossing = np.stack(crossing_rows)
    image_ids = frame["image_name"].to_numpy()
    unique_images = np.unique(image_ids)

    def image_aggregate(array: np.ndarray) -> np.ndarray:
        return np.stack([array[image_ids == image].mean(axis=0) for image in unique_images])

    target_image = image_aggregate(target)
    source_image = image_aggregate(source)
    instantaneous_crossing_image = image_aggregate(instantaneous_crossing)
    crossing_image = image_aggregate(crossing)
    return {
        "target": target,
        "source": source,
        "instantaneous_crossing": instantaneous_crossing,
        "crossing": crossing,
        "target_image": target_image,
        "source_image": source_image,
        "instantaneous_crossing_image": instantaneous_crossing_image,
        "crossing_image": crossing_image,
        "unique_images": unique_images,
        "run_metrics": pd.DataFrame(run_rows),
    }


def curve_ci(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = array.mean(axis=0)
    sem = array.std(axis=0, ddof=1) / np.sqrt(array.shape[0])
    return (
        mean,
        np.clip(mean - 1.96 * sem, 0.0, 1.0),
        np.clip(mean + 1.96 * sem, 0.0, 1.0),
    )


def summarize_mode(frame: pd.DataFrame, trajectories: dict) -> dict:
    run_metrics = trajectories["run_metrics"]
    final_target = bootstrap_image_mean(
        pd.Series(trajectories["target"][:, -1]), frame["image_name"]
    )
    crossing = bootstrap_image_mean(
        pd.Series(trajectories["crossing"][:, -1]), frame["image_name"]
    )
    target_majority = bootstrap_image_mean(
        run_metrics["final_target_probability_ge_0_5"],
        frame["image_name"],
    )
    final_target_side = bootstrap_image_mean(
        run_metrics["final_target_side"], frame["image_name"]
    )
    restricted_crossing = bootstrap_image_mean(
        run_metrics["restricted_first_crossing_step"], frame["image_name"]
    )
    proposals_per_step = bootstrap_image_mean(
        run_metrics["proposals_per_recorded_step"], frame["image_name"]
    )
    successful_steps = run_metrics.loc[
        run_metrics["crossed"].astype(bool), "first_crossing_step"
    ]
    per_image_crossing = (
        run_metrics.assign(
            successful_step=run_metrics["first_crossing_step"].where(
                run_metrics["crossed"].astype(bool)
            )
        )
        .groupby("image_name")["successful_step"]
        .median()
        .dropna()
    )
    initial_target = float(trajectories["target"][:, 0].mean())
    target_curve = trajectories["target_image"].mean(axis=0)
    source_curve = trajectories["source_image"].mean(axis=0)
    mean_crossings = np.flatnonzero(target_curve >= source_curve)
    target_side_curve = trajectories["instantaneous_crossing_image"].mean(axis=0)
    majority_steps = np.flatnonzero(target_side_curve >= 0.5)
    return {
        "images": int(frame["image_name"].nunique()),
        "runs": int(len(frame)),
        "initial_target_probability": initial_target,
        "final_target_probability": final_target[0],
        "final_target_probability_ci_low": final_target[1],
        "final_target_probability_ci_high": final_target[2],
        "crossing_rate": crossing[0],
        "crossing_rate_ci_low": crossing[1],
        "crossing_rate_ci_high": crossing[2],
        "target_probability_ge_0_5": target_majority[0],
        "target_probability_ge_0_5_ci_low": target_majority[1],
        "target_probability_ge_0_5_ci_high": target_majority[2],
        "median_first_crossing_successes": float(successful_steps.median()),
        "median_image_level_first_crossing": float(per_image_crossing.median()),
        "mean_curve_crossing_step": (
            int(mean_crossings[0]) if len(mean_crossings) else None
        ),
        "majority_target_side_step": (
            int(majority_steps[0]) if len(majority_steps) else None
        ),
        "final_target_side_rate": final_target_side[0],
        "final_target_side_rate_ci_low": final_target_side[1],
        "final_target_side_rate_ci_high": final_target_side[2],
        "restricted_mean_first_crossing_step": restricted_crossing[0],
        "restricted_mean_first_crossing_step_ci_low": restricted_crossing[1],
        "restricted_mean_first_crossing_step_ci_high": restricted_crossing[2],
        "mean_proposals_per_recorded_step": proposals_per_step[0],
        "mean_proposals_per_recorded_step_ci_low": proposals_per_step[1],
        "mean_proposals_per_recorded_step_ci_high": proposals_per_step[2],
    }


def paired_image_contrast(
    stats: dict[str, dict], metric: str, cat_minus_dog: bool = False
) -> tuple[float, float, float]:
    image_values = {}
    for mode in ("dog_to_dog", "dog_to_cat"):
        image_values[mode] = (
            stats[mode]["trajectories"]["run_metrics"]
            .groupby("image_name")[metric]
            .mean()
        )
    common = image_values["dog_to_dog"].index.intersection(
        image_values["dog_to_cat"].index
    )
    dog = image_values["dog_to_dog"].loc[common].to_numpy(dtype=float)
    cat = image_values["dog_to_cat"].loc[common].to_numpy(dtype=float)
    difference = cat - dog if cat_minus_dog else dog - cat
    rng = np.random.default_rng(44)
    samples = rng.choice(difference, (500, len(difference)), replace=True).mean(axis=1)
    return (
        float(difference.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def comparison_summary(stats: dict[str, dict]) -> dict:
    specifications = {
        "ever_crossing_rate_dog_minus_cat": ("crossed", False),
        "final_target_side_rate_dog_minus_cat": ("final_target_side", False),
        "target_probability_ge_0_5_dog_minus_cat": (
            "final_target_probability_ge_0_5",
            False,
        ),
        "restricted_crossing_steps_cat_minus_dog": (
            "restricted_first_crossing_step",
            True,
        ),
        "final_target_probability_dog_minus_cat": (
            "final_target_probability",
            False,
        ),
        "proposals_per_step_dog_minus_cat": (
            "proposals_per_recorded_step",
            False,
        ),
    }
    output = {}
    for name, (metric, cat_minus_dog) in specifications.items():
        value, low, high = paired_image_contrast(stats, metric, cat_minus_dog)
        output[name] = value
        output[f"{name}_ci_low"] = low
        output[f"{name}_ci_high"] = high
    dog_keys = stats["dog_to_dog"]["frame"][
        ["image_name", "repeat_index", "seed"]
    ].rename(columns={"seed": "dog_seed"})
    cat_keys = stats["dog_to_cat"]["frame"][
        ["image_name", "repeat_index", "seed"]
    ].rename(columns={"seed": "cat_seed"})
    paired = dog_keys.merge(cat_keys, on=["image_name", "repeat_index"])
    output["seed_matched_runs"] = int(paired["dog_seed"].eq(paired["cat_seed"]).sum())
    output["total_matched_runs"] = int(len(paired))
    return output


def plot_summary(stats: dict[str, dict], output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.05))
    steps = np.arange(51)

    for mode in ("dog_to_dog", "dog_to_cat"):
        values = stats[mode]["trajectories"]
        mean, low, high = curve_ci(values["target_image"])
        axes[0].plot(steps, mean, color=COLORS[mode], lw=2, label=LABELS[mode])
        axes[0].fill_between(steps, low, high, color=COLORS[mode], alpha=0.18)

        mean, low, high = curve_ci(values["crossing_image"])
        axes[1].plot(steps, mean, color=COLORS[mode], lw=2, label=LABELS[mode])
        axes[1].fill_between(steps, low, high, color=COLORS[mode], alpha=0.18)

    axes[0].set(
        xlabel="U-turn step", ylabel=r"target probability ($\uparrow$)", ylim=(0, 1)
    )
    axes[1].set(
        xlabel="U-turn step",
        ylabel=r"target exceeds source ($\uparrow$)",
        ylim=(0, 1),
    )
    axes[0].legend(frameon=True, framealpha=0.82, edgecolor="#D0D0D0")

    modes = ["dog_to_dog", "dog_to_cat"]
    x = np.arange(2)
    final = [stats[mode]["summary"]["final_target_probability"] for mode in modes]
    crossing = [stats[mode]["summary"]["crossing_rate"] for mode in modes]
    width = 0.34
    axes[2].bar(
        x - width / 2,
        final,
        width,
        color=[COLORS[mode] for mode in modes],
        alpha=0.72,
        label="final target prob.",
    )
    axes[2].bar(
        x + width / 2,
        crossing,
        width,
        color=[COLORS[mode] for mode in modes],
        alpha=1,
        hatch="//",
        label="crossing rate",
    )
    axes[2].set_xticks(x, ["dog to dog", "dog to cat"])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel(r"rate / probability ($\uparrow$)")
    axes[2].legend(frameon=True, framealpha=0.82, edgecolor="#D0D0D0", fontsize=8)

    fig.suptitle("Cached selection-based image steering at $\\rho=0.1$", y=1.01)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"cached_image_steering_summary.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_trajectory_summary(stats: dict[str, dict], output_dir: Path) -> None:
    rows = []
    for mode in ("dog_to_dog", "dog_to_cat"):
        trajectories = stats[mode]["trajectories"]
        for metric, key in (
            ("target_probability", "target_image"),
            ("source_probability", "source_image"),
            (
                "instantaneous_crossing_rate",
                "instantaneous_crossing_image",
            ),
            ("cumulative_crossing_rate", "crossing_image"),
        ):
            mean, low, high = curve_ci(trajectories[key])
            for step, (value, ci_low, ci_high) in enumerate(
                zip(mean, low, high)
            ):
                rows.append(
                    {
                        "mode": mode,
                        "step": step,
                        "metric": metric,
                        "mean": value,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "images": trajectories[key].shape[0],
                        "runs_per_image": 4,
                    }
                )
    pd.DataFrame(rows).to_csv(
        output_dir / "cached_image_steering_trajectories.csv", index=False
    )


def write_difficulty_csv(
    stats: dict[str, dict], comparison: dict, output_dir: Path
) -> None:
    dog = stats["dog_to_dog"]["summary"]
    cat = stats["dog_to_cat"]["summary"]
    rows = [
        {
            "metric": "ever target-over-source crossing rate",
            "dog_to_dog": dog["crossing_rate"],
            "dog_to_cat": cat["crossing_rate"],
            "contrast": comparison["ever_crossing_rate_dog_minus_cat"],
            "contrast_definition": "dog_to_dog - dog_to_cat",
            "contrast_ci_low": comparison[
                "ever_crossing_rate_dog_minus_cat_ci_low"
            ],
            "contrast_ci_high": comparison[
                "ever_crossing_rate_dog_minus_cat_ci_high"
            ],
        },
        {
            "metric": "final target-side occupancy",
            "dog_to_dog": dog["final_target_side_rate"],
            "dog_to_cat": cat["final_target_side_rate"],
            "contrast": comparison["final_target_side_rate_dog_minus_cat"],
            "contrast_definition": "dog_to_dog - dog_to_cat",
            "contrast_ci_low": comparison[
                "final_target_side_rate_dog_minus_cat_ci_low"
            ],
            "contrast_ci_high": comparison[
                "final_target_side_rate_dog_minus_cat_ci_high"
            ],
        },
        {
            "metric": "restricted mean first-crossing step (never=51)",
            "dog_to_dog": dog["restricted_mean_first_crossing_step"],
            "dog_to_cat": cat["restricted_mean_first_crossing_step"],
            "contrast": comparison["restricted_crossing_steps_cat_minus_dog"],
            "contrast_definition": "dog_to_cat - dog_to_dog",
            "contrast_ci_low": comparison[
                "restricted_crossing_steps_cat_minus_dog_ci_low"
            ],
            "contrast_ci_high": comparison[
                "restricted_crossing_steps_cat_minus_dog_ci_high"
            ],
        },
        {
            "metric": "final target probability >= 0.5",
            "dog_to_dog": dog["target_probability_ge_0_5"],
            "dog_to_cat": cat["target_probability_ge_0_5"],
            "contrast": comparison[
                "target_probability_ge_0_5_dog_minus_cat"
            ],
            "contrast_definition": "dog_to_dog - dog_to_cat",
            "contrast_ci_low": comparison[
                "target_probability_ge_0_5_dog_minus_cat_ci_low"
            ],
            "contrast_ci_high": comparison[
                "target_probability_ge_0_5_dog_minus_cat_ci_high"
            ],
        },
        {
            "metric": "proposals per recorded U-turn step",
            "dog_to_dog": dog["mean_proposals_per_recorded_step"],
            "dog_to_cat": cat["mean_proposals_per_recorded_step"],
            "contrast": comparison["proposals_per_step_dog_minus_cat"],
            "contrast_definition": "dog_to_dog - dog_to_cat",
            "contrast_ci_low": comparison[
                "proposals_per_step_dog_minus_cat_ci_low"
            ],
            "contrast_ci_high": comparison[
                "proposals_per_step_dog_minus_cat_ci_high"
            ],
        },
    ]
    pd.DataFrame(rows).to_csv(
        output_dir / "selection_steering_difficulty_summary.csv", index=False
    )


def write_markdown(
    stats: dict[str, dict], comparison: dict, output_dir: Path
) -> None:
    dog = stats["dog_to_dog"]["summary"]
    cat = stats["dog_to_cat"]["summary"]
    lines = [
        "# Cached image steering summary",
        "",
        "These are the original probability-maximizing proposal-selection runs at "
        "$\\rho=0.1$, not the Metropolis-Hastings experiment. The comparison uses "
        "100 strict dog starting images, four paired stochastic seeds per image, "
        "and 50 sequential U-turn steps.",
        "",
        "At every step, the algorithm draws 64 U-turn proposals, keeps the proposal "
        "with the highest target probability if it strictly improves the current "
        "state, and retries up to four batches (256 proposals). If no proposal "
        "improves the target, the image is kept unchanged for that step. Dog-to-dog "
        "targets the highest-scoring non-source dog class in the starting image. "
        "Dog-to-cat targets the summed probability of ImageNet cat classes 281--285 "
        "and compares it with summed dog probability over classes 151--268.",
        "",
        "| target | images | runs | target prob. start -> final | target > source | "
        "target prob. >= 0.5 | median crossing step | 50% target-side step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("dog_to_dog", "dog_to_cat"):
        row = stats[mode]["summary"]
        lines.append(
            f"| {LABELS[mode]} | {row['images']} | {row['runs']} | "
            f"{row['initial_target_probability']:.3f} -> "
            f"{row['final_target_probability']:.3f} "
            f"[{row['final_target_probability_ci_low']:.3f}, "
            f"{row['final_target_probability_ci_high']:.3f}] | "
            f"{row['crossing_rate']:.3f} "
            f"[{row['crossing_rate_ci_low']:.3f}, "
            f"{row['crossing_rate_ci_high']:.3f}] | "
            f"{row['target_probability_ge_0_5']:.3f} "
            f"[{row['target_probability_ge_0_5_ci_low']:.3f}, "
            f"{row['target_probability_ge_0_5_ci_high']:.3f}] | "
            f"{row['median_first_crossing_successes']:.1f} | "
            f"{row['majority_target_side_step']} |"
        )
    lines += [
        "",
        "## Direct difficulty comparison",
        "",
        "| quantity | dog to dog | dog to cat | paired contrast |",
        "|---|---:|---:|---:|",
        (
            "| Ever reaches target side | "
            f"{dog['crossing_rate']:.3f} | {cat['crossing_rate']:.3f} | "
            f"+{comparison['ever_crossing_rate_dog_minus_cat']:.3f} "
            f"[{comparison['ever_crossing_rate_dog_minus_cat_ci_low']:.3f}, "
            f"{comparison['ever_crossing_rate_dog_minus_cat_ci_high']:.3f}] "
            "for dog-to-dog |"
        ),
        (
            "| On target side at step 50 | "
            f"{dog['final_target_side_rate']:.3f} | "
            f"{cat['final_target_side_rate']:.3f} | "
            f"+{comparison['final_target_side_rate_dog_minus_cat']:.3f} "
            f"[{comparison['final_target_side_rate_dog_minus_cat_ci_low']:.3f}, "
            f"{comparison['final_target_side_rate_dog_minus_cat_ci_high']:.3f}] "
            "for dog-to-dog |"
        ),
        (
            "| Restricted mean crossing step (never = 51) | "
            f"{dog['restricted_mean_first_crossing_step']:.3f} | "
            f"{cat['restricted_mean_first_crossing_step']:.3f} | "
            f"+{comparison['restricted_crossing_steps_cat_minus_dog']:.3f} "
            f"[{comparison['restricted_crossing_steps_cat_minus_dog_ci_low']:.3f}, "
            f"{comparison['restricted_crossing_steps_cat_minus_dog_ci_high']:.3f}] "
            "steps for dog-to-cat |"
        ),
        (
            "| Proposals per recorded step | "
            f"{dog['mean_proposals_per_recorded_step']:.1f} | "
            f"{cat['mean_proposals_per_recorded_step']:.1f} | "
            f"{comparison['proposals_per_step_dog_minus_cat']:+.1f} "
            f"[{comparison['proposals_per_step_dog_minus_cat_ci_low']:.1f}, "
            f"{comparison['proposals_per_step_dog_minus_cat_ci_high']:.1f}] "
            "dog-to-dog minus dog-to-cat |"
        ),
        "",
        "The matched curves first satisfy mean target probability >= mean source "
        f"probability at step {dog['mean_curve_crossing_step']} for dog-to-dog and "
        f"step {cat['mean_curve_crossing_step']} for dog-to-cat. Half of the "
        f"population is currently on the target side by step "
        f"{dog['majority_target_side_step']} versus "
        f"{cat['majority_target_side_step']}. Thus dog-to-cat is both slower and "
        "less reliable despite the same proposal budget.",
        "",
        "Confidence intervals are 500-resample bootstraps over starting images after "
        "averaging the four stochastic runs for each image. All "
        f"{comparison['seed_matched_runs']} run pairs use identical random seeds.",
    ]
    (output_dir / "cached_image_steering_summary.md").write_text(
        "\n".join(lines) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/work/pcsl/Noam/sequential_diffusion"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    active = read_active_images(args.root / "metadata/dog_image_list_strict_100.txt")
    inputs = {
        "dog_to_cat": args.root / "results/steering_meta_v2_multi/steering_summary.csv",
        "dog_to_dog": args.root
        / "results/steering_dog2dog_v1_multi/steering_summary.csv",
    }
    stats = {}
    json_payload = {}
    paired_seeds = None
    for mode, path in inputs.items():
        frame = clean_summary(path, active, mode, preferred_seeds=paired_seeds)
        if mode == "dog_to_cat":
            paired_seeds = {
                (row.image_name, int(row.repeat_index)): int(row.seed)
                for row in frame.itertuples()
            }
        trajectories = trajectory_statistics(frame)
        summary = summarize_mode(frame, trajectories)
        stats[mode] = {"frame": frame, "trajectories": trajectories, "summary": summary}
        json_payload[mode] = summary

    comparison = comparison_summary(stats)
    json_payload["comparison"] = comparison
    (args.output_dir / "cached_image_steering_summary.json").write_text(
        json.dumps(json_payload, indent=2) + "\n"
    )
    write_trajectory_summary(stats, args.output_dir)
    write_difficulty_csv(stats, comparison, args.output_dir)
    write_markdown(stats, comparison, args.output_dir)
    plot_summary(stats, args.output_dir)


if __name__ == "__main__":
    main()
