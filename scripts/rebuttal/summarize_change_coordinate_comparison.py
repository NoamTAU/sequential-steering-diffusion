#!/usr/bin/env python3
"""Summarize single-versus-sequential results at matched change coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "musiq": ("MUSIQ", "higher"),
    "topiq_nr": ("TOPIQ-NR", "higher"),
    "manifold_precision_k3": ("manifold precision", "higher"),
    "manifold_density_k3": ("manifold density", "higher"),
    "manifold_nearest_distance": ("nearest-real distance", "lower"),
    "convnext_class_perplexity": ("class perplexity", "lower"),
    "fid_matched_200": ("global ImageNet FID-200", "lower"),
    "kid": ("global ImageNet KID", "lower"),
    "source_fid_200": ("source-cohort FID-200", "lower"),
    "convnext_start_class_retained": ("starting-class retention", "higher"),
}


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def attach_iqa(
    frame: pd.DataFrame, iqa: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    return frame.merge(iqa[[*keys, "musiq", "topiq_nr"]], on=keys)


def compare(
    coordinate: str,
    single: pd.DataFrame,
    sequential: pd.DataFrame,
    x_column: str,
) -> list[dict[str, object]]:
    single = single.sort_values(x_column)
    sequential = sequential[
        sequential["step"].gt(1)
        & sequential[x_column].between(
            single[x_column].min(), single[x_column].max()
        )
    ].copy()
    rows: list[dict[str, object]] = []
    for column, (label, direction) in METRICS.items():
        matched_single = np.interp(
            sequential[x_column],
            single[x_column],
            single[column],
        )
        difference = sequential[column].to_numpy() - matched_single
        sequential_better = (
            difference > 0 if direction == "higher" else difference < 0
        )
        rows.append(
            {
                "coordinate": coordinate,
                "metric": label,
                "better_direction": direction,
                "overlap_min": single[x_column].min(),
                "overlap_max": single[x_column].max(),
                "sequential_points": len(sequential),
                "mean_sequential_minus_single": difference.mean(),
                "median_sequential_minus_single": np.median(difference),
                "sequential_better_count": sequential_better.sum(),
                "sequential_better_fraction": sequential_better.mean(),
            }
        )
    return rows


def write_report(frame: pd.DataFrame, output: Path) -> None:
    path = frame[frame["coordinate"].eq("accumulated path")]
    net = frame[frame["coordinate"].eq("normalized net displacement")]

    def value(table: pd.DataFrame, metric: str, column: str) -> float:
        return float(table.loc[table["metric"].eq(metric), column].iloc[0])

    lines = [
        "# Matched image-change coordinate comparison",
        "",
        "## Coordinates",
        "",
        r"- Accumulated path: "
        r"`S_n = sum_j max(0, 1 - e(x^(j-1))^T e(x^(j)))`. "
        "It counts every step, including motion that later reverses.",
        r"- Net displacement: "
        r"`D_tilde_n = max(0, 1 - e(x^0)^T e(x^n)) / D_infinity`. "
        "It measures only the final image's displacement from its start; "
        "`D_tilde=1` is independent direct-diffusion displacement.",
        "",
        "For each sequential aggregate with `step > 1` inside the single-U-turn "
        "coordinate range, the single-U-turn metric is linearly interpolated "
        "at exactly the same coordinate. Differences below are descriptive; "
        "the aggregate points are correlated and are not treated as independent "
        "replicates.",
        "",
        "## Accumulated-path result",
        "",
        (
            f"Only `{int(path['sequential_points'].iloc[0])}` genuinely "
            "sequential aggregate points overlap the single-U-turn path range "
            f"`[{path['overlap_min'].iloc[0]:.3f}, "
            f"{path['overlap_max'].iloc[0]:.3f}]`. In this early overlap, "
            "sequential U-turns have higher predicted perceptual quality and "
            "better per-image fidelity at a fixed amount of total motion:"
        ),
        "",
        (
            f"- MUSIQ difference: "
            f"`{value(path, 'MUSIQ', 'mean_sequential_minus_single'):+.3f}`."
        ),
        (
            f"- TOPIQ-NR difference: "
            f"`{value(path, 'TOPIQ-NR', 'mean_sequential_minus_single'):+.3f}`."
        ),
        (
            f"- Manifold-precision difference: "
            f"`{value(path, 'manifold precision', 'mean_sequential_minus_single'):+.3f}`."
        ),
        (
            f"- Nearest-real-distance difference: "
            f"`{value(path, 'nearest-real distance', 'mean_sequential_minus_single'):+.3f}`."
        ),
        "",
        "Global FID moves in the opposite direction in this small overlap, while "
        "KID improves. These set-level metrics remain coverage- and "
        "sample-size-sensitive and are not used as the sole image-quality claim.",
        "",
        "## Net-displacement result",
        "",
        (
            f"`{int(net['sequential_points'].iloc[0])}` sequential aggregates "
            "overlap the single-U-turn endpoint range. FID, manifold precision, "
            "source drift, and starting-class retention are close on average at "
            "matched endpoint displacement, but repeated denoising incurs extra "
            "no-reference perceptual degradation:"
        ),
        "",
        (
            f"- MUSIQ difference: "
            f"`{value(net, 'MUSIQ', 'mean_sequential_minus_single'):+.3f}`; "
            f"sequential is better at only "
            f"`{100 * value(net, 'MUSIQ', 'sequential_better_fraction'):.1f}%` "
            "of matched points."
        ),
        (
            f"- TOPIQ-NR difference: "
            f"`{value(net, 'TOPIQ-NR', 'mean_sequential_minus_single'):+.3f}`; "
            f"sequential is better at only "
            f"`{100 * value(net, 'TOPIQ-NR', 'sequential_better_fraction'):.1f}%` "
            "of matched points."
        ),
        (
            f"- Manifold-precision difference: "
            f"`{value(net, 'manifold precision', 'mean_sequential_minus_single'):+.3f}`."
        ),
        (
            f"- Global FID difference: "
            f"`{value(net, 'global ImageNet FID-200', 'mean_sequential_minus_single'):+.3f}`."
        ),
        "",
        "The endpoint result is therefore not that the two procedures are "
        "identical. They can reach similar endpoint semantics and distribution "
        "statistics, while many small denoising operations add perceptual damage "
        "that one large U-turn does not incur.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-results-dir", type=Path, required=True)
    parser.add_argument("--path-results-dir", type=Path, required=True)
    parser.add_argument("--net-results-dir", type=Path, required=True)
    parser.add_argument("--iqa-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    single_iqa = read(
        args.iqa_results_dir / "single_uturn_shared20_iqa_summary.csv"
    )
    sequential_iqa = read(
        args.iqa_results_dir / "sequential_uturn_iqa_summary.csv"
    )

    single_path = attach_iqa(
        read(
            args.path_results_dir
            / "single_uturn_shared20_corrected_path_summary.csv"
        ),
        single_iqa,
        ["noise_step", "rho"],
    )
    sequential_path = attach_iqa(
        read(
            args.path_results_dir
            / "sequential_uturn_shared20_corrected_path_summary.csv"
        ),
        sequential_iqa,
        ["noise_step", "rho", "step"],
    )
    single_net = attach_iqa(
        read(
            args.net_results_dir
            / "single_uturn_shared20_net_displacement_summary.csv"
        ),
        single_iqa,
        ["noise_step", "rho"],
    )
    sequential_net = attach_iqa(
        read(
            args.net_results_dir
            / "sequential_uturn_shared20_net_displacement_summary.csv"
        ),
        sequential_iqa,
        ["noise_step", "rho", "step"],
    )

    rows = [
        *compare(
            "accumulated path",
            single_path,
            sequential_path,
            "clip_cumulative_path",
        ),
        *compare(
            "normalized net displacement",
            single_net,
            sequential_net,
            "normalized_clip_net_displacement",
        ),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(rows)
    output.to_csv(
        args.output_dir / "matched_change_coordinate_comparison.csv",
        index=False,
    )
    write_report(
        output,
        args.output_dir / "matched_change_coordinate_comparison.md",
    )


if __name__ == "__main__":
    main()
