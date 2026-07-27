#!/usr/bin/env python3
"""Plot image rebuttal metrics on corrected step-to-step CLIP path length."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_no_reference_iqa import IQA_SPECS
from plot_net_displacement_image_rebuttal_figures import (
    add_corrected_distance,
    read_csv,
    summarize_distance,
)
from plot_standalone_image_rebuttal_figures import (
    QUALITY_METRICS,
    plot_quality_vs_change,
    style,
)


SEED = 44


def add_path(frame: pd.DataFrame, path_file: Path) -> pd.DataFrame:
    values = np.load(path_file)
    if len(frame) != len(values):
        raise RuntimeError(
            f"{path_file}: {len(values)} values for {len(frame)} rows"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path_file} contains non-finite values")
    frame = frame.copy()
    frame["clip_cumulative_path_corrected"] = values
    return frame


def summarize_path(
    frame: pd.DataFrame,
    keys: tuple[str, ...],
    seed: int,
) -> pd.DataFrame:
    frame = frame.rename(
        columns={
            "clip_cumulative_path_corrected": "clip_net_distance_corrected"
        }
    )
    summary = summarize_distance(frame, keys, seed)
    return summary.rename(
        columns={
            "clip_net_distance_corrected": "clip_cumulative_path",
            "clip_net_distance_corrected_ci_low": (
                "clip_cumulative_path_ci_low"
            ),
            "clip_net_distance_corrected_ci_high": (
                "clip_cumulative_path_ci_high"
            ),
        }
    )


def merge_path_metrics(
    metrics: pd.DataFrame,
    paths: pd.DataFrame,
    keys: tuple[str, ...],
) -> pd.DataFrame:
    path_columns = [
        *keys,
        "clip_cumulative_path",
        "clip_cumulative_path_ci_low",
        "clip_cumulative_path_ci_high",
    ]
    return metrics.drop(
        columns=[
            column
            for column in (
                "clip_cumulative_path",
                "clip_cumulative_path_ci_low",
                "clip_cumulative_path_ci_high",
            )
            if column in metrics
        ]
    ).merge(
        paths[path_columns],
        on=list(keys),
        how="inner",
        validate="1:1",
    )


def write_index(
    output_dir: Path,
    quality_entries: list[tuple[str, str]],
    iqa_entries: list[tuple[str, str]],
) -> None:
    lines = [
        "# Corrected accumulated-path figure index",
        "",
        "These figures use the total step-to-step trajectory length",
        "",
        r"`S_n = sum_j max(0, 1 - e(x^(j-1))^T e(x^(j)))`,",
        "",
        "with corrected `[0,1]` CLIP preprocessing. This quantity counts "
        "reversed motion and is not bounded by independent-sample distance.",
        "",
        "| Family | Quantity | PNG | PDF |",
        "|---|---|---|---|",
    ]
    for family, folder, entries in (
        ("reference/distribution metrics", "reference_metrics", quality_entries),
        ("no-reference IQA", "no_reference_iqa", iqa_entries),
    ):
        for title, stem in entries:
            lines.append(
                f"| {family} | {title} | "
                f"[PNG]({folder}/{stem}.png) | "
                f"[PDF]({folder}/{stem}.pdf) |"
            )
    (output_dir / "corrected_accumulated_path_figure_index.md").write_text(
        "\n".join(lines) + "\n"
    )


def write_report(
    output_dir: Path,
    single: pd.DataFrame,
    sequential: pd.DataFrame,
) -> None:
    terminal_step = int(sequential["step"].max())
    terminal = sequential[
        sequential["step"].eq(terminal_step)
    ].sort_values("rho")
    lines = [
        "# Corrected accumulated CLIP-patch path length",
        "",
        "## Definition",
        "",
        r"`S_n = sum_{j=1}^n max(0, 1 - e(x^(j-1))^T e(x^(j)))`.",
        "",
        "Every step-to-step movement is counted, including changes that later "
        "reverse. The quantity is therefore unbounded under repeated U-turns.",
        "",
        "CLIP inputs use `[0,1]` before official channel normalization.",
        "",
        "## Endpoint checks",
        "",
        "| condition | accumulated path |",
        "|---|---:|",
        (
            f"| single, rho={single.sort_values('rho').iloc[-1]['rho']:.3f} | "
            f"{single.sort_values('rho').iloc[-1]['clip_cumulative_path']:.6f} |"
        ),
    ]
    for row in terminal.itertuples(index=False):
        lines.append(
            f"| sequential, rho={row.rho:.3f}, step={terminal_step} | "
            f"{row.clip_cumulative_path:.6f} |"
        )
    (output_dir / "corrected_accumulated_path_results.md").write_text(
        "\n".join(lines) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-manifest", type=Path, required=True)
    parser.add_argument("--sequential-manifest", type=Path, required=True)
    parser.add_argument("--shared-start-manifest", type=Path, required=True)
    parser.add_argument("--single-distances", type=Path, required=True)
    parser.add_argument("--sequential-path", type=Path, required=True)
    parser.add_argument("--quality-results-dir", type=Path, required=True)
    parser.add_argument("--iqa-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    shared_ids = set(
        read_csv(args.shared_start_manifest)["image_id"].astype(str)
    )
    single_rows = add_corrected_distance(
        args.single_manifest, args.single_distances
    )
    single_rows = single_rows[
        single_rows["image_id"].astype(str).isin(shared_ids)
    ].copy()
    single_rows["clip_cumulative_path_corrected"] = single_rows[
        "clip_net_distance_corrected"
    ]

    sequential_rows = add_path(
        read_csv(args.sequential_manifest), args.sequential_path
    )
    sequential_rows = sequential_rows[
        sequential_rows["image_id"].astype(str).isin(shared_ids)
    ].copy()
    if single_rows["image_id"].nunique() != 20:
        raise RuntimeError("Single-U-turn path cohort is not the shared 20")
    if sequential_rows["image_id"].nunique() != 20:
        raise RuntimeError("Sequential-U-turn path cohort is not the shared 20")

    single_x = summarize_path(
        single_rows,
        ("noise_step", "rho"),
        SEED,
    )
    sequential_x = summarize_path(
        sequential_rows,
        ("noise_step", "rho", "step"),
        SEED + 100,
    )

    quality_single = merge_path_metrics(
        read_csv(
            args.quality_results_dir
            / "single_uturn_shared20_quality_summary.csv"
        ),
        single_x,
        ("noise_step", "rho"),
    )
    quality_sequential = merge_path_metrics(
        read_csv(
            args.quality_results_dir
            / "sequential_uturn_shared20_quality_summary.csv"
        ),
        sequential_x,
        ("noise_step", "rho", "step"),
    )
    quality_direct = read_csv(
        args.quality_results_dir / "direct_diffusion_quality_summary.csv"
    )
    iqa_single = merge_path_metrics(
        read_csv(
            args.iqa_results_dir / "single_uturn_shared20_iqa_summary.csv"
        ),
        single_x,
        ("noise_step", "rho"),
    )
    iqa_sequential = merge_path_metrics(
        read_csv(args.iqa_results_dir / "sequential_uturn_iqa_summary.csv"),
        sequential_x,
        ("noise_step", "rho", "step"),
    )
    iqa_direct = read_csv(
        args.iqa_results_dir / "direct_diffusion_iqa_summary.csv"
    )
    iqa_original = read_csv(
        args.iqa_results_dir / "original_start_iqa_summary.csv"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality_single.to_csv(
        args.output_dir
        / "single_uturn_shared20_corrected_path_summary.csv",
        index=False,
    )
    quality_sequential.to_csv(
        args.output_dir
        / "sequential_uturn_shared20_corrected_path_summary.csv",
        index=False,
    )

    style()
    quality_output = args.output_dir / "reference_metrics"
    quality_entries = []
    for spec in QUALITY_METRICS:
        stem = plot_quality_vs_change(
            quality_single,
            quality_sequential,
            quality_direct,
            spec,
            quality_output,
        )
        if stem is not None:
            quality_entries.append((spec.title, stem))

    iqa_output = args.output_dir / "no_reference_iqa"
    iqa_entries = []
    for spec in IQA_SPECS:
        stem = plot_quality_vs_change(
            iqa_single,
            iqa_sequential,
            iqa_direct,
            spec,
            iqa_output,
            original=iqa_original,
        )
        if stem is not None:
            iqa_entries.append((spec.title, stem))

    write_index(args.output_dir, quality_entries, iqa_entries)
    write_report(args.output_dir, single_x, sequential_x)
    print(
        f"Wrote {len(quality_entries)} reference/distribution and "
        f"{len(iqa_entries)} IQA corrected-path figures to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
