#!/usr/bin/env python3
"""Generate duplicate rebuttal figures on normalized start-to-current displacement."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_no_reference_iqa import IQA_SPECS
from plot_standalone_image_rebuttal_figures import (
    QUALITY_METRICS,
    plot_quality_vs_net_displacement,
    style,
)


SEED = 44
RESAMPLES = 500


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def summarize_distance(
    frame: pd.DataFrame,
    keys: tuple[str, ...],
    seed: int,
) -> pd.DataFrame:
    rows = []
    for group_index, (key, group) in enumerate(
        frame.groupby(list(keys), sort=True)
    ):
        if not isinstance(key, tuple):
            key = (key,)
        image_means = (
            group.groupby("image_id", sort=False)["clip_net_distance_corrected"]
            .mean()
            .to_numpy(dtype=float)
        )
        if len(image_means) == 0:
            raise RuntimeError(f"Empty distance group for {key}")
        rng = np.random.default_rng(seed + group_index)
        indices = rng.integers(
            0,
            len(image_means),
            size=(RESAMPLES, len(image_means)),
        )
        draws = image_means[indices].mean(axis=1)
        row = dict(zip(keys, key))
        row.update(
            {
                "samples": len(group),
                "images": len(image_means),
                "clip_net_distance_corrected": float(image_means.mean()),
                "clip_net_distance_corrected_ci_low": float(
                    np.quantile(draws, 0.025)
                ),
                "clip_net_distance_corrected_ci_high": float(
                    np.quantile(draws, 0.975)
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_corrected_distance(
    manifest_path: Path,
    distance_path: Path,
) -> pd.DataFrame:
    frame = read_csv(manifest_path)
    distances = np.load(distance_path)
    if len(frame) != len(distances):
        raise RuntimeError(
            f"{manifest_path}: {len(frame)} rows but {len(distances)} distances"
        )
    if not np.isfinite(distances).all():
        raise RuntimeError(f"{distance_path} contains non-finite values")
    frame["clip_net_distance_corrected"] = distances
    return frame


def normalize_distance(
    frame: pd.DataFrame,
    direct_reference: float,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["normalized_clip_net_displacement"] = (
        frame["clip_net_distance_corrected"] / direct_reference
    )
    frame["normalized_clip_net_displacement_ci_low"] = (
        frame["clip_net_distance_corrected_ci_low"] / direct_reference
    )
    frame["normalized_clip_net_displacement_ci_high"] = (
        frame["clip_net_distance_corrected_ci_high"] / direct_reference
    )
    return frame


def merge_metrics(
    metrics: pd.DataFrame,
    distances: pd.DataFrame,
    keys: tuple[str, ...],
) -> pd.DataFrame:
    distance_columns = [
        *keys,
        "clip_net_distance_corrected",
        "clip_net_distance_corrected_ci_low",
        "clip_net_distance_corrected_ci_high",
        "normalized_clip_net_displacement",
        "normalized_clip_net_displacement_ci_low",
        "normalized_clip_net_displacement_ci_high",
    ]
    return metrics.merge(
        distances[distance_columns],
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
        "# Normalized net-displacement figure index",
        "",
        "These figures duplicate the accumulated-change metric family using "
        "bounded start-to-current displacement. Existing cumulative-path "
        "figures remain unchanged.",
        "",
        "| Family | Quantity | PNG | PDF |",
        "|---|---|---|---|",
    ]
    for family, entries in (
        ("reference/distribution metrics", quality_entries),
        ("no-reference IQA", iqa_entries),
    ):
        folder = (
            "reference_metrics"
            if family == "reference/distribution metrics"
            else "no_reference_iqa"
        )
        for title, stem in entries:
            lines.append(
                f"| {family} | {title} | "
                f"[PNG]({folder}/{stem}.png) | "
                f"[PDF]({folder}/{stem}.pdf) |"
            )
    (output_dir / "net_displacement_figure_index.md").write_text(
        "\n".join(lines) + "\n"
    )


def write_report(
    output_dir: Path,
    direct_summary: pd.DataFrame,
    single: pd.DataFrame,
    sequential: pd.DataFrame,
) -> None:
    reference = float(direct_summary["direct_clip_net_distance"].iloc[0])
    low = float(direct_summary["direct_clip_net_distance_ci_low"].iloc[0])
    high = float(direct_summary["direct_clip_net_distance_ci_high"].iloc[0])
    single_terminal = single.sort_values("rho").iloc[-1]
    terminal_step = int(sequential["step"].max())
    terminal = sequential[sequential["step"].eq(terminal_step)].sort_values("rho")
    lines = [
        "# Normalized net CLIP-patch displacement",
        "",
        "## Definition",
        "",
        r"Let `e(x)` be the flattened, L2-normalized final CLIP ViT-B/32 "
        r"patch-token representation. The raw source-to-current displacement is",
        "",
        r"`D_n = max(0, 1 - e(x^(0))^T e(x^(n)))`.",
        "",
        "The plotting coordinate is",
        "",
        r"`D_tilde_n = D_n / D_infinity`,",
        "",
        "where `D_infinity` is estimated by crossing all 20 exact starting "
        "images with all 200 independent direct-diffusion samples.",
        "",
        f"- `D_infinity = {reference:.6f} [{low:.6f}, {high:.6f}]`",
        "- `D_tilde = 0` means unchanged from the exact start.",
        "- `D_tilde = 1` is the mean displacement of independent direct "
        "sampling from total noise.",
        "",
        "CLIP input images were represented in `[0,1]` before applying the "
        "official CLIP channel mean and standard deviation. This corrects the "
        "legacy extractor's inconsistent `[-1,1]` input range; old figures "
        "were not overwritten.",
        "",
        "## Endpoint checks",
        "",
        "| condition | raw net displacement | normalized displacement |",
        "|---|---:|---:|",
        (
            f"| single, rho={single_terminal['rho']:.3f} | "
            f"{single_terminal['clip_net_distance_corrected']:.6f} | "
            f"{single_terminal['normalized_clip_net_displacement']:.3f} |"
        ),
    ]
    for row in terminal.itertuples(index=False):
        lines.append(
            f"| sequential, rho={row.rho:.3f}, step={terminal_step} | "
            f"{row.clip_net_distance_corrected:.6f} | "
            f"{row.normalized_clip_net_displacement:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `single_uturn_shared20_net_displacement_summary.csv`",
            "- `sequential_uturn_shared20_net_displacement_summary.csv`",
            "- `direct_clip_reference_summary.csv`",
            "- `net_displacement_figure_index.md`",
        ]
    )
    (output_dir / "net_displacement_results.md").write_text(
        "\n".join(lines) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-manifest", type=Path, required=True)
    parser.add_argument("--sequential-manifest", type=Path, required=True)
    parser.add_argument("--shared-start-manifest", type=Path, required=True)
    parser.add_argument("--single-distances", type=Path, required=True)
    parser.add_argument("--sequential-distances", type=Path, required=True)
    parser.add_argument("--direct-reference-summary", type=Path, required=True)
    parser.add_argument("--quality-results-dir", type=Path, required=True)
    parser.add_argument("--iqa-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    direct_summary = read_csv(args.direct_reference_summary)
    reference = float(direct_summary["direct_clip_net_distance"].iloc[0])
    direct_reference = (
        reference,
        float(direct_summary["direct_clip_net_distance_ci_low"].iloc[0]),
        float(direct_summary["direct_clip_net_distance_ci_high"].iloc[0]),
    )
    if not 0 < reference < 2:
        raise RuntimeError(f"Invalid direct-reference displacement {reference}")

    shared_ids = set(
        read_csv(args.shared_start_manifest)["image_id"].astype(str)
    )
    single_rows = add_corrected_distance(
        args.single_manifest, args.single_distances
    )
    single_rows = single_rows[
        single_rows["image_id"].astype(str).isin(shared_ids)
    ].copy()
    sequential_rows = add_corrected_distance(
        args.sequential_manifest, args.sequential_distances
    )
    sequential_rows = sequential_rows[
        sequential_rows["image_id"].astype(str).isin(shared_ids)
    ].copy()
    if single_rows["image_id"].nunique() != 20:
        raise RuntimeError("Single-U-turn corrected cohort is not the shared 20")
    if sequential_rows["image_id"].nunique() != 20:
        raise RuntimeError(
            "Sequential-U-turn corrected cohort is not the shared 20"
        )

    single_x = summarize_distance(
        single_rows,
        ("noise_step", "rho"),
        SEED,
    )
    sequential_x = summarize_distance(
        sequential_rows,
        ("noise_step", "rho", "step"),
        SEED + 100,
    )
    single_x = normalize_distance(single_x, reference)
    sequential_x = normalize_distance(sequential_x, reference)

    quality_single = merge_metrics(
        read_csv(
            args.quality_results_dir
            / "single_uturn_shared20_quality_summary.csv"
        ),
        single_x,
        ("noise_step", "rho"),
    )
    quality_sequential = merge_metrics(
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
    iqa_single = merge_metrics(
        read_csv(
            args.iqa_results_dir / "single_uturn_shared20_iqa_summary.csv"
        ),
        single_x,
        ("noise_step", "rho"),
    )
    iqa_sequential = merge_metrics(
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
        args.output_dir / "single_uturn_shared20_net_displacement_summary.csv",
        index=False,
    )
    quality_sequential.to_csv(
        args.output_dir
        / "sequential_uturn_shared20_net_displacement_summary.csv",
        index=False,
    )
    direct_summary.to_csv(
        args.output_dir / "direct_clip_reference_summary.csv",
        index=False,
    )

    style()
    quality_output = args.output_dir / "reference_metrics"
    quality_entries = []
    for spec in QUALITY_METRICS:
        stem = plot_quality_vs_net_displacement(
            quality_single,
            quality_sequential,
            quality_direct,
            spec,
            quality_output,
            direct_reference,
        )
        if stem is not None:
            quality_entries.append((spec.title, stem))

    iqa_output = args.output_dir / "no_reference_iqa"
    iqa_entries = []
    for spec in IQA_SPECS:
        stem = plot_quality_vs_net_displacement(
            iqa_single,
            iqa_sequential,
            iqa_direct,
            spec,
            iqa_output,
            direct_reference,
            original=iqa_original,
        )
        if stem is not None:
            iqa_entries.append((spec.title, stem))

    write_index(args.output_dir, quality_entries, iqa_entries)
    write_report(args.output_dir, direct_summary, single_x, sequential_x)
    print(
        f"Wrote {len(quality_entries)} reference/distribution and "
        f"{len(iqa_entries)} IQA figures to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
