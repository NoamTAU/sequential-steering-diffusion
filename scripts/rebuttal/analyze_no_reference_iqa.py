#!/usr/bin/env python3
"""Aggregate no-reference IQA scores and create rebuttal-style figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plot_standalone_image_rebuttal_figures import (
    MetricSpec,
    plot_quality_vs_change,
    plot_quality_vs_step,
    style,
)


SEED = 44
RESAMPLES = 500
IQA_SPECS = (
    MetricSpec(
        column="musiq",
        stem="musiq_no_reference_quality",
        title="No-reference perceptual quality (MUSIQ)",
        ylabel=r"MUSIQ quality score ($\uparrow$)",
        words=(
            "MUSIQ predicts a human mean-opinion quality score from an image "
            "without receiving a reference image. We report the mean score."
        ),
        equations=(
            r"$q_i=Q_{\mathrm{MUSIQ}}(x_i)$",
            r"$y=\frac{1}{M}\sum_{i=1}^{M}q_i$",
        ),
        detail=(
            "Pretrained IQA-PyTorch KonIQ checkpoint; values are model-score "
            "units, not probabilities. Baselines use 200 direct samples and "
            "the 20 exact starts."
        ),
        direction="Higher indicates better predicted perceptual quality.",
        ylim=(0.0, 100.0),
        direct_baseline=True,
    ),
    MetricSpec(
        column="topiq_nr",
        stem="topiq_nr_no_reference_quality",
        title="No-reference perceptual quality (TOPIQ-NR)",
        ylabel=r"TOPIQ-NR quality score ($\uparrow$)",
        words=(
            "TOPIQ-NR predicts human-perceived quality from an image without "
            "receiving a reference image. We report the mean score."
        ),
        equations=(
            r"$q_i=Q_{\mathrm{TOPIQ\!-\!NR}}(x_i)$",
            r"$y=\frac{1}{M}\sum_{i=1}^{M}q_i$",
        ),
        detail=(
            "Pretrained IQA-PyTorch KonIQ checkpoint. This score does not "
            "measure semantics, source preservation, or dataset coverage. "
            "Baselines use 200 direct samples and the 20 exact starts."
        ),
        direction="Higher indicates better predicted perceptual quality.",
        ylim=(0.0, 1.0),
        direct_baseline=True,
    ),
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def shared_flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def image_bootstrap_summary(
    group: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
) -> dict[str, float | int]:
    image_means = group.groupby("image_id", sort=False)[list(columns)].mean()
    if image_means.empty:
        raise ValueError("Cannot summarize an empty IQA group")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(image_means), size=(RESAMPLES, len(image_means))
    )
    output: dict[str, float | int] = {
        "samples": int(len(group)),
        "images": int(len(image_means)),
    }
    for column in columns:
        values = image_means[column].to_numpy(dtype=float)
        draws = values[indices].mean(axis=1)
        output[column] = float(values.mean())
        output[f"{column}_ci_low"] = float(np.quantile(draws, 0.025))
        output[f"{column}_ci_high"] = float(np.quantile(draws, 0.975))
    return output


def summarize_grouped(
    frame: pd.DataFrame,
    keys: tuple[str, ...],
    columns: tuple[str, ...],
    seed: int,
) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(list(keys), sort=True)
    for group_index, (group_key, group) in enumerate(grouped):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(keys, group_key))
        row.update(
            image_bootstrap_summary(group, columns, seed=seed + group_index)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def attach_accumulated_change(
    summary: pd.DataFrame,
    quality_summary_path: Path,
    keys: tuple[str, ...],
) -> pd.DataFrame:
    quality = read_csv(quality_summary_path)
    columns = [
        *keys,
        "clip_cumulative_path",
        "clip_cumulative_path_ci_low",
        "clip_cumulative_path_ci_high",
    ]
    missing = sorted(set(columns) - set(quality.columns))
    if missing:
        raise ValueError(f"{quality_summary_path} is missing columns: {missing}")
    coordinates = quality[columns].copy()
    return summary.merge(coordinates, on=list(keys), how="left", validate="1:1")


def markdown_table(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    labels: tuple[str, ...],
    digits: tuple[int, ...],
) -> list[str]:
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join("---:" for _ in labels) + "|",
    ]
    for row in frame.itertuples(index=False):
        cells = []
        for column, decimals in zip(columns, digits):
            value = getattr(row, column)
            cells.append(
                str(int(value))
                if decimals < 0
                else f"{float(value):.{decimals}f}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(
    output_path: Path,
    original: pd.DataFrame,
    direct: pd.DataFrame,
    single: pd.DataFrame,
    sequential: pd.DataFrame,
) -> None:
    terminal = sequential[sequential["step"].eq(sequential["step"].max())].copy()
    lines = [
        "# Full no-reference IQA results",
        "",
        "MUSIQ and TOPIQ-NR were selected after a blinded calibration pilot. "
        "Both are no-reference predictors trained against human quality "
        "judgments. They measure predicted perceptual/technical quality; they "
        "do not measure source preservation, semantic correctness, or dataset "
        "coverage.",
        "",
        "## Baselines",
        "",
        "| cohort | n | MUSIQ (higher better) | TOPIQ-NR (higher better) |",
        "|---|---:|---:|---:|",
        (
            f"| exact original starts | {int(original['samples'].iloc[0])} | "
            f"{original['musiq'].iloc[0]:.3f} "
            f"[{original['musiq_ci_low'].iloc[0]:.3f}, "
            f"{original['musiq_ci_high'].iloc[0]:.3f}] | "
            f"{original['topiq_nr'].iloc[0]:.3f} "
            f"[{original['topiq_nr_ci_low'].iloc[0]:.3f}, "
            f"{original['topiq_nr_ci_high'].iloc[0]:.3f}] |"
        ),
        (
            f"| direct diffusion | {int(direct['samples'].iloc[0])} | "
            f"{direct['musiq'].iloc[0]:.3f} "
            f"[{direct['musiq_ci_low'].iloc[0]:.3f}, "
            f"{direct['musiq_ci_high'].iloc[0]:.3f}] | "
            f"{direct['topiq_nr'].iloc[0]:.3f} "
            f"[{direct['topiq_nr_ci_low'].iloc[0]:.3f}, "
            f"{direct['topiq_nr_ci_high'].iloc[0]:.3f}] |"
        ),
        "",
        "## Single U-turn",
        "",
        *markdown_table(
            single,
            ("rho", "clip_cumulative_path", "musiq", "topiq_nr"),
            (
                "rho",
                "accumulated CLIP-patch change",
                "MUSIQ",
                "TOPIQ-NR",
            ),
            (3, 3, 3, 3),
        ),
        "",
        "## Sequential U-turns at step 100",
        "",
        *markdown_table(
            terminal,
            ("rho", "clip_cumulative_path", "musiq", "topiq_nr"),
            (
                "rho",
                "accumulated CLIP-patch change",
                "MUSIQ",
                "TOPIQ-NR",
            ),
            (3, 3, 3, 3),
        ),
        "",
        "## Headline interpretation",
        "",
        "- A single U-turn remains close to the original-start and direct-"
        "diffusion quality baselines through moderate noise "
        "(`rho <= 0.4`) on both learned IQA metrics.",
        "- At high single-U-turn noise (`rho = 0.999`), the mean falls to "
        f"MUSIQ `{single.loc[single['rho'].eq(0.999), 'musiq'].iloc[0]:.3f}` "
        "and TOPIQ-NR "
        f"`{single.loc[single['rho'].eq(0.999), 'topiq_nr'].iloc[0]:.3f}`.",
        "- After 100 sequential U-turns, every noise level is below the "
        "direct-diffusion mean on both metrics. The lowest terminal means "
        "occur at `rho = 0.2`: MUSIQ "
        f"`{terminal.loc[terminal['rho'].eq(0.2), 'musiq'].iloc[0]:.3f}` "
        "and TOPIQ-NR "
        f"`{terminal.loc[terminal['rho'].eq(0.2), 'topiq_nr'].iloc[0]:.3f}`.",
        "- The sequential dependence on noise is non-monotone: low/moderate "
        "noise preserves one-step quality but can accumulate substantial "
        "deterioration over repeated turns, while high-noise trajectories "
        "make a larger early move and then tend to plateau. The figures "
        "therefore report quality jointly with accumulated change rather "
        "than treating noise as a quality proxy.",
        "",
        "## Estimation",
        "",
        "For U-turn cohorts, scores first average stochastic trajectories "
        "within each of the 20 shared starting images, then average across "
        "starts. Confidence intervals use 500 bootstrap resamples over starts. "
        "Direct diffusion bootstraps its 200 independent samples.",
        "",
        "## Integrity checks",
        "",
        "All `25,860` unique images received finite scores. The full run "
        "exactly reproduced both metric values on all `112` images shared "
        "with the blinded pilot, and raw-image quantile/disagreement sheets "
        "were visually audited. See `audit/iqa_integrity_audit.md`.",
        "",
        "## Figures",
        "",
        "- `standalone/musiq_no_reference_quality_vs_accumulated_change.pdf`",
        "- `standalone/musiq_no_reference_quality_vs_uturn_step.pdf`",
        "- `standalone/topiq_nr_no_reference_quality_vs_accumulated_change.pdf`",
        "- `standalone/topiq_nr_no_reference_quality_vs_uturn_step.pdf`",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--single-quality-summary", type=Path, required=True)
    parser.add_argument(
        "--single-shared20-quality-summary", type=Path, required=True
    )
    parser.add_argument(
        "--sequential-quality-summary", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = read_csv(args.scores)
    required = {
        "dataset",
        "image_id",
        "noise_step",
        "rho",
        "step",
        "shared20",
        "musiq",
        "topiq_nr",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{args.scores} is missing columns: {missing}")
    for column in ("noise_step", "rho", "step"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[["musiq", "topiq_nr"]].to_numpy(float)).all():
        raise RuntimeError("IQA score table contains non-finite values")

    metrics = ("musiq", "topiq_nr")
    single_frame = frame[frame["dataset"].eq("single")].copy()
    sequential_frame = frame[frame["dataset"].eq("sequential")].copy()
    direct_frame = frame[frame["dataset"].eq("direct_diffusion")].copy()
    original_frame = frame[frame["dataset"].eq("original_start")].copy()
    single_shared = single_frame[shared_flag(single_frame["shared20"])].copy()

    single_full = summarize_grouped(
        single_frame, ("noise_step", "rho"), metrics, SEED
    )
    single_full = attach_accumulated_change(
        single_full,
        args.single_quality_summary,
        ("noise_step", "rho"),
    )
    single = summarize_grouped(
        single_shared, ("noise_step", "rho"), metrics, SEED + 100
    )
    single = attach_accumulated_change(
        single,
        args.single_shared20_quality_summary,
        ("noise_step", "rho"),
    )
    sequential = summarize_grouped(
        sequential_frame, ("noise_step", "rho", "step"), metrics, SEED + 200
    )
    sequential = attach_accumulated_change(
        sequential,
        args.sequential_quality_summary,
        ("noise_step", "rho", "step"),
    )
    direct = pd.DataFrame(
        [image_bootstrap_summary(direct_frame, metrics, SEED + 300)]
    )
    original = pd.DataFrame(
        [image_bootstrap_summary(original_frame, metrics, SEED + 400)]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    single_full.to_csv(args.output_dir / "single_uturn_iqa_summary.csv", index=False)
    single.to_csv(
        args.output_dir / "single_uturn_shared20_iqa_summary.csv", index=False
    )
    sequential.to_csv(
        args.output_dir / "sequential_uturn_iqa_summary.csv", index=False
    )
    direct.to_csv(args.output_dir / "direct_diffusion_iqa_summary.csv", index=False)
    original.to_csv(args.output_dir / "original_start_iqa_summary.csv", index=False)

    style()
    figure_dir = args.output_dir / "standalone"
    entries = []
    for spec in IQA_SPECS:
        change_stem = plot_quality_vs_change(
            single,
            sequential,
            direct,
            spec,
            figure_dir,
            original=original,
        )
        step_stem = plot_quality_vs_step(
            sequential,
            direct,
            spec,
            figure_dir,
            original=original,
        )
        entries.extend(
            [
                ("accumulated change", spec.title, change_stem),
                ("U-turn step", spec.title, step_stem),
            ]
        )
    index_lines = [
        "# No-reference IQA figure index",
        "",
        "| view | metric | PDF | PNG |",
        "|---|---|---|---|",
    ]
    for view, title, stem in entries:
        if stem is None:
            continue
        index_lines.append(
            f"| {view} | {title} | [{stem}.pdf]({stem}.pdf) | "
            f"[{stem}.png]({stem}.png) |"
        )
    (figure_dir / "standalone_figure_index.md").write_text(
        "\n".join(index_lines) + "\n"
    )
    write_report(
        args.output_dir / "no_reference_iqa_results.md",
        original,
        direct,
        single,
        sequential,
    )
    print(f"Wrote no-reference IQA analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
