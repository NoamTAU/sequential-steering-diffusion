#!/usr/bin/env python3
"""Create one-panel image rebuttal figures with exact metric definitions."""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


PAPER_CMAP = LinearSegmentedColormap.from_list(
    "paper_noise", ["#D96B6B", "#E8A27E", "#91AFC9", "#3B6FB6"]
)
METHOD_GRAY = "#5C5C5C"
DIRECT_BLACK = "#202020"
ORIGINAL_GRAY = "#8A8A8A"
BLUE = "#3B6FB6"
RED = "#D96B6B"


@dataclass(frozen=True)
class MetricSpec:
    column: str
    stem: str
    title: str
    ylabel: str
    words: str
    equations: tuple[str, ...]
    detail: str
    direction: str
    scale: float = 1.0
    ylim: tuple[float, float] | None = None
    direct_baseline: bool = False


QUALITY_METRICS = (
    MetricSpec(
        column="manifold_precision_k3",
        stem="manifold_precision",
        title="Real-manifold membership",
        ylabel=r"manifold precision, $k=3$ ($\uparrow$)",
        words=(
            "Fraction of generated images that lie inside at least one local "
            "neighborhood of the real ImageNet validation features."
        ),
        equations=(
            r"$R_j=\|r_j-r_{j,(3)}\|_2$",
            r"$P_i=\mathbf{1}\{\exists j:\|g_i-r_j\|_2\leq R_j\}$",
            r"$y=\frac{1}{M}\sum_{i=1}^{M}P_i$",
        ),
        detail=(
            "Here r_j and g_i are 2048-dimensional Inception features, and "
            "r_{j,(3)} is the third-nearest real neighbor. Reference: all "
            "50,000 ImageNet validation images."
        ),
        direction="Higher is better.",
        ylim=(0.0, 1.035),
        direct_baseline=True,
    ),
    MetricSpec(
        column="manifold_density_k3",
        stem="manifold_density",
        title="Local real-manifold support",
        ylabel=r"manifold density, $k=3$ ($\uparrow$)",
        words=(
            "Average number of real-data neighborhoods that contain a "
            "generated image, normalized by k. Density can exceed one."
        ),
        equations=(
            r"$R_j=\|r_j-r_{j,(3)}\|_2$",
            r"$D_i=\frac{1}{3}\sum_j\mathbf{1}\{\|g_i-r_j\|_2\leq R_j\}$",
            r"$y=\frac{1}{M}\sum_{i=1}^{M}D_i$",
        ),
        detail=(
            "The same Inception features and 50,000-image reference set are "
            "used as for manifold precision."
        ),
        direction="Higher is better.",
        direct_baseline=True,
    ),
    MetricSpec(
        column="manifold_nearest_distance",
        stem="nearest_real_inception_distance",
        title="Distance to the real manifold",
        ylabel=r"nearest-real feature distance ($\downarrow$)",
        words=(
            "Mean Euclidean distance from each generated image to its nearest "
            "real ImageNet validation image in Inception feature space."
        ),
        equations=(
            r"$d_i=\min_j\|g_i-r_j\|_2$",
            r"$y=\frac{1}{M}\sum_{i=1}^{M}d_i$",
        ),
        detail=(
            "Features are 2048-dimensional Inception features; the reference "
            "contains all 50,000 ImageNet validation images."
        ),
        direction="Lower is better.",
        direct_baseline=True,
    ),
    MetricSpec(
        column="convnext_class_perplexity",
        stem="convnext_class_perplexity",
        title="Classifier uncertainty",
        ylabel=r"ConvNeXt class perplexity ($\downarrow$)",
        words=(
            "Effective number of ImageNet classes supported by the ConvNeXt "
            "posterior for one image, averaged over images. This is a "
            "classifier-uncertainty score, not model likelihood."
        ),
        equations=(
            r"$p_{ic}=\operatorname{softmax}(z(x_i))_c$",
            r"$H_i=-\sum_{c=1}^{1000}p_{ic}\log p_{ic}$",
            r"$y=\frac{1}{M}\sum_{i=1}^{M}\exp(H_i)$",
        ),
        detail=(
            "The implementation averages per-image exp(entropy); it does not "
            "exponentiate the mean entropy."
        ),
        direction="Lower means a more class-confident image.",
        direct_baseline=True,
    ),
    MetricSpec(
        column="convnext_max_probability",
        stem="convnext_max_class_probability",
        title="Per-image semantic confidence",
        ylabel=r"maximum class probability ($\uparrow$)",
        words=(
            "Mean probability assigned by ConvNeXt to its most likely "
            "ImageNet class for each generated image."
        ),
        equations=(
            r"$p_{ic}=\operatorname{softmax}(z(x_i))_c$",
            r"$y=\frac{1}{M}\sum_{i=1}^{M}\max_c p_{ic}$",
        ),
        detail="ConvNeXt-Base is used as the fixed ImageNet classifier.",
        direction="Higher means greater classifier confidence.",
        ylim=(0.0, 1.0),
        direct_baseline=True,
    ),
    MetricSpec(
        column="convnext_feature_cosine_distance",
        stem="paired_convnext_feature_distance",
        title="Perceptual-semantic change",
        ylabel=r"paired ConvNeXt feature distance ($\downarrow$)",
        words=(
            "Cosine distance between each U-turn output and its exact starting "
            "image in the 1024-dimensional penultimate ConvNeXt feature space."
        ),
        equations=(
            r"$\widehat h(x)=h(x)/\|h(x)\|_2$",
            r"$d_i=1-\widehat h(x_i^{(0)})^\top\widehat h(x_i)$",
            r"$y=\frac{1}{M}\sum_{i=1}^{M}\max(0,d_i)$",
        ),
        detail=(
            "This is a paired preservation metric. It does not measure whether "
            "the output is realistic relative to ImageNet as a whole."
        ),
        direction="Lower means closer to the exact starting image.",
        ylim=(0.0, 1.0),
    ),
    MetricSpec(
        column="convnext_start_class_probability",
        stem="starting_class_probability",
        title="Original semantic confidence",
        ylabel=r"starting-class probability ($\uparrow$)",
        words=(
            "Probability that the output assigns to the ConvNeXt top-1 class "
            "of its exact starting image."
        ),
        equations=(
            r"$c_i^{(0)}=\arg\max_c z_c(x_i^{(0)})$",
            r"$y=\frac{1}{M}\sum_{i=1}^{M}p_{i,c_i^{(0)}}$",
        ),
        detail=(
            "The starting label is classifier-defined separately for every "
            "source image; no ground-truth class label is injected."
        ),
        direction="Higher means stronger preservation of source semantics.",
        ylim=(0.0, 1.0),
    ),
    MetricSpec(
        column="convnext_start_class_retained",
        stem="starting_class_retention",
        title="Top-1 class retention",
        ylabel=r"starting top-1 class retained ($\uparrow$)",
        words=(
            "Fraction of outputs whose ConvNeXt top-1 prediction is unchanged "
            "from the exact starting image."
        ),
        equations=(
            r"$c_i^{(0)}=\arg\max_c z_c(x_i^{(0)})$",
            r"$R_i=\mathbf{1}\{\arg\max_c z_c(x_i)=c_i^{(0)}\}$",
            r"$y=\frac{1}{M}\sum_{i=1}^{M}R_i$",
        ),
        detail=(
            "This is a paired identity-preservation rate, not a measure of "
            "unconditional image quality."
        ),
        direction="Higher means more source-class preservation.",
        ylim=(0.0, 1.0),
    ),
    MetricSpec(
        column="kid",
        stem="global_imagenet_kid",
        title="Global ImageNet kernel distance",
        ylabel=r"KID to ImageNet, $\times 10^3$ ($\downarrow$)",
        words=(
            "Unbiased squared maximum mean discrepancy between generated and "
            "real Inception features with the standard cubic polynomial kernel."
        ),
        equations=(
            r"$k(a,b)=(a^\top b/2048+1)^3$",
            r"$\widehat{\mathrm{KID}}=\overline{k}_{G,G}^{\,i\ne j}"
            r"+\overline{k}_{R,R}^{\,i\ne j}-2\overline{k}_{G,R}$",
            r"$y=10^3\,\operatorname{mean}_{s=1}^{100}"
            r"\widehat{\mathrm{KID}}_s$",
        ),
        detail=(
            "Each estimate uses 100 random paired subsets of 100 features. "
            "The interval is the 2.5-97.5 percentile range across subsets. "
            "Reference: 50,000 ImageNet validation images."
        ),
        direction=(
            "Lower is better, but this distribution-level score rewards "
            "coverage and can be negative near zero because it is unbiased."
        ),
        scale=1000.0,
        direct_baseline=True,
    ),
    MetricSpec(
        column="fid_matched_200",
        stem="global_imagenet_fid200",
        title="Global ImageNet Fréchet distance",
        ylabel=r"FID-200 to ImageNet ($\downarrow$)",
        words=(
            "Fréchet distance between the empirical Gaussian fitted to 200 "
            "balanced generated features and the Gaussian fitted to all real "
            "ImageNet validation features."
        ),
        equations=(
            r"$d_\mu=\|\mu_G-\mu_R\|_2^2$",
            r"$d_\Sigma=\operatorname{Tr}"
            r"(\Sigma_G+\Sigma_R-2(\Sigma_G\Sigma_R)^{1/2})$",
            r"$y=d_\mu+d_\Sigma$",
        ),
        detail=(
            "The generated set has 200 rows balanced across starting images. "
            "The 2048-dimensional sample covariance is rank-deficient, so the "
            "code evaluates the same empirical FID through its 200x200 Gram "
            "eigenvalues. No FID confidence interval is plotted."
        ),
        direction=(
            "Lower is better as a distribution-fit score; it is coverage- and "
            "sample-size-sensitive, not a pure per-image quality score."
        ),
        direct_baseline=True,
    ),
    MetricSpec(
        column="source_kid",
        stem="source_cohort_kid",
        title="Drift from the starting cohort",
        ylabel=r"source-set KID, $\times 10^3$ ($\downarrow$)",
        words=(
            "The same unbiased KID statistic, but the reference distribution "
            "is the exact 20-image source cohort rather than all ImageNet."
        ),
        equations=(
            r"$k(a,b)=(a^\top b/2048+1)^3$",
            r"$\widehat{\mathrm{KID}}_{\rm src}="
            r"\overline{k}_{G,G}^{\,i\ne j}"
            r"+\overline{k}_{S,S}^{\,i\ne j}-2\overline{k}_{G,S}$",
            r"$y=10^3\,\operatorname{mean}_{s=1}^{100}"
            r"\widehat{\mathrm{KID}}_{{\rm src},s}$",
        ),
        detail=(
            "The 20 exact starts are each repeated ten times to form a balanced "
            "200-row reference. KID uses 100 random subsets of 100."
        ),
        direction="Lower means less distributional drift from the source cohort.",
        scale=1000.0,
    ),
    MetricSpec(
        column="source_fid_200",
        stem="source_cohort_fid200",
        title="Fréchet drift from the starting cohort",
        ylabel=r"source-set FID-200 ($\downarrow$)",
        words=(
            "Empirical FID between 200 balanced generated features and a "
            "200-row reference made from the exact 20 starting images."
        ),
        equations=(
            r"$d_\mu=\|\mu_G-\mu_S\|_2^2$",
            r"$d_\Sigma=\operatorname{Tr}"
            r"(\Sigma_G+\Sigma_S-2(\Sigma_G\Sigma_S)^{1/2})$",
            r"$y=d_\mu+d_\Sigma$",
        ),
        detail=(
            "Each exact source feature is repeated ten times. This measures "
            "cohort drift, not unconditional image quality. No FID confidence "
            "interval is plotted."
        ),
        direction="Lower means less drift from the starting distribution.",
        ylim=(0.0, 350.0),
    ),
)


STEP_METRIC_STEMS = {
    "kid": "global_imagenet_kid",
    "fid_matched_200": "global_imagenet_fid200",
    "convnext_max_probability": "convnext_max_class_probability",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.titleweight": "normal",
            "legend.framealpha": 0.82,
            "legend.edgecolor": "#D0D0D0",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def wrapped(text: str, width: int = 55) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def definition_panel(
    axis: plt.Axes,
    words: str,
    equations: Iterable[str],
    detail: str,
    direction: str,
    x_definition: bool = False,
    net_displacement_reference: tuple[float, float, float] | None = None,
) -> None:
    axis.axis("off")
    axis.axvline(0.0, color="#D8D8D8", lw=0.8, ymin=0.02, ymax=0.98)
    x = 0.06
    y = 0.97

    def place(
        text: str,
        *,
        fontsize: float,
        weight: str = "normal",
        color: str = "#202020",
        gap: float = 0.018,
        linespacing: float = 1.28,
    ) -> None:
        nonlocal y
        artist = axis.text(
            x,
            y,
            text,
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            va="top",
            linespacing=linespacing,
            transform=axis.transAxes,
        )
        axis.figure.canvas.draw()
        renderer = axis.figure.canvas.get_renderer()
        bbox = artist.get_window_extent(renderer=renderer)
        axis_bbox = axis.get_window_extent(renderer=renderer)
        y -= bbox.height / axis_bbox.height + gap

    place("Definition", fontsize=11, weight="semibold", gap=0.026)
    place(wrapped(words, 57), fontsize=8.9, gap=0.025)
    for equation in equations:
        place(equation, fontsize=9.4, gap=0.018)
    if x_definition:
        y -= 0.005
        place(
            "Accumulated-change axis",
            fontsize=9.2,
            weight="semibold",
            gap=0.022,
        )
        place(
            r"$\widehat e_j="
            r"\operatorname{vec}(E_{\rm CLIP}^{\rm patch}(x^{(j)}))/"
            r"\|\operatorname{vec}(E_{\rm CLIP}^{\rm patch}(x^{(j)}))\|_2$",
            fontsize=8.2,
            gap=0.018,
        )
        place(
            r"$S_n=\sum_{j=1}^{n}"
            r"\max\{0,1-\widehat e_{j-1}^{\top}\widehat e_j\}$",
            fontsize=9.1,
            gap=0.022,
        )
        place(
            wrapped(
                "The embedding uses all CLIP ViT-B/32 patch tokens. For a single "
                "U-turn, n=1; sequential paths sum every realized step.",
                57,
            ),
            fontsize=8.1,
            gap=0.025,
        )
    if net_displacement_reference is not None:
        reference, reference_low, reference_high = net_displacement_reference
        y -= 0.005
        place(
            "Net-displacement axis",
            fontsize=9.2,
            weight="semibold",
            gap=0.022,
        )
        place(
            r"$\widehat e(x)="
            r"\operatorname{vec}(E_{\rm CLIP}^{\rm patch}(x))/"
            r"\|\operatorname{vec}(E_{\rm CLIP}^{\rm patch}(x))\|_2$",
            fontsize=8.0,
            gap=0.016,
        )
        place(
            r"$D_n=\max\{0,1-\widehat e(x^{(0)})^\top"
            r"\widehat e(x^{(n)})\},\qquad"
            r"\widetilde D_n=D_n/\widehat D_\infty$",
            fontsize=8.7,
            gap=0.017,
        )
        place(
            (
                rf"$\widehat D_\infty={reference:.3f}$ "
                rf"$[{reference_low:.3f},{reference_high:.3f}]$"
            ),
            fontsize=8.7,
            gap=0.018,
        )
        place(
            wrapped(
                "D-infinity is the mean distance over all 20 exact starts "
                "crossed with 200 independent direct-diffusion samples. Thus "
                "zero is unchanged and one is independent-sample displacement.",
                57,
            ),
            fontsize=7.8,
            gap=0.024,
        )
    place(wrapped(detail, 57), fontsize=7.9, color="#505050", gap=0.024)
    place(
        wrapped(direction, 57),
        fontsize=8.7,
        weight="semibold",
        gap=0.0,
    )


def figure_axes() -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    fig = plt.figure(figsize=(12.8, 5.25), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.62, 0.04, 1.28))
    plot_axis = fig.add_subplot(grid[0, 0])
    colorbar_axis = fig.add_subplot(grid[0, 1])
    definition_axis = fig.add_subplot(grid[0, 2])
    return fig, plot_axis, colorbar_axis, definition_axis


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    aliases: tuple[str, ...] = (),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for output_stem in (stem, *aliases):
        for suffix in ("pdf", "png"):
            fig.savefig(
                output_dir / f"{output_stem}.{suffix}",
                dpi=300,
                bbox_inches="tight",
            )
    plt.close(fig)


def add_rho_colorbar(
    fig: plt.Figure, axis: plt.Axes, minimum: float, maximum: float
) -> Normalize:
    norm = Normalize(minimum, maximum)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=PAPER_CMAP)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=axis)
    colorbar.set_label(r"noise fraction $\rho$", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    return norm


def finite_column(frame: pd.DataFrame, column: str) -> bool:
    return column in frame and np.isfinite(frame[column].to_numpy(dtype=float)).any()


def informative_ylim(
    axis: plt.Axes,
    values: Iterable[np.ndarray | pd.Series | list[float]],
    bounds: tuple[float, float] | None = None,
) -> None:
    finite = []
    for value in values:
        array = np.asarray(value, dtype=float).reshape(-1)
        finite.extend(array[np.isfinite(array)].tolist())
    if not finite:
        return
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    span = maximum - minimum
    if span <= 1e-12:
        span = max(abs(maximum), 1.0) * 0.1
    pad = 0.075 * span
    low = minimum - pad
    high = maximum + pad
    if bounds is not None:
        bound_low, bound_high = bounds
        full_span = bound_high - bound_low
        if minimum - bound_low < 0.08 * full_span:
            low = bound_low
        else:
            low = max(bound_low, low)
        if bound_high - maximum < 0.08 * full_span:
            high = bound_high + 0.02 * full_span
        else:
            high = min(bound_high, high)
    axis.set_ylim(low, high)


def plot_quality_vs_change(
    single: pd.DataFrame,
    sequential: pd.DataFrame,
    direct: pd.DataFrame,
    spec: MetricSpec,
    output_dir: Path,
    original: pd.DataFrame | None = None,
) -> str | None:
    if not finite_column(single, spec.column) or not finite_column(
        sequential, spec.column
    ):
        return None
    fig, axis, colorbar_axis, definition_axis = figure_axes()
    rho_min = min(single["rho"].min(), sequential["rho"].min())
    rho_max = max(single["rho"].max(), sequential["rho"].max())
    norm = add_rho_colorbar(fig, colorbar_axis, rho_min, rho_max)
    ci_low = f"{spec.column}_ci_low"
    ci_high = f"{spec.column}_ci_high"
    y_extent: list[np.ndarray | pd.Series | list[float]] = []

    for rho, group in sequential.groupby("rho", sort=True):
        group = group.sort_values("clip_cumulative_path")
        x = group["clip_cumulative_path"].to_numpy(dtype=float)
        y = group[spec.column].to_numpy(dtype=float) * spec.scale
        y_extent.append(y)
        color = PAPER_CMAP(norm(rho))
        if finite_column(group, ci_low) and finite_column(group, ci_high):
            y_extent.extend(
                [
                    group[ci_low].to_numpy(dtype=float) * spec.scale,
                    group[ci_high].to_numpy(dtype=float) * spec.scale,
                ]
            )
            axis.fill_between(
                x,
                group[ci_low].to_numpy(dtype=float) * spec.scale,
                group[ci_high].to_numpy(dtype=float) * spec.scale,
                color=color,
                alpha=0.09,
                linewidth=0,
            )
        axis.plot(x, y, color=color, marker="o", ms=3.0, lw=1.45, alpha=0.92)

    single = single.sort_values("clip_cumulative_path")
    y_extent.append(single[spec.column].to_numpy(dtype=float) * spec.scale)
    if finite_column(single, ci_low) and finite_column(single, ci_high):
        y_extent.extend(
            [
                single[ci_low].to_numpy(dtype=float) * spec.scale,
                single[ci_high].to_numpy(dtype=float) * spec.scale,
            ]
        )
    axis.plot(
        single["clip_cumulative_path"],
        single[spec.column] * spec.scale,
        color=METHOD_GRAY,
        ls="--",
        lw=1.25,
        zorder=3,
    )
    for row in single.itertuples(index=False):
        value = float(getattr(row, spec.column)) * spec.scale
        kwargs = {}
        if ci_low in single and ci_high in single:
            low = float(getattr(row, ci_low)) * spec.scale
            high = float(getattr(row, ci_high)) * spec.scale
            if np.isfinite(low) and np.isfinite(high):
                kwargs["yerr"] = [[max(0.0, value - low)], [max(0.0, high - value)]]
        axis.errorbar(
            float(row.clip_cumulative_path),
            value,
            fmt="D",
            ms=5.2,
            color=PAPER_CMAP(norm(float(row.rho))),
            markeredgecolor="#333333",
            markeredgewidth=0.5,
            ecolor=PAPER_CMAP(norm(float(row.rho))),
            elinewidth=0.7,
            capsize=1.5,
            zorder=4,
            **kwargs,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_GRAY,
            ls="--",
            marker="D",
            markerfacecolor="white",
            label="single U-turn",
        ),
        Line2D(
            [0],
            [0],
            color=METHOD_GRAY,
            ls="-",
            marker="o",
            markerfacecolor="white",
            label="sequential U-turns",
        ),
    ]
    if spec.direct_baseline and finite_column(direct, spec.column):
        baseline = float(direct[spec.column].iloc[0]) * spec.scale
        y_extent.append([baseline])
        baseline_low = f"{spec.column}_ci_low"
        baseline_high = f"{spec.column}_ci_high"
        if finite_column(direct, baseline_low) and finite_column(
            direct, baseline_high
        ):
            y_extent.extend(
                [
                    direct[baseline_low].to_numpy(dtype=float) * spec.scale,
                    direct[baseline_high].to_numpy(dtype=float) * spec.scale,
                ]
            )
            axis.axhspan(
                float(direct[baseline_low].iloc[0]) * spec.scale,
                float(direct[baseline_high].iloc[0]) * spec.scale,
                color=DIRECT_BLACK,
                alpha=0.055,
                linewidth=0,
            )
        axis.axhline(baseline, color=DIRECT_BLACK, ls=":", lw=1.25)
        handles.append(
            Line2D(
                [0],
                [0],
                color=DIRECT_BLACK,
                ls=":",
                label="direct diffusion",
            )
        )
    if original is not None and finite_column(original, spec.column):
        baseline = float(original[spec.column].iloc[0]) * spec.scale
        y_extent.append([baseline])
        baseline_low = f"{spec.column}_ci_low"
        baseline_high = f"{spec.column}_ci_high"
        if finite_column(original, baseline_low) and finite_column(
            original, baseline_high
        ):
            y_extent.extend(
                [
                    original[baseline_low].to_numpy(dtype=float) * spec.scale,
                    original[baseline_high].to_numpy(dtype=float) * spec.scale,
                ]
            )
            axis.axhspan(
                float(original[baseline_low].iloc[0]) * spec.scale,
                float(original[baseline_high].iloc[0]) * spec.scale,
                color=ORIGINAL_GRAY,
                alpha=0.04,
                linewidth=0,
            )
        axis.axhline(
            baseline,
            color=ORIGINAL_GRAY,
            ls="-.",
            lw=1.05,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=ORIGINAL_GRAY,
                ls="-.",
                label="original starts",
            )
        )

    axis.set_xscale("log")
    axis.set_xlabel("accumulated CLIP-patch change")
    axis.set_ylabel(spec.ylabel)
    axis.set_title(spec.title)
    informative_ylim(axis, y_extent, bounds=spec.ylim)
    axis.legend(handles=handles, fontsize=7.4, loc="best")
    distribution_metrics = {
        "kid",
        "fid_matched_200",
        "source_kid",
        "source_fid_200",
    }
    aggregation_note = (
        ""
        if spec.column in distribution_metrics
        else " Scalar per-image scores first average stochastic outputs within "
        "each of 20 starts, then average starts; their 95% intervals use 500 "
        "bootstrap resamples over starts."
    )
    definition_panel(
        definition_axis,
        spec.words,
        spec.equations,
        spec.detail + aggregation_note,
        spec.direction,
        x_definition=True,
    )
    stem = f"{spec.stem}_vs_accumulated_change"
    save_figure(fig, output_dir, stem)
    return stem


def plot_quality_vs_net_displacement(
    single: pd.DataFrame,
    sequential: pd.DataFrame,
    direct: pd.DataFrame,
    spec: MetricSpec,
    output_dir: Path,
    direct_reference: tuple[float, float, float],
    original: pd.DataFrame | None = None,
) -> str | None:
    x_column = "normalized_clip_net_displacement"
    x_ci_low = f"{x_column}_ci_low"
    x_ci_high = f"{x_column}_ci_high"
    if (
        not finite_column(single, spec.column)
        or not finite_column(sequential, spec.column)
        or not finite_column(single, x_column)
        or not finite_column(sequential, x_column)
    ):
        return None

    fig, axis, colorbar_axis, definition_axis = figure_axes()
    rho_min = min(single["rho"].min(), sequential["rho"].min())
    rho_max = max(single["rho"].max(), sequential["rho"].max())
    norm = add_rho_colorbar(fig, colorbar_axis, rho_min, rho_max)
    ci_low = f"{spec.column}_ci_low"
    ci_high = f"{spec.column}_ci_high"
    y_extent: list[np.ndarray | pd.Series | list[float]] = []
    x_extent: list[np.ndarray | pd.Series | list[float]] = [[0.0, 1.0]]

    for rho, group in sequential.groupby("rho", sort=True):
        group = group.sort_values("step")
        x = group[x_column].to_numpy(dtype=float)
        y = group[spec.column].to_numpy(dtype=float) * spec.scale
        x_extent.append(x)
        y_extent.append(y)
        color = PAPER_CMAP(norm(rho))
        if finite_column(group, ci_low) and finite_column(group, ci_high):
            y_extent.extend(
                [
                    group[ci_low].to_numpy(dtype=float) * spec.scale,
                    group[ci_high].to_numpy(dtype=float) * spec.scale,
                ]
            )
            axis.fill_between(
                x,
                group[ci_low].to_numpy(dtype=float) * spec.scale,
                group[ci_high].to_numpy(dtype=float) * spec.scale,
                color=color,
                alpha=0.09,
                linewidth=0,
            )
        axis.plot(
            x,
            y,
            color=color,
            marker="o",
            ms=3.0,
            lw=1.45,
            alpha=0.92,
        )

    single = single.sort_values(x_column)
    x_extent.append(single[x_column].to_numpy(dtype=float))
    y_extent.append(single[spec.column].to_numpy(dtype=float) * spec.scale)
    if finite_column(single, x_ci_low) and finite_column(single, x_ci_high):
        x_extent.extend(
            [
                single[x_ci_low].to_numpy(dtype=float),
                single[x_ci_high].to_numpy(dtype=float),
            ]
        )
    if finite_column(single, ci_low) and finite_column(single, ci_high):
        y_extent.extend(
            [
                single[ci_low].to_numpy(dtype=float) * spec.scale,
                single[ci_high].to_numpy(dtype=float) * spec.scale,
            ]
        )
    axis.plot(
        single[x_column],
        single[spec.column] * spec.scale,
        color=METHOD_GRAY,
        ls="--",
        lw=1.25,
        zorder=3,
    )
    for row in single.itertuples(index=False):
        x_value = float(getattr(row, x_column))
        y_value = float(getattr(row, spec.column)) * spec.scale
        kwargs: dict[str, list[list[float]]] = {}
        if ci_low in single and ci_high in single:
            low = float(getattr(row, ci_low)) * spec.scale
            high = float(getattr(row, ci_high)) * spec.scale
            if np.isfinite(low) and np.isfinite(high):
                kwargs["yerr"] = [
                    [max(0.0, y_value - low)],
                    [max(0.0, high - y_value)],
                ]
        if x_ci_low in single and x_ci_high in single:
            low = float(getattr(row, x_ci_low))
            high = float(getattr(row, x_ci_high))
            if np.isfinite(low) and np.isfinite(high):
                kwargs["xerr"] = [
                    [max(0.0, x_value - low)],
                    [max(0.0, high - x_value)],
                ]
        axis.errorbar(
            x_value,
            y_value,
            fmt="D",
            ms=5.2,
            color=PAPER_CMAP(norm(float(row.rho))),
            markeredgecolor="#333333",
            markeredgewidth=0.5,
            ecolor=PAPER_CMAP(norm(float(row.rho))),
            elinewidth=0.7,
            capsize=1.5,
            zorder=4,
            **kwargs,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_GRAY,
            ls="--",
            marker="D",
            markerfacecolor="white",
            label="single U-turn",
        ),
        Line2D(
            [0],
            [0],
            color=METHOD_GRAY,
            ls="-",
            marker="o",
            markerfacecolor="white",
            label="sequential U-turns",
        ),
    ]
    axis.axvline(1.0, color=DIRECT_BLACK, ls=":", lw=0.8, alpha=0.45)
    if spec.direct_baseline and finite_column(direct, spec.column):
        baseline = float(direct[spec.column].iloc[0]) * spec.scale
        y_extent.append([baseline])
        baseline_low = f"{spec.column}_ci_low"
        baseline_high = f"{spec.column}_ci_high"
        if finite_column(direct, baseline_low) and finite_column(
            direct, baseline_high
        ):
            low = float(direct[baseline_low].iloc[0]) * spec.scale
            high = float(direct[baseline_high].iloc[0]) * spec.scale
            y_extent.extend([[low], [high]])
            axis.axhspan(
                low,
                high,
                color=DIRECT_BLACK,
                alpha=0.055,
                linewidth=0,
            )
        axis.axhline(baseline, color=DIRECT_BLACK, ls=":", lw=1.25)
        axis.plot(
            [1.0],
            [baseline],
            marker="*",
            ms=8,
            color=DIRECT_BLACK,
            zorder=5,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=DIRECT_BLACK,
                ls=":",
                marker="*",
                label="direct diffusion ($x=1$)",
            )
        )
    else:
        handles.append(
            Line2D(
                [0],
                [0],
                color=DIRECT_BLACK,
                ls=":",
                label="independent-sample scale ($x=1$)",
            )
        )

    if original is not None and finite_column(original, spec.column):
        baseline = float(original[spec.column].iloc[0]) * spec.scale
        y_extent.append([baseline])
        baseline_low = f"{spec.column}_ci_low"
        baseline_high = f"{spec.column}_ci_high"
        if finite_column(original, baseline_low) and finite_column(
            original, baseline_high
        ):
            low = float(original[baseline_low].iloc[0]) * spec.scale
            high = float(original[baseline_high].iloc[0]) * spec.scale
            y_extent.extend([[low], [high]])
            axis.axhspan(
                low,
                high,
                color=ORIGINAL_GRAY,
                alpha=0.04,
                linewidth=0,
            )
        axis.axhline(baseline, color=ORIGINAL_GRAY, ls="-.", lw=1.05)
        axis.plot(
            [0.0],
            [baseline],
            marker="s",
            ms=5,
            color=ORIGINAL_GRAY,
            zorder=5,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=ORIGINAL_GRAY,
                ls="-.",
                marker="s",
                label="original starts ($x=0$)",
            )
        )

    finite_x = np.concatenate(
        [
            np.asarray(values, dtype=float).reshape(-1)
            for values in x_extent
        ]
    )
    finite_x = finite_x[np.isfinite(finite_x)]
    x_max = max(1.08, float(finite_x.max()) + 0.035)
    axis.set_xlim(-0.025, x_max)
    axis.set_xlabel(
        r"normalized net CLIP-patch displacement from start, "
        r"$\widetilde D_n$"
    )
    axis.set_ylabel(spec.ylabel)
    axis.set_title(spec.title)
    informative_ylim(axis, y_extent, bounds=spec.ylim)
    axis.legend(handles=handles, fontsize=7.1, loc="best")

    distribution_metrics = {
        "kid",
        "fid_matched_200",
        "source_kid",
        "source_fid_200",
    }
    aggregation_note = (
        ""
        if spec.column in distribution_metrics
        else " Scalar per-image scores first average stochastic outputs within "
        "each of 20 starts, then average starts; their 95% intervals use 500 "
        "bootstrap resamples over starts."
    )
    definition_panel(
        definition_axis,
        spec.words,
        spec.equations,
        spec.detail + aggregation_note,
        spec.direction,
        net_displacement_reference=direct_reference,
    )
    stem = f"{spec.stem}_vs_normalized_net_displacement"
    save_figure(fig, output_dir, stem)
    return stem


def plot_quality_vs_step(
    sequential: pd.DataFrame,
    direct: pd.DataFrame,
    spec: MetricSpec,
    output_dir: Path,
    original: pd.DataFrame | None = None,
) -> str | None:
    if not finite_column(sequential, spec.column):
        return None
    fig, axis, colorbar_axis, definition_axis = figure_axes()
    norm = add_rho_colorbar(
        fig, colorbar_axis, sequential["rho"].min(), sequential["rho"].max()
    )
    ci_low = f"{spec.column}_ci_low"
    ci_high = f"{spec.column}_ci_high"
    y_extent: list[np.ndarray | pd.Series | list[float]] = []
    for rho, group in sequential.groupby("rho", sort=True):
        group = group.sort_values("step")
        color = PAPER_CMAP(norm(rho))
        x = group["step"].to_numpy(dtype=float)
        y = group[spec.column].to_numpy(dtype=float) * spec.scale
        y_extent.append(y)
        if finite_column(group, ci_low) and finite_column(group, ci_high):
            y_extent.extend(
                [
                    group[ci_low].to_numpy(dtype=float) * spec.scale,
                    group[ci_high].to_numpy(dtype=float) * spec.scale,
                ]
            )
            axis.fill_between(
                x,
                group[ci_low].to_numpy(dtype=float) * spec.scale,
                group[ci_high].to_numpy(dtype=float) * spec.scale,
                color=color,
                alpha=0.09,
                linewidth=0,
            )
        axis.plot(x, y, color=color, marker="o", ms=3.0, lw=1.45)
    if spec.direct_baseline and finite_column(direct, spec.column):
        y_extent.append([float(direct[spec.column].iloc[0]) * spec.scale])
        axis.axhline(
            float(direct[spec.column].iloc[0]) * spec.scale,
            color=DIRECT_BLACK,
            ls=":",
            lw=1.25,
            label="direct diffusion",
        )
    if original is not None and finite_column(original, spec.column):
        baseline = float(original[spec.column].iloc[0]) * spec.scale
        y_extent.append([baseline])
        baseline_low = f"{spec.column}_ci_low"
        baseline_high = f"{spec.column}_ci_high"
        if finite_column(original, baseline_low) and finite_column(
            original, baseline_high
        ):
            low = float(original[baseline_low].iloc[0]) * spec.scale
            high = float(original[baseline_high].iloc[0]) * spec.scale
            y_extent.extend([[low], [high]])
            axis.axhspan(
                low,
                high,
                color=ORIGINAL_GRAY,
                alpha=0.04,
                linewidth=0,
            )
        axis.axhline(
            baseline,
            color=ORIGINAL_GRAY,
            ls="-.",
            lw=1.05,
            label="original starts",
        )
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=8)
    axis.set_xlabel("sequential U-turn step")
    axis.set_ylabel(spec.ylabel)
    axis.set_title(f"{spec.title} over repeated U-turns")
    informative_ylim(axis, y_extent, bounds=spec.ylim)
    distribution_metrics = {
        "kid",
        "fid_matched_200",
        "source_kid",
        "source_fid_200",
    }
    aggregation_note = (
        ""
        if spec.column in distribution_metrics
        else " Each curve fixes rho. Scalar per-image scores first average "
        "stochastic trajectories within each of 20 starts, then average "
        "starts; their 95% intervals use 500 bootstrap resamples over starts."
    )
    definition_panel(
        definition_axis,
        spec.words,
        spec.equations,
        spec.detail + aggregation_note,
        spec.direction,
    )
    stem = f"{spec.stem}_vs_uturn_step"
    save_figure(fig, output_dir, stem)
    return stem


def simple_axes() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig = plt.figure(figsize=(12.3, 5.0), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.48, 1.12))
    return fig, fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])


def plot_mh_summary_metric(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    words: str,
    equations: tuple[str, ...],
    direction: str,
    output_dir: Path,
    stem: str,
    ylim: tuple[float, float] | None = None,
) -> str:
    fig, axis, definition_axis = simple_axes()
    markers = {"dog_class": "o", "cat": "s"}
    labels = {"dog_class": "dog to dog", "cat": "dog to cat"}
    colors = {0.0: "#777777", 1.0: BLUE, 4.0: RED}
    y_extent: list[np.ndarray | pd.Series | list[float]] = []
    for mode in ("dog_class", "cat"):
        for energy_lambda in sorted(summary["energy_lambda"].unique()):
            group = summary[
                summary["target_mode"].eq(mode)
                & summary["energy_lambda"].eq(energy_lambda)
            ].sort_values("rho")
            if group.empty:
                continue
            label = rf"{labels[mode]}, $\lambda={energy_lambda:g}$"
            color = colors.get(float(energy_lambda), "#333333")
            y_extent.append(group[metric].to_numpy(dtype=float))
            axis.plot(
                group["rho"],
                group[metric],
                marker=markers[mode],
                color=color,
                lw=1.45,
                ms=4.5,
                label=label,
            )
            low_column = f"{metric}_ci_low"
            high_column = f"{metric}_ci_high"
            if finite_column(group, low_column) and finite_column(
                group, high_column
            ):
                y_extent.extend(
                    [
                        group[low_column].to_numpy(dtype=float),
                        group[high_column].to_numpy(dtype=float),
                    ]
                )
                axis.fill_between(
                    group["rho"],
                    group[low_column],
                    group[high_column],
                    color=color,
                    alpha=0.07,
                    linewidth=0,
                )
    axis.set_xlabel(r"noise fraction $\rho$")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    informative_ylim(axis, y_extent, bounds=ylim)
    axis.legend(fontsize=7.4, ncol=2)
    definition_panel(
        definition_axis,
        words,
        equations,
        (
            "Dog-to-dog uses a specific target dog class versus the source "
            "dog class. Dog-to-cat sums ConvNeXt probabilities over ImageNet "
            "cat classes 281-285 versus dog classes 151-268. Intervals are "
            "500-resample bootstraps over independent chains."
        ),
        direction,
    )
    save_figure(fig, output_dir, stem)
    return stem


def plot_mh_focused_endpoint(
    summary: pd.DataFrame, output_dir: Path
) -> str | None:
    focused = summary[
        summary["target_mode"].eq("dog_class")
        & np.isclose(summary["rho"], 0.4)
        & summary["energy_lambda"].isin([0.0, 4.0])
    ].sort_values("energy_lambda")
    if len(focused) != 2:
        return None
    fig, axis, definition_axis = simple_axes()
    values = focused["final_target_probability"].to_numpy(dtype=float)
    low = focused["final_target_probability_ci_low"].to_numpy(dtype=float)
    high = focused["final_target_probability_ci_high"].to_numpy(dtype=float)
    axis.bar([0, 1], values, color=["#777777", RED], width=0.58)
    axis.errorbar(
        [0, 1],
        values,
        yerr=[values - low, high - values],
        fmt="none",
        color="black",
        capsize=3,
        lw=1,
    )
    for index, value in enumerate(values):
        axis.text(index, high[index] + 0.012, f"{value:.3f}", ha="center", fontsize=9)
    axis.set_xticks([0, 1], [r"$H=0$", r"$H=-4\log p_{\rm target}$"])
    axis.set_ylabel(r"final target-class probability ($\uparrow$)")
    axis.set_title(r"Dog-to-dog MH-rule steering at $\rho=0.4$")
    axis.set_ylim(0, max(0.5, float(high.max()) * 1.18))
    definition_panel(
        definition_axis,
        (
            "Mean endpoint probability of the selected target dog class under "
            "the unsteered chain and the classifier-energy MH-rule chain."
        ),
        (
            r"$H_\lambda(x)=-\lambda\log(\max\{p_{\rm target}(x),10^{-8}\})$",
            r"$A(x\to x')=\min\{1,\exp[-H_\lambda(x')+H_\lambda(x)]\}$",
            r"$y=\frac{1}{K}\sum_{r=1}^{K}p_{\rm target}(x_T^{(r)})$",
        ),
        (
            "K=12 independent 50-step chains per condition. Error bars are "
            "500-resample bootstrap 95% intervals over chains. This applies "
            "the paper's energy-only MH acceptance rule to an approximate "
            "neural U-turn proposal; it is not a proof of exact stationarity."
        ),
        "Higher is better for steering success.",
    )
    stem = "mh_rule_focused_final_target_probability"
    save_figure(fig, output_dir, stem)
    return stem


def plot_mh_trajectory(
    trajectories: pd.DataFrame,
    output_dir: Path,
    mode: str,
    rho: float,
    stem_suffix: str,
) -> str | None:
    subset = trajectories[
        trajectories["target_mode"].eq(mode)
        & np.isclose(trajectories["rho"], rho)
    ]
    if subset.empty:
        return None
    fig, axis, definition_axis = simple_axes()
    colors = {0.0: "#777777", 1.0: BLUE, 4.0: RED}
    for energy_lambda, group in subset.groupby("energy_lambda", sort=True):
        group = group.sort_values("step")
        color = colors.get(float(energy_lambda), "#333333")
        axis.plot(
            group["step"],
            group["mean_target_probability"],
            color=color,
            lw=1.7,
            label=rf"$\lambda={energy_lambda:g}$",
        )
        axis.fill_between(
            group["step"],
            group["q025_target_probability"],
            group["q975_target_probability"],
            color=color,
            alpha=0.11,
            linewidth=0,
        )
    mode_label = "dog to dog" if mode == "dog_class" else "dog to cat"
    axis.set_xlabel("U-turn step")
    axis.set_ylabel(r"mean target probability ($\uparrow$)")
    axis.set_title(rf"MH-rule {mode_label} trajectory, $\rho={rho:g}$")
    informative_ylim(
        axis,
        [
            subset["q025_target_probability"],
            subset["q975_target_probability"],
        ],
        bounds=(0.0, 1.0),
    )
    axis.legend(fontsize=8)
    definition_panel(
        definition_axis,
        (
            "Mean classifier probability of the steering target at each "
            "accepted-chain state."
        ),
        (
            r"$p_{\rm target}(x_t)=\operatorname{softmax}(z(x_t))_{c_*}$"
            if mode == "dog_class"
            else r"$p_{\rm target}(x_t)=\sum_{c\in\mathcal{C}_{\rm cat}}"
            r"\operatorname{softmax}(z(x_t))_c$",
            r"$y_t=\frac{1}{K}\sum_{r=1}^{K}p_{\rm target}(x_t^{(r)})$",
        ),
        (
            "The shaded interval is the pointwise 2.5-97.5 percentile range "
            "across independent chains. The chain uses the classifier-energy "
            "Hamiltonian and the energy-only MH acceptance rule defined in "
            "the focused endpoint figure."
        ),
        "Higher is better for steering success.",
    )
    stem = f"mh_rule_target_probability_trajectory_{stem_suffix}"
    save_figure(fig, output_dir, stem)
    return stem


def plot_mh_focused_probability_crossing(
    trajectories: pd.DataFrame, output_dir: Path
) -> str | None:
    required = {
        "mean_target_probability",
        "mean_source_probability",
    }
    if not required.issubset(trajectories.columns):
        return None
    focused = trajectories[
        trajectories["target_mode"].eq("dog_class")
        & np.isclose(trajectories["rho"], 0.4)
        & trajectories["energy_lambda"].isin([0.0, 4.0])
    ]
    if focused.empty:
        return None

    fig, axis, definition_axis = simple_axes()
    colors = {0.0: "#777777", 4.0: RED}
    y_extent: list[np.ndarray | pd.Series | list[float]] = []
    crossing_details = []
    for energy_lambda in (0.0, 4.0):
        group = focused[
            focused["energy_lambda"].eq(energy_lambda)
        ].sort_values("step")
        if group.empty:
            continue
        color = colors[energy_lambda]
        axis.plot(
            group["step"],
            group["mean_target_probability"],
            color=color,
            lw=1.9,
            label=rf"target, $\lambda={energy_lambda:g}$",
        )
        axis.plot(
            group["step"],
            group["mean_source_probability"],
            color=color,
            lw=1.55,
            ls="--",
            label=rf"source, $\lambda={energy_lambda:g}$",
        )
        y_extent.extend(
            [
                group["mean_target_probability"],
                group["mean_source_probability"],
            ]
        )
        crossed = group[
            group["mean_target_probability"].ge(
                group["mean_source_probability"]
            )
        ]
        if not crossed.empty:
            crossing_step = int(crossed.iloc[0]["step"])
            crossing_details.append(
                rf"$t^*_{{\rm mean}}={crossing_step}$ for "
                rf"$\lambda={energy_lambda:g}$"
            )
            axis.axvline(
                crossing_step,
                color=color,
                lw=1.0,
                ls=":",
                alpha=0.85,
            )
            target_value = float(
                crossed.iloc[0]["mean_target_probability"]
            )
            axis.scatter(
                [crossing_step],
                [target_value],
                color=color,
                s=24,
                zorder=4,
            )
        else:
            crossing_details.append(
                rf"no mean crossing by step 50 for "
                rf"$\lambda={energy_lambda:g}$"
            )

    axis.set_xlabel("U-turn step")
    axis.set_ylabel(r"mean classifier probability")
    axis.set_title(
        r"Dog-to-dog MH steering: source-target crossing, $\rho=0.4$"
    )
    informative_ylim(axis, y_extent, bounds=(0.0, 1.0))
    axis.legend(fontsize=7.8, ncol=2)
    crossing_note = (
        "; ".join(crossing_details)
        if crossing_details
        else "The mean target and source curves do not cross."
    )
    definition_panel(
        definition_axis,
        (
            "Mean classifier probabilities of the selected target dog class "
            "and the source dog class along the accepted U-turn chain."
        ),
        (
            r"$\bar p_a(t)=\frac{1}{K}\sum_{r=1}^{K}p_a(x_t^{(r)}),"
            r"\quad a\in\{\mathrm{source},\mathrm{target}\}$",
            r"$t^*_{\rm mean}=\min\{t:\bar p_{\rm target}(t)"
            r"\geq\bar p_{\rm source}(t)\}$",
        ),
        (
            "Solid lines are target probabilities and dashed lines are source "
            "probabilities. Dotted vertical lines mark crossings of the mean "
            f"curves. {crossing_note}. The chain uses 12 independent runs per "
            "condition and 50 U-turn steps."
        ),
        (
            "Earlier target-over-source crossing means faster steering. "
            "The chain-level crossing CDF is the inferential crossing result."
        ),
    )
    stem = "mh_rule_focused_source_target_crossing_trajectory"
    save_figure(fig, output_dir, stem)
    return stem


def plot_mh_crossing_trajectory(
    trajectories: pd.DataFrame,
    output_dir: Path,
    mode: str,
    rho: float,
    stem_suffix: str,
) -> str | None:
    required = {
        "cumulative_crossing_rate",
        "cumulative_crossing_ci_low",
        "cumulative_crossing_ci_high",
    }
    if not required.issubset(trajectories.columns):
        return None
    subset = trajectories[
        trajectories["target_mode"].eq(mode)
        & np.isclose(trajectories["rho"], rho)
    ]
    if subset.empty:
        return None
    fig, axis, definition_axis = simple_axes()
    colors = {0.0: "#777777", 1.0: BLUE, 4.0: RED}
    for energy_lambda, group in subset.groupby("energy_lambda", sort=True):
        group = group.sort_values("step")
        color = colors.get(float(energy_lambda), "#333333")
        axis.plot(
            group["step"],
            group["cumulative_crossing_rate"],
            color=color,
            lw=1.7,
            label=rf"$\lambda={energy_lambda:g}$",
        )
        axis.fill_between(
            group["step"],
            group["cumulative_crossing_ci_low"],
            group["cumulative_crossing_ci_high"],
            color=color,
            alpha=0.11,
            linewidth=0,
        )
    mode_label = "dog to dog" if mode == "dog_class" else "dog to cat"
    axis.set_xlabel("U-turn step")
    axis.set_ylabel(r"$P(\tau_{\rm cross}\leq t)$ ($\uparrow$)")
    axis.set_title(rf"MH-rule {mode_label} crossing, $\rho={rho:g}$")
    maximum_crossing = float(subset["cumulative_crossing_rate"].max())
    if maximum_crossing <= 1e-12:
        axis.set_ylim(0, 0.12)
        axis.text(
            float(subset["step"].max()) / 2,
            0.055,
            "No target-over-source crossings observed",
            ha="center",
            va="center",
            fontsize=9,
            color="#555555",
        )
    else:
        axis.set_ylim(0, 1)
    axis.legend(fontsize=8)
    run_counts = (
        subset.groupby("energy_lambda", sort=True)["runs"].max().to_dict()
    )
    run_text = ", ".join(
        rf"$K={int(count)}$ for $\lambda={energy_lambda:g}$"
        for energy_lambda, count in run_counts.items()
    )
    definition_panel(
        definition_axis,
        (
            "Cumulative fraction of independent chains that have reached a "
            "state where target probability is at least source probability by "
            "the plotted U-turn step."
        ),
        (
            r"$\tau_r=\min\{t:p_{\rm target}(x_t^{(r)})"
            r"\geq p_{\rm source}(x_t^{(r)})\}$",
            r"$y_t=P(\tau_{\rm cross}\leq t)"
            r"=\frac{1}{K}\sum_{r=1}^{K}\mathbf{1}\{\tau_r\leq t\}$",
        ),
        (
            "Bands are pointwise 500-resample bootstrap 95% intervals over "
            "independent chains. The target/source definitions and "
            "classifier-energy MH rule are the same as in the endpoint figure. "
            f"At this setting: {run_text}."
        ),
        (
            "Earlier rise means earlier crossing events. Steering additionally "
            "requires an endpoint improvement relative to the unsteered chain."
        ),
    )
    stem = f"mh_rule_cumulative_crossing_{stem_suffix}"
    save_figure(fig, output_dir, stem)
    return stem


def plot_mh_focused_target_side_trajectory(
    trajectories: pd.DataFrame, output_dir: Path
) -> str | None:
    required = {
        "instantaneous_crossing_rate",
        "instantaneous_crossing_ci_low",
        "instantaneous_crossing_ci_high",
    }
    if not required.issubset(trajectories.columns):
        return None
    focused = trajectories[
        trajectories["target_mode"].eq("dog_class")
        & np.isclose(trajectories["rho"], 0.4)
        & trajectories["energy_lambda"].isin([0.0, 4.0])
    ]
    if focused.empty:
        return None

    fig, axis, definition_axis = simple_axes()
    colors = {0.0: "#777777", 4.0: RED}
    threshold_notes = []
    for energy_lambda in (0.0, 4.0):
        group = focused[
            focused["energy_lambda"].eq(energy_lambda)
        ].sort_values("step")
        if group.empty:
            continue
        color = colors[energy_lambda]
        axis.plot(
            group["step"],
            group["instantaneous_crossing_rate"],
            color=color,
            lw=1.8,
            label=rf"$\lambda={energy_lambda:g}$",
        )
        axis.fill_between(
            group["step"],
            group["instantaneous_crossing_ci_low"],
            group["instantaneous_crossing_ci_high"],
            color=color,
            alpha=0.11,
            linewidth=0,
        )
        majority = group[group["instantaneous_crossing_rate"].ge(0.5)]
        if majority.empty:
            threshold_notes.append(
                rf"$q_t$ never reaches 0.5 for $\lambda={energy_lambda:g}$"
            )
        else:
            threshold_step = int(majority.iloc[0]["step"])
            threshold_notes.append(
                rf"$q_t$ first reaches 0.5 at $t={threshold_step}$ for "
                rf"$\lambda={energy_lambda:g}$"
            )
            axis.axvline(
                threshold_step,
                color=color,
                lw=1.0,
                ls=":",
                alpha=0.85,
            )
    axis.axhline(0.5, color="#B5B5B5", lw=0.9, ls="--")
    axis.set_xlabel("U-turn step")
    axis.set_ylabel(r"$P(p_{\rm target}\geq p_{\rm source}\mid t)$ ($\uparrow$)")
    axis.set_title(
        r"Dog-to-dog MH steering: target-side occupancy, $\rho=0.4$"
    )
    axis.set_ylim(0, 1)
    axis.legend(fontsize=8)
    definition_panel(
        definition_axis,
        (
            "Fraction of independent chains currently on the target side at "
            "each U-turn step. Unlike the crossing CDF, this does not retain "
            "earlier transient crossings."
        ),
        (
            r"$I_r(t)=\mathbf{1}\{p_{\rm target}(x_t^{(r)})"
            r"\geq p_{\rm source}(x_t^{(r)})\}$",
            r"$q_t=\frac{1}{K}\sum_{r=1}^{K}I_r(t)$",
        ),
        (
            "The estimate uses 12 independent chains per condition. Bands are "
            "pointwise 500-resample bootstrap 95% intervals. "
            + "; ".join(threshold_notes)
            + "."
        ),
        (
            "A sustained increase relative to the unsteered chain is evidence "
            "of steering; isolated crossings are not sufficient."
        ),
    )
    stem = "mh_rule_focused_target_side_occupancy_trajectory"
    save_figure(fig, output_dir, stem)
    return stem


def plot_cached_trajectory(
    trajectories: pd.DataFrame,
    metric: str,
    output_dir: Path,
    stem: str,
) -> str | None:
    subset = trajectories[trajectories["metric"].eq(metric)]
    if subset.empty:
        return None
    fig, axis, definition_axis = simple_axes()
    labels = {"dog_to_dog": "dog to dog", "dog_to_cat": "dog to cat"}
    colors = {"dog_to_dog": BLUE, "dog_to_cat": RED}
    y_extent: list[np.ndarray | pd.Series | list[float]] = []
    majority_details = {}
    endpoint_details = {}
    for mode in ("dog_to_dog", "dog_to_cat"):
        group = subset[subset["mode"].eq(mode)].sort_values("step")
        if group.empty:
            continue
        group = group.sort_values("step")
        plot_method = axis.step if metric == "cumulative_crossing_rate" else axis.plot
        plot_kwargs = {"where": "post"} if metric == "cumulative_crossing_rate" else {}
        plot_method(
            group["step"],
            group["mean"],
            color=colors[mode],
            lw=1.8,
            label=labels[mode],
            **plot_kwargs,
        )
        axis.fill_between(
            group["step"],
            group["ci_low"],
            group["ci_high"],
            color=colors[mode],
            alpha=0.13,
            linewidth=0,
        )
        y_extent.extend([group["ci_low"], group["ci_high"]])
        if metric == "cumulative_crossing_rate":
            majority = group[group["mean"].ge(0.5)]
            majority_details[mode] = (
                int(majority.iloc[0]["step"]) if not majority.empty else None
            )
            endpoint_details[mode] = float(group.iloc[-1]["mean"])
    axis.set_xlabel("U-turn step")
    axis.legend(fontsize=8)
    if metric == "target_probability":
        axis.set_ylabel(r"mean target probability ($\uparrow$)")
        axis.set_title(r"Selection-based steering at $\rho=0.1$")
        words = (
            "Classifier probability of the selected target, averaged first "
            "over four trajectories for each source image and then over images."
        )
        equations = (
            r"$p_{\rm target}(x_t)=p_{c_*}(x_t)$"
            r" or $\sum_{c\in\mathcal{C}_{\rm cat}}p_c(x_t)$",
            r"$y_t=\frac{1}{N}\sum_{s=1}^{N}\frac{1}{R_s}"
            r"\sum_{r=1}^{R_s}p_{\rm target}(x_{srt})$",
        )
        direction = "Higher is better for steering success."
        informative_ylim(axis, y_extent, bounds=(0.0, 1.0))
    elif metric == "instantaneous_crossing_rate":
        axis.set_ylabel(
            r"$P(p_{\rm target}\geq p_{\rm source}\mid t)$ ($\uparrow$)"
        )
        axis.set_title(
            r"Selection-based target-side occupancy at $\rho=0.1$"
        )
        axis.set_ylim(0, 1)
        words = (
            "Fraction of trajectories currently at a state where target "
            "probability is at least source probability at the plotted step."
        )
        equations = (
            r"$I_{sr}(t)=\mathbf{1}\{p_{\rm target}(x_{srt})"
            r"\geq p_{\rm source}(x_{srt})\}$",
            r"$y_t=\frac{1}{N}\sum_{s=1}^{N}\frac{1}{R_s}"
            r"\sum_{r=1}^{R_s}I_{sr}(t)$",
        )
        direction = (
            "Higher means more current chain states lie on the target side."
        )
    else:
        axis.set_ylabel(r"$P(\tau_{\rm cross}\leq t)$ ($\uparrow$)")
        axis.set_title(
            r"Steering difficulty: dog to dog versus dog to cat, $\rho=0.1$"
        )
        axis.set_ylim(0, 1)
        axis.set_xlim(0, 50)
        axis.axhline(0.5, color="#777777", lw=0.8, ls=":", zorder=0)
        annotation_positions = {
            "dog_to_dog": (5.0, 0.69),
            "dog_to_cat": (14.0, 0.42),
        }
        for mode in ("dog_to_dog", "dog_to_cat"):
            crossing_step = majority_details.get(mode)
            if crossing_step is None:
                continue
            group = subset[subset["mode"].eq(mode)].set_index("step")
            crossing_value = float(group.loc[crossing_step, "mean"])
            axis.axvline(
                crossing_step,
                ymax=crossing_value,
                color=colors[mode],
                lw=0.9,
                ls=":",
                alpha=0.9,
            )
            axis.scatter(
                [crossing_step],
                [crossing_value],
                color=colors[mode],
                s=24,
                zorder=5,
            )
            axis.annotate(
                f"50% by step {crossing_step}",
                xy=(crossing_step, crossing_value),
                xytext=annotation_positions[mode],
                textcoords="data",
                fontsize=8,
                color=colors[mode],
                arrowprops={
                    "arrowstyle": "->",
                    "color": colors[mode],
                    "lw": 0.8,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
            )
            endpoint = endpoint_details[mode]
            y_offset = -0.033 if mode == "dog_to_dog" else 0.015
            axis.annotate(
                f"{100 * endpoint:.1f}% by step 50",
                xy=(50, endpoint),
                xytext=(48.8, endpoint + y_offset),
                ha="right",
                va="center",
                fontsize=8,
                color=colors[mode],
            )
        words = (
            "Empirical cumulative distribution of the first U-turn step at "
            "which target probability reaches or exceeds source probability."
        )
        equations = (
            r"$\tau_{sr}=\min\{j:p_{\rm target}(x_{srj})"
            r"\geq p_{\rm source}(x_{srj})\}$",
            r"$y_t=\frac{1}{N}\sum_{s=1}^{N}\frac{1}{R_s}"
            r"\sum_{r=1}^{R_s}\mathbf{1}\{\tau_{sr}\leq t\}$",
        )
        direction = (
            "Earlier rise and a higher endpoint mean easier, more reliable steering. "
            "Dog-to-cat is substantially harder."
        )
    sample_detail = (
        "The estimate uses 100 strict dog starting images and four "
        "stochastic runs per image. Bands are mean +/- 1.96 standard "
        "errors across the 100 image-level means. Dog-to-cat uses summed "
        "cat probability versus summed dog probability."
    )
    if metric == "cumulative_crossing_rate":
        sample_detail += (
            " Half the runs cross by step "
            f"{majority_details.get('dog_to_dog')} for dog-to-dog and step "
            f"{majority_details.get('dog_to_cat')} for dog-to-cat; endpoints "
            f"are {100 * endpoint_details.get('dog_to_dog', np.nan):.1f}% and "
            f"{100 * endpoint_details.get('dog_to_cat', np.nan):.1f}%."
        )
    definition_panel(
        definition_axis,
        words,
        equations,
        sample_detail,
        direction,
    )
    aliases = (
        ("selection_dog_to_dog_vs_dog_to_cat_difficulty",)
        if metric == "cumulative_crossing_rate"
        else ()
    )
    save_figure(fig, output_dir, stem, aliases=aliases)
    return aliases[0] if aliases else stem


def plot_cached_probability_crossing(
    trajectories: pd.DataFrame,
    output_dir: Path,
    mode: str,
) -> str | None:
    target = trajectories[
        trajectories["metric"].eq("target_probability")
        & trajectories["mode"].eq(mode)
    ].sort_values("step")
    source = trajectories[
        trajectories["metric"].eq("source_probability")
        & trajectories["mode"].eq(mode)
    ].sort_values("step")
    if target.empty or source.empty:
        return None
    target = target.set_index("step")
    source = source.set_index("step")
    common_steps = target.index.intersection(source.index)
    target = target.loc[common_steps]
    source = source.loc[common_steps]

    fig, axis, definition_axis = simple_axes()
    target_color = BLUE if mode == "dog_to_dog" else RED
    axis.plot(
        common_steps,
        target["mean"],
        color=target_color,
        lw=1.9,
        label="target probability",
    )
    axis.fill_between(
        common_steps,
        target["ci_low"],
        target["ci_high"],
        color=target_color,
        alpha=0.13,
        linewidth=0,
    )
    axis.plot(
        common_steps,
        source["mean"],
        color="#555555",
        lw=1.65,
        ls="--",
        label="source probability",
    )
    axis.fill_between(
        common_steps,
        source["ci_low"],
        source["ci_high"],
        color="#777777",
        alpha=0.08,
        linewidth=0,
    )

    crossed = target[target["mean"].ge(source["mean"])]
    crossing_text = "The mean curves do not cross."
    if not crossed.empty:
        crossing_step = int(crossed.index[0])
        crossing_value = float(target.loc[crossing_step, "mean"])
        axis.axvline(
            crossing_step,
            color=target_color,
            lw=1.1,
            ls=":",
        )
        axis.scatter(
            [crossing_step],
            [crossing_value],
            color=target_color,
            s=26,
            zorder=4,
        )
        axis.annotate(
            rf"$t^*_{{\rm mean}}={crossing_step}$",
            xy=(crossing_step, crossing_value),
            xytext=(7, 9),
            textcoords="offset points",
            fontsize=8,
            color=target_color,
        )
        crossing_text = (
            f"The target and source mean curves first cross at U-turn step "
            f"{crossing_step}."
        )

    mode_label = "dog to dog" if mode == "dog_to_dog" else "dog to cat"
    axis.set_xlabel("U-turn step")
    axis.set_ylabel("mean classifier probability")
    axis.set_title(
        rf"Selection-based {mode_label} crossing, $\rho=0.1$"
    )
    informative_ylim(
        axis,
        [
            target["ci_low"],
            target["ci_high"],
            source["ci_low"],
            source["ci_high"],
        ],
        bounds=(0.0, 1.0),
    )
    axis.legend(fontsize=8)
    class_detail = (
        "The target and source are two specific ImageNet dog classes."
        if mode == "dog_to_dog"
        else (
            "Dog-to-cat compares summed cat probability with summed dog "
            "probability."
        )
    )
    definition_panel(
        definition_axis,
        (
            "Image-level mean classifier probabilities of the steering target "
            "and the source class or class family along the U-turn chain."
        ),
        (
            r"$\bar p_a(t)=\frac{1}{N}\sum_{s=1}^{N}\frac{1}{R_s}"
            r"\sum_{r=1}^{R_s}p_a(x_{srt})$",
            r"$t^*_{\rm mean}=\min\{t:\bar p_{\rm target}(t)"
            r"\geq\bar p_{\rm source}(t)\}$",
        ),
        (
            "The estimate uses 100 starting images and four stochastic runs "
            f"per image. {crossing_text} {class_detail}"
        ),
        (
            "Target rising above source is successful steering. The separate "
            "crossing CDF reports chain-level uncertainty."
        ),
    )
    stem = f"selection_source_target_crossing_{mode}"
    save_figure(fig, output_dir, stem)
    return stem


def plot_cached_bars(
    payload: dict,
    metric: str,
    output_dir: Path,
    stem: str,
) -> str:
    modes = ("dog_to_dog", "dog_to_cat")
    labels = ("dog to dog", "dog to cat")
    colors = [BLUE, RED]
    values = np.asarray([payload[mode][metric] for mode in modes], dtype=float)
    low = np.asarray([payload[mode][f"{metric}_ci_low"] for mode in modes], dtype=float)
    high = np.asarray(
        [payload[mode][f"{metric}_ci_high"] for mode in modes], dtype=float
    )
    fig, axis, definition_axis = simple_axes()
    axis.bar([0, 1], values, color=colors, width=0.58)
    axis.errorbar(
        [0, 1],
        values,
        yerr=[values - low, high - values],
        fmt="none",
        color="black",
        capsize=3,
        lw=1,
    )
    axis.set_xticks([0, 1], labels)
    axis.set_ylim(0, 1)
    if metric == "final_target_probability":
        axis.set_ylabel(r"final target probability ($\uparrow$)")
        axis.set_title("Selection-based steering endpoints")
        words = "Mean classifier target probability at the final U-turn state."
        equations = (
            r"$y=\frac{1}{N}\sum_{s=1}^{N}\frac{1}{R_s}"
            r"\sum_{r=1}^{R_s}p_{\rm target}(x_{srT})$",
        )
        direction = "Higher is better for steering success."
    elif metric == "crossing_rate":
        axis.set_ylabel(r"target-over-source crossing rate ($\uparrow$)")
        axis.set_title("Selection-based steering success")
        words = (
            "Fraction of trajectories that reach target probability greater "
            "than or equal to source probability at least once."
        )
        equations = (
            r"$I_{sr}=\mathbf{1}\{\exists t:"
            r"p_{\rm target}(x_{srt})\geq p_{\rm source}(x_{srt})\}$",
            r"$y=\frac{1}{N}\sum_{s=1}^{N}\frac{1}{R_s}\sum_{r=1}^{R_s}I_{sr}$",
        )
        direction = "Higher means more trajectories reach the target side."
    else:
        axis.set_ylabel(r"final target-majority rate ($\uparrow$)")
        axis.set_title("Selection-based target-majority endpoints")
        words = (
            "Fraction of final states whose target classifier probability is "
            "at least one half."
        )
        equations = (
            r"$J_{sr}=\mathbf{1}\{p_{\rm target}(x_{srT})\geq 0.5\}$",
            r"$y=\frac{1}{N}\sum_{s=1}^{N}\frac{1}{R_s}\sum_{r=1}^{R_s}J_{sr}$",
        )
        direction = "Higher means more endpoints are target-majority."
    definition_panel(
        definition_axis,
        words,
        equations,
        (
            "The estimate uses 100 strict dog starting images and four "
            r"stochastic runs per image at $\rho=0.1$. Error bars are "
            "500-resample bootstrap 95% intervals over image-level means. "
            "This is proposal selection, not the MH-rule experiment."
        ),
        direction,
    )
    save_figure(fig, output_dir, stem)
    return stem


def write_index(entries: list[tuple[str, str, str]], output_path: Path) -> None:
    lines = [
        "# Standalone image-rebuttal figures",
        "",
        "Every file contains one data plot and a separate definition panel with the "
        "implemented y-axis statistic. PNG and PDF versions are generated.",
        "",
        "| Group | Quantity | Figure stem |",
        "|---|---|---|",
    ]
    for group, quantity, stem in entries:
        lines.append(f"| {group} | {quantity} | `{stem}.png` |")
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    style()

    quality_root = args.results_root / "image_quality"
    quality_output = quality_root / "standalone"
    single = pd.read_csv(quality_root / "single_uturn_shared20_quality_summary.csv")
    sequential = pd.read_csv(
        quality_root / "sequential_uturn_shared20_quality_summary.csv"
    )
    direct = pd.read_csv(quality_root / "direct_diffusion_quality_summary.csv")
    entries: list[tuple[str, str, str]] = []
    by_column = {spec.column: spec for spec in QUALITY_METRICS}
    for spec in QUALITY_METRICS:
        stem = plot_quality_vs_change(
            single, sequential, direct, spec, quality_output
        )
        if stem is not None:
            entries.append(("quality vs accumulated change", spec.title, stem))
    for column, stem_prefix in STEP_METRIC_STEMS.items():
        spec = by_column[column]
        step_spec = MetricSpec(
            **{
                **spec.__dict__,
                "stem": stem_prefix,
            }
        )
        stem = plot_quality_vs_step(sequential, direct, step_spec, quality_output)
        if stem is not None:
            entries.append(("quality vs chain step", spec.title, stem))
    write_index(entries, quality_output / "standalone_figure_index.md")

    mh_root = args.results_root / "mh_steering"
    mh_output = mh_root / "standalone"
    mh_summary = pd.read_csv(mh_root / "mh_image_steering_summary.csv")
    mh_entries: list[tuple[str, str, str]] = []
    mh_specs = (
        (
            "final_target_probability",
            "Final target probability",
            r"final target probability ($\uparrow$)",
            "Mean target classifier probability at the terminal chain state.",
            (
                r"$y=\frac{1}{K}\sum_{r=1}^{K}"
                r"p_{\rm target}(x_T^{(r)})$",
            ),
            "Higher is better for steering success.",
            "mh_rule_final_target_probability_vs_rho",
            (0.0, 1.0),
        ),
        (
            "acceptance_rate",
            "MH-rule acceptance rate",
            "accepted proposal fraction",
            "Fraction of U-turn proposals accepted by the energy-only MH rule.",
            (
                r"$A_t=\min\{1,\exp[-H(x_t')+H(x_{t-1})]\}$",
                r"$y=\frac{1}{K}\sum_{r=1}^{K}\frac{1}{T}"
                r"\sum_{t=1}^{T}\mathbf{1}\{u_{rt}<A_{rt}\}$",
            ),
            "This is diagnostic; neither larger nor smaller is universally better.",
            "mh_rule_acceptance_rate_vs_rho",
            (0.0, 1.0),
        ),
        (
            "crossed",
            "Target-over-source crossing rate",
            r"target-over-source crossing rate ($\uparrow$)",
            (
                "Fraction of chains that reach target probability greater than "
                "or equal to source probability at least once."
            ),
            (
                r"$I_r=\mathbf{1}\{\exists t:"
                r"p_{\rm target}(x_t^{(r)})\geq p_{\rm source}(x_t^{(r)})\}$",
                r"$y=\frac{1}{K}\sum_{r=1}^{K}I_r$",
            ),
            "Higher means more chains reach the target side.",
            "mh_rule_crossing_rate_vs_rho",
            (0.0, 1.0),
        ),
    )
    for metric, title, ylabel, words, equations, direction, stem, ylim in mh_specs:
        created = plot_mh_summary_metric(
            mh_summary,
            metric,
            title,
            ylabel,
            words,
            equations,
            direction,
            mh_output,
            stem,
            ylim,
        )
        mh_entries.append(("MH-rule steering", title, created))
    focused = plot_mh_focused_endpoint(mh_summary, mh_output)
    if focused:
        mh_entries.append(("MH-rule steering", "Focused endpoint result", focused))
    mh_trajectory_path = mh_root / "mh_image_steering_trajectories.csv"
    if mh_trajectory_path.exists():
        mh_trajectories = pd.read_csv(mh_trajectory_path)
        focused_crossing = plot_mh_focused_probability_crossing(
            mh_trajectories, mh_output
        )
        if focused_crossing:
            mh_entries.append(
                (
                    "MH-rule trajectories",
                    "Focused source-target crossing trajectory",
                    focused_crossing,
                )
            )
        focused_target_side = plot_mh_focused_target_side_trajectory(
            mh_trajectories, mh_output
        )
        if focused_target_side:
            mh_entries.append(
                (
                    "MH-rule trajectories",
                    "Focused target-side occupancy trajectory",
                    focused_target_side,
                )
            )
        for mode in ("dog_class", "cat"):
            for rho in sorted(mh_trajectories["rho"].unique()):
                suffix = f"{mode}_rho_{rho:g}".replace(".", "p")
                created = plot_mh_trajectory(
                    mh_trajectories, mh_output, mode, float(rho), suffix
                )
                if created:
                    mh_entries.append(
                        (
                            "MH-rule trajectories",
                            f"{mode}, rho={rho:g}",
                            created,
                        )
                    )
                crossing_created = plot_mh_crossing_trajectory(
                    mh_trajectories, mh_output, mode, float(rho), suffix
                )
                if crossing_created:
                    mh_entries.append(
                        (
                            "MH-rule trajectories",
                            f"{mode} crossing, rho={rho:g}",
                            crossing_created,
                        )
                    )
    write_index(mh_entries, mh_output / "standalone_figure_index.md")

    cached_root = args.results_root / "cached_steering"
    cached_output = cached_root / "standalone"
    cached_payload = json.loads(
        (cached_root / "cached_image_steering_summary.json").read_text()
    )
    cached_entries: list[tuple[str, str, str]] = []
    for metric, stem, label in (
        (
            "final_target_probability",
            "selection_final_target_probability",
            "Final target probability",
        ),
        ("crossing_rate", "selection_crossing_rate", "Crossing rate"),
        (
            "target_probability_ge_0_5",
            "selection_target_majority_rate",
            "Target-majority rate",
        ),
    ):
        created = plot_cached_bars(
            cached_payload, metric, cached_output, stem
        )
        cached_entries.append(("selection steering", label, created))
    cached_trajectory_path = cached_root / "cached_image_steering_trajectories.csv"
    if cached_trajectory_path.exists():
        cached_trajectories = pd.read_csv(cached_trajectory_path)
        for metric, stem, label in (
            (
                "target_probability",
                "selection_target_probability_trajectory",
                "Target probability trajectory",
            ),
            (
                "instantaneous_crossing_rate",
                "selection_target_side_occupancy_trajectory",
                "Target-side occupancy trajectory",
            ),
            (
                "cumulative_crossing_rate",
                "selection_crossing_trajectory",
                "Cumulative crossing trajectory",
            ),
        ):
            created = plot_cached_trajectory(
                cached_trajectories, metric, cached_output, stem
            )
            if created:
                cached_entries.append(("selection steering", label, created))
        for mode in ("dog_to_dog", "dog_to_cat"):
            created = plot_cached_probability_crossing(
                cached_trajectories, cached_output, mode
            )
            if created:
                mode_label = (
                    "Dog-to-dog source-target crossing"
                    if mode == "dog_to_dog"
                    else "Dog-to-cat source-target crossing"
                )
                cached_entries.append(
                    ("selection steering", mode_label, created)
                )
    write_index(
        cached_entries, cached_output / "standalone_figure_index.md"
    )


if __name__ == "__main__":
    main()
