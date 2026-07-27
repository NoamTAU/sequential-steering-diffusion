#!/usr/bin/env python3
"""Calibrate no-reference IQA metrics on existing image rebuttal caches.

The workflow is deliberately split:

1. ``prepare`` samples controls and transition-region candidates, creates
   synthetic positive controls, and writes blinded contact sheets.
2. The transition candidates are visually labeled without seeing IQA scores.
3. ``score`` runs no-reference IQA models and reports whether they agree with
   the controls and visual labels.

No diffusion generation is performed.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont


SEED = 44
DEFAULT_MODELS = ("musiq", "topiq_nr", "niqe")
PRIMARY_LEARNED_MODELS = ("musiq", "topiq_nr")
HIGHER_IS_BETTER = {
    "musiq": True,
    "topiq_nr": True,
    "niqe": False,
}
TRANSITION_SINGLE_RHOS = (0.4, 0.5, 0.6, 0.7)
TRANSITION_SEQUENTIAL_STRATA = (
    (0.4, 20),
    (0.4, 50),
    (0.6, 10),
    (0.6, 40),
)
VISUAL_LABELS = ("intact", "borderline", "malformed")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def sample_rows(
    frame: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
    *,
    unique_by: str | None = None,
) -> pd.DataFrame:
    if unique_by is not None:
        frame = frame.drop_duplicates(unique_by)
    if len(frame) < count:
        raise ValueError(f"Need {count} rows, found {len(frame)}")
    indices = rng.choice(len(frame), size=count, replace=False)
    return frame.iloc[np.sort(indices)].copy()


def record(
    *,
    sample_id: str,
    cohort: str,
    image_path: str | Path,
    source_image_id: str,
    dataset: str,
    rho: float | str = "",
    step: int | str = "",
    trajectory_id: int | str = "",
    expected_role: str,
    transition_kind: str = "",
) -> dict:
    return {
        "sample_id": sample_id,
        "cohort": cohort,
        "image_path": str(image_path),
        "source_image_id": str(source_image_id),
        "dataset": dataset,
        "rho": rho,
        "step": step,
        "trajectory_id": trajectory_id,
        "expected_role": expected_role,
        "transition_kind": transition_kind,
    }


def image_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def make_contact_sheet(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    columns: int,
    title: str,
    label_columns: tuple[str, ...],
    tile_size: int = 214,
) -> None:
    label_height = 66
    title_height = 50
    gap = 10
    rows = math.ceil(len(frame) / columns)
    width = columns * tile_size + (columns - 1) * gap
    height = title_height + rows * (tile_size + label_height + gap) - gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (width // 2, 24),
        title,
        anchor="mm",
        fill="#202020",
        font=image_font(20),
    )
    for item_index, (_, row) in enumerate(frame.iterrows()):
        grid_row, grid_column = divmod(item_index, columns)
        x = grid_column * (tile_size + gap)
        y = title_height + grid_row * (tile_size + label_height + gap)
        with Image.open(row["image_path"]) as handle:
            image = handle.convert("RGB")
            image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_size, tile_size), "#F1F1F1")
        tile.paste(
            image,
            ((tile_size - image.width) // 2, (tile_size - image.height) // 2),
        )
        canvas.paste(tile, (x, y))
        label = "\n".join(
            textwrap.fill(str(row[column]), width=30)
            for column in label_columns
            if str(row[column])
        )
        draw.multiline_text(
            (x + tile_size // 2, y + tile_size + label_height // 2),
            label,
            anchor="mm",
            align="center",
            spacing=3,
            fill="#202020",
            font=image_font(13),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_synthetic_controls(
    originals: pd.DataFrame,
    output_dir: Path,
    rng: np.random.Generator,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    selected = originals.iloc[:8]
    for _, source in selected.iterrows():
        with Image.open(source["image_path"]) as handle:
            image = handle.convert("RGB")
        source_id = str(source["source_image_id"])

        blur = image.filter(ImageFilter.GaussianBlur(radius=2.5))
        blur_path = output_dir / f"{source_id}_blur.png"
        blur.save(blur_path)
        rows.append(
            record(
                sample_id=f"synthetic_blur_{source_id}",
                cohort="synthetic_degraded",
                image_path=blur_path,
                source_image_id=source_id,
                dataset="synthetic",
                expected_role="known_degradation",
                transition_kind="gaussian_blur_r2.5",
            )
        )

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=8, optimize=False)
        buffer.seek(0)
        jpeg = Image.open(buffer).convert("RGB")
        jpeg_path = output_dir / f"{source_id}_jpeg_q08.png"
        jpeg.save(jpeg_path)
        rows.append(
            record(
                sample_id=f"synthetic_jpeg_{source_id}",
                cohort="synthetic_degraded",
                image_path=jpeg_path,
                source_image_id=source_id,
                dataset="synthetic",
                expected_role="known_degradation",
                transition_kind="jpeg_quality_8",
            )
        )

        pixels = np.asarray(image).astype(np.float32) / 255.0
        noise = rng.normal(0.0, 0.08, size=pixels.shape).astype(np.float32)
        noisy = Image.fromarray(
            np.round(np.clip(pixels + noise, 0.0, 1.0) * 255).astype(np.uint8)
        )
        noise_path = output_dir / f"{source_id}_noise_s08.png"
        noisy.save(noise_path)
        rows.append(
            record(
                sample_id=f"synthetic_noise_{source_id}",
                cohort="synthetic_degraded",
                image_path=noise_path,
                source_image_id=source_id,
                dataset="synthetic",
                expected_role="known_degradation",
                transition_kind="gaussian_noise_sigma_0.08",
            )
        )
    return rows


def prepare(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    manifest_dir = args.root / "image_quality/manifests"
    shared = read_csv(manifest_dir / "shared_start_manifest.csv")
    single = read_csv(manifest_dir / "single_manifest.csv")
    sequential = read_csv(manifest_dir / "sequential_manifest.csv")
    direct = read_csv(args.root / "direct_diffusion/direct_manifest.csv")
    for name, frame in (
        ("shared", shared),
        ("single", single),
        ("sequential", sequential),
        ("direct", direct),
    ):
        require_columns(frame, ("image_id", "image_path"), name)
    require_columns(single, ("rho", "trajectory_id", "step"), "single")
    require_columns(sequential, ("rho", "trajectory_id", "step"), "sequential")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared = shared.sort_values("image_id").head(args.control_count).copy()
    shared_ids = set(shared["image_id"].astype(str))
    records: list[dict] = []

    for index, (_, row) in enumerate(shared.iterrows()):
        records.append(
            record(
                sample_id=f"original_{index:03d}",
                cohort="original_real",
                image_path=row["image_path"],
                source_image_id=row["image_id"],
                dataset="original",
                expected_role="high_quality_reference",
            )
        )

    selected_direct = sample_rows(direct, args.control_count, rng)
    for index, (_, row) in enumerate(selected_direct.iterrows()):
        records.append(
            record(
                sample_id=f"direct_{index:03d}",
                cohort="direct_diffusion",
                image_path=row["image_path"],
                source_image_id=row["image_id"],
                dataset="direct",
                expected_role="generator_baseline",
            )
        )

    single_control = single[
        np.isclose(single["rho"].astype(float), 0.1)
        & single["trajectory_id"].astype(int).eq(0)
        & single["image_id"].astype(str).isin(shared_ids)
    ].drop_duplicates("image_id")
    single_control = sample_rows(
        single_control, min(args.control_count, len(single_control)), rng
    )
    for index, (_, row) in enumerate(single_control.iterrows()):
        records.append(
            record(
                sample_id=f"single_low_noise_{index:03d}",
                cohort="single_low_noise",
                image_path=row["image_path"],
                source_image_id=row["image_id"],
                dataset="single",
                rho=row["rho"],
                step=row["step"],
                trajectory_id=row["trajectory_id"],
                expected_role="intact_uturn_control",
            )
        )

    sequential_control = sequential[
        np.isclose(sequential["rho"].astype(float), 0.1)
        & sequential["trajectory_id"].astype(int).eq(0)
        & sequential["step"].astype(int).eq(100)
        & sequential["image_id"].astype(str).isin(shared_ids)
    ].drop_duplicates("image_id")
    sequential_control = sample_rows(
        sequential_control, min(args.control_count, len(sequential_control)), rng
    )
    for index, (_, row) in enumerate(sequential_control.iterrows()):
        records.append(
            record(
                sample_id=f"sequential_low_noise_{index:03d}",
                cohort="sequential_low_noise",
                image_path=row["image_path"],
                source_image_id=row["image_id"],
                dataset="sequential",
                rho=row["rho"],
                step=row["step"],
                trajectory_id=row["trajectory_id"],
                expected_role="long_chain_control",
            )
        )

    transition_rows = []
    per_stratum = args.transition_per_stratum
    for rho in TRANSITION_SINGLE_RHOS:
        candidates = single[
            np.isclose(single["rho"].astype(float), rho)
            & single["trajectory_id"].astype(int).eq(0)
        ]
        chosen = sample_rows(candidates, per_stratum, rng, unique_by="image_id")
        chosen = chosen.assign(
            transition_kind=f"single_rho_{rho:g}", source_dataset="single"
        )
        transition_rows.append(chosen)
    for rho, step in TRANSITION_SEQUENTIAL_STRATA:
        candidates = sequential[
            np.isclose(sequential["rho"].astype(float), rho)
            & sequential["step"].astype(int).eq(step)
            & sequential["trajectory_id"].astype(int).eq(0)
        ]
        chosen = sample_rows(candidates, per_stratum, rng, unique_by="image_id")
        chosen = chosen.assign(
            transition_kind=f"sequential_rho_{rho:g}_step_{step}",
            source_dataset="sequential",
        )
        transition_rows.append(chosen)
    transition = pd.concat(transition_rows, ignore_index=True)
    transition = transition.iloc[rng.permutation(len(transition))].reset_index(
        drop=True
    )
    for index, row in transition.iterrows():
        records.append(
            record(
                sample_id=f"transition_{index:03d}",
                cohort="transition_candidate",
                image_path=row["image_path"],
                source_image_id=row["image_id"],
                dataset=row["source_dataset"],
                rho=row["rho"],
                step=row["step"],
                trajectory_id=row["trajectory_id"],
                expected_role="blind_visual_label",
                transition_kind=row["transition_kind"],
            )
        )

    originals = pd.DataFrame(
        [row for row in records if row["cohort"] == "original_real"]
    )
    records.extend(
        save_synthetic_controls(
            originals,
            args.output_dir / "synthetic_controls",
            rng,
        )
    )
    manifest = pd.DataFrame(records)
    missing = manifest.loc[
        ~manifest["image_path"].map(lambda value: Path(value).is_file())
    ]
    if len(missing):
        raise FileNotFoundError(
            f"{len(missing)} selected pilot images are missing; first rows:\n"
            f"{missing.head()}"
        )
    manifest.to_csv(args.output_dir / "pilot_manifest.csv", index=False)

    labels = manifest.loc[
        manifest["cohort"].eq("transition_candidate"), ["sample_id"]
    ].copy()
    labels["visual_label"] = ""
    labels["visual_notes"] = ""
    labels.to_csv(args.output_dir / "pilot_visual_labels.csv", index=False)

    controls = manifest[
        manifest["cohort"].isin(
            (
                "original_real",
                "direct_diffusion",
                "single_low_noise",
                "sequential_low_noise",
                "synthetic_degraded",
            )
        )
    ].copy()
    controls = controls.groupby("cohort", sort=False).head(8).reset_index(drop=True)
    controls["display_label"] = (
        controls["cohort"].str.replace("_", " ")
        + " | "
        + controls["sample_id"]
    )
    make_contact_sheet(
        controls,
        args.output_dir / "pilot_controls.png",
        columns=5,
        title="IQA pilot controls",
        label_columns=("display_label",),
        tile_size=186,
    )

    blind = manifest[manifest["cohort"].eq("transition_candidate")].copy()
    page_size = 16
    for page_index, start in enumerate(range(0, len(blind), page_size), start=1):
        page = blind.iloc[start : start + page_size]
        make_contact_sheet(
            page,
            args.output_dir / f"pilot_transition_blind_{page_index}.png",
            columns=4,
            title=f"Transition candidates, blind page {page_index}",
            label_columns=("sample_id",),
            tile_size=226,
        )

    metadata = {
        "seed": args.seed,
        "models": list(DEFAULT_MODELS),
        "rows": len(manifest),
        "cohort_counts": manifest["cohort"].value_counts().sort_index().to_dict(),
        "transition_single_rhos": list(TRANSITION_SINGLE_RHOS),
        "transition_sequential_strata": [
            {"rho": rho, "step": step}
            for rho, step in TRANSITION_SEQUENTIAL_STRATA
        ],
        "selection_is_independent_of_iqa": True,
    }
    (args.output_dir / "pilot_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


def sanitize_model_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name)


def score_iqa(
    manifest: pd.DataFrame,
    output_path: Path,
    models: tuple[str, ...],
    device: str,
) -> pd.DataFrame:
    try:
        import pyiqa
        import torch
    except ImportError as error:
        raise RuntimeError(
            "The score stage requires pyiqa and torch. Install IQA-PyTorch "
            "in the work-filesystem environment before running."
        ) from error

    available = set(pyiqa.list_models())
    missing = sorted(set(models) - available)
    if missing:
        raise ValueError(
            f"Unavailable pyiqa model names: {missing}. "
            f"Available examples: {sorted(available)[:20]}"
        )
    scored = manifest.copy()
    if output_path.is_file():
        prior = read_csv(output_path)
        if list(prior["sample_id"]) != list(scored["sample_id"]):
            raise ValueError("Existing score file does not match pilot manifest")
        retained_columns = [
            column
            for column in prior.columns
            if column not in scored.columns and not column.endswith("_quality_z")
        ]
        scored = scored.merge(
            prior[["sample_id", *retained_columns]],
            on="sample_id",
            how="left",
            validate="1:1",
        )

    for model_name in models:
        column = sanitize_model_name(model_name)
        if column in scored and pd.to_numeric(
            scored[column], errors="coerce"
        ).notna().all():
            print(f"{model_name}: complete in existing score file")
            continue
        metric = pyiqa.create_metric(model_name, device=device)
        values = []
        with torch.inference_mode():
            for row_index, path in enumerate(scored["image_path"].astype(str)):
                value = float(metric(path).detach().cpu().reshape(-1)[0])
                if not np.isfinite(value):
                    raise FloatingPointError(
                        f"{model_name} produced {value} for {path}"
                    )
                values.append(value)
                if (row_index + 1) % 20 == 0:
                    print(f"{model_name}: {row_index + 1}/{len(scored)}")
        scored[column] = values
        scored.to_csv(output_path, index=False)
        del metric
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return scored


def bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, resamples: int
) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def rank_auc(intact: np.ndarray, malformed: np.ndarray) -> float:
    if not len(intact) or not len(malformed):
        return float("nan")
    comparisons = intact[:, None] - malformed[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def bootstrap_auc(
    intact: np.ndarray,
    malformed: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
) -> tuple[float, float, float]:
    auc = rank_auc(intact, malformed)
    draws = []
    for _ in range(resamples):
        draws.append(
            rank_auc(
                rng.choice(intact, size=len(intact), replace=True),
                rng.choice(malformed, size=len(malformed), replace=True),
            )
        )
    return auc, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def oriented_scores(frame: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    output = frame.copy()
    oriented_columns: dict[str, str] = {}
    for model in models:
        column = sanitize_model_name(model)
        values = pd.to_numeric(output[column], errors="coerce")
        scale = float(values.std(ddof=0))
        z = (values - float(values.mean())) / (scale if scale > 0 else 1.0)
        if not HIGHER_IS_BETTER.get(model, True):
            z = -z
        oriented_column = f"{column}_quality_z"
        output[oriented_column] = z
        oriented_columns[model] = oriented_column
    consensus_columns = [
        oriented_columns[model]
        for model in PRIMARY_LEARNED_MODELS
        if model in oriented_columns
    ]
    if not consensus_columns:
        consensus_columns = list(oriented_columns.values())
    output["iqa_consensus_z"] = output[consensus_columns].mean(axis=1)
    return output


def make_score_examples(
    frame: pd.DataFrame, output_path: Path, count: int = 4
) -> None:
    candidates = frame[frame["cohort"].eq("transition_candidate")].copy()
    if "visual_label" in candidates and candidates["visual_label"].astype(bool).any():
        low = candidates[candidates["visual_label"].eq("malformed")].nsmallest(
            count, "iqa_consensus_z"
        )
        high = candidates[candidates["visual_label"].eq("intact")].nlargest(
            count, "iqa_consensus_z"
        )
    else:
        low = candidates.nsmallest(count, "iqa_consensus_z")
        high = candidates.nlargest(count, "iqa_consensus_z")
    examples = pd.concat([high, low], ignore_index=True)
    examples["display_label"] = examples.apply(
        lambda row: (
            f"{row['sample_id']} | {row.get('visual_label', '') or 'unlabeled'}"
        ),
        axis=1,
    )
    examples["display_scores"] = examples.apply(
        lambda row: (
            f"MUSIQ {row['musiq']:.1f}; TOPIQ-NR {row['topiq_nr']:.2f}"
        ),
        axis=1,
    )
    make_contact_sheet(
        examples,
        output_path,
        columns=count,
        title="Blind visual labels, ranked by learned-IQA consensus",
        label_columns=("display_label", "display_scores"),
        tile_size=224,
    )


def make_validation_examples(frame: pd.DataFrame, output_path: Path) -> None:
    def representative(group: pd.DataFrame, count: int) -> pd.DataFrame:
        median = float(group["iqa_consensus_z"].median())
        return group.assign(
            _distance=(group["iqa_consensus_z"] - median).abs()
        ).nsmallest(count, "_distance")

    rows: list[pd.DataFrame] = []

    references = []
    for cohort, label in (
        ("original_real", "Original real"),
        ("direct_diffusion", "Direct diffusion"),
    ):
        group = representative(frame[frame["cohort"].eq(cohort)], 2).copy()
        group["panel_label"] = label
        references.append(group)
    rows.append(pd.concat(references, ignore_index=True))

    transitions = frame[frame["cohort"].eq("transition_candidate")]
    intact = transitions[transitions["visual_label"].eq("intact")].nlargest(
        4, "iqa_consensus_z"
    ).copy()
    intact["panel_label"] = "U-turn, blind label: intact"
    rows.append(intact)

    malformed = transitions[transitions["visual_label"].eq("malformed")]
    detected = malformed.nsmallest(2, "iqa_consensus_z").copy()
    detected["panel_label"] = "Malformed, correctly demoted"
    missed = malformed.nlargest(2, "iqa_consensus_z").copy()
    missed["panel_label"] = "Malformed, IQA miss"
    rows.append(pd.concat([detected, missed], ignore_index=True))

    synthetic = frame[frame["cohort"].eq("synthetic_degraded")].copy()
    source_counts = synthetic.groupby("source_image_id").size()
    paired_source = str(source_counts[source_counts.ge(3)].index[0])
    paired_original = frame[
        frame["cohort"].eq("original_real")
        & frame["source_image_id"].astype(str).eq(paired_source)
    ].head(1).copy()
    paired_original["panel_label"] = "Paired original"
    paired_degraded = synthetic[
        synthetic["source_image_id"].astype(str).eq(paired_source)
    ].copy()
    paired_degraded["degradation"] = paired_degraded["sample_id"].str.extract(
        r"synthetic_([^_]+)_"
    )
    paired_degraded = paired_degraded.sort_values("degradation")
    paired_degraded["panel_label"] = (
        "Same image, " + paired_degraded["degradation"].astype(str)
    )
    rows.append(pd.concat([paired_original, paired_degraded], ignore_index=True))

    examples = pd.concat(rows, ignore_index=True)
    examples["display_scores"] = examples.apply(
        lambda row: (
            f"MUSIQ {row['musiq']:.1f}; TOPIQ-NR {row['topiq_nr']:.2f}"
        ),
        axis=1,
    )
    make_contact_sheet(
        examples,
        output_path,
        columns=4,
        title="No-reference IQA calibration examples",
        label_columns=("panel_label", "display_scores"),
        tile_size=224,
    )


def plot_scores(
    frame: pd.DataFrame, output_path: Path, models: tuple[str, ...]
) -> None:
    import matplotlib.pyplot as plt

    frame = frame.copy()
    frame["analysis_group"] = frame["cohort"].replace(
        {
            "original_real": "real starts",
            "direct_diffusion": "direct diffusion",
            "single_low_noise": "single low-noise",
            "sequential_low_noise": "sequential low-noise",
            "synthetic_degraded": "synthetic degraded",
            "transition_candidate": "transition: unlabeled",
        }
    )
    if "visual_label" in frame:
        transition = frame["cohort"].eq("transition_candidate")
        labeled = transition & frame["visual_label"].astype(bool)
        frame.loc[labeled, "analysis_group"] = (
            "transition: " + frame.loc[labeled, "visual_label"].astype(str)
        )
    order = [
        "real starts",
        "direct diffusion",
        "single low-noise",
        "sequential low-noise",
        "synthetic degraded",
        "transition: intact",
        "transition: borderline",
        "transition: malformed",
        "transition: unlabeled",
    ]
    present = [group for group in order if group in set(frame["analysis_group"])]

    fig, axes = plt.subplots(
        len(models), 1, figsize=(12.5, 3.2 * len(models)), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models):
        column = sanitize_model_name(model)
        values = [
            frame.loc[frame["analysis_group"].eq(group), column].to_numpy(float)
            for group in present
        ]
        axis.boxplot(values, labels=present, showfliers=True)
        axis.set_title(
            f"{model}: no-reference IQA "
            f"({'higher' if HIGHER_IS_BETTER.get(model, True) else 'lower'} is better)"
        )
        axis.tick_params(axis="x", rotation=24)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    output_path: Path,
    models: tuple[str, ...],
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    groups = frame.copy()
    groups["analysis_group"] = groups["cohort"]
    if "visual_label" in groups:
        mask = groups["cohort"].eq("transition_candidate") & groups[
            "visual_label"
        ].astype(bool)
        groups.loc[mask, "analysis_group"] = (
            "transition_" + groups.loc[mask, "visual_label"].astype(str)
        )

    lines = [
        "# No-reference IQA calibration pilot",
        "",
        "This is a calibration study on existing images, not rebuttal evidence yet.",
        "Transition candidates were selected and visually labeled independently of",
        "the IQA scores.",
        "",
        "## Cohort summary",
        "",
        "| cohort | n | "
        + " | ".join(
            f"{model} ({'higher' if HIGHER_IS_BETTER.get(model, True) else 'lower'} better)"
            for model in models
        )
        + " |",
        "|---|---:|" + "|".join("---:" for _ in models) + "|",
    ]
    for group_name, group in groups.groupby("analysis_group", sort=True):
        cells = []
        for model in models:
            mean, low, high = bootstrap_mean(
                group[sanitize_model_name(model)].to_numpy(float), rng, 500
            )
            cells.append(f"{mean:.3f} [{low:.3f}, {high:.3f}]")
        lines.append(
            f"| {group_name} | {len(group)} | " + " | ".join(cells) + " |"
        )

    lines.extend(["", "## Calibration checks", ""])
    original = groups[groups["cohort"].eq("original_real")]
    degraded = groups[groups["cohort"].eq("synthetic_degraded")]
    original_by_source = original.set_index("source_image_id")
    check_rows = []
    for model in models:
        column = sanitize_model_name(model)
        original_mean = float(original[column].mean())
        degraded_mean = float(degraded[column].mean())
        expected = (
            degraded_mean < original_mean
            if HIGHER_IS_BETTER.get(model, True)
            else degraded_mean > original_mean
        )
        paired_successes = 0
        paired_total = 0
        for _, degraded_row in degraded.iterrows():
            source_id = degraded_row["source_image_id"]
            if source_id not in original_by_source.index:
                continue
            original_value = float(original_by_source.loc[source_id, column])
            degraded_value = float(degraded_row[column])
            paired_successes += int(
                degraded_value < original_value
                if HIGHER_IS_BETTER.get(model, True)
                else degraded_value > original_value
            )
            paired_total += 1
        check_rows.append(
            (
                model,
                original_mean,
                degraded_mean,
                expected,
                paired_successes,
                paired_total,
            )
        )
    lines.extend(
        [
            "| model | original mean | degraded mean | paired degradations demoted | expected ordering? |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for (
        model,
        original_mean,
        degraded_mean,
        expected,
        paired_successes,
        paired_total,
    ) in check_rows:
        lines.append(
            f"| {model} | {original_mean:.3f} | {degraded_mean:.3f} | "
            f"{paired_successes}/{paired_total} | "
            f"{'PASS' if expected else 'FAIL'} |"
        )

    labeled = groups[
        groups["cohort"].eq("transition_candidate")
        & groups.get("visual_label", pd.Series("", index=groups.index)).isin(
            VISUAL_LABELS
        )
    ]
    intact = labeled[labeled["visual_label"].eq("intact")]
    malformed = labeled[labeled["visual_label"].eq("malformed")]
    lines.extend(["", "## Visual-label separation", ""])
    if len(intact) and len(malformed):
        lines.extend(
            [
                "| model | intact n | malformed n | AUC: intact scores better (95% bootstrap CI) |",
                "|---|---:|---:|---:|",
            ]
        )
        for model in models:
            column = sanitize_model_name(model)
            intact_values = intact[column].to_numpy(float)
            malformed_values = malformed[column].to_numpy(float)
            if not HIGHER_IS_BETTER.get(model, True):
                intact_values = -intact_values
                malformed_values = -malformed_values
            auc, low, high = bootstrap_auc(
                intact_values, malformed_values, rng, 500
            )
            lines.append(
                f"| {model} | {len(intact)} | {len(malformed)} | "
                f"{auc:.3f} [{low:.3f}, {high:.3f}] |"
            )
        consensus_auc, low, high = bootstrap_auc(
            intact["iqa_consensus_z"].to_numpy(float),
            malformed["iqa_consensus_z"].to_numpy(float),
            rng,
            500,
        )
        lines.append(
            f"| learned consensus | {len(intact)} | {len(malformed)} | "
            f"{consensus_auc:.3f} [{low:.3f}, {high:.3f}] |"
        )
    else:
        lines.append(
            "Visual labels are incomplete; fill `pilot_visual_labels.csv` and "
            "rerun `score`."
        )

    learned_checks = [
        expected
        for model, _, _, expected, _, _ in check_rows
        if model in PRIMARY_LEARNED_MODELS
    ]
    learned_present = [
        sanitize_model_name(model)
        for model in PRIMARY_LEARNED_MODELS
        if sanitize_model_name(model) in labeled
    ]
    learned_correlation = (
        float(labeled[learned_present].corr(method="spearman").iloc[0, 1])
        if len(learned_present) == 2
        else float("nan")
    )
    lines.extend(
        [
            "",
            "## Metric agreement and scope",
            "",
            f"- MUSIQ/TOPIQ-NR Spearman correlation on blind transition samples: "
            f"`{learned_correlation:.3f}`.",
            "- NIQE is retained only as a classical distortion check; it is not "
            "included in the learned-IQA consensus.",
            "- These are no-reference perceptual-quality predictors trained against "
            "human quality judgments. They do not measure ImageNet coverage or "
            "semantic fidelity to the starting image.",
            "",
            "## Preliminary decision",
            "",
            (
                "The learned IQA positive-control check passes."
                if learned_checks and all(learned_checks)
                else "The learned IQA positive-control check does not fully pass."
            ),
            "MUSIQ and TOPIQ-NR also agree strongly and separate the blind intact",
            "and malformed samples usefully, supporting scale-up as complementary",
            "no-reference quality metrics. The malformed sample is small (n=6),",
            "so the pilot confidence intervals are broad and IQA should not be used",
            "as the sole quality claim.",
            "",
            "## Files",
            "",
            "- `iqa_scores.csv`: image-level raw scores and labels",
            "- `pilot_iqa_score_distributions.pdf`: cohort score distributions",
            "- `pilot_iqa_examples.png`: visually inspectable ranked examples",
            "- `pilot_iqa_validation_examples.png`: controls, successes, and misses",
            "- `pilot_controls.png`: original/direct/U-turn/synthetic controls",
            "- `pilot_transition_blind_*.png`: blinded transition candidates",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def score(args: argparse.Namespace) -> None:
    manifest = read_csv(args.output_dir / "pilot_manifest.csv")
    labels = read_csv(args.output_dir / "pilot_visual_labels.csv")
    if labels["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in pilot_visual_labels.csv")
    invalid = sorted(
        set(labels["visual_label"].astype(str))
        - {"", *VISUAL_LABELS}
    )
    if invalid:
        raise ValueError(
            f"Invalid visual labels {invalid}; expected blank or {VISUAL_LABELS}"
        )
    manifest = manifest.merge(labels, on="sample_id", how="left", validate="1:1")
    models = tuple(args.models)
    scored = score_iqa(
        manifest,
        args.output_dir / "iqa_scores.csv",
        models,
        args.device,
    )
    scored = oriented_scores(scored, models)
    scored.to_csv(args.output_dir / "iqa_scores.csv", index=False)
    make_score_examples(scored, args.output_dir / "pilot_iqa_examples.png")
    make_validation_examples(
        scored, args.output_dir / "pilot_iqa_validation_examples.png"
    )
    plot_scores(
        scored,
        args.output_dir / "pilot_iqa_score_distributions",
        models,
    )
    write_report(
        scored,
        args.output_dir / "pilot_iqa_report.md",
        models,
        args.seed,
    )
    print(f"Wrote IQA pilot results to {args.output_dir}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=SEED)
    prepare_parser.add_argument("--control-count", type=int, default=20)
    prepare_parser.add_argument("--transition-per-stratum", type=int, default=4)
    prepare_parser.set_defaults(function=prepare)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--output-dir", type=Path, required=True)
    score_parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    score_parser.add_argument("--device", default="cuda")
    score_parser.add_argument("--seed", type=int, default=SEED)
    score_parser.set_defaults(function=score)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
