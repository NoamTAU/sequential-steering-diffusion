#!/usr/bin/env python3
"""Audit image rebuttal manifests, cached metrics, and source images."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional


EXPECTED_ROWS = {
    "single": 13_640,
    "sequential": 12_000,
    "direct": 200,
    "reference": 50_000,
}
FEATURE_FILES = (
    "inception_features.npy",
    "pixel_rmse.npy",
    "pixel_l1.npy",
    "convnext_max_probability.npy",
    "convnext_entropy.npy",
)


class Audit:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.rows.append((name, "PASS" if condition else "FAIL", detail))

    def warn(self, name: str, detail: str) -> None:
        self.rows.append((name, "WARN", detail))

    @property
    def failed(self) -> bool:
        return any(status == "FAIL" for _, status, _ in self.rows)


def finite_array(array: np.ndarray, chunk_size: int = 2048) -> bool:
    return all(
        np.isfinite(np.asarray(array[start : start + chunk_size])).all()
        for start in range(0, len(array), chunk_size)
    )


def duplicate_feature_rows(array: np.ndarray) -> int:
    hashes = set()
    for row in array:
        digest = hashlib.blake2b(
            np.ascontiguousarray(row).view(np.uint8), digest_size=12
        ).digest()
        hashes.add(digest)
    return len(array) - len(hashes)


def resized_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image = transform_functional.resize(
            image,
            [256, 256],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return transform_functional.pil_to_tensor(image).float().div_(255)


def audit_manifest(
    audit: Audit,
    name: str,
    frame: pd.DataFrame,
    expected_rows: int,
    require_start: bool,
) -> None:
    audit.check(
        f"{name}: manifest row count",
        len(frame) == expected_rows,
        f"observed {len(frame):,}; expected {expected_rows:,}",
    )
    image_paths = frame["image_path"].astype(str)
    missing_images = sum(not Path(path).is_file() for path in image_paths)
    audit.check(
        f"{name}: output files",
        missing_images == 0,
        f"{missing_images:,} missing of {len(frame):,}",
    )
    duplicate_paths = int(image_paths.duplicated().sum())
    audit.check(
        f"{name}: unique output paths",
        duplicate_paths == 0,
        f"{duplicate_paths:,} duplicated paths",
    )
    if require_start:
        start_paths = frame["start_path"].astype(str)
        missing_starts = sum(not Path(path).is_file() for path in start_paths.unique())
        audit.check(
            f"{name}: start files",
            missing_starts == 0,
            f"{missing_starts:,} missing unique start paths",
        )
    if "npz_path" in frame:
        npz_paths = frame["npz_path"].astype(str).unique()
        missing_npz = sum(not Path(path).is_file() for path in npz_paths)
        audit.check(
            f"{name}: trajectory caches",
            missing_npz == 0,
            f"{missing_npz:,} missing of {len(npz_paths):,} unique NPZ files",
        )


def audit_features(
    audit: Audit,
    name: str,
    feature_dir: Path,
    expected_rows: int,
    require_pixel_finite: bool,
) -> None:
    for filename in FEATURE_FILES:
        path = feature_dir / filename
        audit.check(f"{name}: {filename} exists", path.is_file(), str(path))
        if not path.is_file():
            continue
        array = np.load(path, mmap_mode="r")
        audit.check(
            f"{name}: {filename} row alignment",
            len(array) == expected_rows,
            f"shape={array.shape}",
        )
        if filename.startswith("pixel_") and not require_pixel_finite:
            all_nan = bool(np.isnan(np.asarray(array)).all())
            audit.check(
                f"{name}: {filename} direct/reference convention",
                all_nan,
                "all values should be NaN when no start image is defined",
            )
        else:
            audit.check(
                f"{name}: {filename} finite",
                finite_array(array),
                f"shape={array.shape}, dtype={array.dtype}",
            )

    inception_path = feature_dir / "inception_features.npy"
    if inception_path.is_file():
        features = np.load(inception_path, mmap_mode="r")
        duplicate_rows = duplicate_feature_rows(features)
        detail = f"{duplicate_rows:,} exact duplicate rows of {len(features):,}"
        if duplicate_rows:
            audit.warn(f"{name}: exact duplicate Inception rows", detail)
        else:
            audit.check(f"{name}: exact duplicate Inception rows", True, detail)


def audit_pixel_distances(
    audit: Audit,
    name: str,
    frame: pd.DataFrame,
    feature_dir: Path,
    seed: int,
    sample_size: int = 20,
) -> None:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(frame), size=min(sample_size, len(frame)), replace=False)
    stored_rmse = np.load(feature_dir / "pixel_rmse.npy", mmap_mode="r")
    stored_l1 = np.load(feature_dir / "pixel_l1.npy", mmap_mode="r")
    rmse_errors = []
    l1_errors = []
    for index in indices:
        image = resized_tensor(frame.iloc[index]["image_path"])
        start = resized_tensor(frame.iloc[index]["start_path"])
        difference = image - start
        observed_rmse = float(difference.square().mean().sqrt())
        observed_l1 = float(difference.abs().mean())
        rmse_errors.append(abs(observed_rmse - float(stored_rmse[index])))
        l1_errors.append(abs(observed_l1 - float(stored_l1[index])))
    max_rmse_error = max(rmse_errors, default=float("inf"))
    max_l1_error = max(l1_errors, default=float("inf"))
    audit.check(
        f"{name}: recomputed pixel distances",
        max(max_rmse_error, max_l1_error) < 2e-6,
        (
            f"{len(indices)} sampled pairs; max RMSE error={max_rmse_error:.2e}, "
            f"max L1 error={max_l1_error:.2e}"
        ),
    )


def audit_random_images(
    audit: Audit,
    name: str,
    frame: pd.DataFrame,
    seed: int,
    sample_size: int = 100,
) -> None:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(frame), size=min(sample_size, len(frame)), replace=False)
    failures = 0
    dimensions = set()
    for index in indices:
        try:
            with Image.open(frame.iloc[index]["image_path"]) as handle:
                dimensions.add(handle.size)
                handle.verify()
        except Exception:
            failures += 1
    audit.check(
        f"{name}: sampled image integrity",
        failures == 0,
        f"{failures} corrupt of {len(indices)} sampled; dimensions={sorted(dimensions)}",
    )


def audit_direct_hashes(audit: Audit, frame: pd.DataFrame) -> None:
    hashes = []
    for path in frame["image_path"].astype(str):
        hashes.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    audit.check(
        "direct: distinct JPEG bytes",
        len(set(hashes)) == len(hashes),
        f"{len(set(hashes)):,} unique hashes of {len(hashes):,}",
    )


def audit_start_image_consistency(
    audit: Audit, name: str, frame: pd.DataFrame, paths_per_image: int = 5
) -> None:
    inconsistent = 0
    checked = 0
    for _, group in frame.groupby("image_id", sort=False):
        paths = group["start_path"].drop_duplicates().head(paths_per_image)
        hashes = []
        for path in paths:
            with Image.open(path) as handle:
                pixels = np.asarray(handle.convert("RGB"))
            hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
        inconsistent += int(len(set(hashes)) > 1)
        checked += 1
    audit.check(
        f"{name}: starting image consistent across runs",
        inconsistent == 0,
        (
            f"{inconsistent} inconsistent image IDs of {checked}; "
            f"checked up to {paths_per_image} start files per image"
        ),
    )


def audit_paired_features(
    audit: Audit, name: str, feature_dir: Path, expected_rows: int
) -> None:
    specifications = {
        "convnext_feature_cosine_distance.npy": (0.0, 2.0),
        "convnext_start_class_probability.npy": (0.0, 1.0),
        "convnext_start_class_retained.npy": (0.0, 1.0),
        "convnext_start_class.npy": (0.0, 999.0),
    }
    for filename, (minimum, maximum) in specifications.items():
        path = feature_dir / filename
        audit.check(f"{name}: {filename} exists", path.is_file(), str(path))
        if not path.is_file():
            continue
        array = np.load(path, mmap_mode="r")
        valid = (
            len(array) == expected_rows
            and finite_array(array)
            and float(array.min()) >= minimum
            and float(array.max()) <= maximum
        )
        audit.check(
            f"{name}: {filename} valid",
            valid,
            (
                f"shape={array.shape}; range=[{float(array.min()):.6f}, "
                f"{float(array.max()):.6f}]"
            ),
        )


def audit_manifold_features(
    audit: Audit, name: str, feature_dir: Path, expected_rows: int
) -> None:
    specifications = {
        "manifold_precision_k3.npy": (0.0, 1.0),
        "manifold_density_k3.npy": (0.0, float("inf")),
        "manifold_nearest_distance.npy": (0.0, float("inf")),
        "manifold_realism_score_k3.npy": (0.0, float("inf")),
    }
    for filename, (minimum, maximum) in specifications.items():
        path = feature_dir / filename
        audit.check(f"{name}: {filename} exists", path.is_file(), str(path))
        if not path.is_file():
            continue
        array = np.load(path, mmap_mode="r")
        valid = (
            len(array) == expected_rows
            and finite_array(array)
            and float(array.min()) >= minimum
            and float(array.max()) <= maximum
        )
        audit.check(
            f"{name}: {filename} valid",
            valid,
            (
                f"shape={array.shape}; range=[{float(array.min()):.6f}, "
                f"{float(array.max()):.6f}]"
            ),
        )


def audit_shared_source(audit: Audit, root: Path) -> None:
    manifest_path = root / "image_quality/manifests/shared_start_manifest.csv"
    feature_path = root / "image_quality/features/shared_start/inception_features.npy"
    audit.check("shared source: manifest exists", manifest_path.is_file(), str(manifest_path))
    audit.check("shared source: features exist", feature_path.is_file(), str(feature_path))
    if not manifest_path.is_file() or not feature_path.is_file():
        return
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    features = np.load(feature_path, mmap_mode="r")
    files_exist = frame["image_path"].map(lambda path: Path(path).is_file()).all()
    audit.check(
        "shared source: exact 20-image cohort",
        len(frame) == 20
        and frame["image_id"].nunique() == 20
        and files_exist
        and features.shape == (20, 2048)
        and finite_array(features),
        (
            f"manifest rows={len(frame)}, IDs={frame['image_id'].nunique()}, "
            f"feature shape={features.shape}"
        ),
    )

    summaries = {
        "single": (
            root / "image_quality/results/single_uturn_shared20_quality_summary.csv",
            11,
            800,
        ),
        "sequential": (
            root
            / "image_quality/results/sequential_uturn_shared20_quality_summary.csv",
            60,
            200,
        ),
    }
    source_columns = [
        "source_kid",
        "source_kid_ci_low",
        "source_kid_ci_high",
        "source_fid_200",
        "source_fid_mean_term",
        "source_fid_covariance_term",
    ]
    paired_columns = [
        "convnext_feature_cosine_distance",
        "convnext_start_class_probability",
        "convnext_start_class_retained",
        "manifold_precision_k3",
        "manifold_density_k3",
        "manifold_nearest_distance",
    ]
    for name, (path, expected_rows, expected_samples) in summaries.items():
        audit.check(f"shared {name}: summary exists", path.is_file(), str(path))
        if not path.is_file():
            continue
        summary = pd.read_csv(path)
        required = source_columns + paired_columns
        valid = (
            len(summary) == expected_rows
            and summary["samples"].eq(expected_samples).all()
            and set(required).issubset(summary.columns)
            and np.isfinite(summary[required].to_numpy(dtype=float)).all()
        )
        audit.check(
            f"shared {name}: source and paired metrics valid",
            bool(valid),
            (
                f"rows={len(summary)}, samples={sorted(summary['samples'].unique())}, "
                f"source FID range=[{summary.get('source_fid_200', pd.Series([np.nan])).min():.3f}, "
                f"{summary.get('source_fid_200', pd.Series([np.nan])).max():.3f}]"
            ),
        )


def audit_metric_summaries(audit: Audit, results_dir: Path) -> None:
    specifications = {
        "single": ("single_uturn_quality_summary.csv", 11, 1_240),
        "sequential": ("sequential_uturn_quality_summary.csv", 60, 200),
        "direct": ("direct_diffusion_quality_summary.csv", 1, 200),
    }
    for name, (filename, expected_rows, expected_samples) in specifications.items():
        path = results_dir / filename
        audit.check(f"{name}: summary exists", path.is_file(), str(path))
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        audit.check(
            f"{name}: summary dimensions",
            len(frame) == expected_rows
            and frame["samples"].eq(expected_samples).all(),
            (
                f"rows={len(frame)} (expected {expected_rows}); "
                f"sample counts={sorted(frame['samples'].unique())}"
            ),
        )
        finite = np.isfinite(
            frame[["kid", "kid_ci_low", "kid_ci_high", "fid_matched_200"]]
            .to_numpy(dtype=float)
        ).all()
        audit.check(
            f"{name}: KID/FID finite",
            bool(finite),
            (
                f"KID range [{frame['kid'].min():.5f}, {frame['kid'].max():.5f}], "
                f"FID range [{frame['fid_matched_200'].min():.2f}, "
                f"{frame['fid_matched_200'].max():.2f}]"
            ),
        )


def audit_clip_paths(audit: Audit, results_dir: Path, feature_dir: Path) -> None:
    single_clip = np.load(
        feature_dir / "single/clip_net_distance.npy", mmap_mode="r"
    )
    audit.check(
        "single: CLIP-patch change valid",
        len(single_clip) == EXPECTED_ROWS["single"]
        and finite_array(single_clip)
        and bool((single_clip >= 0).all()),
        (
            f"shape={single_clip.shape}, range=[{single_clip.min():.6f}, "
            f"{single_clip.max():.6f}]"
        ),
    )

    path = results_dir / "sequential_sample_metrics.csv"
    frame = pd.read_csv(path)
    columns = ["clip_net_distance", "clip_cumulative_path"]
    finite_nonnegative = np.isfinite(frame[columns].to_numpy()).all() and bool(
        (frame[columns] >= -1e-8).all().all()
    )
    audit.check(
        "sequential: CLIP-patch metrics valid",
        len(frame) == EXPECTED_ROWS["sequential"] and finite_nonnegative,
        f"rows={len(frame):,}",
    )
    net_path = feature_dir / "sequential/clip_net_distance_from_images.npy"
    cumulative_path = (
        feature_dir / "sequential/clip_cumulative_path_from_images.npy"
    )
    image_derived_valid = net_path.is_file() and cumulative_path.is_file()
    if image_derived_valid:
        net = np.load(net_path, mmap_mode="r")
        cumulative = np.load(cumulative_path, mmap_mode="r")
        image_derived_valid = (
            len(net) == len(frame)
            and len(cumulative) == len(frame)
            and finite_array(net)
            and finite_array(cumulative)
            and np.allclose(frame["clip_net_distance"], net, atol=1e-7)
            and np.allclose(
                frame["clip_cumulative_path"], cumulative, atol=1e-7
            )
        )
    audit.check(
        "sequential: image-derived CLIP cache alignment",
        bool(image_derived_valid),
        (
            f"net cache={net_path.exists()}, cumulative cache={cumulative_path.exists()}"
        ),
    )

    trajectory_failures = 0
    grouping = ["image_id", "noise_step", "trajectory_id"]
    for _, group in frame.groupby(grouping, sort=False):
        values = group.sort_values("step")["clip_cumulative_path"].to_numpy()
        trajectory_failures += int(np.any(np.diff(values) < -1e-7))
    audit.check(
        "sequential: cumulative paths monotone per trajectory",
        trajectory_failures == 0,
        f"{trajectory_failures} non-monotone trajectories",
    )

    means = (
        frame.groupby(["rho", "step"], as_index=False)["clip_cumulative_path"]
        .mean()
        .sort_values(["rho", "step"])
    )
    mean_failures = sum(
        np.any(np.diff(group["clip_cumulative_path"].to_numpy()) < -1e-7)
        for _, group in means.groupby("rho")
    )
    audit.check(
        "sequential: mean cumulative paths monotone",
        mean_failures == 0,
        f"{mean_failures} rho curves non-monotone",
    )


def audit_mh_steering(audit: Audit, root: Path) -> None:
    summary_path = root / "mh_steering/summary/mh_image_steering_summary.csv"
    audit.check("MH steering: summary exists", summary_path.is_file(), str(summary_path))
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        focused = summary[
            summary["target_mode"].eq("dog_class")
            & summary["rho"].eq(0.4)
            & summary["energy_lambda"].isin([0.0, 4.0])
        ]
        valid = len(focused) == 2 and focused["runs"].eq(12).all()
        audit.check(
            "MH steering: focused summary counts",
            bool(valid),
            (
                focused[
                    ["energy_lambda", "runs", "final_target_probability"]
                ].to_dict("records")
            ),
        )

    run_counts = {}
    finite_failures = 0
    image_failures = 0
    for energy_lambda in (0, 4):
        pattern = f"*/mode_dog_class_rho_400_lambda_{energy_lambda}_rep_*"
        run_dirs = sorted((root / "mh_steering").glob(pattern))
        run_counts[energy_lambda] = len(run_dirs)
        for run_dir in run_dirs:
            data_path = run_dir / "mh_data.npz"
            image_path = run_dir / "step_050.jpeg"
            if not image_path.is_file():
                image_failures += 1
            if not data_path.is_file():
                finite_failures += 1
                continue
            with np.load(data_path) as data:
                for key in (
                    "target_probability",
                    "source_probability",
                    "max_probability",
                    "accepted",
                ):
                    finite_failures += int(
                        key not in data or not np.isfinite(data[key]).all()
                    )
    audit.check(
        "MH steering: focused raw runs",
        run_counts == {0: 12, 4: 12}
        and finite_failures == 0
        and image_failures == 0,
        (
            f"run counts={run_counts}; invalid arrays={finite_failures}; "
            f"missing terminal images={image_failures}"
        ),
    )


def write_report(audit: Audit, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Image rebuttal result audit",
        "",
        "This audit checks manifest-to-feature alignment, cached values, source and "
        "output files, selected recomputations, and steering-run completeness.",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for name, status, detail in audit.rows:
        escaped = str(detail).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {name} | **{status}** | {escaped} |")
    lines.extend(
        [
            "",
            f"**Overall:** {'FAIL' if audit.failed else 'PASS'}",
            "",
            "Warnings are reported for inspection but do not invalidate row alignment "
            "or metric computation.",
        ]
    )
    (output_dir / "image_rebuttal_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    pd.DataFrame(audit.rows, columns=["check", "status", "detail"]).to_csv(
        output_dir / "image_rebuttal_audit.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_dir = args.root / "image_quality/manifests"
    feature_root = args.root / "image_quality/features"
    results_dir = args.root / "image_quality/results"
    frames = {
        "single": pd.read_csv(
            manifest_dir / "single_manifest.csv", keep_default_na=False
        ),
        "sequential": pd.read_csv(
            manifest_dir / "sequential_manifest.csv", keep_default_na=False
        ),
        "direct": pd.read_csv(
            args.root / "direct_diffusion/direct_manifest.csv",
            keep_default_na=False,
        ),
    }

    audit = Audit()
    audit_manifest(
        audit, "single", frames["single"], EXPECTED_ROWS["single"], require_start=True
    )
    audit_manifest(
        audit,
        "sequential",
        frames["sequential"],
        EXPECTED_ROWS["sequential"],
        require_start=True,
    )
    audit_manifest(
        audit, "direct", frames["direct"], EXPECTED_ROWS["direct"], require_start=False
    )

    audit_features(
        audit,
        "single",
        feature_root / "single",
        EXPECTED_ROWS["single"],
        require_pixel_finite=True,
    )
    audit_features(
        audit,
        "sequential",
        feature_root / "sequential",
        EXPECTED_ROWS["sequential"],
        require_pixel_finite=True,
    )
    audit_features(
        audit,
        "direct",
        args.root / "direct_diffusion/features",
        EXPECTED_ROWS["direct"],
        require_pixel_finite=False,
    )

    reference_path = feature_root / "reference/inception_features.npy"
    reference = np.load(reference_path, mmap_mode="r")
    audit.check(
        "reference: Inception feature shape and finiteness",
        reference.shape == (EXPECTED_ROWS["reference"], 2048)
        and finite_array(reference),
        f"shape={reference.shape}, dtype={reference.dtype}",
    )

    audit_pixel_distances(
        audit, "single", frames["single"], feature_root / "single", seed=44
    )
    audit_pixel_distances(
        audit, "sequential", frames["sequential"], feature_root / "sequential", seed=45
    )
    audit_start_image_consistency(audit, "single", frames["single"])
    audit_start_image_consistency(audit, "sequential", frames["sequential"])
    audit_paired_features(
        audit,
        "single",
        feature_root / "single",
        EXPECTED_ROWS["single"],
    )
    audit_paired_features(
        audit,
        "sequential",
        feature_root / "sequential",
        EXPECTED_ROWS["sequential"],
    )
    audit_manifold_features(
        audit,
        "single",
        feature_root / "single",
        EXPECTED_ROWS["single"],
    )
    audit_manifold_features(
        audit,
        "sequential",
        feature_root / "sequential",
        EXPECTED_ROWS["sequential"],
    )
    audit_manifold_features(
        audit,
        "direct",
        args.root / "direct_diffusion/features",
        EXPECTED_ROWS["direct"],
    )
    for offset, name in enumerate(("single", "sequential", "direct")):
        audit_random_images(audit, name, frames[name], seed=46 + offset)
    audit_direct_hashes(audit, frames["direct"])
    audit_metric_summaries(audit, results_dir)
    audit_shared_source(audit, args.root)
    audit_clip_paths(audit, results_dir, feature_root)
    audit_mh_steering(audit, args.root)
    write_report(audit, args.output_dir)
    print(f"overall={'FAIL' if audit.failed else 'PASS'}")
    for name, status, detail in audit.rows:
        if status != "PASS":
            print(f"{status:4s} {name}: {detail}")
    raise SystemExit(1 if audit.failed else 0)


if __name__ == "__main__":
    main()
