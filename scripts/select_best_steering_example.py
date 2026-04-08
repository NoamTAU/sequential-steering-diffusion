import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path


def read_image_map(path: Path) -> dict[str, str]:
    with path.open("r") as f:
        return {Path(line.strip()).stem: line.strip() for line in f if line.strip()}


def load_image_summary(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        image_name = str(row.get("image_name", "")).strip()
        if image_name:
            out[image_name] = row
    return out


def load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def parse_repeat_index(run_dir: Path):
    m = re.search(r"_rep(\d+)$", run_dir.name)
    if m:
        return int(m.group(1))
    return None


def find_image_name(run_dir: Path):
    for part in run_dir.parts:
        if part.startswith("ILSVRC2012_val_"):
            return part
    return None


def iter_run_dirs(root: Path):
    for steering_path in glob.glob(str(root / "**" / "steering_data.npz"), recursive=True):
        yield Path(steering_path).parent


def collect_meta_runs(root: Path, active_images: set[str], require_repeat_index: bool):
    runs_by_image = defaultdict(list)
    for run_dir in iter_run_dirs(root):
        image_name = find_image_name(run_dir)
        if not image_name or image_name not in active_images:
            continue
        repeat_index = parse_repeat_index(run_dir)
        if require_repeat_index and repeat_index is None:
            continue
        runs_by_image[image_name].append(
            {
                "run_dir": str(run_dir),
                "repeat_index": repeat_index,
            }
        )
    return runs_by_image


def collect_dog_runs(root: Path, active_images: set[str], require_repeat_index: bool):
    runs_by_combo = defaultdict(list)
    for run_dir in iter_run_dirs(root):
        image_name = find_image_name(run_dir)
        if not image_name or image_name not in active_images:
            continue
        repeat_index = parse_repeat_index(run_dir)
        if require_repeat_index and repeat_index is None:
            continue

        auto_info = load_json(run_dir / "auto_target.json")
        start_info = load_json(run_dir / "start_image_info.json")
        orig = auto_info.get("orig_class_idx")
        target = auto_info.get("target_class_idx")
        top1 = start_info.get("top1_class_idx")
        if orig is None or target is None or top1 is None:
            continue
        if orig != top1:
            continue
        if orig == target:
            continue

        key = (image_name, int(orig), int(target))
        runs_by_combo[key].append(
            {
                "run_dir": str(run_dir),
                "repeat_index": repeat_index,
            }
        )
    return runs_by_combo


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def choose_best(meta_runs_by_image, dog_runs_by_combo, image_map, image_summary):
    candidates = []
    for (image_name, orig_idx, target_idx), dog_runs in dog_runs_by_combo.items():
        meta_runs = meta_runs_by_image.get(image_name, [])
        if not meta_runs:
            continue
        summary = image_summary.get(image_name, {})
        candidates.append(
            {
                "image_name": image_name,
                "orig_class_idx": orig_idx,
                "target_class_idx": target_idx,
                "meta_count": len(meta_runs),
                "dog_count": len(dog_runs),
                "score": min(len(meta_runs), len(dog_runs)),
                "total": len(meta_runs) + len(dog_runs),
                "image_path": image_map[image_name],
                "top1_prob": parse_float(summary.get("top1_prob")),
                "top1_class_idx": summary.get("top1_class_idx", ""),
                "true_class_idx": summary.get("true_class_idx", ""),
                "top1_matches_true": summary.get("top1_matches_true", ""),
            }
        )
    candidates.sort(key=lambda row: (row["score"], row["total"], row["top1_prob"]), reverse=True)
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-image-list", required=True)
    parser.add_argument("--meta-root", default="/work/pcsl/Noam/sequential_diffusion/results/steering_meta_v2_multi")
    parser.add_argument("--dog-root", default="/work/pcsl/Noam/sequential_diffusion/results/steering_dog2dog_v1_multi")
    parser.add_argument("--image-summary-csv", default="scripts/dog_image_summary.csv")
    parser.add_argument("--target-total", type=int, default=20)
    parser.add_argument("--require-repeat-index", action="store_true")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--format", choices=["text", "json", "shell"], default="text")
    args = parser.parse_args()

    image_map = read_image_map(Path(args.active_image_list))
    image_summary = load_image_summary(Path(args.image_summary_csv))
    active_images = set(image_map)
    meta_runs_by_image = collect_meta_runs(Path(args.meta_root), active_images, args.require_repeat_index)
    dog_runs_by_combo = collect_dog_runs(Path(args.dog_root), active_images, args.require_repeat_index)
    candidates = choose_best(meta_runs_by_image, dog_runs_by_combo, image_map, image_summary)
    if not candidates:
        raise SystemExit("No valid image/source/target combination found.")

    selected = candidates[: max(1, args.top_k)]
    for row in selected:
        row["meta_extra_needed"] = max(0, args.target_total - row["meta_count"])
        row["dog_extra_needed"] = max(0, args.target_total - row["dog_count"])

    if args.format == "json":
        payload = selected[0] if args.top_k == 1 else selected
        print(json.dumps(payload, indent=2))
        return
    if args.format == "shell":
        best = selected[0]
        for key in [
            "image_name",
            "image_path",
            "orig_class_idx",
            "target_class_idx",
            "top1_prob",
            "top1_class_idx",
            "true_class_idx",
            "top1_matches_true",
            "meta_count",
            "dog_count",
            "meta_extra_needed",
            "dog_extra_needed",
        ]:
            print(f"{key.upper()}={best[key]}")
        return

    header = "Best clean steering example:" if args.top_k == 1 else f"Top {len(selected)} clean steering examples:"
    print(header)
    for idx, row in enumerate(selected, start=1):
        if args.top_k > 1:
            print(f"[{idx}]")
        for key in [
            "image_name",
            "image_path",
            "orig_class_idx",
            "target_class_idx",
            "top1_prob",
            "top1_class_idx",
            "true_class_idx",
            "top1_matches_true",
            "meta_count",
            "dog_count",
            "meta_extra_needed",
            "dog_extra_needed",
        ]:
            print(f"{key}: {row[key]}")
        if idx != len(selected):
            print()


if __name__ == "__main__":
    main()
