import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path


def read_image_names(path: Path) -> set[str]:
    with path.open("r") as f:
        return {Path(line.strip()).stem for line in f if line.strip()}


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


def choose_best(meta_runs_by_image, dog_runs_by_combo):
    candidates = []
    for (image_name, orig_idx, target_idx), dog_runs in dog_runs_by_combo.items():
        meta_runs = meta_runs_by_image.get(image_name, [])
        if not meta_runs:
            continue
        candidates.append(
            {
                "image_name": image_name,
                "orig_class_idx": orig_idx,
                "target_class_idx": target_idx,
                "meta_count": len(meta_runs),
                "dog_count": len(dog_runs),
                "score": min(len(meta_runs), len(dog_runs)),
                "total": len(meta_runs) + len(dog_runs),
            }
        )
    candidates.sort(key=lambda row: (row["score"], row["total"]), reverse=True)
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-image-list", required=True)
    parser.add_argument("--meta-root", default="/work/pcsl/Noam/sequential_diffusion/results/steering_meta_v2_multi")
    parser.add_argument("--dog-root", default="/work/pcsl/Noam/sequential_diffusion/results/steering_dog2dog_v1_multi")
    parser.add_argument("--target-total", type=int, default=20)
    parser.add_argument("--require-repeat-index", action="store_true")
    parser.add_argument("--format", choices=["text", "json", "shell"], default="text")
    args = parser.parse_args()

    active_images = read_image_names(Path(args.active_image_list))
    meta_runs_by_image = collect_meta_runs(Path(args.meta_root), active_images, args.require_repeat_index)
    dog_runs_by_combo = collect_dog_runs(Path(args.dog_root), active_images, args.require_repeat_index)
    candidates = choose_best(meta_runs_by_image, dog_runs_by_combo)
    if not candidates:
        raise SystemExit("No valid image/source/target combination found.")

    best = candidates[0]
    best["meta_extra_needed"] = max(0, args.target_total - best["meta_count"])
    best["dog_extra_needed"] = max(0, args.target_total - best["dog_count"])
    best["image_path"] = f"/work/pcsl/Noam/diffusion_datasets/selected_images/{best['image_name']}.JPEG"

    if args.format == "json":
        print(json.dumps(best, indent=2))
        return
    if args.format == "shell":
        for key in [
            "image_name",
            "image_path",
            "orig_class_idx",
            "target_class_idx",
            "meta_count",
            "dog_count",
            "meta_extra_needed",
            "dog_extra_needed",
        ]:
            print(f"{key.upper()}={best[key]}")
        return

    print("Best clean steering example:")
    for key in [
        "image_name",
        "image_path",
        "orig_class_idx",
        "target_class_idx",
        "meta_count",
        "dog_count",
        "meta_extra_needed",
        "dog_extra_needed",
    ]:
        print(f"{key}: {best[key]}")


if __name__ == "__main__":
    main()
