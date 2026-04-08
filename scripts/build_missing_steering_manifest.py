import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_image_paths(path: Path):
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(Path(line))
    return rows


def read_rows(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def normalize_repeat(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def is_truthy(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def collect_meta_repeats(rows, active_images):
    out = defaultdict(set)
    for row in rows:
        image_name = row.get("image_name", "").strip()
        if image_name not in active_images:
            continue
        if not is_truthy(row.get("is_top1_dog", "")):
            continue
        repeat_index = normalize_repeat(row.get("repeat_index"))
        if repeat_index is None:
            continue
        out[image_name].add(repeat_index)
    return out


def collect_dog_repeats(rows, active_images):
    out = defaultdict(set)
    for row in rows:
        image_name = row.get("image_name", "").strip()
        if image_name not in active_images:
            continue
        if not is_truthy(row.get("is_top1_dog", "")):
            continue
        repeat_index = normalize_repeat(row.get("repeat_index"))
        if repeat_index is None:
            continue
        try:
            top1 = int(str(row.get("top1_class_idx", "")).strip())
            orig = int(str(row.get("orig_class_idx", "")).strip())
            target = int(str(row.get("target_class_idx", "")).strip())
        except ValueError:
            continue
        if orig != top1 or orig == target:
            continue
        out[image_name].add(repeat_index)
    return out


def build_manifest(image_paths, existing_repeats, expected_repeats, base_seed):
    rows = []
    for image_index, image_path in enumerate(image_paths):
        image_name = image_path.stem
        seen = existing_repeats.get(image_name, set())
        for repeat_index in range(expected_repeats):
            if repeat_index in seen:
                continue
            seed = base_seed + image_index * 1000 + repeat_index
            rows.append(
                {
                    "image_path": str(image_path),
                    "image_name": image_name,
                    "image_index": image_index,
                    "repeat_index": repeat_index,
                    "seed": seed,
                }
            )
    return rows


def write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "image_name", "image_index", "repeat_index", "seed"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-image-list", required=True)
    parser.add_argument("--meta-csv", required=True)
    parser.add_argument("--dog-csv", required=True)
    parser.add_argument("--expected-repeats", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20260407)
    parser.add_argument("--meta-out", required=True)
    parser.add_argument("--dog-out", required=True)
    args = parser.parse_args()

    image_paths = read_image_paths(Path(args.active_image_list))
    active_images = {p.stem for p in image_paths}
    meta_rows = read_rows(Path(args.meta_csv))
    dog_rows = read_rows(Path(args.dog_csv))

    meta_existing = collect_meta_repeats(meta_rows, active_images)
    dog_existing = collect_dog_repeats(dog_rows, active_images)

    meta_manifest = build_manifest(image_paths, meta_existing, args.expected_repeats, args.base_seed)
    dog_manifest = build_manifest(image_paths, dog_existing, args.expected_repeats, args.base_seed)

    write_manifest(Path(args.meta_out), meta_manifest)
    write_manifest(Path(args.dog_out), dog_manifest)

    print(f"Wrote {len(meta_manifest)} missing dog->cat tasks to {args.meta_out}")
    print(f"Wrote {len(dog_manifest)} missing dog->dog tasks to {args.dog_out}")


if __name__ == "__main__":
    main()
