import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def read_image_names(path: Path) -> set[str]:
    with path.open("r") as f:
        return {Path(line.strip()).stem for line in f if line.strip()}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def normalize_repeat(value: str):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def collect_meta(rows: list[dict], active_images: set[str], require_repeat_index: bool):
    by_image = defaultdict(list)
    for row in rows:
        image_name = row.get("image_name", "").strip()
        if image_name not in active_images:
            continue
        repeat_index = normalize_repeat(row.get("repeat_index"))
        if require_repeat_index and repeat_index is None:
            continue
        if not is_truthy(row.get("is_top1_dog", "")):
            continue
        by_image[image_name].append(row)
    return by_image


def collect_dog(rows: list[dict], active_images: set[str], require_repeat_index: bool):
    by_combo = defaultdict(list)
    for row in rows:
        image_name = row.get("image_name", "").strip()
        if image_name not in active_images:
            continue
        repeat_index = normalize_repeat(row.get("repeat_index"))
        if require_repeat_index and repeat_index is None:
            continue
        if not is_truthy(row.get("is_top1_dog", "")):
            continue
        try:
            top1 = int(str(row.get("top1_class_idx", "")).strip())
            orig = int(str(row.get("orig_class_idx", "")).strip())
            target = int(str(row.get("target_class_idx", "")).strip())
        except ValueError:
            continue
        if orig != top1:
            continue
        if orig == target:
            continue
        by_combo[(image_name, orig, target)].append(row)
    return by_combo


def unique_repeat_count(rows: list[dict]) -> int:
    repeats = {normalize_repeat(row.get("repeat_index")) for row in rows}
    repeats.discard(None)
    if repeats:
        return len(repeats)
    return len(rows)


def find_best_combo(meta_by_image, dog_by_combo):
    best = None
    for (image_name, orig_idx, target_idx), dog_rows in dog_by_combo.items():
        meta_rows = meta_by_image.get(image_name, [])
        if not meta_rows:
            continue
        candidate = {
            "image_name": image_name,
            "orig_class_idx": orig_idx,
            "target_class_idx": target_idx,
            "meta_repeats": unique_repeat_count(meta_rows),
            "dog_repeats": unique_repeat_count(dog_rows),
            "score": min(unique_repeat_count(meta_rows), unique_repeat_count(dog_rows)),
            "total": unique_repeat_count(meta_rows) + unique_repeat_count(dog_rows),
        }
        if best is None or (candidate["score"], candidate["total"]) > (best["score"], best["total"]):
            best = candidate
    return best


def count_images_with_at_least(meta_by_image, threshold: int):
    return sum(unique_repeat_count(rows) >= threshold for rows in meta_by_image.values())


def count_dog_combos_with_at_least(dog_by_combo, threshold: int):
    return sum(unique_repeat_count(rows) >= threshold for rows in dog_by_combo.values())


def count_common_images(meta_by_image, dog_by_combo, threshold: int):
    dog_best_by_image = {}
    for (image_name, orig_idx, target_idx), rows in dog_by_combo.items():
        repeats = unique_repeat_count(rows)
        current = dog_best_by_image.get(image_name)
        candidate = (repeats, orig_idx, target_idx)
        if current is None or candidate > current:
            dog_best_by_image[image_name] = candidate
    total = 0
    for image_name, meta_rows in meta_by_image.items():
        if unique_repeat_count(meta_rows) < threshold:
            continue
        dog_info = dog_best_by_image.get(image_name)
        if dog_info and dog_info[0] >= threshold:
            total += 1
    return total


def histogram(values):
    counts = Counter(values)
    return ", ".join(f"{k}:{counts[k]}" for k in sorted(counts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-image-list", required=True)
    parser.add_argument("--meta-csv", required=True)
    parser.add_argument("--dog-csv", required=True)
    parser.add_argument("--expected-repeats", type=int, default=4)
    parser.add_argument("--require-repeat-index", action="store_true")
    args = parser.parse_args()

    active_images = read_image_names(Path(args.active_image_list))
    meta_rows = read_rows(Path(args.meta_csv))
    dog_rows = read_rows(Path(args.dog_csv))

    meta_by_image = collect_meta(meta_rows, active_images, args.require_repeat_index)
    dog_by_combo = collect_dog(dog_rows, active_images, args.require_repeat_index)

    meta_repeat_counts = [unique_repeat_count(rows) for rows in meta_by_image.values()]
    dog_repeat_counts = [unique_repeat_count(rows) for rows in dog_by_combo.values()]

    best = find_best_combo(meta_by_image, dog_by_combo)

    print(f"Active pilot images: {len(active_images)}")
    print(f"Expected repeats per regime: {args.expected_repeats}")
    print(f"Meta runs (dog->cat) on active images: {sum(len(v) for v in meta_by_image.values())}")
    print(f"Dog runs (dog->dog) on active images: {sum(len(v) for v in dog_by_combo.values())}")
    print(f"Images with >=1 dog->cat run: {len(meta_by_image)}")
    print(f"Image/source/target combos with >=1 dog->dog run: {len(dog_by_combo)}")
    print(
        f"Images with >={args.expected_repeats} dog->cat repeats: "
        f"{count_images_with_at_least(meta_by_image, args.expected_repeats)}"
    )
    print(
        f"Image/source/target combos with >={args.expected_repeats} dog->dog repeats: "
        f"{count_dog_combos_with_at_least(dog_by_combo, args.expected_repeats)}"
    )
    print(
        f"Common images with >={args.expected_repeats} repeats in both regimes: "
        f"{count_common_images(meta_by_image, dog_by_combo, args.expected_repeats)}"
    )
    print(f"dog->cat repeat histogram: {histogram(meta_repeat_counts) if meta_repeat_counts else 'empty'}")
    print(f"dog->dog repeat histogram: {histogram(dog_repeat_counts) if dog_repeat_counts else 'empty'}")

    if best is not None:
        print("Best clean matched example:")
        print(f"  image_name={best['image_name']}")
        print(f"  orig_class_idx={best['orig_class_idx']}")
        print(f"  target_class_idx={best['target_class_idx']}")
        print(f"  meta_repeats={best['meta_repeats']}")
        print(f"  dog_repeats={best['dog_repeats']}")


if __name__ == "__main__":
    main()
