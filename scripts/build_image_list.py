import argparse
import glob
import random
from pathlib import Path


def collect_images(image_dir: Path, pattern: str, recursive: bool) -> list[str]:
    glob_pattern = str(image_dir / ("**" / Path(pattern) if recursive else Path(pattern)))
    return sorted(glob.glob(glob_pattern, recursive=recursive))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Directory containing source images")
    parser.add_argument("--out", required=True, help="Output txt file with one image path per line")
    parser.add_argument("--pattern", default="*.JPEG", help="Glob pattern for image files")
    parser.add_argument("--recursive", action="store_true", help="Search recursively")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap on number of images")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when max-images > 0")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    image_paths = collect_images(image_dir, args.pattern, args.recursive)
    if args.max_images > 0 and len(image_paths) > args.max_images:
        rng = random.Random(args.seed)
        image_paths = sorted(rng.sample(image_paths, args.max_images))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for path in image_paths:
            f.write(path + "\n")

    print(f"Wrote {len(image_paths)} image paths to {out_path}")


if __name__ == "__main__":
    main()
