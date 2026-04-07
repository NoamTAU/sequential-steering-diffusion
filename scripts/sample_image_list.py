import argparse
import random
from pathlib import Path


def read_lines(path: Path) -> list[str]:
    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image-list txt file")
    parser.add_argument("--output", required=True, help="Output sampled txt file")
    parser.add_argument("--num-images", type=int, required=True, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    args = parser.parse_args()

    rows = read_lines(Path(args.input))
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.num_images > len(rows):
        raise ValueError(f"Requested {args.num_images} images but only {len(rows)} are available")

    rng = random.Random(args.seed)
    sample = sorted(rng.sample(rows, args.num_images))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in sample:
            f.write(row + "\n")

    print(f"Sampled {len(sample)} / {len(rows)} images from {args.input}")
    print(f"Seed: {args.seed}")
    print(f"Wrote sample to {out_path}")


if __name__ == "__main__":
    main()
