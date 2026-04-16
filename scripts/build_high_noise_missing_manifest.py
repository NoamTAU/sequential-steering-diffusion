import argparse
import csv
from pathlib import Path


def read_image_rows(path):
    rows = []
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            p = Path(line)
            rows.append({"image_name": p.stem, "image_path": str(p)})
    return rows


def trajectory_complete(traj_dir, expected_uturns):
    start_file = traj_dir / "uturn_000.jpeg"
    final_file = traj_dir / f"uturn_{expected_uturns:03d}.jpeg"
    if not start_file.exists():
        start_file = traj_dir / "uturn_000.png"
    if not final_file.exists():
        final_file = traj_dir / f"uturn_{expected_uturns:03d}.png"
    return start_file.exists() and final_file.exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--noise-steps", nargs="+", type=int, required=True)
    parser.add_argument("--expected-trajectories", type=int, required=True)
    parser.add_argument("--expected-uturns", type=int, required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    image_rows = read_image_rows(args.image_list)
    results_root = Path(args.results_root)
    manifest_rows = []

    for image_row in image_rows:
        image_name = image_row["image_name"]
        image_path = image_row["image_path"]
        for noise in args.noise_steps:
            noise_dir = results_root / image_name / f"noise_step_{noise}"
            traj_dirs = sorted(p for p in noise_dir.glob("trajectory_*") if p.is_dir())
            complete_count = sum(1 for traj_dir in traj_dirs if trajectory_complete(traj_dir, args.expected_uturns))
            if complete_count >= args.expected_trajectories:
                continue
            reason = "missing_noise_dir" if not noise_dir.exists() else "incomplete_pair"
            manifest_rows.append(
                {
                    "image_name": image_name,
                    "image_path": image_path,
                    "noise_step": noise,
                    "reason": reason,
                    "trajectories_found": len(traj_dirs),
                    "complete_trajectories": complete_count,
                }
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "image_path",
                "noise_step",
                "reason",
                "trajectories_found",
                "complete_trajectories",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} missing/incomplete pairs to {out_path}")


if __name__ == "__main__":
    main()
