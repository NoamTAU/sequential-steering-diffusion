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
            for traj_idx in range(args.expected_trajectories):
                traj_dir = noise_dir / f"trajectory_{traj_idx:03d}"
                if trajectory_complete(traj_dir, args.expected_uturns):
                    continue
                manifest_rows.append(
                    {
                        "image_name": image_name,
                        "image_path": image_path,
                        "noise_step": noise,
                        "trajectory_idx": traj_idx,
                        "traj_dir_exists": int(traj_dir.exists()),
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
                "trajectory_idx",
                "traj_dir_exists",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} missing trajectories to {out_path}")


if __name__ == "__main__":
    main()
