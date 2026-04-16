import argparse
import os
from pathlib import Path


def read_image_names(path):
    out = []
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            out.append(Path(line).stem)
    return out


def trajectory_complete(traj_dir, expected_uturns):
    start_file = traj_dir / "uturn_000.jpeg"
    final_file = traj_dir / f"uturn_{expected_uturns:03d}.jpeg"
    if not start_file.exists():
        start_file = traj_dir / "uturn_000.png"
    if not final_file.exists():
        final_file = traj_dir / f"uturn_{expected_uturns:03d}.png"
    return start_file.exists(), final_file.exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--classifier-name", default="convnext_base")
    parser.add_argument("--noise-steps", nargs="+", type=int, required=True)
    parser.add_argument("--expected-trajectories", type=int, required=True)
    parser.add_argument("--expected-uturns", type=int, required=True)
    args = parser.parse_args()

    image_names = read_image_names(args.image_list)
    results_root = Path(args.results_root)
    analysis_root = Path(args.analysis_root)

    print(f"Images in sweep: {len(image_names)}")
    print(f"Noise steps: {args.noise_steps}")
    print(f"Expected trajectories per image/noise: {args.expected_trajectories}")
    print(f"Expected final U-turn index: {args.expected_uturns}")
    print()

    generation_ok = 0
    eval_ok = 0
    total_pairs = len(image_names) * len(args.noise_steps)

    for noise in args.noise_steps:
        print(f"[noise={noise}]")
        missing_generation = []
        incomplete_generation = []
        missing_eval = []

        for image_name in image_names:
            noise_dir = results_root / image_name / f"noise_step_{noise}"
            traj_dirs = sorted(p for p in noise_dir.glob("trajectory_*") if p.is_dir())
            complete_count = 0
            for traj_dir in traj_dirs:
                has_start, has_final = trajectory_complete(traj_dir, args.expected_uturns)
                if has_start and has_final:
                    complete_count += 1

            if complete_count >= args.expected_trajectories:
                generation_ok += 1
            else:
                if not noise_dir.exists():
                    missing_generation.append(image_name)
                else:
                    incomplete_generation.append((image_name, len(traj_dirs), complete_count))

            eval_file = (
                analysis_root
                / args.classifier_name
                / image_name
                / f"noise_{noise}"
                / "sequential_activations_v2.pk"
            )
            if eval_file.exists():
                eval_ok += 1
            else:
                missing_eval.append(image_name)

        print(f"  complete generation pairs: {generation_ok}/{total_pairs} cumulative")
        print(f"  complete evaluation pairs: {eval_ok}/{total_pairs} cumulative")
        if missing_generation:
            print(f"  missing noise directories: {len(missing_generation)}")
            for name in missing_generation[:10]:
                print(f"    {name}")
        if incomplete_generation:
            print(f"  incomplete generation directories: {len(incomplete_generation)}")
            for name, found, complete in incomplete_generation[:10]:
                print(f"    {name}: trajectories_found={found}, complete_trajectories={complete}")
        if missing_eval:
            print(f"  missing evaluation files: {len(missing_eval)}")
            for name in missing_eval[:10]:
                print(f"    {name}")
        if not (missing_generation or incomplete_generation or missing_eval):
            print("  all expected generation and evaluation outputs are present")
        print()

    print("Summary")
    print(f"  generation complete: {generation_ok}/{total_pairs}")
    print(f"  evaluation complete: {eval_ok}/{total_pairs}")


if __name__ == "__main__":
    main()
