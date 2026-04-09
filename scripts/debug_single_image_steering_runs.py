import argparse
import glob
import json
import re
from pathlib import Path


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


def parse_target_from_run_dir(run_dir: Path):
    for part in run_dir.parts:
        m = re.fullmatch(r"target(?:_auto)?_(\d+)", part)
        if m:
            return int(m.group(1))
    return None


def debug(root: Path, image_name: str, target_idx: int | None):
    pattern = str(root / "**" / image_name / "**")
    candidates = sorted({Path(p).parent for p in glob.glob(pattern + "/steering_data.npz", recursive=True)})
    all_dirs = sorted(
        {
            Path(p)
            for p in glob.glob(str(root / "**" / image_name / "**"), recursive=True)
            if Path(p).is_dir() and Path(p).name.startswith("noise_")
        }
    )

    print(f"Root: {root}")
    print(f"Image: {image_name}")
    print(f"Noise dirs found: {len(all_dirs)}")
    print(f"Dirs with steering_data.npz: {len(candidates)}")
    print()

    rows = []
    for run_dir in all_dirs:
        has_data = (run_dir / "steering_data.npz").exists()
        start_info = load_json(run_dir / "start_image_info.json")
        auto_info = load_json(run_dir / "auto_target.json")
        top1 = start_info.get("top1_class_idx")
        orig = auto_info.get("orig_class_idx", top1)
        target = auto_info.get("target_class_idx", parse_target_from_run_dir(run_dir))
        repeat = parse_repeat_index(run_dir)
        passes = (
            has_data
            and repeat is not None
            and top1 is not None
            and orig is not None
            and target is not None
            and int(orig) == int(top1)
            and int(orig) != int(target)
            and (target_idx is None or int(target) == int(target_idx))
        )
        rows.append((passes, repeat, has_data, top1, orig, target, run_dir))

    rows.sort(key=lambda row: (row[1] is None, -1 if row[1] is None else row[1], str(row[6])))
    passing = [row for row in rows if row[0]]
    print(f"Passing strict runs: {len(passing)}")
    print()
    for passes, repeat, has_data, top1, orig, target, run_dir in rows:
        mark = "PASS" if passes else "DROP"
        print(
            f"{mark} repeat={repeat} data={has_data} top1={top1} "
            f"orig={orig} target={target} dir={run_dir}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--target-idx", type=int, default=None)
    args = parser.parse_args()
    debug(Path(args.root), args.image_name, args.target_idx)


if __name__ == "__main__":
    main()
