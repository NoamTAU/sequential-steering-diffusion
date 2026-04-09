import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

import numpy as np


def find_transition_idx(orig_probs, target_probs):
    if orig_probs is None or target_probs is None:
        return None
    n = min(len(orig_probs), len(target_probs))
    if n == 0:
        return None
    diff = np.array(target_probs[:n]) - np.array(orig_probs[:n])
    hits = np.where(diff >= 0)[0]
    if len(hits) == 0:
        return None
    return int(hits[0])


def load_optional_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def parse_target_from_path(run_path):
    for part in run_path.parts:
        match = re.fullmatch(r"target(?:_auto)?_(\d+)", part)
        if match:
            return int(match.group(1))
    return ""


def summarize_run(run_dir):
    run_path = Path(run_dir)
    steering_path = run_path / "steering_data.npz"
    if not steering_path.exists():
        return None

    data = np.load(steering_path, allow_pickle=True)
    attempts = data["attempts"] if "attempts" in data else None
    score_mode = None
    if "score_mode" in data:
        score_mode_raw = data["score_mode"]
        if np.ndim(score_mode_raw) == 0:
            score_mode = str(score_mode_raw.item())
        elif len(score_mode_raw) > 0:
            score_mode = str(score_mode_raw[0])

    is_meta = "probs_cat" in data and "probs_dog" in data
    if is_meta:
        orig_probs = np.array(data["probs_dog"], dtype=float)
        target_probs = np.array(data["probs_cat"], dtype=float)
        mode_family = "meta_cat"
    else:
        if "probs_orig" not in data or "probs_target" not in data:
            return None
        orig_probs = np.array(data["probs_orig"], dtype=float)
        target_probs = np.array(data["probs_target"], dtype=float)
        mode_family = "dog2dog"

    n_steps_recorded = int(len(target_probs) - 1)
    transition_idx = find_transition_idx(orig_probs, target_probs)
    final_target_prob = float(target_probs[-1]) if len(target_probs) else None
    final_orig_prob = float(orig_probs[-1]) if len(orig_probs) else None
    max_target_prob = float(np.max(target_probs)) if len(target_probs) else None
    total_attempts = int(np.sum(attempts[1:])) if attempts is not None and len(attempts) > 1 else 0
    n_skips = int(np.sum(np.array(attempts[1:], dtype=int) > 0)) if attempts is not None and len(attempts) > 1 else 0

    start_info = load_optional_json(run_path / "start_image_info.json")
    auto_target = load_optional_json(run_path / "auto_target.json")

    image_name = run_path.parent.parent.name if len(run_path.parents) >= 2 else ""
    run_dir_name = run_path.name
    repeat_match = re.search(r"_rep(\d+)$", run_dir_name)
    repeat_index = int(repeat_match.group(1)) if repeat_match else ""

    return {
        "run_dir": str(run_path),
        "run_dir_name": run_dir_name,
        "image_name": image_name,
        "mode_family": mode_family,
        "score_mode": score_mode or "",
        "repeat_index": repeat_index,
        "n_steps_recorded": n_steps_recorded,
        "transition_idx": "" if transition_idx is None else transition_idx,
        "crossed": transition_idx is not None,
        "final_target_prob": final_target_prob,
        "final_orig_prob": final_orig_prob,
        "max_target_prob": max_target_prob,
        "total_attempts": total_attempts,
        "n_nonzero_attempt_steps": n_skips,
        "top1_class_idx": start_info.get("top1_class_idx", ""),
        "is_top1_dog": start_info.get("is_top1_dog", ""),
        "orig_class_idx": auto_target.get("orig_class_idx", start_info.get("top1_class_idx", "")),
        "target_class_idx": auto_target.get("target_class_idx", parse_target_from_path(run_path)),
    }


def iter_runs(root):
    for steering_path in glob.glob(os.path.join(root, "**", "steering_data.npz"), recursive=True):
        yield os.path.dirname(steering_path)


def write_csv(rows, out_csv):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root steering results directory")
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(set(iter_runs(args.root))):
        row = summarize_run(run_dir)
        if row is not None:
            rows.append(row)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    write_csv(rows, args.out_csv)
    print(f"Wrote {len(rows)} run summaries to {args.out_csv}")


if __name__ == "__main__":
    main()
