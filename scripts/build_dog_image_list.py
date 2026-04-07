import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch as th
from PIL import Image
from torchvision.transforms import Resize

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from guided_diffusion.torch_classifiers import load_classifier


DOG_INDICES = list(range(151, 269))


def read_image_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def read_ground_truth(path):
    with open(path, "r") as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    # Standard ILSVRC2012 validation labels are 1-based.
    return [label - 1 for label in labels]


def classify_image(classifier, preprocess, device, image_path, image_size):
    resize = Resize([image_size, image_size], Image.BICUBIC)
    start_pil = resize(Image.open(image_path).convert("RGB"))
    start_tensor = th.tensor(np.array(start_pil)).float() / 127.5 - 1
    start_tensor = start_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    with th.no_grad():
        clf_dtype = next(classifier.parameters()).dtype
        logits = classifier(preprocess(start_tensor).to(clf_dtype))
        probs = th.nn.functional.softmax(logits, dim=1)
    top1_idx = int(th.argmax(logits, dim=1).item())
    top1_prob = float(probs[0, top1_idx].item())
    return {
        "image_path": image_path,
        "image_name": Path(image_path).stem,
        "top1_class_idx": top1_idx,
        "top1_prob": top1_prob,
        "is_top1_dog": top1_idx in DOG_INDICES,
    }


def write_csv(rows, csv_path):
    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_dog_list(rows, out_path):
    with open(out_path, "w") as f:
        for row in rows:
            if row["is_top1_dog"]:
                f.write(row["image_path"] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--dog-list-out", required=True)
    parser.add_argument("--summary-csv-out", required=True)
    parser.add_argument(
        "--ground-truth-file",
        default="",
        help="Optional ILSVRC2012 validation ground-truth file (1 label per line, 1-based class ids).",
    )
    parser.add_argument(
        "--require-true-dog",
        action="store_true",
        help="Write only images whose ground-truth ImageNet label is a dog class.",
    )
    parser.add_argument(
        "--require-classifier-match",
        action="store_true",
        help="When ground truth is provided, also require classifier top-1 to match the true label.",
    )
    parser.add_argument("--classifier-name", default="convnext_base")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--classifier-use-fp16", action="store_true")
    args = parser.parse_args()

    image_paths = read_image_list(args.image_list)
    gt_labels = None
    if args.ground_truth_file:
        gt_labels = read_ground_truth(args.ground_truth_file)
        if len(gt_labels) != len(image_paths):
            raise ValueError(
                f"Ground-truth label count ({len(gt_labels)}) does not match image list length ({len(image_paths)})."
            )

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    classifier, preprocess, _ = load_classifier(args.classifier_name)
    classifier.to(device)
    classifier.eval()
    if args.classifier_use_fp16:
        if hasattr(classifier, "convert_to_fp16"):
            classifier.convert_to_fp16()
        else:
            classifier.half()

    rows = []
    for idx, image_path in enumerate(image_paths):
        row = classify_image(classifier, preprocess, device, image_path, args.image_size)
        if gt_labels is not None:
            true_class_idx = gt_labels[idx]
            row["true_class_idx"] = true_class_idx
            row["is_true_dog"] = true_class_idx in DOG_INDICES
            row["top1_matches_true"] = row["top1_class_idx"] == true_class_idx
        else:
            row["true_class_idx"] = ""
            row["is_true_dog"] = ""
            row["top1_matches_true"] = ""
        rows.append(row)
        print(
            f"{row['image_name']}: top1={row['top1_class_idx']} "
            f"prob={row['top1_prob']:.4f} clf_dog={row['is_top1_dog']}"
        )

    os.makedirs(os.path.dirname(args.summary_csv_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.dog_list_out), exist_ok=True)
    write_csv(rows, args.summary_csv_out)
    selected_rows = rows
    if args.require_true_dog:
        if gt_labels is None:
            raise ValueError("--require-true-dog requires --ground-truth-file")
        selected_rows = [row for row in selected_rows if row["is_true_dog"]]
    else:
        selected_rows = [row for row in selected_rows if row["is_top1_dog"]]
    if args.require_classifier_match:
        if gt_labels is None:
            raise ValueError("--require-classifier-match requires --ground-truth-file")
        selected_rows = [row for row in selected_rows if row["top1_matches_true"]]

    write_dog_list(selected_rows, args.dog_list_out)
    n_dogs = len(selected_rows)
    print(f"Wrote {len(rows)} image summaries to {args.summary_csv_out}")
    print(f"Wrote {n_dogs} dog images to {args.dog_list_out}")
    if gt_labels is not None:
        n_true_dogs = sum(1 for row in rows if row["is_true_dog"])
        n_match = sum(1 for row in rows if row["top1_matches_true"] is True)
        print(f"Ground-truth dog images in list: {n_true_dogs}")
        print(f"Classifier top-1 matches ground truth on: {n_match} images")


if __name__ == "__main__":
    main()
