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
    parser.add_argument("--classifier-name", default="convnext_base")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--classifier-use-fp16", action="store_true")
    args = parser.parse_args()

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
    for image_path in read_image_list(args.image_list):
        row = classify_image(classifier, preprocess, device, image_path, args.image_size)
        rows.append(row)
        print(
            f"{row['image_name']}: top1={row['top1_class_idx']} "
            f"prob={row['top1_prob']:.4f} dog={row['is_top1_dog']}"
        )

    os.makedirs(os.path.dirname(args.summary_csv_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.dog_list_out), exist_ok=True)
    write_csv(rows, args.summary_csv_out)
    write_dog_list(rows, args.dog_list_out)
    n_dogs = sum(1 for row in rows if row["is_top1_dog"])
    print(f"Wrote {len(rows)} image summaries to {args.summary_csv_out}")
    print(f"Wrote {n_dogs} dog images to {args.dog_list_out}")


if __name__ == "__main__":
    main()
