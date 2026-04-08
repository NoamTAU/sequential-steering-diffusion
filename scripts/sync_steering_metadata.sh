#!/usr/bin/env bash
set -euo pipefail

METADATA_DIR="${1:-/work/pcsl/Noam/sequential_diffusion/metadata}"
IMAGE_DIR="${2:-/work/pcsl/Noam/diffusion_datasets/all_images}"
GROUND_TRUTH_FILE="${3:-/work/pcsl/imagenet/imagenet1k/devkit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt}"
SAMPLE_SIZE="${4:-100}"
SAMPLE_SEED="${5:-20260407}"

mkdir -p "$METADATA_DIR"

python scripts/build_image_list.py \
  --image-dir "$IMAGE_DIR" \
  --out "$METADATA_DIR/imagenet_val_image_list.txt" \
  --pattern '*.JPEG'

python scripts/build_dog_image_list.py \
  --image-list "$METADATA_DIR/imagenet_val_image_list.txt" \
  --dog-list-out "$METADATA_DIR/dog_image_list_strict.txt" \
  --summary-csv-out "$METADATA_DIR/dog_image_summary.csv" \
  --ground-truth-file "$GROUND_TRUTH_FILE" \
  --require-true-dog \
  --require-classifier-match \
  --classifier-use-fp16

python scripts/sample_image_list.py \
  --input "$METADATA_DIR/dog_image_list_strict.txt" \
  --output "$METADATA_DIR/dog_image_list_strict_100.txt" \
  --num-images "$SAMPLE_SIZE" \
  --seed "$SAMPLE_SEED"

echo "Metadata synced under: $METADATA_DIR"
