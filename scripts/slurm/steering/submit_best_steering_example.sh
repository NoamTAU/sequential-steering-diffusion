#!/usr/bin/env bash
set -euo pipefail

ACTIVE_IMAGE_LIST="${1:-scripts/dog_image_list_strict_100.txt}"
TARGET_TOTAL="${2:-20}"
BASE_SEED="${3:-20260408}"

eval "$(
  python scripts/select_best_steering_example.py \
    --active-image-list "$ACTIVE_IMAGE_LIST" \
    --target-total "$TARGET_TOTAL" \
    --require-repeat-index \
    --format shell
)"

echo "Selected image: $IMAGE_NAME"
echo "Image path: $IMAGE_PATH"
echo "Source class: $ORIG_CLASS_IDX"
echo "Target class: $TARGET_CLASS_IDX"
echo "Existing dog->cat runs: $META_COUNT"
echo "Existing dog->dog runs: $DOG_COUNT"
echo "Extra dog->cat runs needed: $META_EXTRA_NEEDED"
echo "Extra dog->dog runs needed: $DOG_EXTRA_NEEDED"

bash scripts/slurm/steering/submit_single_image_extension.sh \
  "$IMAGE_PATH" \
  "$ORIG_CLASS_IDX" \
  "$TARGET_CLASS_IDX" \
  "$META_EXTRA_NEEDED" \
  "$DOG_EXTRA_NEEDED" \
  "$BASE_SEED"
