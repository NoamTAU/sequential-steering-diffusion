#!/usr/bin/env bash
set -euo pipefail

IMAGE_LIST_FILE="${1:-scripts/dog_image_list.txt}"
REPEATS="${2:-5}"

if [ ! -f "$IMAGE_LIST_FILE" ]; then
  echo "Missing image list: $IMAGE_LIST_FILE"
  exit 1
fi

N_IMAGES=$(grep -cve '^\s*$' "$IMAGE_LIST_FILE")
if [ "$N_IMAGES" -le 0 ]; then
  echo "Image list is empty: $IMAGE_LIST_FILE"
  exit 1
fi

ARRAY_END=$((N_IMAGES * REPEATS - 1))
echo "Submitting multi-image steering arrays"
echo "Image list: $IMAGE_LIST_FILE"
echo "Images: $N_IMAGES"
echo "Repeats per image: $REPEATS"
echo "Array range: 0-$ARRAY_END"

sbatch \
  --array="0-$ARRAY_END" \
  --export=ALL,IMAGE_LIST_FILE="$IMAGE_LIST_FILE",REPEATS="$REPEATS" \
  scripts/slurm/steering/run_steering_meta_catprob_multi.slurm

sbatch \
  --array="0-$ARRAY_END" \
  --export=ALL,IMAGE_LIST_FILE="$IMAGE_LIST_FILE",REPEATS="$REPEATS" \
  scripts/slurm/steering/run_steering_dog2dog_prob_multi.slurm
