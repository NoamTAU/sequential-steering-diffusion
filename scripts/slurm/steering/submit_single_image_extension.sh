#!/usr/bin/env bash
set -euo pipefail

IMG="${1:?usage: submit_single_image_extension.sh IMAGE_PATH DOG2DOG_TARGET_IDX META_EXTRA DOG_EXTRA [BASE_SEED]}"
DOG2DOG_TARGET_IDX="${2:?usage: submit_single_image_extension.sh IMAGE_PATH DOG2DOG_TARGET_IDX META_EXTRA DOG_EXTRA [BASE_SEED]}"
META_EXTRA="${3:?usage: submit_single_image_extension.sh IMAGE_PATH DOG2DOG_TARGET_IDX META_EXTRA DOG_EXTRA [BASE_SEED]}"
DOG_EXTRA="${4:?usage: submit_single_image_extension.sh IMAGE_PATH DOG2DOG_TARGET_IDX META_EXTRA DOG_EXTRA [BASE_SEED]}"
BASE_SEED="${5:-20260408}"

if [ "$META_EXTRA" -le 0 ] && [ "$DOG_EXTRA" -le 0 ]; then
  echo "Nothing to submit."
  exit 0
fi

echo "Submitting single-image extension"
echo "Image: $IMG"
echo "Fixed dog->dog target: $DOG2DOG_TARGET_IDX"
echo "Extra dog->cat runs: $META_EXTRA"
echo "Extra dog->dog runs: $DOG_EXTRA"
echo "Base seed: $BASE_SEED"

if [ "$META_EXTRA" -gt 0 ]; then
  sbatch \
    --array="0-$((META_EXTRA - 1))" \
    --export=ALL,IMG="$IMG",BASE_SEED="$BASE_SEED" \
    scripts/slurm/steering/run_steering_meta_catprob_single_repeat.slurm
fi

if [ "$DOG_EXTRA" -gt 0 ]; then
  sbatch \
    --array="0-$((DOG_EXTRA - 1))" \
    --export=ALL,IMG="$IMG",TARGET_CLASS_IDX="$DOG2DOG_TARGET_IDX",BASE_SEED="$BASE_SEED" \
    scripts/slurm/steering/run_steering_dog2dog_prob_single_fixed_target.slurm
fi
