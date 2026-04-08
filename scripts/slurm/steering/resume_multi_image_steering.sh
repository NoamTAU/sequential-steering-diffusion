#!/usr/bin/env bash
set -euo pipefail

ACTIVE_IMAGE_LIST="${1:-scripts/dog_image_list_strict_100.txt}"
EXPECTED_REPEATS="${2:-4}"
BASE_SEED="${3:-20260407}"
META_CSV="${4:-/work/pcsl/Noam/sequential_diffusion/results/steering_meta_v2_multi/steering_summary.csv}"
DOG_CSV="${5:-/work/pcsl/Noam/sequential_diffusion/results/steering_dog2dog_v1_multi/steering_summary.csv}"
MANIFEST_ROOT="${6:-/work/pcsl/Noam/sequential_diffusion/metadata/manifests}"
META_MANIFEST="${7:-$MANIFEST_ROOT/steering_meta_missing_manifest.csv}"
DOG_MANIFEST="${8:-$MANIFEST_ROOT/steering_dog_missing_manifest.csv}"

mkdir -p "$MANIFEST_ROOT"

python scripts/build_missing_steering_manifest.py \
  --active-image-list "$ACTIVE_IMAGE_LIST" \
  --meta-csv "$META_CSV" \
  --dog-csv "$DOG_CSV" \
  --expected-repeats "$EXPECTED_REPEATS" \
  --base-seed "$BASE_SEED" \
  --meta-out "$META_MANIFEST" \
  --dog-out "$DOG_MANIFEST"

META_TASKS=$(( $(wc -l < "$META_MANIFEST") - 1 ))
DOG_TASKS=$(( $(wc -l < "$DOG_MANIFEST") - 1 ))

echo "Missing dog->cat tasks: $META_TASKS"
echo "Missing dog->dog tasks: $DOG_TASKS"

if [ "$META_TASKS" -gt 0 ]; then
  sbatch \
    --array="0-$((META_TASKS - 1))" \
    --export=ALL,MANIFEST_FILE="$META_MANIFEST" \
    scripts/slurm/steering/run_steering_meta_catprob_manifest.slurm
fi

if [ "$DOG_TASKS" -gt 0 ]; then
  sbatch \
    --array="0-$((DOG_TASKS - 1))" \
    --export=ALL,MANIFEST_FILE="$DOG_MANIFEST" \
    scripts/slurm/steering/run_steering_dog2dog_prob_manifest.slurm
fi
