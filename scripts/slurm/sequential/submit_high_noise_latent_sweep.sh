#!/bin/bash

set -euo pipefail

IMAGE_LIST_FILE=${IMAGE_LIST_FILE:-scripts/image_list.txt}
NOISE_LEVELS_CSV=${NOISE_LEVELS_CSV:-300,400,500}
NOISE_LEVELS_SPEC=${NOISE_LEVELS_SPEC:-${NOISE_LEVELS_CSV//,/:}}
NUM_UTURNS=${NUM_UTURNS:-100}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-20}
OUTPUT_DIR=${OUTPUT_DIR:-/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns}

if [ ! -f "$IMAGE_LIST_FILE" ]; then
    echo "Missing image list: $IMAGE_LIST_FILE" >&2
    exit 1
fi

NUM_IMAGES=$(grep -cve '^\s*$' "$IMAGE_LIST_FILE")
IFS=':' read -r -a NOISE_LEVELS <<< "$NOISE_LEVELS_SPEC"
NUM_NOISE_LEVELS=${#NOISE_LEVELS[@]}

if [ "$NUM_IMAGES" -le 0 ] || [ "$NUM_NOISE_LEVELS" -le 0 ]; then
    echo "Need at least one image and one noise level" >&2
    exit 1
fi

ARRAY_END=$((NUM_IMAGES * NUM_NOISE_LEVELS - 1))

echo "Submitting high-noise latent sweep"
echo "  images: $NUM_IMAGES"
echo "  noises: ${NOISE_LEVELS[*]}"
echo "  trajectories per noise: $NUM_TRAJECTORIES"
echo "  U-turns per trajectory: $NUM_UTURNS"
echo "  array: 0-$ARRAY_END"

sbatch \
  --array=0-"$ARRAY_END" \
  --export="ALL,IMAGE_LIST_FILE=$IMAGE_LIST_FILE,NOISE_LEVELS_SPEC=$NOISE_LEVELS_SPEC,NUM_UTURNS=$NUM_UTURNS,NUM_TRAJECTORIES=$NUM_TRAJECTORIES,OUTPUT_DIR=$OUTPUT_DIR" \
  scripts/slurm/sequential/run_high_noise_latent_sweep.slurm
