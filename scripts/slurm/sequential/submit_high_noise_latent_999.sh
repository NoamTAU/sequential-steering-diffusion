#!/bin/bash

set -euo pipefail

IMAGE_LIST_FILE=${IMAGE_LIST_FILE:-/work/pcsl/Noam/sequential_diffusion/metadata/high_noise_image_list.txt}
NUM_UTURNS=${NUM_UTURNS:-100}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-10}
OUTPUT_DIR=${OUTPUT_DIR:-/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns}

echo "Submitting dedicated sequential latent sweep for noise=999"
echo "  image list: $IMAGE_LIST_FILE"
echo "  trajectories: $NUM_TRAJECTORIES"
echo "  U-turns: $NUM_UTURNS"

IMAGE_LIST_FILE="$IMAGE_LIST_FILE" \
NOISE_LEVELS_SPEC=999 \
NUM_UTURNS="$NUM_UTURNS" \
NUM_TRAJECTORIES="$NUM_TRAJECTORIES" \
OUTPUT_DIR="$OUTPUT_DIR" \
bash scripts/slurm/sequential/submit_high_noise_latent_sweep.sh
