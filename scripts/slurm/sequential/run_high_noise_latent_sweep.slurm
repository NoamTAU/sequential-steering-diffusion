#!/bin/bash

# Slurm array for unguided sequential U-turn sweeps at high noise.
# Submit via submit_high_noise_latent_sweep.sh so the array size matches
# (num_images x num_noise_levels).

#SBATCH --job-name=latent_high_noise
#SBATCH --array=0-0
#SBATCH --output=slurm_logs/latent_high_noise_%A_%a.out
#SBATCH --error=slurm_logs/latent_high_noise_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

set -euo pipefail

IMAGE_LIST_FILE=${IMAGE_LIST_FILE:-scripts/image_list.txt}
NOISE_LEVELS_CSV=${NOISE_LEVELS_CSV:-300,400,500}
NUM_UTURNS=${NUM_UTURNS:-100}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-20}
OUTPUT_DIR=${OUTPUT_DIR:-/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns}
MODEL_PATH=${MODEL_PATH:-/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/models/256x256_diffusion_uncond.pt}

IFS=',' read -r -a NOISE_LEVELS <<< "$NOISE_LEVELS_CSV"
NUM_NOISE_LEVELS=${#NOISE_LEVELS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}
IMAGE_INDEX=$((TASK_ID / NUM_NOISE_LEVELS))
NOISE_INDEX=$((TASK_ID % NUM_NOISE_LEVELS))
CURRENT_NOISE_STEP=${NOISE_LEVELS[$NOISE_INDEX]}
START_IMAGE=$(sed -n "$((IMAGE_INDEX + 1))p" "$IMAGE_LIST_FILE")

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Image list: $IMAGE_LIST_FILE"
echo "Image index: $IMAGE_INDEX"
echo "Noise index: $NOISE_INDEX"
echo "Noise step: $CURRENT_NOISE_STEP"
echo "Start image: $START_IMAGE"

if [ -z "$START_IMAGE" ]; then
    echo "Error: no image found for index $IMAGE_INDEX"
    exit 1
fi

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

python scripts/sequential_uturns.py \
    --start_image_path "$START_IMAGE" \
    --output_dir "$OUTPUT_DIR" \
    --num_uturns "$NUM_UTURNS" \
    --num_trajectories "$NUM_TRAJECTORIES" \
    --noise_step "$CURRENT_NOISE_STEP" \
    --model_path "$MODEL_PATH"

