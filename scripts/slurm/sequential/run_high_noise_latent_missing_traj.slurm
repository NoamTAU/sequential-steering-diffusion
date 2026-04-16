#!/bin/bash

# Fill missing high-noise unguided trajectories one trajectory per job.

#SBATCH --job-name=latent_high_traj
#SBATCH --array=0-0
#SBATCH --output=slurm_logs/latent_high_traj_%A_%a.out
#SBATCH --error=slurm_logs/latent_high_traj_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

set -euo pipefail

MANIFEST_CSV=${MANIFEST_CSV:?MANIFEST_CSV is required}
NUM_UTURNS=${NUM_UTURNS:-100}
OUTPUT_DIR=${OUTPUT_DIR:-/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns}
MODEL_PATH=${MODEL_PATH:-/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/models/256x256_diffusion_uncond.pt}

LINE_NO=$((SLURM_ARRAY_TASK_ID + 2))
ROW=$(sed -n "${LINE_NO}p" "$MANIFEST_CSV")
if [ -z "$ROW" ]; then
    echo "No manifest row for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

IFS=',' read -r IMAGE_NAME IMAGE_PATH NOISE_STEP TRAJECTORY_IDX TRAJ_DIR_EXISTS <<< "$ROW"

echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Image: $IMAGE_NAME"
echo "Path: $IMAGE_PATH"
echo "Noise: $NOISE_STEP"
echo "Trajectory index: $TRAJECTORY_IDX"
echo "Trajectory dir exists: $TRAJ_DIR_EXISTS"

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

python scripts/sequential_uturns.py \
    --start_image_path "$IMAGE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_uturns "$NUM_UTURNS" \
    --noise_step "$NOISE_STEP" \
    --trajectory_idx "$TRAJECTORY_IDX" \
    --model_path "$MODEL_PATH"
