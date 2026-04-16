#!/bin/bash

# Rebuild specific image/noise pairs from a manifest, backing up any partial
# result tree before rerunning so completed pairs are untouched.

#SBATCH --job-name=latent_high_manifest
#SBATCH --array=0-0
#SBATCH --output=slurm_logs/latent_high_manifest_%A_%a.out
#SBATCH --error=slurm_logs/latent_high_manifest_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

set -euo pipefail

MANIFEST_CSV=${MANIFEST_CSV:?MANIFEST_CSV is required}
NUM_UTURNS=${NUM_UTURNS:-100}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-20}
OUTPUT_DIR=${OUTPUT_DIR:-/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns}
MODEL_PATH=${MODEL_PATH:-/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/models/256x256_diffusion_uncond.pt}
BACKUP_PARTIAL=${BACKUP_PARTIAL:-1}

LINE_NO=$((SLURM_ARRAY_TASK_ID + 2))
ROW=$(sed -n "${LINE_NO}p" "$MANIFEST_CSV")
if [ -z "$ROW" ]; then
    echo "No manifest row for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

IFS=',' read -r IMAGE_NAME IMAGE_PATH NOISE_STEP REASON TRAJ_FOUND COMPLETE_TRAJ <<< "$ROW"
PAIR_DIR="${OUTPUT_DIR}/${IMAGE_NAME}/noise_step_${NOISE_STEP}"

echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Image: $IMAGE_NAME"
echo "Path: $IMAGE_PATH"
echo "Noise: $NOISE_STEP"
echo "Reason: $REASON"
echo "Found trajectories: $TRAJ_FOUND"
echo "Complete trajectories: $COMPLETE_TRAJ"
echo "Pair dir: $PAIR_DIR"

if [ "$BACKUP_PARTIAL" = "1" ] && [ -d "$PAIR_DIR" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    BACKUP_DIR="${PAIR_DIR}_partial_backup_${TS}"
    echo "Backing up partial pair dir to $BACKUP_DIR"
    mv "$PAIR_DIR" "$BACKUP_DIR"
fi

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

python scripts/sequential_uturns.py \
    --start_image_path "$IMAGE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_uturns "$NUM_UTURNS" \
    --num_trajectories "$NUM_TRAJECTORIES" \
    --noise_step "$NOISE_STEP" \
    --model_path "$MODEL_PATH"
