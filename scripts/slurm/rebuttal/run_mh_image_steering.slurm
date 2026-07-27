#!/bin/bash

#SBATCH --job-name=mh_image_steer
#SBATCH --output=slurm_logs/mh_image_steer_%A_%a.out
#SBATCH --error=slurm_logs/mh_image_steer_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

set -euo pipefail

: "${MANIFEST:?Set MANIFEST to the MH steering CSV}"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
IFS=',' read -r IMAGE_PATH TARGET_MODE ORIG_IDX TARGET_IDX NOISE_STEP ENERGY_LAMBDA REPEAT_INDEX SEED NUM_STEPS <<< "$LINE"

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

python scripts/rebuttal/mh_image_steering.py \
  --start_image_path "$IMAGE_PATH" \
  --target_mode "$TARGET_MODE" \
  --orig_class_idx "$ORIG_IDX" \
  --target_class_idx "$TARGET_IDX" \
  --noise_step "$NOISE_STEP" \
  --energy_lambda "$ENERGY_LAMBDA" \
  --repeat_index "$REPEAT_INDEX" \
  --seed "$SEED" \
  --num_steps "$NUM_STEPS"
