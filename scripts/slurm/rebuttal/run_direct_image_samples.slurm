#!/bin/bash

#SBATCH --job-name=direct_image_samples
#SBATCH --output=slurm_logs/direct_image_samples_%A_%a.out
#SBATCH --error=slurm_logs/direct_image_samples_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

set -euo pipefail

ROOT=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx/direct_diffusion
SEED=$((43 + SLURM_ARRAY_TASK_ID))

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

python scripts/rebuttal/generate_direct_image_samples.py \
  --output_dir "$ROOT/images" \
  --num_samples 50 \
  --batch_size 10 \
  --seed "$SEED"
