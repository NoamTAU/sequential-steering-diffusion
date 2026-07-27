#!/bin/bash

#SBATCH --job-name=image_quality
#SBATCH --output=slurm_logs/image_quality_%j.out
#SBATCH --error=slurm_logs/image_quality_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

set -euo pipefail

: "${MANIFEST:?Set MANIFEST to a CSV manifest}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a feature output directory}"

BATCH_SIZE=${BATCH_SIZE:-64}
WORKERS=${WORKERS:-12}
SKIP_CONVNEXT=${SKIP_CONVNEXT:-0}

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

EXTRA_ARGS=()
if [ "$SKIP_CONVNEXT" = "1" ]; then
  EXTRA_ARGS+=(--skip-convnext)
fi

python scripts/rebuttal/extract_image_quality_features.py \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --resume \
  "${EXTRA_ARGS[@]}"
