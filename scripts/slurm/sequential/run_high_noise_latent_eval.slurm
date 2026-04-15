#!/bin/bash

# Single-job latent evaluation for a targeted unguided high-noise sweep.

#SBATCH --job-name=latent_high_eval
#SBATCH --output=slurm_logs/latent_high_eval_%j.out
#SBATCH --error=slurm_logs/latent_high_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

set -euo pipefail

DATA_DIR=${DATA_DIR:-/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns}
OUTPUT_BASE=${OUTPUT_BASE:-/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion/scripts/sequential_analysis_results}
STATS_DIR=${STATS_DIR:-$OUTPUT_BASE}
IMAGE_LIST_FILE=${IMAGE_LIST_FILE:-}
NOISE_LEVELS_CSV=${NOISE_LEVELS_CSV:-300,400,500}
CLASSIFIER_NAME=${CLASSIFIER_NAME:-convnext_base}

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

CMD=(
  python scripts/evaluate_all_images_all_noises.py
  --data_dir "$DATA_DIR"
  --output_base "$OUTPUT_BASE"
  --stats_dir "$STATS_DIR"
  --classifier_name "$CLASSIFIER_NAME"
  --noise_steps ${NOISE_LEVELS_CSV//,/ }
)

if [ -n "$IMAGE_LIST_FILE" ]; then
  CMD+=(--image_list "$IMAGE_LIST_FILE")
fi

echo "Running latent evaluation on data_dir=$DATA_DIR"
echo "Image list: ${IMAGE_LIST_FILE:-<all images under data_dir>}"
echo "Noise levels: $NOISE_LEVELS_CSV"

"${CMD[@]}"
