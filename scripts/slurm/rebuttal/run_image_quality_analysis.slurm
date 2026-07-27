#!/bin/bash

#SBATCH --job-name=image_quality_analysis
#SBATCH --output=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx/logs/image_quality_analysis_%j.out
#SBATCH --error=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx/logs/image_quality_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=06:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx/code_sync

set -euo pipefail

ROOT=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx
COMPUTE_FID=${COMPUTE_FID:-0}

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

EXTRA_ARGS=()
if [ "$COMPUTE_FID" = "1" ]; then
  EXTRA_ARGS+=(--compute-fid)
fi

python analyze_image_quality_rebuttal.py \
  --manifest-dir "$ROOT/image_quality/manifests" \
  --feature-dir "$ROOT/image_quality/features" \
  --output-dir "$ROOT/image_quality/results" \
  --steering-summary-json "$ROOT/cached_steering/cached_image_steering_summary.json" \
  --mh-steering-summary-csv "$ROOT/mh_steering/summary/mh_image_steering_summary.csv" \
  --direct-manifest "$ROOT/direct_diffusion/direct_manifest.csv" \
  --direct-feature-dir "$ROOT/direct_diffusion/features" \
  "${EXTRA_ARGS[@]}"
