#!/bin/bash

#SBATCH --job-name=direct_quality
#SBATCH --output=slurm_logs/direct_quality_%j.out
#SBATCH --error=slurm_logs/direct_quality_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

set -euo pipefail

ROOT=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

python scripts/rebuttal/build_direct_image_manifest.py \
  --image-dir "$ROOT/direct_diffusion/images" \
  --output "$ROOT/direct_diffusion/direct_manifest.csv" \
  --expected 200

python scripts/rebuttal/extract_image_quality_features.py \
  --manifest "$ROOT/direct_diffusion/direct_manifest.csv" \
  --output-dir "$ROOT/direct_diffusion/features" \
  --batch-size 64 \
  --workers 12
