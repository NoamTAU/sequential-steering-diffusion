#!/bin/bash

#SBATCH --job-name=repair_seq_clip
#SBATCH --output=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx/logs/repair_seq_clip_%j.out
#SBATCH --error=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx/logs/repair_seq_clip_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

set -euo pipefail

ROOT=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

python scripts/rebuttal/repair_sequential_clip_endpoints.py \
  --manifest "$ROOT/image_quality/manifests/sequential_manifest.csv" \
  --output "$ROOT/image_quality/features/sequential/clip_metric_overrides.csv"
