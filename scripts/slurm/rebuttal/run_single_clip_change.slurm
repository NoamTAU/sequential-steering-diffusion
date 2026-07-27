#!/bin/bash

#SBATCH --job-name=single_clip_change
#SBATCH --output=slurm_logs/single_clip_change_%j.out
#SBATCH --error=slurm_logs/single_clip_change_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

set -euo pipefail

ROOT=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

python scripts/rebuttal/extract_single_clip_change.py \
  --manifest "$ROOT/image_quality/manifests/single_manifest.csv" \
  --output "$ROOT/image_quality/features/single/clip_net_distance.npy" \
  --batch-size 64 \
  --workers 12
