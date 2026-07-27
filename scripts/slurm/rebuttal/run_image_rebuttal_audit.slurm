#!/bin/bash

#SBATCH --job-name=image_rebuttal_audit
#SBATCH --output=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx/logs/image_rebuttal_audit_%j.out
#SBATCH --error=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx/logs/image_rebuttal_audit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

set -euo pipefail

ROOT=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

python scripts/rebuttal/audit_image_rebuttal_results.py \
  --root "$ROOT" \
  --output-dir "$ROOT/image_quality/audit"
