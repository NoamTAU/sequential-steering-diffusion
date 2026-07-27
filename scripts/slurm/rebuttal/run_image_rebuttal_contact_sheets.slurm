#!/bin/bash

#SBATCH --job-name=image_contact_sheets
#SBATCH --output=slurm_logs/image_contact_sheets_%j.out
#SBATCH --error=slurm_logs/image_contact_sheets_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nlevi/Noam/SingleMaskDiffusion/guided-diffusion

set -euo pipefail

ROOT=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx

source ~/Noam/miniconda3/etc/profile.d/conda.sh
conda activate llm_physics

python scripts/rebuttal/make_image_rebuttal_contact_sheets.py \
  --root "$ROOT" \
  --output-dir "$ROOT/image_quality/audit/contact_sheets"
