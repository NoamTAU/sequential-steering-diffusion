#!/usr/bin/env bash
#SBATCH --job-name=clip_path
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH --time=02:00:00
#SBATCH --output=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx/image_quality/corrected_accumulated_path/logs/extract-%j.out

set -euo pipefail

ROOT=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx
CODE_ROOT="${ROOT}/code_sync"
QUALITY_ROOT="${ROOT}/image_quality"
OUTPUT_ROOT="${QUALITY_ROOT}/corrected_accumulated_path/raw"
SCRIPT="${CODE_ROOT}/extract_corrected_clip_trajectory_path.py"

mkdir -p "${OUTPUT_ROOT}" "${QUALITY_ROOT}/corrected_accumulated_path/logs"
export PYTHONUNBUFFERED=1

/home/nlevi/Noam/miniconda3/bin/python "${SCRIPT}" \
  --manifest "${QUALITY_ROOT}/manifests/sequential_manifest.csv" \
  --output-dir "${OUTPUT_ROOT}" \
  --download-root /home/nlevi/.cache/clip
