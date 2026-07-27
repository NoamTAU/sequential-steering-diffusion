#!/usr/bin/env bash
#SBATCH --job-name=clip_net
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx/image_quality/net_displacement/logs/extract-%A_%a.out

set -euo pipefail

ROOT=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx
CODE_ROOT="${ROOT}/code_sync"
QUALITY_ROOT="${ROOT}/image_quality"
OUTPUT_ROOT="${QUALITY_ROOT}/net_displacement/raw"
SCRIPT="${CODE_ROOT}/extract_corrected_clip_net_displacement.py"

mkdir -p "${OUTPUT_ROOT}" "${QUALITY_ROOT}/net_displacement/logs"
export PYTHONUNBUFFERED=1

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    /home/nlevi/Noam/miniconda3/bin/python "${SCRIPT}" pairs \
      --manifest "${QUALITY_ROOT}/manifests/single_manifest.csv" \
      --output "${OUTPUT_ROOT}/single_clip_net_distance_corrected.npy" \
      --download-root /home/nlevi/.cache/clip
    ;;
  1)
    /home/nlevi/Noam/miniconda3/bin/python "${SCRIPT}" pairs \
      --manifest "${QUALITY_ROOT}/manifests/sequential_manifest.csv" \
      --output "${OUTPUT_ROOT}/sequential_clip_net_distance_corrected.npy" \
      --download-root /home/nlevi/.cache/clip
    ;;
  2)
    /home/nlevi/Noam/miniconda3/bin/python "${SCRIPT}" reference \
      --start-manifest "${QUALITY_ROOT}/manifests/shared_start_manifest.csv" \
      --direct-manifest "${ROOT}/direct_diffusion/direct_manifest.csv" \
      --output-dir "${OUTPUT_ROOT}/direct_reference" \
      --download-root /home/nlevi/.cache/clip
    ;;
  *)
    echo "Unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac
