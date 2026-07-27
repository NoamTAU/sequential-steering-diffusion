#!/bin/bash
#SBATCH --job-name=iqa_pilot
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH --time=01:00:00
#SBATCH --output=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx/iqa_pilot/slurm-%j.out

set -euo pipefail

ROOT=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx
PYTHON=/home/nlevi/Noam/miniconda3/bin/python

export PYTHONPATH="${ROOT}/iqa_python:${PYTHONPATH:-}"
export TORCH_HOME="${ROOT}/iqa_cache/torch"
export XDG_CACHE_HOME="${ROOT}/iqa_cache/xdg"
export HF_HOME="${ROOT}/iqa_cache/huggingface"
export MPLCONFIGDIR="${ROOT}/iqa_cache/matplotlib"

mkdir -p \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${MPLCONFIGDIR}"

"${PYTHON}" "${ROOT}/code_sync/pilot_no_reference_iqa.py" score \
  --output-dir "${ROOT}/iqa_pilot" \
  --models musiq topiq_nr niqe \
  --device cuda \
  --seed 44
