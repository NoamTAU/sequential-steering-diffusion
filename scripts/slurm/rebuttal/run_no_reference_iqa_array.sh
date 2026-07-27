#!/usr/bin/env bash
#SBATCH --job-name=iqa_full
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=23G
#SBATCH --time=01:00:00
#SBATCH --array=0-31
#SBATCH --output=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx/iqa_full/logs/score-%A_%a.out

set -euo pipefail

ROOT=/work/pcsl/Noam/sequential_diffusion/rebuttal/4AwjOJ5qrx
CODE_ROOT="${ROOT}/code_sync"
IQA_ROOT="${ROOT}/iqa_full"
NUM_SHARDS=16
MODELS=(musiq topiq_nr)

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
SHARD_ID=$((SLURM_ARRAY_TASK_ID / 2))
MODEL="${MODELS[$MODEL_INDEX]}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/iqa_python"
export TORCH_HOME="${ROOT}/iqa_cache/torch"
export XDG_CACHE_HOME="${ROOT}/iqa_cache/xdg"
export HF_HOME="${ROOT}/iqa_cache/huggingface"
export MPLCONFIGDIR="${ROOT}/iqa_cache/matplotlib"

/home/nlevi/Noam/miniconda3/bin/python \
  "${CODE_ROOT}/score_no_reference_iqa.py" score-shard \
  --manifest "${IQA_ROOT}/iqa_full_manifest.csv" \
  --output-dir "${IQA_ROOT}/scores" \
  --model "${MODEL}" \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --device cuda \
  --checkpoint-every 100
