#!/bin/bash

set -euo pipefail

MANIFEST_CSV=${MANIFEST_CSV:?MANIFEST_CSV is required}
NUM_UTURNS=${NUM_UTURNS:-100}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-20}
OUTPUT_DIR=${OUTPUT_DIR:-/work/pcsl/Noam/sequential_diffusion/results/sequential_uturns}
BACKUP_PARTIAL=${BACKUP_PARTIAL:-1}

if [ ! -f "$MANIFEST_CSV" ]; then
    echo "Missing manifest: $MANIFEST_CSV" >&2
    exit 1
fi

NUM_ROWS=$(( $(wc -l < "$MANIFEST_CSV") - 1 ))
if [ "$NUM_ROWS" -le 0 ]; then
    echo "Manifest is empty: $MANIFEST_CSV" >&2
    exit 0
fi

ARRAY_END=$((NUM_ROWS - 1))
echo "Submitting manifest rerun for $NUM_ROWS missing/incomplete pairs"
echo "  manifest: $MANIFEST_CSV"
echo "  array: 0-$ARRAY_END"

sbatch \
  --array=0-"$ARRAY_END" \
  --export=ALL,MANIFEST_CSV="$MANIFEST_CSV",NUM_UTURNS="$NUM_UTURNS",NUM_TRAJECTORIES="$NUM_TRAJECTORIES",OUTPUT_DIR="$OUTPUT_DIR",BACKUP_PARTIAL="$BACKUP_PARTIAL" \
  scripts/slurm/sequential/run_high_noise_latent_manifest.slurm
