#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash download_plots.sh user@host [dest]"
  echo "Example: bash download_plots.sh nlevi@pcsl ~/Downloads/seq_plots"
  exit 1
fi

REMOTE="$1"
DEST="${2:-$HOME/Downloads/seq_plots}"
REMOTE_DIR="/work/pcsl/Noam/sequential_diffusion/results/plots"

mkdir -p "$DEST"

if command -v rsync >/dev/null 2>&1; then
  rsync -avz --progress "${REMOTE}:${REMOTE_DIR}/" "${DEST}/"
else
  scp -r "${REMOTE}:${REMOTE_DIR}" "${DEST}/"
fi
