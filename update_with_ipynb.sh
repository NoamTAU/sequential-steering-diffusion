#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[update-ipynb] status before pull"
git status -sb

dirty=0
if [ -n "$(git status --porcelain)" ]; then
  dirty=1
  echo "[update-ipynb] working tree dirty; stashing changes"
  git stash push -u -m "update_with_ipynb auto-stash"
fi

git pull --rebase

if [ "$dirty" -eq 1 ]; then
  echo "[update-ipynb] restoring stashed changes"
  if ! git stash pop; then
    echo "[update-ipynb] stash pop had conflicts. Resolve conflicts, then rerun."
    exit 1
  fi
fi

# Optional: clean generated PDFs so they don't keep showing as modified.
# Disable by running: CLEAN_OUTPUTS=0 ./update_with_ipynb.sh
if [ "${CLEAN_OUTPUTS:-1}" -eq 1 ]; then
  for f in notebooks/_exports/*.pdf notebooks/*.pdf; do
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      git restore "$f" || true
    fi
  done
fi

echo "[update-ipynb] status after pull"
git status -sb
