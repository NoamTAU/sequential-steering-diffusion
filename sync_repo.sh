#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash sync_repo.sh \"commit message\""
  exit 1
fi

msg="$1"

echo "[sync] status before pull"
git status -sb

echo "[sync] pulling latest (rebase)"
dirty=0
if [ -n "$(git status --porcelain)" ]; then
  dirty=1
  echo "[sync] working tree dirty; stashing changes"
  git stash push -u -m "sync_repo auto-stash"
fi

git pull --rebase

if [ "$dirty" -eq 1 ]; then
  echo "[sync] restoring stashed changes"
  if ! git stash pop; then
    echo "[sync] stash pop had conflicts. Resolve conflicts, then rerun."
    exit 1
  fi
fi

echo "[sync] status after pull"
git status -sb

echo "[sync] staging changes"
git add -A

if git diff --cached --quiet; then
  echo "[sync] no changes to commit"
  exit 0
fi

echo "[sync] committing"
git commit -m "$msg"

echo "[sync] pushing"
git push
