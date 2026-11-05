#!/usr/bin/env bash
# Mac -> GPU box: copy the committed public repo (incl. .git, so runs record the commit; R5).
# Private notes (notes/, CLAUDE.md, .git-private/), runs/ and the venv are never sent;
# --delete never touches runs/ on the box (excluded paths are protected).
# Refuses a dirty tree: a run's commit hash must describe the code that ran.
#   bash scripts/gpu/push_code.sh            # then on the box: bash scripts/gpu/bootstrap.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "push_code: working tree has uncommitted changes; commit first (R5)" >&2
  exit 1
fi
echo "push_code: $(git rev-parse --short HEAD) -> $GPU_HOST:$GPU_DIR"
"${SSH[@]}" "$GPU_HOST" "mkdir -p '$GPU_DIR'"
rsync "${RSYNC_FLAGS[@]}" --delete \
  --exclude '.venv/' --include 'runs/.gitkeep' --exclude 'runs/*' --exclude 'notes/' --exclude 'CLAUDE.md' \
  --exclude '.git-private/' --exclude 'wandb/' --exclude 'outputs/' --exclude 'multirun/' \
  --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
  --exclude '.mypy_cache/' --exclude '.DS_Store' \
  ./ "$GPU_HOST:$GPU_DIR/"
# rsync -a keeps the Mac's uid; git on the box then refuses the repo ("dubious ownership")
# and runs could not record their commit (R5).
"${SSH[@]}" "$GPU_HOST" "git config --global --add safe.directory \"\$HOME/$GPU_DIR\" && cd '$GPU_DIR' && echo \"box at \$(git rev-parse --short HEAD), dirty files: \$(git status --porcelain --untracked-files=no | wc -l)\""
