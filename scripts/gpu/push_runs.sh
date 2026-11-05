#!/usr/bin/env bash
# Mac -> new GPU box, after a crash (D31): tell the box which policies are already trained.
# 1. Locally, unfinished run directories (no DONE: the run the crash interrupted, pulled
#    half-way) are moved to runs/_partial/, so the retrained run lands in a clean directory.
# 2. Finished runs are copied to the box WITHOUT checkpoints and traces (they stay on the Mac):
#    only the small files the sweep reads (DONE, configs, metrics, eval results), plus the
#    ledger and gate records. Relaunching the sweep then skips every finished job; the
#    backend pick and the gate re-check from these files in seconds.
#   bash scripts/gpu/push_code.sh && bash scripts/gpu/push_runs.sh     # then bootstrap + launch
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
mkdir -p runs
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
moved=0
for d in runs/*/; do
  d="${d%/}"; name="$(basename "$d")"
  [[ "$name" == _* ]] && continue
  if [[ ! -f "$d/DONE" ]]; then
    mkdir -p runs/_partial
    mv "$d" "runs/_partial/${name}_${stamp}"
    echo "push_runs: moved unfinished $name -> runs/_partial/"
    moved=$((moved + 1))
  fi
done
n_done="$(find runs -mindepth 2 -maxdepth 2 -name DONE | grep -v '/_' | wc -l | tr -d ' ')"
echo "push_runs: $n_done finished runs, $moved unfinished moved aside"
"${SSH[@]}" "$GPU_HOST" "mkdir -p '$GPU_DIR/runs'"
rsync "${RSYNC_FLAGS[@]}" \
  --exclude 'checkpoints/' --exclude 'traces/' --exclude '_partial/' --exclude '_sweeps/' \
  --exclude '*.mp4' --exclude '*.png' \
  runs/ "$GPU_HOST:$GPU_DIR/runs/"
"${SSH[@]}" "$GPU_HOST" "cd '$GPU_DIR' && echo \"box now has \$(find runs -mindepth 2 -maxdepth 2 -name DONE | grep -v '/_' | wc -l) finished runs\""
