#!/usr/bin/env bash
# Mac -> new GPU box, after a crash (D31): tell the box which policies are already trained.
# Only this session's runs are touched: directories whose name ends in _<session commit>.
# 1. Locally, this session's unfinished run directories (no DONE: the run the crash
#    interrupted, pulled half-way) are moved to runs/_partial/, so the retrained run lands in
#    a clean directory. Other directories (older runs, demos) are left alone.
# 2. This session's finished runs are copied to the box WITHOUT checkpoints and traces:
#    only the small files the sweep reads (DONE, configs, metrics, eval results), plus the
#    ledger and gate records. Relaunching the sweep then skips every finished job; the
#    backend pick and the gate re-check from these files in seconds.
#   bash scripts/gpu/push_code.sh <commit> && bash scripts/gpu/push_runs.sh <commit>
#   (<commit> = the session's commit, in runs/_ledger.jsonl), then bootstrap + relaunch.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
sha="$(git rev-parse --short "${1:?usage: push_runs.sh <session commit>}^{commit}")"
mkdir -p runs
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
moved=0
for d in runs/*_"$sha"/; do
  [[ -d "$d" ]] || continue
  d="${d%/}"; name="$(basename "$d")"
  [[ "$name" == _* ]] && continue
  if [[ ! -f "$d/DONE" ]]; then
    mkdir -p runs/_partial
    mv "$d" "runs/_partial/${name}_${stamp}"
    echo "push_runs: moved unfinished $name -> runs/_partial/"
    moved=$((moved + 1))
  fi
done
n_done="$(ls -d runs/*_"$sha"/DONE 2>/dev/null | wc -l | tr -d ' ')"
echo "push_runs: session $sha: $n_done finished runs, $moved unfinished moved aside"
"${SSH[@]}" "$GPU_HOST" "mkdir -p '$GPU_DIR/runs'"
# Filter order matters: the first matching rule wins.
rsync "${RSYNC_FLAGS[@]}" \
  --exclude 'checkpoints/' --exclude 'traces/' \
  --include "*_${sha}/" --include "*_${sha}/**" \
  --include '_ledger.jsonl' --include '_gate/' --include "_gate/${sha}.json" \
  --exclude '*' \
  runs/ "$GPU_HOST:$GPU_DIR/runs/"
"${SSH[@]}" "$GPU_HOST" "cd '$GPU_DIR' && echo \"box now has \$(find runs -mindepth 2 -maxdepth 2 -name DONE | grep -v '/_' | wc -l) finished runs\""
