#!/usr/bin/env bash
# Mac: pull runs/ from the GPU box (D9: no bucket; pull, since the box cannot reach the Mac).
#   bash scripts/gpu/sync_down.sh               # one pull, then verify nothing is left to copy
#   bash scripts/gpu/sync_down.sh --watch [s]   # pull every s seconds (default 300) while a job
#                                               # runs; keeps the Mac awake (caffeinate)
# Only after a verified final pull may the instance be destroyed (CLAUDE.md Compute).
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
mkdir -p runs

pull() { rsync "${RSYNC_FLAGS[@]}" "$GPU_HOST:$GPU_DIR/runs/" runs/; }

verify() {
  # Dry run with checksums: lists any file that still differs from the box.
  local pending
  pending="$(rsync -anc --out-format='%n' "$GPU_HOST:$GPU_DIR/runs/" runs/ | grep -v '/$' || true)"
  if [[ -n "$pending" ]]; then
    echo "sync_down: NOT verified; still differing:" >&2
    echo "$pending" >&2
    return 1
  fi
  echo "sync_down: verified, local runs/ matches $GPU_HOST:$GPU_DIR/runs/"
}

if [[ "${1:-}" == "--watch" ]]; then
  interval="${2:-300}"
  if [[ -z "${CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
    CAFFEINATED=1 exec caffeinate -i "$0" "$@"
  fi
  while true; do
    if pull; then echo "$(date '+%F %T') pulled"; else echo "$(date '+%F %T') pull failed" >&2; fi
    sleep "$interval"
  done
else
  pull
  verify
fi
