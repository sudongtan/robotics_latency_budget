#!/usr/bin/env bash
# Mac -> GPU box: copy the source runs that fine-tune jobs start from (follow-up F1, C_ft).
# Only what scripts/train.py:_init_from reads: DONE, config.yaml, meta.json and the final
# checkpoint. Run after push_code.sh and before launching the sweep.
#   bash scripts/gpu/push_init.sh                     # A_mlp_s0..4 at 4a4a467 (default)
#   bash scripts/gpu/push_init.sh 'A_mlp_s*_4a4a467'
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
pattern="${1:-A_mlp_s[0-4]_4a4a467}"
files=()
for d in runs/$pattern; do
  [[ -f "$d/DONE" ]] || { echo "push_init: $d is not a finished run" >&2; exit 1; }
  ck="$(ls "$d"/checkpoints/it*.msgpack | sort | tail -1)"
  files+=("$d/DONE" "$d/config.yaml" "$d/meta.json" "$ck")
done
(( ${#files[@]} )) || { echo "push_init: no runs match $pattern" >&2; exit 1; }
printf '%s\n' "${files[@]}" > /tmp/push_init_files.$$
rsync "${RSYNC_FLAGS[@]}" --files-from=/tmp/push_init_files.$$ ./ "$GPU_HOST:$GPU_DIR/"
rm -f /tmp/push_init_files.$$
echo "push_init: copied $(( ${#files[@]} / 4 )) source runs ($pattern)"
