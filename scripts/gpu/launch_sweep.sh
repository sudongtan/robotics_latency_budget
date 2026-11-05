#!/usr/bin/env bash
# Run ON the GPU box: run every job of a sweep file in sequence (one GPU), detached with
# nohup so the ssh session can drop. Logs: runs/_sweeps/<sweep>_<time>/{jobNN.log,status}.
# The Mac pulls runs/ during the sweep with sync_down.sh --watch.
#   bash scripts/gpu/launch_sweep.sh configs/sweeps/s1_calibration.yaml
#   bash scripts/gpu/launch_sweep.sh --dry-run configs/sweeps/s1_calibration.yaml
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
# No MUJOCO_GL default: training does not render, and MUJOCO_GL=egl without libEGL makes
# `import mujoco` fail. Set MUJOCO_GL=egl only for rendering jobs (CLAUDE.md).

dry=0
if [[ "${1:-}" == "--dry-run" ]]; then dry=1; shift; fi
sweep="${1:?usage: launch_sweep.sh [--dry-run] configs/sweeps/<name>.yaml}"
cmds="$(uv run python scripts/gpu/sweep.py "$sweep")"   # one command per line
if [[ $dry == 1 ]]; then echo "$cmds"; exit 0; fi
n_jobs="$(echo "$cmds" | wc -l | tr -d ' ')"

logdir="runs/_sweeps/$(basename "$sweep" .yaml)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$logdir"
echo "$cmds" > "$logdir/commands.txt"
cat > "$logdir/run.sh" <<RUN
#!/usr/bin/env bash
cd "$PWD"
i=0
while IFS= read -r cmd; do
  i=\$((i + 1)); log=\$(printf '%s/job%02d.log' "$logdir" \$i)
  echo "\$(date '+%FT%T%z') start \$i: \$cmd" >> "$logdir/status"
  if bash -c "\$cmd" < /dev/null > "\$log" 2>&1; then r=ok; else r="FAILED(\$?)"; fi
  echo "\$(date '+%FT%T%z') \$r \$i" >> "$logdir/status"
done < "$logdir/commands.txt"
echo "\$(date '+%FT%T%z') SWEEP DONE" >> "$logdir/status"
RUN
nohup bash "$logdir/run.sh" > "$logdir/nohup.log" 2>&1 &
echo "launched $n_jobs jobs (pid $!); status: $logdir/status"
