# Shared settings for scripts/gpu/*.sh (sourced, not run).
# GPU_HOST: ssh alias of the rented box (~/.ssh/config `Host gpu`; update HostName/Port per
#           vast.ai instance, F12). GPU_DIR: repo location on the box.
GPU_HOST="${GPU_HOST:-gpu}"
GPU_DIR="${GPU_DIR:-latency_budget}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Portable rsync flags only: macOS ships openrsync (F14), which lacks GNU-only options.
RSYNC_FLAGS=(-az --partial)
# GPU_SSH_CONFIG: optional ssh config file (local Docker test); default ~/.ssh/config.
SSH=(ssh)
if [[ -n "${GPU_SSH_CONFIG:-}" ]]; then
  SSH=(ssh -F "$GPU_SSH_CONFIG")
  RSYNC_FLAGS+=(-e "ssh -F $GPU_SSH_CONFIG")
fi
