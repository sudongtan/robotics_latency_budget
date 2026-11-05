#!/usr/bin/env bash
# Run ON the GPU box, from the repo pushed by push_code.sh. Installs uv, syncs the same
# uv.lock as the Mac with the CUDA extra, checks JAX sees the GPU, runs the test suite's
# fast subset, then a ~2-minute smoke training run (CLAUDE.md Compute).
#   bash scripts/gpu/bootstrap.sh            # full
#   BOOTSTRAP_CPU=1 SMOKE_ENVS=16 SMOKE_STEPS=3200 SMOKE_ARGS='ppo.num_minibatches=2 train.eval_envs=4' \
#     bash scripts/gpu/bootstrap.sh   # local test without a GPU (no CUDA extra, tiny smoke)
set -euo pipefail
cd "$(dirname "$0")/../.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if [[ "${BOOTSTRAP_CPU:-0}" == 1 ]]; then
  uv sync --frozen
else
  uv sync --frozen --extra cuda
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

uv run python - <<'PY'
import os, jax
devs = jax.devices()
print("jax", jax.__version__, devs)
if os.environ.get("BOOTSTRAP_CPU") != "1":
    assert devs[0].platform == "gpu", f"JAX does not see a GPU: {devs}"
PY

# Fast tests only (the full suite runs on the Mac before every session; CLAUDE.md).
uv run pytest -q tests/test_gae.py tests/test_ppo_shapes.py tests/test_policy_inputs.py

# ~2-minute smoke run: real config, small budget. Overwrites nothing (run dirs are unique;
# delete an old runs/gpusmoke_* first when re-bootstrapping the same commit).
SMOKE_STEPS="${SMOKE_STEPS:-5e6}"
SMOKE_ENVS="${SMOKE_ENVS:-8192}"
rm -rf runs/gpusmoke_A_mlp_s0_*
uv run python scripts/train.py experiment=A seed=0 train.run_prefix=gpusmoke_ \
  ppo.num_envs="$SMOKE_ENVS" train.total_steps="$SMOKE_STEPS" train.num_evals=1 \
  train.eval_envs=128 train.log_every=5 ${SMOKE_ARGS:-}
echo "bootstrap: OK"
