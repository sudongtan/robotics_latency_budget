# Compute log

One row per GPU session. Mac work is not logged.

| Session | Date | Provider / GPU | Jobs run | Hours | Cost | Notes |
|---|---|---|---|---|---|---|

## Planned

**Session S** (single rental for all GPU work): backend benchmark, calibration against Brax PPO,
automatic gate, main sweep (A, B × 5 seeds; C, D, E, F × 10), ablations. 101 jobs. Estimated
≈ 36 h on one RTX 4090 / A100-class GPU, assuming 200k env-steps/s (19 h at 400k, 70 h at 100k;
the first job measures the real speed); ≈ 3 h until the gate. Job list:
`configs/sweeps/session_s.yaml`.

Instance requirements for session S (jobs run one at a time on one GPU):

| Resource | Need | Why |
|---|---|---|
| GPU | 1 × RTX 4090 / A100 class, ≥ 24 GB VRAM (16 GB likely enough) | 8192 envs + rollout buffer (≈ 0.3 GB) + eval chunks of 8192 envs; JAX preallocates 75 % of VRAM. A second GPU would sit idle |
| NVIDIA driver | recent (CUDA 12 capable); `bootstrap.sh` stops if JAX does not see the GPU | `jax[cuda12]` wheels bring their own CUDA libraries |
| CPU | 4–8 vCPUs | training is GPU-bound; CPU does XLA compilation (≈ 1–2 min per job) and Python |
| RAM | ≥ 32 GB (16 GB minimum) | CUDA/XLA runtime and compilation; traces (≈ 50 MB per set) pass through host memory |
| Disk | ≥ 50 GB | venv + CUDA wheels ≈ 7 GB, uv cache ≈ 7 GB, Go1 assets (menagerie) 2 GB, results ≈ 7 GB (checkpoints 1.6 MB × 10 × 98 runs, probing traces ≈ 4.5 GB: 128 episodes × 3 sets × 25 runs) |
| Rental type | on-demand, SSH as root | no bucket: results exist only on the box until pulled (D9) |
| Laptop | awake and on power for the whole session, ≥ 10 GB free | `sync_down.sh --watch` pulls results every 5 min |

**Priority tiers** (if the session is cut, the most important results already exist):

| Stopped after (≈ cumulative h) | Analyzable |
|---|---|
| calibration + gate (3 h) | our PPO vs Brax PPO; one-seed pilot curves |
| 3 seeds of the core conditions (10 h) | every pre-registered hypothesis, wide CIs |
| core conditions incl. observation-delay runs at 5 seeds (16 h) | all hypotheses at 5 seeds |
| main comparisons at 10 seeds (23 h) | the pre-registered seed counts |
| fixed-delay specialists (28 h) | + reference curves |
| exploratory ablations (36 h) | + motor-lag transfer, history length |

The cost cap runs only the tiers that fit in its budget (50 h) at the measured speed.

**Monitoring:** `scripts/gpu/health.py --watch 300` runs on the laptop next to the result pulls
and alerts (terminal + macOS notification + `runs/_health.log`) on low disk or memory on either
machine, an idle GPU, failed or stalled jobs, the gate result, a dead sweep runner, an unreachable
GPU machine, or pulls that stopped.

**If the GPU machine dies mid-session:** finished runs are already on the laptop (pulled every
5 min); only the run in progress is lost (≈ 20–30 min of GPU time). Restart on a new machine:
`push_code.sh <session commit>` (the commit in `runs/_ledger.jsonl`; run ids include it, so
another commit would retrain everything), `push_runs.sh <session commit>`, `bootstrap.sh`, then relaunch the same
sweep file. Finished runs are skipped, the interrupted one is retrained.
