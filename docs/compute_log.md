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
| Disk | ≥ 50 GB | venv + CUDA wheels ≈ 7 GB, uv cache ≈ 7 GB, Go1 assets (menagerie) 2 GB, results ≈ 3 GB (checkpoints 1.6 MB × 10 × 98 runs, traces ≈ 1.1 GB) |
| Rental type | on-demand, SSH as root | no bucket: results exist only on the box until pulled (D9) |
| Laptop | awake and on power for the whole session, ≈ 3 GB free | `sync_down.sh --watch` pulls results every 5 min |

**If the GPU machine dies mid-session:** finished runs are already on the laptop (pulled every
5 min); only the run in progress is lost (≈ 20–30 min of GPU time). Restart on a new machine:
`push_code.sh <session commit>` (the commit in `runs/_ledger.jsonl`; run ids include it, so
another commit would retrain everything), `push_runs.sh <session commit>`, `bootstrap.sh`, then relaunch the same
sweep file. Finished runs are skipped, the interrupted one is retrained.
