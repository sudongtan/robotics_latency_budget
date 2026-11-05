# Latency Budget: how much delay can a learned robot controller handle?

Robots always act on slightly outdated information, because sensing, computing and moving
take time. This project measures how much delay a learned robot controller can handle
before it fails, and which training method makes it most tolerant. Today engineers mostly
find this out by trial and error.

## The question

A four-legged robot (Unitree Go1, simulated in MuJoCo) learns to walk with reinforcement
learning (PPO). Between the moment it senses its body and the moment its motors respond,
tens of milliseconds pass. In simulation this delay is usually zero; on real hardware it
is not.

The project asks:

- **How does performance degrade as delay grows?** Gradually, or does it fall off a cliff?
- **Which fix works best for the training cost?** Three common approaches are compared
  under identical conditions:
  1. **Randomizing the delay** during training.
  2. **Giving the policy a short history** of its recent actions.
  3. **Giving the policy memory** (a recurrent network, GRU).
- **Does it matter where the delay is?** Late sensor readings vs. late motor commands,
  and pure delay vs. a slow motor response.
- **Do memory-based policies learn to estimate their own delay?** Checked by probing
  their internal activations.

## Approach

- Six training conditions (naive baseline, fixed-delay specialists, randomization only,
  history, GRU, and an "oracle" that is told the true delay), all trained with the same
  algorithm, reward, network size and training budget, so differences come from the
  method alone.
- Each policy is tested across a grid of delays from 0 to 120 ms, including delays longer
  than any seen in training, plus changed friction, payload and motor strength.
- Hypotheses, the definition of success and the rule for judging each hypothesis are
  written down and frozen **before** the main experiments run
  ([`docs/preregistration.md`](docs/preregistration.md)); results are reported whether they
  support them or not.
- Statistics: 5 random seeds per condition, 10 for the four main comparison methods;
  interquartile mean with bootstrap confidence intervals; no best-seed numbers.
- Before the main experiments, our PPO is checked against the reference implementation
  (Brax PPO) on the baseline condition; the experiments run only if it matches.
- Every figure regenerates from saved runs with one command.

## Planned deliverables

- **Headline figure:** success rate vs. delay, one curve per method, with confidence intervals.
- **Side-by-side videos:** baseline vs. best method at ~60 ms delay.
- **Reproducible code:** one command per figure; reported compute cost.

## Progress

| Phase | Status |
|---|---|
| 0. Setup: environment, tooling, simulator running | Done: simulator runs and renders on laptop; environment documented; tooling and experiment tracking set up |
| 1. Delay and motor-lag simulation, with tests | Done: action/observation delay (4 ms resolution, jitter), explicit motor control with motor lag; 58 tests incl. no-future-leakage and parity with the original simulator; demo video of a feedback controller failing at 80 ms delay |
| 2. Own PPO implementation, validated against a reference | Laptop part done: JAX PPO with MLP, history and GRU policies, matching the Playground/Brax reference setup; tested against NumPy and Brax references; all three policies learn in short laptop runs; evaluation grid, gate and full GPU sweep built and rehearsed. Validation against Brax runs at the start of the single GPU session |
| 3. Pre-registration of hypotheses | Done: hypotheses, success definition and decision rules frozen before any main-sweep run (`docs/preregistration.md`) |
| 4. Main experiment sweep (GPU) | Not started |
| 5. Evaluation and first results | Not started |
| 6. Ablations | Not started |
| 7. Analysis: cross-evaluation, probing, gait | Not started |
| 8. Paper, videos, write-up | Not started |

Current state: everything the GPU session runs is built and rehearsed on the laptop, and the pre-registration is frozen. Next is the single GPU session: calibration against Brax PPO, an automatic gate, the main sweep and the ablations.

## Tech stack

Python, JAX, Flax, MuJoCo / MJX, MuJoCo Playground, Hydra, Weights & Biases, rliable.
Development and analysis run on a laptop (Apple Silicon); only training runs on rented GPUs.

## Running it

Requires [uv](https://docs.astral.sh/uv/) and macOS (Apple Silicon) or Linux.

```bash
uv sync                                         # install pinned dependencies
uv run pytest -q                                # 116 tests (~4 min): delays, no future leakage,
                                                # parity, PPO vs. NumPy/Brax references, evaluation
uv run python scripts/delay_feedback_demo.py    # demo video -> runs/phase1_delay_demo_*/
scripts/mjpython.sh scripts/view_env.py         # macOS: interactive viewer of the robot model

# Short training run on a laptop (~10 min): baseline condition A, small batch
uv run python scripts/train.py experiment=A ppo.num_envs=64 ppo.num_minibatches=4 \
    train.total_steps=2e6 train.run_prefix=smoke_ mjx_eval=smoke
uv run python scripts/plot_curves.py runs/smoke_A_*      # -> runs/<id>/curves.png
```

A full run (`uv run python scripts/train.py experiment=C seed=0`) trains for 200M steps with
8192 parallel simulations, then evaluates the policy on the whole delay grid; it is meant for
a GPU. Conditions are `A`, `B20`, `B40`, `B60`, `C`, `D`, `E`, `F` and `ablations/*`. Each run
writes `runs/<experiment>_<policy>_s<seed>_<commit>/`: config, commit hash, training curves,
checkpoints and per-episode evaluation results (`eval_mjx.parquet`).

All GPU work happens in one rented session, driven from the laptop:

```bash
bash scripts/gpu/push_code.sh       # laptop -> GPU machine (committed code only)
bash scripts/gpu/bootstrap.sh       # on the GPU machine: install, check GPU, short test run
bash scripts/gpu/launch_sweep.sh configs/sweeps/session_s.yaml   # on the GPU machine
bash scripts/gpu/sync_down.sh --watch                            # laptop: pull results during the run
```

The sweep runs a speed benchmark, the comparison with Brax PPO, an automatic pass/fail check
([`scripts/gpu/gate.py`](scripts/gpu/gate.py)), then the main experiments and ablations.

## Repository guide

- `docs/` — pre-registration (frozen), environment notes, compute log, results (as they are written).
- `src/latrl/envs/` — the simulation layer: `latency.py` (delay buffers), `actuator.py`
  (motor control and motor lag), `go1.py` (the robot environment with delays added),
  `specs.py` (observation layout and delay settings shared by all code).
- `src/latrl/ppo/` — the training algorithm (PPO): `trainer.py` (parallel simulation, rollouts,
  updates; one code path for feedforward and recurrent policies), `gae.py` (advantage
  estimates), `normalizer.py`, `checkpoint.py`.
- `src/latrl/policies/` — the networks (`networks.py`: MLP and GRU, matched in size), what each
  method's policy sees (`inputs.py`: observation, history, or observation + true delay), and the
  action distribution.
- `src/latrl/eval/mjx_eval.py` — evaluation after training: the delay grid, falls, velocity
  tracking, energy use, smoothness and gait statistics per episode; recordings for probing.
- `configs/` — all settings (Hydra); no hard-coded numbers in code. `configs/experiment/` holds one
  file per training condition, `configs/sweeps/session_s.yaml` the full GPU job list.
- `scripts/` — runnable entry points (`train.py`, `brax_reference.py`, `plot_curves.py`, demos);
  `scripts/gpu/` — the GPU session tooling. `tests/` — the test suite.
- `runs/` — every result, with its config, commit hash and seed (not in git; synced from the GPU).
