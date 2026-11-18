# Latency Budget: how much delay can a learned robot controller handle?

Robots act on slightly outdated information, because sensing, computing and moving take time.
This project measures how much delay a learned walking controller (simulated Unitree Go1, PPO,
MuJoCo/MJX) tolerates before it fails, and which training method makes it most tolerant.

Between the moment a robot senses its body and the moment its motors respond, tens of
milliseconds pass. In simulation this delay is usually zero; on real hardware it is not. Today
engineers mostly find out how much delay a controller tolerates by trial and error.

## Research question

**How well does each method cope with delay?** Three common ways to make a learned controller
tolerate delay are trained under identical conditions and measured on the same tests:

1. **Randomizing the delay** during training (C).
2. **Adding a short history** of recent observations to what the policy sees (D).
3. **Giving the policy memory**, a recurrent network (GRU) (E).

They are compared with a controller trained without delay (A) and with an "oracle" that is told the
true delay (F). For each method: does it learn to walk, how much delay does it tolerate, what does
it cost at zero delay and in training time, and does it learn to estimate the delay internally?

The project began with six pre-registered hypotheses. They turned out to be poorly designed, and
most of their tests say little; the results below are reported without them. The hypotheses, the
analysis they produced and what was wrong with the design are on a separate page:
[`docs/hypotheses.md`](docs/hypotheses.md).

## Results

![Success rate vs. test action latency, F2](docs/figures/f2_success_vs_action_latency.png)

Five methods, same PPO recipe, reward, network size and training budget, 10 seeds each, tested on
action delays from 0 to 120 ms. Success = no fall in 20 s and velocity tracked within a fixed
tolerance. **L50** = the delay at which success drops to 50 %. Numbers are interquartile means over
all seeds with 95 % bootstrap CIs; no best-seed numbers.

| Method (trained on random delay 0–40 ms, except A) | Seeds that learn to walk | L50 (ms) | Success at 0 ms |
|---|---|---|---|
| A: no delay in training, MLP | 10/10 | 20 [20, 21] | 94 % |
| C: random delay, MLP | 8/10 | 56 [36, 57] | 89 % |
| D: random delay, MLP + last 4 observations | **2/10** | 0 [0, 22] | 3 % |
| **E: random delay, GRU (memory)** | 9/10 | **66 [54, 67]** | **93 %** |
| F: random delay, MLP told the true delay | 8/10 | 51 [26, 54] | 86 % |

1. **A controller trained without delay fails at about one control step.** It acts every 20 ms, and
   its success drops from 94 % to 50 % at 20 ms of delay and to almost 0 by 32 ms.
2. **Training with random delay triples the tolerable delay** (L50 20 → 56 ms for the plain MLP) at a
   small cost at zero delay (89 % vs 94 %).
3. **Memory (GRU) is the best method:** the highest tolerance (66 ms, well beyond the 40 ms seen in
   training), almost no cost at zero delay, and the most reliable training (9/10 seeds). It costs
   about 1.9× the training time of an MLP.
4. **The GRU learns to estimate the delay.** A linear probe reads the true delay from its memory
   (R² 0.78), far above the no-delay controller (0.41) and an untrained GRU (0.43).
5. **A short observation history fails to learn**: 8 of 10 seeds learn to stand still instead of
   walking. The 2 seeds that walk are as robust as the GRU (L50 ≈ 66 ms), but two seeds prove nothing.
6. **Telling the policy the true delay does not help** (F is no better than C).
7. **Training on too wide a delay range breaks learning.** The original study randomized delay over
   0–80 ms; there, the MLP-based methods never learned to walk (0/10) and only the GRU did (7/10).
   Narrowing to 0–40 ms (above), or fine-tuning a controller that already walks, fixes it: a
   fine-tuned MLP reaches L50 ≈ 81 ms (exploratory, 5 seeds).

Full numbers and caveats: [`docs/results.md`](docs/results.md) (original study) and
`runs/_analysis_f2/` (F2, `scripts/analyze_f2.py`). Limits: one simulator family, one
robot, one PPO recipe and budget; delay tested on the action side only in F2.

## How the study was run

- **Fixed definitions.** The definition of success, the delay grid and the statistics were frozen
  before the experiments ([`docs/preregistration.md`](docs/preregistration.md)) and are unchanged.
  The pre-registered hypotheses are no longer used to report results (see
  [`docs/hypotheses.md`](docs/hypotheses.md)).
- **Controlled.** Same algorithm, reward, network size and training budget for every method; our
  PPO was checked against the reference Brax PPO (98 % of its return) before any experiment.
- **Checked twice.** The original study's test-time results replicate in native MuJoCo (the CPU simulator) on the same
  episodes.
- **Cheap and reproducible.** Three rented RTX 4090 sessions, about 35 GPU hours and $13 in total
  ([`docs/compute_log.md`](docs/compute_log.md)); every number and figure regenerates from saved
  runs with one command.

### History

| Stage | What happened |
|---|---|
| Original study (Nov 2026) | 8 conditions + ablations, 0–80 ms randomized delay. Most methods never learned to walk; closed as a negative result ([`docs/results.md`](docs/results.md)) |
| Follow-up F1 | Diagnosis: the 0–80 ms range, not the critic, caused the failure; narrower range and fine-tuning both fix it |
| F2 | Method comparison rerun on 0–40 ms, 10 seeds per method (results above) |
| Not done | Videos, gait analysis, observation-side delay under 0–40 ms, a paper |

## References

Prior work, the theory behind each method, and the tools used: [`docs/references.md`](docs/references.md).
The closest prior work is Li & Sun (arXiv 2610.32277, 2026), who showed delay randomization widens
the delay margin of a humanoid motion tracker; this project replicates that on a quadruped and adds
the history, memory, oracle and probing comparisons.

## Tech stack

Python, JAX, Flax, MuJoCo / MJX, MuJoCo Playground, Hydra, Weights & Biases, rliable.
Development and analysis run on a laptop (Apple Silicon); only training runs on rented GPUs.

## Running it

Requires [uv](https://docs.astral.sh/uv/) and macOS (Apple Silicon) or Linux.

```bash
uv sync                                         # install pinned dependencies
uv run pytest -q                                # 180 tests (~6 min): delays, no future leakage,
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
a GPU. Conditions are `A`, `B20`, `B40`, `B60`, `C`, `D`, `E`, `F`, `ablations/*`, `followup/*` and `f2/*`. Each run
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
([`scripts/gpu/gate.py`](scripts/gpu/gate.py)), then the experiments in priority order: first
every condition needed for the hypotheses at 5 seeds, then extra seeds, then reference and
exploratory runs. If the session is cut short, or the measured speed means the budget only
covers part of it, the most important results are complete.
It survives losing the GPU machine: finished runs are listed in `runs/_ledger.jsonl` and skipped
when the sweep is relaunched on a new machine (`scripts/gpu/push_runs.sh <commit>`), so at most the
run in progress is repeated.

## Repository guide

- `docs/` — pre-registration (frozen), original hypotheses and their problems, results, references,
  environment notes, compute log.
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
