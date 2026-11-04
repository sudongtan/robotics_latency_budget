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
- Hypotheses are written down and committed **before** the main experiments run, and the
  results are reported whether they support them or not.
- Statistics: at least 5 random seeds per condition, interquartile mean with bootstrap
  confidence intervals; no best-seed numbers.
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
| 2. Own PPO implementation, validated against a reference | Not started |
| 3. Pre-registration of hypotheses | Not started |
| 4. Main experiment sweep (GPU) | Not started |
| 5. Evaluation and first results | Not started |
| 6. Ablations | Not started |
| 7. Analysis: cross-evaluation, probing, gait | Not started |
| 8. Paper, videos, write-up | Not started |

Current state: simulation layer complete; next is the PPO implementation (Phase 2).

## Tech stack

Python, JAX, Flax, MuJoCo / MJX, MuJoCo Playground, Hydra, Weights & Biases, rliable.
Development and analysis run on a laptop (Apple Silicon); only training runs on rented GPUs.

## Repository guide

- `docs/` — pre-registration, environment notes, compute log, results (as they are written).
- `src/latrl/`, `scripts/`, `configs/` — code (as it lands).
