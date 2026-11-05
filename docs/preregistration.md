# Pre-registration

**Status: FROZEN** (approved by the owner on 2026-11-05). Nothing above "Post-hoc notes" changes
after this point (project rule R6); later notes go in that section, labeled.

## Freeze

- Hypotheses frozen at commit: `c29e780`
- Date: 2026-11-05

## 1. Question

For a quadruped (Unitree Go1, MuJoCo Playground joystick task on flat ground, 50 Hz control)
trained with PPO in simulation, how does sensing-to-actuation latency degrade velocity
tracking, and which mitigation buys the most robustness per unit of training cost: latency
randomization, observation/action history, a recurrent policy, or knowing the true latency?

H1 and H2 replicate, for a quadruped velocity tracker, results that Li & Sun (arXiv
2610.32277, 2026) report for a humanoid motion tracker. H3–H6 are not tested there.

Our 0–80 ms training range and 0–120 ms test range go beyond typical proprioceptive latency
(~5–30 ms on real robots); 40–100 ms is the regime of vision or offboard compute.

## 2. Hypotheses

- **H1** (replication) A policy trained at 0 ms latency fails sharply, not gracefully, beyond a
  critical latency of ~2–3 control steps (40–60 ms).
- **H2** (replication) Latency randomization raises the critical latency but lowers nominal
  (0 ms) performance.
- **H3** Adding action history to the observation recovers most of the nominal cost of H2.
- **H4** A GRU policy matches history stacking in robustness but costs more wall-clock to train.
- **H5** At equal total delay on the 20 ms grid, observation and action latency are
  interchangeable for constant delay: a policy trained on one channel and evaluated on the other
  is within 10 percentage points of success of same-channel evaluation, at every test latency
  that is a multiple of 20 ms. (Delayed-MDP theory predicts this; sub-step delays and jitter are
  exploratory.)
- **H6** History/GRU policies' internal features linearly encode the current latency (probe R²
  clearly above a baseline-policy and a random-network control). Prior evidence is weak
  (arXiv 2603.21268: R² 0.01 linear / 0.04 MLP from an LSTM with five randomized factors).

## 3. Conditions

All conditions: same reward, same PPO configuration (`configs/ppo/default.yaml`, copied from
the Playground Go1 reference), same total environment steps (200M), actor networks within
±10 % parameter count. The critic is identical for all conditions and never sees the latency.

| ID | Training action latency | Actor input | Actor | Seeds |
|---|---|---|---|---|
| A | 0 | observation | MLP | 5 |
| B20 / B40 / B60 | fixed 20 / 40 / 60 ms | observation | MLP | 5 each |
| C | U{0..80 ms} per episode | observation | MLP | 10 |
| D | U{0..80 ms} | last 4 observation frames (incl. last 4 actions) | MLP | 10 |
| E | U{0..80 ms} | observation | GRU | 10 |
| F | U{0..80 ms} | observation + true latency | MLP | 10 |

Ablations (pre-committed): D with history length H ∈ {2, 8} (3 seeds; H = 1 is C, H = 4 is D);
C, D, E trained with observation delay U{0..80 ms} only (5 seeds); C, D, E trained with motor
time constant T_m ~ U[0, 40 ms] only (5 seeds).

Seeds are 0..n−1 for every condition; no run is excluded or replaced. A run that fails to
learn is reported, not rerun with other settings.

## 4. Evaluation

Every run is evaluated in MJX after training with the deterministic policy, 256 episodes of
20 s per test setting, the same episodes (initial state, commands, dynamics draw) for every
setting and every run:

- action latency 0–120 ms every 4 ms, jitter off and on (±12 ms, p = 0.1 per step; exploratory);
- observation latency 0–120 ms every 20 ms;
- motor time constant 0–60 ms every 5 ms.

**Fall:** trunk height < 0.15 m or tilt > 60° at any control step.

**Success:** no fall for 20 s **and** mean linear-velocity tracking error (excluding
the first 0.5 s) ≤ τ, where τ is the 95th percentile of the per-episode tracking error of
condition A's non-fallen episodes at 0 ms (pooled over A's 5 seeds). τ is computed once from
those runs before any other condition is analysed and recorded here as a post-hoc number.

## 5. Metrics and statistics

Per seed: success rate per test setting; `L50` = smallest test latency with success < 50 %
(linear interpolation on the grid); delay margin (Li & Sun) = largest grid latency with success
≥ 50 % there and at every smaller latency; `L90`, `L10` and transition width `W = L10 − L90`
(right-censored if success stays ≥ 10 % up to 120 ms). Also tracking errors, cost of transport,
action jerk, gait statistics, and wall-clock to reach 90 % of the run's final return.

Across seeds: interquartile mean (IQM) with stratified bootstrap 95 % CIs (rliable,
5000 resamples). Differences between conditions use bootstrap CIs of the difference of IQMs.
No best-seed numbers. Six hypotheses, one primary test each; no multiplicity correction; all
results reported.

## 6. Decision rules

| | Supported if |
|---|---|
| H1 | IQM(W) of A ≤ 25 ms **and** IQM(L50) of A in [30, 70] ms |
| H2 | L50(C) − L50(A) > 0 with CI excluding 0 **and** nominal tracking error at 0 ms of C − A > 0 with CI excluding 0 |
| H3 | recovery R = (err_C − err_D) / (err_C − err_A) at 0 ms ≥ 0.5, CI lower bound > 0 (only tested if H2's nominal cost holds) |
| H4 | 90 % CI of L50(E) − L50(D) within ±10 ms **and** wall-clock ratio E / D > 1 with CI excluding 1 |
| H5 | for each of C, D, E: \|success(cross-channel) − success(same-channel)\| ≤ 10 pp at every multiple of 20 ms from 20 to 120 ms (IQM over seeds) |
| H6 | ridge-probe R² (episode-level 5-fold split) of D and E features exceeds both controls (A's features, a random network) by ≥ 0.2, CI lower bound above the controls' CI upper bound |

"Not supported" if the rule fails with CIs that exclude the supporting region; otherwise
"inconclusive".

## 7. Exploratory (not hypotheses)

Jitter, sub-step delays, delay vs. motor-lag cross-evaluation, sim-to-sim transfer (MJX →
native MuJoCo), dynamics perturbations, gait changes with latency, time-resolved latency probes.

## Post-hoc notes

(Added after the main sweep starts, labeled, never replacing text above.)
