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

(Added after the freeze, labeled, never replacing text above.)

### 2026-11-05: clarifications and deviations before any data

Written after a review of the frozen text and before session S (no main-sweep run exists).
They resolve ambiguities and fix one modelling error; no hypothesis or threshold changes.

**Clarifications of the decision rules (§5, §6)**

1. **Censored L50.** If success stays ≥ 50 % at every action latency up to 120 ms, that seed's
   L50 is right-censored and enters IQMs and differences as 120 ms (conservative: it understates
   the more robust condition). The number of censored seeds per condition is reported. Same rule
   for L90 / L10 in W (§5). H4's robustness part: if at least half the seeds of D or of E are
   censored, the L50 comparison is uninformative and H4 is judged instead on mean success over the
   action-latency grid 0–120 ms (no jitter), equivalence margin ±5 percentage points (90 % CI).
2. **H5 pairing.** "Same-channel" and "cross-channel" refer to the same policy: a policy trained
   with action delay (C, D, E) evaluated on the action-latency axis (same channel) and on the
   observation-latency axis (cross channel) at equal total delay; for the observation-trained
   policies (the ablations C, D, E with observation delay) the roles swap. The rule must hold for
   each of the six policies (IQM over seeds) at 20, 40, …, 120 ms. Points where both success values
   are ≥ 95 % or both ≤ 5 % are reported as floor/ceiling points; H5 is "supported" only if at most
   half of each policy's points are floor/ceiling, otherwise "inconclusive".
3. **Nominal tracking error (H2, H3).** Per-episode linear-velocity tracking error (`lin_err`,
   m/s, excluding the first 0.5 s), at 0 ms action delay, jitter off, non-fallen episodes only;
   mean over episodes per seed; IQM over seeds. H3's recovery R uses these IQMs, with a bootstrap
   CI; if the CI of err_C − err_A includes 0, H3 is "inconclusive" (nothing to recover).
4. **Wall-clock (H4).** Total training wall-clock for 200M steps (excluding the evaluation grid),
   measured on the same GPU and backend within one session. Time to 90 % of final return (§5) is
   reported as a secondary measure and is not used for H4.
5. **Condition F's latency input under jitter** is the delay drawn at the previous control step
   (one step stale when jitter changes it). Jitter results are exploratory (§7).

**Deviations before data**

1. **Actuator-lag model** (affects the motor-lag ablation and evaluation axis only). The first
   implementation low-pass filtered the whole PD torque. That puts the lag inside each joint's
   feedback loop, which is unstable for T_m > Kd/Kp ≈ 14 ms: the robot falls at zero action, before
   the policy acts. Most of the planned range (training U[0, 40 ms], evaluation 0–60 ms) would have
   tested an unstable joint controller rather than a slow actuator. The lag now filters the joint
   target the PD loop tracks; the loop is stable for any T_m. Ranges unchanged.
2. **Critic input (§3).** "The critic never sees the latency" should read "never sees the latency
   value". Its input (Playground's `privileged_state`) includes the delivered joint torques, which
   are computed from the delayed target, so they carry an indirect latency signal. This is the same
   for every condition; no actor sees it.
3. **Operational:** probing traces use 128 episodes per set (was 32), and the GPU session stops
   automatically if its projected hours exceed a cap. Neither affects any hypothesis test.
   Robustness at intermediate training checkpoints is computed afterwards on the laptop from the
   saved checkpoints (exploratory, for "robustness per unit of training cost").
