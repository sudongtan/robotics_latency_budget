# The original hypotheses, their analysis, and what was wrong with the design

The project started with six pre-registered hypotheses and a decision rule for each
([`preregistration.md`](preregistration.md), frozen before any experiment and left unchanged).
The main README no longer reports results through them: it describes how each method performs
under delay. This page keeps the hypotheses, shows what the pre-registered analysis returned, and
explains why most of it says little. The data behind both is the same; only the analysis differs.

## Studies

| Name | What it was | Training delay |
|---|---|---|
| Original study (Nov 5–6) | 8 conditions + ablations, 96 runs; all six hypotheses tested ([`results.md`](results.md)) | 0–80 ms random |
| F1 (Nov 17) | Exploratory diagnosis of why most original policies never learned; MLP variants only, 20 runs | 0–40 ms, or fine-tuning |
| F2 (Nov 17–18) | Pre-registered rerun of H2, H3, H4 and H6 with the F1 fix, 50 runs | 0–40 ms random |

## The hypotheses

| | Prediction (abridged; full text in the pre-registration §2) | Decision rule (§6) |
|---|---|---|
| H1 | A policy trained without delay fails sharply, at 2–3 control steps (40–60 ms) | transition width W ≤ 25 ms and L50 in [30, 70] ms |
| H2 | Delay randomization raises the critical delay but lowers performance at 0 ms | L50(C) > L50(A) and tracking error at 0 ms C > A, both CIs excluding 0 |
| H3 | Adding history recovers most of H2's cost at 0 ms | recovery (err_C − err_D) / (err_C − err_A) ≥ 0.5, CI lower bound > 0 |
| H4 | A GRU matches history in robustness but costs more to train | 90 % CI of L50(E) − L50(D) within ±10 ms, and train time E/D > 1 |
| H5 | Sensor and motor delay of the same size hurt equally | cross- vs same-channel success within 10 pp at 20–120 ms, for C, D, E trained on either channel |
| H6 | History and GRU policies encode the current delay internally | probe R² of D and E above both controls (A, random network) by ≥ 0.2 |

A = no delay in training; C = random delay, MLP; D = random delay, MLP + last 4 observations;
E = random delay, GRU; F = random delay, MLP told the true delay.

## What the pre-registered analysis returned

| | Original study (0–80 ms) | F2 (0–40 ms) | What the verdict actually reflects |
|---|---|---|---|
| H1 | not supported: sharp (W = 17 ms) but at L50 = 20 ms, not 30–70 | not computed (F2 retrained A, so it could be) | A real test: the fall is sharp, but at one control step |
| H2 | not supported | **supported**: L50 +36 ms [8, 37] over A; error at 0 ms +0.0075 m/s [0.0055, 0.27] | Original: C never learned to walk (0/10). F2: a real test. F1 (exploratory) agrees |
| H3 | not supported | not supported: R = −88 [−117, −1] | Both times D stood still (0/10, then 8 of 10 seeds); history was never measured |
| H4 | not supported | not supported: L50 E − D = +66 ms [43, 66]; train time E/D = 1.9 | The cost part is real (GRU ≈ 2× slower); the robustness part compares the GRU with a standing policy |
| H5 | inconclusive: C, D and sensor-trained D at the floor | dropped: F2 trained no sensor-delay policies | Original: three of six policies never learned. Never tested since |
| H6 | not supported: D R² −0.06, E 0.58 [0.24, 0.77] | not supported: D −0.01, E 0.78 [0.71, 0.80] vs controls 0.41–0.43 | Fails on D, which stands. The GRU half passes in F2 |

Formal outputs: `runs/_analysis/hypotheses.json` (original) and `runs/_analysis_f2/hypotheses.json`
(F2), from `scripts/analyze.py` and `scripts/analyze_f2.py`.

Of twelve verdicts across the two studies, two are informative tests of the hypothesis as stated
(H1 original, H2 in F2), plus the cost half of H4. The rest are dominated by policies that did not
learn to walk.

## What was wrong with the design

1. **The hypotheses compared methods with each other in a chain, so one failed method took out
   several hypotheses.** H3 compares history with randomization, H4 the GRU with history, H6 both.
   Three hypotheses depend on the history policy D, and only in one variant (4 frames). D failed to
   learn twice, and all three became uninformative, although the GRU, which H4 and H6 are also
   about, worked well. Comparing each method with a common reference (A and C) would have isolated
   the failure to one hypothesis.
2. **Learnability was assumed, never checked before freezing.** The hypotheses presuppose that
   every method learns to walk under one fixed training recipe (rule R1: no per-condition tuning).
   No condition was piloted first. In the original study the 0–80 ms range stopped most methods
   from learning; in F2, D passed a weak gate (walks in 1 of 2 pilot seeds) and then walked in
   2 of 10.
3. **The decision rules had no outcome for "a condition did not learn".** They return "not
   supported" for comparisons against a policy that stands still, which reads as evidence against
   the hypothesis when nothing was measured.
4. **The robustness measure mixes learning and robustness.** L50 is averaged over all seeds, and a
   standing seed scores 0. A method that learns in few seeds therefore scores low however robust
   its walking seeds are (D: 0 ms over all seeds, about 66 ms over its 2 walking seeds).
5. **Hypotheses and runs drifted apart.** F2 dropped H5 to save GPU time (no sensor-delay runs) and
   did not compute H1, although its A runs allow it.
6. **H1's predicted range was a guess.** 40–60 ms came from intuition about control steps, not from
   a pilot. The baseline was measured only after the rule was frozen.

The questions behind the hypotheses were sound; the way they were turned into tests was not. The
README therefore reports what each method does under delay, descriptively, with all seeds and no
selection. Comparisons there are post-hoc, not pre-registered tests.
