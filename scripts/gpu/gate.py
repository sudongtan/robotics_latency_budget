"""Automatic gate between calibration and the main sweep of session S (D27, D29).

Reads the calibration runs in runs/ and checks pass criteria fixed before renting. Exit 0 =
pass (the sweep continues), 1 = fail (launch_sweep.sh stops the queue). Writes
runs/_gate/<commit>.json with every number it used, so the decision is auditable (R5).

    uv run python scripts/gpu/gate.py              # checks runs of the current commit
    uv run python scripts/gpu/gate.py --commit abc1234 --runs-dir runs

Criteria (all must hold; seeds averaged, never best-seed):
- G1 complete: our A seeds 0, 1, Brax-on-our-env seeds 0, 1, and our C, D, E seed 0 finished
  (DONE) with finite final eval returns.
- G2 parity with Brax: mean final eval return of our A ≥ 0.9 × Brax's (same env, same config,
  both deterministic evals).
- G3 converged: for our A, the gain from the eval at ≈80 % of training to the final eval is
  ≤ 5 % of the final value (else the step budget is too small for the main sweep); needs
  ≥ 5 evals so the 80 % point is resolved.
- G4 walks: our A's final eval survival (no upstream termination over 20 s) ≥ 0.9.
- G5 pipeline: C, D, E each wrote eval_mjx.parquet with the full grid (82 cells × episodes).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys

import pandas as pd

RATIO_MIN, GAIN_MAX, SURVIVAL_MIN, MIN_EVALS = 0.9, 0.05, 0.9, 5


def _evals(run: pathlib.Path) -> list[dict]:
    path = run / "eval.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _final(run: pathlib.Path, key: str) -> float:
    ev = _evals(run)
    return float(ev[-1][key]) if ev and ev[-1].get(key) is not None else math.nan


def _at_fraction(run: pathlib.Path, key: str, frac: float) -> float:
    ev = _evals(run)
    if not ev:
        return math.nan
    target = frac * ev[-1]["env_steps"]
    row = min(ev, key=lambda r: abs(r["env_steps"] - target))
    return float(row[key])


def check(runs_dir: pathlib.Path, commit: str, n_cells: int) -> dict:
    def run(name: str) -> pathlib.Path:
        return runs_dir / f"{name}_{commit}"

    ours_a = [run(f"A_mlp_s{s}") for s in (0, 1)]
    brax_a = [run(f"brax_ours_s{s}") for s in (0, 1)]
    pipeline = [run("C_mlp_s0"), run("D_hist4_s0"), run("E_gru_s0")]
    out: dict = {"commit": commit, "criteria": {}}
    c = out["criteria"]

    finals_ours = [_final(r, "eval/episode_return") for r in ours_a]
    finals_brax = [_final(r, "eval/episode_reward") for r in brax_a]
    finals_pipe = [_final(r, "eval/episode_return") for r in pipeline]
    done = all((r / "DONE").exists() for r in ours_a + brax_a + pipeline)
    finite = all(math.isfinite(v) for v in finals_ours + finals_brax + finals_pipe)
    c["G1_complete"] = {"pass": done and finite, "done": done, "finite": finite}

    mo, mb = sum(finals_ours) / 2, sum(finals_brax) / 2
    ratio = mo / mb if mb > 0 else math.nan
    c["G2_parity"] = {
        "pass": ratio >= RATIO_MIN,
        "ours": finals_ours,
        "brax": finals_brax,
        "ratio": ratio,
        "min": RATIO_MIN,
    }

    at80 = [_at_fraction(r, "eval/episode_return", 0.8) for r in ours_a]
    gain = (mo - sum(at80) / 2) / abs(mo) if mo else math.nan
    n_evals = min(len(_evals(r)) for r in ours_a)  # < 5 evals: "80 %" point is not resolved
    c["G3_converged"] = {
        "pass": gain <= GAIN_MAX and n_evals >= MIN_EVALS,
        "at_80pct": at80,
        "final": finals_ours,
        "relative_gain": gain,
        "max": GAIN_MAX,
        "n_evals": n_evals,
    }

    surv = [_final(r, "eval/survived") for r in ours_a]
    c["G4_walks"] = {"pass": sum(surv) / 2 >= SURVIVAL_MIN, "survived": surv, "min": SURVIVAL_MIN}

    rows = {}
    for r in pipeline:
        p = r / "eval_mjx.parquet"
        rows[r.name] = len(pd.read_parquet(p)) if p.exists() else 0
    episodes = None
    for r in pipeline:
        cfg_path = r / "config.yaml"
        if cfg_path.exists():
            from omegaconf import OmegaConf

            episodes = int(OmegaConf.load(cfg_path).mjx_eval.episodes)
    expected = n_cells * (episodes or 0)
    c["G5_pipeline"] = {
        "pass": expected > 0 and all(v == expected for v in rows.values()),
        "rows": rows,
        "expected": expected,
    }

    out["pass"] = all(v["pass"] for v in c.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=pathlib.Path, default=pathlib.Path("runs"))
    ap.add_argument("--commit", default=None)
    ap.add_argument("--cells", type=int, default=None, help="grid cells (default: full.yaml)")
    args = ap.parse_args()
    commit = (
        args.commit
        or subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    if args.cells is None:
        from omegaconf import OmegaConf

        from latrl.eval.mjx_eval import grid

        root = pathlib.Path(__file__).resolve().parents[2]
        args.cells = len(grid(OmegaConf.load(root / "configs" / "mjx_eval" / "full.yaml")))
    result = check(args.runs_dir, commit, args.cells)
    out_dir = args.runs_dir / "_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{commit}.json").write_text(json.dumps(result, indent=2, default=str))
    for k, v in result["criteria"].items():
        print(f"{k}: {'PASS' if v['pass'] else 'FAIL'}  {v}")
    print("GATE", "PASS" if result["pass"] else "FAIL")
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
