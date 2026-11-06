"""Pre-registered analysis of session S (prereg §4–§6): tables and verdicts H1–H5.

    uv run python scripts/analyze.py                 # configs/analysis/default.yaml
    uv run python scripts/analyze.py n_boot=500      # quick look (not for reporting)

Reads only finished runs at the session commit (R5) and writes `runs/_analysis/`:
    per_setting.parquet   success / fall / gait per (run, axis, test latency)
    per_run.csv           L50, L90, L10, W, delay margin, nominal metrics, timing per run
    conditions.csv        IQM + 95 % CI per condition for the per-run metrics
    hypotheses.json       verdicts H1–H5 with the estimates behind them
    meta.json             analysis commit, config, run list

First recomputes τ from condition A alone and stops if it differs from the recorded value (D39).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from omegaconf import OmegaConf

from latrl.analysis import hypotheses as H
from latrl.analysis import stats as S
from latrl.analysis import summarize as sm
from latrl.utils.provenance import git_commit

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY_COLS = [
    "L50",
    "margin",
    "L90",
    "L10",
    "W",
    "mean_success",
    "L50_jit",
    "mean_success_jit",
    "L50_obs",
    "mean_success_obs",
    "L50_tm",
    "mean_success_tm",
    "nominal_success",
    "nominal_lin_err",
    "nominal_yaw_err",
    "nominal_cot",
    "nominal_jerk",
    "train_s",
    "t90_train_s",
    "final_return",
]


def recompute_tau(runs, cfg) -> float:
    a = runs[runs["cond"] == "A"]
    ep = sm.load_episodes(list(a["path"]), tau=np.inf)
    z = ep[(ep["axis"] == "action") & (ep["x_ms"] == 0.0) & ~ep["fell"]]
    return float(np.percentile(z["lin_err"], cfg["tau_percentile"]))


def condition_table(pr, cfg, rng):
    rows = []
    for cond, g in pr.groupby("cond", sort=True):
        row = {"cond": cond, "n_seeds": len(g), "L50_censored": int(g["L50_cens"].sum())}
        for c in SUMMARY_COLS:
            v = g.sort_values("seed")[c].to_numpy(dtype=float)
            if np.isnan(v).any():
                # Delay margin is undefined for a seed below 50 % success at 0 ms: report the
                # count instead of an IQM over the seeds where it exists (selection).
                row[f"{c}_defined"] = int((~np.isnan(v)).sum())
                continue
            e = S.iqm_ci(v, cfg["n_boot"], rng, cfg["ci"])
            row |= {c: e.point, f"{c}_lo": e.lo, f"{c}_hi": e.hi}
        rows.append(row)
    import pandas as pd

    return pd.DataFrame(rows)


def main() -> None:
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/analysis/default.yaml"), OmegaConf.from_cli(sys.argv[1:])
    )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    runs_dir = ROOT / "runs"
    out = runs_dir / "_analysis"
    out.mkdir(exist_ok=True)

    runs = sm.experiment_runs(runs_dir, cfg["commit"])
    print(
        f"{len(runs)} finished experiment runs at {cfg['commit']}:",
        dict(runs.groupby("cond").size()),
    )

    tau = recompute_tau(runs, cfg)
    if not np.isclose(tau, cfg["tau"], rtol=0, atol=1e-7):
        raise SystemExit(f"tau recomputed {tau:.8f} != recorded {cfg['tau']:.8f}; stop (D39)")
    print(f"tau = {tau:.6f} m/s (matches the recorded value)")

    ep = sm.load_episodes(list(runs["path"]), cfg["tau"])
    ps = sm.per_setting(ep)
    ps.to_parquet(out / "per_setting.parquet", index=False)
    pr = sm.per_run(ps, runs, cfg)
    pr.to_csv(out / "per_run.csv", index=False)

    rng = np.random.default_rng(cfg["seed"])
    ct = condition_table(pr, cfg, rng)
    ct.to_csv(out / "conditions.csv", index=False)

    rh = np.random.default_rng(cfg["seed"] + 1)
    r2 = H.h2(pr, cfg, rh)
    hyp = {
        "H1": H.h1(pr, cfg, rh),
        "H2": r2,
        "H3": H.h3(pr, cfg, rh, r2["nominal_cost_holds"]),
        "H4": H.h4(pr, cfg, rh),
        "H5": H.h5(ps, pr, cfg, rh),
        "H6": {"verdict": "pending", "reason": "probing analysis (Phase 7)"},
    }
    (out / "hypotheses.json").write_text(json.dumps(hyp, indent=2, default=float))
    sha, dirty = git_commit()
    (out / "meta.json").write_text(
        json.dumps(
            {
                "analysis_commit": sha,
                "dirty": dirty,
                "config": cfg,
                "tau_recomputed": tau,
                "runs": runs["run_id"].tolist(),
            },
            indent=2,
        )
    )
    for k, v in hyp.items():
        print(f"{k}: {v['verdict']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
