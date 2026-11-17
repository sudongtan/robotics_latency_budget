"""Follow-up F1 (exploratory): robustness of the F1 arms vs. A and B on the action-delay grid.

    uv run python scripts/analyze_followup.py              # n_boot from configs/analysis
    uv run python scripts/analyze_followup.py n_boot=500   # quick look

Uses the Phase 5 definitions unchanged (prereg §4: tau, success, L50, delay margin) and the
same bootstrap. Reads finished runs at the session-S commit (A, B20, B40, B60, C, E) and at
the F1 commit (C control, C_ft, C_r40, C_hcrit); the F1 control is labelled `C@f1`.
Writes runs/_analysis_f1/{per_setting.parquet, per_run.csv, conditions.csv, curves.csv}.
Exploratory: no pre-registered verdicts (prereg post-hoc note 2026-11-17).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from latrl.analysis import stats as S
from latrl.analysis import summarize as sm
from latrl.utils.provenance import git_commit

ROOT = pathlib.Path(__file__).resolve().parents[1]
F1_COMMIT = "650d2d7"
S_CONDS = ["A", "B20", "B40", "B60", "C", "E"]
F1_CONDS = {"C": "C@f1", "C_ft": "C_ft", "C_r40": "C_r40", "C_hcrit": "C_hcrit"}
COLS = ["L50", "margin", "mean_success", "nominal_success", "final_return"]
WALK = 25.0


def main() -> None:
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/analysis/default.yaml"), OmegaConf.from_cli(sys.argv[1:])
    )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    runs_dir = ROOT / "runs"
    out = runs_dir / "_analysis_f1"
    out.mkdir(exist_ok=True)

    s = sm.experiment_runs(runs_dir, cfg["commit"])
    s = s[s["cond"].isin(S_CONDS)]
    f = sm.experiment_runs(runs_dir, F1_COMMIT)
    f = f[f["cond"].isin(F1_CONDS)].assign(cond=lambda d: d["cond"].map(F1_CONDS))
    runs = pd.concat([s, f], ignore_index=True)
    print(dict(runs.groupby("cond").size()))

    ep = sm.load_episodes(list(runs["path"]), cfg["tau"])
    ps = sm.per_setting(ep)
    ps.to_parquet(out / "per_setting.parquet", index=False)
    pr = sm.per_run(ps, runs, cfg)
    pr["walks"] = pr["final_return"] > WALK
    pr.to_csv(out / "per_run.csv", index=False)

    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for cond, g in pr.groupby("cond", sort=False):
        row = {"cond": cond, "n": len(g), "walks": int(g["walks"].sum())}
        for c in COLS:
            v = g.sort_values("seed")[c].to_numpy(dtype=float)
            if np.isnan(v).any():
                row[f"{c}_defined"] = int((~np.isnan(v)).sum())
                continue
            e = S.iqm_ci(v, cfg["n_boot"], rng, cfg["ci"])
            row |= {c: e.point, f"{c}_lo": e.lo, f"{c}_hi": e.hi}
        rows.append(row)
    ct = pd.DataFrame(rows)
    ct.to_csv(out / "conditions.csv", index=False)

    # IQM over seeds of per-run success at each test delay (the headline-figure curve).
    ps = ps[ps["axis"] == "action"].merge(runs[["run_id", "cond"]], on="run_id")
    cv = (
        ps.groupby(["cond", "x_ms"])["success"]
        .apply(lambda x: float(S.iqm(x.to_numpy())))
        .unstack("x_ms")
    )
    cv.to_csv(out / "curves.csv")

    sha, dirty = git_commit()
    (out / "meta.json").write_text(
        json.dumps(
            {"analysis_commit": sha, "dirty": dirty, "config": cfg, "runs": list(runs["run_id"])},
            indent=2,
        )
    )
    pd.set_option("display.width", 200)
    show = ["cond", "n", "walks"] + [
        c
        for c in ["L50", "L50_lo", "L50_hi", "margin", "mean_success", "nominal_success"]
        if c in ct
    ]
    print(ct[show].round(3).to_string(index=False))
    xs = [0, 20, 40, 60, 80, 100, 120]
    print("\nIQM success at test action delay (ms):")
    print(cv[[x for x in cv.columns if int(x) in xs]].round(2).to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
