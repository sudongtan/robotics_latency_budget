"""F2 pre-registered analysis (prereg post-hoc note 2026-11-17 "F2"): H2, H3, H4 (+ H6 if probed).

    uv run python scripts/analyze_f2.py              # n_boot etc. from configs/analysis
    uv run python scripts/analyze_f2.py n_boot=500   # quick look (not for reporting)

Reads only finished runs at the F2 commit (R5). Uses the Phase 5 metrics and the §6 decision
functions of latrl.analysis.hypotheses unchanged, with C_r40, D_r40, E_r40 in the roles of C, D,
E. tau stays 0.1552 (recorded 2026-11-06); tau recomputed from F2's A is reported as a check only.
Secondary (descriptive): walk rate per condition and the metrics over walking seeds only.
H6 is read from runs/_analysis_f2/h6.json, written by

    uv run python scripts/probe.py \
        'runs=runs/{A_mlp,C_r40_mlp,D_r40_hist4,E_r40_gru,F_r40_mlppriv}_s*_bc378de' \
        commit=bc378de 'h6.probed=[D_r40,E_r40]' \
        'representation={E_r40: hidden, default: features}' out=runs/_analysis_f2

Writes runs/_analysis_f2/{per_setting.parquet, per_run.csv, conditions.csv,
conditions_walking.csv, curves.csv, hypotheses.json, meta.json}. Verdicts are labelled
"F2 (0-40 ms training)" and do not replace the first version's.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze import condition_table, recompute_tau  # noqa: E402

from latrl.analysis import hypotheses as H  # noqa: E402
from latrl.analysis import summarize as sm  # noqa: E402
from latrl.utils.provenance import git_commit  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
F2_COMMIT = "bc378de"
CONDS = ["A", "C_r40", "D_r40", "E_r40", "F_r40"]
ROLES = {"C_r40": "C", "D_r40": "D", "E_r40": "E"}  # F2 condition -> §6 role
WALK = 25.0  # final eval return; walk/stand split (F43)
LABEL = "F2 (0-40 ms training)"


def main() -> None:
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/analysis/default.yaml"),
        {"commit": F2_COMMIT},
        OmegaConf.from_cli(sys.argv[1:]),
    )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    out = ROOT / "runs" / "_analysis_f2"
    out.mkdir(exist_ok=True)

    runs = sm.experiment_runs(ROOT / "runs", cfg["commit"])
    runs = runs[runs["cond"].isin(CONDS)].reset_index(drop=True)
    print(f"{len(runs)} finished runs at {cfg['commit']}:", dict(runs.groupby("cond").size()))

    tau_f2 = recompute_tau(runs, cfg)
    print(f"tau used {cfg['tau']:.6f}; recomputed from F2's A {tau_f2:.6f} (check only)")

    ep = sm.load_episodes(list(runs["path"]), cfg["tau"])
    ps = sm.per_setting(ep)
    ps.to_parquet(out / "per_setting.parquet", index=False)
    pr = sm.per_run(ps, runs, cfg)
    pr["walks"] = pr["final_return"] > WALK
    pr.to_csv(out / "per_run.csv", index=False)

    rng = np.random.default_rng(cfg["seed"])
    ct = condition_table(pr, cfg, rng)
    ct = ct.merge(pr.groupby("cond")["walks"].sum().astype(int).rename("walks"), on="cond")
    ct.to_csv(out / "conditions.csv", index=False)
    # Secondary: walking seeds only (selection on outcome; descriptive, never a verdict).
    cw = condition_table(pr[pr["walks"]], cfg, np.random.default_rng(cfg["seed"]))
    cw.to_csv(out / "conditions_walking.csv", index=False)

    act = ps[ps["axis"] == "action"].merge(runs[["run_id", "cond"]], on="run_id")
    cv = act.groupby(["cond", "x_ms"])["success"].apply(lambda x: float(sm.stats.iqm(x.to_numpy())))
    cv = cv.unstack("x_ms")
    cv.to_csv(out / "curves.csv")

    # §6 rules unchanged, F2 conditions in the C/D/E roles.
    role = pr.assign(cond=pr["cond"].replace(ROLES))
    rh = np.random.default_rng(cfg["seed"] + 1)
    h2 = H.h2(role, cfg, rh)
    h6f = out / "h6.json"
    hyp = {
        "label": LABEL,
        "roles": ROLES,
        "H2": h2,
        "H3": H.h3(role, cfg, rh, h2["nominal_cost_holds"]),
        "H4": H.h4(role, cfg, rh),
        "H6": (
            json.loads(h6f.read_text())
            if h6f.exists()
            else {"verdict": "pending", "reason": "run scripts/probe.py (see docstring)"}
        ),
        "not_retested": {"H1": "A's training unchanged", "H5": "no cross-channel runs in F2"},
    }
    (out / "hypotheses.json").write_text(json.dumps(hyp, indent=2, default=float))
    sha, dirty = git_commit()
    (out / "meta.json").write_text(
        json.dumps(
            {
                "analysis_commit": sha,
                "dirty": dirty,
                "config": cfg,
                "tau_recomputed_f2_A": tau_f2,
                "runs": runs["run_id"].tolist(),
            },
            indent=2,
        )
    )

    pd.set_option("display.width", 200)
    show = ["cond", "n_seeds", "walks", "L50_censored"] + [
        c
        for c in ["L50", "L50_lo", "L50_hi", "margin", "nominal_success", "nominal_lin_err"]
        if c in ct
    ]
    print(ct[[c for c in show if c in ct]].round(3).to_string(index=False))
    print("\nwalking seeds only (descriptive):")
    print(cw[[c for c in show if c in cw]].round(3).to_string(index=False))
    xs = [0, 20, 40, 60, 80, 100, 120]
    print("\nIQM success at test action delay (ms):")
    print(cv[[x for x in cv.columns if int(x) in xs]].round(2).to_string())
    for k in ("H2", "H3", "H4", "H6"):
        print(f"{k} [{LABEL}]: {hyp[k]['verdict']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
