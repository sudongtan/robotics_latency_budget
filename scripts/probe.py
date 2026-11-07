"""H6: probe the true action latency from policy features (plan step 30, prereg §6).

    uv run python scripts/probe.py                    # configs/probe/default.yaml
    uv run python scripts/probe.py n_boot=500 workers=4

Per run and trace set: ridge probe (H6 test, nested episode-level CV) and a 1x64 MLP probe
(secondary) on the policy's representation; for the probed conditions (D, E) also on a
randomly initialized network of the same architecture fed the same observations.
Writes runs/_analysis/probe_per_run.csv and runs/_analysis/h6.json.
"""

from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "_analysis"


def _expand(pattern: str) -> list[pathlib.Path]:
    if "{" in pattern:
        pre, rest = pattern.split("{", 1)
        alts, post = rest.split("}", 1)
        return sorted({p for a in alts.split(",") for p in _expand(pre + a + post)})
    return sorted(pathlib.Path(p) for p in glob.glob(str(ROOT / pattern)))


def probe_run(args) -> list[dict]:
    run_dir, cfg = args
    os.environ.setdefault(
        "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    import jax
    import numpy as np

    from latrl.analysis import probing as P
    from latrl.analysis.summarize import parse_run_id
    from latrl.eval.load import load_run

    info = parse_run_id(run_dir.name)
    cond, seed = info["cond"], int(info["seed"])
    reps = cfg["representation"]
    main_rep = reps.get(cond, reps["default"])
    extra = ["features"] if main_rep != "features" else []
    probed = cond in cfg["h6"]["probed"]
    tr = ts = None
    if probed:
        _, _, tr, ts = load_run(run_dir)
        rnd = tr.actor.init(jax.random.fold_in(jax.random.PRNGKey(int(cfg["seed"])), seed))
    rows_out = []
    for tset in cfg["sets"]:
        z = np.load(run_dir / "traces" / f"{tset}.npz")
        target, alive = z[cfg["target"]], z["alive"]
        sources = [("policy", r, z[r]) for r in [main_rep, *extra]]
        if probed:
            sources.append(("random", main_rep, P.recompute(tr, rnd, ts.norm, z["obs"], main_rep)))
        for source, rep, arr in sources:
            rows = P.design(arr, target, alive, int(cfg["start_step"]), int(cfg["stride"]))
            # same fold assignment for every source of this run and set
            rng = np.random.default_rng([int(cfg["seed"]), seed])
            t0 = time.time()
            r2_ridge, _, alphas = P.ridge_cv(
                rows, int(cfg["folds"]), int(cfg["inner_folds"]), cfg["alphas"], rng
            )
            rng = np.random.default_rng([int(cfg["seed"]), seed])
            r2_mlp = P.mlp_cv(rows, int(cfg["folds"]), cfg["mlp_probe"], rng)
            y_ep = target[np.unique(rows.groups), 0]
            rows_out.append(
                dict(
                    run_id=run_dir.name,
                    condition=cond,
                    seed=seed,
                    set=tset,
                    source=source,
                    rep=rep,
                    primary=rep == main_rep,
                    r2_ridge=r2_ridge,
                    r2_mlp=r2_mlp,
                    alpha_median=float(np.median(alphas)),
                    n_rows=len(rows.y),
                    n_episodes=len(y_ep),
                    alive_frac=float(alive[:, int(cfg["start_step"]) :].mean()),
                    target_sd=float(rows.y.std()),
                    seconds=time.time() - t0,
                )
            )
    return rows_out


def h6(df, cfg) -> dict:
    import numpy as np

    from latrl.analysis import stats as S

    rng = np.random.default_rng(int(cfg["seed"]) + 6)
    nb, ci, m = int(cfg["n_boot"]), float(cfg["ci"]), float(cfg["h6"]["margin"])
    d = df[(df.set == "lat") & df.primary]

    def r2s(cond, source="policy"):
        return d[(d.condition == cond) & (d.source == source)].sort_values("seed").r2_ridge.values

    out, all_sup, any_excl = {"conditions": {}}, True, False
    for x in cfg["h6"]["probed"]:
        ex = S.iqm_ci(r2s(x), nb, rng, ci)
        res = {"r2": ex.as_dict(), "controls": {}}
        sup_x = True
        for c in cfg["h6"]["controls"]:
            ctrl = r2s(c) if c != "random" else r2s(x, "random")
            ec = S.iqm_ci(ctrl, nb, rng, ci)
            diff = S.iqm_diff_ci(r2s(x), ctrl, nb, rng, ci)
            sup = diff.point >= m and ex.lo > ec.hi
            excl = diff.hi < m  # CI excludes the supporting region
            res["controls"][c] = {
                "r2": ec.as_dict(),
                "diff": diff.as_dict(),
                "supported": bool(sup),
                "excluded": bool(excl),
            }
            sup_x &= sup
            any_excl |= excl
        res["supported"] = bool(sup_x)
        out["conditions"][x] = res
        all_sup &= sup_x
    out["verdict"] = S.verdict(all_sup, any_excl)
    out["rule"] = (
        f"ridge R2 (lat set, {cfg['folds']}-fold episode-level CV) of each of "
        f"{cfg['h6']['probed']} exceeds each control {cfg['h6']['controls']} by >= {m} (IQM over "
        f"seeds), and its CI lower bound is above the control's CI upper bound"
    )
    return out


def main() -> None:
    import pandas as pd

    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/probe/default.yaml"), OmegaConf.from_cli(sys.argv[1:])
    )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    runs = [r for r in _expand(cfg["runs"]) if (r / "DONE").exists() and (r / "traces").exists()]
    print(f"{len(runs)} runs with traces", flush=True)
    rows = []
    with mp.get_context("spawn").Pool(int(cfg["workers"])) as pool:
        for res in pool.imap_unordered(probe_run, [(r, cfg) for r in runs]):
            print(f"done {res[0]['run_id']}", flush=True)
            rows += res
    df = pd.DataFrame(rows).sort_values(["condition", "seed", "set", "source", "rep"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "probe_per_run.csv", index=False)
    res = h6(df, cfg)
    (OUT / "h6.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v["r2"] for k, v in res["conditions"].items()}, indent=1))
    print("H6:", res["verdict"])


if __name__ == "__main__":
    main()
