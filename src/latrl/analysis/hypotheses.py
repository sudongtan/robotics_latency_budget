"""Decision rules H1–H5 (prereg §6 and post-hoc clarifications 1–4). H6 (probing) is Phase 7.

Each test returns a dict with the verdict, the estimates it used, and the rule as text, so
`docs/results.md` can show exactly why a verdict came out the way it did.

"Not supported" = the rule fails with CIs that exclude the supporting region (prereg §6). For
conjunctions, a conjunct whose CI lies entirely outside its supporting region makes the
hypothesis "not supported"; any other failure is "inconclusive".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from latrl.analysis import stats

S = stats


def _col(pr: pd.DataFrame, cond: str, col: str) -> np.ndarray:
    v = pr[pr["cond"] == cond].sort_values("seed")[col].to_numpy(dtype=float)
    if v.size == 0:
        raise ValueError(f"no runs for condition {cond}")
    return v


def h1(pr: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> dict:
    dc, nb, lv = cfg["decision"], cfg["n_boot"], cfg["ci"]
    w = S.iqm_ci(_col(pr, "A", "W"), nb, rng, lv)
    l50 = S.iqm_ci(_col(pr, "A", "L50"), nb, rng, lv)
    wmax, (lo, hi) = dc["h1_w_max_ms"], dc["h1_l50_range_ms"]
    ok = w.point <= wmax and lo <= l50.point <= hi
    excl = w.lo > wmax or l50.hi < lo or l50.lo > hi
    return {
        "verdict": S.verdict(ok, excl),
        "rule": f"IQM(W_A) <= {wmax} ms and IQM(L50_A) in [{lo}, {hi}] ms",
        "W_A": w.as_dict(),
        "L50_A": l50.as_dict(),
        "censored_W_A": int(pr[pr["cond"] == "A"]["L10_cens"].sum()),
        "censored_L50_A": int(pr[pr["cond"] == "A"]["L50_cens"].sum()),
    }


def h2(pr: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> dict:
    nb, lv = cfg["n_boot"], cfg["ci"]
    dl = S.iqm_diff_ci(_col(pr, "C", "L50"), _col(pr, "A", "L50"), nb, rng, lv)
    de = S.iqm_diff_ci(
        _col(pr, "C", "nominal_lin_err"), _col(pr, "A", "nominal_lin_err"), nb, rng, lv
    )
    ok = dl.lo > 0 and de.lo > 0
    excl = dl.hi <= 0 or de.hi <= 0
    return {
        "verdict": S.verdict(ok, excl),
        "rule": "L50(C) - L50(A) > 0 and err0(C) - err0(A) > 0, both CIs excluding 0",
        "dL50_C_minus_A": dl.as_dict(),
        "dErr0_C_minus_A": de.as_dict(),
        "censored_L50_C": int(pr[pr["cond"] == "C"]["L50_cens"].sum()),
        "nominal_cost_holds": bool(de.lo > 0),
    }


def h3(pr: pd.DataFrame, cfg: dict, rng: np.random.Generator, nominal_cost_holds: bool) -> dict:
    nb, lv, rmin = cfg["n_boot"], cfg["ci"], cfg["decision"]["h3_r_min"]
    ea, ec, ed = (_col(pr, c, "nominal_lin_err") for c in ("A", "C", "D"))
    r = S.recovery_ci(ea, ec, ed, nb, rng, lv)
    out = {
        "rule": f"R = (err_C - err_D)/(err_C - err_A) >= {rmin}, CI lower bound > 0; "
        "inconclusive if the CI of err_C - err_A includes 0",
        "R": r.as_dict(),
        "err0_IQM": {c: float(S.iqm(e)) for c, e in (("A", ea), ("C", ec), ("D", ed))},
    }
    if not nominal_cost_holds:
        return {"verdict": S.INCONCLUSIVE, "reason": "H2's nominal cost does not hold"} | out
    return {"verdict": S.verdict(r.point >= rmin and r.lo > 0, r.hi < rmin)} | out


def h4(pr: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> dict:
    dc, nb = cfg["decision"], cfg["n_boot"]
    cens_d = int(_col(pr, "D", "L50_cens").sum())
    cens_e = int(_col(pr, "E", "L50_cens").sum())
    n_d, n_e = len(_col(pr, "D", "L50")), len(_col(pr, "E", "L50"))
    fallback = cens_d >= n_d / 2 or cens_e >= n_e / 2
    if fallback:
        m = dc["h4_equiv_success_pp"]
        diff = S.iqm_diff_ci(
            100 * _col(pr, "E", "mean_success"),
            100 * _col(pr, "D", "mean_success"),
            nb,
            rng,
            dc["h4_ci"],
        )
        measure = "mean success over action grid 0-120 ms (pp)"
    else:
        m = dc["h4_equiv_ms"]
        diff = S.iqm_diff_ci(_col(pr, "E", "L50"), _col(pr, "D", "L50"), nb, rng, dc["h4_ci"])
        measure = "L50 (ms)"
    ratio = S.iqm_ratio_ci(_col(pr, "E", "train_s"), _col(pr, "D", "train_s"), nb, rng, cfg["ci"])
    equiv = -m <= diff.lo and diff.hi <= m
    equiv_excl = diff.lo > m or diff.hi < -m
    ok = equiv and ratio.lo > 1
    excl = equiv_excl or ratio.hi < 1
    return {
        "verdict": S.verdict(ok, excl),
        "rule": f"{int(100 * dc['h4_ci'])} % CI of E - D within +-{m} on {measure}; "
        "wall-clock ratio E/D > 1 with CI excluding 1",
        "measure": measure,
        "fallback_used": fallback,
        "censored_L50": {"D": cens_d, "E": cens_e},
        "diff_E_minus_D": diff.as_dict(),
        "equivalent": bool(equiv),
        "train_s_ratio_E_over_D": ratio.as_dict(),
        "train_s_IQM": {c: float(S.iqm(_col(pr, c, "train_s"))) for c in ("D", "E")},
    }


H5_POLICIES = {  # policy: (same-channel axis, cross-channel axis)
    "C": ("action", "obs"),
    "D": ("action", "obs"),
    "E": ("action", "obs"),
    "C_obs": ("obs", "action"),
    "D_obs": ("obs", "action"),
    "E_obs": ("obs", "action"),
}


def h5(ps: pd.DataFrame, pr: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> dict:
    """Same policy on both channels at 20, 40, ..., 120 ms; paired over seeds."""
    dc, nb, lv = cfg["decision"], cfg["n_boot"], cfg["ci"]
    tol = dc["h5_max_pp"]
    floor, ceil = dc["h5_floor_ceiling"]
    xs = [20.0 * k for k in range(1, 7)]
    per_policy, all_ok, any_excl, too_many_fc = {}, True, False, False
    for pol, (same, cross) in H5_POLICIES.items():
        ids = pr[pr["cond"] == pol].sort_values("seed")["run_id"].tolist()
        sub = ps[ps["run_id"].isin(ids) & ps["x_ms"].isin(xs)]
        tab = sub.pivot_table(index=["run_id", "x_ms"], columns="axis", values="success")
        pts, fc = [], 0
        for x in xs:
            a = 100 * np.array([tab.loc[(i, x), same] for i in ids])
            b = 100 * np.array([tab.loc[(i, x), cross] for i in ids])
            idx = rng.integers(0, len(ids), size=(nb, len(ids)))
            boot = S.iqm(b[idx], axis=1) - S.iqm(a[idx], axis=1)
            d = float(S.iqm(b) - S.iqm(a))
            lo, hi = S.percentile_ci(boot, lv)
            sa, sb = float(S.iqm(a)), float(S.iqm(b))
            is_fc = (sa >= 100 * ceil and sb >= 100 * ceil) or (
                sa <= 100 * floor and sb <= 100 * floor
            )
            fc += is_fc
            ok = abs(d) <= tol
            excl = lo > tol or hi < -tol
            all_ok &= ok
            any_excl |= excl
            pts.append(
                {
                    "x_ms": x,
                    "same": sa,
                    "cross": sb,
                    "diff_pp": d,
                    "lo": lo,
                    "hi": hi,
                    "within": ok,
                    "excluded": excl,
                    "floor_ceiling": bool(is_fc),
                }
            )
        too_many_fc |= fc > len(xs) / 2
        per_policy[pol] = {
            "same_axis": same,
            "cross_axis": cross,
            "n_seeds": len(ids),
            "floor_ceiling_points": fc,
            "points": pts,
        }
    if any_excl:
        v = S.NOT_SUPPORTED
    elif all_ok and not too_many_fc:
        v = S.SUPPORTED
    else:
        v = S.INCONCLUSIVE
    return {
        "verdict": v,
        "rule": f"|success(cross) - success(same)| <= {tol} pp at 20..120 ms for each of the "
        "six policies (IQM over seeds); <= half floor/ceiling points per policy",
        "policies": per_policy,
    }
