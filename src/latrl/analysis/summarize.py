"""From `runs/<id>/eval_mjx.parquet` (one row per episode) to per-setting and per-run tables.

Only finished runs (`DONE`) at the session commit are read (R5). Episode success (prereg §4):
no fall in 20 s and mean linear-velocity tracking error (first 0.5 s excluded, `lin_err`) ≤ τ.

Test axes in the parquet (written by `latrl.eval.mjx_eval`) and their x coordinate in ms:
    action         action_delay_ms   0–120 every 4 ms (substeps of 4 ms), no jitter
    action_jitter  action_delay_ms   same grid, jitter ±12 ms, p = 0.1 per control step
    obs            obs_delay_ms      0–120 every 20 ms (control steps)
    tm             t_m_s × 1000      motor time constant 0–60 every 5 ms
"""

from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pandas as pd

from latrl.analysis import stats
from latrl.utils import runs as runs_util

RUN_RE = re.compile(
    r"^(?P<cond>.+)_(?P<policy>mlp|mlppriv|hist\d+|gru)_s(?P<seed>\d+)_(?P<commit>[0-9a-f]{7,})$"
)
NOT_EXPERIMENTS = ("bench_", "brax_", "gpusmoke_", "smoke_", "phase1_")
AXES = ("action", "action_jitter", "obs", "tm")


def parse_run_id(run_id: str) -> dict | None:
    m = RUN_RE.match(run_id)
    if m is None or run_id.startswith(NOT_EXPERIMENTS):
        return None
    return {"cond": m["cond"], "policy": m["policy"], "seed": int(m["seed"]), "commit": m["commit"]}


def experiment_runs(runs_dir: pathlib.Path, commit: str) -> pd.DataFrame:
    """Finished experiment runs at `commit`: run_id, cond, policy, seed, path."""
    rows = []
    for d in runs_util.finished(runs_dir):
        p = parse_run_id(d.name)
        if p and p["commit"] == commit:
            rows.append({"run_id": d.name, **p, "path": d})
    return pd.DataFrame(rows).sort_values(["cond", "seed"]).reset_index(drop=True)


def x_ms(df: pd.DataFrame) -> pd.Series:
    """Test-axis coordinate in ms for each row."""
    x = pd.Series(np.nan, index=df.index)
    act = df["axis"].isin(["action", "action_jitter"])
    x[act] = df.loc[act, "action_delay_ms"]
    obs = df["axis"] == "obs"
    x[obs] = df.loc[obs, "obs_delay_ms"]
    tm = df["axis"] == "tm"
    x[tm] = (df.loc[tm, "t_m_s"] * 1000.0).round(6)
    return x


def load_episodes(paths: list[pathlib.Path], tau: float) -> pd.DataFrame:
    cols = [
        "run_id",
        "axis",
        "action_delay_ms",
        "obs_delay_ms",
        "t_m_s",
        "episode",
        "fell",
        "lin_err",
        "yaw_err",
        "cot",
        "jerk",
        "duty_factor",
        "step_freq_hz",
        "apex_m",
        "stance_width_m",
        "return",
    ]
    df = pd.concat(
        [pd.read_parquet(p / "eval_mjx.parquet", columns=cols) for p in paths], ignore_index=True
    )
    df["x_ms"] = x_ms(df)
    df["success"] = (~df["fell"]) & (df["lin_err"] <= tau)
    return df


def per_setting(ep: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, axis, x): success / fall rates and means over non-fallen episodes."""
    g = ep.groupby(["run_id", "axis", "x_ms"], sort=True)
    out = g.agg(n=("success", "size"), success=("success", "mean"), fall=("fell", "mean"))
    nf = ep[~ep["fell"]].groupby(["run_id", "axis", "x_ms"])
    means = nf[
        [
            "lin_err",
            "yaw_err",
            "cot",
            "jerk",
            "duty_factor",
            "step_freq_hz",
            "apex_m",
            "stance_width_m",
        ]
    ].mean()
    return out.join(means).reset_index()


def curve(ps: pd.DataFrame, run_id: str, axis: str) -> tuple[np.ndarray, np.ndarray]:
    c = ps[(ps["run_id"] == run_id) & (ps["axis"] == axis)].sort_values("x_ms")
    return c["x_ms"].to_numpy(), c["success"].to_numpy()


def train_timing(run_dir: pathlib.Path) -> dict[str, float]:
    """Training wall-clock (prereg post-hoc clarification 4) and time to 90 % of final return.

    `train_s` (last metrics line) is the training loop's own clock: rollouts + updates +
    compilation, excluding the periodic evals and the evaluation grid. `wall_s` adds the
    periodic evals (10 per run). Time to 90 %: first periodic eval whose return reaches 90 %
    of the final eval's return (10 evals per run, so coarse: 20M-step resolution).
    """
    last = None
    with (run_dir / "metrics.jsonl").open() as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    evals = [json.loads(line) for line in (run_dir / "eval.jsonl").open() if line.strip()]
    final = evals[-1]["eval/episode_return"]
    hit = next(e for e in evals if e["eval/episode_return"] >= 0.9 * final)
    return {
        "train_s": float(last["train_s"]),
        "wall_s": float(last["wall_s"]),
        "final_return": float(final),
        "t90_train_s": float(hit["train_s"]),
        "t90_steps": float(hit["env_steps"]),
    }


def per_run(ps: pd.DataFrame, runs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-run robustness and nominal metrics."""
    rows = []
    for r in runs.itertuples():
        row = {"run_id": r.run_id, "cond": r.cond, "policy": r.policy, "seed": r.seed}
        for axis, tag in (
            ("action", ""),
            ("action_jitter", "_jit"),
            ("obs", "_obs"),
            ("tm", "_tm"),
        ):
            x, s = curve(ps, r.run_id, axis)
            l50 = stats.crossing(x, s, cfg["success_threshold"])
            l90 = stats.crossing(x, s, cfg["w_hi"])
            l10 = stats.crossing(x, s, cfg["w_lo"])
            row |= {
                f"L50{tag}": l50.value,
                f"L50{tag}_cens": l50.censored,
                f"L90{tag}": l90.value,
                f"L10{tag}": l10.value,
                f"L10{tag}_cens": l10.censored,
                f"W{tag}": l10.value - l90.value,
                f"margin{tag}": stats.delay_margin(x, s, cfg["success_threshold"]),
                f"mean_success{tag}": float(np.mean(s)),
            }
        nom = ps[(ps["run_id"] == r.run_id) & (ps["axis"] == "action") & (ps["x_ms"] == 0.0)]
        row |= {
            "nominal_success": float(nom["success"].iloc[0]),
            "nominal_fall": float(nom["fall"].iloc[0]),
            "nominal_lin_err": float(nom["lin_err"].iloc[0]),
            "nominal_yaw_err": float(nom["yaw_err"].iloc[0]),
            "nominal_cot": float(nom["cot"].iloc[0]),
            "nominal_jerk": float(nom["jerk"].iloc[0]),
        }
        params = json.loads((r.path / "params.json").read_text())
        row |= {"actor_params": params["actor"]} | train_timing(r.path)
        rows.append(row)
    return pd.DataFrame(rows)
