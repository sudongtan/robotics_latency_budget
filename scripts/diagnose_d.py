"""Stage 1 diagnosis of D_r40's learning failure (log D52): what the standing policies do.

    uv run python scripts/diagnose_d.py            # replays + first-layer weights
    uv run python scripts/diagnose_d.py part=weights

Replay: each run's checkpoints in native MuJoCo on the first `episodes` MJX grid episodes at the
configured action delays. Per episode, while upright after metric_start_step: commanded speed,
achieved speed along the command, linear-velocity tracking error, foot touchdowns per second
(rising edges of the floor-contact sensors), and whether it fell.
Weights (history policies): share of the first-layer weight norm per history frame (frame 0 =
newest), on the proprio + last-action inputs (the command is the same in all frames).
Writes runs/_diagnose_d/{replay.csv, weights.csv}. Exploratory; no verdicts.
"""

from __future__ import annotations

import multiprocessing as mp
import pathlib
import sys

import jax.numpy as jp
import numpy as np
import pandas as pd
from mujoco_playground._src.locomotion.go1 import go1_constants as consts
from omegaconf import OmegaConf

from latrl.envs.specs import LAST_ACT, OBS_SIZE, PROPRIO
from latrl.eval.load import checkpoints, load_run
from latrl.eval.native_eval import MS_PER_SUBSTEP, Cell, NativeEvaluator

ROOT = pathlib.Path(__file__).resolve().parents[1]


def replay(job: tuple) -> pd.DataFrame:
    run, ck, delays, ncfg, steps = job
    ev = NativeEvaluator(run, ncfg, None if ck == "final" else f"{ck}.msgpack")
    fall = ncfg["fall"]
    t0 = int(ncfg["metric_start_step"])
    out = []
    for sub in delays:
        env = ev._env(Cell("action", action_delay=sub))
        m0 = env.models[0]
        adr = [m0.sensor_adr[m0.sensor(f"{g}_floor_found").id] for g in consts.FEET_GEOMS]
        n = ev.e
        hist = ev._init_hist(jp.asarray(env.obs, jp.float32))
        h = ev.tr.actor.initial_carry(n)
        first = jp.ones(n, bool)
        alive = np.ones(n, bool)
        prev = np.ones((n, 4), bool)
        acc = {k: np.zeros(n) for k in ("m", "cmd", "along", "err", "td")}
        for t in range(steps):
            a, h = ev._policy(
                hist,
                h,
                first,
                jp.asarray(env.action_delay, jp.int32),
                jp.asarray(env.obs_delay, jp.int32),
            )
            env.step(np.asarray(a, np.float64))
            alive &= ~((env.height() < fall["min_height"]) | (env.upz() < fall["min_upz"]))
            c = np.stack([d.sensordata[adr] > 0 for d in env.datas])
            w = (alive & (t >= t0)).astype(float)
            cmd = ev.exo.command[:, t + 1, :2]
            v = env.local_linvel()[:, :2]
            cn = np.linalg.norm(cmd, axis=-1)
            acc["m"] += w
            acc["cmd"] += w * cn
            acc["along"] += w * np.sum(v * cmd, -1) / np.maximum(cn, 1e-6)
            acc["err"] += w * np.linalg.norm(cmd - v, axis=-1)
            acc["td"] += w * np.sum(c & ~prev, -1)
            prev = c
            hist = ev._push(hist, jp.asarray(env.obs, jp.float32))
            first = jp.zeros(n, bool)
        m = np.maximum(acc["m"], 1)
        dt = float(ev.env.dt)
        out.append(
            pd.DataFrame(
                {
                    "run_id": run.name,
                    "checkpoint": ck,
                    "delay_ms": sub * MS_PER_SUBSTEP,
                    "episode": np.arange(n),
                    "fell": ~alive,
                    "cmd_speed": acc["cmd"] / m,
                    "speed_along_cmd": acc["along"] / m,
                    "lin_err": acc["err"] / m,
                    "touchdowns_per_s_per_foot": acc["td"] / m / dt / 4,
                }
            )
        )
    print(run.name, ck, flush=True)
    return pd.concat(out)


def weights(run: pathlib.Path) -> dict:
    _, _, tr, ts = load_run(run)
    k = np.asarray(ts.params["actor"]["params"]["Dense_0"]["kernel"])
    hl = tr.inp.history_len
    rows = np.r_[np.arange(OBS_SIZE)[PROPRIO], np.arange(OBS_SIZE)[LAST_ACT]]
    sq = np.array([np.sum(k[f * OBS_SIZE + rows] ** 2) for f in range(hl)])
    sp = np.array([np.sum(k[f * OBS_SIZE + np.arange(OBS_SIZE)[PROPRIO]] ** 2) for f in range(hl)])
    sa = np.array([np.sum(k[f * OBS_SIZE + np.arange(OBS_SIZE)[LAST_ACT]] ** 2) for f in range(hl)])
    d = {"run_id": run.name}
    for f in range(hl):
        d[f"frame{f}"] = sq[f] / sq.sum()
        d[f"proprio{f}"] = sp[f] / sp.sum()
        d[f"act{f}"] = sa[f] / sa.sum()
    return d


def main() -> None:
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/diagnose_d/default.yaml"), OmegaConf.from_cli(sys.argv[1:])
    )
    out = ROOT / cfg.out
    out.mkdir(parents=True, exist_ok=True)
    ncfg = OmegaConf.to_container(OmegaConf.load(ROOT / cfg.native), resolve=True)
    ncfg["episodes"] = int(cfg.episodes)
    runs = [
        ROOT / "runs" / f"{stem}_s{s}_{cfg.commit}" for stem, ss in cfg.runs.items() for s in ss
    ]
    part = cfg.get("part", "all")
    if part in ("all", "weights"):
        dr = sorted((ROOT / "runs").glob(f"D_r40_hist4_s*_{cfg.commit}"))
        w = pd.DataFrame([weights(r) for r in dr])
        w.to_csv(out / "weights.csv", index=False)
        print(w.round(3).to_string(index=False))
    if part in ("all", "replay"):
        for r in runs:
            assert checkpoints(r), r
        jobs = [
            (r, ck, list(cfg.action_delays), ncfg, int(cfg.steps))
            for r in runs
            for ck in cfg.checkpoints
        ]
        with mp.get_context("spawn").Pool(int(cfg.workers)) as pool:
            df = pd.concat(pool.map(replay, jobs, chunksize=1))
        df.to_csv(out / "replay.csv", index=False)
        print("wrote", out / "replay.csv")


if __name__ == "__main__":
    main()
