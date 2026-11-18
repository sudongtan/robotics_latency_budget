"""F2 videos: trained policies walking (or not) under action delay, in native MuJoCo on the Mac.

    uv run python scripts/make_videos.py                 # all videos in configs/video/default.yaml
    uv run python scripts/make_videos.py only=grid       # one video

No cherry-picking: the seed per condition is fixed by rule (lower-median L50, see the config) and
every panel plays the same episode, drawn once at random. Each panel shows the condition's success
rate at that delay (IQM over all 10 seeds, MJX grid, runs/_analysis_f2/curves.csv), so a viewer can
tell whether the clip is typical, and marks the moment this episode falls, if it does.
Writes runs/_videos_f2/<video>.mp4 (+ meta.json) and docs/figures/f2_<video>.gif where configured.
"""

from __future__ import annotations

import json
import pathlib
import sys

import jax.numpy as jp
import mediapy
import mujoco
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from latrl.eval.native_eval import MS_PER_SUBSTEP, Cell, NativeEvaluator
from latrl.utils.provenance import git_commit

ROOT = pathlib.Path(__file__).resolve().parents[1]


def median_seed(pr: pd.DataFrame, cond: str) -> int:
    g = pr[pr["cond"] == cond].sort_values(["L50", "seed"]).reset_index(drop=True)
    return int(g["seed"].iloc[(len(g) - 1) // 2])


def rollout(ev: NativeEvaluator, cell: Cell, ep: int, steps: int, cfg) -> tuple[list, int]:
    """Run all loaded episodes (the batch the evaluator holds), render episode `ep`."""
    env = ev._env(cell)
    n = ev.e
    fall = ev.ecfg["fall"]
    hist = ev._init_hist(jp.asarray(env.obs, jp.float32))
    h = ev.tr.actor.initial_carry(n)
    first = jp.ones(n, bool)
    r = mujoco.Renderer(env.models[ep], cfg.height, cfg.width)
    frames, fall_step = [], -1
    for t in range(steps):
        a, h = ev._policy(
            hist,
            h,
            first,
            jp.asarray(env.action_delay, jp.int32),
            jp.asarray(env.obs_delay, jp.int32),
        )
        env.step(np.asarray(a, np.float64))
        d = env.datas[ep]
        if fall_step < 0 and (
            env.height()[ep] < fall["min_height"] or env.upz()[ep] < fall["min_upz"]
        ):
            fall_step = t
        r.update_scene(d, camera=cfg.camera)
        frames.append(r.render())
        hist = ev._push(hist, jp.asarray(env.obs, jp.float32))
        first = jp.zeros(n, bool)
    r.close()
    return frames, fall_step


def label(frames: list, lines: list[str], fall_step: int, fps: int) -> list:
    font = ImageFont.load_default(size=13)
    out = []
    for t, f in enumerate(frames):
        img = Image.fromarray(f)
        dr = ImageDraw.Draw(img)
        text = list(lines)
        if 0 <= fall_step <= t:
            text.append(f"fell at {fall_step / fps:.1f} s")
        y = 4
        for s in text:
            box = dr.textbbox((6, y), s, font=font)
            dr.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0))
            dr.text((6, y), s, fill=(255, 255, 255), font=font)
            y = box[3] + 5
        out.append(np.asarray(img))
    return out


def main() -> None:
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/video/default.yaml"), OmegaConf.from_cli(sys.argv[1:])
    )
    ncfg = OmegaConf.to_container(OmegaConf.load(ROOT / cfg.native), resolve=True)
    an = ROOT / cfg.analysis
    pr = pd.read_csv(an / "per_run.csv")
    curves = pd.read_csv(an / "curves.csv", index_col="cond")
    curves.columns = [float(c) for c in curves.columns]
    ep = int(np.random.default_rng(cfg.episode_seed).integers(cfg.episode_pool))
    ncfg["episodes"] = ep + 1  # the evaluator loads episodes 0..ep (paired with the MJX grid)
    steps = int(cfg.duration_s * cfg.fps)
    out = ROOT / cfg.out
    out.mkdir(parents=True, exist_ok=True)
    seeds = {c: median_seed(pr, c) for c in cfg.conds}
    print("episode", ep, "seeds", seeds)

    only = cfg.get("only")
    evs: dict[str, NativeEvaluator] = {}
    meta = {"episode": ep, "seeds": seeds, "commit": cfg.commit, "videos": {}}
    for name, v in cfg.videos.items():
        if only and name != only:
            continue
        rows = []
        for c in v.rows:
            run = ROOT / "runs" / f"{cfg.conds[c].stem}_s{seeds[c]}_{cfg.commit}"
            if c not in evs:
                evs[c] = NativeEvaluator(run, ncfg)
            panels = []
            for ms in v.delays_ms:
                sub = round(ms / MS_PER_SUBSTEP)
                frames, fs = rollout(evs[c], Cell("action", action_delay=sub), ep, steps, cfg)
                sr = 100 * curves.loc[c, float(ms)]
                lines = [cfg.conds[c].label, f"delay {ms} ms", f"success {sr:.0f} % (all seeds)"]
                panels.append(label(frames, lines, fs, cfg.fps))
                meta["videos"].setdefault(name, []).append(
                    {"cond": c, "run": run.name, "delay_ms": ms, "fell_at_step": fs}
                )
                print(f"{name}: {c} {ms} ms fell_at_step={fs}", flush=True)
            rows.append([np.concatenate(fs, axis=1) for fs in zip(*panels, strict=True)])
        video = [np.concatenate(fs, axis=0) for fs in zip(*rows, strict=True)]
        mediapy.write_video(out / f"{name}.mp4", video, fps=cfg.fps)
        if v.gif:
            small = [
                np.asarray(Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2)))
                for f in video[::3]
            ]
            gif = ROOT / cfg.gif_dir / f"f2_{name}.gif"
            mediapy.write_video(gif, small, fps=cfg.fps / 3, codec="gif")
            print("wrote", gif)
        print("wrote", out / f"{name}.mp4")
    sha, dirty = git_commit()
    meta |= {"video_commit": sha, "dirty": dirty, "config": OmegaConf.to_container(cfg)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
