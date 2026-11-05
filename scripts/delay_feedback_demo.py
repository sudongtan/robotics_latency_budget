"""Visual check that action delay degrades closed-loop control (Phase 1 step 9, D12).

A scripted feedback controller holds the standing pose at 50 Hz using the joint angles it
*observes*, and the trunk gets a sideways push. The same controller runs at several
action delays; the videos are tiled side by side.

Why feedback: delaying an open-loop motion only shifts it in time. A feedback controller
reacts to a state that has already moved on; with enough delay it pushes at the wrong
moment, the oscillation grows, and the robot falls.

Usage:
    uv run python scripts/delay_feedback_demo.py [key=value ...]   # overrides config
Outputs runs/phase1_delay_demo_s{seed}_{commit}/: video.mp4, metrics.json, meta.json.
"""

from __future__ import annotations

import json
import pathlib

import jax
import jax.numpy as jp
import mediapy
import mujoco
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from latrl.envs.go1 import make_env
from latrl.envs.specs import PROPRIO
from latrl.utils.provenance import git_commit, write_run_meta

ROOT = pathlib.Path(__file__).resolve().parents[1]
JOINT_POS = slice(9, 21)  # within PROPRIO: joint angles relative to the default pose


def controller(obs: jax.Array, gain: float, action_scale: float) -> jax.Array:
    """q* − q_default = gain · (q_default − q_measured), expressed as a policy action."""
    dq = obs[PROPRIO][JOINT_POS]
    return jp.clip(-gain * dq / action_scale, -1.0, 1.0)


def run(env, cfg, d_substeps: int):
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    ctrl = jax.jit(lambda o: controller(o, cfg.gain, env._config.action_scale))
    state = env.with_latency(reset(jax.random.PRNGKey(cfg.seed)), action_delay=d_substeps)
    state = state.replace(data=state.data.replace(qvel=jp.zeros_like(state.data.qvel)))
    n = round(cfg.duration_s / env.dt)
    p0 = round(cfg.push.start_s / env.dt)
    p1 = p0 + round(cfg.push.duration_s / env.dt)
    torso = env._torso_body_id
    traj, tilt = [], []
    for t in range(n):
        force = jp.zeros((env.mjx_model.nbody, 6))
        if p0 <= t < p1:
            force = force.at[torso, 1].set(cfg.push.force_n)
        state = state.replace(data=state.data.replace(xfrc_applied=force))
        state = step(state, ctrl(state.obs["state"]))
        traj.append(state)
        up = np.asarray(env.get_upvector(state.data))
        tilt.append(float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))))
    return traj, np.array(tilt)


def label(frames, text):
    out = []
    for f in frames:
        img = Image.fromarray(f)
        draw = ImageDraw.Draw(img)
        draw.rectangle(draw.textbbox((8, 6), text), fill=(0, 0, 0))
        draw.text((8, 6), text, fill=(255, 255, 255))
        out.append(np.asarray(img))
    return out


def main() -> None:
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/demo/delay_feedback.yaml"), OmegaConf.from_cli()
    )
    env_cfg = OmegaConf.load(ROOT / cfg.env)
    env_cfg.noise_level = 0.0  # deterministic demo
    env_cfg.latency.action.hi = env_cfg.latency.action.max
    env = make_env(env_cfg)
    sub_ms = env.sim_dt * 1000

    sha, _ = git_commit()
    run_dir = ROOT / "runs" / f"phase1_delay_demo_s{cfg.seed}_{sha}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_meta(
        run_dir,
        {"demo": OmegaConf.to_container(cfg), "env": OmegaConf.to_container(env_cfg)},
        cfg.seed,
    )

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True  # draw the push arrow
    panels, metrics = [], {}
    for ms in cfg.action_delays_ms:
        d = round(ms / sub_ms)
        traj, tilt = run(env, cfg, d)
        fell = bool(tilt.max() > cfg.fall_tilt_deg)
        metrics[f"{ms}ms"] = {
            "delay_substeps": d,
            "max_tilt_deg": round(float(tilt.max()), 1),
            "final_tilt_deg": round(float(tilt[-1]), 1),
            "fell": fell,
        }
        print(ms, "ms", metrics[f"{ms}ms"])
        frames = env.render(
            traj,
            height=cfg.video.height,
            width=cfg.video.width,
            camera=cfg.video.camera,
            scene_option=opt,
        )
        panels.append(label(frames, f"action delay {ms} ms" + ("  - FELL" if fell else "")))
    video = [np.concatenate(fs, axis=1) for fs in zip(*panels, strict=True)]
    mediapy.write_video(run_dir / "video.mp4", video, fps=cfg.video.fps)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", run_dir)


if __name__ == "__main__":
    main()
