"""Sim-to-sim evaluation in native MuJoCo for finished runs (plan step 23; exploratory).

    uv run python scripts/eval_native.py                       # configs/native_eval/default.yaml
    uv run python scripts/eval_native.py runs='runs/A_*_4a4a467' workers=4 episodes=16

Writes runs/<id>/eval_native.parquet (one row per episode and cell); cells already in an
existing file are skipped (new cells are appended). Native episode i is MJX grid episode i,
so rows pair with eval_mjx.parquet.
"""

from __future__ import annotations

import glob
import multiprocessing as mp
import os
import pathlib
import sys
import time

from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELL_KEYS = (
    "axis",
    "action_delay",
    "obs_delay",
    "t_m",
    "jitter",
    "friction",
    "payload_kg",
    "motor",
)


def _expand(pattern: str) -> list[pathlib.Path]:
    if "{" in pattern:
        pre, rest = pattern.split("{", 1)
        alts, post = rest.split("}", 1)
        return sorted({p for a in alts.split(",") for p in _expand(pre + a + post)})
    return sorted(pathlib.Path(p) for p in glob.glob(str(ROOT / pattern)))


def cells(cfg) -> list:
    from latrl.eval.native_eval import Cell

    out = [Cell("action", action_delay=int(d)) for d in cfg["action_delays"]]
    p = cfg["perturb"]
    for d in p["action_delays"]:
        out += [Cell("friction", action_delay=int(d), friction=float(f)) for f in p["friction"]]
        out += [Cell("payload", action_delay=int(d), payload_kg=float(k)) for k in p["payload_kg"]]
        out += [Cell("motor", action_delay=int(d), motor=float(s)) for s in p["motor"]]
    return out


def evaluate(args) -> str:
    run_dir, cfg = args
    os.environ.setdefault(
        "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    import pandas as pd

    from latrl.eval.native_eval import NativeEvaluator

    out = run_dir / "eval_native.parquet"
    want = cells(cfg)
    have = pd.read_parquet(out) if out.exists() else None
    if have is not None:
        done = {tuple(r) for r in have[list(CELL_KEYS)].itertuples(index=False)}
        want = [c for c in want if tuple(getattr(c, k) for k in CELL_KEYS) not in done]
        if not want:
            return f"skip {run_dir.name}"
    t = time.time()
    ev = NativeEvaluator(run_dir, cfg)
    df = pd.concat([ev.run_cell(c) for c in want], ignore_index=True)
    if have is not None:
        df = pd.concat([have, df], ignore_index=True)
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return f"done {run_dir.name}: {len(df)} episodes in {time.time() - t:.0f}s"


def main() -> None:
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/native_eval/default.yaml"), OmegaConf.from_cli(sys.argv[1:])
    )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    runs = [r for r in _expand(cfg["runs"]) if (r / "DONE").exists()]
    print(
        f"{len(runs)} runs, {len(cells(cfg))} cells x {cfg['episodes']} episodes each", flush=True
    )
    with mp.get_context("spawn").Pool(int(cfg["workers"])) as pool:
        for msg in pool.imap_unordered(evaluate, [(r, cfg) for r in runs]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
