"""Robustness at every saved checkpoint (exploratory: robustness per unit of training cost).

    uv run python scripts/eval_checkpoints.py            # configs/native_eval/default.yaml
                                                          #   (checkpoints: runs, delays, episodes)

Native MuJoCo, the first `checkpoints.episodes` episodes of the MJX grid, action-delay axis
only. Writes runs/<id>/eval_checkpoints.parquet (skipped if present).
"""

from __future__ import annotations

import multiprocessing as mp
import pathlib
import sys
import time

from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def evaluate(args) -> str:
    run_dir, cfg = args
    import json

    import pandas as pd

    from latrl.eval.load import checkpoints
    from latrl.eval.native_eval import Cell, NativeEvaluator

    out = run_dir / "eval_checkpoints.parquet"
    if out.exists():
        return f"skip {run_dir.name}"
    t = time.time()
    c = cfg["checkpoints"]
    ecfg = dict(cfg, episodes=int(c["episodes"]))
    steps = {}
    for line in (run_dir / "eval.jsonl").open():
        e = json.loads(line)
        steps[int(e["iter"])] = (int(e["env_steps"]), float(e["train_s"]))
    dfs = []
    for ck in checkpoints(run_dir):
        it = int(ck.stem[2:])
        ev = NativeEvaluator(run_dir, ecfg, checkpoint=ck.name)
        for d in c["action_delays"]:
            df = ev.run_cell(Cell("action", action_delay=int(d)))
            df["iter"], (df["env_steps"], df["train_s"]) = it, steps.get(it, (-1, float("nan")))
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return f"done {run_dir.name}: {len(df)} episodes in {time.time() - t:.0f}s"


def main() -> None:
    from eval_native import _expand

    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/native_eval/default.yaml"), OmegaConf.from_cli(sys.argv[1:])
    )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    runs = [r for r in _expand(cfg["checkpoints"]["runs"]) if (r / "DONE").exists()]
    print(f"{len(runs)} runs", flush=True)
    with mp.get_context("spawn").Pool(int(cfg["workers"])) as pool:
        for msg in pool.imap_unordered(evaluate, [(r, cfg) for r in runs]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
