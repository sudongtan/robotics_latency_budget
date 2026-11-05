"""Restartable runs (D31): skip finished, move interrupted aside, ledger, finished list."""

import json
import pathlib
import subprocess
import time

from latrl.utils import runs

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_prepare_new_skip_partial(tmp_path):
    d = tmp_path / "A_mlp_s0_abc"
    assert runs.prepare(d) == "new" and not d.exists()
    d.mkdir()
    (d / "metrics.jsonl").write_text("{}\n")  # interrupted: no DONE
    assert runs.prepare(d) == "new"
    assert not d.exists()
    moved = list((tmp_path / "_partial").iterdir())
    assert len(moved) == 1 and moved[0].name.startswith("A_mlp_s0_abc_")
    assert (moved[0] / "metrics.jsonl").exists()
    d.mkdir()
    runs.finish(d, {"wall_s": 1.0})
    assert runs.prepare(d) == "skip"
    assert runs.finished(tmp_path) == [d]  # _partial is not a run


def test_ledger_appends(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        runs.finish(tmp_path / name, {"seed": 0})
    rows = [json.loads(x) for x in (tmp_path / "_ledger.jsonl").read_text().splitlines()]
    assert [r["run_id"] for r in rows] == ["a", "b"]
    assert not list(tmp_path.glob("*/DONE.tmp"))


def _train(runs_dir):
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train.py",
        "experiment=A",
        "ppo.num_envs=8",
        "ppo.num_minibatches=2",
        "train.total_steps=160",
        "train.eval_envs=2",
        "train.num_evals=1",
        "ppo.episode_length=20",
        "mjx_eval.enabled=false",
        f"train.runs_dir={runs_dir}",
    ]
    t = time.time()
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return out, time.time() - t


def test_train_skips_finished_and_restarts_interrupted(tmp_path):
    out, _ = _train(tmp_path)
    (run,) = runs.finished(tmp_path)
    done_before = (run / "DONE").read_text()
    out, dt = _train(tmp_path)
    assert "already finished, skipping" in out
    assert (run / "DONE").read_text() == done_before  # untouched
    (run / "DONE").unlink()  # simulate a crash before the end
    out, _ = _train(tmp_path)
    assert "moved interrupted" in out
    assert runs.finished(tmp_path) == [run]
    assert len(list((tmp_path / "_partial").iterdir())) == 1
    ledger = (tmp_path / "_ledger.jsonl").read_text().splitlines()
    assert len(ledger) == 2  # first run + rerun; the skip adds nothing
