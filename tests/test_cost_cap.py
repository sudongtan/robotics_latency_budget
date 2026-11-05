"""Cost cap step (scripts/gpu/cost_cap.py): projection from the benchmark, skip finished runs."""

import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHA = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
).stdout.strip()


def _setup(tmp_path, rate):
    runs = tmp_path / "runs"
    bench = runs / f"bench_jax_A_mlp_s0_{SHA}"
    bench.mkdir(parents=True)
    (bench / "metrics.jsonl").write_text(json.dumps({"env_steps": 2e7, "train_s": 2e7 / rate}))
    (tmp_path / "impl").write_text("jax")
    sweep = tmp_path / "s.yaml"
    sweep.write_text(
        "jobs:\n"
        '  - {step: "uv run python scripts/gpu/cost_cap.py X @LOGDIR@/impl --max-hours 1"}\n'
        '  - {experiment: A, seeds: [0, 1], overrides: ["env.impl=@IMPL@"]}\n'
        '  - {experiment: E, seeds: [0], overrides: ["env.impl=@IMPL@"]}\n'
        '  - {script: scripts/brax_reference.py, seeds: [0], overrides: ["env=ours"]}\n'
    )
    return runs, sweep


def _run(tmp_path, runs, sweep, max_hours):
    r = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/gpu/cost_cap.py",
            str(sweep),
            str(tmp_path / "impl"),
            "--max-hours",
            str(max_hours),
            "--runs-dir",
            str(runs),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return r.returncode, json.loads((tmp_path / "cost_cap.json").read_text())


def test_projection_and_cap(tmp_path):
    runs, sweep = _setup(tmp_path, rate=200e3)
    code, rec = _run(tmp_path, runs, sweep, max_hours=100)
    assert code == 0 and rec["pass"] and rec["jobs_remaining"] == 4
    grid = 82 * 256 * 1000
    expect = ((2 * (2e8 + grid) + 1.5 * (2e8 + grid) + 2e8) / 200e3 / 3600) + 4 * 1.5 / 60
    assert abs(rec["projected_hours"] - expect) < 1e-6
    code, rec = _run(tmp_path, runs, sweep, max_hours=1)
    assert code == 1 and not rec["pass"]


def test_finished_runs_are_not_counted(tmp_path):
    runs, sweep = _setup(tmp_path, rate=200e3)
    done = runs / f"A_mlp_s0_{SHA}"
    done.mkdir()
    (done / "DONE").write_text("{}")
    code, rec = _run(tmp_path, runs, sweep, max_hours=100)
    assert rec["jobs_remaining"] == 3 and rec["jobs_already_done"] == 1
