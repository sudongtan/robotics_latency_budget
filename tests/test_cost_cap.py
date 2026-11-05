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
        "  - {tier: 0}\n"
        '  - {experiment: A, seeds: [0, 1], overrides: ["env.impl=@IMPL@"]}\n'
        "  - {tier: 1}\n"
        '  - {experiment: E, seeds: [0], overrides: ["env.impl=@IMPL@"]}\n'
        "  - {tier: 2}\n"
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


GRID = 82 * 256 * 1000
H_A = (2e8 + GRID) / 200e3 / 3600 + 1.5 / 60  # one MLP run at 200k steps/s
H_E = 1.5 * (2e8 + GRID) / 200e3 / 3600 + 1.5 / 60
H_B = 2e8 / 200e3 / 3600 + 1.5 / 60


def test_all_tiers_fit(tmp_path):
    runs, sweep = _setup(tmp_path, rate=200e3)
    code, rec = _run(tmp_path, runs, sweep, max_hours=100)
    assert code == 0 and rec["pass"] and rec["stop_before_tier"] is None
    assert abs(rec["projected_hours_all"] - (2 * H_A + H_E + H_B)) < 1e-6
    assert not (tmp_path / "stop_before_tier").exists()


def test_cap_trims_whole_tiers(tmp_path):
    runs, sweep = _setup(tmp_path, rate=200e3)
    cap = 2 * H_A + H_E + 0.01  # tiers 0 and 1 fit, tier 2 does not
    code, rec = _run(tmp_path, runs, sweep, max_hours=cap)
    assert code == 0 and rec["runs_tiers"] == [0, 1] and rec["stop_before_tier"] == 2
    assert (tmp_path / "stop_before_tier").read_text() == "2"


def test_nothing_fits_stops(tmp_path):
    runs, sweep = _setup(tmp_path, rate=200e3)
    code, rec = _run(tmp_path, runs, sweep, max_hours=0.1)
    assert code == 1 and not rec["pass"]


def test_finished_runs_are_not_counted(tmp_path):
    runs, sweep = _setup(tmp_path, rate=200e3)
    done = runs / f"A_mlp_s0_{SHA}"
    done.mkdir()
    (done / "DONE").write_text("{}")
    code, rec = _run(tmp_path, runs, sweep, max_hours=100)
    assert rec["jobs_already_done"] == 1 and rec["tiers"]["0"]["jobs"] == 1


def test_env_var_overrides_cap(tmp_path, monkeypatch):
    runs, sweep = _setup(tmp_path, rate=200e3)
    monkeypatch.setenv("COST_CAP_HOURS", "100")
    code, rec = _run(tmp_path, runs, sweep, max_hours=0.1)  # flag says 0.1 h
    assert code == 0 and rec["max_hours"] == 100 and rec["stop_before_tier"] is None
