"""launch_sweep.sh runner: TIER markers, stop_before_tier (cost cap, D36), STEP failure."""

import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _launch(tmp_path, body, stop_tier=None):
    sweep = tmp_path / "t.yaml"
    sweep.write_text("jobs:\n" + body)
    out = subprocess.run(
        ["bash", "scripts/gpu/launch_sweep.sh", str(sweep)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    logdir = ROOT / out.split("status: ")[1].strip().rsplit("/status", 1)[0]
    if stop_tier is not None:  # what cost_cap.py would write (before the first TIER line)
        (logdir / "stop_before_tier").write_text(str(stop_tier))
    status = logdir / "status"
    for _ in range(100):
        if status.exists() and "SWEEP DONE" in status.read_text():
            break
        time.sleep(0.2)
    return logdir, status.read_text()


BODY = (
    '  - {step: "sleep 1"}\n'  # gives the test time to write stop_before_tier
    "  - {tier: 0, label: calib}\n"
    '  - {step: "echo t0"}\n'
    "  - {tier: 1, label: core}\n"
    '  - {step: "echo t1"}\n'
    "  - {tier: 2, label: extra}\n"
    '  - {step: "echo t2"}\n'
)


def test_runs_all_tiers_without_stop_file(tmp_path):
    logdir, st = _launch(tmp_path, BODY)
    try:
        assert "TIER 2 extra" in st and "echo t2" in st and "SKIPPING" not in st
    finally:
        subprocess.run(["rm", "-rf", str(logdir)])


def test_stops_at_cost_cap_tier(tmp_path):
    logdir, st = _launch(tmp_path, BODY, stop_tier=2)
    try:
        assert "echo t1" in st and "echo t2" not in st
        assert "SKIPPING tier 2 and later (cost cap)" in st and "SWEEP DONE" in st
    finally:
        subprocess.run(["rm", "-rf", str(logdir)])


def test_failed_step_stops_queue(tmp_path):
    logdir, st = _launch(tmp_path, '  - {step: "false"}\n  - {step: "echo never"}\n')
    try:
        assert "STEP FAILED" in st and "echo never" not in st
    finally:
        subprocess.run(["rm", "-rf", str(logdir)])
