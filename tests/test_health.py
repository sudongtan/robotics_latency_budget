"""Session health monitor (scripts/gpu/health.py): alert logic on synthetic probes."""

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("health", ROOT / "scripts" / "gpu" / "health.py")
health = importlib.util.module_from_spec(spec)
sys.modules["health"] = health  # dataclasses needs the module registered
spec.loader.exec_module(health)
LIM = health.Limits()
MAC = {"free_gb": 200.0, "last_pull_min": 2.0}

STATUS = [
    "2026-11-06T10:00:00+0000 start 1 [RUN]: uv run python scripts/train.py experiment=A",
    "2026-11-06T10:20:00+0000 ok 1",
    "2026-11-06T10:20:00+0000 start 2 [RUN]: uv run python scripts/train.py experiment=C",
]


def probe(**kw):
    p = {
        "disk_free_gb": 40.0,
        "disk_used_pct": 30.0,
        "ram_avail_gb": 50.0,
        "gpu_util": 95.0,
        "gpu_mem_used": 18000.0,
        "gpu_mem_total": 24000.0,
        "status_file": "s",
        "runner": 1.0,
        "log_age_s": 30.0,
        "status": list(STATUS),
    }
    p.update(kw)
    return p


def run(p, mac=MAC, state=None):
    state = {} if state is None else state
    return health.assess(p, mac, state, LIM), state


def test_healthy():
    (alerts, infos, summary), _ = run(probe())
    assert alerts == set() and infos == []
    assert any("jobs ok 1, failed 0" in s for s in summary)


def test_parse_probe():
    p = health.parse_probe(
        "disk_free_gb=12.5\nrunner=1\nstatus_file=runs/_sweeps/x/status\nS:a b\n"
    )
    assert p["disk_free_gb"] == 12.5 and p["runner"] == 1.0 and p["status"] == ["a b"]


def test_disk_and_memory():
    (alerts, _, _), _ = run(
        probe(disk_free_gb=5.0, disk_used_pct=95.0, ram_avail_gb=1.0, gpu_mem_used=23900.0)
    )
    text = " ".join(alerts)
    assert "disk low" in text and "RAM low" in text and "GPU memory" in text
    (alerts, _, _), _ = run(probe(), mac={"free_gb": 3.0, "last_pull_min": 1.0})
    assert any("Mac disk low" in a for a in alerts)


def test_unreachable_needs_two_checks():
    state = {}
    (a1, _, _), state = run(None, state=state)
    (a2, _, _), state = run(None, state=state)
    assert a1 == set() and any("unreachable" in a for a in a2)
    (a3, _, _), state = run(probe(), state=state)
    assert a3 == set()


def test_failed_job_is_info_gate_failure_is_alert():
    st = STATUS + ["2026-11-06T10:40:00+0000 FAILED(1) 2"]
    (alerts, infos, _), state = run(probe(status=st))
    assert alerts == set() and any("job failed" in i for i in infos)
    (alerts, infos, _), _ = run(probe(status=st), state=state)
    assert infos == []  # reported once
    st2 = st + [
        "x start 3 [GATE]: gate",
        "x FAILED(1) 3",
        "x GATE FAILED: queue stopped",
        "x SWEEP DONE",
    ]
    (alerts, infos, _), _ = run(probe(status=st2, runner=0.0), state=state)
    assert any("sweep stopped" in a for a in alerts)
    assert any("sweep done" in i for i in infos)


def test_gate_pass_reported_once():
    st = STATUS + ["x start 3 [GATE]: gate", "x ok 3"]
    (_, infos, _), state = run(probe(status=st))
    assert any("gate PASSED" in i for i in infos)
    (_, infos, _), _ = run(probe(status=st), state=state)
    assert infos == []


def test_stall_runner_pull_idle():
    (alerts, _, _), _ = run(probe(log_age_s=45 * 60))
    assert any("silent" in a for a in alerts)
    (alerts, _, _), _ = run(probe(runner=0.0))
    assert any("runner not running" in a for a in alerts)
    (alerts, _, _), _ = run(probe(), mac={"free_gb": 200.0, "last_pull_min": 40.0})
    assert any("has not pulled" in a for a in alerts)
    state = {}
    for i in range(3):
        (alerts, _, _), state = run(probe(gpu_util=2.0), state=state)
        assert any("idle" in a for a in alerts) == (i == 2)


def test_no_running_alerts_after_done():
    st = STATUS + ["x ok 2", "x SWEEP DONE"]
    (alerts, _, _), _ = run(
        probe(status=st, runner=0.0, log_age_s=99999.0, gpu_util=0.0),
        mac={"free_gb": 200.0, "last_pull_min": 999.0},
    )
    assert alerts == set()
