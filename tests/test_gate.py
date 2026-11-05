"""Automatic gate (scripts/gpu/gate.py) and backend picker on synthetic run directories."""

import importlib.util
import json
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHA = "abc1234"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "gpu" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load("gate")
pick = _load("pick_backend")


def _run(runs, name, evals, key="eval/episode_return", done=True, grid_rows=None):
    d = runs / f"{name}_{SHA}"
    d.mkdir(parents=True)
    rows = [{"env_steps": s, key: r, "eval/survived": sv} for s, r, sv in evals]
    (d / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if done:
        (d / "DONE").write_text("{}")
    if grid_rows is not None:
        pd.DataFrame({"x": range(grid_rows)}).to_parquet(d / "eval_mjx.parquet")
        (d / "config.yaml").write_text("mjx_eval:\n  episodes: 4\n")
    return d


CURVE = [(20, 10.0, 0.9), (40, 15, 0.9), (60, 18, 0.9), (80, 19.5, 0.95), (100, 20.0, 0.96)]
# converged: +2.5 % after 80 %


def _setup(tmp_path, ours=CURVE, brax_final=20.0, grid_rows=40, done=True):
    runs = tmp_path / "runs"
    for s in (0, 1):
        _run(runs, f"A_mlp_s{s}", ours, done=done)
        _run(runs, f"brax_ours_s{s}", [(100, brax_final, 1.0)], key="eval/episode_reward")
    for name in ("C_mlp_s0", "D_hist4_s0", "E_gru_s0"):
        _run(runs, name, CURVE, grid_rows=grid_rows)
    return runs


def test_gate_passes(tmp_path):
    res = gate.check(_setup(tmp_path), SHA, n_cells=10)
    assert res["pass"], res


@pytest.mark.parametrize(
    "kwargs, failing",
    [
        ({"brax_final": 30.0}, "G2_parity"),  # ours 20 < 0.9 × 30
        (
            {
                "ours": [
                    (20, 5.0, 0.9),
                    (40, 6, 0.9),
                    (60, 8, 0.9),
                    (80, 10.0, 0.95),
                    (100, 20.0, 0.96),
                ]
            },
            "G3_converged",
        ),
        ({"ours": [(50, 20.0, 0.96), (100, 20.0, 0.96)]}, "G3_converged"),  # too few evals
        (
            {
                "ours": [
                    (20, 19, 0.5),
                    (40, 19, 0.5),
                    (60, 19, 0.5),
                    (80, 19.5, 0.5),
                    (100, 20, 0.5),
                ]
            },
            "G4_walks",
        ),
        ({"grid_rows": 39}, "G5_pipeline"),
        ({"done": False}, "G1_complete"),
    ],
)
def test_gate_fails(tmp_path, kwargs, failing):
    res = gate.check(_setup(tmp_path, **kwargs), SHA, n_cells=10)
    assert not res["pass"]
    assert not res["criteria"][failing]["pass"]


def test_pick_backend(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for impl, t in (("jax", 100.0), ("warp", 60.0)):
        d = runs / f"bench_{impl}_A_mlp_s0_{SHA}"
        d.mkdir(parents=True)
        (d / "metrics.jsonl").write_text(json.dumps({"train_s": t}) + "\n")
        (d / "DONE").write_text("{}")
    assert pick.train_s(runs / f"bench_warp_A_mlp_s0_{SHA}") == 60.0
    (runs / f"bench_warp_A_mlp_s0_{SHA}" / "DONE").unlink()  # unfinished run loses
    assert pick.train_s(runs / f"bench_warp_A_mlp_s0_{SHA}") == float("inf")


def test_session_sweep_expands():
    sweep = _load("sweep")
    cmds = sweep.commands(str(ROOT / "configs" / "sweeps" / "session_s.yaml"))
    kinds = [c.split()[0] for c in cmds]
    assert kinds.count("GATE") == 1 and kinds.count("STEP") == 2 and kinds.count("RUN") == 101
    gate_i = kinds.index("GATE")
    pick, cap = (i for i, k in enumerate(kinds) if k == "STEP")
    tiers = {int(c.split()[1]): i for i, c in enumerate(cmds) if c.startswith("TIER")}
    assert list(tiers) == [0, 1, 2, 3, 4]
    assert "pick_backend" in cmds[pick] and "cost_cap" in cmds[cap]
    assert pick < cap < tiers[0] < gate_i < tiers[1] < tiers[2] < tiers[3] < tiers[4]
    calib = " ".join(cmds[tiers[0] : gate_i])
    for name in (
        "experiment=A seed=0",
        "experiment=A seed=1",
        "experiment=C seed=0",
        "experiment=D seed=0",
        "experiment=E seed=0",
        "env=ours",
    ):
        assert name in calib, name  # everything the gate reads runs before it
    assert all("@IMPL@" in c for c in cmds[cap + 1 :] if c.startswith("RUN"))


def _tier(cmds, n):
    idx = [i for i, c in enumerate(cmds) if c.startswith(("TIER", "GATE"))] + [len(cmds)]
    start = next(i for i, c in enumerate(cmds) if c.startswith(f"TIER {n}"))
    end = next(i for i in idx if i > start)
    return [c for c in cmds[start:end] if c.startswith("RUN")]


def test_tier_contents_and_balanced_rounds():
    """D36: tier 1 holds every core condition incl. the obs-delay runs H5 needs, seeds 0-4;
    within a tier, seed s of every condition precedes seed s+1 of any."""
    import re

    cmds = _load("sweep").commands(str(ROOT / "configs" / "sweeps" / "session_s.yaml"))

    def pairs(n):
        return [
            (
                re.search(r"experiment=(\S+)", c).group(1).split("/")[-1],
                int(re.search(r"seed=(\d+)", c).group(1)),
            )
            for c in _tier(cmds, n)
        ]

    t1 = pairs(1)
    t1_all = set(t1) | {("A", 0), ("A", 1), ("C", 0), ("D", 0), ("E", 0)}  # + calibration
    core = ["A", "C", "D", "E", "F", "C_obs", "D_obs", "E_obs"]
    assert t1_all == {(e, s) for e in core for s in range(5)}
    for n in (1, 2, 3, 4):
        seeds = [s for _, s in pairs(n)]
        assert seeds == sorted(seeds), n
    assert {e for e, _ in pairs(2)} == {"C", "D", "E", "F"}
    assert {s for _, s in pairs(2)} == set(range(5, 10))
    assert {e for e, _ in pairs(3)} == {"B20", "B40", "B60"}
    assert {e for e, _ in pairs(4)} == {"C_tm", "D_tm", "E_tm", "DH2", "DH8"}


def test_session_sweep_matches_generator():
    """configs/sweeps/session_s.yaml is generated; hand edits would drift from the script."""
    gen = _load("make_session_sweep")
    expected = gen.HEADER + "\n".join(gen.lines()) + "\n"
    assert (ROOT / "configs" / "sweeps" / "session_s.yaml").read_text() == expected
