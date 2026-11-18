"""Learnability gate (F50): a condition passes only if enough pilot seeds finished walking."""

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("learn_gate", ROOT / "scripts/gpu/learn_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def _run(d: pathlib.Path, ret: float | None, done: bool = True) -> None:
    d.mkdir(parents=True)
    if ret is not None:
        rows = [{"eval/episode_return": 3.0}, {"eval/episode_return": ret}]
        (d / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if done:
        (d / "DONE").write_text("")


def test_gate(tmp_path):
    _run(tmp_path / "X_mlp_s0_abc1234", 28.0)
    _run(tmp_path / "X_mlp_s1_abc1234", 19.5)
    _run(tmp_path / "Y_gru_s0_abc1234", 19.5)
    _run(tmp_path / "Y_gru_s1_abc1234", 20.0)
    _run(tmp_path / "Z_mlp_s0_abc1234", 29.0)
    _run(tmp_path / "Z_mlp_s1_abc1234", 29.0, done=False)  # still running: not counted
    ok, lines = gate.check(tmp_path, ["X_mlp"], [0, 1], "abc1234", 25.0, 1)
    assert ok and lines[0].startswith("PASS")
    ok, lines = gate.check(tmp_path, ["X_mlp", "Y_gru"], [0, 1], "abc1234", 25.0, 1)
    assert not ok and lines[1].startswith("FAIL")
    ok, _ = gate.check(tmp_path, ["Z_mlp"], [0, 1], "abc1234", 25.0, 1)
    assert not ok  # a missing or unfinished pilot run fails the gate
    ok, _ = gate.check(tmp_path, ["X_mlp"], [0, 1], "abc1234", 25.0, 2)
    assert not ok


def test_cli_as_the_sweep_calls_it(tmp_path, capsys):
    """The exact argument order of the F2 sweep line (D51: names after --seeds)."""
    _run(tmp_path / "X_mlp_s0_abc1234", 28.0)
    _run(tmp_path / "X_mlp_s1_abc1234", 19.5)
    _run(tmp_path / "Y_gru_s0_abc1234", 19.5)
    _run(tmp_path / "Y_gru_s1_abc1234", 20.0)
    base = ["--runs-dir", str(tmp_path), "--commit", "abc1234", "--seeds", "0,1"]
    for names, code in ((["X_mlp"], 0), (["X_mlp", "Y_gru"], 1)):
        try:
            gate.main(base + names)
        except SystemExit as e:
            assert e.code == code
    assert "FAIL Y_gru" in capsys.readouterr().out
