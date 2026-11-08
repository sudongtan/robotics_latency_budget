"""Analysis statistics (prereg §5): crossings, delay margin, IQM, bootstrap, verdicts."""

import numpy as np
import pytest

from latrl.analysis import stats as S
from latrl.analysis.summarize import parse_run_id

X = np.arange(0, 121, 4.0)


def test_crossing_interpolates_between_grid_points():
    s = np.where(X < 40, 1.0, 0.0)
    s[X == 40] = 0.6
    s[X == 44] = 0.2  # 0.6 at 40, 0.2 at 44 -> 0.5 crossed at 41
    c = S.crossing(X, s, 0.5)
    assert c.value == pytest.approx(41.0) and not c.censored


def test_crossing_censored_and_at_start():
    assert S.crossing(X, np.ones_like(X), 0.5) == S.Crossing(120.0, True)
    assert S.crossing(X, np.zeros_like(X), 0.5) == S.Crossing(0.0, False)


def test_crossing_uses_first_drop_not_last():
    s = np.ones_like(X)
    s[5] = 0.0  # dip at 20 ms, recovers afterwards
    assert S.crossing(X, s, 0.5).value == pytest.approx(18.0)


def test_delay_margin():
    s = np.ones_like(X)
    s[X >= 48] = 0.3
    assert S.delay_margin(X, s) == 44.0
    s[2] = 0.4  # dip at 8 ms: margin stops before it
    assert S.delay_margin(X, s) == 4.0
    assert np.isnan(S.delay_margin(X, np.zeros_like(X)))
    assert S.delay_margin(X, np.ones_like(X)) == 120.0


def test_iqm_matches_rliable():
    from rliable import metrics

    x = np.random.default_rng(0).normal(size=10)
    assert S.iqm(x) == pytest.approx(metrics.aggregate_iqm(x[:, None]))


def test_bootstrap_ci_is_deterministic_and_covers_point():
    x = np.random.default_rng(1).normal(50, 5, size=10)
    a = S.iqm_ci(x, 2000, np.random.default_rng(7), 0.95)
    b = S.iqm_ci(x, 2000, np.random.default_rng(7), 0.95)
    assert a == b and a.lo < a.point < a.hi


def test_diff_ci_detects_shift():
    rng = np.random.default_rng(2)
    x, y = rng.normal(60, 3, 10), rng.normal(40, 3, 10)
    d = S.iqm_diff_ci(x, y, 2000, rng, 0.95)
    assert d.lo > 10 and d.point == pytest.approx(S.iqm(x) - S.iqm(y))


def test_recovery_point():
    ea, ec, ed = np.full(5, 0.10), np.full(5, 0.14), np.full(5, 0.11)
    r = S.recovery_ci(ea, ec, ed, 200, np.random.default_rng(0), 0.95)
    assert r.point == pytest.approx(0.75)


def test_verdict():
    assert S.verdict(True, False) == S.SUPPORTED
    assert S.verdict(False, True) == S.NOT_SUPPORTED
    assert S.verdict(False, False) == S.INCONCLUSIVE


@pytest.mark.parametrize(
    "rid,expect",
    [
        ("A_mlp_s0_4a4a467", ("A", "mlp", 0)),
        ("C_obs_mlp_s3_4a4a467", ("C_obs", "mlp", 3)),
        ("DH8_hist8_s2_4a4a467", ("DH8", "hist8", 2)),
        ("F_mlppriv_s9_4a4a467", ("F", "mlppriv", 9)),
        ("E_tm_gru_s4_4a4a467", ("E_tm", "gru", 4)),
    ],
)
def test_parse_run_id(rid, expect):
    p = parse_run_id(rid)
    assert (p["cond"], p["policy"], p["seed"]) == expect


@pytest.mark.parametrize(
    "rid",
    [
        "bench_jax_A_mlp_s0_4a4a467",
        "smoke_A_mlp_s0_17163eb",
        "brax_ours_s0_4a4a467",
        "phase1_delay_demo_s0_b7141ef",
    ],
)
def test_parse_run_id_skips_non_experiments(rid):
    assert parse_run_id(rid) is None
