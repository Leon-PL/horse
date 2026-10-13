"""The plan-4c harness must find a residual when one exists, and only then.

A harness that reports "the model beats the market" on data where the market
is already right would be the stablemate leak again; one that cannot find a
planted residual would bury a real lead. Both are tested on synthetic races.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "wf_market_offset", ROOT / "scripts" / "wf_market_offset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _races(n_races: int, residual: float, seed: int) -> pd.DataFrame:
    """Races whose winner follows the market plus `residual` * x0."""
    rng = np.random.default_rng(seed)
    k = 8
    race = np.repeat(np.arange(n_races), k)
    x0, x1 = rng.normal(size=n_races * k), rng.normal(size=n_races * k)
    log_mkt = rng.normal(size=n_races * k) * 1.2
    mkt = np.exp(log_mkt); mkt /= np.bincount(race, mkt)[race]
    true = np.exp(np.log(mkt) + residual * x0); true /= np.bincount(race, true)[race]
    won = np.zeros(n_races * k, dtype=int)
    for r in range(n_races):
        idx = np.arange(r * k, (r + 1) * k)
        won[rng.choice(idx, p=true[idx])] = 1
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(race // 40, unit="D")
    return pd.DataFrame({"race_id": race.astype(str), "race_date": dates, "won": won,
                         "x0": x0, "x1": x1, "mkt": mkt, "offset": np.log(mkt),
                         "_entry": 1 / mkt, "_close": 1 / mkt})


@pytest.mark.parametrize("residual, expect_gain", [(0.6, True), (0.0, False)])
def test_offset_model_learns_only_a_real_residual(mod, residual, expect_gain):
    train = _races(3000, residual, seed=1)
    test = _races(1500, residual, seed=2)
    test["race_id"] = "t" + test["race_id"]
    booster, best_iter, _ = mod.fit_offset_model(train, ["x0", "x1"])
    probs = mod.predict(booster, best_iter, test, ["x0", "x1"])
    block = mod.score_block(test, probs)
    delta = block["delta_ll"].mean()
    # Every race's probabilities must still sum to one.
    sums = pd.Series(probs).groupby(test["race_id"].to_numpy()).sum()
    assert np.allclose(sums, 1.0)
    if expect_gain:
        assert delta < -0.01, f"missed a planted residual: delta {delta:+.4f}"
    else:
        assert abs(delta) < 0.01, f"invented an edge from noise: delta {delta:+.4f}"


def test_race_softmax_and_logloss(mod):
    scores = np.log(np.array([0.5, 0.3, 0.2, 0.9, 0.1]))
    starts = np.array([0, 3])
    p = mod.race_softmax(scores, starts)
    assert np.allclose(p, [0.5, 0.3, 0.2, 0.9, 0.1])
    ll = mod.race_logloss_vector(p, np.array([0, 1, 0, 1, 0]), starts)
    assert np.allclose(ll, [-np.log(0.3), -np.log(0.9)])
