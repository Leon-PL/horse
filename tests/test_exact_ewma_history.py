import numpy as np
import pandas as pd
import pytest

from src.feature_engineer import _fast_grouped_rolling_core, _rolling_with_seed


def test_missing_results_beyond_tail_do_not_erase_known_form():
    frame = pd.DataFrame({"entity": "A", "race_id": [f"r{i:04d}" for i in range(600)],
        "race_date": pd.date_range("2020-01-01", periods=600),
        "value": np.r_[np.ones(20), np.full(400, np.nan), np.full(180, .4)]})
    ops = [("form", "ewma", 1-2**(-1/20))]
    full = _fast_grouped_rolling_core(frame, "entity", "value", ops)["form"]
    _, state = _rolling_with_seed(frame.iloc[:350], "entity", "value", ops, {})
    actual, _ = _rolling_with_seed(frame.iloc[350:].reset_index(drop=True), "entity", "value", ops, state)
    np.testing.assert_array_equal(actual["form"], full[350:])
    assert actual["form"][0] == 1.


def test_many_chunk_boundaries_and_same_race_peers_match_one_full_calculation():
    rng = np.random.default_rng(912)
    n = 4200
    frame = pd.DataFrame({"entity": np.tile(["A", "A", "B"], n//3),
        "race_id": [f"r{i//3:05d}" for i in range(n)],
        "race_date": pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n)//3, unit="D"),
        "value": rng.uniform(0,1,n)})
    frame.loc[rng.random(n)<.35, "value"] = np.nan
    ops = [("mean3", "mean", 3), ("fast", "ewma", .3), ("slow", "ewma", 1-2**(-1/20))]
    full = _fast_grouped_rolling_core(frame, "entity", "value", ops)
    for size in [117, 333, 900]:
        state, pieces = {}, {name: [] for name, _, _ in ops}
        for start in range(0, n, size):
            actual, state = _rolling_with_seed(frame.iloc[start:start+size].reset_index(drop=True), "entity", "value", ops, state)
            for name in pieces:
                pieces[name].append(actual[name])
        for name, parts in pieces.items():
            np.testing.assert_allclose(np.concatenate(parts), full[name], rtol=0, atol=2e-14, equal_nan=True)


def test_legacy_truncated_state_requires_a_rebuild():
    frame = pd.DataFrame({"entity": ["A"], "race_id": ["a"], "race_date": [pd.Timestamp("2025-01-01")], "value": [1.]})
    with pytest.raises(ValueError, match="Legacy truncated"):
        _rolling_with_seed(frame, "entity", "value", [("form", "ewma", .1)], {"A": [.5,.6]})
