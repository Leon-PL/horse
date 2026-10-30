"""Generated fields and histories exercise existing causal/state contracts."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

from src.each_way import adjust_place_probs_for_race  # noqa: E402
from src.feature_engineer import _fast_grouped_rolling_core, _rolling_with_seed  # noqa: E402
from src.forecast_evaluation import valid_race_mask  # noqa: E402
from src.result_status import pending_card, result_labels  # noqa: E402

BOUNDED = settings(max_examples=40, deadline=None, derandomize=True, database=None)


@st.composite
def fields(draw):
    size = draw(st.integers(2, 18))
    winner = draw(st.integers(0, size - 1))
    nonfinish = draw(st.lists(st.booleans(), min_size=size, max_size=size))
    order = draw(st.permutations(tuple(range(size))))
    positions = [1 if i == winner else (0 if nonfinish[i] else 2) for i in range(size)]
    return pd.DataFrame({
        "race_id": ["race-a"] * size,
        "horse_name": [f"Horse {i}" for i in range(size)],
        "race_date": [pd.Timestamp("2026-01-01")] * size,
        "finish_position": positions,
        "won": [int(i == winner) for i in range(size)],
        "num_runners": [size] * size,
    }).iloc[list(order)].reset_index(drop=True)


@BOUNDED
@given(fields())
def test_permutations_and_real_nonfinishers_preserve_complete_field_eligibility(frame):
    assert valid_race_mask(frame).all()
    assert valid_race_mask(frame.iloc[::-1]).all()
    labels = result_labels(frame)
    assert labels.loc[frame.finish_position.eq(0)].eq("DNF").all()
    pending = pending_card(frame)
    assert not valid_race_mask(pending).any()
    assert result_labels(pending).eq("").all()


@BOUNDED
@given(fields())
def test_missing_runner_rejects_its_entire_field_without_poisoning_another(frame):
    good = frame.copy()
    good["race_id"] = "race-b"
    combined = pd.concat([frame.iloc[:-1], good], ignore_index=True)
    eligible = valid_race_mask(combined)
    assert not eligible.loc[combined.race_id.eq("race-a")].any()
    assert eligible.loc[combined.race_id.eq("race-b")].all()


@BOUNDED
@given(st.data(), st.integers(2, 20))
def test_place_projection_preserves_mass_bounds_and_win_floors(data, size):
    raw = np.array(data.draw(st.lists(st.floats(0, 1, allow_nan=False),
                                      min_size=size, max_size=size)))
    weights = np.array(data.draw(st.lists(st.floats(.01, 1, allow_nan=False),
                                          min_size=size, max_size=size)))
    win = weights / weights.sum()
    slots = data.draw(st.integers(1, min(4, size)))
    result = adjust_place_probs_for_race(raw, win, slots)
    assert np.isfinite(result).all()
    assert (result >= win - 1e-12).all()
    assert (result <= 1 + 1e-12).all()
    assert result.sum() == pytest.approx(slots, abs=2e-10)


@settings(max_examples=16, deadline=None, derandomize=True, database=None)
@given(st.data(), st.integers(8, 80))
def test_generated_chunk_boundaries_match_full_causal_history(data, days):
    values = data.draw(st.lists(st.one_of(st.none(), st.floats(0, 1, allow_nan=False)),
                                min_size=days * 3, max_size=days * 3))
    cut = data.draw(st.integers(1, days - 1)) * 3
    frame = pd.DataFrame({
        "entity": np.tile(["A", "A", "B"], days),
        "race_id": np.repeat([f"r{i:03d}" for i in range(days)], 3),
        "race_date": np.repeat(pd.date_range("2026-01-01", periods=days), 3),
        "value": pd.Series(values, dtype=float),
    })
    ops = [("mean3", "mean", 3), ("ewma", "ewma", .2)]
    full = _fast_grouped_rolling_core(frame, "entity", "value", ops)
    prefix, state = _rolling_with_seed(frame.iloc[:cut], "entity", "value", ops, {})
    suffix, _ = _rolling_with_seed(frame.iloc[cut:].reset_index(drop=True),
                                   "entity", "value", ops, state)
    future_changed = frame.copy()
    future_changed.loc[cut:, "value"] = .99
    changed = _fast_grouped_rolling_core(future_changed, "entity", "value", ops)
    for name, _, _ in ops:
        np.testing.assert_allclose(np.r_[prefix[name], suffix[name]], full[name],
                                   atol=2e-13, rtol=0, equal_nan=True)
        np.testing.assert_allclose(changed[name][:cut], full[name][:cut],
                                   atol=0, rtol=0, equal_nan=True)
