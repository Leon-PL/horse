"""Compiled history kernels against the pandas calls they stand in for."""
import numpy as np
import pandas as pd
import pytest

from src.fe_incremental import (
    CountsAccumulator, PlainCountsAccumulator, _group_totals, _nb_group_cumsum,
)
from src.feature_engineer import _nb_expanding_prior


def noisy(rng, n):
    values = rng.normal(0, 1, n) * 10.0 ** rng.integers(-6, 7, n)
    values[rng.random(n) < .15] = np.nan
    values[rng.random(n) < .2] = 2.5
    return values


@pytest.mark.parametrize("seed", range(5))
def test_group_cumsum_matches_pandas(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(-1, 40, 5000)
    values = noisy(rng, 5000)
    frame = pd.Series(values).groupby(labels)
    out, totals = _nb_group_cumsum(values, labels.astype(np.int64), 40)
    expected = frame.cumsum().to_numpy()
    known = labels >= 0
    assert np.array_equal(out[known], expected[known], equal_nan=True)
    assert np.isnan(out[~known]).all()
    assert np.array_equal(totals, frame.sum().reindex(range(40), fill_value=0.0).to_numpy())


def test_totals_follow_pandas_when_a_value_is_infinite():
    values = np.array([1.0, np.inf, 3.0, -np.inf, 2.0, 5.0])
    labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    _, totals = _nb_group_cumsum(values, labels, 3)
    summed = _group_totals(values, labels, totals)
    assert np.array_equal(summed, pd.Series(values).groupby(labels).sum().to_numpy())


def test_accumulators_keep_a_float_state_a_later_chunk_can_extend():
    keys, races = np.array(["a", "a", "b", "a"]), np.array([1, 1, 1, 2])
    values = np.array([1.0, 2.0, 4.0, 8.0])
    race_safe = CountsAccumulator()
    assert race_safe.transform(keys, races, values).tolist() == [0.0, 0.0, 0.0, 3.0]
    assert race_safe.to_state() == {"a": 11.0, "b": 4.0}
    plain = PlainCountsAccumulator({"a": 10.0})
    assert plain.transform(keys, values).tolist() == [10.0, 11.0, 0.0, 13.0]
    assert plain.to_state() == {"a": 21.0, "b": 4.0}


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("split", [0, 1, 37, 400])
def test_expanding_prior_matches_pandas_from_any_row(seed, split):
    rng = np.random.default_rng(seed)
    values = noisy(rng, 400)
    values[rng.random(400) < .03] = np.inf
    values[rng.random(400) < .03] = -np.inf
    values[100:180] = 7.25  # a long run of one value resets the deviation
    shifted = pd.Series(values).shift(1).expanding()
    medians, deviations = _nb_expanding_prior(values[:split], values[split:])
    assert np.array_equal(medians, shifted.median().to_numpy()[split:], equal_nan=True)
    assert np.array_equal(deviations, shifted.std().to_numpy()[split:], equal_nan=True)


def test_expanding_prior_with_nothing_seen():
    medians, deviations = _nb_expanding_prior(np.empty(0), np.array([np.nan, 1.0]))
    assert np.isnan(medians[:2]).tolist() == [True, True]
    assert np.isnan(deviations).all()
