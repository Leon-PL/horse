import numpy as np
import pytest

from src.forecast_evaluation import paired_interval, probability_metrics


@pytest.mark.parametrize("p, y, g", [
    ([.5, .5, .5], [1, 0, 0], [3]),  # incomplete/unnormalised book
    ([np.nan, .5], [1, 0], [2]),
    ([.5, .5], [1, 1], [2]),
    ([.5, .5], [1, 0], [2.5]),
    ([.5, .5], [1, 0], [3]),
    ([], [], []),
])
def test_invalid_forecast_cannot_be_scored(p, y, g):
    with pytest.raises(ValueError):
        probability_metrics(p, y, g)


def test_tied_winner_accuracy_is_permutation_invariant():
    a = probability_metrics([.4, .4, .2], [1, 0, 0], [3])[0]
    b = probability_metrics([.2, .4, .4], [0, 0, 1], [3])[0]
    assert a["top1_accuracy"] == b["top1_accuracy"] == .5
    assert a["brier"] == pytest.approx(b["brier"])


def test_date_cluster_interval_retains_race_weighted_estimand():
    # A block's size must not change the target to an unweighted day average.
    delta = np.array([1., 1., 1., -1.])
    result = paired_interval(delta, blocks=["a", "a", "a", "b"], draws=100)
    assert result["mean"] == .5
    assert result["ci95"] == [-1., 1.]
