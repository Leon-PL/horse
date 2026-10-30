import json

import numpy as np
import pytest

from src.tail_smoothing import (STRENGTHS, activation_summary, fit_strength,
                                smooth_tail, validate_strength)


def test_identity_and_no_activated_tail_are_exact():
    p, q = [.2, .8], [.3, .7]
    assert np.array_equal(smooth_tail(p, q, [2], 0), p)
    assert np.array_equal(smooth_tail(p, q, [2], .25), p)


def test_only_severely_suppressed_tail_is_raised_and_other_odds_preserved():
    p, q = np.array([1e-9, .003, .2, .796999999]), [.004, .005, .3, .691]
    actual = smooth_tail(p, q, [4], .25)
    assert actual.sum() == pytest.approx(1)
    assert actual[0] >= .25 * q[0] / 1.25
    assert actual[0] > .00099
    # The .3% runner is not suppressed enough; it keeps its odds to the others.
    np.testing.assert_allclose(actual[1:] / actual[1], p[1:] / p[1], rtol=1e-14)
    summary = activation_summary(p, q, [4], .25)
    assert summary["directly_lifted_runners"] == 1
    assert summary["affected_races"] == 1


def test_no_cross_race_normalization_and_permutation_equivariance():
    p, q = [0, 1, .3, .7], [.01, .99, .4, .6]
    actual = smooth_tail(p, q, [2, 2], .1)
    np.testing.assert_allclose(actual[2:], p[2:])
    permuted = smooth_tail([1, 0, .7, .3], [.99, .01, .6, .4], [2, 2], .1)
    np.testing.assert_array_equal(actual, permuted[[1, 0, 3, 2]])
    assert actual[:2].sum() == pytest.approx(1)
    noisy_p = [0, 1, .3 + 1e-16, .7]
    assert np.array_equal(smooth_tail(noisy_p, q, [2, 2], .1)[2:], noisy_p[2:])


def test_tail_boundary_and_floor_ratio_are_respected():
    # A runner at exactly 1% is outside the fixed tail, even when suppressed.
    p, q = [.01, .001, .989], [.1, .01, .89]
    actual = smooth_tail(p, q, [3], .25)
    assert actual[0] / actual[2] == pytest.approx(p[0] / p[2])
    assert actual[1] > p[1]


def test_sparse_winning_tail_can_learn_positive_strength_with_score_safeguards():
    p, q = [1e-9, 1 - 1e-9], [.004, .996]
    result = fit_strength(p, q, [1, 0], [2])
    assert result["learned_strength"] == max(STRENGTHS)
    assert all(c["eligible"] for c in result["candidates"])
    accepted = validate_strength(result["learned_strength"], p, q, [1, 0], [2])
    assert accepted["accepted"] and accepted["selected_strength"] == .25
    rejected = validate_strength(.25, p, q, [0, 1], [2])
    assert not rejected["accepted"] and rejected["selected_strength"] == 0


def test_losing_tail_does_not_force_a_correction_and_serialization_replays():
    p, q = [1e-9, 1 - 1e-9], [.004, .996]
    assert fit_strength(p, q, [0, 1], [2])["learned_strength"] == 0
    strength = json.loads(json.dumps({"strength": .03}))["strength"]
    assert np.array_equal(smooth_tail(p, q, [2], strength), smooth_tail(p, q, [2], .03))


@pytest.mark.parametrize("strength", [-.1, .26, np.nan])
def test_invalid_strength_rejected(strength):
    with pytest.raises(ValueError):
        smooth_tail([.1, .9], [.2, .8], [2], strength)


@pytest.mark.parametrize("p,q,g", [([.1, .8], [.2, .8], [2]),
    ([.1, .9], [0, 1], [2]), ([.1, .9], [.2, .8], [1, 1]),
    ([.1, .9], [.2, .8], [2.5]), ([np.nan, 1], [.2, .8], [2])])
def test_malformed_or_partial_fields_rejected(p, q, g):
    with pytest.raises(ValueError):
        smooth_tail(p, q, g, .1)
