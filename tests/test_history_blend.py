import json

import numpy as np
import pytest

from src.history_blend import CONTEXTS, apply_context_blend, contexts_from_flags, fit_context_blend


def test_context_uses_whole_field_and_requires_observed_flags():
    assert contexts_from_flags([1, 0, 0, 0], [2, 2]).tolist() == [1, 0]
    with pytest.raises(ValueError, match="binary"):
        contexts_from_flags([np.nan, 0], [2])


def test_one_coefficient_applies_to_every_runner_in_a_field():
    p, q = [.1, .9, .7, .3], [.3, .7, .6, .4]
    actual = apply_context_blend(p, q, [2, 2], [1, 0], dict(zip(CONTEXTS, [0, 1])))
    np.testing.assert_array_equal(actual, [.3, .7, .7, .3])
    np.testing.assert_array_equal(actual.reshape(2, 2).sum(axis=1), [1, 1])


def test_zero_weights_and_serialization_are_exact():
    weights = dict(zip(CONTEXTS, [0., .27]))
    restored = json.loads(json.dumps(weights))
    p, q = [.2, .8, .7, .3], [.3, .7, .6, .4]
    assert np.array_equal(apply_context_blend(p, q, [2, 2], [0, 1], restored),
                          apply_context_blend(p, q, [2, 2], [0, 1], weights))
    assert np.array_equal(apply_context_blend(p, q, [2, 2], [0, 1], dict(zip(CONTEXTS, [0, 0]))), p)


def synthetic():
    p = np.tile([.7, .3], 100)
    q = np.r_[np.tile([.6, .4], 50), np.tile([.9, .1], 50)]
    y = np.tile([1, 0], 100)
    return (p, q, y, np.repeat(2, 100), np.r_[np.zeros(50), np.ones(50)])


def test_weights_fit_earlier_contexts_and_validation_only_accepts_or_rejects():
    fit = synthetic()
    result = fit_context_blend(fit, fit)
    assert result[CONTEXTS[0]]["selected_weight"] == 0
    assert result[CONTEXTS[1]]["selected_weight"] == 1
    validation = list(synthetic())
    validation[2] = validation[2].copy()
    validation[2][100:] = np.tile([0, 1], 50)
    rejected = fit_context_blend(fit, validation)
    assert rejected[CONTEXTS[1]]["fit"]["weight"] == 1
    assert rejected[CONTEXTS[1]]["selected_weight"] == 0


def test_insufficient_earlier_races_fall_back_to_identity():
    p, q, y, g, c = synthetic()
    small = p[:20], q[:20], y[:20], g[:10], c[:10]
    result = fit_context_blend(small, small)
    assert all(s["selected_weight"] == 0 for s in result.values())


@pytest.mark.parametrize("weights", [{CONTEXTS[0]: 0}, dict(zip(CONTEXTS, [0, 1.2]))])
def test_invalid_weights_fail_closed(weights):
    with pytest.raises(ValueError):
        apply_context_blend([.2, .8], [.3, .7], [2], [0], weights)


def test_partial_field_is_not_silently_normalized():
    with pytest.raises(ValueError, match="normalize"):
        apply_context_blend([.2, .7], [.3, .7], [2], [1], dict(zip(CONTEXTS, [0, .5])))
