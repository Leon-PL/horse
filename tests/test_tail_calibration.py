import json

import numpy as np
import pandas as pd
import pytest

from src.tail_calibration import blend, fit_weight, validate_weight
from scripts.benchmark_tail_calibration import stage_frame


def test_blend_is_normalized_exact_identity_and_replays():
    p = np.array([.9, .1, .2, .8])
    q = np.array([.7, .3, .4, .6])
    assert np.array_equal(blend(p, q, 0), p)
    w = json.loads(json.dumps({"weight": .17}))["weight"]
    actual = blend(p, q, w)
    np.testing.assert_allclose(actual.reshape(2, 2).sum(axis=1), 1, atol=1e-15)
    assert np.array_equal(actual, blend(p, q, .17))


def test_brier_constraint_and_convex_minimum_match_grid():
    p = np.array([.8, .2, .7, .3, .6, .4, .9, .1])
    q = np.array([.6, .4, .4, .6, .7, .3, .6, .4])
    y = np.array([1, 0, 0, 1, 1, 0, 1, 0])
    result = fit_weight(p, q, y, [2] * 4)
    a, b = result["brier_quadratic_a"], result["brier_linear_b"]
    for w in (0, .1, .5, 1):
        observed = np.mean((blend(p, q, w) - y) ** 2) - np.mean((p - y) ** 2)
        assert observed == pytest.approx(a * w ** 2 + b * w, abs=1e-15)
    grid = np.linspace(0, result["brier_feasible_upper"], 10001)
    losses = np.array([-np.log(blend(p, q, w)[y == 1]).mean() for w in grid])
    actual = -np.log(blend(p, q, result["weight"])[y == 1]).mean()
    assert actual <= losses.min() + 1e-12
    assert result["adjusted"]["brier"] <= result["original"]["brier"] + 1e-14


def test_equal_or_worse_fit_chooses_identity():
    p = np.array([.8, .2])
    assert fit_weight(p, p, [1, 0], [2])["weight"] == 0
    assert fit_weight(p, [.5, .5], [1, 0], [2])["weight"] == 0


def test_validation_rejects_without_refitting_and_can_accept():
    p, q = [.8, .2], [.6, .4]
    rejected = validate_weight(.2, p, q, [1, 0], [2])
    assert not rejected["accepted"] and rejected["selected_weight"] == 0
    accepted = validate_weight(.2, p, q, [0, 1], [2])
    assert accepted["accepted"] and accepted["selected_weight"] == .2


def test_extreme_probability_is_lifted_without_clipping_other_runners():
    actual = blend([1e-9, 1 - 1e-9], [.004, .996], .1)
    assert actual[0] == pytest.approx(.0004000009)
    assert actual.sum() == pytest.approx(1)
    assert -np.log(actual[0]) < 8 < -np.log(1e-9)


def test_chronology_development_and_whole_field_guards_retain_dnf():
    frame = pd.DataFrame({"race_id": [1, 1, 2, 2],
        "horse_name": ["a", "b", "c", "d"], "race_date": pd.to_datetime(["2026-01-01"] * 2 + ["2026-04-01"] * 2),
        "won": [1, 0, 0, 1], "finish_position": [1, 0, 0, 1], "num_runners": [2] * 4})
    actual = stage_frame(frame, "test", "2025-12-22")
    assert len(actual) == 2 and actual.finish_position.eq(0).sum() == 1
    with pytest.raises(ValueError, match="embargo"):
        stage_frame(frame, "test", "2025-12-30")
    with pytest.raises(ValueError, match="whole fields"):
        stage_frame(frame.iloc[[0]], "test", "2025-12-22")


@pytest.mark.parametrize("weight", [-.1, 1.1, np.nan])
def test_invalid_weights_rejected(weight):
    with pytest.raises(ValueError):
        blend([.5, .5], [.4, .6], weight)


def test_malformed_fields_rejected_before_fitting():
    with pytest.raises(ValueError, match="sum to one"):
        fit_weight([.5, .4], [.5, .5], [1, 0], [2])
    with pytest.raises(ValueError, match="strictly positive"):
        fit_weight([0, 1], [.5, .5], [1, 0], [2])
