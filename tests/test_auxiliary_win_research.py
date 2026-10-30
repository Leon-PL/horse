import numpy as np
import pytest

from src.auxiliary_win_research import auxiliary_win_probabilities, fit_auxiliary_win, place_utility_book


def inputs():
    win = np.array([.5, .3, .2, .4, .3, .2, .1])
    raw_place = np.array([.6, .8, .3, .5, .8, .6, .2])
    groups = np.array([3, 4])
    won = np.array([0, 1, 0, 0, 1, 0, 0])
    return win, raw_place, groups, won


def test_identity_replays_existing_win_forecasts_exactly():
    win, place, groups, _ = inputs()
    np.testing.assert_array_equal(auxiliary_win_probabilities(win, place, groups), win)


def test_place_scores_do_not_contain_win_floors_or_depend_on_win_values():
    win, place, groups, _ = inputs()
    other = np.array([.2, .5, .3, .1, .2, .3, .4])
    first = place_utility_book(win, place, groups)
    np.testing.assert_array_equal(first, place_utility_book(other, place, groups))
    assert first[1] > first[0] > first[2]
    np.testing.assert_allclose([first[:3].sum(), first[3:].sum()], 1., atol=1e-15)


def test_positive_place_weight_moves_forecast_towards_its_ranking():
    win, place, groups, _ = inputs()
    corrected = auxiliary_win_probabilities(win, place, groups, alpha=.4)
    assert corrected[1] > win[1]
    assert corrected[4] > win[4]
    np.testing.assert_allclose([corrected[:3].sum(), corrected[3:].sum()], 1., atol=1e-14)


def test_extreme_raw_place_scores_stay_finite_and_constant_scores_are_neutral():
    win, place, groups, _ = inputs()
    place[0], place[1] = 0., 1.
    corrected = auxiliary_win_probabilities(win, place, groups, alpha=.3)
    assert np.isfinite(corrected).all() and (corrected > 0).all()
    np.testing.assert_allclose(auxiliary_win_probabilities(win, np.full(len(win), .5), groups, alpha=.3), win, atol=1e-15)


def test_temperature_control_cannot_use_place_and_fits_stay_bounded():
    win, place, groups, won = inputs()
    control = fit_auxiliary_win(win, place, groups, won, allow_place=False)
    joint = fit_auxiliary_win(win, place, groups, won)
    assert control["alpha"] == 0
    for fitted in (control, joint):
        assert fitted["accepted_fit"] and fitted["converged"]
        assert 0 <= fitted["alpha"] <= 1 and .5 <= fitted["beta"] <= 1.5
        assert fitted["objective"] <= fitted["baseline_objective"] + 1e-10


@pytest.mark.parametrize("value", [np.nan, np.inf, -.1, 1.1])
def test_invalid_raw_place_does_not_become_a_forecast(value):
    win, place, groups, _ = inputs()
    place[0] = value
    with pytest.raises(ValueError):
        place_utility_book(win, place, groups)


def test_partial_fields_and_fractional_group_sizes_are_rejected():
    win, place, groups, _ = inputs()
    with pytest.raises(ValueError):
        place_utility_book(win, place, [2.5, 4.5])
    with pytest.raises(ValueError):
        place_utility_book(win[:-1], place[:-1], groups)
