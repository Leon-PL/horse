import numpy as np
import pandas as pd
import pytest

from src.recent_probability import calendar_split, SmoothWinMap, RecentPlaceMap, project_places


def test_calendar_window_is_bounded_in_days_and_has_no_overlap():
    dates = pd.date_range("2023-01-01", "2026-10-27")
    train, cal = calendar_split(dates)
    assert dates[cal].min() == pd.Timestamp("2026-07-30")
    assert dates[train].max() == pd.Timestamp("2026-07-23")
    assert not (train & cal).any()
    # Many races on one date cannot move the date boundary.
    more = dates.append(pd.DatetimeIndex(["2026-10-20"]*2000))
    a, b = calendar_split(more)
    np.testing.assert_array_equal(a[:len(dates)], train)
    np.testing.assert_array_equal(b[:len(dates)], cal)


def test_smooth_win_map_retains_tail_mass_and_complete_field_totals():
    p = SmoothWinMap(.9, -.2).predict([.001, .2, .4, .3, .01], [5])
    assert p.min() > 1e-6
    assert p.sum() == pytest.approx(1.)
    assert np.argmax(p) == 2
    with pytest.raises(ValueError):
        SmoothWinMap.fit([.2, .3], [1, 1], [2])


def test_vectorised_place_projection_matches_existing_rule():
    from src.each_way import adjust_place_probs_for_race
    raw = np.array([.1, .2, .7, .4, .5, .3, .05])
    win = np.array([.02, .05, .5, .15, .15, .1, .03])
    actual = project_places(raw, win, [7], [2])
    expected = adjust_place_probs_for_race(raw, win, 2)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def test_missing_slots_projection_has_correct_mass_and_win_floors():
    raw = [.1, .8, .2, .4, .7, .3]
    win = [.001, .8, .009, .05, .1, .04]
    p = project_places(raw, win, [6], [3], expected_slots=[2.8])
    assert p.sum() == pytest.approx(2.8)
    assert np.all(p >= win)
    assert np.all(p <= 1)
    with pytest.raises(ValueError):
        project_places(raw, win, [6], [3], expected_slots=[.5])


def test_fitted_place_map_learns_missing_slots_from_prior_results_only():
    groups = np.array([5]*30)
    raw = np.tile([.6, .5, .3, .2, .1], 30)
    win = np.tile([.4, .3, .15, .1, .05], 30)
    positions = np.tile([1, 2, 0, 0, 0], 30)
    places = np.array([3]*30)
    codes = np.array(["Chase"]*30)
    mapping = RecentPlaceMap.fit(raw, win, (positions > 0).astype(int), positions, groups, places, codes)
    assert mapping.expected_slots(["Chase"], [3])[0] < 3
    assert mapping.expected_slots(["Flat"], [3])[0] == 3
    p = mapping.predict(raw, win, groups, places, codes)
    np.testing.assert_allclose(p.reshape(-1,5).sum(axis=1), mapping.expected_slots(codes, places))
