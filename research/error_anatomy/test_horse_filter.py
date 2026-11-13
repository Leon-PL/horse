import numpy as np

from research.error_anatomy.horse_filter import _filter


def run(values, days, young=None, horse=None):
    n = len(values)
    horse = np.zeros(n, np.int64) if horse is None else np.asarray(horse, np.int64)
    young = np.zeros(n, np.bool_) if young is None else np.asarray(young, np.bool_)
    return _filter(horse, np.asarray(days, np.int64), np.asarray(values, float), young, 1., 1., .01)


def test_first_run_and_same_day_runs_see_no_observation_of_that_day():
    mean, var = run([2., 4., 6.], [10, 10, 20])
    assert mean[0] == 0 and mean[1] == 0 and var[0] == 1 and var[1] == 1
    # Both runs of day 10 are folded in before day 20: gains 1/2 then 1/3.
    assert np.isclose(mean[2], 2.)
    assert np.isclose(var[2], 1 / 3 + .01 * 10)


def test_a_later_value_cannot_change_an_earlier_row():
    a, _ = run([1., 2., 3., 4.], [1, 5, 9, 30])
    b, _ = run([1., 2., 3., 99.], [1, 5, 9, 30])
    assert np.array_equal(a, b)
    c, _ = run([1., 2., 77., 4.], [1, 5, 9, 30])
    assert np.array_equal(a[:3], c[:3]) and a[3] != c[3]


def test_time_off_widens_uncertainty_faster_for_the_young_and_never_past_the_prior():
    _, old = run([1., 1.], [0, 20])
    _, young = run([1., 1.], [0, 20], young=[True, True])
    _, long = run([1., 1.], [0, 5000])
    assert np.isclose(old[1], .5 + .2) and np.isclose(young[1], .5 + .4) and long[1] == 1.


def test_horses_are_separate_and_missing_values_are_skipped():
    mean, var = run([5., np.nan, 3., 1.], [0, 3, 0, 3], horse=[0, 0, 1, 1])
    assert mean[2] == 0 and var[2] == 1
    assert np.isclose(mean[1], 2.5) and np.isclose(mean[3], 1.5)
