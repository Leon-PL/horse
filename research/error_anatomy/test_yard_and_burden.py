import numpy as np

from research.error_anatomy.burden import _history


def test_weight_history_reads_earlier_dates_only():
    horse = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    day = np.array([1, 5, 5, 9, 9], dtype=np.int64)
    weight = np.array([150., 160., 140., 155., 150.])
    won = np.array([1., 0., 1., 0., 0.])
    out = _history(horse, day, weight, won)
    assert np.isnan(out[0]).all() and np.isnan(out[4]).all()          # no earlier run
    # Both runs on day 5 see only day 1: 150 carried, and won under.
    assert out[1].tolist() == [10., 10., 10., 0.]
    assert out[2].tolist() == [-10., -10., -10., 1.]
    # Day 9 sees 150, 160, 140: mean 150, heaviest win 150, heaviest 160, one run at or above 155.
    assert out[3].tolist() == [5., 5., -5., 1.]
