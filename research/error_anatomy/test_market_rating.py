import numpy as np

from research.error_anatomy.market_rating import _rate


def rate(logp, horse, day, starts):
    n = len(horse)
    return _rate(np.asarray(horse, np.int64), np.asarray(day, np.int64), np.asarray(starts, np.int64),
                 np.asarray(logp, float), np.zeros(n, np.bool_), max(horse) + 1, 1., .3, .001)


HORSE = [0, 1, 2, 0, 3, 1, 3, 0, 1]
DAY = [1, 1, 1, 5, 5, 5, 5, 9, 9]
STARTS = [0, 3, 5, 7, 9]  # two races on day 5
LOGP = np.log([.6, .3, .1, .7, .3, .5, .5, .5, .5])


def test_prior_uses_only_earlier_dates():
    mean, var, runs = rate(LOGP, HORSE, DAY, STARTS)
    assert np.all(mean[:3] == 0) and np.all(runs[:3] == 0)
    changed = LOGP.copy()
    changed[3:7] = np.log([.2, .8, .9, .1])  # day 5 prices
    other, _, _ = rate(changed, HORSE, DAY, STARTS)
    assert np.array_equal(mean[:7], other[:7])   # nothing on day 5 or before moves
    assert not np.array_equal(mean[7:], other[7:])


def test_shorter_priced_horse_gets_the_higher_rating_and_ratings_link_through_a_rival():
    mean, _, runs = rate(LOGP, HORSE, DAY, STARTS)
    assert mean[3] > mean[5]            # horse 0 was 0.6, horse 1 was 0.3 on day 1
    assert runs[7] == 2 and runs[8] == 2


def test_ratings_link_through_a_rival():
    # Horse 1 meets horse 2 on day 1, then horse 3 at level prices on day 5; horse 3 is read on day 9.
    def later(x):
        return rate(np.log([x, 1 - x, .5, .5, 1.]), [1, 2, 1, 3, 3], [1, 1, 5, 5, 9], [0, 2, 4, 5])[0][-1]
    assert later(.8) > later(.5) > later(.2)


def test_unpriced_rows_do_not_update():
    logp = LOGP.copy()
    logp[:3] = np.nan
    mean, _, runs = rate(logp, HORSE, DAY, STARTS)
    assert np.all(mean[3:7] == 0) and np.all(runs[3:7] == 0)
