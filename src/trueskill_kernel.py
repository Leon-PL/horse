"""Compiled TrueSkill sweep for races rated as free-for-alls.

``trueskill.TrueSkill.rate`` builds a Python factor graph per race, which cost
about a minute per 60,000 rows. This module sends the same messages in the same
order and with the same floating-point operations, specialised to teams of one
runner with unit weights, so every rating agrees with the library to the bit.
``tests/test_trueskill_kernel.py`` holds it to that against the library.

Two things look redundant and are not:

* Squares go through ``pow`` with an exponent the compiler cannot see, because
  the library's ``x ** 2`` is ``pow(x, 2.0)``, which is not always ``x * x``.
* Ratings round-trip through (pi, tau) on the way in and out, as ``Rating``
  objects do, because that round trip changes the last bits.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

_SQRT2 = math.sqrt(2)
_PDF_SCALE = 1 / math.sqrt(2 * math.pi)
_INF = math.inf
_NAT = np.iinfo(np.int64).min

# How one race ended.
_RATED = 0
_NOT_RATED = 1  # the library raised FloatingPointError or ValueError
_FAILED = 2  # the library raised ZeroDivisionError, which nothing catches

# Rows of the scratch table: two (pi, tau) rows per variable or message.
_RATING = 0  # variables
_PERF = 2
_TEAM = 4
_DIFF = 6
_LIKE_RATING = 8  # messages: likelihood -> rating
_LIKE_PERF = 10  # likelihood -> performance
_TEAM_PERF = 12  # team sum -> performance
_TEAM_TEAM = 14  # team sum -> team
_DIFF_LEFT = 16  # difference sum -> team on the left
_DIFF_RIGHT = 18  # difference sum -> team on the right
_DIFF_DIFF = 20  # difference sum -> difference
_TRUNC = 22  # truncation -> difference
_ROWS = 24


@nb.njit(error_model="numpy")
def _erfc(x):
    z = abs(x)
    t = 1. / (1. + z / 2.)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (
        0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (
            0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
                -0.82215223 + t * 0.17087277
            )))
        )))
    )))
    return 2. - r if x < 0 else r


@nb.njit(error_model="numpy")
def _cdf(x):
    return 0.5 * _erfc(-x / _SQRT2)


@nb.njit(error_model="numpy")
def _pdf(x, two):
    return _PDF_SCALE * math.exp(-(x ** two / 2.))


@nb.njit(error_model="numpy")
def _v_win(diff, margin, two):
    x = diff - margin
    denom = _cdf(x)
    return _pdf(x, two) / denom if denom != 0.0 else -x


@nb.njit(error_model="numpy")
def _v_draw(diff, margin, two):
    abs_diff = abs(diff)
    a, b = margin - abs_diff, -margin - abs_diff
    denom = _cdf(a) - _cdf(b)
    numer = _pdf(b, two) - _pdf(a, two)
    value = numer / denom if denom != 0.0 else a
    return -value if diff < 0 else value


@nb.njit(error_model="numpy")
def _truncation_win(diff, margin, two):
    """(v, w, valid) for one place decided by a clear margin."""
    x = diff - margin
    v = _v_win(diff, margin, two)
    w = v * (v + x)
    return v, w, 0.0 < w and w < 1.0


@nb.njit(error_model="numpy")
def _truncation_draw(diff, margin, two):
    """(v, w, valid) for a dead heat."""
    v = _v_draw(diff, margin, two)
    abs_diff = abs(diff)
    a, b = margin - abs_diff, -margin - abs_diff
    denom = _cdf(a) - _cdf(b)
    if denom == 0.0:
        return v, 0.0, False
    level = _v_draw(abs_diff, margin, two)
    return v, level ** two + (a * _pdf(a, two) - b * _pdf(b, two)) / denom, True


@nb.njit(error_model="numpy")
def _sum_one(pi, tau):
    """SumFactor.update over one term; (pi, tau, failed)."""
    mu = 0.0 + (tau / pi if pi != 0.0 else pi)
    pi_inv = 0.0 + 1.0 / pi if pi != 0.0 else _INF
    if pi_inv == 0.0:
        return 0.0, 0.0, True
    out = 1. / pi_inv
    return out, out * mu, False


@nb.njit(error_model="numpy")
def _sum_two(pi_a, tau_a, coeff_a, pi_b, tau_b, coeff_b):
    """SumFactor.update over two terms; (pi, tau, failed)."""
    mu = 0.0 + coeff_a * (tau_a / pi_a if pi_a != 0.0 else pi_a)
    pi_inv = 0.0 + 1.0 / pi_a if pi_a != 0.0 else _INF
    mu += coeff_b * (tau_b / pi_b if pi_b != 0.0 else pi_b)
    if pi_inv != _INF:
        pi_inv = pi_inv + 1.0 / pi_b if pi_b != 0.0 else _INF
    if pi_inv == 0.0:
        return 0.0, 0.0, True
    out = 1. / pi_inv
    return out, out * mu, False


@nb.njit(error_model="numpy")
def _send(s, variable, message, i, j, pi, tau):
    """Variable.update_message: replace a stored message and the variable."""
    s[variable, i] = (s[variable, i] - s[message, j]) + pi
    s[variable + 1, i] = (s[variable + 1, i] - s[message + 1, j]) + tau
    s[message, j] = pi
    s[message + 1, j] = tau


@nb.njit(error_model="numpy")
def _rest(s, variable, message, i, j):
    """The variable without one message: (pi, tau) of ``val / msg``."""
    return s[variable, i] - s[message, j], s[variable + 1, i] - s[message + 1, j]


@nb.njit(error_model="numpy")
def _difference_down(s, j):
    left = _rest(s, _TEAM, _DIFF_LEFT, j, j)
    right = _rest(s, _TEAM, _DIFF_RIGHT, j + 1, j)
    pi, tau, failed = _sum_two(left[0], left[1], 1.0, right[0], right[1], -1.0)
    if not failed:
        _send(s, _DIFF, _DIFF_DIFF, j, j, pi, tau)
    return failed


@nb.njit(error_model="numpy")
def _difference_to_left(s, j):
    """SumFactor.up(0): the difference and the right team set the left team."""
    diff = _rest(s, _DIFF, _DIFF_DIFF, j, j)
    right = _rest(s, _TEAM, _DIFF_RIGHT, j + 1, j)
    pi, tau, failed = _sum_two(diff[0], diff[1], 1.0, right[0], right[1], 1.0)
    if not failed:
        _send(s, _TEAM, _DIFF_LEFT, j, j, pi, tau)
    return failed


@nb.njit(error_model="numpy")
def _difference_to_right(s, j):
    """SumFactor.up(1): the left team and the difference set the right team."""
    left = _rest(s, _TEAM, _DIFF_LEFT, j, j)
    diff = _rest(s, _DIFF, _DIFF_DIFF, j, j)
    pi, tau, failed = _sum_two(left[0], left[1], 1.0, diff[0], diff[1], -1.0)
    if not failed:
        _send(s, _TEAM, _DIFF_RIGHT, j + 1, j, pi, tau)
    return failed


@nb.njit(error_model="numpy")
def _truncate(s, j, tie, margin, two):
    """TruncateFactor.up; (delta, outcome)."""
    div_pi, div_tau = _rest(s, _DIFF, _TRUNC, j, j)
    if div_pi < 0.0:
        return 0.0, _NOT_RATED
    sqrt_pi = math.sqrt(div_pi)
    diff, scaled = div_tau / sqrt_pi, margin * sqrt_pi
    if tie:
        v, w, valid = _truncation_draw(diff, scaled, two)
    else:
        v, w, valid = _truncation_win(diff, scaled, two)
    if not valid:
        return 0.0, _NOT_RATED
    denom = 1. - w
    pi, tau = div_pi / denom, (div_tau + sqrt_pi * v) / denom
    var_pi, var_tau = s[_DIFF, j], s[_DIFF + 1, j]
    s[_TRUNC, j] = pi + s[_TRUNC, j] - var_pi
    s[_TRUNC + 1, j] = tau + s[_TRUNC + 1, j] - var_tau
    s[_DIFF, j], s[_DIFF + 1, j] = pi, tau
    pi_delta = abs(var_pi - pi)
    if pi_delta == _INF:
        return 0.0, _RATED
    delta, root = abs(var_tau - tau), math.sqrt(pi_delta)
    return (root if root > delta else delta), _RATED


@nb.njit(error_model="numpy")
def _rate(s, order, position, count, ratings, beta2, tau2, margin, min_delta, two, minus_two):
    """Rate one race in place: ``ratings[:, count]`` becomes the new ratings.

    ``order`` lists runners by finishing position and ``position`` holds their
    ranks in that order. Returns how the race ended; on anything but
    ``_RATED`` the ratings are only partly written and must be ignored.
    """
    diffs = count - 1
    s[:, :count] = 0.0

    # Gray arrows: prior, performance, team.
    for x in range(count):
        runner = order[x]
        pi = ratings[1, runner] ** minus_two
        tau = pi * ratings[0, runner]
        mu = tau / pi if pi != 0.0 else pi
        sigma = math.sqrt(1 / pi) if pi != 0.0 else _INF
        sigma = math.sqrt(sigma ** two + tau2)
        pi = sigma ** minus_two
        s[_RATING, x], s[_RATING + 1, x] = pi, pi * mu

        a = 1. / (1. + beta2 * s[_RATING, x])
        _send(s, _PERF, _LIKE_PERF, x, x, a * s[_RATING, x], a * s[_RATING + 1, x])
        pi, tau, failed = _sum_one(*_rest(s, _PERF, _TEAM_PERF, x, x))
        if failed:
            return _FAILED
        _send(s, _TEAM, _TEAM_TEAM, x, x, pi, tau)

    # Arrows 1, 2 and 3 until the update is small.
    for _ in range(10):
        if diffs == 1:
            if _difference_down(s, 0):
                return _FAILED
            delta, outcome = _truncate(s, 0, position[0] == position[1], margin, two)
            if outcome != _RATED:
                return outcome
        else:
            delta = 0.0
            for x in range(diffs - 1):
                if _difference_down(s, x):
                    return _FAILED
                moved, outcome = _truncate(s, x, position[x] == position[x + 1], margin, two)
                if outcome != _RATED:
                    return outcome
                if moved > delta:
                    delta = moved
                if _difference_to_right(s, x):
                    return _FAILED
            for x in range(diffs - 1, 0, -1):
                if _difference_down(s, x):
                    return _FAILED
                moved, outcome = _truncate(s, x, position[x] == position[x + 1], margin, two)
                if outcome != _RATED:
                    return outcome
                if moved > delta:
                    delta = moved
                if _difference_to_left(s, x):
                    return _FAILED
        if delta <= min_delta:
            break

    if _difference_to_left(s, 0) or _difference_to_right(s, diffs - 1):
        return _FAILED

    # The remaining arrows back up to the ratings.
    for x in range(count):
        pi, tau, failed = _sum_one(*_rest(s, _TEAM, _TEAM_TEAM, x, x))
        if failed:
            return _FAILED
        _send(s, _PERF, _TEAM_PERF, x, x, pi, tau)
    for x in range(count):
        msg_pi, msg_tau = _rest(s, _PERF, _LIKE_PERF, x, x)
        a = 1. / (1. + beta2 * msg_pi)
        _send(s, _RATING, _LIKE_RATING, x, x, a * msg_pi, a * msg_tau)

    for x in range(count):
        pi, tau = s[_RATING, x], s[_RATING + 1, x]
        if pi < 0.0:
            return _NOT_RATED
        mu = tau / pi if pi != 0.0 else pi
        sigma = math.sqrt(1 / pi) if pi != 0.0 else _INF
        if sigma == 0.0:
            return _NOT_RATED
        pi = sigma ** minus_two
        tau = pi * mu
        runner = order[x]
        ratings[0, runner] = tau / pi if pi != 0.0 else pi
        ratings[1, runner] = math.sqrt(1 / pi) if pi != 0.0 else _INF
    return _RATED


@nb.njit(cache=True, error_model="numpy")
def rate_races(
    entity, finish, score, dates, rows, offsets, nanos_per_day,
    state_mu, state_sigma, rated, last_seen,
    prior_mu, prior_sigma, idle_c2, beta2, tau2, margin, min_delta, dnf_as_loss,
    mu_col, sigma_col, has_col, two, minus_two,
):
    """Sweep races in order, filling the pre-race columns and the state arrays.

    ``entity`` is -1 for rows with no key, and ``rows[offsets[r]:offsets[r+1]]``
    are the rows of race ``r``. With ``score`` false only the pre-race columns
    are filled. Returns the first race the library could not have rated without
    raising an uncaught error, or -1.
    """
    size = 0
    for race in range(len(offsets) - 1):
        size = max(size, offsets[race + 1] - offsets[race])
    s = np.zeros((_ROWS, size))
    ratings = np.empty((2, size))
    runners = np.empty(size, dtype=np.int64)
    order = np.empty(size, dtype=np.int64)
    position = np.empty(size, dtype=np.int64)
    ranked = np.empty(size, dtype=np.int64)

    for race in range(len(offsets) - 1):
        start, stop = offsets[race], offsets[race + 1]
        race_date = dates[rows[start]]
        for at in range(start, stop):
            row = rows[at]
            who = entity[row]
            if who < 0:
                continue
            if rated[who]:
                previous = last_seen[who]
                if race_date == _NAT or previous == _NAT:
                    days = math.nan
                else:
                    days = float(race_date - previous) / nanos_per_day
                    if 0.0 > days:
                        days = 0.0
                months = days / 30.44
                sigma = math.sqrt(state_sigma[who] ** two + idle_c2 * months)
                if prior_sigma < sigma:
                    sigma = prior_sigma
                mu_col[row] = state_mu[who]
                sigma_col[row] = sigma
                has_col[row] = 1
            else:
                mu_col[row] = prior_mu
                sigma_col[row] = prior_sigma

        if not score:
            continue

        last_finisher, finishers, unplaced = 0.0, 0, 0
        for at in range(start, stop):
            place = finish[rows[at]]
            if math.isfinite(place):
                if place > 0:
                    if finishers == 0 or place > last_finisher:
                        last_finisher = place
                    finishers += 1
                elif place == 0:
                    unplaced += 1
        unplaced_rank = last_finisher + 1.0
        level_last = dnf_as_loss and finishers > 0 and unplaced > 0

        count = 0
        for at in range(start, stop):
            row = rows[at]
            place = finish[row]
            if entity[row] < 0 or not math.isfinite(place):
                continue
            if place > 0:
                ranked[count] = int(place)
            elif place == 0 and level_last:
                ranked[count] = int(unplaced_rank)
            else:
                continue
            runners[count] = row
            ratings[0, count] = mu_col[row]
            ratings[1, count] = sigma_col[row]
            count += 1
        if count < 2:
            continue

        for x in range(count):  # stable, as the library's sort by rank is
            at = x
            while at > 0 and ranked[order[at - 1]] > ranked[x]:
                order[at] = order[at - 1]
                at -= 1
            order[at] = x
        for x in range(count):
            position[x] = ranked[order[x]]

        outcome = _rate(s, order, position, count, ratings, beta2, tau2, margin, min_delta, two, minus_two)
        if outcome == _FAILED:
            return race
        if outcome == _NOT_RATED:
            continue
        for x in range(count):
            who = entity[runners[x]]
            state_mu[who] = ratings[0, x]
            state_sigma[who] = ratings[1, x]
            rated[who] = True
            last_seen[who] = race_date
    return -1
