"""Inactivity is elapsed time, not evidence from a failed/pending appearance."""
from copy import deepcopy

import numpy as np
import pytest

from src import ratings


def _pass(function, state, events):
    keys, dates, finishes, ids = [], [], [], []
    groups = {}
    for race, (day, runners) in enumerate(events):
        indices = []
        for key, position in runners:
            indices.append(len(keys))
            keys.append(key)
            dates.append(day)
            finishes.append(position)
        ids.append(race)
        groups[race] = np.array(indices)
    return function(keys, np.array(finishes, dtype=float), np.array(dates, dtype="datetime64[ns]"),
                    np.array(ids), groups, state=state, return_state=True)


@pytest.fixture(params=["glicko1", "glicko2"])
def sweep(request):
    date = np.datetime64("2025-01-01", "ns")
    if request.param == "glicko1":
        state = {"ratings": {"A": 1600., "B": 1500., "T": 1450.},
                 "rds": {"A": 100., "B": 100., "T": 100.},
                 "last_dt": {k: date for k in ("A", "B", "T")},
                 "counts": {k: 10 for k in ("A", "B", "T")}}
        return ratings._glicko_pass, state
    state = {"mus": {"A": 100./ratings._G2_SCALE, "B": 0., "T": -50./ratings._G2_SCALE},
             "phis": {k: 100./ratings._G2_SCALE for k in ("A", "B", "T")},
             "sigmas": {k: .06 for k in ("A", "B", "T")},
             "last_dt": {k: date for k in ("A", "B", "T")}}
    return ratings._glicko2_pass, state


def test_nonfinish_does_not_count_elapsed_time_twice(sweep):
    function, state = sweep
    direct = _pass(function, state, [("2025-03-01", [("A", 1), ("B", 2)])])
    interrupted = _pass(function, state, [
        ("2025-02-01", [("A", 0)]),
        ("2025-03-01", [("A", 1), ("B", 2)]),
    ])
    # The current policy excludes non-finishers from rating updates. Their
    # mere appearance must therefore not alter the next rating's uncertainty.
    for expected, actual in zip(direct[:-1], interrupted[:-1]):
        np.testing.assert_allclose(expected, actual[-2:], rtol=0, atol=1e-12)
    assert direct[-1] == interrupted[-1]


def test_pending_cards_do_not_modify_carried_state(sweep):
    function, state = sweep
    before = deepcopy(state)
    result = _pass(function, state, [("2025-02-01", [("A", np.nan), ("B", np.nan)])])
    assert state == before
    assert result[-1] == before
    assert result[1][0] > 100.


def test_duplicate_entity_has_identical_prerace_uncertainty(sweep):
    function, state = sweep
    result = _pass(function, state, [("2025-02-01", [("T", 1), ("T", 2), ("B", 3)])])
    assert result[1][0] == result[1][1]
    # Order of a stable's runners cannot change its pre-race features.
    reversed_result = _pass(function, state, [("2025-02-01", [("T", 2), ("T", 1), ("B", 3)])])
    np.testing.assert_allclose(result[1], reversed_result[1], rtol=0, atol=1e-12)


def test_split_after_nonfinish_matches_uninterrupted_pass(sweep):
    function, state = sweep
    events = [("2025-02-01", [("A", 0)]), ("2025-03-01", [("A", 1), ("B", 2)])]
    batch = _pass(function, state, events)
    first = _pass(function, state, events[:1])
    second = _pass(function, first[-1], events[1:])
    for expected, actual in zip(batch[:-1], second[:-1]):
        np.testing.assert_allclose(expected[-2:], actual, rtol=0, atol=1e-12)
    assert batch[-1] == second[-1]
