from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pandas as pd
import pytest

from src import feature_engineer as fe
from src.feature_context import counts_state_scope, current_context, current_counts_state


def test_nested_execution_restores_outer_state_and_timeline_cache():
    frame = pd.DataFrame({"horse_id": [1, 1], "race_id": [1, 2],
                          "race_date": pd.to_datetime(["2025-01-01", "2025-01-02"])})
    state = {"outer": {}}
    with counts_state_scope(state) as outer:
        cached = fe._timeline(frame, ["horse_id"], "race_date")
        with pytest.raises(ValueError):
            with counts_state_scope({}) as inner:
                assert inner.timelines is not outer.timelines
                assert fe._timeline(frame, ["horse_id"], "race_date") is not cached
                raise ValueError("injected failure")
        assert current_counts_state() is state
        assert fe._timeline(frame, ["horse_id"], "race_date") is cached
    assert current_context() is None


def test_parallel_execution_owns_its_accumulators_and_timeline_cache():
    barrier = Barrier(2)
    def run(value):
        state = {}
        frame = pd.DataFrame({"horse_id": [1, 1], "race_id": [1, 2],
                              "race_date": pd.to_datetime(["2025-01-01", "2025-01-02"])})
        with counts_state_scope(state) as context:
            barrier.wait(timeout=20)
            assert current_counts_state() is state
            result = fe._cumulative(np.array(["A", "A"]), np.array([1, 2]),
                                    {"m": np.array([value, value], dtype=float)})
            line = fe._timeline(frame, ["horse_id"], "race_date")
            barrier.wait(timeout=20)
            assert current_context() is context
            assert fe._timeline(frame, ["horse_id"], "race_date") is line
            return result["m"], state["m"]["A"], context.timelines
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(run, [1., 3.]))
    np.testing.assert_array_equal(a[0], [0., 1.])
    np.testing.assert_array_equal(b[0], [0., 3.])
    assert a[1] == 2. and b[1] == 6.
    assert a[2] is not b[2]
    assert current_context() is None
