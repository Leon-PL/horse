import numpy as np
import pandas as pd

from research.trainer_moves_098.inputs import build


def _frame(rows):
    return pd.DataFrame(rows, columns=["race_id", "horse_id", "race_date", "trainer", "trainer_elo", "won", "odds"])


def test_move_licence_change_and_runs_in_yard():
    f = _frame([
        (1, 7, "2024-01-01", "A Smith", 1500.0, 0, 3.0),
        (2, 7, "2024-02-01", "B Jones", 1600.0, 0, 3.0),
        (3, 7, "2024-03-01", "B Jones", 1610.0, 0, 3.0),
        (4, 7, "2024-04-01", "B & C Jones", 1620.0, 0, 3.0),  # licence change, same yard
        (5, 8, "2024-01-01", "A Smith", 1500.0, 1, 3.0),
    ])
    out = build(f).set_index("race_id")
    assert np.isnan(out.loc[1, "tm_elo_change"]) and out.loc[1, "tm_runs_in_yard"] == 0
    assert out.loc[2, "tm_elo_change"] == 100 and out.loc[2, "tm_runs_in_yard"] == 0
    assert out.loc[3, "tm_elo_change"] == 110 and out.loc[3, "tm_runs_in_yard"] == 1
    assert out.loc[4, "tm_elo_change"] == 120 and out.loc[4, "tm_runs_in_yard"] == 2
    assert np.isnan(out.loc[5, "tm_elo_change"])


def test_arrival_edge_uses_only_earlier_dates():
    # Horse 1 arrives at "New" and wins at 4.0 in a two-runner race; a same-day runner must not see it.
    f = _frame([
        (1, 1, "2024-01-01", "Old", 1500.0, 0, 2.0),
        (1, 2, "2024-01-01", "X", 1500.0, 1, 2.0),
        (2, 1, "2024-02-01", "New", 1600.0, 1, 2.0),
        (2, 3, "2024-02-01", "Y", 1500.0, 0, 2.0),
        (3, 4, "2024-02-01", "New", 1600.0, 0, 2.0),
        (3, 5, "2024-02-01", "Z", 1500.0, 1, 2.0),
        (4, 6, "2024-03-01", "New", 1600.0, 0, 2.0),
        (4, 7, "2024-03-01", "Z", 1500.0, 1, 2.0),
    ])
    out = build(f).set_index(["race_id", "horse_id"])
    assert out.loc[(2, 1), "tm_arrivals_n"] == 0 and out.loc[(3, 4), "tm_arrivals_n"] == 0
    assert out.loc[(4, 6), "tm_arrivals_n"] == 1
    assert np.isclose(out.loc[(4, 6), "tm_arrival_edge"], 0.5 / 31)
