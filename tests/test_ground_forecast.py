"""Idea C's ground preference must describe only a horse's previous runs."""

import numpy as np
import pandas as pd

from scripts.ground_forecast import going_score, ground_preference


def test_going_score_reads_the_primary_description():
    assert going_score("Good to Soft (Soft in places)") == 4.0
    assert going_score("Soft (Heavy in places)") == 5.0
    assert going_score("Good (Good to Firm in places)") == 3.0
    assert going_score("Good to Firm") == 2.0
    assert going_score("Yielding to Soft") == 4.5
    assert np.isnan(going_score(None))


def _race(rid, date, going, horses, finish, odds):
    return pd.DataFrame({"race_id": rid, "race_date": date, "off_time": "14:00", "going": going,
                         "surface": "TURF", "horse_id": horses, "finish_position": finish, "odds": odds})


def test_ground_preference_is_prior_only_and_signed():
    # Horse 1 beats its SP rank on soft, disappoints on firm.
    res = pd.concat([
        _race(1, "2024-01-01", "Soft", [1, 2, 3], [1, 2, 3], [5.0, 2.0, 3.0]),
        _race(2, "2024-02-01", "Good to Firm", [1, 2, 3], [3, 1, 2], [2.0, 5.0, 3.0]),
        _race(3, "2024-03-01", "Heavy", [1, 2, 3], [1, 2, 3], [2.0, 3.0, 5.0]),
    ], ignore_index=True)
    out = ground_preference(res)
    h1 = out[res["horse_id"] == 1]["pref"].to_numpy()
    assert h1[0] == 0.0                 # no history before its first run
    assert h1[1] > 0                    # one soft run, beat its SP rank
    assert h1[2] > h1[1]                # then disappointed on firm
    # The current race never enters: change race 3's result, race 3's pref is unchanged.
    res2 = res.copy()
    res2.loc[(res2["race_id"] == 3) & (res2["horse_id"] == 1), "finish_position"] = 3
    assert ground_preference(res2)["pref"].iloc[6] == out["pref"].iloc[6]
