import numpy as np
import pandas as pd
import pytest

from research.error_anatomy.entity_residuals import run_residuals
from research.error_anatomy.franking import SHRINK, franking_inputs


def meetings(days=7, runners=4, seed=1):
    """The same four horses meet every day, two races a day swapping partners."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(days):
        order = rng.permutation(8)
        for race in range(2):
            field = order[race * 4:(race + 1) * 4]
            finish = rng.permutation(runners) + 1
            for k, horse in enumerate(field):
                rows.append({"race_id": f"{day}-{race}", "horse_id": int(horse),
                             "race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=7 * day),
                             "finish_position": int(finish[k]), "won": int(finish[k] == 1), "p_single": .25})
    return pd.DataFrame(rows)


def columns(values):
    return [c for c in values if c not in ("race_id", "horse_id")]


def by_key(values):
    return values.sort_values(["race_id", "horse_id"]).reset_index(drop=True)


def test_inputs_use_only_rival_runs_before_today():
    frame = meetings()
    before = by_key(franking_inputs(frame, frame))
    changed = frame.copy()
    last = changed.race_date.eq(changed.race_date.max()).to_numpy()
    changed.loc[last, "finish_position"] = changed.loc[last].groupby("race_id").finish_position.transform(
        lambda x: x.to_numpy()[::-1])
    changed["won"] = changed.finish_position.eq(1).astype(int)
    after = by_key(franking_inputs(changed, changed))
    pd.testing.assert_frame_equal(before, after)
    first = frame.race_date.eq(frame.race_date.min())
    keys = set(zip(frame.race_id[first], frame.horse_id[first]))
    opening = before[[k in keys for k in zip(before.race_id, before.horse_id)]]
    assert (opening[columns(opening)] == 0).all().all()  # no earlier race, so nothing to frank


def test_one_rival_run_matches_the_definition():
    # Race A: horses 1 (first) and 2 (second). Then 2 runs in race B against 3. Then 1 runs in race C.
    frame = pd.DataFrame({
        "race_id": ["A", "A", "B", "B", "C", "C"], "horse_id": [1, 2, 2, 3, 1, 4],
        "race_date": pd.to_datetime(["2025-01-01"] * 2 + ["2025-01-10"] * 2 + ["2025-01-20"] * 2),
        "finish_position": [1, 2, 1, 2, 2, 1], "won": [1, 0, 1, 0, 0, 1],
        "p_single": [.5, .5, .2, .8, .5, .5]})
    values = franking_inputs(frame, frame).set_index(["race_id", "horse_id"])
    rival = run_residuals(frame).order.iloc[2]  # horse 2's run in race B
    assert values.loc[("C", 1), "fr_last_behind_runs"] == 1
    assert values.loc[("C", 1), "fr_last_behind_order"] == pytest.approx(rival / (1 + SHRINK), rel=1e-5)
    assert values.loc[("C", 1), "fr_last_ahead_runs"] == 0
    # Horse 2 in race B: its rival from race A (horse 1) has not run again before race B.
    assert values.loc[("B", 2), "fr_last_ahead_runs"] == 0
