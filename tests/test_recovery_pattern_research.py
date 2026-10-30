from copy import deepcopy
import math

import numpy as np
import pandas as pd
import pytest

from src.recovery_pattern_research import FEATURES, recovery_pattern_features


def history(dates):
    rows = []
    for i, date in enumerate(dates):
        for horse, identity, pos in [("Routine", 1, 1), ("Rival", 2, 2)]:
            rows.append({"race_id": str(i), "race_date": pd.Timestamp(date), "horse_id": identity,
                "horse_name": horse, "finish_position": pos, "won": int(pos == 1),
                "num_runners": 2, "finish_pos_label": ""})
    return pd.DataFrame(rows)


def test_own_routine_and_unusually_longer_gap_have_correct_direction():
    frame = history(["2025-01-01", "2025-01-11", "2025-01-21", "2025-01-31", "2025-02-10", "2025-03-02"])
    out = recovery_pattern_features(frame)
    assert out.iloc[:8][list(FEATURES[:2])].isna().all().all()
    np.testing.assert_allclose(out.iloc[8:10][FEATURES[0]], 0.)
    np.testing.assert_allclose(out.iloc[8:10][FEATURES[1]], math.log(11))
    np.testing.assert_allclose(out.iloc[10:12][FEATURES[0]], math.log(21) - math.log(11))
    assert out.iloc[8][FEATURES[2]] == 3


def test_only_last_five_intervals_are_retained():
    frame = history(pd.date_range("2025-01-01", periods=10, freq="10D"))
    out, seed = recovery_pattern_features(frame, return_state=True)
    assert out.iloc[-1][FEATURES[2]] == 5
    assert all(len(v["intervals"]) == 5 for v in seed["history"].values())


def test_old_intervals_expire_and_do_not_invent_a_stable_routine():
    frame = history(["2021-01-01", "2021-01-11", "2021-01-21", "2021-01-31", "2025-01-01"])
    out = recovery_pattern_features(frame)
    assert out.iloc[-2:][FEATURES[2]].eq(0).all()
    assert out.iloc[-2:][list(FEATURES[:2])].isna().all().all()


def test_same_day_and_future_results_cannot_change_earlier_values():
    frame = history(pd.date_range("2025-01-01", periods=7, freq="10D"))
    out = recovery_pattern_features(frame)
    changed = frame.copy()
    late = changed.race_date.ge("2025-02-10")
    changed.loc[late, "finish_position"] = 0
    changed.loc[late, "won"] = 0
    pd.testing.assert_frame_equal(recovery_pattern_features(changed).loc[~late], out.loc[~late], check_exact=True)
    # Removing the present day's outcomes does not affect its own descriptors.
    pd.testing.assert_frame_equal(recovery_pattern_features(changed).loc[changed.race_date.eq("2025-02-10")],
                                  out.loc[frame.race_date.eq("2025-02-10")], check_exact=True)


def test_prefix_append_permutation_and_pending_state_are_exact():
    frame = history(pd.date_range("2025-01-01", periods=8, freq="10D"))
    full = recovery_pattern_features(frame)
    prefix, seed = recovery_pattern_features(frame.iloc[:8], return_state=True)
    before = deepcopy(seed)
    pd.testing.assert_frame_equal(prefix, full.iloc[:8], check_exact=True)
    pd.testing.assert_frame_equal(recovery_pattern_features(frame.iloc[8:], state=seed), full.iloc[8:], check_exact=True)
    pd.testing.assert_frame_equal(recovery_pattern_features(frame.sample(frac=1, random_state=3)).sort_index(), full, check_exact=True)
    card = frame.iloc[8:10].assign(finish_position=0, won=0)
    _, returned = recovery_pattern_features(card, state=seed, return_state=True)
    assert seed == before == returned
    with pytest.raises(ValueError, match="precede"):
        recovery_pattern_features(frame.iloc[:2], state=seed)


def test_nonfinish_is_a_real_start_but_nonstarter_is_not():
    frame = history(["2025-01-01", "2025-01-11", "2025-01-21", "2025-01-31", "2025-02-10"])
    frame.loc[2, ["finish_position", "won", "finish_pos_label"]] = [0, 0, "F"]
    frame.loc[3, ["finish_position", "won", "finish_pos_label"]] = [0, 0, "NR"]
    out = recovery_pattern_features(frame)
    assert out.iloc[8][FEATURES[2]] == 3
    assert out.iloc[9][FEATURES[2]] == 2
    assert np.isnan(out.iloc[9][FEATURES[0]])


def test_same_day_duplicate_actual_start_is_not_arbitrarily_ordered():
    frame = history(["2025-01-01"])
    duplicate = pd.concat([frame, frame.iloc[[0]].assign(race_id="another")], ignore_index=True)
    with pytest.raises(ValueError, match="ordering"):
        recovery_pattern_features(duplicate)


def test_numeric_ids_preserve_history_across_changed_source_spelling():
    frame = history(pd.date_range("2025-01-01", periods=5, freq="10D"))
    expected = recovery_pattern_features(frame)
    changed = frame.copy()
    changed.loc[8, "horse_name"] = "ROUTINE (IRE)"
    changed["horse_id"] = changed.horse_id.astype(str)
    pd.testing.assert_frame_equal(recovery_pattern_features(changed), expected, check_exact=True)
