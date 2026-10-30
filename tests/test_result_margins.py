import numpy as np
import pandas as pd

from src.result_margins import cumulative_beaten_lengths


def frame(positions, gaps):
    return pd.DataFrame({"race_id": 1, "finish_position": positions, "lengths_behind": gaps})


def test_source_yarmouth_result_accumulates_adjacent_gaps():
    f = frame([1, 2, 3, 4, 5], [0, .3, .2, .75, 1.5])
    np.testing.assert_allclose(cumulative_beaten_lengths(f), [0, .3, .5, 1.25, 2.75])
    assert f.lengths_behind.iloc[2] == .2


def test_unknown_gap_does_not_make_downstream_horses_close_finishers():
    f = frame([1, 2, 3, 4, 0], [np.nan, 2, np.nan, .2, 0])
    np.testing.assert_allclose(cumulative_beaten_lengths(f), [0, 2, np.nan, np.nan, np.nan], equal_nan=True)


def test_missing_finishers_and_negative_gaps_invalidate_the_chain():
    np.testing.assert_allclose(cumulative_beaten_lengths(frame([1, 3], [0, 2])), [0, np.nan], equal_nan=True)
    np.testing.assert_allclose(cumulative_beaten_lengths(frame([1, 2, 3], [0, -1, 2])), [0, np.nan, np.nan], equal_nan=True)


def test_dead_heats_are_equal_and_permutation_invariant():
    f = frame([1, 2, 2, 4], [0, 2, 0, .5])
    expected = cumulative_beaten_lengths(f)
    np.testing.assert_allclose(expected, [0, 2, 2, 2.5])
    pd.testing.assert_series_equal(cumulative_beaten_lengths(f.sample(frac=1, random_state=4)).sort_index(), expected)
    np.testing.assert_allclose(cumulative_beaten_lengths(frame([1, 1, 3], [0, 0, .5])), [0, 0, .5])


def test_conflicting_dead_heat_and_cards_are_unknown():
    np.testing.assert_allclose(cumulative_beaten_lengths(frame([1, 2, 2, 4], [0, 2, 1, .5])), [0, np.nan, np.nan, np.nan], equal_nan=True)
    assert cumulative_beaten_lengths(frame([0, 0], [np.nan, np.nan])).isna().all()


def test_races_do_not_share_history_and_nonunique_indices_are_preserved():
    f = pd.concat([frame([1, 2], [0, 3]), frame([1, 2], [0, .5]).assign(race_id=2)])
    out = cumulative_beaten_lengths(f)
    np.testing.assert_allclose(out, [0, 3, 0, .5])
    assert out.index.equals(f.index)
