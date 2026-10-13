"""Sibling-form features must not see the present.

Two leakage rules apply here and both are easy to get wrong:

* a horse is not its own sibling, so its own earlier runs must be subtracted
  from the dam's totals; and
* siblings frequently run on the same day, and at prediction time that day's
  earlier results do not exist — so a run on date D may only see sibling form
  through D-1 (the project's date-strict rule, CLAUDE.md hard rule 1).

A leak in either direction would produce a feature that looks strong in
backtest and is unavailable live, which is the failure mode that has cost this
project the most.
"""

import numpy as np
import pandas as pd
import pytest

from src.sibling_features import (
    SIBLING_FEATURES,
    compute_sibling_features,
)


def frame(rows):
    """rows: (race_id, date, dam, horse, won, official_rating)"""
    return pd.DataFrame(
        rows, columns=["race_id", "race_date", "dam", "horse_name", "won", "official_rating"],
    ).assign(race_date=lambda d: pd.to_datetime(d.race_date))


class TestSelfExclusion:
    def test_a_horse_is_not_its_own_sibling(self):
        """One horse, three winning runs, no siblings: must stay at zero."""
        out = compute_sibling_features(frame([
            ("r1", "2025-01-01", "Dam A", "Solo", 1, 80),
            ("r2", "2025-02-01", "Dam A", "Solo", 1, 85),
            ("r3", "2025-03-01", "Dam A", "Solo", 1, 90),
        ]))
        assert out["sibling_runs"].tolist() == [0.0, 0.0, 0.0]
        assert out["sibling_wins"].tolist() == [0.0, 0.0, 0.0]
        assert out["sibling_count"].tolist() == [0.0, 0.0, 0.0]
        assert out["has_sibling_history"].tolist() == [0, 0, 0]
        assert out["sibling_win_rate_shrunk"].isna().all()

    def test_a_sibling_s_prior_runs_are_counted(self):
        out = compute_sibling_features(frame([
            ("r1", "2025-01-01", "Dam A", "Elder", 1, 90),
            ("r2", "2025-02-01", "Dam A", "Elder", 0, 88),
            ("r3", "2025-03-01", "Dam A", "Younger", 0, np.nan),
        ]))
        assert out["sibling_runs"].iloc[2] == 2.0
        assert out["sibling_wins"].iloc[2] == 1.0
        assert out["sibling_count"].iloc[2] == 1.0
        assert out["has_sibling_history"].iloc[2] == 1

    def test_own_record_is_subtracted_when_both_have_run(self):
        """Elder has 2 runs, Younger 1; Younger's second run sees only Elder."""
        out = compute_sibling_features(frame([
            ("r1", "2025-01-01", "Dam A", "Elder", 1, 90),
            ("r2", "2025-02-01", "Dam A", "Younger", 1, 70),
            ("r3", "2025-03-01", "Dam A", "Elder", 0, 92),
            ("r4", "2025-04-01", "Dam A", "Younger", 0, 75),
        ]))
        # Younger's run on 04-01: siblings = Elder's two runs, one of them a win.
        assert out["sibling_runs"].iloc[3] == 2.0
        assert out["sibling_wins"].iloc[3] == 1.0
        assert out["sibling_count"].iloc[3] == 1.0

    def test_best_official_rating_excludes_the_horse_s_own_best(self):
        """The top-two-by-horse bookkeeping exists for exactly this case."""
        out = compute_sibling_features(frame([
            ("r1", "2025-01-01", "Dam A", "Star", 1, 120),   # the good one
            ("r2", "2025-02-01", "Dam A", "Plain", 0, 70),
            ("r3", "2025-03-01", "Dam A", "Star", 1, 125),
        ]))
        # Star's own 125/120 must not be its own sibling_best_or; Plain's 70 is.
        assert out["sibling_best_or"].iloc[2] == 70.0
        # Plain, conversely, does see Star's 120.
        assert out["sibling_best_or"].iloc[1] == 120.0


class TestDateStrictness:
    def test_siblings_running_the_same_day_are_invisible_to_each_other(self):
        """The rule that makes the feature available live."""
        out = compute_sibling_features(frame([
            ("r1", "2025-05-01", "Dam A", "First", 1, 100),
            ("r2", "2025-05-01", "Dam A", "Second", 0, 90),
        ]))
        assert out["sibling_runs"].tolist() == [0.0, 0.0], (
            "a same-day sibling result is not known at prediction time"
        )
        assert out["has_sibling_history"].tolist() == [0, 0]

    def test_the_next_day_does_see_it(self):
        out = compute_sibling_features(frame([
            ("r1", "2025-05-01", "Dam A", "First", 1, 100),
            ("r2", "2025-05-01", "Dam A", "Second", 0, 90),
            ("r3", "2025-05-02", "Dam A", "Third", 0, np.nan),
        ]))
        assert out["sibling_runs"].iloc[2] == 2.0
        assert out["sibling_wins"].iloc[2] == 1.0

    def test_a_future_sibling_run_never_leaks_backwards(self):
        out = compute_sibling_features(frame([
            ("r1", "2025-01-01", "Dam A", "Early", 0, 60),
            ("r2", "2026-01-01", "Dam A", "Late", 1, 130),
        ]))
        assert out["sibling_runs"].iloc[0] == 0.0, "Early must not see Late"
        assert np.isnan(out["sibling_best_or"].iloc[0])

    def test_settled_rows_do_not_change_when_later_races_are_appended(self):
        """The store appends; a settled row's features must be immutable."""
        base = frame([
            ("r1", "2025-01-01", "Dam A", "Elder", 1, 90),
            ("r2", "2025-02-01", "Dam A", "Younger", 0, 70),
        ])
        extended = pd.concat([base, frame([
            ("r3", "2025-03-01", "Dam A", "Third", 1, 110),
        ])], ignore_index=True)
        first = compute_sibling_features(base)
        again = compute_sibling_features(extended).iloc[:2]
        pd.testing.assert_frame_equal(
            first.reset_index(drop=True), again.reset_index(drop=True),
        )


class TestSeparationAndShape:
    def test_different_dams_never_mix(self):
        out = compute_sibling_features(frame([
            ("r1", "2025-01-01", "Dam A", "A1", 1, 100),
            ("r2", "2025-01-01", "Dam B", "B1", 1, 100),
            ("r3", "2025-02-01", "Dam B", "B2", 0, np.nan),
        ]))
        assert out["sibling_runs"].iloc[2] == 1.0
        assert out["sibling_count"].iloc[2] == 1.0

    def test_dam_names_are_normalised_but_unknowns_do_not_group(self):
        out = compute_sibling_features(frame([
            ("r1", "2025-01-01", "  dam a ", "X", 1, 90),
            ("r2", "2025-02-01", "DAM A", "Y", 0, np.nan),
            ("r3", "2025-01-01", "UNKNOWN", "P", 1, 90),
            ("r4", "2025-02-01", "UNKNOWN", "Q", 0, np.nan),
        ]))
        assert out["sibling_runs"].iloc[1] == 1.0, "case/space variants are one dam"
        assert out["sibling_runs"].iloc[3] == 0.0, "'UNKNOWN' must not group horses"

    def test_win_rate_is_shrunk_not_raw(self):
        """One winning sibling run must not read as a 100% producer."""
        out = compute_sibling_features(frame([
            ("r1", "2025-01-01", "Dam A", "Elder", 1, 90),
            ("r2", "2025-02-01", "Dam A", "Younger", 0, np.nan),
        ]))
        assert 0.0 < out["sibling_win_rate_shrunk"].iloc[1] < 0.2

    def test_no_sibling_record_leaves_the_rate_missing_not_average(self):
        """The RTV lesson: do not imply an average producer with a median fill."""
        out = compute_sibling_features(frame([
            ("r1", "2025-01-01", "Dam A", "Only", 0, 80),
        ]))
        assert np.isnan(out["sibling_win_rate_shrunk"].iloc[0])

    def test_missing_dam_column_degrades_to_empties(self):
        df = frame([("r1", "2025-01-01", "Dam A", "X", 1, 90)]).drop(columns=["dam"])
        out = compute_sibling_features(df)
        assert list(out.columns) == list(SIBLING_FEATURES)
        assert out["has_sibling_history"].tolist() == [0]

    def test_output_is_aligned_and_complete(self):
        df = frame([
            ("r1", "2025-01-01", "Dam A", "X", 1, 90),
            ("r1", "2025-01-01", "Dam B", "Y", 0, 85),
        ])
        out = compute_sibling_features(df)
        assert list(out.columns) == list(SIBLING_FEATURES)
        assert out.index.equals(df.index)

    def test_race_relative_variant_is_centred_within_race(self):
        out = compute_sibling_features(frame([
            ("r0", "2025-01-01", "Dam A", "A1", 0, 100),
            ("r0", "2025-01-01", "Dam B", "B1", 0, 60),
            ("r1", "2025-02-01", "Dam A", "A2", 0, np.nan),
            ("r1", "2025-02-01", "Dam B", "B2", 0, np.nan),
        ]))
        pair = out["sibling_best_or_vs_field"].iloc[2:]
        assert pair.sum() == pytest.approx(0.0), "within-race deviations must cancel"
        assert pair.iloc[0] > 0 > pair.iloc[1]
