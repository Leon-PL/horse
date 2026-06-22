"""Tests for the data-quality betting gates (src/quality_control.py)."""

import pandas as pd

from src.quality_control import assess_quality, apply_betting_gate


def _card():
    # Race A: a short-priced thin-history raider (horse A1, SP 4.0, 0 prior runs)
    #         plus a normal runner. Race B: all well-exposed runners.
    return pd.DataFrame({
        "race_id": ["A", "A", "B", "B"],
        "horse_name": ["A1", "A2", "B1", "B2"],
        "odds": [4.0, 6.0, 3.0, 5.0],
        "horse_prev_races": [0, 12, 20, 8],
        "is_value": [True, True, True, True],
        "is_ew_value": [False, True, True, False],
    })


def test_thin_history_flags_short_priced_no_form():
    qc = assess_quality(_card())
    flagged = qc["runner_flagged"]
    assert flagged.tolist() == [True, False, False, False]
    assert "Thin-history" in qc["reason"].iloc[0]
    assert qc["reason"].iloc[1] == ""


def test_block_is_race_level():
    qc = assess_quality(_card())
    # The whole of race A is blocked (it contains the thin raider); B is clean.
    assert qc["race_blocked"].tolist() == [True, True, False, False]
    assert qc["blocked_race_ids"] == {"A"}
    assert qc["summary"]["n_races_blocked"] == 1
    assert qc["summary"]["n_runners_flagged"] == 1


def test_apply_gate_suppresses_blocked_race_bets():
    df = _card()
    apply_betting_gate(df, ["is_value", "is_ew_value"])
    # Race A bets suppressed (both runners), race B untouched.
    assert df["is_value"].tolist() == [False, False, True, True]
    assert df["is_ew_value"].tolist() == [False, False, True, False]


def test_threshold_tunes_what_counts_as_contender():
    df = _card()
    # Tighten the contender band below A1's SP -> it's no longer a contender.
    qc = assess_quality(df, {"thin_history_contender_max_odds": 3.0})
    assert qc["runner_flagged"].sum() == 0


def test_disabled_passes_everything():
    qc = assess_quality(_card(), {"enabled": False})
    assert qc["runner_flagged"].sum() == 0
    assert qc["race_blocked"].sum() == 0


def test_missing_columns_degrade_gracefully():
    df = pd.DataFrame({"race_id": ["A"], "horse_name": ["x"]})  # no odds/prev_races
    qc = assess_quality(df)
    assert qc["runner_flagged"].sum() == 0
