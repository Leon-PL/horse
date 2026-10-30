import pandas as pd

from scripts.collect_missing_history_results import cohort


def test_jumps_and_cheltenham_cohorts_use_only_pre_race_experience():
    frame = pd.DataFrame({"horse_id": [1, 2, 3, 4, 5], "horse_name": list("ABCDE"),
        "race_date": pd.to_datetime(["2025-03-01"]*5),
        "race_type": ["Chase", "Hurdle", "Flat", "Chase", "Hurdle"],
        "track": ["Cheltenham", "Cheltenham", "Ascot", "Cheltenham", "Cheltenham"],
        "horse_prev_races": [0, 0, 0, 0, 3], "days_since_last_run": [24, 0, 15, 0, 20],
        "prior_form_runs": [1, 4, 3, 0, 6]})
    assert [r["horse_id"] for r in cohort(frame, "cheltenham")] == ["1", "2"]
    assert cohort(frame, "jumps") == cohort(frame.assign(won=1, finish_position=1, odds=1.1), "jumps")


def test_membership_uses_first_eligible_date_and_excludes_reserved_periods():
    frame = pd.DataFrame({"horse_id": [1, 1, 2], "horse_name": ["A", "A", "B"],
        "race_date": pd.to_datetime(["2025-10-01", "2025-03-01", "2026-10-01"]),
        "race_type": "Hurdle", "track": "Cheltenham", "horse_prev_races": 0,
        "days_since_last_run": 20, "prior_form_runs": 3})
    result = cohort(frame, "cheltenham")
    assert len(result) == 1
    assert result[0]["first_local_date"] == "2025-03-01"


def test_top_jumps_scope_covers_other_major_venues_without_inferring_country():
    frame = pd.DataFrame({"horse_id": [1, 2, 3], "horse_name": ["A", "B", "C"],
        "race_date": pd.to_datetime(["2025-03-01"]*3), "race_type": ["Chase", "Hurdle", "Flat"],
        "track": ["Kempton", "Cheltenham", "Ascot"], "race_class": ["Class 1", "Class 2", "Class 1"],
        "horse_prev_races": 0, "days_since_last_run": 20, "prior_form_runs": 4})
    result = cohort(frame, "jumps_class1")
    assert [r["horse_id"] for r in result] == ["1"]
