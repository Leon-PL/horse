import numpy as np
import pandas as pd
import pytest

from src.forecast_benchmark import corrected_probabilities, fit_corrections, match_market_fields
from scripts.benchmark_forecasts import gap_segments


def fields():
    frame = pd.DataFrame({"race_id": ["r"] * 3, "race_date": ["2026-07-01"] * 3,
                          "track": ["Newcastle"] * 3, "off_time": ["14:00"] * 3,
                          "horse_name": ["Horse One", "Horse Two", "Horse Three"],
                          "won": [1, 0, 0], "finish_position": [1, 2, 0], "num_runners": [3] * 3})
    source = pd.DataFrame({"race_date": ["2026-07-01"] * 3, "track_key": ["newcastle"] * 3,
                           "off_key": ["14:00"] * 3, "horse_key": ["horse one", "horse two", "horse three"],
                           "market_id": ["m"] * 3, "selection_id": [1, 2, 3],
                           "market_probability": [.5, .3, .2], "active_runners": [3] * 3,
                           "decision_ms": [1000] * 3, "observed_ms": [900] * 3,
                           "price_changed_ms": [800] * 3, "won": [1, 0, 0],
                           "withdrawn_after_decision": [False] * 3})
    return frame, source


def test_identical_field_aligns_by_identity_and_preserves_dnf():
    frame, source = fields()
    paired, report = match_market_fields(frame.sample(frac=1, random_state=8), source)
    assert report["paired_races"] == 1 and len(paired) == 3
    assert paired.loc[paired.horse_name.eq("Horse Three"), "finish_position"].item() == 0
    assert paired.loc[paired.horse_name.eq("Horse One"), "market_probability"].item() == .5


@pytest.mark.parametrize("problem", ["missing_runner", "late_withdrawal", "future_price",
                                     "duplicate_key", "wrong_winner", "different_field"])
def test_bad_overlap_excludes_whole_race(problem):
    frame, source = fields()
    if problem == "missing_runner":
        source = source.iloc[:2]
    elif problem == "late_withdrawal":
        source.loc[2, "withdrawn_after_decision"] = True
    elif problem == "future_price":
        source.loc[2, "price_changed_ms"] = 1001
    elif problem == "duplicate_key":
        source.loc[2, "horse_key"] = "horse one"
    elif problem == "wrong_winner":
        source["won"] = [0, 1, 0]
    elif problem == "different_field":
        source.loc[2, "horse_key"] = "a different horse"
    paired, _ = match_market_fields(frame, source)
    assert paired.empty


def test_feature_join_collision_is_rejected_instead_of_picking_a_race():
    frame, source = fields()
    collision = frame.assign(race_id="another")
    paired, report = match_market_fields(pd.concat([frame, collision], ignore_index=True), source)
    assert paired.empty and report["ambiguous_feature_key_races"] == 2


def calibration_frame(uniform=False):
    rows = []
    for day in range(40):
        for race in range(10):
            for runner in range(2):
                winner = runner == race % 2
                rows.append({"race_id": f"{day}-{race}", "horse_name": str(runner),
                             "race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day),
                             "won": int(winner), "model_probability": .5 if uniform else (.8 if winner else .2),
                             "market_probability": .5})
    return pd.DataFrame(rows)


def test_correction_identity_fallback_is_exact_market():
    frame = calibration_frame(uniform=True)
    recipe = fit_corrections(frame)
    assert recipe["temperature"] == 1
    assert recipe["anchor"] == {"alpha": 0, "beta": 1}
    np.testing.assert_array_equal(corrected_probabilities(frame, recipe)["market_combination"], frame.market_probability)


def test_correction_selection_is_chronological_bounded_and_row_order_invariant():
    frame = calibration_frame()
    a, b = fit_corrections(frame), fit_corrections(frame.sample(frac=1, random_state=7))
    assert a == b
    assert pd.Timestamp(a["fit_last_date"]) <= pd.Timestamp(a["validation_first_date"]) - pd.Timedelta(days=7)
    assert 0 <= a["anchor"]["alpha"] <= 1 and .5 <= a["anchor"]["beta"] <= 1.5
    assert a["anchor"]["alpha"] > 0
    corrected = corrected_probabilities(frame, a)
    np.testing.assert_allclose(corrected["market_combination"].reshape(-1, 2).sum(axis=1), 1)


def test_interleaved_races_are_rejected_before_combining_probabilities():
    frame = calibration_frame(uniform=True).iloc[:4].copy()
    frame = frame.iloc[[0, 2, 1, 3]]
    with pytest.raises(ValueError, match="contiguous"):
        corrected_probabilities(frame, {"temperature": 1, "anchor": {"alpha": 0, "beta": 1}})


def test_experience_diagnostics_use_prior_history_instead_of_scrape_time_totals():
    rows = []
    for race in range(60):
        for runner in range(2):
            rows.append({"race_id": race, "race_date": "2025-01-01", "horse_name": str(runner),
                         "won": int(runner == 0), "prior_form_runs": 0 if race < 30 else 10,
                         "horse_runs": 1000, "market_probability": .5})
    frame = pd.DataFrame(rows)
    p = np.full(len(frame), .5)
    segments = gap_segments(frame, {"market": p, "independent": p})
    assert segments["exposure=contains lightly raced"]["races"] == 30
    assert segments["exposure=all at least 3 starts"]["races"] == 30
