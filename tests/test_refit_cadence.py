import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_independent_features import score_pair
from scripts.benchmark_refit_cadence import assemble_month, cadence_advances, month_schedule


def test_calendar_schedule_preserves_quarter_start_models_and_six_new_fits():
    schedule = month_schedule(["2025Q3", "2025Q4", "2026Q1"], 7)
    assert len(schedule) == 9
    assert sum(not month["reuse_quarterly_model"] for month in schedule) == 6
    january = next(month for month in schedule if month["month"] == "2026-01")
    assert january["as_of"] == "2025-12-25"
    assert schedule[1]["as_of"] == "2025-07-25"
    # Calendar anchoring does not move forward if Jan 1-2 have no results.
    assert january["decision_start"] == "2026-01-01"


def test_month_alignment_keeps_dnf_fields_and_rejects_missing_runners_or_late_fit():
    expected = pd.DataFrame({"race_id": [1, 1, 1], "horse_name": ["A", "B", "C"],
                             "race_date": ["2025-08-02"] * 3, "won": [1, 0, 0],
                             "finish_position": [1, 2, 0], "baseline_probability": [.5, .3, .2]})
    actual = pd.DataFrame({"race_id": [1, 1, 1], "horse_name": ["C", "A", "B"],
                           "candidate_probability": [.1, .6, .3]})
    specification = month_schedule(["2025Q3"], 7)[1]
    def combine(value, cutoff="2025-07-25"):
        return assemble_month(expected, value, baseline_last="2025-06-24", candidate_last=cutoff,
                              specification=specification)
    result = combine(actual)
    assert len(result) == 3 and result.finish_position.eq(0).sum() == 1
    np.testing.assert_allclose(result.candidate_probability, [.6, .3, .1])
    assert result.candidate_parameter_age_days.eq(8).all()
    with pytest.raises(ValueError, match="fields differ"):
        combine(actual.iloc[:2])
    with pytest.raises(ValueError, match="after the registered cutoff"):
        combine(actual, "2025-07-26")


def test_cadence_gate_requires_all_quarters_and_date_bound():
    frame = pd.DataFrame({"race_id": np.repeat(range(10), 2), "race_date": ["2025-01-01"] * 20,
                          "won": [1, 0] * 10, "baseline_probability": [.5, .5] * 10,
                          "candidate_probability": [.7, .3] * 10})
    pooled = score_pair(frame)
    assert cadence_advances(pooled, [pooled] * 3)
    assert not cadence_advances(pooled, [pooled] * 2)
    altered = {**pooled, "paired_delta_candidate_minus_baseline": {
        **pooled["paired_delta_candidate_minus_baseline"], "date_cluster_race_log_loss": {"ci95": [-1., .001]}}}
    assert not cadence_advances(altered, [pooled] * 3)
