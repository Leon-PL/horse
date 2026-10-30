from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_atr_invariance import audit
from scripts.benchmark_independent_features import advancement, check_eligibility, score_pair


@pytest.fixture(scope="module")
def registration():
    root = Path(__file__).resolve().parents[1]
    def names(path):
        return [line.strip() for line in (root / path).read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    unsafe = [f"{name}{suffix}" for name in ("atr_speed_score", "atr_stride_x_gallop")
              for suffix in ("", "_vs_field", "_z_in_race", "_rank_pct_in_race", "_vs_best_in_race")]
    return {"baseline_features": names("feature_sets/win.txt"),
            "candidate_features": names("feature_sets/research_win_with_atr.txt"),
            "unsafe_composites_requiring_exclusion_or_rebuild": unsafe}


def test_every_added_atr_column_is_invariant_to_future_and_current_day_results(registration):
    result = audit(registration)
    assert result["passed"] and result["eligible_atr_columns"] == 206
    assert all(not changes for changes in result["checks"].values())
    assert result["quarantined_negative_control"]
    assert result["prior_information_sensitivity_columns"] > 0
    check_eligibility(registration, result)
    altered = {**registration, "candidate_features": registration["candidate_features"] + ["atr_speed_score"]}
    with pytest.raises(ValueError):
        check_eligibility(altered, result)


def pair():
    return pd.DataFrame({"race_id": np.repeat(np.arange(10), 2), "race_date": ["2025-01-01"] * 20,
                         "won": [1, 0] * 10, "baseline_probability": [.5, .5] * 10,
                         "candidate_probability": [.7, .3] * 10})


def test_advancement_requires_every_registered_fold_and_all_probability_scores():
    score = score_pair(pair())
    assert advancement(score, [score] * 3)
    assert not advancement(score, [score] * 2)
    worse_brier = {**score, "candidate": {**score["candidate"], "brier": 1.}}
    assert not advancement(worse_brier, [score] * 3)
    worse_binary = {**score, "candidate": {**score["candidate"], "binary_log_loss": 1.}}
    assert not advancement(worse_binary, [score] * 3)


def test_pair_scoring_rejects_interleaved_fields():
    with pytest.raises(ValueError, match="contiguous"):
        score_pair(pair().iloc[[0, 2, 1, 3]])
