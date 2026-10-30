import numpy as np
import pandas as pd
import pytest

from scripts.assess_forecast_parity import parity_evidence, upper_bound


def test_cluster_bootstrap_preserves_runner_weighting_and_small_difference_precision():
    # Two races of unequal size: runner mean .00012, race mean .00015.
    values = np.array([.0002] * 2 + [.0001] * 8)
    result = upper_bound(values, ["small"] * 2 + ["large"] * 8, draws=1000)
    assert result["mean"] == pytest.approx(.00012, abs=1e-10)
    assert result["upper95"] == pytest.approx(.0002, abs=1e-10)
    assert result["clusters"] == 2


def test_identical_market_forecast_has_exact_zero_paired_uncertainty():
    frame = pd.DataFrame({"race_id": ["a", "a", "b", "b", "b"],
                          "race_date": ["2025-01-01"] * 2 + ["2025-01-02"] * 3,
                          "won": [1, 0, 0, 1, 0]})
    p = np.array([.7, .3, .1, .6, .3])
    result = parity_evidence(frame, {"market": p, "candidate": p.copy()}, draws=100)
    assert result["candidate"]["within_proposed_margins_in_both_analyses"]
    for name in ("race_bootstrap", "date_cluster_sensitivity"):
        assert result["candidate"][name]["runner_brier"]["upper95"] == 0
        assert result["candidate"][name]["race_log_loss"]["upper95"] == 0


def test_interleaved_fields_cannot_silently_score_as_different_races():
    frame = pd.DataFrame({"race_id": ["a", "b", "a", "b"],
                          "race_date": ["2025-01-01"] * 4, "won": [1, 0, 0, 1]})
    p = np.full(4, .5)
    with pytest.raises(ValueError, match="contiguous"):
        parity_evidence(frame, {"market": p, "candidate": p.copy()})
