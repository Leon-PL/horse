import numpy as np
import pandas as pd

from scripts.benchmark_independent_features import score_pair
from scripts.benchmark_race_objective import candidate_advances


def test_two_candidate_gate_requires_all_periods_and_date_sensitivity():
    frame = pd.DataFrame({"race_id": np.repeat(range(8), 2), "race_date": ["2025-01-01"] * 16,
                          "won": [1, 0] * 8, "baseline_probability": [.5, .5] * 8,
                          "candidate_probability": [.7, .3] * 8})
    pooled = score_pair(frame)
    assert candidate_advances(pooled, [pooled] * 3)
    assert not candidate_advances(pooled, [pooled] * 2)
    date_worse = {**pooled, "paired_delta_candidate_minus_baseline": {
        **pooled["paired_delta_candidate_minus_baseline"], "date_cluster_race_log_loss": {"ci95": [-1, .01]}}}
    assert not candidate_advances(date_worse, [pooled] * 3)
