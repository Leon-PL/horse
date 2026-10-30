import copy

import numpy as np
import pandas as pd
import pytest

import config
from scripts.benchmark_history_features import align_reference, qualify, verify_reference


def fixture():
    frame = pd.DataFrame({"race_id": [1, 1], "horse_name": ["a", "b"],
                          "race_date": pd.to_datetime(["2025-07-01"] * 2), "won": [1, 0]})
    reference = frame.copy()
    reference["baseline_probability"] = [.7, .3]
    return frame, reference


def test_reference_reuse_requires_identical_identity_labels_and_native_predictions():
    frame, reference = fixture()
    p, error = align_reference(frame, reference, np.array([.7, .3]))
    assert np.array_equal(p, [.7, .3]) and error == 0
    reference["won"] = [0, 1]
    with pytest.raises(ValueError, match="labels"):
        align_reference(frame, reference, np.array([.7, .3]))


def test_reference_never_silently_drops_a_runner_or_changes_forecasts():
    frame, reference = fixture()
    with pytest.raises(ValueError, match="universe"):
        align_reference(frame, reference.iloc[:1], np.array([.7, .3]))
    with pytest.raises(ValueError, match="replay"):
        align_reference(frame, reference, np.array([.6, .4]))


def test_reference_alignment_is_by_identity_not_row_order():
    frame, reference = fixture()
    p, error = align_reference(frame, reference.iloc[::-1], np.array([.7, .3]))
    assert error == 0 and np.array_equal(p, [.7, .3])


def test_incompatible_reference_dataset_recipe_or_source_is_rejected():
    params = {"classifier": {"n_estimators": 2000}}
    manifest = {"source_file_sha256": "features", "params": params, "seed": 42, "threads": 2,
                "feature_dtype": "float32 for both variants", "recency_half_life_days": config.RECENCY_HALF_LIFE_DAYS,
                "calibration": config.WIN_CALIBRATION_METHOD, "distillation": bool(config.WIN_SOFT_LABEL_DISTILL),
                "code_sha256": {"src/model.py": "native"}}
    verify_reference(manifest, {"src/model.py": "native"}, "features", params)
    with pytest.raises(ValueError, match="recipe"):
        verify_reference(manifest, {"src/model.py": "native"}, "another", params)
    with pytest.raises(ValueError, match="code"):
        verify_reference(manifest, {"src/model.py": "changed"}, "features", params)


def test_advancement_needs_all_periods_and_date_cluster_support():
    result = {"baseline": {"brier": .1, "binary_log_loss": .3, "race_log_loss": 1.8},
              "candidate": {"brier": .09, "binary_log_loss": .29, "race_log_loss": 1.79},
              "paired_delta_candidate_minus_baseline": {"race_log_loss": {"ci95": [-.02, -.001]},
                                                       "date_cluster_race_log_loss": {"ci95": [-.03, -.001]}}}
    assert not qualify(result, [result, result])
    assert qualify(result, [result] * 3)
    changed = copy.deepcopy(result)
    changed["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]["ci95"][1] = .001
    assert not qualify(changed, [result] * 3)
