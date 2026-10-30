import numpy as np
import pandas as pd
import pytest

from scripts.screen_calibration_choice import temporal_split, fit_map, apply_map, eligible


def history():
    dates = pd.date_range("2025-01-01", periods=60)
    return pd.DataFrame({"race_id": np.repeat(np.arange(120), 3),
        "race_date": np.repeat(dates, 6), "horse_name": ["a", "b", "c"]*120,
        "finish_position": [1, 2, 3]*120, "won": [1, 0, 0]*120,
        "raw_probability": [.4, .15, .08]*120})


def test_choice_keeps_whole_dates_and_seven_day_purge():
    frame = history()
    fit, validation = temporal_split(frame)
    assert not (fit & validation).any()
    assert frame.loc[fit, "race_date"].max()+pd.Timedelta(days=7) <= frame.loc[validation, "race_date"].min()
    assert not set(frame.loc[fit, "race_id"]) & set(frame.loc[validation, "race_id"])
    assert frame.groupby("race_date").apply(lambda g: fit.loc[g.index].nunique(), include_groups=False).max() == 1
    assert frame.groupby("race_date").apply(lambda g: validation.loc[g.index].nunique(), include_groups=False).max() == 1


def test_choice_rejects_out_of_order_history():
    with pytest.raises(ValueError, match="chronological"):
        temporal_split(history().iloc[::-1])


@pytest.mark.parametrize("family", ["native", "platt_only", "race_temperature"])
def test_each_family_predicts_complete_race_distribution(family):
    frame = history()
    fitted = fit_map(family, frame)
    probabilities = apply_map(fitted, frame)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.reshape(-1, 3).sum(axis=1), 1., atol=1e-12)
    if family != "native":
        assert (probabilities > 0).all()


def test_choice_requires_brier_binary_and_both_log_loss_intervals():
    result = {"baseline": {"brier": .1, "binary_log_loss": .3},
        "candidate": {"brier": .099, "binary_log_loss": .299},
        "paired_delta_candidate_minus_baseline": {"race_log_loss": {"ci95": [-.01, -.001]},
            "date_cluster_race_log_loss": {"ci95": [-.02, -.0001]}}}
    assert eligible(result)
    result["candidate"]["brier"] = .101
    assert not eligible(result)
    result["candidate"]["brier"] = .099
    result["candidate"]["binary_log_loss"] = .301
    assert not eligible(result)
    result["candidate"]["binary_log_loss"] = .299
    result["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]["ci95"][1] = .001
    assert not eligible(result)
