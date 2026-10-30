import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from src.confirmation_planning import race_differences, variance_estimates, sample_size, planning_scenario
from src.forecast_evaluation import probability_metrics


def fields():
    return pd.DataFrame({"race_id": ["a", "a", "b", "b", "b"],
        "race_date": ["2025-01-01"]*2+["2025-01-02"]*3,
        "horse_name": ["A", "B", "C", "D", "E"], "won": [1, 0, 0, 1, 0],
        "finish_position": [1, 0, 2, 1, 0], "num_runners": [2, 2, 3, 3, 3],
        "candidate": [.6, .4, .2, .5, .3], "market": [.7, .3, .1, .6, .3]})


def test_loss_differences_match_shared_proper_scores_with_dnfs_and_unequal_fields():
    frame = fields()
    losses = race_differences(frame, "candidate", "market")
    p, _, _ = probability_metrics(frame.candidate, frame.won, [2, 3])
    q, _, _ = probability_metrics(frame.market, frame.won, [2, 3])
    assert losses.race_log_loss.mean() == pytest.approx(p["race_log_loss"]-q["race_log_loss"])
    assert losses.brier_total.sum()/losses.runners.sum() == pytest.approx(p["brier"]-q["brier"])
    assert losses.runners.tolist() == [2, 3]
    permuted = race_differences(frame.sample(frac=1, random_state=4), "candidate", "market")
    pd.testing.assert_frame_equal(losses, permuted)


@pytest.mark.parametrize("problem", ["partial", "duplicate", "labels", "probabilities", "sum"])
def test_malformed_whole_fields_cannot_supply_power_evidence(problem):
    frame = fields()
    if problem == "partial": frame = frame.iloc[1:]
    elif problem == "duplicate": frame.loc[1, "horse_name"] = "A"
    elif problem == "labels": frame.loc[1, "won"] = 1
    elif problem == "probabilities": frame.loc[0, "candidate"] = np.nan
    else: frame.loc[0, "market"] = .6
    with pytest.raises(ValueError):
        race_differences(frame, "candidate", "market")


def test_runner_brier_ratio_influence_uses_size_centering():
    races = pd.DataFrame({"race_id": ["a", "b", "c"],
        "race_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "runners": [2, 8, 5], "race_log_loss": [-1., 0., 1.], "brier_total": [.4, .8, -.1]})
    stats = variance_estimates(races)
    theta = 1.1/15
    influence = (races.brier_total-theta*races.runners)/5
    assert stats["observed_mean_candidate_minus_market"]["runner_brier"] == pytest.approx(theta)
    assert theta != pytest.approx((races.brier_total/races.runners).mean())
    assert stats["variance_per_race"]["race"]["runner_brier"] == pytest.approx(np.var(influence, ddof=1))
    assert stats["variance_per_race"]["date"]["runner_brier"] == pytest.approx(np.var(influence, ddof=1))


def test_same_day_loss_dependence_inflates_date_variance():
    races = pd.DataFrame({"race_id": list("abcdefgh"),
        "race_date": ["2025-01-01"]*4+["2025-01-02"]*4,
        "runners": [5]*8, "race_log_loss": [-1]*4+[1]*4, "brier_total": [-.1]*4+[.1]*4})
    stats = variance_estimates(races)
    assert stats["variance_per_race"]["race"]["race_log_loss"] == pytest.approx(8/7)
    assert stats["variance_per_race"]["date"]["race_log_loss"] == pytest.approx(8)
    assert stats["dates"] == 2


def test_normal_formula_targets_four_gate_joint_power_conservatively():
    n = sample_size(1., margin=1., assumed_difference=0.)
    z = NormalDist().inv_cdf(.95)
    assert n == math.ceil((2*z)**2) == 11
    marginal_power = NormalDist().cdf(math.sqrt(n)-z)
    assert 1-4*(1-marginal_power) >= .8
    assert sample_size(1., margin=.5, assumed_difference=0.) >= 4*n-3
    assert sample_size(0., margin=.1, assumed_difference=0.) == 2


@pytest.mark.parametrize("difference", [.002, .003])
def test_no_large_sample_can_deliver_high_power_for_null_side_mean(difference):
    assert sample_size(.1, margin=.002, assumed_difference=difference) is None


@pytest.mark.parametrize("kwargs", [{"variance": -1}, {"variance": float("nan")},
    {"alpha": .9}, {"joint_power": .2}, {"gates": 0}, {"gates": True}])
def test_invalid_planning_assumptions_fail(kwargs):
    args = {"variance": 1., "margin": .1, "assumed_difference": 0., **kwargs}
    with pytest.raises(ValueError):
        sample_size(**args)


def test_scenario_uses_largest_slice_and_sampling_requirement():
    stats = {"variance_per_race": {"race": {"race_log_loss": 1., "runner_brier": .01},
                                   "date": {"race_log_loss": 4., "runner_brier": .02}}}
    result = planning_scenario({"pooled": stats}, margins={"race_log_loss": .002, "runner_brier": .0001},
        assumed_differences={"race_log_loss": 0., "runner_brier": 0.})
    assert result["limiting_requirement"]["metric"] == "runner_brier"
    assert result["limiting_requirement"]["sampling"] == "date"
    assert result["required_races"] == sample_size(.02, margin=.0001, assumed_difference=0.)
