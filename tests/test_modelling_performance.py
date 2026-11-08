"""Performance-target, chronological-residual and heterogeneous ensemble contracts."""
from dataclasses import FrozenInstanceError, replace

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from src.modelling.candidate import WinCandidate, fit_candidate
from src.modelling.components import component_model, spec_from_dict, training_columns
from src.modelling.evaluation import component_oof
from src.modelling.performance import PerformanceModel, PerformanceSpec, performance_targets, win_probabilities
from src.modelling.pool import LogPool
from src.modelling.spec import ModelSpec
from src.wf_eval import Fold


def fields(days=180):
    rng = np.random.default_rng(214)
    ids = np.repeat(np.arange(days), 6)
    signal = rng.normal(size=len(ids))
    performance = signal + rng.normal(scale=.7, size=len(ids))
    rank = pd.Series(performance).groupby(ids).rank(ascending=False).to_numpy(dtype=int)
    lengths = np.repeat(np.maximum.reduceat(performance, np.arange(0, len(ids), 6)), 6) - performance
    return pd.DataFrame({"race_id": ids, "horse_id": np.tile(np.arange(6), days),
        "horse_name": np.tile([f"horse-{n}" for n in range(6)], days),
        "race_date": np.repeat(pd.date_range("2024-01-01", periods=days), 6), "num_runners": 6,
        "won": (rank == 1).astype(int), "finish_position": np.where(rank == 6, 0, rank),
        "beaten_lengths": lengths, "distance_furlongs": 8., "signal": signal,
        "odds": 7., "bsp": 6.})


def recipe(**kwargs):
    params = (("n_estimators", 15), ("learning_rate", .1), ("num_leaves", 7),
              ("min_child_samples", 12), ("colsample_bytree", 1.), ("subsample", 1.))
    spec = PerformanceSpec("performance", ("signal",), params=params, screen_params=(("n_estimators", 12),),
        early_stopping_rounds=3, burn_in_months=0, quadrature_points=128)
    return replace(spec, **kwargs)


def test_performance_recipe_is_resolved_immutable_and_dispatches():
    spec = recipe()
    assert spec_from_dict(spec.to_dict()) == spec
    assert isinstance(component_model(spec), PerformanceModel)
    assert training_columns([spec]) == ("beaten_lengths", "distance_furlongs")
    with pytest.raises(FrozenInstanceError):
        spec.margin_cap_lb = 999
    for change in ({"margin_cap_lb": 0}, {"sigma_floor": np.nan}, {"residual_splits": 2},
                   {"quadrature_points": 10}, {"features": ("beaten_lengths",)}):
        with pytest.raises(ValueError):
            recipe(**change)
    with pytest.raises(ValueError, match="Unknown"):
        spec_from_dict({"kind": "invented"})


def test_performance_targets_never_invent_non_finisher_or_missing_figures():
    frame = pd.DataFrame({"finish_position": [1, 2, 3, 0, 4, 5, 6, 7],
        "beaten_lengths": [np.nan, 2, 100, 1, np.nan, -1, 5, 2],
        "distance_furlongs": [8, 8, 5, 8, 8, 8, 0, np.nan]})
    target = performance_targets(frame, recipe())
    np.testing.assert_allclose(target[:3], [0, -3.75, -30])
    assert np.isnan(target[3:]).all()


def test_normal_integration_matches_analytic_two_runner_result_and_symmetries():
    means, sigma = np.array([1., -2.]), np.array([2., 4.])
    expected = ndtr((means[0] - means[1]) / np.sqrt(np.sum(sigma ** 2)))
    p = win_probabilities(means, sigma, [2])
    np.testing.assert_allclose(p, [expected, 1 - expected], atol=1e-8)
    np.testing.assert_allclose(win_probabilities(10 + means * 3, sigma * 3, [2]), p, atol=1e-12)
    np.testing.assert_allclose(win_probabilities([0.] * 4, [2.] * 4, [4]), .25, atol=1e-12)
    np.testing.assert_array_equal(win_probabilities(np.tile(means, 5), np.tile(sigma, 5), [2] * 5,
        chunk_cells=10), np.tile(p, 5))
    for bad in ([1], [3], [2.5]):
        with pytest.raises(ValueError, match="complete"):
            win_probabilities(means, sigma, bad)


@pytest.mark.parametrize("capacity", ["screen", "native"])
def test_complete_field_prediction_needs_no_outcomes_and_survives_reload(tmp_path, capacity):
    frame, spec = fields(), recipe()
    model = PerformanceModel(spec).fit(frame, capacity=capacity)
    inference = frame.drop(columns=["won", "finish_position", "beaten_lengths", "distance_furlongs"])
    expected = model.predict(inference)
    assert model.lineage["figure_rows"] == len(frame) * 5 // 6
    assert model.lineage["missing_figures"] == len(frame) // 6
    assert model.lineage["calibration_fit_boundary"] < model.lineage["calibration_first_date"]
    for fold in model.lineage["residual_folds"]:
        assert pd.Timestamp(fold["test_first_date"]) - pd.Timestamp(fold["training_last_date"]) >= pd.Timedelta(days=7)
    np.testing.assert_allclose(pd.Series(expected).groupby(frame.race_id).sum(), 1., atol=1e-12)
    details = model.predict_details(inference)
    assert (details.sigma >= spec.sigma_floor).all()
    shuffled = inference.sample(frac=1, random_state=7)
    np.testing.assert_allclose(model.predict(shuffled), expected[shuffled.index], atol=1e-12)
    for race in (0, 100, 179):
        selected = inference.race_id.eq(race)
        np.testing.assert_allclose(model.predict(inference.loc[selected]), expected[selected], atol=1e-12)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(inference), expected)
    with pytest.raises(ValueError, match="complete"):
        model.predict(inference.iloc[1:])


def test_later_margin_labels_cannot_change_earlier_residual_fit_or_scale():
    frame, spec = fields(), recipe()
    original = PerformanceModel(spec).fit(frame, capacity="screen")
    changed = frame.copy()
    later = changed.race_date.ge(original.lineage["calibration_first_date"]) & changed.finish_position.gt(1)
    changed.loc[later, "beaten_lengths"] = 20.
    other = PerformanceModel(spec).fit(changed, capacity="screen")
    assert original.scale == other.scale
    assert original.lineage["residual_folds"] == other.lineage["residual_folds"]
    assert not np.allclose(original.predict(frame), other.predict(frame))


def test_performance_oof_cache_tracks_margin_labels_and_mixed_candidates_reload(tmp_path):
    frame = fields(200)
    day = (frame.race_date - frame.race_date.min()).dt.days
    folds = [Fold("earlier", day.lt(105).to_numpy(), day.between(112, 133).to_numpy(), 0),
             Fold("later", day.lt(155).to_numpy(), day.between(162, 199).to_numpy(), 0)]
    performance = recipe(features=("signal", "distance_furlongs"))
    win = ModelSpec("generalist", ("signal",), params=performance.params, screen_params=performance.screen_params,
        burn_in_months=0, early_stopping_rounds=3)
    records, _ = component_oof(frame, [win, performance], folds=folds, capacity="screen", cache_dir=tmp_path / "cache")
    replayed, _ = component_oof(frame, [win, performance], folds=folds, capacity="screen", cache_dir=tmp_path / "cache")
    pd.testing.assert_frame_equal(records, replayed)
    candidate = fit_candidate(frame, [win, performance], records, as_of=frame.race_date.max(), capacity="screen")
    path = tmp_path / "candidate.joblib"
    candidate.save(path)
    expected = candidate.predict(frame.drop(columns=["won", "finish_position", "beaten_lengths"]))
    np.testing.assert_array_equal(WinCandidate.load(path).predict(frame), expected)
    changed = frame.copy()
    changed.loc[changed.finish_position.gt(1), "beaten_lengths"] += 1.
    updated, _ = component_oof(changed, [performance], folds=folds, capacity="screen", cache_dir=tmp_path / "cache")
    assert set(updated.cache_key).isdisjoint(set(records.loc[records.component.eq(performance.id), "cache_key"]))
    assert len(list((tmp_path / "cache").glob("*/receipt.json"))) == 6


def test_unavailable_performance_expert_keeps_exact_generalist_fallback():
    p = np.array([.4, .2, .15, .1, .1, .05])
    pooled = LogPool(("generalist", "performance"), weights=(.6, .4))
    np.testing.assert_array_equal(pooled.predict(np.column_stack([p, np.full(6, np.nan)]), np.zeros(6)), p)
