"""Final fitting uses every allowed row and none of the calendar holdout."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config
from src.model import RacePredictor, _SoftLabelClassifier, prepare_multi_target_data


@pytest.fixture
def history(monkeypatch):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", ["signal"])
    monkeypatch.setattr(config, "BURN_IN_MONTHS", 0)
    monkeypatch.setattr(config, "PURGE_DAYS", 7)
    rows = []
    for day in range(150):
        for race in range(1 + day % 2):
            for runner, position in enumerate([1, 2, 3, 4, 0]):
                rows.append({"race_id": f"{day}-{race}",
                             "race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day),
                             "horse_name": f"horse-{runner}", "num_runners": 5,
                             "finish_position": position, "won": int(position == 1),
                             "lengths_behind": float(runner), "handicap": 0,
                             "signal": float(day), "odds": float(runner + 2)})
    return pd.DataFrame(rows)


def test_calendar_holdout_is_whole_dates_and_purged(history):
    data = prepare_multi_target_data(history, holdout_days=30)
    cutoff = history.race_date.max() - pd.Timedelta(days=29)
    assert data["test_df"].race_date.min() == cutoff
    assert data["test_df"].race_date.max() == history.race_date.max()
    assert data["train_df"].race_date.max() <= cutoff - pd.Timedelta(days=7)
    assert not set(data["train_df"].race_id) & set(data["test_df"].race_id)
    expected = history[history.race_date >= cutoff]
    assert len(data["test_df"]) == len(expected)
    assert data["test_df"].finish_position.eq(0).sum() == expected.finish_position.eq(0).sum()


@pytest.mark.parametrize("days", [0, -1, True, 1.5, 151])
def test_invalid_or_untrainable_holdout_fails(history, days):
    with pytest.raises(ValueError):
        prepare_multi_target_data(history, holdout_days=days)


def test_training_only_keeps_latest_history_without_fabricated_test(history):
    data = prepare_multi_target_data(history, training_only=True)
    assert len(data["X_train"]) == len(history)
    assert len(data["X_test"]) == len(data["groups_test"]) == 0
    assert data["train_df"].race_date.max() == history.race_date.max()
    with pytest.raises(ValueError, match="cannot also reserve"):
        prepare_multi_target_data(history, holdout_days=30, training_only=True)


@pytest.mark.parametrize("win_only", [True, False])
def test_as_of_native_fit_uses_latest_allowed_date_and_never_future(history, monkeypatch, win_only, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 2)
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_DISTILL", False)
    predictor = RacePredictor(frameworks={"classifier": "lgbm", "place": "lgbm"})
    predictor._autotune_njobs = 1
    cutoff = history.race_date.max() - pd.Timedelta(days=20)
    last_signal = float(history.loc[history.race_date <= cutoff, "signal"].max())
    fits = {"classifier": [], "place": []}
    for role, name in (("classifier", "_train_win_classifier"), ("place", "_train_place_classifier")):
        original = getattr(predictor, name)

        def fit(X, y, original=original, role=role, **kwargs):
            fits[role].append((len(X), float(X[:, 0].max())))
            assert X[:, 0].max() <= last_signal
            if kwargs.get("eval_set"):
                assert kwargs["eval_set"][0][0][:, 0].max() <= last_signal
            return original(X, y, **kwargs)
        monkeypatch.setattr(predictor, name, fit)
    info = predictor.fit_as_of(history, cutoff, win_only=win_only,
                              params={r: {"n_estimators": 8, "min_child_samples": 3} for r in fits})
    assert (int((history.race_date <= cutoff).sum()), last_signal) in fits["classifier"]
    assert bool(fits["place"]) != win_only
    assert (predictor.place_model is None) == win_only
    assert info["training_last_date"] == str(cutoff.date())
    assert info["holdout_races"] == 0 and predictor.metrics is None and predictor.test_analysis is None
    cal = info["calibration"]
    assert pd.Timestamp(cal["base_training_last_date"]) <= pd.Timestamp(cal["first_date"]) - pd.Timedelta(days=7)
    future = history.loc[history.race_date > cutoff]
    prediction = predictor.predict_races(future)
    np.testing.assert_allclose(prediction.groupby("race_id").win_probability.sum(), 1)
    with monkeypatch.context() as win_context:
        def no_place(*args, **kwargs):
            pytest.fail("Win-only inference must not compute place probabilities")
        win_context.setattr(predictor, "_compute_place_probabilities", no_place)
        win_prediction = predictor.predict_races(future, win_only=True)
    np.testing.assert_allclose(win_prediction.win_probability, prediction.win_probability, atol=1e-12)
    assert "place_probability" not in win_prediction and "value_score" not in win_prediction
    path = tmp_path / "as_of.joblib"
    predictor.save(str(path))
    restored = RacePredictor()
    restored.load(str(path))
    assert restored.eval_split_info == info
    replay = restored.predict_races(future)
    np.testing.assert_allclose(prediction.win_probability, replay.win_probability, atol=1e-12)


def test_as_of_future_changes_cannot_change_fit(history, monkeypatch):
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 2)
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_DISTILL", False)
    cutoff = history.race_date.max() - pd.Timedelta(days=20)
    changed = history.copy()
    changed.loc[changed.race_date > cutoff, ["signal", "won", "odds"]] = 999
    predictors = [RacePredictor(), RacePredictor()]
    for predictor, frame in zip(predictors, [history, changed]):
        predictor._autotune_njobs = 1
        predictor.fit_as_of(frame, cutoff, win_only=True,
                            params={"classifier": {"n_estimators": 8, "min_child_samples": 3}})
    assert predictors[0].clf_model.booster_.model_to_string() == predictors[1].clf_model.booster_.model_to_string()


@pytest.mark.parametrize("selected,expected", [
    (SimpleNamespace(best_iteration_=7), 7),
    (SimpleNamespace(best_iteration_=0, n_estimators_=11), 11),
    (_SoftLabelClassifier(SimpleNamespace(best_iteration_=9)), 9),
])
def test_refit_uses_selected_capacity_all_training_rows_and_no_holdout(history, selected, expected):
    data = prepare_multi_target_data(history, holdout_days=30)
    predictor = RacePredictor(frameworks={"classifier": "lgbm", "place": "lgbm"})
    predictor.eval_split_info = {}
    params = {"n_estimators": 100, "learning_rate": .05}
    weights = np.linspace(1, 2, len(data["X_train"]))
    captured = {}
    final = object()

    def fit(X, y, **kwargs):
        captured.update(X=X, y=y, **kwargs)
        return final

    assert predictor._refit_selected_model(
        selected, "classifier", fit, data["X_train"], data["y_train_won"], params,
        sample_weight=weights, soft_y=data["y_train_won"].astype(float),
    ) is final
    assert captured["X"] is data["X_train"]
    assert captured["y"] is data["y_train_won"]
    assert captured["sample_weight"] is weights
    assert len(captured["soft_y"]) == len(data["X_train"])
    assert captured["X"][:, 0].max() < data["X_test"][:, 0].min()
    assert "eval_set" not in captured
    assert captured["params"]["n_estimators"] == expected
    assert params["n_estimators"] == 100
    assert predictor.eval_split_info["refit_rounds"]["classifier"] == expected


def test_refit_refuses_unknown_capacity(history):
    predictor = RacePredictor(frameworks={"classifier": "lgbm", "place": "lgbm"})
    with pytest.raises(ValueError, match="selected tree count"):
        predictor._refit_selected_model(SimpleNamespace(), "classifier", None,
                                        np.ones((3, 1)), np.array([1, 0, 0]), {})


def test_real_training_refits_both_models_without_holdout_rows(history, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CV_N_FOLDS", 2)
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 2)
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_DISTILL", False)
    predictor = RacePredictor(frameworks={"classifier": "lgbm", "place": "lgbm"})
    # Deliberately miscalibrated independent marginals make the adjustment
    # observable; old evaluation reported .9 while serving constrained them.
    monkeypatch.setattr(predictor, "_calibrate_place_probs", lambda p: np.full(len(p), .9))
    prepared = prepare_multi_target_data(history, holdout_days=30)
    calls = {"classifier": [], "place": []}
    for role, method in (("classifier", "_train_win_classifier"), ("place", "_train_place_classifier")):
        original = getattr(predictor, method)

        def fit(X, y, original=original, role=role, **kwargs):
            calls[role].append((len(X), float(X[:, 0].max()), kwargs.get("eval_set")))
            return original(X, y, **kwargs)

        monkeypatch.setattr(predictor, method, fit)
    params = {role: {"n_estimators": 8, "max_depth": 3, "min_child_samples": 3}
              for role in calls}
    predictor.train(history, save=False, params=params, holdout_days=30, blend_market_odds=False)
    for role in calls:
        assert any(rows == len(prepared["X_train"]) and validation is None
                   for rows, _, validation in calls[role])
        assert all(last_day < prepared["X_test"][:, 0].min()
                   for _, last_day, _ in calls[role])
        assert 1 <= predictor.eval_split_info["refit_rounds"][role] <= 8
    assert predictor.eval_split_info["holdout_first_date"] == str(prepared["test_df"].race_date.min())[:10]
    saved = predictor.test_analysis["predictions"]
    public = predictor.predict_races(prepared["test_df"])
    aligned = saved.merge(public[["race_id", "horse_name", "place_probability"]],
                          on=["race_id", "horse_name"], validate="one_to_one")
    np.testing.assert_allclose(aligned.place_prob, aligned.place_probability)
    assert not np.allclose(aligned.place_prob, .9)
