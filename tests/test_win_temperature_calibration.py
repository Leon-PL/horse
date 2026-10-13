"""Race-temperature integration and compatibility of saved win calibrators."""
import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

import config
from src.model import RacePredictor, _IdentityScaler, _grouped_normalize
from src.probability_calibration import fit_temperature, race_temperature


class _ColumnClassifier:
    def predict_proba(self, X):
        p = np.asarray(X)[:, 0].astype(float)
        return np.column_stack([1 - p, p])


def _predictor(method="race_temperature"):
    predictor = RacePredictor(win_calibration_method=method)
    predictor.feature_cols = ["signal"]
    predictor.scaler = _IdentityScaler()
    predictor.clf_model = _ColumnClassifier()
    predictor.place_model = _ColumnClassifier()
    predictor.calibration_temp = 0.7
    return predictor


def _races():
    return pd.DataFrame({
        "race_id": ["a"] * 5 + ["b"] * 7,
        "horse_name": [f"horse-{i}" for i in range(12)],
        "signal": np.linspace(0.02, 0.9, 12),
        "num_runners": [5] * 5 + [7] * 7,
        "handicap": 0,
        "odds": np.linspace(2, 25, 12),
    })


@pytest.mark.parametrize("temperature", [0.25, 1.0, 4.0, 1e-320, 1e300])
def test_temperature_stable_extremes_and_unit_race_sums(temperature):
    raw = np.array([0, 0, 0, 1, 1e-300, 0, 1, 0, 0.01, 0.3])
    sizes = np.array([3, 3, 1, 3])
    actual = race_temperature(raw, sizes, temperature)
    assert np.isfinite(actual).all()
    assert ((actual >= 0) & (actual <= 1)).all()
    np.testing.assert_allclose(np.add.reduceat(actual, [0, 3, 6, 7]), 1)
    np.testing.assert_allclose(actual[:3], 1 / 3)
    assert actual[6] == 1
    assert actual[-1] >= actual[-2] >= actual[-3]


@pytest.mark.parametrize("temperature", [0, -1, np.nan, np.inf])
def test_temperature_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        race_temperature([0.2, 0.8], [2], temperature)


@pytest.mark.parametrize("groups", [[3], [0, 2], [1.5, 0.5], [np.nan], [[2]]])
def test_temperature_rejects_misaligned_races(groups):
    with pytest.raises(ValueError):
        race_temperature([0.2, 0.8], groups, 1)


@pytest.mark.parametrize("raw", [[np.nan, 0.4], [np.inf, 0.2], [-0.1, 0.4], [1.1, 0.3]])
def test_temperature_rejects_invalid_probabilities(raw):
    with pytest.raises(ValueError, match="Probabilities"):
        race_temperature(raw, [2], 1)


def test_temperature_matches_original_benchmark_formula():
    raw = np.array([0.1, 0.04, 0.2, 0.07, 0.3, 0.12, 0.06])
    groups = np.array([3, 4])
    starts = np.array([0, 3])
    scores = np.log(np.clip(raw, 1e-12, 1)) / 0.925729
    scores -= np.repeat(np.maximum.reduceat(scores, starts), groups)
    expected = _grouped_normalize(np.exp(scores), groups)
    np.testing.assert_allclose(race_temperature(raw, groups, 0.925729), expected, atol=1e-15)


def test_temperature_fit_ignores_whole_invalid_outcome_races():
    raw = np.tile([0.99, 0.01], 100)
    y = np.tile([1.0, 0.0], 100)
    y[-60:] = np.tile([0.0, 1.0], 30)
    groups = np.full(100, 2)
    expected = fit_temperature(raw, groups, y)
    assert expected > 1
    actual = fit_temperature(
        np.r_[raw, 0.5, 0.5, 0.2, 0.8, 0.5], np.r_[groups, 2, 2, 1],
        np.r_[y, 0, 0, 1, 1, 1],
    )
    assert actual == pytest.approx(expected)
    assert fit_temperature([0.5, 0.5], [2], [0, 0]) == 1
    assert fit_temperature([], [], []) == 1
    assert race_temperature([], [], 1).size == 0
    assert fit_temperature(np.tile([0.5, 0.5], 50), np.full(50, 2), np.tile([1, 0], 50)) == 1


@pytest.mark.parametrize("market_anchor", [None, {"alpha": 0.3, "beta": 0.8}])
def test_temperature_shuffled_batch_matches_individual_races(market_anchor):
    predictor = _predictor()
    predictor.market_anchor = market_anchor
    frame = _races().sample(frac=1, random_state=8)
    actual = predictor.predict_races(frame)
    np.testing.assert_allclose(actual.groupby("race_id")["win_probability"].sum(), 1)
    columns = ["win_probability", "place_probability", "predicted_rank", "value_score"]
    for race_id, race in frame.groupby("race_id", sort=False):
        single = predictor.predict_race(race).set_index("horse_name").sort_index()
        batch = actual[actual["race_id"] == race_id].set_index("horse_name").sort_index()
        np.testing.assert_allclose(batch[columns], single[columns], atol=1e-12)


def test_shared_fit_replaces_stale_calibration_and_matches_temperature_helper():
    predictor = _predictor()
    predictor.platt_a, predictor.platt_b = 2.0, -1.0
    predictor.win_iso_cal = IsotonicRegression().fit([0, 1], [0.1, 0.9])
    raw = np.tile([0.99, 0.01], 100)
    groups = np.full(100, 2)
    positions = np.tile([1, 2], 100)
    positions[-60:] = np.tile([2, 1], 30)
    predictor._fit_win_calibrator(raw, groups, positions)
    assert predictor.win_iso_cal is None
    assert (predictor.platt_a, predictor.platt_b) == (1.0, 0.0)
    assert predictor.calibration_temp == fit_temperature(raw, groups, positions == 1)
    np.testing.assert_allclose(
        predictor._calibrate_win_probs(raw, groups),
        race_temperature(raw, groups, predictor.calibration_temp),
    )


@pytest.mark.parametrize("method", ["platt_isotonic", "race_temperature"])
def test_calibration_method_round_trips_independently_of_config(tmp_path, monkeypatch, method):
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path))
    predictor = _predictor(method)
    predictor.save()
    monkeypatch.setattr(config, "WIN_CALIBRATION_METHOD", "race_temperature", raising=False)
    restored = RacePredictor()
    restored.load()
    assert restored.win_calibration_method == method
    assert restored.calibration_temp == predictor.calibration_temp
    np.testing.assert_allclose(
        restored.predict_races(_races())["win_probability"],
        predictor.predict_races(_races())["win_probability"],
    )


def test_legacy_artifact_ignores_historical_temperature_and_new_config(tmp_path, monkeypatch):
    iso = IsotonicRegression(out_of_bounds="clip").fit([0, 0.1, 0.4, 1], [0, 0.08, 0.5, 1])
    artifact = tmp_path / "legacy.joblib"
    joblib.dump({"feature_cols": ["signal"], "calibration_temp": 3.5,
                 "platt_a": 0.8, "platt_b": -0.3, "win_iso_cal": iso}, artifact)
    monkeypatch.setattr(config, "WIN_CALIBRATION_METHOD", "race_temperature", raising=False)
    restored = RacePredictor()
    assert restored.win_calibration_method == "race_temperature"
    restored.load(str(artifact))
    assert restored.win_calibration_method == "platt_isotonic"
    raw, groups = np.array([0.1, 0.4, 0.02, 0.7, 0.1]), np.array([2, 3])
    platt = expit(0.8 * np.log(raw / (1 - raw)) - 0.3)
    expected = _grouped_normalize(np.clip(iso.predict(_grouped_normalize(platt, groups)), 1e-9, 1), groups)
    np.testing.assert_allclose(restored._calibrate_win_probs(raw, groups), expected, atol=1e-15)


def test_config_selects_new_models_and_unknown_method_fails_loudly(monkeypatch):
    monkeypatch.setattr(config, "WIN_CALIBRATION_METHOD", "race_temperature", raising=False)
    assert RacePredictor().win_calibration_method == "race_temperature"
    assert RacePredictor(win_calibration_method="platt_isotonic").win_calibration_method == "platt_isotonic"
    with pytest.raises(ValueError, match="WIN_CALIBRATION_METHOD"):
        RacePredictor(win_calibration_method="typo")


@pytest.mark.parametrize("saved_temperature", [0, -1, np.nan, np.inf])
def test_invalid_saved_temperature_fails_at_load(tmp_path, saved_temperature):
    artifact = tmp_path / "invalid.joblib"
    joblib.dump({"feature_cols": ["signal"], "win_calibration_method": "race_temperature",
                 "calibration_temp": saved_temperature}, artifact)
    with pytest.raises(ValueError, match="temperature"):
        RacePredictor().load(str(artifact))


def test_slab_uses_shared_fit_and_held_out_predictions(monkeypatch):
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 0)
    monkeypatch.setattr(RacePredictor, "_train_win_classifier", lambda *a, **kw: _ColumnClassifier())
    monkeypatch.setattr(RacePredictor, "_train_place_classifier", lambda *a, **kw: _ColumnClassifier())
    groups = np.full(50, 5)
    X = np.random.default_rng(7).uniform(0.02, 0.98, (groups.sum(), 1))
    positions = np.tile(np.arange(1, 6), len(groups))
    predictor = _predictor()
    assert predictor._fit_training_slab_calibrators(
        X, groups, (positions == 1).astype(int), (positions <= 3).astype(int),
        positions, model_params=None, sample_weight=None,
    )
    boundary = int(groups[:40].sum())
    expected = fit_temperature(X[boundary:, 0], groups[40:], positions[boundary:] == 1)
    assert predictor.calibration_temp == expected
    assert predictor.win_iso_cal is None


def test_fold_fits_temperature_only_on_training_slab(monkeypatch):
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 0)
    monkeypatch.setattr(config, "FEATURE_PRUNE_FRACTION", 0)
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_DISTILL", False)
    monkeypatch.setattr(RacePredictor, "_train_win_classifier", lambda *a, **kw: _ColumnClassifier())
    monkeypatch.setattr(RacePredictor, "_train_place_classifier", lambda *a, **kw: _ColumnClassifier())
    groups = np.full(50, 5)
    rng = np.random.default_rng(11)
    X = rng.uniform(0.02, 0.98, (groups.sum(), 1))
    positions = np.tile(np.arange(1, 6), len(groups))
    frame = pd.DataFrame({
        "race_id": np.repeat(np.arange(len(groups)), groups),
        "race_date": pd.date_range("2025-01-01", periods=len(groups)).repeat(5),
        "finish_position": positions, "won": (positions == 1).astype(int),
        "num_runners": 5,
    })
    predictor = _predictor()
    test_X = rng.uniform(0.01, 0.99, (12, 1))
    actual = predictor.train_on_fold(
        X, test_X, frame, groups, np.array([5, 7]), ["signal"], blend_market_odds=False,
    )
    expected_temperature = fit_temperature(X[200:, 0], groups[40:], positions[200:] == 1)
    assert predictor.calibration_temp == expected_temperature
    np.testing.assert_allclose(actual, race_temperature(test_X[:, 0], [5, 7], expected_temperature))
