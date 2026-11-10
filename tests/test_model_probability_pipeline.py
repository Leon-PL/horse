"""Regression coverage for race grouping and calibration/serving parity."""

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

import config
import src.model as model_module
from src.model import RacePredictor, _IdentityScaler


class _ColumnClassifier:
    def __init__(self, column):
        self.column = column

    def predict_proba(self, X):
        p = np.asarray(X)[:, self.column].astype(float)
        return np.column_stack([1.0 - p, p])


class _ConstantCalibration:
    def predict(self, probabilities):
        return np.full(len(probabilities), 0.99)


def _predictor():
    predictor = RacePredictor()
    predictor.feature_cols = ["signal", "place_signal"]
    predictor.scaler = _IdentityScaler()
    predictor.clf_model = _ColumnClassifier(0)
    predictor.place_model = _ColumnClassifier(1)
    predictor.platt_a, predictor.platt_b = 0.8, -0.3
    predictor.place_platt_a, predictor.place_platt_b = 0.9, 0.1
    predictor.win_iso_cal = IsotonicRegression(out_of_bounds="clip").fit(
        [0, 0.1, 0.3, 0.7, 1], [0, 0.08, 0.25, 0.8, 1]
    )
    return predictor


def _races():
    rows = []
    for race, size in [("first", 5), ("second", 8), ("third", 6)]:
        for runner in range(size):
            rows.append({
                "race_id": race, "horse_name": f"{race}-{runner}",
                "signal": (runner + 1) / (size + 2),
                "place_signal": (size - runner) / (size + 1),
                "num_runners": size, "handicap": 0,
                "odds": 2.0 + 2.0 * runner,
            })
    return pd.DataFrame(rows)


def test_missing_model_signal_is_not_silently_replaced_with_zero():
    frame = _races().drop(columns="signal")
    with pytest.raises(ValueError, match="Prediction frame.*missing: signal"):
        _predictor().predict_races(frame)
    assert "signal" not in frame


@pytest.mark.parametrize("market_anchor", [None, {"alpha": 0.3, "beta": 0.8}])
def test_interleaved_batch_matches_individual_races(market_anchor):
    predictor = _predictor()
    predictor.market_anchor = market_anchor
    frame = _races().sample(frac=1, random_state=13)
    assert not frame["race_id"].equals(frame["race_id"].sort_values())
    actual = predictor.predict_races(frame)
    assert list(actual["race_id"].unique()) == list(frame["race_id"].unique())
    np.testing.assert_allclose(actual.groupby("race_id")["win_probability"].sum(), 1)

    numeric = ["win_probability", "place_probability", "predicted_rank",
               "implied_prob", "value_score"]
    for race_id, race in frame.groupby("race_id", sort=False):
        expected = predictor.predict_race(race).set_index("horse_name").sort_index()
        batched = actual[actual["race_id"] == race_id].set_index("horse_name").sort_index()
        np.testing.assert_allclose(batched[numeric], expected[numeric], rtol=1e-12, atol=1e-12)


def test_batch_rejects_missing_race_identity():
    frame = _races()
    frame.loc[0, "race_id"] = None
    with pytest.raises(ValueError, match="race_id for every runner"):
        _predictor().predict_races(frame)


def test_replacing_calibrators_is_independent_of_previous_mapping(monkeypatch):
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 0)
    monkeypatch.setattr(RacePredictor, "_optimise_platt_calibration",
                        staticmethod(lambda *args: (1.0, 0.0)))
    monkeypatch.setattr(RacePredictor, "_optimise_place_platt", lambda *args: (1.0, 0.0))
    monkeypatch.setattr(RacePredictor, "_train_win_classifier", lambda *args, **kwargs: _ColumnClassifier(0))
    monkeypatch.setattr(RacePredictor, "_train_place_classifier", lambda *args, **kwargs: _ColumnClassifier(1))
    rng = np.random.default_rng(42)
    groups = np.full(50, 5)
    features = rng.uniform(0.05, 0.95, (groups.sum(), 2))
    positions = np.concatenate([rng.permutation(np.arange(1, 6)) for _ in groups])
    predictors = [RacePredictor(win_calibration_method="platt_isotonic"),
                  RacePredictor(win_calibration_method="platt_isotonic")]
    predictors[1].win_iso_cal = _ConstantCalibration()
    predictors[1].place_iso_cal = _ConstantCalibration()
    for predictor in predictors:
        assert predictor._fit_training_slab_calibrators(
            features, groups, (positions == 1).astype(int), (positions <= 3).astype(int),
            positions, model_params=None, sample_weight=None,
        )
        assert predictor.win_iso_cal is not None
        assert predictor.place_iso_cal is not None
    for predictor in predictors[1:]:
        np.testing.assert_allclose(
            predictor._calibrate_win_probs(features[:, 0], groups),
            predictors[0]._calibrate_win_probs(features[:, 0], groups),
        )
        np.testing.assert_allclose(
            predictor._calibrate_place_probs(features[:, 1]),
            predictors[0]._calibrate_place_probs(features[:, 1]),
        )


def test_framework_defaults_follow_project_configuration(monkeypatch):
    monkeypatch.setattr(config, "SUB_MODEL_FRAMEWORKS", {"classifier": "lgbm"})
    assert RacePredictor().frameworks == {"classifier": "lgbm"}
    assert RacePredictor(frameworks={"place": "lgbm"}).frameworks == {"classifier": "lgbm", "place": "lgbm"}


@pytest.mark.parametrize("framework", ["xgb", "cat"])
def test_removed_frameworks_are_rejected(framework):
    with pytest.raises(ValueError, match="Unsupported framework"):
        RacePredictor(frameworks={"place": framework})


def test_calibration_slab_purges_dates_and_preserves_soft_targets(monkeypatch):
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 0)
    monkeypatch.setattr(config, "PURGE_DAYS", 3)
    groups = np.full(60, 5)
    dates = pd.date_range("2025-01-01", periods=20).repeat(3)
    features = np.random.default_rng(7).uniform(0.05, 0.95, (groups.sum(), 2))
    positions = np.tile(np.arange(1, 6), len(groups))
    soft_targets = np.linspace(0.02, 0.8, groups.sum())
    captured = {}

    def train_win(self, X, y, **kwargs):
        captured["rows"] = len(X)
        captured["soft_y"] = kwargs["soft_y"]
        return _ColumnClassifier(0)

    monkeypatch.setattr(RacePredictor, "_train_win_classifier", train_win)
    monkeypatch.setattr(RacePredictor, "_train_place_classifier", lambda *args, **kwargs: _ColumnClassifier(1))
    predictor = RacePredictor()
    assert predictor._fit_training_slab_calibrators(
        features, groups, (positions == 1).astype(int), (positions <= 3).astype(int),
        positions, model_params=None, sample_weight=None, race_dates=dates,
        soft_y=soft_targets,
    )
    calibration_start = dates[int(len(groups) * 0.8)]
    train_races = captured["rows"] // 5
    assert dates[train_races - 1] <= calibration_start - pd.Timedelta(days=3)
    np.testing.assert_array_equal(captured["soft_y"], soft_targets[:captured["rows"]])


def test_fold_honours_market_blending_distillation_and_local_fast_params(monkeypatch):
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_DISTILL", True)
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_BLEND", 0.5)
    monkeypatch.setattr(config, "FEATURE_PRUNE_FRACTION", 0)
    frame = _races()
    frame["race_date"] = pd.Timestamp("2025-01-01")
    frame["finish_position"] = frame.groupby("race_id", sort=False).cumcount() + 1
    frame["won"] = (frame["finish_position"] == 1).astype(int)
    frame["bsp"] = frame["odds"]
    groups = frame.groupby("race_id", sort=False).size().to_numpy()
    X = frame[["signal", "place_signal"]].to_numpy()
    captured = {}
    params = {"classifier": {"n_estimators": 12}, "place": {"n_estimators": 18}}
    defaults = dict(config.CLASSIFIER_PARAMS)

    def fit_calibration(self, **kwargs):
        captured["calibration"] = kwargs
        if kwargs["market_odds"] is not None:
            self.market_anchor = {"alpha": 0.0, "beta": 1.0}
        return True

    def fit_win(self, X, y, **kwargs):
        captured["win"] = kwargs
        return _ColumnClassifier(0)

    def fit_place(self, X, y, **kwargs):
        captured["place"] = kwargs
        return _ColumnClassifier(1)

    monkeypatch.setattr(RacePredictor, "_fit_training_slab_calibrators", fit_calibration)
    monkeypatch.setattr(RacePredictor, "_train_win_classifier", fit_win)
    monkeypatch.setattr(RacePredictor, "_train_place_classifier", fit_place)
    predictor = RacePredictor(frameworks={"classifier": "lgbm", "place": "lgbm"})
    win, place = predictor.train_on_fold(
        X, X, frame, groups, groups, ["signal", "place_signal"],
        params=params, fast_fold=True, return_place_probs=True,
        test_df=frame, blend_market_odds=True,
    )
    expected_market, _ = model_module.market_probs_from_odds(frame["odds"].to_numpy(), groups)
    np.testing.assert_allclose(win, expected_market)
    expected_teacher = 0.5 * expected_market + 0.5 * frame["won"].to_numpy()
    np.testing.assert_allclose(captured["win"]["soft_y"], expected_teacher)
    np.testing.assert_array_equal(captured["calibration"]["soft_y"], captured["win"]["soft_y"])
    assert len(captured["calibration"]["race_dates"]) == len(groups)
    assert captured["win"]["params"]["n_estimators"] == 6
    assert captured["place"]["params"]["n_estimators"] == 9
    assert params == {"classifier": {"n_estimators": 12}, "place": {"n_estimators": 18}}
    assert config.CLASSIFIER_PARAMS == defaults
    assert place.shape == win.shape


def test_seed_round_trips_with_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path))
    predictor = _predictor()
    predictor.random_seed = 1729
    predictor.save()
    restored = RacePredictor(random_seed=3)
    restored.load()
    assert restored.random_seed == 1729
    np.testing.assert_allclose(
        restored.predict_races(_races())["win_probability"],
        predictor.predict_races(_races())["win_probability"],
    )


@pytest.mark.parametrize("role", ["classifier", "place"])
def test_model_factory_uses_run_seed(role, monkeypatch):
    class Model:
        def __init__(self, **kwargs):
            self.params = kwargs

        def fit(self, X, y, **kwargs):
            return self

    monkeypatch.setattr(model_module, "_ensure_lightgbm", lambda: None)
    monkeypatch.setattr(model_module, "LGBMClassifier", Model)
    predictor = RacePredictor(frameworks={"classifier": "lgbm", "place": "lgbm"}, random_seed=1729)
    method = predictor._train_win_classifier if role == "classifier" else predictor._train_place_classifier
    # The binary path; the exploded win model's seed is covered in test_exploded_win_model.
    kwargs = {"objective": "cross_entropy"} if role == "classifier" else {}
    fitted = method(np.ones((10, 2)), np.tile([0, 1], 5), **kwargs)
    assert fitted.params["random_state"] == 1729


def test_exchange_settlement_and_targets_never_enter_form_features(monkeypatch):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", None)
    frame = pd.DataFrame({
        "horse_win_rate": [0.1, 0.2], "bsp": [5.0, 8.0], "place_bsp": [2.0, 3.0],
        "placed_bf": [1, 0], "placed": [1, 0], "relevance": [5, 0],
        "ltp_preoff": [4.0, 9.0], "ltp_60s": [4.0, 9.0],
        "ltp_300s": [4.0, 9.0], "ltp_last": [4.0, 9.0], "win_ltp300_raw": [4.0, 9.0],
    })
    assert model_module.get_feature_columns(frame) == ["horse_win_rate"]
    # An allowlist must never override mandatory leakage exclusions.
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", list(frame.columns))
    with pytest.raises(ValueError, match="FEATURE_ALLOWLIST contains excluded"):
        model_module.get_feature_columns(frame)


@pytest.mark.parametrize("explicit_allowlist", [False, True])
def test_future_dependent_sectional_composites_never_enter_model_features(monkeypatch, explicit_allowlist):
    unsafe = [f"{base}{suffix}" for base in ("atr_speed_score", "atr_stride_x_gallop")
              for suffix in ("", "_vs_field", "_z_in_race", "_rank_pct_in_race", "_vs_best_in_race")]
    frame = pd.DataFrame({"horse_win_rate": [.1, .2], "atr_finishing_speed_pct_avg_3": [98., 101.],
                          **{name: [1., 2.] for name in unsafe}})
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", list(frame) if explicit_allowlist else None)
    if explicit_allowlist:
        with pytest.raises(ValueError, match="FEATURE_ALLOWLIST contains excluded"):
            model_module.get_feature_columns(frame)
    else:
        assert model_module.get_feature_columns(frame) == ["horse_win_rate", "atr_finishing_speed_pct_avg_3"]


@pytest.mark.parametrize("contents", ["", "# nothing selected\n", "[]", "[not json]", '[12]'])
def test_invalid_allowlist_file_fails_closed(contents, tmp_path, monkeypatch):
    allowlist = tmp_path / "features.txt"
    allowlist.write_text(contents)
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", str(allowlist))
    with pytest.raises(ValueError, match="FEATURE_ALLOWLIST"):
        model_module.get_feature_columns(pd.DataFrame({"feature": [1, 2]}))


def test_missing_allowlist_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError, match="FEATURE_ALLOWLIST"):
        model_module.get_feature_columns(pd.DataFrame({"feature": [1, 2]}))


@pytest.mark.parametrize("allowlist", [["typo"], [], ["bsp"]])
def test_unmatched_or_empty_allowlist_never_expands_to_all_features(allowlist, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", allowlist)
    with pytest.raises(ValueError, match="FEATURE_ALLOWLIST"):
        model_module.get_feature_columns(pd.DataFrame({"feature": [1, 2], "bsp": [3, 4]}))


def test_allowlist_requires_all_requested_eligible_features(tmp_path, monkeypatch):
    allowlist = tmp_path / "features.txt"
    allowlist.write_text("# selected features\nfeature_b\nmissing_optional_feature\n")
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", allowlist)
    frame = pd.DataFrame({"feature_a": [1, 2], "feature_b": [3, 4]})
    with pytest.raises(ValueError, match="missing: missing_optional_feature"):
        model_module.get_feature_columns(frame)
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", ["feature_b"])
    assert model_module.get_feature_columns(frame) == ["feature_b"]
