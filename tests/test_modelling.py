"""Chronology, explicit recipes and whole-field behaviour of the new win path."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

import config
from src.modelling.spec import ModelSpec
from src.modelling.win import WinModel
from src.modelling.pool import LogPool
from src.modelling.data import ordered_frame, numeric_matrix
from src.modelling.evaluation import component_oof, nested_pool_evaluation
from src.modelling.candidate import WinCandidate, save_candidate
from src.wf_eval import Fold, oof_predict_expanding


def fields(n=160, runners=6):
    rng = np.random.default_rng(91)
    race = np.repeat(np.arange(n), runners)
    x = rng.normal(size=len(race))
    strength = x + rng.gumbel(size=len(x))
    rank = pd.Series(strength).groupby(race).rank(ascending=False).astype(int).to_numpy()
    return pd.DataFrame({"race_id": [f"r{r:04d}" for r in race], "horse_id": np.tile(np.arange(runners), n),
        "horse_name": np.tile([f"Horse {i}" for i in range(runners)], n),
        "race_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(race, unit="D"),
        "num_runners": runners, "won": (rank == 1).astype(int), "finish_position": np.where(rank == runners, 0, rank),
        "lengths_behind": (rank - 1).astype(float), "race_type": "Flat", "handicap": 0,
        "odds": np.full(len(x), runners + 1.), "bsp": np.full(len(x), runners),
        "signal": x, "noise": rng.normal(size=len(x))})


def recipe(**kwargs):
    spec = ModelSpec("generalist", ("signal", "noise"), burn_in_months=0, early_stopping_rounds=5,
        params=tuple(sorted({"n_estimators": 35, "learning_rate": .1, "num_leaves": 7,
            "min_child_samples": 15, "colsample_bytree": 1., "subsample": 1.}.items())),
        screen_params=(("n_estimators", 20), ("learning_rate", .1)))
    return replace(spec, **kwargs)


def forward_folds(frame):
    day = (frame.race_date - frame.race_date.min()).dt.days.to_numpy()
    return [Fold("early", day < 90, (day >= 100) & (day < 120), 10 * 6),
            Fold("late", day < 120, (day >= 130) & (day < 150), 10 * 6)]


def test_specs_resolve_defaults_and_are_immutable(monkeypatch):
    spec = recipe()
    assert ModelSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(FrozenInstanceError):
        spec.seed = 123
    monkeypatch.setattr(config, "CLASSIFIER_PARAMS", {"n_estimators": 9999})
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_BLEND", 0.)
    assert dict(spec.params)["n_estimators"] == 35
    assert spec.soft_label_blend == .5
    assert spec.signature != spec.subset(("signal",)).signature
    with pytest.raises(ValueError, match="excluded"):
        recipe(features=("won",))
    with pytest.raises(ValueError, match="unique"):
        recipe(features=("signal", "signal"))
    with pytest.raises(ValueError):
        recipe(params=(("learning_rate", float("nan")),))


def test_explicit_folds_never_split_a_date_and_fail_before_fitting():
    frame = fields()
    calls = []
    bad = forward_folds(frame)
    test = bad[-1].test.copy()
    test[np.flatnonzero(test)[0]] = False
    bad[-1] = replace(bad[-1], test=test)
    with pytest.raises(ValueError, match="split"):
        oof_predict_expanding(frame, folds=bad, min_train_rows=1,
            fit_predict=lambda tr, te: calls.append(te) or np.full(len(te), 1 / 6))
    assert not calls
    observed = []
    def predict(train, test):
        assert train.race_date.max() <= test.race_date.min() - pd.Timedelta(days=7)
        assert not set(train.race_id) & set(test.race_id)
        assert test.groupby("race_id").size().eq(6).all()
        observed.extend(test.index)
        return np.full(len(test), 1 / 6)
    p = oof_predict_expanding(frame, folds=forward_folds(frame), min_train_rows=1, fit_predict=predict)
    assert np.isfinite(p).sum() == len(observed)
    # The default block splitter also keeps every date/race intact.
    oof_predict_expanding(frame, n_splits=7, min_train_rows=1, fit_predict=predict)


def test_fold_purge_and_duplicate_tests_are_rejected():
    frame = fields()
    fold = forward_folds(frame)[0]
    with pytest.raises(ValueError, match="purge"):
        oof_predict_expanding(frame, folds=[replace(fold, train=frame.race_date.lt(frame.loc[fold.test, "race_date"].min()).to_numpy())],
            fit_predict=lambda tr, te: np.zeros(len(te)), min_train_rows=1)
    with pytest.raises(ValueError, match="only one"):
        oof_predict_expanding(frame, folds=[fold, replace(fold, label="duplicate")],
            fit_predict=lambda tr, te: np.zeros(len(te)), min_train_rows=1)


@pytest.mark.parametrize("objective", ["exploded", "cross_entropy"])
def test_models_keep_recipes_separate_and_restore_prediction_order(objective):
    frame = fields()
    original = config.FEATURE_ALLOWLIST
    spec = recipe(objective=objective)
    model = WinModel(spec).fit(frame, capacity="screen")
    other = WinModel(recipe(id="signal-only", features=("signal",), soft_label_blend=0.)).fit(frame, capacity="screen")
    assert model.spec.features == ("signal", "noise") and other.spec.features == ("signal",)
    assert config.FEATURE_ALLOWLIST == original
    assert model.lineage["calibration_fit_boundary"] < model.lineage["calibration_first_date"]
    assert model.lineage["training_runners"] == len(frame)  # DNFs remain in training.
    expected = model.predict(frame)
    shuffled = frame.sample(frac=1, random_state=9)
    actual = model.predict(shuffled)
    np.testing.assert_allclose(actual, expected[shuffled.index], atol=1e-12)
    assert pd.Series(actual).groupby(shuffled.race_id.to_numpy()).sum().sub(1).abs().max() < 1e-10
    with pytest.raises(ValueError, match="complete"):
        model.predict(frame.iloc[1:])
    with pytest.raises(ValueError, match="required|missing|Missing"):
        model.predict(frame.drop(columns="signal"))


def test_native_model_matches_legacy_fit_recipe(monkeypatch):
    from src.model import RacePredictor
    frame = fields(200)
    monkeypatch.setattr(config, "CLASSIFIER_PARAMS", {**config.CLASSIFIER_PARAMS, "n_estimators": 45, "learning_rate": .1})
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 5)
    monkeypatch.setattr(config, "BURN_IN_MONTHS", 0)
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", ["signal", "noise"])
    spec = ModelSpec("generalist", ("signal", "noise"))
    modern = WinModel(spec).fit(frame, capacity="native")
    legacy = RacePredictor()
    legacy._autotune_njobs = 1
    legacy.fit_as_of(frame, frame.race_date.max(), win_only=True)
    x = frame[list(spec.features)].to_numpy()
    np.testing.assert_allclose(modern.estimator.booster_.predict(x, raw_score=True),
        legacy.clf_model.booster_.predict(x, raw_score=True), atol=1e-9)
    assert modern.temperature == pytest.approx(legacy.calibration_temp, abs=1e-8)


def test_pool_identity_fallback_and_invalid_distributions():
    frame = fields(3)
    p = np.tile(np.array([.4, .2, .15, .1, .1, .05]), 3)
    single = LogPool(("generalist",)).fit(p[:, None], frame.won, frame.race_id)
    np.testing.assert_array_equal(single.predict(p[:, None], frame.race_id), p)
    experts = np.column_stack([p, p[::-1]])
    experts[:6, 1] = np.nan
    experts[6, 1] = np.nan  # partial coverage masks the whole expert field.
    pool = LogPool(("generalist", "expert"), weights=(0., 1.))
    result = pool.predict(experts, frame.race_id)
    np.testing.assert_array_equal(result[:12], p[:12])
    np.testing.assert_array_equal(result[12:], experts[12:, 1])
    experts[12:, 1] = .5
    with pytest.raises(ValueError, match="sum"):
        pool.predict(experts, frame.race_id)


def test_pool_learns_complementary_forecasts_and_supports_interleaved_rows():
    frame = fields(80)
    base = np.full(len(frame), 1 / 6)
    expert = np.where(frame.won, .7, .3 / 5)
    matrix = np.column_stack([base, expert])
    pool = LogPool(("generalist", "expert")).fit(matrix, frame.won, frame.race_id)
    assert pool.weights[1] > .8
    p = pool.predict(matrix, frame.race_id)
    shuffled = frame.sample(frac=1, random_state=8)
    np.testing.assert_allclose(pool.predict(matrix[shuffled.index], shuffled.race_id), p[shuffled.index])
    invalid = np.ones_like(matrix, dtype=bool)
    invalid[0, 1] = False
    with pytest.raises(ValueError, match="whole races"):
        pool.predict(matrix, frame.race_id, invalid)


def test_oof_cache_is_keyed_and_single_component_stack_is_identity(tmp_path, monkeypatch):
    frame = fields()
    spec = recipe()
    records, models = component_oof(frame, [spec], folds=forward_folds(frame), capacity="screen", cache_dir=tmp_path)
    assert records.groupby(["component", "race_id"]).size().eq(6).all()
    assert (pd.to_datetime(records.training_last_date) < records.race_date).all()
    def refuse(*args, **kwargs):
        raise AssertionError("valid cached fold was refitted")
    monkeypatch.setattr(WinModel, "fit", refuse)
    cached, _ = component_oof(frame, [spec], folds=forward_folds(frame), capacity="screen", cache_dir=tmp_path)
    pd.testing.assert_frame_equal(records, cached)
    covered = frame.loc[frame.race_id.isin(records.race_id)].reset_index(drop=True)
    dates = covered.race_date
    fold = Fold("meta", dates.lt("2023-05-01").to_numpy(), dates.ge("2023-05-11").to_numpy(), 0)
    report, predicted, _ = nested_pool_evaluation(covered, records, (spec.id,), folds=[fold])
    assert report["pool"]["race_log_loss"] == report["baseline"]["race_log_loss"]
    np.testing.assert_array_equal(predicted.pool_probability, predicted._component_0)
    candidate = WinCandidate(models, LogPool((spec.id,)), {"test": True})
    path = tmp_path / "candidate.joblib"
    candidate.save(path)
    np.testing.assert_array_equal(WinCandidate.load(path).predict(frame), candidate.predict(frame))
    file = next(tmp_path.glob("*/predictions.parquet"))
    file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mismatch"):
        component_oof(frame, [spec], folds=forward_folds(frame), capacity="screen", cache_dir=tmp_path)


def test_field_validation_rejects_ambiguous_identities_and_infinities():
    frame = fields(3)
    malformed = frame.copy()
    malformed.loc[0, "horse_id"] = malformed.loc[1, "horse_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        ordered_frame(malformed)
    frame.loc[0, "signal"] = np.inf
    with pytest.raises(ValueError, match="infinities"):
        numeric_matrix(frame, ("signal",))


def test_candidate_registry_round_trip_cannot_be_promoted(tmp_path, monkeypatch):
    import src.run_store as runs
    frame = fields()
    model = WinModel(recipe()).fit(frame, capacity="screen")
    candidate = WinCandidate({"generalist": model}, LogPool(("generalist",)), {"as_of": "2023-06-09"})
    monkeypatch.setattr(runs, "RUNS_DIR", str(tmp_path / "runs"))
    pointer = tmp_path / "production.json"
    monkeypatch.setattr(runs, "PRODUCTION_POINTER", str(pointer))
    receipt = {"fe_signature": "current", "data_signature": "current", "dataset_manifest": {"id": "fixture"}}
    run = save_candidate(candidate, receipt)
    np.testing.assert_array_equal(runs.load_run_predictor(run).predict(frame), candidate.predict(frame))
    assert not runs.promote(run, allow_feature_mismatch=True)
    assert not pointer.exists()
