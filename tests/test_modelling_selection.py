"""Selection uses the serving learner, preserves precision, and confirms later."""
import numpy as np
import pandas as pd
import pytest

from src.modelling.spec import ModelSpec
from src.modelling.selection import select_features, matrix_signature
from src.modelling.win import WinModel
from src.modelling.text import TextExpert


def data():
    rng = np.random.default_rng(19)
    race = np.repeat(np.arange(290), 6)
    signal, noise = rng.normal(size=(2, len(race)))
    strength = pd.Series(signal + rng.gumbel(size=len(race)))
    finish = strength.groupby(race).rank(ascending=False).astype(int).to_numpy()
    frame = pd.DataFrame({"race_id": [f"r{r:04d}" for r in race], "horse_id": np.tile(np.arange(6), 290),
        "horse_name": np.tile([f"Horse {i}" for i in range(6)], 290), "num_runners": 6,
        "race_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(race, unit="D"),
        "finish_position": finish, "won": (finish == 1).astype(int), "odds": 7., "bsp": 6.})
    # Late availability must not influence hygiene chosen on earlier dates.
    late = (frame.race_date >= "2023-07-01").to_numpy(float)
    matrix = np.column_stack([signal, noise, signal.copy(), late])
    names = ["signal", "noise", "duplicate_signal", "late_only"]
    spec = ModelSpec("generalist", ("signal", "noise"), burn_in_months=0,
        params=(("min_child_samples", 10), ("num_leaves", 7)),
        screen_params=(("n_estimators", 12), ("learning_rate", .2)))
    return frame, matrix, names, spec


def test_selection_confirms_later_resumes_and_does_not_mutate_defaults(tmp_path, monkeypatch):
    frame, matrix, names, spec = data()
    import config
    configured = config.FEATURE_ALLOWLIST
    result = select_features(frame, matrix, names, spec, selection_end="2023-06-30",
        confirmation_start="2023-07-08", n_min=1, min_train_periods=1,
        capacity="screen", output=tmp_path)
    assert result["hygiene_rejections"]["duplicate_signal"] == "duplicate of signal"
    assert result["hygiene_rejections"]["late_only"] == "constant"
    assert result["confirmation"]["baseline"]["races"] > 0
    assert config.FEATURE_ALLOWLIST == configured
    assert spec.features == ("signal", "noise")
    records = pd.read_parquet(tmp_path / "confirmation_selected.parquet")
    assert records.race_date.min() >= pd.Timestamp("2023-07-08")
    assert (tmp_path / "selected_features.txt").read_text().splitlines() == result["selected_spec"]["features"]
    def refuse(*args, **kwargs):
        raise AssertionError("completed selection was refitted")
    monkeypatch.setattr(WinModel, "fit", refuse)
    again = select_features(frame, matrix, names, spec, selection_end="2023-06-30",
        confirmation_start="2023-07-08", n_min=1, min_train_periods=1,
        capacity="screen", output=tmp_path)
    assert result == again
    changed = matrix.copy()
    changed[0, 0] += 1
    with pytest.raises(ValueError, match="registration"):
        select_features(frame, changed, names, spec, selection_end="2023-06-30",
            confirmation_start="2023-07-08", n_min=1, min_train_periods=1, capacity="screen", output=tmp_path)


def test_selection_refuses_holdout_and_overlapping_periods(tmp_path):
    frame, matrix, names, spec = data()
    with pytest.raises(ValueError, match="gap"):
        select_features(frame, matrix, names, spec, selection_end="2023-06-30", confirmation_start="2023-07-01")
    frame.race_date += pd.DateOffset(years=4)
    with pytest.raises(ValueError, match="holdout"):
        select_features(frame, matrix, names, spec, selection_end="2027-06-30", confirmation_start="2027-07-08")


def test_matrix_signature_preserves_precision_and_order():
    matrix = np.array([[1e10, 1e10 + 1], [2., 3.]], dtype=np.float64)
    original = matrix_signature(matrix, ["a", "b"])
    assert original != matrix_signature(matrix.astype(np.float32), ["a", "b"])
    assert original != matrix_signature(matrix, ["b", "a"])
    from src.feature_selection import hygiene_filter
    kept, _ = hygiene_filter(matrix, ["a", "b"], dtype=np.float64)
    assert kept == ["a", "b"]


def test_explicit_text_expert_never_trains_at_prediction_and_masks_absent_races(monkeypatch):
    frame, _, _, _ = data()
    frame = frame.iloc[:240].copy()
    frame["tf_comment"] = np.where(frame.won, "progressing well", "needs to improve")
    frame["tf_verdict"] = "Horse 0 may go well."
    expert = TextExpert(min_df=5).fit(frame)
    assert expert.lineage["training_last_date"] == str(frame.race_date.max().date())
    from src import rtv_text
    def refuse(*args, **kwargs):
        raise AssertionError("prediction tried fitting a text model")
    monkeypatch.setattr(rtv_text, "fit_softmax", refuse)
    predicted = expert.predict(frame)
    np.testing.assert_allclose(pd.Series(predicted).groupby(frame.race_id).sum(), 1)
    empty = frame.copy()
    empty[["tf_comment", "tf_verdict"]] = None
    assert np.isnan(expert.predict(empty)).all()
    assert np.isnan(expert.predict(empty.drop(columns=["tf_comment", "tf_verdict"]))).all()


def test_model_feature_catalog_distinguishes_learned_text_and_prior_market():
    from src.modelling.catalog import feature_metadata
    assert feature_metadata("horse_prior_market_logrel_3_vs_field")["family"] == "market"
    assert feature_metadata("rtv_text_prob")["kind"] == "learned_prediction"
    assert feature_metadata("jt_win_rate")["family"] == "connections"
    assert feature_metadata("unclassified")["individual_dependencies"] is None
