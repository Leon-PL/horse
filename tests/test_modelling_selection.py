"""Selection uses the serving learner, preserves precision, and confirms later."""
import numpy as np
import pandas as pd
import pytest

from src.modelling.spec import ModelSpec
from src.modelling.selection import select_features, matrix_signature
from src.modelling.win import WinModel


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




def test_model_feature_catalog_distinguishes_learned_text_and_prior_market():
    from src.modelling.catalog import feature_metadata
    assert feature_metadata("horse_prior_market_logrel_3_vs_field")["family"] == "market"
    assert feature_metadata("rtv_text_prob")["kind"] == "learned_prediction"
    assert feature_metadata("jt_win_rate")["family"] == "connections"
    assert feature_metadata("unclassified")["individual_dependencies"] is None


def wide_data():
    """Six races a day for 290 days: two real signals, near-copies of the first, and noise."""
    rng = np.random.default_rng(23)
    race = np.repeat(np.arange(290 * 3), 6)
    first, second = rng.normal(size=(2, len(race)))
    noise = rng.normal(size=(4, len(race)))
    strength = pd.Series(1.4 * first + .9 * second + rng.gumbel(size=len(race)))
    finish = strength.groupby(race).rank(ascending=False).astype(int).to_numpy()
    frame = pd.DataFrame({"race_id": [f"r{r:04d}" for r in race], "horse_id": np.tile(np.arange(6), 290 * 3),
        "horse_name": np.tile([f"Horse {i}" for i in range(6)], 290 * 3), "num_runners": 6,
        "race_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(race // 3, unit="D"),
        "finish_position": finish, "won": (finish == 1).astype(int), "odds": 7., "bsp": 6.})
    matrix = np.column_stack([first, second, np.exp(first), first + rng.normal(scale=.02, size=len(race)), *noise])
    names = ["first", "second", "first_exp", "first_again", "n0", "n1", "n2", "n3"]
    spec = ModelSpec("generalist", ("first", "n0"), burn_in_months=0,
        params=(("min_child_samples", 10), ("num_leaves", 7)),
        screen_params=(("n_estimators", 25), ("learning_rate", .2)))
    return frame, matrix, names, spec


def test_selection_drops_near_copies_and_ranks_by_loss_prevented(tmp_path):
    frame, matrix, names, spec = wide_data()
    result = select_features(frame, matrix, names, spec, selection_end="2023-06-30",
        confirmation_start="2023-07-08", n_min=2, min_train_periods=1, capacity="screen",
        importance="permutation", confirm_capacity="native", output=tmp_path)
    # The two rewritten copies of the first signal are dropped before any fit, in favour of the baseline input.
    assert set(result["near_copy_rejections"]) == {"first_exp", "first_again"}
    assert all("near-copy of first " in reason for reason in result["near_copy_rejections"].values())
    # Shuffling a real signal costs loss; elimination to two inputs keeps exactly the two signals.
    import json
    last = sorted(tmp_path.glob("selection_*.json"))[-1]
    step = json.loads(last.read_text())
    assert step["ranking"] == "permutation" and sorted(step["features"]) == ["first", "second"]
    assert set(result["selected_spec"]["features"]) >= {"first", "second"}
    # The second capacity confirms the baseline and selected lists only.
    assert set(result["confirmation_native"]) == {"baseline", "selected"}
    assert result["confirmation_native"]["selected"]["paired_vs_baseline"]["mean"] < 0
    assert "broad" in result["confirmation"]


def test_near_copy_filter_can_be_turned_off(tmp_path):
    frame, matrix, names, spec = wide_data()
    result = select_features(frame, matrix, names, spec, selection_end="2023-06-30",
        confirmation_start="2023-07-08", n_min=6, min_train_periods=1, capacity="screen", near_copy=None)
    assert result["near_copy_rejections"] == {} and result["confirmation"]["broad"]["features"] == 8


def test_admission_separates_information_from_extra_inputs(tmp_path):
    from src.modelling.selection import admit_features
    frame, matrix, names, spec = wide_data()
    good = admit_features(frame, matrix, names, spec, ["second"], start="2023-07-08", min_train_periods=1,
                          output=tmp_path)
    assert good["with_candidates_vs_twin"]["mean"] < -0.02 and good["with_candidates_vs_baseline"]["mean"] < -0.02
    assert (tmp_path / "admission.json").exists()
    noise = admit_features(frame, matrix, names, spec, ["n1", "n2"], start="2023-07-08", min_train_periods=1)
    low, high = noise["with_candidates_vs_twin"]["ci95"]
    assert low < 0 < high                                  # noise is no better than its shuffled twin
    assert good["with_candidates_vs_twin"]["ci95"][1] < 0  # a real signal is
    with pytest.raises(ValueError, match="new inputs"):
        admit_features(frame, matrix, names, spec, ["first"], start="2023-07-08", min_train_periods=1)


def test_admission_refuses_candidates_that_are_context_columns():
    from src.modelling.selection import admit_features
    frame, matrix, names, spec = wide_data()
    matrix = np.column_stack([matrix, frame.num_runners.to_numpy(float)])
    with pytest.raises(ValueError, match="context columns"):
        admit_features(frame, matrix, [*names, "num_runners"], spec, ["num_runners"], start="2023-07-08",
                       min_train_periods=1)
