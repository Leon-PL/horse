"""A teacher-only refit can reuse a place head only on identical model inputs."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_teacher_refresh_waits_for_the_corrected_parent_install(tmp_path, monkeypatch):
    import json
    from scripts import refit_bsp_teacher as refresh
    monkeypatch.setattr(refresh, "CORRECTIONS", tmp_path)
    (tmp_path / "deployment_status.json").write_text(json.dumps({"stage": "normal_full_refit"}))
    with pytest.raises(ValueError, match="Finish and verify"):
        refresh.run()


def test_reused_place_matches_a_complete_native_refit_after_teacher_changes(tmp_path, monkeypatch):
    import config
    from src.model import RacePredictor
    from scripts.refit_bsp_teacher import TeacherRefreshPredictor
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", ["signal", "second_signal"])
    monkeypatch.setattr(config, "BURN_IN_MONTHS", 0)
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 2)
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_DISTILL", True)
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_BLEND", .5)
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path / "models"))
    rng = np.random.default_rng(27)
    rows = []
    for day, date in enumerate(pd.date_range("2025-01-01", periods=180)):
        for horse in range(6):
            finish = 1+(horse-day%6)%6
            rows.append({"race_id": day, "race_date": date, "horse_name": str(horse),
                "num_runners": 6, "handicap": 0, "won": int(finish==1), "finish_position": finish,
                "lengths_behind": float(finish-1), "odds": 2.+horse, "bsp": 2.+horse,
                "signal": horse+rng.normal(0,.2), "second_signal": rng.normal()})
    before = pd.DataFrame(rows)
    after = before.copy()
    after["bsp"] = 7.-after.horse_name.astype(int)
    cutoff = str(before.race_date.max().date())
    params = {role: {"n_estimators": 8, "max_depth": 2, "min_child_samples": 3} for role in ("classifier", "place")}
    parent = RacePredictor(); parent._autotune_njobs = 1
    parent.fit_as_of(before, cutoff, params=params)
    parent_path = tmp_path / "parent.joblib"
    parent.save(str(parent_path))
    complete = RacePredictor(); complete._autotune_njobs = 1
    complete.fit_as_of(after, cutoff, params=params)
    reused = TeacherRefreshPredictor(parent_path=parent_path, parent_origin={"run_id": "parent"},
        checkpoint_directory=tmp_path / "components", checkpoint_manifest="teacher-data-update")
    reused._autotune_njobs = 1
    # A win-only update must not silently fit any place learner again.
    monkeypatch.setattr(reused, "_train_place_classifier", lambda *a, **k: pytest.fail("Place fit repeated"))
    reused.fit_as_of(after, cutoff, params=params)
    _, matrix = complete._prepare_prediction_frame(after)
    assert not np.allclose(parent.clf_model.predict_proba(matrix), complete.clf_model.predict_proba(matrix))
    np.testing.assert_array_equal(parent.place_model.predict_proba(matrix), complete.place_model.predict_proba(matrix))
    actual = reused.predict_races(after).sort_values(["race_id", "horse_name"])
    expected = complete.predict_races(after).sort_values(["race_id", "horse_name"])
    for column in ("win_probability", "place_probability"):
        np.testing.assert_array_equal(actual[column], expected[column])
    assert reused.eval_split_info["reused_place_component"]["run_id"] == "parent"


def test_place_reuse_refuses_changed_dates_or_scaling():
    from sklearn.preprocessing import StandardScaler
    from scripts.refit_bsp_teacher import reuse_place
    parent = SimpleNamespace(feature_cols=["a", "b"], frameworks={"classifier": "lgbm", "place": "lgbm"},
        scaler=StandardScaler().fit([[1, 2], [3, 6]]), place_model=object(),
        place_platt_a=1., place_platt_b=0., place_iso_cal=None,
        eval_split_info={"as_of": "2025-01-01", "training_runners": 2, "training_races": 1,
            "training_first_date": "2025-01-01", "training_last_date": "2025-01-01",
            "calibration": {"last_date": "2025-01-01"}, "refit_rounds": {"place": 8}})
    target = deepcopy(parent)
    target.eval_split_info["as_of"] = "2025-01-02"
    with pytest.raises(ValueError, match="scope"):
        reuse_place(target, parent, {})
    target = deepcopy(parent)
    target.scaler.scale_[0] += .1
    with pytest.raises(AssertionError, match="preprocessing"):
        reuse_place(target, parent, {})
