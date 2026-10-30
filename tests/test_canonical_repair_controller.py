"""Exercise native fits, component restart and the recipe certificate on a tiny fixture."""
import json
import numpy as np
import pandas as pd
import pytest


def test_controller_resumes_fits_and_certifies_the_saved_native_recipe(tmp_path, monkeypatch):
    import config
    from scripts import integrate_canonical_repairs as controller
    from src import run_store
    from src.deployment_validation import training_recipe, load_validation_report
    from src.featured_dataset import fe_signature
    from src.model import RacePredictor
    from src.training import TrainingConfig

    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", ["signal"])
    monkeypatch.setattr(config, "BURN_IN_MONTHS", 0)
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 2)
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_DISTILL", True)
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(controller, "OUTPUT", tmp_path)
    # Source/manifest verification has its own checks; here failure injection
    # isolates the save/certificate/restart path without private project data.
    monkeypatch.setattr(controller, "verify", lambda *args: None)
    rng = np.random.default_rng(26)
    dates = pd.date_range("2025-01-01", "2026-03-31")
    rows = []
    for index, day in enumerate(dates):
        for horse in range(6):
            rows.append({"race_id": index, "race_date": day, "horse_name": str(horse),
                "finish_position": horse+1, "won": int(horse==0), "num_runners": 6, "handicap": 0,
                "odds": 6., "bsp": 6., "lengths_behind": float(horse), "signal": rng.normal()})
    frame = pd.DataFrame(rows)
    frame.to_parquet(tmp_path / "corrected_inputs.parquet", index=False)
    test = frame.loc[frame.race_date.between("2026-01-01", "2026-03-31")].copy()
    keys = ["race_id", "horse_name", "race_date", "won", "finish_position", "num_runners", "handicap"]
    test[keys].assign(win_probability=1/6, place_probability=1/3).to_parquet(tmp_path / "baseline_2026Q1_predictions.parquet", index=False)
    cfg = TrainingConfig(refresh_data=False, threads=2,
        hyperparams={"classifier": {"n_estimators": 8, "max_depth": 2}, "place": {"n_estimators": 8, "max_depth": 2}})
    from scripts.canonical_training_cache import CheckpointPredictor
    scope = "fixture-native-estimator-scope"
    manifest = {"historical_cutoff": "2025-12-22", "recipe": training_recipe(cfg),
        "historical_component_cache_scope":scope,
        "fe_signature": fe_signature(), "adoption_policy": "test mandatory correctness"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "source_snapshot.zip").write_bytes(b"fixture archive fingerprint")
    # A preceding win-only check should supply all win estimators; adding the
    # place head must not fit them again, including its calibration component.
    prior = CheckpointPredictor(checkpoint_directory=tmp_path / "components/historical", checkpoint_manifest=scope,
        random_seed=cfg.seed, frameworks=cfg.frameworks, win_calibration_method=cfg.win_calibration_method)
    prior._autotune_njobs = 2
    prior.fit_as_of(frame, "2025-12-22", params=cfg.hyperparams, dataset_start=frame.race_date.min(), win_only=True)
    original_win = RacePredictor._train_win_classifier
    recent_win_fits = []
    def only_new_recent_win(self, matrix, labels, **kwargs):
        recent_win_fits.append(len(matrix))
        assert len(recent_win_fits) == 1, "Previously cached win estimators must be reused"
        return original_win(self,matrix,labels,**kwargs)
    #038 needs one different earlier win head for honest place-calibration
    #floors. The three existing win components must still be exact cache hits.
    monkeypatch.setattr(RacePredictor, "_train_win_classifier", only_new_recent_win)
    original_report = controller.write_validation_report
    def interrupted(*args, **kwargs):
        raise OSError("interrupted certificate publication")
    monkeypatch.setattr(controller, "write_validation_report", interrupted)
    with pytest.raises(OSError, match="certificate publication"):
        controller.qualify(manifest, cfg)
    assert (tmp_path / "historical_model.joblib").exists()
    assert list((tmp_path / "components/historical").glob("*.joblib"))
    assert len(recent_win_fits) == 1
    # Force the restart to reconstruct from exact component checkpoints rather
    # than the whole predictor. No native learner is allowed to train again.
    (tmp_path / "historical_model.joblib").unlink()
    monkeypatch.setattr(RacePredictor, "_train_win_classifier", lambda *a, **k: pytest.fail("Win fit repeated"))
    monkeypatch.setattr(RacePredictor, "_train_place_classifier", lambda *a, **k: pytest.fail("Place fit repeated"))
    monkeypatch.setattr(controller, "write_validation_report", original_report)
    reference = controller.qualify(manifest, cfg)
    assert reference["features"] == ["signal"]
    assert load_validation_report(tmp_path / "validation.json", training_recipe(cfg))["validation_run_id"] == reference["validation_run_id"]


def test_cache_transfer_rejects_changed_settings_even_with_a_matching_hash():
    from scripts import integrate_canonical_repairs as controller
    from scripts.benchmark_independent_features import canonical_hash
    from src.training import TrainingConfig
    cfg=TrainingConfig(refresh_data=False,threads=2)
    semantics=controller.native_component_semantics(cfg)
    status={"historical_component_cache_scope":canonical_hash(semantics),"historical_component_cache_semantics":semantics}
    assert controller.verified_historical_cache_scope(status,cfg)==status["historical_component_cache_scope"]
    semantics["seed"]+=1
    status["historical_component_cache_scope"]=canonical_hash(semantics)
    with pytest.raises(ValueError,match="different learner"):
        controller.verified_historical_cache_scope(status,cfg)
    assert controller.verified_historical_cache_scope({},cfg) is None
