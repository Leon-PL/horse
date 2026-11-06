"""Deployment uses the evaluated recipe, recent allowed labels, and no test claims."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src import run_store as registry
from src import training
from src.deployment_validation import training_recipe, write_validation_report
from src.model import RacePredictor


@pytest.fixture
def evaluated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", ["signal", "second_signal"])
    monkeypatch.setattr(config, "BURN_IN_MONTHS", 0)
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 2)
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_DISTILL", False)
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(registry, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(registry, "PRODUCTION_POINTER", str(tmp_path / "production.json"))
    frame = pd.DataFrame([
        {"race_id": f"r-{day}", "race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day),
         "horse_name": f"h-{runner}", "num_runners": 5, "handicap": 0,
         "signal": float(np.sin(day) + runner), "second_signal": float(np.cos(day) * runner), "odds": float(runner + 2),
         "won": int(runner == day % 4),
         "finish_position": (0 if runner == 4 else (1 if runner == day % 4 else 2 + runner)),
         "lengths_behind": float(runner), "track": "Test", "off_time": "14:30"}
        for day in range(150) for runner in range(5)])
    monkeypatch.setattr(training, "process_data", lambda: frame.copy())
    monkeypatch.setattr(training, "get_featured_dataset", lambda processed: frame.copy())
    monkeypatch.setattr(training, "data_signature", lambda df, cutoff: "frozen-data")
    instances = []

    class SingleWorkerPredictor(RacePredictor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._autotune_njobs = 1
            instances.append(self)

    monkeypatch.setattr(training, "RacePredictor", SingleWorkerPredictor)
    cfg = training.TrainingConfig(refresh_data=False, holdout_days=30,
        hyperparams={role: {"n_estimators": 8, "min_child_samples": 3}
                     for role in ("classifier", "place")})
    run_id = training.train_run(cfg)
    certificate = tmp_path / "validation.json"
    write_validation_report(run_id, certificate)
    return cfg, run_id, certificate, frame, instances


def test_native_deployment_refit_and_saved_serving_round_trip(evaluated, monkeypatch):
    cfg, reference, certificate, history, instances = evaluated
    monkeypatch.setattr(training, "evaluate", lambda *a: pytest.fail("Deployment must not invent test scores"))
    cutoff = "2025-05-15"
    deployed = training.train_run(replace(cfg, name="deployment", fit_as_of=cutoff,
                                        validation_report=str(certificate)))
    meta = registry.load_run_meta(deployed)
    tc = meta["training_config"]
    assert meta["metrics"] == meta["test_analysis"] == {}
    assert tc["validation_reference"]["validation_run_id"] == reference
    assert tc["split"]["as_of"] == tc["split"]["training_last_date"] == cutoff
    assert tc["split"]["holdout_races"] == tc["split"]["validation_races"] == 0
    assert tc["split"]["training_runners"] == int(history.race_date.le(cutoff).sum())
    assert not tc["split"]["win_only"]
    assert registry.get_production_run_id() is None
    restored = RacePredictor()
    restored.load(str(Path(registry.RUNS_DIR) / deployed / "model.joblib"))
    test = history.loc[history.race_date.gt(cutoff)].sample(frac=1, random_state=7)
    before, after = instances[-1].predict_races(test), restored.predict_races(test)
    for column in ("win_probability", "place_probability"):
        np.testing.assert_allclose(before[column], after[column], atol=1e-12, rtol=0)
        assert after[column].between(0, 1).all()
    np.testing.assert_allclose(after.groupby("race_id").win_probability.sum(), 1, atol=1e-12)
    assert restored.market_anchor is None


@pytest.mark.parametrize("change", ["seed", "parameters", "features", "distillation"])
def test_changed_recipe_rejected_before_loading_data(evaluated, monkeypatch, change):
    cfg, _, certificate, _, _ = evaluated
    if change == "seed":
        cfg = replace(cfg, seed=43)
    elif change == "parameters":
        cfg = replace(cfg, hyperparams={"classifier": {"n_estimators": 10}})
    elif change == "features":
        monkeypatch.setattr(config, "FEATURE_ALLOWLIST", ["different"])
    else:
        monkeypatch.setattr(config, "WIN_SOFT_LABEL_DISTILL", True)
    monkeypatch.setattr(training, "process_data", lambda **kw: pytest.fail("Must validate before data access"))
    with pytest.raises(ValueError, match="recipe differs"):
        training.train_run(replace(cfg, fit_as_of="2025-05-15", validation_report=str(certificate)))


def test_tampered_validation_evidence_rejected(evaluated):
    cfg, reference, certificate, _, _ = evaluated
    metadata = Path(registry.RUNS_DIR) / reference / "meta.json"
    metadata.write_text(metadata.read_text() + "\n")
    with pytest.raises(ValueError, match="evidence changed"):
        training.train_run(replace(cfg, fit_as_of="2025-05-15", validation_report=str(certificate)))


def test_deployment_rejects_whole_inconsistent_fields_and_retains_dnfs(evaluated):
    cfg, _, certificate, history, _ = evaluated
    history.loc[history.race_id.eq("r-50"), "num_runners"] = 6
    rid = training.train_run(replace(cfg, fit_as_of="2025-05-15", validation_report=str(certificate)))
    split = registry.load_run_meta(rid)["training_config"]["split"]
    assert split["input_field_validation"]["excluded_races"] == 1
    assert split["input_field_validation"]["excluded_rows"] == 5
    assert split["training_runners"] == int(history.race_date.le("2025-05-15").sum()) - 5


def test_deployment_preserves_validated_training_order_when_store_schema_changes(evaluated, monkeypatch):
    cfg, _, certificate, history, instances = evaluated
    # Both inputs exist, but their parquet schema order is reversed.
    reordered = history[["second_signal", *[c for c in history if c != "second_signal"]]]
    monkeypatch.setattr(training, "get_featured_dataset", lambda processed: reordered.copy())
    training.train_run(replace(cfg, fit_as_of="2025-05-15", validation_report=str(certificate)))
    assert instances[-1].feature_cols == ["signal", "second_signal"]


def test_deployment_feature_generation_mismatch_fails_before_data(evaluated, monkeypatch):
    cfg, _, certificate, _, _ = evaluated
    monkeypatch.setattr(training, "fe_signature", lambda: "different feature generation")
    monkeypatch.setattr(training, "process_data", lambda **kw: pytest.fail("No collection or rebuild"))
    with pytest.raises(ValueError, match="features differ"):
        training.train_run(replace(cfg, fit_as_of="2025-05-15", validation_report=str(certificate)))


@pytest.mark.parametrize("cutoff", ["", "bad", "2025-01-01T12:00:00", "2999-01-01"])
def test_invalid_cutoff_fails_before_collection(monkeypatch, cutoff):
    monkeypatch.setattr(training, "process_data", lambda **kw: pytest.fail("No data access"))
    with pytest.raises(ValueError, match="fit_as_of"):
        training.train_run(training.TrainingConfig(fit_as_of=cutoff, validation_report="unused"))


def test_a_refit_without_a_certificate_proceeds_to_the_data(monkeypatch):
    """Experiments may fit through a cutoff; only a certificate that is given is checked."""
    class Reached(Exception):
        pass

    def reached(**kw):
        raise Reached

    import src.deployment_validation as validation
    monkeypatch.setattr(validation, "load_validation_report", lambda *a: pytest.fail("No certificate"))
    monkeypatch.setattr(training, "process_data", reached)
    with pytest.raises(Reached):
        training.train_run(training.TrainingConfig(fit_as_of="2025-05-15", refresh_data=False))


def test_conflicting_modes_fail_before_collection(monkeypatch):
    monkeypatch.setattr(training, "process_data", lambda **kw: pytest.fail("No data access"))
    for cfg, message in [
        (training.TrainingConfig(validation_report="unused"), "requires an explicit"),
        (training.TrainingConfig(fit_as_of="2025-05-15", validation_report="unused", run_walk_forward=True), "separate market"),
        (training.TrainingConfig(fit_as_of="2025-05-15", validation_report="unused", blend_market_odds=True), "separate market"),
    ]:
        with pytest.raises(ValueError, match=message):
            training.train_run(cfg)


def test_deployment_cli_inherits_locked_options(monkeypatch, tmp_path):
    import sys
    import train as cli

    captured = {}
    certificate = tmp_path / "approved.json"
    options = training_recipe(training.TrainingConfig(seed=17, hyperparams={"classifier": {"n_estimators": 11}}))
    certificate.write_text(json.dumps({"recipe": options}))
    monkeypatch.setattr(sys, "argv", ["train.py", "--no-refresh", "--fit-as-of", "2025-05-15",
                                      "--validation-report", str(certificate)])
    monkeypatch.setattr(cli, "train_run", lambda cfg: captured.update(cfg=cfg) or "run")
    monkeypatch.setattr(cli, "_print_summary", lambda rid: None)
    cli.main()
    assert captured["cfg"].fit_as_of == "2025-05-15"
    assert captured["cfg"].validation_report == str(certificate)
    assert captured["cfg"].seed == 17
    assert captured["cfg"].hyperparams == {"classifier": {"n_estimators": 11}}
    monkeypatch.setattr(sys, "argv", ["train.py", "--fit-as-of", "2025-05-15", "--holdout-days", "30"])
    with pytest.raises(SystemExit):
        cli.main()


def test_cli_refit_without_a_certificate_uses_the_given_settings(monkeypatch):
    import sys
    import train as cli

    captured = {}
    monkeypatch.setattr(sys, "argv", ["train.py", "--no-refresh", "--fit-as-of", "2025-05-15",
                                      "--window-days", "400"])
    monkeypatch.setattr(cli, "train_run", lambda cfg: captured.update(cfg=cfg) or "run")
    monkeypatch.setattr(cli, "_print_summary", lambda rid: None)
    cli.main()
    assert captured["cfg"].fit_as_of == "2025-05-15"
    assert captured["cfg"].validation_report is None
    assert captured["cfg"].train_window_days == 400
    monkeypatch.setattr(sys, "argv", ["train.py", "--validation-report", "x.json"])
    with pytest.raises(SystemExit):
        cli.main()
