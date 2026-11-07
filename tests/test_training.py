"""Phase-1 training-core tests: CLV, the production pointer, lean run saves,
and train_run orchestration (mocked — no heavy pipeline)."""

import os

import joblib
import numpy as np
import pandas as pd
import pytest

import config
import src.run_store as rs
import src.training as tr


@pytest.mark.parametrize("accepted", [False, True])
def test_training_cli_reports_promotion_result_and_calendar_setting(monkeypatch, capsys, accepted):
    import sys
    import train as cli

    captured = {}
    monkeypatch.setattr(sys, "argv", ["train.py", "--no-refresh", "--holdout-days", "45", "--promote"])
    monkeypatch.setattr(cli, "train_run", lambda cfg: captured.update(config=cfg) or "saved-run")
    monkeypatch.setattr(cli, "_print_summary", lambda run_id: None)
    monkeypatch.setattr(cli, "promote", lambda run_id: accepted)
    if accepted:
        cli.main()
        assert "Promoted saved-run" in capsys.readouterr().out
    else:
        with pytest.raises(SystemExit, match="Promotion refused"):
            cli.main()
        assert "Promoted" not in capsys.readouterr().out
    assert captured["config"].holdout_days == 45


# ── CLV vs BSP ────────────────────────────────────────────────────────
def test_compute_clv_basic(publish_prices):
    from src.clv import compute_clv

    publish_prices(["Alpha", "Bravo"], [4.0, 2.0])

    bets = pd.DataFrame({
        "race_date": ["2025-01-01", "2025-01-01"],
        "track": ["Ascot", "Ascot"],
        "off_time": ["14:30", "14:30"],
        "horse_name": ["Alpha", "Bravo"],
        "odds": [5.0, 2.0],          # Alpha beat the close (5>4); Bravo matched (2==2)
        "strategy": ["value", "value"],
    })
    out = compute_clv(bets)
    assert out["n_matched"] == 2
    assert out["coverage"] == 1.0
    # mean CLV = mean(5/4, 2/2) = mean(1.25, 1.0) = 1.125
    assert out["overall"]["mean_clv"] == pytest.approx(1.125)
    assert out["overall"]["pct_beat_close"] == pytest.approx(0.5)


def test_compute_clv_without_prices_in_the_dataset():
    from src.clv import compute_clv
    bets = pd.DataFrame({"race_date": ["2025-01-01"], "track": ["Ascot"],
                         "off_time": ["14:30"], "horse_name": ["X"], "odds": [3.0]})
    out = compute_clv(bets)
    assert out["bsp_available"] is False
    assert out["n_matched"] == 0


# ── production pointer ────────────────────────────────────────────────
def _dummy_model_file(path):
    joblib.dump({"feature_cols": ["a", "b"], "frameworks": {"classifier": "lgbm", "place": "lgbm"}}, path)


def test_promote_and_load_production(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; runs.mkdir()
    monkeypatch.setattr(rs, "RUNS_DIR", str(runs))
    monkeypatch.setattr(rs, "PRODUCTION_POINTER", str(tmp_path / "production.json"))
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path / "models"))

    rid = "20250101_120000"
    os.makedirs(runs / rid)
    _dummy_model_file(runs / rid / "model.joblib")

    assert rs.get_production_run_id() is None
    assert rs.promote(rid) is False, "an unsigned run must not promote by default"
    assert rs.promote(rid, allow_feature_mismatch=True) is True
    assert rs.get_production_run_id() == rid
    assert os.path.isfile(rs.PRODUCTION_POINTER)

    pred = rs.load_production_predictor()
    assert pred is not None
    assert pred.feature_cols == ["a", "b"]


def test_promote_missing_model_fails(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; runs.mkdir()
    monkeypatch.setattr(rs, "RUNS_DIR", str(runs))
    monkeypatch.setattr(rs, "PRODUCTION_POINTER", str(tmp_path / "production.json"))
    os.makedirs(runs / "20250101_120000")  # no model.joblib
    assert rs.promote("20250101_120000") is False
    assert rs.get_production_run_id() is None


# ── lean run save (no dataset snapshots) ──────────────────────────────
def test_save_run_is_lean(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; runs.mkdir()
    models = tmp_path / "models"; models.mkdir()
    monkeypatch.setattr(rs, "RUNS_DIR", str(runs))
    monkeypatch.setattr(config, "MODELS_DIR", str(models))
    _dummy_model_file(models / "triple_ensemble_models.joblib")

    rid = rs.save_run(
        name="lean", model_type="race_predictor", data_source="feature_store",
        data_rows=100, n_features=10, elapsed_seconds=1.0,
        metrics={"win_classifier": {"brier_score": 0.1}},
        training_config={"evals": {"clv": {"overall": {"mean_clv": 1.1}}}},
        featured_df=None, processed_df=None,   # lean
    )
    run_dir = runs / rid
    assert (run_dir / "meta.json").is_file()
    assert (run_dir / "model.joblib").is_file()           # model snapshot kept
    assert not (run_dir / "featured_races.parquet").exists()   # NO dataset snapshots
    assert not (run_dir / "processed_races.parquet").exists()


# ── train_run orchestration (mocked) ──────────────────────────────────
@pytest.fixture(autouse=True)
def accepted_training_inputs(monkeypatch):
    monkeypatch.setattr("src.source_dataset.select_dataset", lambda **kwargs: {"id": "training-fixture", "sources": {}})


def test_training_refuses_history_without_a_complete_valid_field(monkeypatch):
    frame = pd.DataFrame({"race_id": ["r1", "r1"],
        "race_date": pd.to_datetime(["2025-01-01"]*2),
        "horse_name": ["A", "B"], "won": [1, 0],
        "finish_position": [1, 0], "num_runners": [3, 3]})
    monkeypatch.setattr(tr, "process_data", lambda: frame.copy())
    monkeypatch.setattr(tr, "get_training_dataset", lambda processed=None: frame.copy())
    monkeypatch.setattr(tr, "RacePredictor", lambda **k: pytest.fail("Invalid fields reached model fitting"))
    monkeypatch.setattr(tr, "save_run", lambda **k: pytest.fail("Invalid fields produced a run"))
    with pytest.raises(ValueError, match="complete single-winner"):
        tr.train_run(tr.TrainingConfig(refresh_data=False))


@pytest.mark.parametrize("walk_forward", [False, True])
@pytest.mark.parametrize("with_incomplete_field", [False, True])
def test_train_run_orchestration(monkeypatch, walk_forward, with_incomplete_field):
    feat = pd.DataFrame({
        "race_id": ["r1", "r1"], "race_date": pd.to_datetime(["2025-01-01"] * 2),
        "horse_name": ["A", "B"], "off_time": ["14:30", "14:30"],
        "won": [1, 0], "finish_position": [1, 0], "num_runners": [2, 2],
    })
    if with_incomplete_field:
        # Only two of the three declared starters were collected. Drop this
        # entire race, while preserving B's genuine non-finish in complete r1.
        feat = pd.concat([feat, feat.assign(race_id="r2", num_runners=3)], ignore_index=True)

    captured = {}

    class _FakePred:
        feature_cols = ["a", "b"]

        def __init__(self, *a, **k):
            captured["predictor_options"] = k
            self.test_analysis = {"stats": {"value": {"roi": 5.0, "bets": 3}}, "bets": pd.DataFrame()}
            self.train_metrics = {}
        def train(self, df, **k):
            captured["training_frame"] = df.copy()
            captured["train_options"] = k
            return {"win_classifier": {"brier_score": 0.12, "log_loss": 0.3}}

    def _fake_save_run(**kwargs):
        captured.update(kwargs)
        return "20250101_000000"

    monkeypatch.setattr(tr, "process_data", lambda: feat.copy())
    monkeypatch.setattr(tr, "get_training_dataset", lambda processed=None: feat.copy())
    monkeypatch.setattr(tr, "RacePredictor", _FakePred)
    monkeypatch.setattr(tr, "fe_signature", lambda: "FE")
    monkeypatch.setattr(tr, "data_signature", lambda df, cutoff=None: "DS")
    monkeypatch.setattr(tr, "compute_clv", lambda bets, ref_df=None: {"overall": {"mean_clv": 1.2}})
    monkeypatch.setattr(tr, "save_run", _fake_save_run)

    import src.backtester as bt

    def _fake_walk_forward(df, **kwargs):
        pd.testing.assert_frame_equal(df, captured["training_frame"])
        captured["walk_forward_options"] = kwargs
        return {"summary": pd.DataFrame()}

    monkeypatch.setattr(bt, "walk_forward_validation", _fake_walk_forward)
    cfg = tr.TrainingConfig(
        name="t", refresh_data=False, seed=123,
        frameworks={"classifier": "lgbm", "place": "lgbm"},
        hyperparams={"classifier": {"n_estimators": 75}},
        run_walk_forward=walk_forward, wf_min_train_months=6,
        wf_fast_fold=False, blend_market_odds=True, win_calibration_method="race_temperature",
        value_config={"value_threshold": 0.08, "ew_enabled": False},
    )
    rid = tr.train_run(cfg)
    assert rid == "20250101_000000"
    # lean: no dataset frames passed to save_run
    assert captured["featured_df"] is None and captured["processed_df"] is None
    assert captured["train_options"]["save"] is False
    assert isinstance(captured["predictor"], _FakePred)
    # evals bundled all three families + signatures recorded
    tc = captured["training_config"]
    assert set(tc["evals"]) == {"calibration", "roi", "clv"}
    assert tc["signatures"] == {"fe_signature": "FE", "data_signature": "DS"}
    assert tc["config"]["name"] == "t"
    assert captured["training_frame"].race_id.tolist() == ["r1", "r1"]
    assert captured["training_frame"].finish_position.tolist() == [1, 0]
    assert tc["split"]["input_field_validation"]["excluded_races"] == int(with_incomplete_field)
    assert tc["split"]["input_field_validation"]["excluded_rows"] == 2*int(with_incomplete_field)
    assert captured["train_options"]["holdout_days"] == cfg.holdout_days == 90
    assert captured["predictor_options"] == {
        "frameworks": cfg.frameworks, "random_seed": 123,
        "win_calibration_method": "race_temperature",
    }
    if walk_forward:
        wf = captured["walk_forward_options"]
        assert wf["frameworks"] == cfg.frameworks
        assert wf["params"] == captured["train_options"]["params"] == cfg.hyperparams
        assert wf["random_seed"] == 123
        assert wf["win_calibration_method"] == "race_temperature"
        assert wf["min_train_months"] == 6
        assert wf["fast_fold"] is False
        assert wf["blend_market_odds"] is True
        assert wf["value_threshold"] == 0.08
        assert wf["ew_enabled"] is False
    else:
        assert "walk_forward_options" not in captured
