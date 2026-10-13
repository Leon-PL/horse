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


# ── CLV vs BSP ────────────────────────────────────────────────────────
def test_compute_clv_basic(tmp_path):
    from src.clv import compute_clv
    from src.betfair import bf_horse_key
    from src.utils import normalise_track_key, normalise_off_time_key

    bsp = pd.DataFrame({
        "race_date": ["2025-01-01", "2025-01-01"],
        "track_key": [normalise_track_key("Ascot"), normalise_track_key("Ascot")],
        "off_key": [normalise_off_time_key("14:30"), normalise_off_time_key("14:30")],
        "horse_key": [bf_horse_key("Alpha"), bf_horse_key("Bravo")],
        "bsp": [4.0, 2.0],
    })
    p = tmp_path / "bsp.parquet"
    bsp.to_parquet(p, index=False)

    bets = pd.DataFrame({
        "race_date": ["2025-01-01", "2025-01-01"],
        "track": ["Ascot", "Ascot"],
        "off_time": ["14:30", "14:30"],
        "horse_name": ["Alpha", "Bravo"],
        "odds": [5.0, 2.0],          # Alpha beat the close (5>4); Bravo matched (2==2)
        "strategy": ["value", "value"],
    })
    out = compute_clv(bets, bsp_path=str(p))
    assert out["n_matched"] == 2
    assert out["coverage"] == 1.0
    # mean CLV = mean(5/4, 2/2) = mean(1.25, 1.0) = 1.125
    assert out["overall"]["mean_clv"] == pytest.approx(1.125)
    assert out["overall"]["pct_beat_close"] == pytest.approx(0.5)


def test_compute_clv_no_bsp_file(tmp_path):
    from src.clv import compute_clv
    bets = pd.DataFrame({"race_date": ["2025-01-01"], "track": ["Ascot"],
                         "off_time": ["14:30"], "horse_name": ["X"], "odds": [3.0]})
    out = compute_clv(bets, bsp_path=str(tmp_path / "missing.parquet"))
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
@pytest.mark.parametrize("walk_forward", [False, True])
def test_train_run_orchestration(monkeypatch, walk_forward):
    feat = pd.DataFrame({
        "race_id": ["r1", "r1"], "race_date": pd.to_datetime(["2025-01-01"] * 2),
        "horse_name": ["A", "B"], "off_time": ["14:30", "14:30"],
    })

    captured = {}

    class _FakePred:
        def __init__(self, *a, **k):
            captured["predictor_options"] = k
            self.test_analysis = {"stats": {"value": {"roi": 5.0, "bets": 3}}, "bets": pd.DataFrame()}
            self.train_metrics = {}
        def train(self, df, **k):
            captured["train_options"] = k
            return {"win_classifier": {"brier_score": 0.12, "log_loss": 0.3}}

    def _fake_save_run(**kwargs):
        captured.update(kwargs)
        return "20250101_000000"

    monkeypatch.setattr(tr, "process_data", lambda save=False: feat.copy())
    monkeypatch.setattr(tr, "get_featured_dataset", lambda processed=None: feat.copy())
    monkeypatch.setattr(tr, "RacePredictor", _FakePred)
    monkeypatch.setattr(tr, "get_feature_columns", lambda df: ["a", "b"])
    monkeypatch.setattr(tr, "fe_signature", lambda: "FE")
    monkeypatch.setattr(tr, "data_signature", lambda df, cutoff=None: "DS")
    monkeypatch.setattr(tr, "compute_clv", lambda bets, ref_df=None: {"overall": {"mean_clv": 1.2}})
    monkeypatch.setattr(tr, "save_run", _fake_save_run)

    import src.backtester as bt

    def _fake_walk_forward(df, **kwargs):
        captured["walk_forward_options"] = kwargs
        return {"summary": pd.DataFrame()}

    monkeypatch.setattr(bt, "walk_forward_validation", _fake_walk_forward)
    cfg = tr.TrainingConfig(
        name="t", refresh_data=False, seed=123,
        frameworks={"classifier": "xgb", "place": "lgbm"},
        hyperparams={"classifier": {"n_estimators": 75}},
        run_walk_forward=walk_forward, wf_min_train_months=6,
        wf_fast_fold=False, blend_market_odds=True, win_calibration_method="race_temperature",
        value_config={"value_threshold": 0.08, "ew_enabled": False},
    )
    rid = tr.train_run(cfg)
    assert rid == "20250101_000000"
    # lean: no dataset frames passed to save_run
    assert captured["featured_df"] is None and captured["processed_df"] is None
    # evals bundled all three families + signatures recorded
    tc = captured["training_config"]
    assert set(tc["evals"]) == {"calibration", "roi", "clv"}
    assert tc["signatures"] == {"fe_signature": "FE", "data_signature": "DS"}
    assert tc["config"]["name"] == "t"
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
