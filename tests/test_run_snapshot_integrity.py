"""Runs own their artifacts and retain provenance from before the fit."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier, Event

import joblib
import pandas as pd
import pytest

import config
from src.model import RacePredictor
from src import run_store as rs


@pytest.fixture
def registry(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(rs, "RUNS_DIR", str(runs))
    monkeypatch.setattr(rs, "PRODUCTION_POINTER", str(models / "production.json"))
    monkeypatch.setattr(config, "MODELS_DIR", str(models))
    return runs, models


def _save(**kwargs):
    return rs.save_run(
        name="test", model_type="race_predictor", data_source="feature_store",
        data_rows=2, n_features=1, elapsed_seconds=1, **kwargs,
    )


def test_concurrent_saves_keep_their_own_predictor_and_unique_id(registry, monkeypatch):
    runs, models = registry
    shared_path = models / "triple_ensemble_models.joblib"
    joblib.dump({"marker": "unrelated global model"}, shared_path)
    monkeypatch.setattr(rs, "_make_run_id", lambda: "20260101_120000")
    barrier = Barrier(2)

    class Predictor:
        def __init__(self, marker):
            self.marker = marker

        def save(self, path):
            barrier.wait(timeout=10)
            joblib.dump({"marker": self.marker}, path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_save, predictor=Predictor(i), metrics={"marker": i})
                   for i in range(2)]
        ids = [future.result(timeout=20) for future in futures]

    assert len(set(ids)) == 2
    assert len(rs.list_runs()) == 2
    for expected, rid in enumerate(ids):
        assert joblib.load(runs / rid / "model.joblib")["marker"] == expected
        assert rs.load_run_meta(rid)["metrics"]["marker"] == expected
    assert joblib.load(shared_path)["marker"] == "unrelated global model"


def test_run_is_invisible_and_not_prunable_until_all_artifacts_saved(registry, monkeypatch):
    runs, _ = registry
    monkeypatch.setattr(rs, "_make_run_id", lambda: "20260101_120000")
    completed_ids = ["20260301_120000", "20260201_120000"]
    for rid in completed_ids:
        completed = runs / rid
        completed.mkdir()
        (completed / "meta.json").write_text(json.dumps({"run_id": rid}))
    ready, release = Event(), Event()

    class PausedPredictor:
        def save(self, path):
            joblib.dump({"marker": "own model"}, path)
            ready.set()
            assert release.wait(timeout=10)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_save, predictor=PausedPredictor())
        assert ready.wait(timeout=10)
        try:
            assert [run["run_id"] for run in rs.list_runs()] == completed_ids
            assert rs.prune_runs(keep_latest=1) == [completed_ids[1]]
            assert (runs / "20260101_120000" / "model.joblib").is_file()
        finally:
            release.set()
        rid = future.result(timeout=20)
    assert rid in [run["run_id"] for run in rs.list_runs()]
    assert (runs / rid / "model.joblib").is_file()


@pytest.mark.parametrize("failure_stage", ["model", "dataset", "metadata"])
def test_failed_save_never_publishes_or_replaces_existing_run(registry, monkeypatch, failure_stage):
    runs, _ = registry
    existing = runs / "20260101_120000"
    existing.mkdir()
    (existing / "model.joblib").write_bytes(b"keep")
    (existing / "meta.json").write_text(json.dumps({"run_id": existing.name}))
    monkeypatch.setattr(rs, "_make_run_id", lambda: existing.name)

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    class Predictor:
        def save(self, path):
            Path(path).write_bytes(b"partial")
            if failure_stage == "model":
                fail()

    if failure_stage == "dataset":
        monkeypatch.setattr(rs, "_snapshot_datasets", fail)
    elif failure_stage == "metadata":
        monkeypatch.setattr(rs.os, "replace", fail)

    with pytest.raises(OSError, match="injected failure"):
        _save(predictor=Predictor())
    assert [run["run_id"] for run in rs.list_runs()] == [existing.name]
    assert list(runs.iterdir()) == [existing]
    assert (existing / "model.joblib").read_bytes() == b"keep"


def test_predictor_explicit_save_round_trip_does_not_touch_global(registry):
    runs, models = registry
    shared_path = models / "triple_ensemble_models.joblib"
    shared_path.write_bytes(b"keep")
    predictor = RacePredictor(random_seed=17, win_calibration_method="race_temperature")
    predictor.feature_cols = ["speed"]
    predictor.calibration_temp = 1.2
    snapshot = runs / "nested" / "model.joblib"
    predictor.save(snapshot)
    loaded = RacePredictor()
    loaded.load(str(snapshot))
    assert loaded.feature_cols == ["speed"]
    assert loaded.calibration_temp == pytest.approx(1.2)
    assert loaded.win_calibration_method == "race_temperature"
    assert loaded.random_seed == 17
    assert shared_path.read_bytes() == b"keep"


