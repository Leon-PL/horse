"""Stored feature inspection, independent model selections and explicit jobs."""
# ruff: noqa: F811 -- shared pytest fixture is injected by name
import hashlib
import json
from pathlib import Path
import threading
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pytest

import config
from src.feature_store import FeatureStore
from src.workbench.feature_catalog import load_sets
from src.workbench.features import Features
from src.workbench.jobs import Jobs
from src.workbench.server import WorkbenchServer
from tests.test_workbench import catalog  # noqa: F401 -- shared isolated fixture


@pytest.fixture
def featured(catalog, monkeypatch):
    folder = catalog.base/"feature_sets"
    folder.mkdir()
    (folder/"general.txt").write_text("signal\nhorse_prev_races\n", encoding="utf-8")
    (folder/"specialist.txt").write_text("signal\nbha_lame_last\n", encoding="utf-8")
    (folder/"registry.json").write_text(json.dumps({"version": 1, "feature_sets": [
        {"id": "general", "label": "General selection", "file": "general.txt"},
        {"id": "specialist", "label": "Specialist selection", "file": "specialist.txt"}],
        "model_roles": [{"id": "general", "label": "General model", "feature_sets": ["general"]},
                        {"id": "specialist", "label": "Specialist model", "scope": "Chases", "feature_sets": ["specialist", "general"]}]}), encoding="utf-8")
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", str(folder/"general.txt"))
    service = Features(catalog)
    store = FeatureStore(str(service.root))
    frame = catalog.canonical("r1")
    frame["race_date"] = pd.to_datetime(frame.race_date)
    frame = frame.assign(signal=[0., np.nan], other=[np.inf, 2.], horse_prev_races=[0., 3.],
                         bha_report_count_5=[0., 1.], bha_lame_last=[np.nan, 0.])
    frame.to_parquet(store.featured_path, index=False)
    before = frame.iloc[1:2].assign(race_id="r0", race_date=pd.Timestamp("2026-05-31"), finish_position=4)
    future = frame.iloc[1:2].assign(race_id="r2", race_date=pd.Timestamp("2026-06-02"))
    pd.concat([before, frame, future], ignore_index=True).to_parquet(store.processed_path, index=False)
    store._save_state({"elo": None, "glicko": None})
    store._save_counts_state({})
    with store._access():
        Path(store.meta_path).write_text(json.dumps({"featured_rows": 2, "last_date": "2026-06-01",
            "fe_signature": service.code_signature(), "dataset_manifest": catalog.manifest()}), encoding="utf-8")
    return service, store


def snapshot(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}


def test_missing_store_and_registry_are_read_only(catalog, monkeypatch):
    monkeypatch.setattr("src.source_dataset.capture_dataset", lambda **kw: pytest.fail("Publication on browse"))
    service = Features(catalog)
    assert service.overview()["store"]["state"] == "missing"
    assert not service.root.exists()
    assert not (catalog.root/"workbench").exists()


def test_model_roles_and_sets_are_many_to_many_without_changing_training(featured):
    service, _ = featured
    original = config.FEATURE_ALLOWLIST
    registry = service.registry()
    assert len(registry["feature_sets"]) == 2 and len(registry["model_roles"]) == 2
    general = service.catalog_rows("general")
    specialist = service.catalog_rows("specialist")
    assert {r["name"] for r in general["rows"]} == {"signal", "horse_prev_races"}
    assert {r["name"] for r in specialist["rows"]} == {"signal", "bha_lame_last"}
    shared = next(r for r in general["rows"] if r["name"] == "signal")
    assert set(shared["feature_sets"]) == {"general", "specialist"}
    assert set(shared["model_roles"]) == {"general", "specialist"}
    assert config.FEATURE_ALLOWLIST == original


def test_inspection_does_not_mutate_artifacts_or_publish(featured, monkeypatch):
    service, store = featured
    before = snapshot(service.root)
    monkeypatch.setattr("src.feature_engineer.engineer_features", lambda **kw: pytest.fail("Feature calculation on GET"))
    monkeypatch.setattr("src.source_dataset.capture_dataset", lambda **kw: pytest.fail("Publication on GET"))
    assert service.overview()["store"]["state"] == "current"
    result = service.race("r1", "general")
    assert len(result["rows"]) == 2
    assert result["rows"][0]["signal"] == 0
    assert result["rows"][1]["signal"] is None
    assert service.value("r1", "h1", "signal")["missing"] is False
    value = service.value("r1", "h2", "signal")
    assert value["missing"] and [r["race_id"] for r in value["history"]] == ["r0"]
    assert "not an exact contributor trace" in value["history_message"]
    assert service.value("r1", "h1", "other")["invalid"]
    assert not service.value("r1", "h1", "other")["missing"]
    assert snapshot(service.root) == before
    assert store.meta()["featured_rows"] == 2


def test_stale_generation_and_dataset_are_explicit(featured):
    service, store = featured
    meta = store.meta()
    meta["fe_signature"] = "old"
    meta.pop("dataset_manifest")
    Path(store.meta_path).write_text(json.dumps(meta))
    status = service.overview()
    assert status["store"]["state"] == "stale"
    assert len(status["store"]["reasons"]) == 2
    assert len(service.race("r1", "general")["rows"]) == 2


def test_saved_model_signature_uses_recorded_training_metadata(featured):
    service, store = featured
    models = service.catalog.base/"models"
    models.mkdir()
    (models/"production.json").write_text(json.dumps({"run_id": "saved-run"}))
    run = service.catalog.root/"runs/saved-run"
    run.mkdir(parents=True)
    metadata = {"training_config": {"signatures": {"fe_signature": store.meta()["fe_signature"]}}}
    (run/"meta.json").write_text(json.dumps(metadata))
    assert service.overview()["saved_model"]["state"] == "definition_match"
    metadata["training_config"]["signatures"]["fe_signature"] = "different"
    (run/"meta.json").write_text(json.dumps(metadata))
    assert service.overview()["saved_model"]["state"] == "definition_mismatch"


def test_build_identity_changes_even_when_definition_signature_does_not(featured):
    service, store = featured
    before = service.overview()["store"]["build_id"]
    frame = pd.read_parquet(store.featured_path)
    frame["signal"] = [3., 4.]
    frame.to_parquet(store.featured_path, index=False)
    assert service.overview()["store"]["build_id"] != before


def test_pending_commit_is_reported_without_recovery(featured):
    service, _ = featured
    journal = service.root/".pending_commit.json"
    journal.write_text('{"stage":".commit-test"}')
    before = snapshot(service.root)
    status = service.overview()
    assert "recovery" in status["store"]["reasons"][0]
    assert snapshot(service.root) == before


def test_quality_preserves_missing_zero_and_invalid_values(featured):
    service, _ = featured
    quality = service.quality("2026-06-01", "2026-06-01", "general")
    signal = next(r for r in quality["rows"] if r["name"] == "signal")
    assert signal["missing_pct"] == 50 and signal["zeros"] == 1 and signal["state"] == "constant"
    assert quality["duplicate_keys"] == 0 and quality["rows_checked"] == 2
    all_values = service.quality("2026-06-01", "2026-06-01", "all")
    assert next(r for r in all_values["rows"] if r["name"] == "other")["infinite"] == 1
    json.dumps(all_values, allow_nan=False)
    with pytest.raises(ValueError, match="31 days"):
        service.quality("2025-01-01", "2026-06-01", "general")


def test_missing_selected_column_is_reported_not_dropped(featured):
    service, _ = featured
    (service.catalog.base/"feature_sets/general.txt").write_text("signal\nnot_built\n")
    r = service.quality("2026-06-01", "2026-06-01", "general")
    assert r["missing_columns"] == ["not_built"]
    row = next(r for r in service.catalog_rows("general")["rows"] if r["name"] == "not_built")
    assert not row["stored"] and row["documentation"] == "family_only"


def test_quality_is_bounded(featured):
    service, store = featured
    sample = pd.read_parquet(store.featured_path).iloc[:1]
    many = pd.concat([sample]*5100, ignore_index=True)
    many["horse_id"] = [str(i) for i in range(len(many))]
    many.to_parquet(store.featured_path, index=False)
    r = service.quality("2026-06-01", "2026-06-01", "general")
    assert r["truncated"] and r["rows_checked"] == 5000


@pytest.mark.parametrize("change", ["path", "role", "duplicates", "label"])
def test_registry_rejects_invalid_bindings(featured, change):
    service, _ = featured
    path = service.catalog.base/"feature_sets/registry.json"
    raw = json.loads(path.read_text())
    if change == "path":
        raw["feature_sets"][0]["file"] = "../../outside.txt"
    if change == "role":
        raw["model_roles"][0]["feature_sets"] = ["not-registered"]
    if change == "duplicates":
        (path.parent/"general.txt").write_text("signal\nsignal\n")
    if change == "label":
        (path.parent/"general.txt").write_text("won\n")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_sets(service.catalog.base)


def test_job_parameters_pin_dataset_and_check_selection(featured):
    service, _ = featured
    manager = Jobs(service.catalog)
    r = manager.validate("feature_update", {"rebuild": True})
    assert r == {"dataset": service.catalog.manifest()["id"], "rebuild": True}
    r = manager.validate("feature_check", {"start": "2026-06-01", "end": "2026-06-01", "feature_set": "specialist"})
    assert r["feature_set_signature"] == service.selection("specialist")["signature"]
    assert r["feature_set"] == "specialist"
    with pytest.raises(ValueError):
        manager.validate("feature_update", {"rebuild": "false"})
    with pytest.raises(ValueError):
        manager.validate("feature_check", {"start": "2026-06-01", "end": "2026-06-01", "feature_set": "missing"})


@pytest.mark.parametrize("append,rebuild", [(True, False), (False, False), (False, True)])
def test_shared_update_uses_pinned_sources_and_existing_atomic_builder(featured, monkeypatch, append, rebuild):
    from contextlib import nullcontext
    from src.source_dataset import current_manifest
    from src.workbench.features import update_store
    service, _ = featured
    dataset = service.catalog.manifest()["id"]
    calls = []
    def process():
        assert current_manifest()["id"] == dataset
        assert Path(current_manifest()["root"]) == service.catalog.sources
        return pd.DataFrame({"x": [1, 2]})
    class Store:
        def __init__(self, root):
            assert Path(root) == service.root
        def _staged_update(self):
            calls.append("stage")
            return nullcontext(self)
        def meta(self):
            return {"fe_signature": "built"}
    def try_append(frame, store):
        calls.append("append")
        return append
    def build(store, chunk, processed):
        calls.append("build")
        assert chunk == 60000 and len(processed) == 2
    monkeypatch.setattr("src.data_processor.process_data", process)
    monkeypatch.setattr("src.feature_store.FeatureStore", Store)
    monkeypatch.setattr("src.store_refresh.append_current_store", try_append)
    monkeypatch.setattr("src.store_build.build_store", build)
    result = update_store(service.catalog, dataset, rebuild)
    assert not result["model_fitted"]
    assert calls == (["stage", "build"] if rebuild else ["append"] if append else ["append", "stage", "build"])


def test_feature_api_and_static_module_use_the_local_server(featured):
    service, _ = featured
    server = WorkbenchServer(("127.0.0.1", 0), service.catalog, Path(__file__).resolve().parents[1]/"ui/dist")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for route in ("features/overview", "features/catalog?feature_set=specialist", "features/races/r1?feature_set=general", "features/races/r1/value?horse_id=h2&column=signal"):
            with urlopen(base+"/api/"+route) as response:
                assert response.status == 200
                json.loads(response.read())
        with urlopen(base+"/features.js") as response:
            assert "javascript" in response.headers["Content-Type"]
        with urlopen(base+"/api/features/races/r1/export?feature_set=general") as response:
            assert b"feature_generation" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_worker_feature_update_calls_existing_owner_without_fitting(featured, monkeypatch):
    from src.workbench import worker
    service, _ = featured
    manager = Jobs(service.catalog)
    manager.root.mkdir(parents=True)
    identifier = "a"*32
    path = manager.root/(identifier+".json")
    params = manager.validate("feature_update", {})
    path.write_text(json.dumps({"id": identifier, "kind": "feature_update", "params": params, "data_dir": str(service.catalog.root)}))
    called = []
    def update(catalog, dataset, rebuild, progress):
        called.append((dataset, rebuild))
        progress(phase="building", rows=2)
        return {"model_fitted": False}
    monkeypatch.setattr("src.workbench.features.update_store", update)
    worker.run(path)
    result = json.loads(path.read_text())
    assert result["state"] == "complete" and not result["result"]["model_fitted"]
    assert called == [(params["dataset"], False)]
