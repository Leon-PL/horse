"""Exercise Workbench's modelling jobs through the supported benchmark and fit."""
# ruff: noqa: F811 -- shared pytest fixtures are injected by name
import json
from pathlib import Path

import pytest

import config
from src.ingestion.storage import atomic_json
from src.source_dataset import current_manifest
from src.workbench import worker
from src.workbench.jobs import Jobs
from src.workbench.modelling import Modelling
from tests.test_modelling import fields
from tests.test_workbench import catalog, http_server, request  # noqa: F401
from tests.test_workbench_features import featured  # noqa: F401


def test_modelling_routes_are_read_only_with_no_history(http_server):
    with request(http_server, "/modelling.js") as response:
        assert "javascript" in response.headers["Content-Type"]
    with request(http_server, "/api/modelling/overview") as response:
        overview = json.load(response)
    assert overview["blockers"] and overview["promoted"] is None
    with request(http_server, "/api/modelling/benchmarks") as response:
        assert not json.load(response)["rows"]
    assert not (http_server.catalog.root/"workbench").exists()
    assert not (http_server.catalog.root/"modelling").exists()


def test_benchmark_to_candidate_preserves_recipe_and_evaluation(featured, monkeypatch):
    features, store = featured
    catalog = features.catalog
    service = Modelling(catalog, features)
    manager = Jobs(catalog, modelling=service)
    manifest = catalog.manifest()
    meta_path = Path(store.meta_path)
    atomic_json(meta_path, {**json.loads(meta_path.read_text()), "data_signature": "test-data"})
    inputs = service.inputs(manifest["id"])
    recipe = features.selection("general")
    spec_args = {"dataset": manifest["id"], "build_id": inputs["store"]["build_id"],
                 "feature_set": recipe["id"], "feature_set_signature": recipe["signature"],
                 "capacity": "screen", "threads": 1, "fold_start": "2025-04-01"}
    with pytest.raises(ValueError, match="feature set changed"):
        manager.validate("model_benchmark", {**spec_args, "feature_set_signature": "changed"})
    with pytest.raises(ValueError, match="feature store changed"):
        manager.validate("model_benchmark", {**spec_args, "build_id": "changed"})
    monkeypatch.setattr(config, "BURN_IN_MONTHS", 0)
    monkeypatch.setattr("src.feature_selection.SCREEN_PARAMS", {"n_estimators": 20, "learning_rate": .1})
    monkeypatch.setattr("src.run_store.RUNS_DIR", str(catalog.root/"runs"))
    frame = fields(n=850).assign(horse_prev_races=0.)

    def dataset(root, *, features, development=True, end=None, **kwargs):
        assert Path(root) == service.features.root
        assert current_manifest()["id"] == manifest["id"]
        assert list(features) == ["signal", "horse_prev_races"]
        loaded = frame if development else frame.loc[frame.race_date.le(end)]
        return loaded.copy(), {
            "dataset_manifest": manifest, "fe_signature": inputs["store"]["signature"],
            "data_signature": "test-data", "dataset_start": "2023-01-01", "excluded_races": 0,
            "evaluation_status": "Retrospective fixture",
        }

    monkeypatch.setattr("src.modelling.__main__.load_dataset", dataset)

    def execute(kind, params, identifier):
        path = manager.root/(identifier+".json")
        atomic_json(path, {"id": identifier, "kind": kind, "params": params, "data_dir": str(catalog.root)})
        worker.run(path)
        return json.loads(path.read_text())

    completed = execute("model_benchmark", manager.validate("model_benchmark", spec_args), "a"*32)
    assert completed["state"] == "complete", completed.get("error")
    identifier = completed["result"]["benchmark_id"]
    row, folder = service.benchmark(identifier, manifest["id"])
    assert row["completed"] and not row["fit_blockers"]
    assert [f["label"] for f in row["folds"]] == ["2025Q2"]
    assert row["report"]["components"]["generalist"]["race_log_loss"] > 0

    fit_args = {**spec_args, "benchmark": identifier, "as_of": "2025-04-29", "name": "Test win model"}
    with pytest.raises(ValueError, match="last scored race"):
        manager.validate("model_fit", {**fit_args, "as_of": "2025-04-01"})
    params = manager.validate("model_fit", fit_args)
    report = folder/"report.json"
    original = json.loads(report.read_text())
    atomic_json(report, {**original, "evaluation_status": "Altered after queuing"})
    stale = execute("model_fit", params, "b"*32)
    assert stale["state"] == "failed" and "pinned benchmark" in stale["error"]
    atomic_json(report, original)

    fitted = execute("model_fit", manager.validate("model_fit", fit_args), "c"*32)
    assert fitted["state"] == "complete", fitted.get("error")
    run = service.run(fitted["result"]["run_id"], manifest["id"])
    assert run["type"] == "win_stack_candidate"
    assert run["trained_through"] == "2025-04-29" and run["has_model"]
    assert run["benchmark_id"] == identifier
    assert run["components"]["generalist"]["recipe"]["features"] == recipe["features"]
    assert not run["metrics"] and not run["promoted"]
    importance, = service.importance(run["id"])["components"]
    assert importance["label"] == "generalist"
    assert {f["feature"] for f in importance["features"]} == set(recipe["features"])
    assert sum(f["gain_share"] for f in importance["features"]) == pytest.approx(1)
    assert importance["features"][0]["splits"] > 0
