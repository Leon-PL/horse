from datetime import datetime, timezone
import json

import pytest

from scripts import project_status as status
from src import forward_store, run_store


def heartbeat(tmp_path, monkeypatch, **overrides):
    now = datetime(2026, 10, 30, 0, 30, tzinfo=timezone.utc)
    study = tmp_path / "study"
    study.mkdir()
    pointer = tmp_path / "forecasts/recorder.json"
    pointer.parent.mkdir()
    pointer.write_text(json.dumps({"study": str(study), "model_id": "current"}))
    value = {"observed_at": now.isoformat(), "last_discovery": now.isoformat(),
        "mode": "recording", "selected_pairs": 0, "model_id": "current", "study": str(study)}
    value.update(overrides)
    (study / "recorder_heartbeat.json").write_text(json.dumps(value))
    monkeypatch.setattr(status.config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(forward_store, "utcnow", lambda: now)
    monkeypatch.setattr(run_store, "get_production_run_id", lambda: "current")
    return now


@pytest.mark.parametrize("discovery", [None, "2026-10-29T23:00:00+00:00", "2026-10-30T01:00:00+00:00"])
def test_alive_process_with_missing_stale_or_future_discovery_is_not_healthy(tmp_path, monkeypatch, discovery):
    heartbeat(tmp_path, monkeypatch, last_discovery=discovery)
    assert status._forward_recording()["state"] == status.WARN


def test_successful_feed_recovery_is_healthy_despite_preserved_old_errors(tmp_path, monkeypatch):
    heartbeat(tmp_path, monkeypatch, recent_errors=[{"stage": "discovery", "error": "old reset"}])
    assert status._forward_recording()["state"] == status.OK


@pytest.mark.parametrize("overrides", [{"model_id": "other"}, {"study": "elsewhere"}, {"study": None},
    {"mode": "stopped"}, {"observed_at": "2026-10-29T23:00:00+00:00"}])
def test_wrong_identity_or_stopped_or_stale_process_is_not_healthy(tmp_path, monkeypatch, overrides):
    heartbeat(tmp_path, monkeypatch, **overrides)
    assert status._forward_recording()["state"] == status.WARN


def test_recorder_for_the_previous_production_model_is_reported(tmp_path, monkeypatch):
    heartbeat(tmp_path, monkeypatch)
    monkeypatch.setattr(run_store, "get_production_run_id", lambda: "new-production")
    assert status._forward_recording()["state"] == status.WARN


def test_normal_fifteen_minute_discovery_interval_allows_a_fresh_heartbeat(tmp_path, monkeypatch):
    heartbeat(tmp_path, monkeypatch, last_discovery="2026-10-30T00:15:00+00:00")
    result = status._forward_recording()
    assert result["state"] == status.OK and result["discovery_age_seconds"] == 900
