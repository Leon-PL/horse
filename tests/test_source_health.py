"""Subset and historical refreshes retain honest per-source coverage."""
from datetime import date, datetime, timedelta, timezone
import json

import pytest

from src.ingestion import auxiliary as source_auxiliary, pipeline as sources
from src.ingestion import results
from src.ingestion.storage import atomic_json


@pytest.fixture
def health(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(sources, "_now", lambda: now)
    yesterday = (date.today()-timedelta(days=1)).isoformat()
    old = (now-timedelta(hours=2)).isoformat()
    previous = {"state": "ok", "checked_at": old, "start": yesterday, "end": yesterday,
                "required_sources": list(sources.DEFAULT_SOURCES),
                "sources": {name: {"state": "ok", "receipt": name} for name in sources.DEFAULT_SOURCES}}
    path = tmp_path/"source_refresh/status.json"
    atomic_json(path, previous)
    return path, previous, now


def test_subset_retains_other_sources_and_their_original_check_times(health, monkeypatch):
    path, before, now = health
    monkeypatch.setattr(source_auxiliary, "refresh_atr", lambda *a, **kw: {"state": "partial", "pending_races": 12})
    result = sources.refresh_sources(1, sources=("atr",))
    saved = json.loads(path.read_text())
    assert result["required_sources"] == ["atr"] and result["state"] == "partial"
    assert saved["required_sources"] == list(sources.DEFAULT_SOURCES)
    for name in ("sl", "rtv", "betfair", "fx"):
        assert saved["sources"][name]["receipt"] == name
        assert saved["sources"][name]["checked_at"] == before["checked_at"]
    assert saved["sources"]["atr"]["checked_at"] == now.isoformat()
    assert saved["checked_at"] == before["checked_at"]
    assert saved["updated_at"] == now.isoformat()
    assert json.loads(path.with_name("last_run.json").read_text()) == result


def test_successful_subset_does_not_erase_another_sources_failure(health, monkeypatch):
    path, _, _ = health
    monkeypatch.setattr(source_auxiliary, "refresh_atr", lambda *a, **kw: {"state": "partial"})
    monkeypatch.setattr(source_auxiliary, "refresh_betfair", lambda *a: {"state": "cached"})
    sources.refresh_sources(1, sources=("atr",))
    assert sources.refresh_sources(1, sources=("betfair",))["state"] == "ok"
    saved = json.loads(path.read_text())
    assert saved["state"] == "partial" and saved["sources"]["atr"]["state"] == "partial"


def test_old_backfill_retains_newer_coverage_and_records_its_own_failure(health, monkeypatch):
    path, before, _ = health
    monkeypatch.setattr(results, "sync_database", lambda **kw: {"state": "error", "error": "old page unavailable"})
    older = (date.today()-timedelta(days=30)).isoformat()
    result = sources.refresh_sources(1, sources=("sl",), start=older, end=older)
    saved = json.loads(path.read_text())
    sl = saved["sources"]["sl"]
    assert result["state"] == "partial" and saved["state"] == "ok"
    assert sl["requested_end"] == before["end"] and sl["checked_at"] == before["checked_at"]
    assert sl["last_attempt"]["requested_end"] == older
    assert sl["last_attempt"]["error"] == "old page unavailable"


def test_subset_cannot_make_stale_other_sources_fresh(health, monkeypatch):
    path, before, now = health
    stale = (now-timedelta(days=2)).isoformat()
    before["sources"]["fx"]["checked_at"] = stale
    atomic_json(path, before)
    monkeypatch.setattr(source_auxiliary, "refresh_betfair", lambda *a: {"state": "cached"})
    sources.refresh_sources(1, sources=("betfair",))
    assert json.loads(path.read_text())["checked_at"] == stale


def test_first_subset_leaves_unknown_required_sources_unverified(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(results, "sync_database", lambda **kw: {"state": "ok"})
    result = sources.refresh_sources(1, sources=("sl",))
    saved = json.loads((tmp_path/"source_refresh/status.json").read_text())
    assert result["state"] == "ok" and saved["state"] == "partial"
    assert set(saved["sources"]) == {"sl"}
