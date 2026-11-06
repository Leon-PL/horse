"""The source coordinator publishes progress without serialising unrelated IO."""
import json
from threading import Barrier, Event, get_ident

from src.ingestion import pipeline as sources, auxiliary
from src.ingestion import results


def test_independent_sources_overlap_only_after_canonical_results(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    ready = Event()
    overlap = Barrier(2)
    def canonical(**kwargs):
        ready.set()
        return {"state": "ok"}
    def independent(*args, **kwargs):
        assert ready.is_set()
        overlap.wait(timeout=5)
        return {"state": "ok"}
    monkeypatch.setattr(results, "sync_database", canonical)
    monkeypatch.setattr(sources, "refresh_rtv_history", independent)
    monkeypatch.setattr(auxiliary, "refresh_betfair", independent)
    report = sources.refresh_sources(1, sources=("sl", "rtv", "betfair"), max_source_workers=2)
    assert report["state"] == "ok"
    assert all(value["state"] == "ok" and "seconds" in value for value in report["sources"].values())


def test_progress_is_saved_and_callbacks_stay_on_the_calling_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(results, "sync_database", lambda **kw: {"state": "ok"})
    monkeypatch.setattr(sources, "refresh_rtv_history", lambda **kw: {"state": "cached"})
    parent = get_ident()
    observed = []
    def callback(source, value):
        on_disk = json.loads((tmp_path/"source_refresh/status.json").read_text())
        observed.append((get_ident(), source, value["state"], on_disk["sources"][source]["state"]))
    report = sources.refresh_sources(1, sources=("sl", "rtv"), progress_callback=callback)
    assert [value[2] for value in observed] == ["running", "ok", "running", "cached"]
    assert all(thread == parent and state == saved for thread, _, state, saved in observed)
    assert report["state"] == "ok"
    assert json.loads((tmp_path/"source_refresh/last_run.json").read_text())["state"] == "ok"
    # Selected collectors completed; other required sources remain unassessed.
    assert json.loads((tmp_path/"source_refresh/status.json").read_text())["state"] == "partial"


def test_a_failed_collector_does_not_cancel_other_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(sources, "refresh_rtv_history", lambda **kw: {"state": "ok"})
    def broken(*args):
        raise RuntimeError("price provider unavailable")
    monkeypatch.setattr(auxiliary, "refresh_betfair", broken)
    report = sources.refresh_sources(1, sources=("rtv", "betfair"))
    assert report["state"] == "partial"
    assert report["sources"]["rtv"]["state"] == "ok"
    assert report["sources"]["betfair"]["error"] == "price provider unavailable"


def test_short_source_finishes_before_a_long_backfill(tmp_path, monkeypatch):
    from src.ingestion import fx
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    release = Event()
    monkeypatch.setattr(fx, "refresh_prize_references", lambda: {"state": "ok"})
    def backfill(*args, **kwargs):
        assert release.wait(timeout=5), "Completed short-source progress was withheld"
        return {"state": "partial", "pending_races": 100}
    monkeypatch.setattr(auxiliary, "refresh_atr", backfill)
    def callback(source, value):
        if source == "fx" and value["state"] == "ok":
            release.set()
    report = sources.refresh_sources(1, sources=("fx", "atr"), max_source_workers=2, progress_callback=callback)
    assert release.is_set() and report["state"] == "partial"
def test_command_defaults_include_the_full_betfair_provider(monkeypatch):
    import sys
    calls = []
    monkeypatch.setattr(sys, "argv", ["ingestion", "--sync", "1"])
    monkeypatch.setattr(sources, "refresh_sources", lambda *a, **kw:
                        calls.append(kw) or {"state": "ok", "sources": {}})
    sources.main()
    assert "betfair" in calls[0]["sources"] and "bsp" not in calls[0]["sources"]
