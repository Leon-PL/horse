"""Source collection must not alter canonical values or invent identity joins."""
from datetime import date
import json
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from src.ingestion import pipeline as sources
from src.ingestion import results


def frames():
    races = pd.DataFrame([{"race_id": 101, "race_date": "2026-10-29", "track_name": "Kempton Park",
        "off_time_utc": "17:30", "is_abandoned": False, "is_arab_race": False}])
    runners = pd.DataFrame([{"race_id": 101, "runner_id": 202, "horse_name": "Test Horse"}])
    canonical = pd.DataFrame([{"race_id": "sl-1", "horse_id": 303, "race_date": "2026-10-29",
        "track": "Kempton", "off_time": "17:30:00", "horse_name": "TEST HORSE"}])
    return races, runners, canonical


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(sources, "_refresh_rtv_features", lambda *args, **kwargs: {"state": "ok"})
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **kw: pytest.fail("No network in source tests"))
    return tmp_path


def test_crosswalk_uses_separate_source_ids_and_shared_normalisers():
    races, runners, canonical = frames()
    linked = sources.match_runners(races, runners, canonical)
    assert linked[["rtv_race_id", "rtv_runner_id", "sl_race_id", "sl_horse_id"]].values.tolist() == [[101, 202, "sl-1", 303]]
    assert "_track" not in races and "_track" not in canonical


@pytest.mark.parametrize("change", ["name", "time", "date", "ambiguous", "abandoned", "arab"])
def test_unsafe_identity_matches_are_rejected(change):
    races, runners, canonical = frames()
    if change == "name":
        runners.loc[0, "horse_name"] = "Renamed Horse"
    elif change == "time":
        races.loc[0, "off_time_utc"] = "18:30"
    elif change == "date":
        races.loc[0, "race_date"] = "2026-10-28"
    elif change == "ambiguous":
        canonical = pd.concat([canonical, canonical.assign(horse_id=404)], ignore_index=True)
    else:
        races.loc[0, "is_abandoned" if change == "abandoned" else "is_arab_race"] = True
    assert sources.match_runners(races, runners, canonical).empty


def test_unified_refresh_runs_results_then_rtv_without_weather(isolated, monkeypatch):
    calls = []
    monkeypatch.setattr(results, "sync_database", lambda **kw: calls.append(kw) or {"state": "ok"})
    monkeypatch.setattr(sources, "refresh_rtv_history", lambda **kw: calls.append(kw) or {"state": "ok"})
    report = sources.refresh_sources(3, enrich=False, sources=("sl", "rtv"))
    assert calls[0]["days_back"] == 3 and calls[0]["enrich"] is False
    assert calls[1] == {"days_back": 10}
    assert report["state"] == "ok"


def test_enriched_refresh_collects_profiles_after_rtv_history(isolated, monkeypatch):
    from src.ingestion import pedigree

    calls = []
    monkeypatch.setattr(results, "sync_database", lambda **kw: calls.append("sl") or {"state": "ok"})
    monkeypatch.setattr(sources, "refresh_rtv_history", lambda **kw: calls.append("rtv") or {"state": "ok"})
    monkeypatch.setattr(pedigree, "backfill_rtv_foaling_profiles",
                        lambda **kw: calls.append(kw) or {"state": "ok", "ok": 1})
    report = sources.refresh_sources(enrich=True, start="2026-10-29", end="2026-10-29",
                                     sources=("sl", "rtv"), pedigree_limit=5)
    assert calls == ["sl", "rtv", {"limit": 5, "start": "2026-10-29", "end": "2026-10-29"}]
    assert report["sources"]["rtv"]["foaling_profiles"]["ok"] == 1


def test_profile_backfill_uses_linked_slug_and_resumes_cached_response(isolated, monkeypatch):
    from src.ingestion import pedigree, rtv_observations

    races, runners, canonical = frames()
    with sqlite3.connect(isolated/"races.db") as con:
        canonical.to_sql("results", con, index=False)
        pd.DataFrame({"horse_id": ["303"], "horse_name": ["Test Horse"], "sire": ["Sire"],
                      "dam": ["Dam"], "foaled": ["2022-01-01"]}).to_sql("horse_pedigree", con, index=False)
    runners = runners.assign(horse_id=700, in_results=True, is_nonrunner=False, res_horse__slug="/test-horse")
    monkeypatch.setattr(rtv_observations, "observation_tables", lambda *a, **kw: {"races": races, "runners": runners})
    assert pedigree.rtv_profile_targets() == ["test-horse"]

    calls = []
    payload = {"horse": {"id": 700, "name": "Test Horse", "sire_name": "Sire", "dam_name": "Dam",
                         "foaling_date": "2022-04-19"}}
    monkeypatch.setattr(sources.rtv, "RTVClient", lambda: SimpleNamespace(
        get=lambda path: (calls.append(path) or 200, json.dumps(payload).encode())))
    first = pedigree.backfill_rtv_foaling_profiles(limit=1)
    second = pedigree.backfill_rtv_foaling_profiles()
    assert (first["state"], first["ok"], first["pending"]) == ("ok", 1, 0)
    assert (second["state"], second["requested"]) == ("cached", 0)
    assert calls == ["/horses/test-horse"]
    runners.loc[0, "horse_name"] = "Different horse"
    assert pedigree.rtv_profile_targets() == []


def test_rtv_outage_does_not_prevent_canonical_sync(isolated, monkeypatch, caplog):
    monkeypatch.setattr(results, "sync_database", lambda **kw: {"state": "ok"})
    def broken(**kwargs):
        raise sources.rtv.RTVTransientError("rate limited")
    monkeypatch.setattr(sources, "refresh_rtv_history", broken)
    assert sources.refresh_sources(1, sources=("sl", "rtv"))["state"] == "partial"
    assert "rate limited" in caplog.text


def test_app_card_capture_keeps_model_inputs_exact(isolated, monkeypatch):
    from src.ingestion import sporting_life as data_scraper
    expected = pd.DataFrame({"race_id": [1], "horse_id": [2]})
    calls = []
    monkeypatch.setattr(data_scraper, "get_scraped_racecards", lambda **kw: expected)
    monkeypatch.setattr(sources, "capture_rtv_snapshot", lambda day: calls.append(day))
    assert sources.get_scraped_racecards("2026-10-30", force_refresh=True) is expected
    assert calls == ["2026-10-30"]


def test_past_day_is_not_reconstructed_as_a_pre_race_capture(isolated):
    assert sources.capture_rtv_snapshot("2025-01-01")["state"] == "skipped"


def test_collection_failure_preserves_successful_manifest_and_is_visible(isolated, monkeypatch):
    root = sources.pipeline_dir()
    previous = {"days": {"2026-10-28": {"directory": "old", "complete": True,
        "checked_at": "2026-10-29T01:00:00+00:00"}}}
    sources.atomic_json(root / "history.json", previous)
    def fail(*args, **kwargs):
        raise sources.rtv.RTVTransientError("503")
    monkeypatch.setattr(sources.rtv, "scrape_days", fail)
    with pytest.raises(sources.rtv.RTVTransientError):
        sources.refresh_rtv_history(1, today=date(2026, 10, 30))
    assert json.loads((root / "history.json").read_text()) == previous
    assert json.loads((root / "history_status.json").read_text())["state"] == "error"


def test_daily_overlay_replaces_all_old_race_rows_without_duplicates(isolated):
    root = sources.pipeline_dir()
    base = root.parent / "tables/races"
    base.mkdir(parents=True)
    pd.DataFrame({"race_id": [1, 2], "race_date": ["2026-10-28", "2026-10-29"], "value": [10, 20]}).to_parquet(base / "2026-Q4.parquet")
    target = root / "history/2026-10-29/new"
    target.mkdir(parents=True)
    pd.DataFrame({"race_id": [2, 3], "race_date": ["2026-10-29"]*2, "value": [99, 30]}).to_parquet(target / "races.parquet")
    sources.atomic_json(root / "history.json", {"days": {"2026-10-29": {"directory": "history/2026-10-29/new"}}})
    result = sources.load_history_table("races", "2026-10-28", "2026-10-29")
    assert result[["race_id", "value"]].values.tolist() == [[1, 10], [2, 99], [3, 30]]
    assert sources.load_history_table("races", "2026-10-28", "2026-10-28").race_id.tolist() == [1]


def test_writer_lock_prevents_concurrent_refresh(isolated):
    with sources._writer(sources.pipeline_dir()) as first:
        assert first
        assert sources.refresh_rtv_history(1)["state"] == "busy"


def test_snapshot_generation_contains_only_the_current_capture_and_is_throttled(isolated, monkeypatch):
    from datetime import timedelta
    moment = sources._now().replace(microsecond=0)
    day = date.today().isoformat()
    calls = []
    monkeypatch.setattr(sources, "_now", lambda: moment)
    monkeypatch.setattr(sources.rtv, "RawStore", lambda *a: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(sources.rtv, "snapshot_day", lambda *a, **kw: calls.append(day) or
        {"races": 1, "requests": 6, "ok": 5, "pace_maps": 1})
    monkeypatch.setattr(sources.rtv, "build_snapshot_tables", lambda *a, **kw: {"pace_maps": pd.DataFrame({
        "fetched_at": [(moment-timedelta(hours=1)).isoformat(), (moment+timedelta(seconds=3)).isoformat()],
        "runner_id": [1, 2]})})
    result = sources.capture_rtv_snapshot(day)
    written = pd.read_parquet(sources.pipeline_dir()/result["directory"]/"pace_maps.parquet")
    assert written.runner_id.tolist() == [2]
    assert sources.capture_rtv_snapshot(day)["state"] == "cached"
    assert calls == [day]


def test_matched_text_keys_rtv_text_to_sl_runners(monkeypatch):
    races, runners, canonical = frames()
    races["card_analyst_verdict_text"] = "Test Horse can follow up."
    runners["card_analyst_comment_text"], runners["comment"] = "Won well last time.", "led, ran on"
    tables = {"races": races, "runners": runners}
    monkeypatch.setattr(sources, "load_history_table", lambda name, start, end: tables[name])
    monkeypatch.setattr(sources, "canonical_rows", lambda start, end: canonical)
    text = sources.load_matched_text("2026-10-29", "2026-10-29")
    assert text.drop(columns=["pace_hint", "draw_comment"]).to_dict("records") == [{"race_id": "sl-1", "horse_id": 303,
        "race_date": "2026-10-29", "tf_comment": "Won well last time.", "tf_verdict": "Test Horse can follow up.",
        "close_up": "led, ran on"}]
    assert text[["pace_hint", "draw_comment"]].isna().all(axis=None)
