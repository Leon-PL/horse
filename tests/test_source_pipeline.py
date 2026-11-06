"""Source collection must not alter canonical values or invent identity joins."""
from datetime import date, datetime, timezone
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src import source_pipeline as sources


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


def test_unified_sync_preserves_canonical_frame_and_refreshes_rtv_without_weather(isolated, monkeypatch):
    from src import database
    expected = pd.DataFrame({"horse_id": [123], "odds": [4.5], "finish_position": [1]})
    calls = []
    monkeypatch.setattr(database, "sync_database", lambda **kw: calls.append(kw) or {"state": "ok"})
    monkeypatch.setattr(database, "load_from_database", lambda: expected)
    monkeypatch.setattr(sources, "refresh_rtv_history", lambda **kw: calls.append(kw) or {"state": "ok"})
    actual = sources.sync_database(days_back=3, enrich=False, sources=("sl", "rtv"))
    assert actual is expected
    assert calls[0]["days_back"] == 3 and calls[0]["enrich"] is False and calls[0]["return_report"] is True
    assert calls[1] == {"days_back": 10}
    assert actual.attrs["source_refresh"]["state"] == "ok"


def test_rtv_outage_does_not_prevent_canonical_sync(isolated, monkeypatch, caplog):
    from src import database
    expected = pd.DataFrame({"race_id": [1]})
    monkeypatch.setattr(database, "sync_database", lambda **kw: {"state": "ok"})
    monkeypatch.setattr(database, "load_from_database", lambda: expected)
    def broken(**kwargs):
        raise sources.rtv.RTVTransientError("rate limited")
    monkeypatch.setattr(sources, "refresh_rtv_history", broken)
    assert sources.sync_database(1, sources=("sl", "rtv")) is expected
    assert expected.attrs["source_refresh"]["state"] == "partial"
    assert "rate limited" in caplog.text


def test_app_card_capture_keeps_model_inputs_exact(isolated, monkeypatch):
    from src import data_scraper
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
    sources._publish(root / "history.json", previous)
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
    sources._publish(root / "history.json", {"days": {"2026-10-29": {"directory": "history/2026-10-29/new"}}})
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
