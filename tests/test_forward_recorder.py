from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from io import StringIO
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.forward_recorder import Recorder, due_events, recorder_lock

NOW = datetime(2026,10,30,12,tzinfo=timezone.utc)


def fake_study(tmp_path):
    (tmp_path/"events").mkdir()
    selected = {}
    study = SimpleNamespace(path=tmp_path, protocol={"first_eligible_date":"2026-10-30", "horizon_seconds":300,
        "model_id":"fixed"}, verify=lambda:None, selected=lambda identity:selected.get(identity),
        selected_records=lambda:iter(selected.values()))
    return study, selected


def event(study, identity, start):
    (study.path/"events"/(identity+".json")).write_text(json.dumps({"event_id":identity,
        "first_start":start.isoformat(), "first_seen_at":(NOW-timedelta(hours=1)).isoformat()}))


def test_only_predeclared_time_and_first_schedule_control_due_events(tmp_path):
    study, selected = fake_study(tmp_path)
    event(study,"1",NOW+timedelta(minutes=7))
    event(study,"2",NOW+timedelta(minutes=20))
    event(study,"3",NOW+timedelta(minutes=4))
    event(study,"4",NOW-timedelta(days=1))
    due, missed = due_events(study,NOW)
    assert [v["event_id"] for v in due] == ["1"]
    assert [v["event_id"] for v in missed] == ["3"]
    selected["1"] = {"capture_id":"success"}
    assert due_events(study,NOW)[0] == []


def test_recorder_restarts_without_unbounded_retries_and_records_discovery_errors(tmp_path):
    study,_ = fake_study(tmp_path)
    event(study,"1",NOW+timedelta(minutes=7))
    calls = []
    def capture(*args,**kwargs):
        calls.append(kwargs)
        return {"paired_available":False}
    def discover(*args,**kwargs):
        raise RuntimeError("feed unavailable")
    recorder = Recorder(study,clock=lambda:NOW,discover=discover,capture=capture,max_attempts=1)
    first = recorder.tick()
    recorder.close()
    assert first["launched_attempts"] == 1
    assert calls[0]["wait_for_issue_window"]
    restarted = Recorder(study,clock=lambda:NOW,discover=discover,capture=capture,max_attempts=1)
    second = restarted.tick()
    restarted.close()
    assert second["launched_attempts"] == len(calls) == 1
    assert second["recent_errors"][0]["stage"] == "discovery"
    assert (study.path/"recorder_heartbeat.json").exists()


def test_discovery_only_never_launches_a_forecast(tmp_path):
    study,_ = fake_study(tmp_path)
    event(study,"1",NOW+timedelta(minutes=7))
    recorder = Recorder(study,clock=lambda:NOW,discover=lambda *a,**kw:None,
        capture=lambda *a,**kw:pytest.fail("Discovery must not infer or wait"))
    assert recorder.tick(discover_only=True)["mode"] == "discovery_only"
    recorder.close()


def test_duplicate_recorder_is_refused_and_lock_is_released(tmp_path):
    study,_ = fake_study(tmp_path)
    with recorder_lock(study):
        with pytest.raises(RuntimeError,match="already owns"):
            with recorder_lock(study):
                pass
    with recorder_lock(study):
        pass


def test_slow_rtv_capture_cannot_occupy_forecast_workers(tmp_path):
    from threading import Event
    study, _ = fake_study(tmp_path)
    event(study, "1", NOW+timedelta(minutes=7))
    release, started, forecast = Event(), Event(), Event()
    def source():
        started.set()
        release.wait(5)
        raise RuntimeError("RTV unavailable")
    recorder = Recorder(study, clock=lambda: NOW, discover=lambda *a, **kw: None,
        capture=lambda *a, **kw: forecast.set(), source_capture=source, workers=1)
    try:
        recorder.tick()
        assert started.wait(2)
        assert forecast.wait(2)
        assert not release.is_set()
        release.set()
        with pytest.raises(RuntimeError, match="RTV unavailable"):
            recorder.source_future.result(timeout=2)
        heartbeat = recorder.tick()
        assert heartbeat["rtv_capture"]["state"] == "error"
        assert heartbeat["recent_errors"][-1]["stage"] == "rtv_snapshot"
    finally:
        release.set()
        recorder.close()


def test_absent_later_api_rows_do_not_erase_verified_withdrawals(tmp_path):
    from src.forward_recorder import collect_withdrawals
    study,_ = fake_study(tmp_path)
    folder = study.path/"outcomes"
    folder.mkdir()
    previous = pd.DataFrame([{"event_id":"old","horse_key":"Horse","confirmed_withdrawn":True,
        "runner_id":"3","market_id":"2","source":"old.json","source_sha256":"preserved","observed_at":NOW.isoformat()}])
    previous.to_parquet(folder/"withdrawals.parquet",index=False)
    actual = collect_withdrawals(study,client=None,clock=lambda:NOW)
    pd.testing.assert_frame_equal(actual,previous)


@pytest.mark.parametrize("append_safe", [True, False])
def test_refresh_fingerprints_canonical_csv_and_never_starts_a_rebuild(tmp_path, monkeypatch, append_safe):
    import config
    from src import database, pedigree_backfill, store_refresh
    from src.data_processor import process_data
    from src.featured_dataset import data_signature
    from src.forward_recorder import refresh_history
    from src import source_pipeline
    monkeypatch.setattr(source_pipeline, "refresh_rtv_history", lambda **kwargs: {"state": "ok"})
    raw = pd.DataFrame({"race_id":["123"], "race_date":["2026-10-28"],
        "horse_name":["Example"], "finish_position":[1], "form":["001"]})
    monkeypatch.setattr(config, "RAW_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pedigree_backfill, "apply_pedigree", lambda frame:frame)
    def sync(*, days_back, enrich):
        assert days_back == 3 and enrich is False
        raw.to_csv(tmp_path/"race_results.csv", index=False)
        return raw.copy()
    monkeypatch.setattr(database, "sync_database", sync)
    expected = process_data(df=pd.read_csv(StringIO(raw.to_csv(index=False))), save=False)
    assert data_signature(expected) != data_signature(process_data(df=raw, save=False))
    def append(frame):
        pd.testing.assert_frame_equal(frame, expected)
        return append_safe
    monkeypatch.setattr(store_refresh, "append_current_store", append)
    monkeypatch.setattr(store_refresh, "get_training_dataset", lambda **kw:pytest.fail("Recorder cannot start a rebuild"))
    verified = []
    study = SimpleNamespace(verify=lambda:verified.append(True))
    if append_safe:
        refresh_history(study)
        assert len(verified) == 2
    else:
        with pytest.raises(RuntimeError, match="maintenance rebuild"):
            refresh_history(study)
        assert len(verified) == 1
