"""Regression checks for failures exposed by the installed source catch-up."""
from datetime import datetime, timedelta, timezone
import json

import pandas as pd
import pytest

from src.ingestion import attheraces as atr
from src.ingestion import auxiliary
from src.ingestion.storage import atomic_json, atomic_parquet, writer_lock


def test_interrupted_price_publication_preserves_the_accepted_cache(tmp_path, monkeypatch):
    destination = tmp_path/"bsp.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(destination, index=False)
    def interrupted(frame, path, **kwargs):
        path.write_bytes(b"incomplete parquet")
        raise RuntimeError("interrupted")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_parquet(pd.DataFrame({"value": [2]}), destination)
    assert pd.read_parquet(destination).value.tolist() == [1]
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.fixture
def atr_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(auxiliary.config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(atr, "load_atr_cache", lambda: pd.DataFrame())
    monkeypatch.setattr(atr, "_load_no_data_race_keys", lambda: set())
    monkeypatch.setattr(auxiliary, "canonical_fields", lambda *a: pd.DataFrame([
        {"race_date": "2026-11-03", "track": "Kempton", "off_time": "12:00"}]))
    now = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(auxiliary, "_utc_now", lambda: now)
    return tmp_path/"source_refresh/atr_recovery.json", now


def test_blocked_atr_retains_queue_and_next_refresh_respects_cooldown(atr_environment, monkeypatch):
    path, now = atr_environment
    calls = []
    def blocked(frame, **kwargs):
        calls.append(kwargs)
        return {"fetched_races": 0, "aborted_blocked": True}
    monkeypatch.setattr(atr, "backfill_atr_sectionals_for_races", blocked)
    first = auxiliary.refresh_atr("2026-11-03", "2026-11-03")
    second = auxiliary.refresh_atr("2026-11-03", "2026-11-03")
    saved = json.loads(path.read_text())
    assert first["state"] == second["state"] == "partial"
    assert saved["pending_races"] == 1 and saved["covered_through"] is None
    assert saved["retry_after"] == (now+timedelta(hours=1)).isoformat()
    assert second["reason"] == "Provider retry cooldown"
    assert calls == [{"max_races": 50, "max_block_retries": 1}]


def test_expired_atr_cooldown_resumes_the_original_gap(atr_environment, monkeypatch):
    path, now = atr_environment
    atomic_json(path, {"start": "2026-06-21", "pending_races": 1,
                      "retry_after": (now-timedelta(seconds=1)).isoformat()})
    monkeypatch.setattr(atr, "backfill_atr_sectionals_for_races", lambda frame, **kw: {"fetched_races": 1, "resolved_races": 1})
    result = auxiliary.refresh_atr("2026-11-03", "2026-11-03")
    saved = json.loads(path.read_text())
    assert result["state"] == "ok" and result["start"] == "2026-06-21"
    assert saved["covered_through"] == "2026-11-03" and "retry_after" not in saved


def test_batched_atr_keys_preserve_the_existing_join_contract():
    fields = pd.DataFrame({"race_date": ["2026-11-03", "2026-11-03 00:00:00", "invalid", "2024-02-29"],
                           "track": [" KEMPTON ", "Wolverhampton", "Down Royal", "Lingfield"],
                           "off_time": ["12:30:00", "1630", "09:00", "9:05"]})
    expected = [atr._race_key(row.race_date, row.track, row.off_time) for row in fields.itertuples()]
    assert auxiliary._atr_race_keys(fields, atr).tolist() == expected


def test_normal_refresh_reports_the_active_full_backfill_without_starting_another(atr_environment, monkeypatch):
    path, _ = atr_environment
    root = path.parent.parent/'atr_cache/rebuild_v2'
    atomic_json(root/'status.json', {'state':'waiting','races':{'data':30,'pending':100,'retry':1},
                                   'saved_rows':248,'total_races':131})
    monkeypatch.setattr(auxiliary,'canonical_fields',lambda *a:pytest.fail('Second ATR collector started'))
    with writer_lock(root/'.worker.lock') as acquired:
        assert acquired
        report = auxiliary.refresh_atr('2026-11-03','2026-11-03')
    assert report['state'] == 'partial' and report['fetched_races'] == 0
    assert report['pending_races'] == 101 and report['backfill']['saved_rows'] == 248
