"""Regression checks for failures exposed by the installed source catch-up."""
from datetime import datetime, timedelta, timezone
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.ingestion import attheraces as atr
from src.ingestion import auxiliary
from src.betfair import attach_bsp
from src.ingestion.storage import atomic_json, writer_lock


def price(horse, **extra):
    return {"race_date": "2026-11-03", "track": "Kempton", "off_time": "12:00",
            "horse_name": horse, "bsp": 4., **extra}


def test_archive_and_public_runner_ids_round_trip_without_losing_timed_prices(tmp_path):
    existing = pd.DataFrame([price("A", bf_runner_id=123, ltp_60s=5.)])
    incoming = pd.DataFrame([price("B", selection_id="45168568"), price("C")])
    merged, receipt = auxiliary.merge_public_bsp(existing, incoming)
    destination = tmp_path/"prices.parquet"
    auxiliary.atomic_parquet(merged, destination)
    restored = pd.read_parquet(destination).set_index("horse_name")
    assert restored.loc["A", "bf_runner_id"] == 123
    assert restored.loc["B", "bf_runner_id"] == 45168568
    assert pd.isna(restored.loc["C", "bf_runner_id"])
    assert restored.loc["A", "ltp_60s"] == 5.
    assert pd.isna(restored.loc["B", "ltp_60s"])
    assert receipt["new_rows"] == 2


def test_source_merge_and_feature_join_share_betfair_horse_keys():
    existing = pd.DataFrame([price("George's Lad", bsp=np.nan)])
    incoming = pd.DataFrame([price("Georges Lad")])
    merged, receipt = auxiliary.merge_public_bsp(existing, incoming)
    assert len(merged) == 1 and receipt["filled_prices"] == 1
    assert attach_bsp(existing.drop(columns="bsp"), merged, cols=("bsp",)).bsp.tolist() == [4.]


def test_invalid_runner_ids_are_rejected_before_publication():
    with pytest.raises(ValueError):
        auxiliary.merge_public_bsp(pd.DataFrame([price("A", bf_runner_id=123)]),
                                   pd.DataFrame([price("B", selection_id="not-an-id")]))


def test_interrupted_price_publication_preserves_the_accepted_cache(tmp_path, monkeypatch):
    destination = tmp_path/"bsp.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(destination, index=False)
    def interrupted(frame, path, **kwargs):
        path.write_bytes(b"incomplete parquet")
        raise RuntimeError("interrupted")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        auxiliary.atomic_parquet(pd.DataFrame({"value": [2]}), destination)
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


def test_bsp_receipt_skips_unchanged_work_but_detects_content_and_canonical_repairs(tmp_path, monkeypatch):
    from src.ingestion import betfair_public as public
    monkeypatch.setattr(auxiliary.config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(auxiliary.config, "PROCESSED_DATA_DIR", str(tmp_path/"processed"))
    raw = tmp_path/"prices.csv"
    raw.write_text("price-v1", encoding="utf-8")
    monkeypatch.setattr(public, "raw_path", lambda *a, **kw: raw)
    monkeypatch.setattr(public, "download", lambda *a, **kw: {"failed": 0, "cached": 2})
    fields = pd.DataFrame([price("A", race_id="1", num_runners=1)]).drop(columns="bsp")
    monkeypatch.setattr(auxiliary, "canonical_fields", lambda *a: fields)
    parsed = []
    def parse(*a):
        parsed.append(a)
        return pd.DataFrame([price("A", selection_id="123")])
    monkeypatch.setattr(public, "parse_file", parse)
    first = auxiliary.refresh_bsp("2026-11-03", "2026-11-03")
    assert first["complete_fields"] == 1
    assert auxiliary.refresh_bsp("2026-11-03", "2026-11-03")["state"] == "cached"
    assert len(parsed) == 2

    # Preserve size and mtime while replacing source content.
    original = raw.stat()
    raw.write_text("price-v2", encoding="utf-8")
    os.utime(raw, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert auxiliary.refresh_bsp("2026-11-03", "2026-11-03")["state"] == "ok"
    assert len(parsed) == 4

    fields.loc[0, "horse_name"] = "A repaired identity"
    changed = auxiliary.refresh_bsp("2026-11-03", "2026-11-03")
    assert len(parsed) == 6 and changed["pending_fields"] == 1

    # A separately repaired archive invalidates the receipt as well.
    path = tmp_path/"processed/betfair_bsp.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "bsp"] = 8.
    frame.to_parquet(path, index=False)
    auxiliary.refresh_bsp("2026-11-03", "2026-11-03")
    assert len(parsed) == 8
    assert pd.read_parquet(path).bsp.iloc[0] == 8.
