"""A failed ATR publication must retain its last accepted cache and skip-list."""
import json

import pandas as pd
import pytest

from src.ingestion import attheraces as atr
from src.ingestion import atr_tabs
from src.ingestion import storage as source_storage


def test_failed_atr_write_keeps_previous_cache(tmp_path, monkeypatch):
    path = tmp_path/"atr.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(path, index=False)
    monkeypatch.setattr(atr, "ATR_CACHE_FILE", str(path))
    monkeypatch.setattr(atr_tabs, 'ATR_CACHE_FILE', str(path))
    def interrupted(frame, destination, **kwargs):
        destination.write_bytes(b"incomplete parquet")
        raise RuntimeError("interrupted")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        atr.save_atr_cache(pd.DataFrame({"value": [2]}))
    assert pd.read_parquet(path).value.tolist() == [1]
    assert not list(tmp_path.glob("*.tmp"))


def test_silent_partial_parquet_is_rejected_before_publication(tmp_path, monkeypatch):
    path = tmp_path/"atr.parquet"
    original = pd.DataFrame.to_parquet
    pd.DataFrame({"value": [1]}).to_parquet(path, index=False)
    monkeypatch.setattr(atr, "ATR_CACHE_FILE", str(path))
    monkeypatch.setattr(atr_tabs, 'ATR_CACHE_FILE', str(path))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda frame, destination, **kwargs:
                        original(frame.iloc[:0], destination, **kwargs))
    with pytest.raises(ValueError, match="row count"):
        atr.save_atr_cache(pd.DataFrame({"value": [2]}))
    assert pd.read_parquet(path).value.tolist() == [1]


def test_failed_no_data_write_keeps_previous_skip_list(tmp_path, monkeypatch):
    path = tmp_path/"absent.json"
    path.write_text(json.dumps({"race_keys": ["confirmed-absent"]}))
    monkeypatch.setattr(atr, "ATR_NO_DATA_FILE", str(path))
    def interrupted(value, stream, **kwargs):
        stream.write("{")
        raise RuntimeError("interrupted")
    monkeypatch.setattr(source_storage.json, "dump", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        atr._save_no_data_race_keys({"confirmed-absent", "another"})
    assert atr._load_no_data_race_keys() == {"confirmed-absent"}
    assert not list(tmp_path.glob("*.tmp"))
