"""Reader locks must not discard accepted data or stop durable publication."""
import json
import os

import pandas as pd
import pytest

from src.ingestion import storage


def write(kind, path, value):
    if kind == "json":
        storage.atomic_json(path, {"value": value})
    else:
        storage.atomic_parquet(pd.DataFrame({"value": [value]}), path)


@pytest.mark.skipif(os.name != "nt", reason="Transient Windows reader locks")
@pytest.mark.parametrize("kind", ["json", "parquet"])
def test_atomic_publication_retries_a_temporary_windows_reader_lock(tmp_path, monkeypatch, kind):
    path = tmp_path / f"accepted.{kind}"
    write(kind, path, 1)
    original = storage.os.replace
    attempts, delays = [], []
    def briefly_locked(source, target):
        attempts.append((source, target))
        if len(attempts) < 3:
            raise PermissionError("A reader temporarily holds the destination")
        original(source, target)
    monkeypatch.setattr(storage.os, "replace", briefly_locked)
    monkeypatch.setattr(storage.time, "sleep", delays.append)
    write(kind, path, 2)
    result = json.loads(path.read_text())["value"] if kind == "json" else pd.read_parquet(path).value.iloc[0]
    assert result == 2 and len(attempts) == 3 and delays == [.05, .1]
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("kind", ["json", "parquet"])
def test_atomic_publication_keeps_accepted_bytes_if_the_lock_does_not_clear(tmp_path, monkeypatch, kind):
    path = tmp_path / f"accepted.{kind}"
    write(kind, path, 1)
    accepted = path.read_bytes()
    attempts, delays = [], []
    def locked(source, target):
        attempts.append((source, target))
        raise PermissionError("Destination remains locked")
    monkeypatch.setattr(storage.os, "replace", locked)
    monkeypatch.setattr(storage.time, "sleep", delays.append)
    with pytest.raises(PermissionError, match="remains locked"):
        write(kind, path, 2)
    assert path.read_bytes() == accepted and not list(tmp_path.glob("*.tmp"))
    assert len(attempts) == (8 if os.name == "nt" else 1)
    assert sum(delays) <= 2.25
