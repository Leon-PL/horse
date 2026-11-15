"""Saved results of history computations that depend only on immutable inputs.

Feature stages that read a whole source history (At The Races run scalars, Racing TV run history) recomputed it
in every process: about 40 s per live card and per build worker. Their inputs are immutable dataset versions,
so the result is saved once per (input identity, computing code) and read back afterwards.

``frame(name, key, compute)`` keeps the latest ``KEEP`` results per name under ``data/processed/fe_cache``. The key
must name every input (version ids, or path, mtime and size of an immutable file) and the source hash of the
module that computes it, so a code change can never be served a stale result. ``memo(name, key, compute)``
does the same for small JSON-serialisable values such as source signatures.
"""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

import config

ROOT = Path(config.PROCESSED_DATA_DIR) / "fe_cache"
KEEP = 3          # saved frames per name
MEMO_KEEP = 200   # saved signatures per name


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def _digest(key):
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _prune(folder, suffix, keep=KEEP):
    files = sorted(folder.glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        try:
            old.unlink()
        except OSError:
            pass


def frame(name, key, compute):
    folder = ROOT / name
    target = folder / f"{_digest(key)}.parquet"
    if target.exists():
        try:
            return pd.read_parquet(target)
        except (OSError, ValueError):
            pass
    result = compute()
    try:
        from src.ingestion.storage import atomic_parquet
        folder.mkdir(parents=True, exist_ok=True)
        atomic_parquet(result, target)
        _prune(folder, ".parquet")
    except OSError:
        pass
    return result


def memo(name, key, compute):
    folder = ROOT / name
    target = folder / f"{_digest(key)}.json"
    if target.exists():
        try:
            return json.loads(target.read_text())["value"]
        except (OSError, ValueError, KeyError):
            pass
    value = compute()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"key": key, "value": value}, default=str))
        os.replace(temporary, target)
        _prune(folder, ".json", MEMO_KEEP)
    except OSError:
        pass
    return value
