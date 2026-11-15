"""Fingerprints of the selected dataset's ATR and Racing TV tables.

``data_signature`` hashes the processed frame (results, pedigree, FX). ATR
and Racing TV observations are read *inside* feature engineering, so neither
that nor the FE signature saw them: a sectional backfill for dates already in
the store left those rows' features stale with nothing reporting it - the
failure class that produced the phantom +15% place edge (ledger 1, 13).

These hash the selected dataset's tables, and their identity and timing
sidecars, up to a cutoff. The store's meta records them when it is built;
``src.store_refresh`` compares them before appending and
``scripts/project_status.py`` against the last committed dataset. This module
is deliberately outside the FE import graph, so it does not move
``fe_signature``.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import pandas as pd


def _sidecar_signature(provider, table, cutoff):
    from src.source_dataset import source_path
    path = source_path(provider, table)
    if not path.exists():
        return None
    files = []
    for name in ("links_"+table, "observations_"+table):
        candidate = source_path(provider, name)
        if candidate.exists():
            stat = candidate.stat()
            files.append((str(candidate), stat.st_mtime_ns, stat.st_size))
    if not files:
        return None
    stat = path.stat()
    return _cached_sidecar_signature(str(path), stat.st_mtime_ns, stat.st_size, tuple(files), str(cutoff) if cutoff is not None else None)


def _saved(name, compute, *key):
    """Disk memo of a signature: the key names immutable files (path, mtime, size) and the cutoff."""
    from src import fe_cache
    return fe_cache.memo(name, [*key, fe_cache.source_hash(__file__)], compute)


@lru_cache(maxsize=64)
def _cached_sidecar_signature(path, modified, size, files, cutoff):
    return _saved("sidecar_signature", lambda: _sidecar_digest(path, files, cutoff), path, modified, size, files, cutoff)


def _sidecar_digest(path, files, cutoff):
    import pyarrow.parquet as pq
    columns = pq.read_schema(path).names
    date_column = "available_date" if "available_date" in columns else "race_date" if "race_date" in columns else None
    raw = pd.read_parquet(path, columns=[date_column]) if date_column else pd.read_parquet(path, columns=[])
    dated = raw["available_date"] if "available_date" in raw else raw.get("race_date")
    selected = raw.index[pd.to_datetime(dated).le(pd.Timestamp(cutoff))] if dated is not None and cutoff is not None else raw.index
    digest = hashlib.sha256()
    found = False
    for sidecar, _, _ in files:
        values = pd.read_parquet(sidecar)
        values = values.loc[values.source_row.isin(selected)].sort_values("source_row")
        # Row ordinals locate receipts; identities/times are the content.
        values = values.drop(columns="source_row")
        digest.update(pd.util.hash_pandas_object(values, index=False).values.tobytes())
        found = True
    return digest.hexdigest()[:16] if found else None


@lru_cache(maxsize=32)
def _rtv_part_signature(path, modified, size, cutoff):
    return _saved("rtv_signature", lambda: _rtv_part_digest(path, cutoff), path, modified, size, cutoff)


def _rtv_part_digest(path, cutoff):
    frame = pd.read_parquet(path)
    if cutoff is not None and not frame.empty:
        dated = frame["available_date"] if "available_date" in frame else frame.race_date
        frame = frame.loc[pd.to_datetime(dated).le(pd.Timestamp(cutoff))]
    keys = [c for c in ("race_date", "race_id", "horse_id", "track_key", "off_key", "horse_key", "observed_at") if c in frame]
    frame = frame.sort_values(keys, kind="stable") if keys else frame
    h = hashlib.sha256(str(len(frame)).encode())
    h.update(pd.util.hash_pandas_object(frame[sorted(frame.columns)].astype("string"), index=False).values.tobytes())
    return h.hexdigest()[:16]


def rtv_cache_signature(cutoff=None) -> dict:
    """Hash only values available to the dated history; include snapshots causally."""
    from src.rtv_features import runs_path, pre_path, cards_path
    from src.rtv_text import scores_path
    output = {}
    paths = (("runs", runs_path()), ("pre", pre_path()), ("cards", cards_path()), ("text", scores_path()))
    # Cards and comment scores describe their own race, so they are hashed by race date.
    for name, path in paths:
        if not path.exists():
            output[name] = None
            continue
        stat = path.stat()
        output[name] = _rtv_part_signature(str(path.resolve()), stat.st_mtime_ns, stat.st_size, str(cutoff) if cutoff is not None else None)
        table = "scores" if name == "text" else name
        sidecar = _sidecar_signature("rtv", table, cutoff)
        if sidecar is not None:
            output[name+"_identity_timing"] = sidecar
    return output

def atr_cache_signature(cutoff) -> str | None:
    """Content hash of the selected ATR rows dated on or before *cutoff*.

    Row order and column order are normalised, so rewriting the same data
    does not change it; any added, removed or edited row at or before the
    cutoff does. ``None`` when the selected dataset has no ATR table.
    """
    from src.source_dataset import source_path
    path = source_path("atr", "sectionals")
    if not path.exists():
        return None
    stat = path.stat()
    sidecar = _sidecar_signature("atr", "sectionals", cutoff)
    return _atr_content_signature(str(path), stat.st_mtime_ns, stat.st_size, str(cutoff), sidecar)


@lru_cache(maxsize=32)
def _atr_content_signature(path, modified, size, cutoff, sidecar):
    return _saved("atr_signature", lambda: _atr_content_digest(path, cutoff, sidecar), path, modified, size, cutoff,
                  sidecar)


def _atr_content_digest(path, cutoff, sidecar):
    cache = pd.read_parquet(path)
    dates = pd.to_datetime(cache["race_date"], errors="coerce")
    sub = cache[dates <= pd.Timestamp(cutoff)]
    sub = sub[sorted(sub.columns)]
    keys = [c for c in ("race_date", "course", "off_time", "horse_name", "horse_key")
            if c in sub.columns]
    if keys:
        sub = sub.sort_values(keys, kind="stable")
    h = hashlib.sha256(str(len(sub)).encode("ascii"))
    h.update(pd.util.hash_pandas_object(sub.astype("string"), index=False).values.tobytes())
    if sidecar is not None:
        h.update(sidecar.encode())
    return h.hexdigest()[:16]
