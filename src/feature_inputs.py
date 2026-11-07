"""Fingerprints of feature inputs that live outside the results database.

``data_signature`` hashes the processed frame, which is built from
``data/races.db``. The ATR sectional cache (``data/atr_cache/
atr_sectionals.parquet``) is read *inside* feature engineering instead
(``add_atr_features`` -> ``merge_atr_sectionals``), so neither the data nor
the FE signature saw it: a sectional backfill for dates already in the store
left those rows' features stale with nothing reporting it - the failure class
that produced the phantom +15% place edge (ledger 1, 13).

The store's meta records ``atr_signature = {"cutoff", "sig"}`` when it is
built, and ``scripts/project_status.py`` recomputes it. This module is
deliberately outside the FE import graph, so it does not move
``fe_signature``. The training refresh, ``src.store_refresh``, compares it
before appending.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import pandas as pd

import config


def _sidecar_signature(provider, table, cutoff):
    from src.source_dataset import source_path
    path = source_path(provider, table)
    if path is None or not path.exists():
        return None
    files = []
    for name in ("links_"+table, "observations_"+table):
        candidate = source_path(provider, name)
        if candidate is not None and candidate.exists():
            stat = candidate.stat()
            files.append((str(candidate), stat.st_mtime_ns, stat.st_size))
    if not files:
        return None
    stat = path.stat()
    return _cached_sidecar_signature(str(path), stat.st_mtime_ns, stat.st_size, tuple(files), str(cutoff) if cutoff is not None else None)


@lru_cache(maxsize=64)
def _cached_sidecar_signature(path, modified, size, files, cutoff):
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
    from src.ingestion.rtv_storage import pin_generation
    from src.source_dataset import current_manifest, registry_root, use_manifest
    from src.ingestion.storage import read_json
    overrides = ("RTV_FEATURE_RUNS", "RTV_FEATURE_PRE", "RTV_FEATURE_CARDS", "RTV_FEATURE_TEXT_SCORES")
    if current_manifest() is None and not any(hasattr(config, name) for name in overrides):
        receipt = read_json(registry_root()/"rtv/latest.json")
        if receipt:
            with use_manifest({"sources": {"rtv": receipt}}):
                return rtv_cache_signature(cutoff)
    output = {}
    with pin_generation():
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

ATR_CACHE_FILE = os.path.join(config.DATA_DIR, "atr_cache", "atr_sectionals.parquet")


def atr_cache_signature(cutoff, path: str = ATR_CACHE_FILE) -> str | None:
    """Content hash of the sectional cache rows dated on or before *cutoff*.

    Row order and column order are normalised, so rewriting the same data
    does not change it; any added, removed or edited row at or before the
    cutoff does. ``None`` when the cache does not exist.
    """
    from src.source_dataset import source_path
    path = source_path("atr", "sectionals", path)
    from src.source_dataset import current_manifest, registry_root
    from src.ingestion.storage import read_json
    if current_manifest() is None and str(path) == ATR_CACHE_FILE:
        receipt = read_json(registry_root()/"atr/latest.json")
        if receipt and "sectionals" in receipt["tables"]:
            from src.source_dataset import use_manifest
            with use_manifest({"sources": {"atr": receipt}}):
                return atr_cache_signature(cutoff)
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    sidecar = _sidecar_signature("atr", "sectionals", cutoff)
    return _atr_content_signature(str(path), stat.st_mtime_ns, stat.st_size, str(cutoff), sidecar)


@lru_cache(maxsize=32)
def _atr_content_signature(path, modified, size, cutoff, sidecar):
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
