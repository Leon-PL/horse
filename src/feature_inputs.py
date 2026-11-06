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
``fe_signature``; folding the check into ``get_featured_dataset`` is the
proper home and waits for the next rebuild that is needed anyway.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import pandas as pd

import config


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
    return output

ATR_CACHE_FILE = os.path.join(config.DATA_DIR, "atr_cache", "atr_sectionals.parquet")


def atr_cache_signature(cutoff, path: str = ATR_CACHE_FILE) -> str | None:
    """Content hash of the sectional cache rows dated on or before *cutoff*.

    Row order and column order are normalised, so rewriting the same data
    does not change it; any added, removed or edited row at or before the
    cutoff does. ``None`` when the cache does not exist.
    """
    if not os.path.exists(path):
        return None
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
    return h.hexdigest()[:16]
