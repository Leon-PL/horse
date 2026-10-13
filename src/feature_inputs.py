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

import pandas as pd

import config

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
