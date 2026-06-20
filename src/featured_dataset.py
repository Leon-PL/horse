"""Single build path for the featured training dataset.

Builds the featured dataset ONCE via :class:`FeatureStore`, then APPENDS new
race dates instead of re-engineering the whole history on every train. A
feature-signature guard forces a clean rebuild whenever the FE code or its
config flags change, so adopting the store never silently serves stale features.

This is the seam that ``train.py`` / ``app.py`` should consume instead of
calling ``engineer_features`` directly. See ``docs/FE_REFACTOR_PLAN.md`` and
``docs/FE_TAXONOMY.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os

import pandas as pd

import config
from src.data_processor import process_data
from src.feature_store import STORE_SCHEMA_VERSION, FeatureStore

logger = logging.getLogger(__name__)

# Config flags that change FE output. Keep in sync with the flags read at
# runtime by feature_engineer.py / ratings.py (see the audit grep in
# docs/FE_REFACTOR_PLAN.md). A change to any of these invalidates the store.
FE_CONFIG_KEYS = [
    "ROLLING_WINDOWS",
    "GLICKO_ENABLED", "GLICKO_JOCKEY_TRAINER", "GLICKO_DIMENSIONAL",
    "GLICKO_MARGIN", "GLICKO2_ENABLED", "GLICKO2_TAU",
    "GLICKO_C", "GLICKO_RD_INIT", "GLICKO_RD_MIN",
    "TRUESKILL_ENABLED", "COLLATERAL_FORM",
    "TE_WINDOW_DAYS", "TE_EWMA_HALF_LIFE_RACES",
    "ELO_K_BASE", "ELO_K_MIN", "ELO_K_DECAY", "ELO_MOMENTUM_ALPHA",
    "MARGIN_ELO_SCALE", "MARGIN_ELO_REF_DIST", "MARGIN_ELO_DNF_PENALTY",
]

_FE_SOURCE_FILES = ("src/feature_engineer.py", "src/ratings.py")


def fe_signature() -> str:
    """Short hash of the FE code + its config flags + the store schema.

    A mismatch with the value stored in the FeatureStore meta means the
    featured rows on disk were produced by different code/config and must be
    rebuilt.
    """
    h = hashlib.sha256()
    for rel in _FE_SOURCE_FILES:
        path = os.path.join(config.BASE_DIR, rel)
        try:
            with open(path, "rb") as fh:
                h.update(fh.read())
        except OSError:
            logger.warning("fe_signature: could not read %s", rel)
    cfg = {k: getattr(config, k, None) for k in FE_CONFIG_KEYS}
    h.update(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8"))
    h.update(str(STORE_SCHEMA_VERSION).encode("utf-8"))
    return h.hexdigest()[:16]


def _stamp_signature(store: FeatureStore, sig: str) -> None:
    m = store.meta()
    m["fe_signature"] = sig
    with open(store.meta_path, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2, sort_keys=True, default=str)


def get_featured_dataset(
    *,
    processed: pd.DataFrame | None = None,
    rebuild: bool = False,
    fast_append: bool = False,
) -> pd.DataFrame:
    """Return the full featured dataset, building or appending as needed.

    * cold store, or ``rebuild=True``, or a changed ``fe_signature`` -> a full
      rebuild via ``FeatureStore.build`` (identical to a from-scratch
      ``engineer_features`` run);
    * otherwise -> ``append`` any race dates newer than the store, then load.

    Args:
        processed: pre-processed history; loaded via ``process_data`` if None.
        rebuild: force a full rebuild even if the store looks current.
        fast_append: use the rating-only fast append path (parity-tested).
    """
    store = FeatureStore()
    sig = fe_signature()
    stored_sig = store.meta().get("fe_signature")
    stale = store.exists() and stored_sig != sig

    if processed is None:
        processed = process_data(save=False)

    if rebuild or not store.exists() or stale:
        if stale and not rebuild:
            logger.info(
                "FeatureStore stale (fe_signature %s != %s) — full rebuild",
                stored_sig, sig,
            )
        featured = store.build(processed)
        _stamp_signature(store, sig)
        return featured

    store.append(processed, fast=fast_append)
    _stamp_signature(store, sig)
    return store.load_featured()
