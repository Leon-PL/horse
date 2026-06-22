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


def data_signature(processed: pd.DataFrame, *, cutoff=None) -> str:
    """Cheap fingerprint of the processed history (optionally up to *cutoff*).

    The append model assumes history is immutable: rows at-or-before the
    store's last date never change. But data backfills/cleanups (e.g. the
    draw backfill, the non-runner purge) edit historical rows *in place*. The
    ``fe_signature`` guard only catches code/config changes, so we also
    fingerprint the data: row count + date span + value sums over a few
    columns (one cheap pass). The sums catch in-place value edits (a draw
    backfill changes ``draw``'s sum) that a row-count check would miss.
    """
    rd = pd.to_datetime(processed["race_date"], errors="coerce")
    sub = processed if cutoff is None else processed[rd <= pd.Timestamp(cutoff)]
    sub_rd = rd if cutoff is None else rd[rd <= pd.Timestamp(cutoff)]
    parts: list = [int(len(sub))]
    if len(sub):
        parts += [str(sub_rd.min()), str(sub_rd.max())]
    for col in ("draw", "won", "finish_position", "num_runners"):
        if col in sub.columns:
            parts.append(float(pd.to_numeric(sub[col], errors="coerce").sum()))
    h = hashlib.sha256(json.dumps(parts, default=str).encode("utf-8"))
    return h.hexdigest()[:16]


def _stamp(store: FeatureStore, fe_sig: str, data_sig: str) -> None:
    m = store.meta()
    m["fe_signature"] = fe_sig
    m["data_signature"] = data_sig
    with open(store.meta_path, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2, sort_keys=True, default=str)


def _rebuild(store: FeatureStore, processed: pd.DataFrame, fe_sig: str) -> pd.DataFrame:
    featured = store.build(processed)
    _stamp(store, fe_sig, data_signature(processed, cutoff=store.last_date()))
    return featured


# Columns the training/eval code needs present in the frame even when they
# aren't model features (settlement, EW pricing, ranking label, BSP/CLV).
_REQUIRED_KEEP = {"num_runners", "handicap", "bsp", "relevance", "finish_time_secs"}


def _model_load_columns(store: FeatureStore) -> list[str] | None:
    """Columns to read when a feature allow-list is active, else None (all).

    Returns meta/target columns + required-for-settlement columns + the
    allow-listed features — everything EXCEPT the numeric feature columns the
    model won't use. Lets training read ~200 of ~1000 columns and avoids
    materialising the full multi-GB frame (the load that freezes low-RAM boxes).
    """
    from src.model import EXCLUDE_COLUMNS, _resolve_feature_allowlist
    allow = _resolve_feature_allowlist()
    path = getattr(store, "featured_path", None)
    if not allow or not path or not os.path.exists(path):
        return None
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pq.read_schema(path)
    keep: list[str] = []
    for field in schema:
        n = field.name
        is_num = (pa.types.is_floating(field.type) or pa.types.is_integer(field.type)
                  or pa.types.is_boolean(field.type))
        is_feature = is_num and n not in EXCLUDE_COLUMNS
        if (not is_feature) or (n in allow) or (n in _REQUIRED_KEEP):
            keep.append(n)
    return keep


def get_featured_dataset(
    *,
    processed: pd.DataFrame | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Return the full featured dataset, building or appending as needed.

    Rebuilds from scratch when the store is cold, ``rebuild=True``, the
    ``fe_signature`` (FE code/config) changed, or the **historical data
    changed beneath the store** (a backfill/cleanup edited rows at-or-before
    the store's last date). Otherwise appends only the newer race dates.

    The append path features NEW ROWS ONLY (ratings + every pandas stage
    seeded from carried state) — history is never recomputed. The
    ``fe_signature`` guard (which folds in ``STORE_SCHEMA_VERSION``) forces a
    full rebuild for any store predating the incremental flip, so the append
    always has carried state to resume from.

    Args:
        processed: pre-processed history; loaded via ``process_data`` if None.
        rebuild: force a full rebuild even if the store looks current.
    """
    store = FeatureStore()
    fe_sig = fe_signature()
    meta = store.meta()

    if processed is None:
        processed = process_data(save=False)

    if rebuild or not store.exists():
        return _rebuild(store, processed, fe_sig)

    if meta.get("fe_signature") != fe_sig:
        logger.info("FeatureStore stale: FE code/config changed — full rebuild")
        return _rebuild(store, processed, fe_sig)

    last = store.last_date()
    if meta.get("data_signature") != data_signature(processed, cutoff=last):
        logger.info(
            "FeatureStore stale: history changed beneath it (backfill/cleanup) "
            "— full rebuild"
        )
        return _rebuild(store, processed, fe_sig)

    store.append(processed)
    _stamp(store, fe_sig, data_signature(processed, cutoff=store.last_date()))
    cols = _model_load_columns(store)
    if cols is not None:
        logger.info("FeatureStore: loading %d of %d columns (allow-list active)",
                    len(cols), len(store.featured_schema_names()))
    return store.load_featured(columns=cols)
