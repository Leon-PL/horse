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
import re

import numpy as np

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

#: Every module whose code can change a *feature value*. A change to any of
#: these invalidates the store. Completeness is enforced by
#: ``tests/test_fe_signature_coverage.py``, which walks the import graph from
#: the FE entry points and fails on anything that is neither registered here
#: nor exempted below — a missing entry silently serves stale features, which
#: is how the 2026-06 "place-market edge" mirage happened.
_FE_SOURCE_FILES = (
    "src/data_processor.py", "src/feature_engineer.py", "src/ratings.py",
    "src/fe_incremental.py", "src/feature_store.py", "src/featured_dataset.py",
    "src/atr_features.py",
    "src/pedigree_backfill.py", "src/track_config.py", "src/utils.py",
    # Merge/column contract consumed by an FE stage: the sectional joins feed
    # add_atr_features.
    "src/attheraces_scraper.py",
    # weather.py, trainer_locations.py and bha_stewards_scraper.py left this
    # list on 2026-10-13 with the weather and trainer-travel stages: nothing
    # reachable from the FE entry points imports them any more.
)

#: Modules reachable from the FE entry points that provably cannot change a
#: feature value, with the reason. Raw-data access is covered separately by
#: :func:`data_signature`; downstream consumers only read features.
FE_SIGNATURE_EXEMPT: dict[str, str] = {
    "database": "raw-data access only; values covered by data_signature",
    "data_scraper": "raw-data access only; values covered by data_signature",
    "data_collector": "raw-data access only; values covered by data_signature",
    "weather": "sync writes raw weather into the DB (covered by data_signature); "
               "no FE stage calls it since the weather stage was removed",
    "feature_inputs": "fingerprints the ATR cache for staleness checks; reads, "
                      "never changes, a feature value",
    "model": "downstream consumer (column selection only), not a value source",
    "probability_calibration": "downstream prediction calibration; cannot change feature values",
    "bet_analysis": "downstream of FE (bet settlement/reporting)",
    "bet_settlement": "downstream of FE (bet settlement/reporting)",
    "each_way": "downstream of FE (bet settlement/reporting)",
    "betfair": "post-race prices joined at eval time, not an FE input",
}


def _portable_config_value(value):
    """Make a config value independent of where the project is checked out.

    Several hashed modules reference path settings (``BASE_DIR``, ``DATA_DIR``
    …). Hashing them verbatim would change the signature whenever the
    checkout moves, forcing a spurious multi-GB store rebuild and making
    signatures incomparable across machines. The *location* of the data says
    nothing about how features are computed — and the data's contents are
    already fingerprinted by :func:`data_signature` — so paths inside the
    project are reduced to a stable project-relative form. A renamed
    sub-directory still changes the signature; a moved project does not.
    """
    if isinstance(value, str) and os.path.isabs(value):
        try:
            rel = os.path.relpath(value, config.BASE_DIR)
        except ValueError:  # different drive on Windows
            return value
        if not rel.startswith(".."):
            return rel.replace(os.sep, "/")
    return value


def fe_signature() -> str:
    """Short hash of the FE code + its config flags + the store schema.

    A mismatch with the value stored in the FeatureStore meta means the
    featured rows on disk were produced by different code/config and must be
    rebuilt.
    """
    h = hashlib.sha256()
    config_keys = set(FE_CONFIG_KEYS)
    for rel in _FE_SOURCE_FILES:
        path = os.path.join(config.BASE_DIR, rel)
        try:
            with open(path, "rb") as fh:
                # Line endings cannot change a feature value (the tokenizer
                # normalises them), but git's autocrlf rewrites them on any
                # checkout, which would force a spurious full rebuild.
                source = fh.read().replace(b"\r\n", b"\n")
            h.update(rel.encode("utf-8"))
            h.update(source)
            text = source.decode("utf-8")
            config_keys.update(re.findall(r"config\.([A-Z][A-Z0-9_]*)", text))
            config_keys.update(re.findall(
                r"getattr\(config,\s*['\"]([A-Z][A-Z0-9_]*)['\"]", text
            ))
        except OSError:
            # Missing optional sources are part of the signature too.
            h.update(f"missing:{rel}".encode("utf-8"))
            logger.warning("fe_signature: could not read %s", rel)
    cfg = {k: _portable_config_value(getattr(config, k, None)) for k in config_keys}
    h.update(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8"))
    h.update(str(STORE_SCHEMA_VERSION).encode("utf-8"))
    return h.hexdigest()[:16]


def data_signature(processed: pd.DataFrame, *, cutoff=None) -> str:
    """Fingerprint historical input values, not just aggregate checksums.

    Sum-only signatures miss swapped results and edits to odds, pedigree,
    ratings, sectionals, and other inputs. Hash each canonical column in row
    order, bounded chunks at a time. Numeric types are normalised so a later
    missing value promoting integers to floats does not invalidate history.
    One-hot columns are derivable from their raw categories and are omitted so
    a category appearing only in new races does not change the old signature.
    """
    rd = pd.to_datetime(processed["race_date"], errors="coerce")
    sub = processed if cutoff is None else processed[rd <= pd.Timestamp(cutoff)]
    h = hashlib.sha256()
    h.update(str(len(sub)).encode("ascii"))
    categorical = ("going", "race_class", "race_type", "region", "sex", "surface")
    derived_prefixes = tuple(f"{c}_" for c in categorical if c in sub.columns)
    # A processed frame also has season dummies, derived from race_date.
    derived_prefixes += ("season_",)
    columns = sorted(c for c in sub.columns if not (
        c.startswith(derived_prefixes)
        and pd.api.types.is_numeric_dtype(sub[c])
        and sub[c].dropna().isin([0, 1]).all()
    ))
    for col in columns:
        h.update(json.dumps(col).encode("utf-8"))
        series = sub[col]
        for start in range(0, len(series), 131072):
            chunk = series.iloc[start:start + 131072]
            if pd.api.types.is_numeric_dtype(series):
                values = chunk.to_numpy(dtype="float64", na_value=np.nan).copy()
                values[values == 0] = 0.0  # canonicalise signed zero
                values[np.isnan(values)] = np.nan
                h.update(values.tobytes())
            else:
                # String normalisation matches parquet's object serialisation.
                values = chunk.astype("string")
                h.update(pd.util.hash_pandas_object(values, index=False).values.tobytes())
    return h.hexdigest()[:16]


def _atr_signature(cutoff) -> dict:
    """Fingerprint of the ATR sectional cache up to *cutoff*.

    The cache is read inside FE (``add_atr_features``), outside both other
    signatures, so a sectional backfill used to leave stored rows stale with
    nothing noticing (ledger 13).
    """
    from src.feature_inputs import atr_cache_signature

    return {"cutoff": str(pd.Timestamp(cutoff).date()), "sig": atr_cache_signature(cutoff)}


def _stamp(store: FeatureStore, fe_sig: str, data_sig: str) -> None:
    m = store.meta()
    m["fe_signature"] = fe_sig
    m["data_signature"] = data_sig
    if store.last_date() is not None:
        m["atr_signature"] = _atr_signature(store.last_date())
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

    atr = meta.get("atr_signature")
    if isinstance(atr, dict) and atr.get("cutoff"):
        if _atr_signature(atr["cutoff"])["sig"] != atr.get("sig"):
            logger.info("FeatureStore stale: ATR sectional cache changed beneath it — full rebuild")
            return _rebuild(store, processed, fe_sig)

    store.append(processed)
    _stamp(store, fe_sig, data_signature(processed, cutoff=store.last_date()))
    cols = _model_load_columns(store)
    if cols is not None:
        logger.info("FeatureStore: loading %d of %d columns (allow-list active)",
                    len(cols), len(store.featured_schema_names()))
    return store.load_featured(columns=cols)
