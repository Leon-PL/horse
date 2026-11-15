"""Signatures and metadata stamps that keep the feature store current.

``fe_signature`` fingerprints the feature code and its config, ``data_signature``
the processed history, and the ATR and Racing TV helpers the caches that feature
stages read directly. ``_stamp`` records them in the store's metadata.

The training refresh (``src.store_refresh.get_training_dataset``) and the
batched builder (``src/store_build.py``) use them to decide
between appending new dates and rebuilding. The older in-memory entry point,
``get_featured_dataset``, was retired on 2026-11-07. See
``docs/architecture/FE_TAXONOMY.md``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import tempfile

import numpy as np

import pandas as pd

import config
from src.feature_store import STORE_SCHEMA_VERSION, FeatureStore

logger = logging.getLogger(__name__)

# Config flags that change FE output. Keep in sync with the flags read at
# runtime by feature_engineer.py / ratings.py (see the audit grep in
# docs/archive/architecture/FE_REFACTOR_PLAN.md). A change to any of these invalidates the store.
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
    "src/source_contracts.py", "src/source_dataset.py",
    "src/verified_race_metadata.py", "reference/verified_race_types.json",
    "reference/verified_runner_ratings.json", "reference/connection_aliases.json",
    "src/prize_currency_features.py", "reference/track_prize_currency.json",
    "src/data_processor.py", "src/feature_engineer.py", "src/ratings.py",
    "src/feature_schema.py",
    "src/feature_context.py",
    "src/foreign_form.py",
    "src/trueskill_kernel.py",
    "src/result_margins.py", "src/runner_identity.py",
    "src/result_status.py", "src/rating_reuse.py",
    "src/feature_semantics.py", "src/surface_features.py",
    "src/established_form_features.py",
    "src/prior_market_features.py", "src/forecast_evaluation.py",
    "src/store_build.py",
    "src/fe_incremental.py", "src/feature_store.py", "src/featured_dataset.py",
    "src/atr_features.py",
    "src/rtv_features.py", "src/rtv_text.py",
    "src/track_config.py", "src/utils.py",
    # Which numeric features the store keeps; saved history results (keyed by inputs and code).
    "src/store_columns.py", "src/fe_cache.py",
    # Column and cache contract read by the sectional join in atr_features.
    "src/ingestion/atr_tabs.py",
    # Raw collection modules are exempt below because accepted source values
    # are fingerprinted separately. BHA report features were removed from the
    # pipeline on 2026-11-15 (research 109: no model reads them, 42% of build time).
)

#: Modules reachable from the FE entry points that provably cannot change a
#: feature value, with the reason. Raw-data access is covered separately by
#: :func:`data_signature`; downstream consumers only read features.
FE_SIGNATURE_EXEMPT: dict[str, str] = {
    "ingestion.publication": "publishes immutable source observations; values and crosswalks covered by accepted source hashes",
    "database": "raw-data access only; values covered by data_signature",
    "identity_schema": "source-table and repair-receipt migrations only; accepted values covered by data_signature",
    "ingestion": "collection namespace only; no feature calculations or import-time collection",
    "ingestion.pipeline": "source orchestration only; published observations covered by data and RTV signatures",
    "ingestion.auxiliary": "collects raw ATR/BSP only; accepted values covered by source-content signatures",
    "ingestion.storage": "source locking, raw retention and atomic publication only",
    "ingestion.attheraces": "raw ATR collector only; observations covered by ATR content signatures",
    "ingestion.atr_jumps": "raw jumps collection only; observations covered by ATR content signatures",
    "ingestion.rtv_storage": "locates the collector's own RTV files for publication; features read published versions covered by rtv_cache_signature",
    "ingestion.fx": "refreshes the committed exchange-rate reference; published FX files covered by prize_reference_signature",
    "ingestion.sporting_life": "raw-data access only; accepted observations covered by data_signature",
    "ingestion.racing_tv": "collects raw RTV payloads; published observations covered by rtv_cache_signature",
    "ingestion.betfair_history": "collects historical market files; accepted prices are raw observations fingerprinted by training/evaluation",
    "ingestion.betfair_credentials": "encrypted historical-data login setup only; cannot change feature values",
    "ingestion.betfair_archives": "imports native market archives; accepted evaluation prices are fingerprinted separately",
    "ingestion.betfair_streams": "parses native market streams; accepted evaluation prices are fingerprinted separately",
    "ingestion.pedigree": "collects and stores native pedigree records; accepted pedigree values enter data_signature",
    "ingestion.results": "collects and stores native result records; accepted result values enter data_signature",
    "ingestion.foreign_form": "collector only; accepted overseas starts are fingerprinted separately",
    "ingestion.rtv_observations": "source-stage materialisation only; published run/card/text values are covered by rtv_cache_signature",
    "ingestion.bha_stewards": "source collection only; no feature stage reads BHA observations since 2026-11-15",
    "ingestion.bha_reports": "source PDF extraction only; accepted extracted text is fingerprinted as BHA source content",
    "wf_eval": "evaluation and source text-fit scheduling; published comment scores are covered by rtv_cache_signature",
    "clv": "downstream evaluation metrics only; cannot change feature values",
    "ingestion.weather": "sync writes raw weather into the DB (covered by data_signature); "
               "no FE stage calls it since the weather stage was removed",
    "feature_inputs": "fingerprints the ATR cache for staleness checks; reads, "
                      "never changes, a feature value",
    "model": "downstream consumer (column selection only), not a value source",
    "probability_calibration": "downstream prediction calibration; cannot change feature values",
    "race_objectives": "win-model training loss (057); downstream of FE, cannot change feature values",
    "feature_selection": "research input-selection harness; reads features, cannot change their values",
    "bet_analysis": "downstream of FE (bet settlement/reporting)",
    "bet_settlement": "downstream of FE (bet settlement/reporting)",
    "each_way": "downstream of FE (bet settlement/reporting)",
    "betfair": "post-race prices joined at eval time, not an FE input",
    "source_prices": "manifest-selected post-race prices used only in training/evaluation",
    "source_health": "consumer readiness reporting only; cannot change feature values",
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


_CONFIG_KEY = re.compile(r"[A-Z][A-Z0-9_]*")
# (path, raw-bytes digest) -> (normalised source, config keys). Parsing every
# FE source takes most of a second and serving asks for the signature often;
# hashing the raw bytes is cheap and, unlike mtimes, cannot go stale.
_SEMANTIC_CACHE: dict[tuple[str, bytes], tuple[bytes, set[str]]] = {}


def _semantic_source(rel: str, source: bytes) -> tuple[bytes, set[str]]:
    """The part of a source that can change a feature value, and its config keys.

    Comments, docstrings, blank lines and formatting cannot change a feature
    value, but hashing raw bytes made every such edit force a ~56-minute store
    rebuild. Python hashes as its AST without docstrings or line numbers, JSON
    as its canonical parsed form. A file that does not parse hashes as raw
    text, so a broken edit still changes the signature. The AST dump can
    differ between Python minor versions; an upgrade may force one rebuild.
    """
    # Line endings cannot change a feature value either, but git's autocrlf
    # rewrites them on checkout.
    source = source.replace(b"\r\n", b"\n")
    try:
        if rel.endswith(".json"):
            value = json.loads(source)
            return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"), set()
        if not rel.endswith(".py"):
            return source, set()
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return source, set()
    keys = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "config" and _CONFIG_KEY.fullmatch(node.attr)):
            keys.add(node.attr)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name) and node.args[0].id == "config"
                and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)
                and _CONFIG_KEY.fullmatch(node.args[1].value)):
            keys.add(node.args[1].value)
    return ast.dump(tree).encode("utf-8"), keys


def fe_signature() -> str:
    """Short hash of the FE code + its config flags + the store schema.

    A mismatch with the value stored in the FeatureStore meta means the
    featured rows on disk were produced by different code/config and must be
    rebuilt. Only edits that can change a value count; see
    :func:`_semantic_source`.
    """
    h = hashlib.sha256()
    config_keys = set(FE_CONFIG_KEYS)
    for rel in _FE_SOURCE_FILES:
        path = os.path.join(config.BASE_DIR, rel)
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
            key = (path, hashlib.sha256(raw).digest())
            if key not in _SEMANTIC_CACHE:
                _SEMANTIC_CACHE[key] = _semantic_source(rel, raw)
            source, keys = _SEMANTIC_CACHE[key]
            h.update(rel.encode("utf-8"))
            h.update(source)
            config_keys.update(keys)
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


def _rtv_signature(cutoff) -> dict:
    from src.feature_inputs import rtv_cache_signature
    return {"cutoff": str(pd.Timestamp(cutoff).date()), "sig": rtv_cache_signature(cutoff)}


def _stamp(store: FeatureStore, fe_sig: str, data_sig: str) -> None:
    """Atomically stamp metadata while retaining the dataset operation's lock."""
    with store._access():
        m = store.meta()
        m["fe_signature"] = fe_sig
        m["data_signature"] = data_sig
        from src.source_dataset import require_manifest
        m["dataset_manifest"] = require_manifest()
        from src.prize_currency_features import reference_signature
        m["prize_reference_signature"] = reference_signature()
        last = store.last_date()
        if last is not None:
            from src.foreign_form import signature as foreign_form_signature
            m["atr_signature"] = _atr_signature(last)
            m["rtv_signature"] = _rtv_signature(last)
            m["foreign_form_signature"] = foreign_form_signature(last)
        fd, temporary = tempfile.mkstemp(prefix=".meta-", suffix=".json", dir=store.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(m, fh, indent=2, sort_keys=True, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary, store.meta_path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)


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
