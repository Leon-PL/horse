"""Training-store refresh without loading two copies of every historical feature.

Feature values and carried state come from the existing native implementation.
Only parquet consolidation is streamed; all five files still commit together.
"""
from pathlib import Path
import logging
from datetime import datetime, timezone

import pandas as pd

from src.data_processor import process_data
from src.feature_store import FeatureStore, STORE_SCHEMA_VERSION, _stringify_objects
from src.featured_dataset import (
    _atr_signature, _rtv_signature, _model_load_columns, _stamp, data_signature, fe_signature,
)
from src.parquet_io import combine_parquet_parts
from src.source_dataset import dataset_build

logger = logging.getLogger(__name__)


@dataset_build
def append_current_store(processed, *, store=None):
    """Return False for a rebuild; otherwise append/no-op under one store lock."""
    store = store or FeatureStore()
    with store._access():
        def rebuild(reason):
            store._rebuild_reason = reason
            return False
        meta = store.meta()
        signature = fe_signature()
        if (not store.exists() or meta.get("schema_version") != STORE_SCHEMA_VERSION
                or meta.get("fe_signature") != signature or not Path(store.counts_state_path).exists()):
            return rebuild("feature_code_or_store_schema_changed")
        if not meta.get("dataset_manifest"):
            # Built from collector caches before source publication existed.
            return rebuild("store_predates_source_publication")
        last = store.last_date()
        if last is None or data_signature(processed, cutoff=last) != meta.get("data_signature"):
            return rebuild("processed_history_changed")
        from src.foreign_form import signature as foreign_form_signature
        if meta.get("foreign_form_signature") != foreign_form_signature(last):
            return rebuild("foreign_form_history_changed")
        atr = meta.get("atr_signature")
        if not atr or atr != _atr_signature(last):
            return rebuild("atr_history_changed")
        current_rtv = _rtv_signature(last)
        if (meta.get("rtv_signature") is not None or any(current_rtv["sig"].values())) and meta.get("rtv_signature") != current_rtv:
            return rebuild("rtv_history_changed")
        dates = pd.to_datetime(processed.race_date)
        new = processed.loc[dates.gt(last)].copy()
        if new.empty:
            # A foreign start after the training cutoff can affect a live
            # card even when there is no new domestic row to append.
            from src.foreign_form import version as foreign_form_version
            from src.source_dataset import require_manifest, use_manifest
            current = require_manifest()
            with use_manifest(meta["dataset_manifest"]):
                old_foreign = foreign_form_version()
            if old_foreign != foreign_form_version():
                _stamp(store, signature, meta["data_signature"])
            return True
        logger.info("Engineering %d new rows with native carried state", len(new))
        featured, advanced, counts = store._fast_featurize(new)
        # Retain native processed-history columns created by the initial build.
        # This frame is small compared with the full historical feature matrix.
        processed_all = pd.concat([store.load_processed(), new], ignore_index=True, sort=False)
        with store._staged_update() as pending:
            part = Path(pending.root) / "new_rows.parquet"
            _stringify_objects(featured).to_parquet(part, index=False)
            logger.info("Streaming historical features and %d new rows", len(featured))
            rows = combine_parquet_parts([store.featured_path, part], pending.featured_path)
            if rows != len(processed_all) or rows != len(processed):
                raise ValueError("Processed and featured history differ; rebuild the store")
            _stringify_objects(processed_all).to_parquet(pending.processed_path, index=False)
            pending._save_state(advanced)
            pending._save_counts_state(counts)
            # Native metadata only needs the featured row count, not values.
            pending._write_meta(processed_all, pd.DataFrame(index=pd.RangeIndex(rows)),
                                op="appended", previous=meta)
            _stamp(pending, signature, data_signature(processed, cutoff=dates.max()))
            if fe_signature() != signature:
                raise RuntimeError("Feature code changed during the store refresh")
        logger.info("Committed %d historical/new feature rows as one generation", rows)
        return True


@dataset_build
def get_training_dataset(*, processed=None, rebuild=False):
    processed = process_data() if processed is None else processed
    reason = "explicit"
    if not rebuild:
        store = FeatureStore()
        with store._access():
            if append_current_store(processed, store=store):
                return store.load_featured(columns=_model_load_columns(store))
            reason = store._rebuild_reason
    # All training uses the same corrected builder. A new definition
    # must rebuild every carried history, without materialising 1,000+ full
    # feature columns at once on this machine.
    from src.store_build import build_store
    from src.ingestion.storage import atomic_json, read_json
    from src.source_dataset import registry_root, current_manifest
    path = registry_root() / "builds.json"
    history = read_json(path).get("builds", [])
    meta = FeatureStore().meta()
    history.append({"started_at": datetime.now(timezone.utc).isoformat(),
                    "reason": reason,
                    "previous_last_date": meta.get("last_date"),
                    "previous_manifest": (meta.get("dataset_manifest") or {}).get("id"),
                    "dataset_manifest": current_manifest().get("id")})
    atomic_json(path, {"builds": history[-1000:]})
    store = FeatureStore()
    with store._staged_update() as pending:
        build_store(pending, 60_000, processed=processed)
    return store.load_featured(columns=_model_load_columns(store))
