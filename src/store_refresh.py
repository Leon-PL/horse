"""Training-store refresh without loading two copies of every historical feature.

Feature values and carried state come from the existing native implementation.
Only parquet consolidation is streamed; all five files still commit together.
"""
from pathlib import Path
import logging

import pandas as pd

from src.data_processor import process_data
from src.feature_store import FeatureStore, STORE_SCHEMA_VERSION, _stringify_objects
from src.featured_dataset import (
    _atr_signature, _rtv_signature, _model_load_columns, _stamp, data_signature, fe_signature,
)
from src.parquet_io import combine_parquet_parts

logger = logging.getLogger(__name__)


def append_current_store(processed, *, store=None):
    """Return False for a rebuild; otherwise append/no-op under one store lock."""
    store = store or FeatureStore()
    with store._access():
        meta = store.meta()
        signature = fe_signature()
        if (not store.exists() or meta.get("schema_version") != STORE_SCHEMA_VERSION
                or meta.get("fe_signature") != signature or not Path(store.counts_state_path).exists()):
            return False
        last = store.last_date()
        if last is None or data_signature(processed, cutoff=last) != meta.get("data_signature"):
            return False
        atr = meta.get("atr_signature")
        if not atr or atr != _atr_signature(last):
            return False
        current_rtv = _rtv_signature(last)
        if (meta.get("rtv_signature") is not None or any(current_rtv["sig"].values())) and meta.get("rtv_signature") != current_rtv:
            return False
        dates = pd.to_datetime(processed.race_date)
        new = processed.loc[dates.gt(last)].copy()
        if new.empty:
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


def get_training_dataset(*, processed=None, rebuild=False):
    processed = process_data() if processed is None else processed
    if not rebuild:
        store = FeatureStore()
        with store._access():
            if append_current_store(processed, store=store):
                return store.load_featured(columns=_model_load_columns(store))
    # All CLI/UI training uses the same corrected builder. A new definition
    # must rebuild every carried history, without materialising 1,000+ full
    # feature columns at once on this machine.
    from scripts.build_feature_store_batched import _build
    store = FeatureStore()
    with store._staged_update() as pending:
        _build(pending, 60_000, processed=processed)
    return store.load_featured(columns=_model_load_columns(store))
