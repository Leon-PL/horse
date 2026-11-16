"""Complete fields and explicit projected reads from a compatible feature store."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.feature_schema import require_numeric_features
from src.forecast_evaluation import valid_race_mask
from src.model import EXCLUDE_COLUMNS


CONTEXT = ("race_id", "horse_id", "horse_name", "race_date", "num_runners", "finish_position", "won", "bsp", "odds")


def ordered_frame(frame, *, labelled=True):
    """Validate fields before sorting; return a private frame and original positions."""
    if frame.empty or frame.columns.has_duplicates:
        raise ValueError("A nonempty frame with unique columns is required")
    required = {"race_id", "horse_id", "horse_name", "race_date", "num_runners"}
    if not required.issubset(frame):
        raise ValueError(f"Missing field identities: {sorted(required - set(frame))}")
    dates = pd.to_datetime(frame.race_date, errors="coerce").dt.normalize()
    if dates.isna().any() or frame.race_id.isna().any() or frame.horse_id.isna().any():
        raise ValueError("Invalid field date or runner identity")
    if frame.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate canonical runner identity")
    if labelled:
        if not {"won", "finish_position"}.issubset(frame) or not valid_race_mask(frame).all():
            raise ValueError("Expected complete, single-winner settled fields, including non-finishers")
    else:
        counts = frame.groupby("race_id", sort=False).race_id.transform("size")
        if (counts.lt(2).any() or counts.ne(pd.to_numeric(frame.num_runners, errors="coerce")).any()
                or dates.groupby(frame.race_id).nunique().ne(1).any()):
            raise ValueError("Prediction requires complete fields on one date per race")
        from src.utils import normalise_horse_key
        keys = pd.DataFrame({"race": frame.race_id, "horse": frame.horse_name.map(normalise_horse_key)})
        if keys.horse.eq("").any() or keys.duplicated().any():
            raise ValueError("Prediction has ambiguous runner names")
    order = pd.DataFrame({"date": dates.to_numpy(), "race": frame.race_id.astype(str).to_numpy(),
                          "horse": frame.horse_name.astype(str).to_numpy(), "position": np.arange(len(frame))})
    positions = order.sort_values(["date", "race", "horse"], kind="stable").position.to_numpy()
    result = frame.iloc[positions].copy().reset_index(drop=True)
    result["race_date"] = dates.iloc[positions].to_numpy()
    return result, positions


def numeric_matrix(frame, features):
    require_numeric_features(frame, list(features), context="ModelSpec")
    x = frame.loc[:, list(features)].to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(x).any():
        raise ValueError("Model inputs may be missing, but cannot contain infinities")
    return x


def load_dataset(store_root=None, *, features=None, development=True, start=None, end=None, label_columns=()):
    """Read only requested model inputs plus context. Never rebuild implicitly.

    ``features=None`` deliberately discovers the eligible numeric universe for
    selection. Development dates are filtered by Parquet before loading rows.
    """
    from src.feature_store import FeatureStore
    from src.featured_dataset import fe_signature
    from src.source_dataset import current_manifest, load_manifest, use_manifest
    from src.data_processor import process_data
    from src.wf_eval import golden_holdout_start, purge_days

    store = FeatureStore(str(store_root) if store_root else None)
    with store._access():
        meta = store.meta()
        if meta.get("fe_signature") != fe_signature() or not meta.get("dataset_manifest"):
            raise ValueError("Feature store is incompatible; rebuild through build_feature_store_batched first")
        recorded = meta["dataset_manifest"]
        manifest = load_manifest(recorded["id"], root=recorded.get("root"))
        active = current_manifest()
        if active and active.get("id") != manifest["id"]:
            raise ValueError("Selected dataset differs from the store dataset")
        # Verify historical values against the exact accepted inputs, rather
        # than today's moving collector pointers. Never append in a read.
        with use_manifest(manifest):
            processed = process_data()
            from src.featured_dataset import data_signature, _atr_signature, _rtv_signature
            last = meta["last_date"]
            checks = {"data_signature": data_signature(processed, cutoff=last),
                      "atr_signature": _atr_signature(last), "rtv_signature": _rtv_signature(last)}
            if any(meta.get(k) != v for k, v in checks.items()):
                raise ValueError("Feature history does not match its accepted inputs; rebuild the store")
        schema = pq.read_schema(store.featured_path)
        eligible = [f.name for f in schema if f.name not in EXCLUDE_COLUMNS and
                    (pa.types.is_integer(f.type) or pa.types.is_floating(f.type) or pa.types.is_boolean(f.type))]
        requested = eligible if features is None else list(features)
        if not requested or set(requested) - set(eligible):
            raise ValueError(f"Unavailable or ineligible model features: {sorted(set(requested) - set(eligible))}")
        missing = set(label_columns) - set(schema.names)
        if missing:
            raise ValueError(f"Unavailable training context: {sorted(missing)}")
        columns = list(dict.fromkeys([c for c in CONTEXT if c in schema.names] + requested + list(label_columns)))
        filters = []
        field = schema.field("race_date")
        def bound(value):
            return str(pd.Timestamp(value).date()) if pa.types.is_string(field.type) or pa.types.is_large_string(field.type) else pd.Timestamp(value)
        if development:
            filters.append(("race_date", "<", bound(golden_holdout_start() - pd.Timedelta(days=purge_days()))))
        if start:
            filters.append(("race_date", ">=", bound(start)))
        if end:
            filters.append(("race_date", "<=", bound(end)))
        frame = pd.read_parquet(store.featured_path, columns=columns, filters=filters or None)
        if fe_signature() != meta["fe_signature"]:
            raise RuntimeError("Feature code changed during the projected read")
    # Current prices are targets/evaluation context, never inference inputs.
    # Read them from the same immutable dataset rather than a collector cache.
    from src.source_prices import attach_bsp
    with use_manifest(manifest):
        frame = attach_bsp(frame, cols=("bsp",), overwrite=True)
    eligible_rows = valid_race_mask(frame)
    # Match the established preparation exclusion for invalid recorded prices.
    if "odds" in frame:
        bad = frame.loc[pd.to_numeric(frame.odds, errors="coerce").lt(1.01), "race_id"]
        eligible_rows &= ~frame.race_id.isin(bad)
    excluded = int(frame.loc[~eligible_rows, "race_id"].nunique())
    frame = frame.loc[eligible_rows]
    # An empty window (e.g. nothing settled after a run's cutoff) is a valid answer;
    # callers such as `src.correction extend` check `frame.empty` themselves.
    frame = ordered_frame(frame)[0] if not frame.empty else frame.reset_index(drop=True)
    receipt = {"dataset_manifest": manifest, "fe_signature": meta["fe_signature"],
               "data_signature": meta["data_signature"], "store_last_date": meta["last_date"],
               "dataset_start": str(pd.to_datetime(processed.race_date).min().date()),
               "excluded_races": excluded, "development_only": development,
               "evaluation_status": "retrospective; existing periods have informed earlier research",
               "features": requested, "store_root": str(Path(store.root).resolve()), "label_columns": list(label_columns)}
    frame.attrs["modelling_receipt"] = receipt
    return frame, receipt
