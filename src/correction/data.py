"""Assemble the correction's history frame from the feature store, the forecast history and prices."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.correction import history as fh
from src.correction import inputs as ci

KEYS = ci.KEYS
TRACKING_FILE = Path(__file__).resolve().parents[2] / "reference" / "modelling" / "correction_tracking.txt"
CARD_COLUMNS = ("form", "hidden_form_runs")


def tracking_columns(exclude=()):
    """Racing TV and At The Races tracking columns passed straight to the correction.

    A frozen list (the store's ``rtv_``/``atr_`` columns the win model did not use when the
    correction was measured), less anything the current win model already has.
    """
    names = [line.strip() for line in TRACKING_FILE.read_text(encoding="utf-8").splitlines()]
    blocked = set(exclude)
    return [n for n in names if n and not n.startswith("#") and n not in blocked]


def _with_forecasts(frame, forecasts):
    held = pd.DataFrame({"race_id": forecasts.race_id.astype(str).to_numpy(),
                         "horse_id": pd.to_numeric(forecasts.horse_id).astype("int64").to_numpy(),
                         "p_base": forecasts.probability.to_numpy(float)})
    keys = pd.DataFrame({"race_id": frame.race_id.astype(str).to_numpy(),
                         "horse_id": pd.to_numeric(frame.horse_id).astype("int64").to_numpy()})
    frame["p_base"] = keys.merge(held, on=KEYS, how="left", validate="one_to_one").p_base.to_numpy()
    # A race counts only with a forecast for its whole field.
    whole = frame.p_base.notna().groupby(frame.race_id).transform("all").to_numpy()
    frame.loc[~whole, "p_base"] = np.nan
    frame["p_market"] = ci.closing_probability(frame).to_numpy()
    return frame


def assemble(store_root=None, *, forecasts=None, development=True, start=None, end=None, exclude=()):
    """Every settled runner in the store with the forecast (NaN where none), market and snapshots.

    Returns the frame (race order, all runs kept as history), row-aligned exchange snapshot prices,
    the tracking column names and the store receipt. ``forecasts`` defaults to the stored history.
    """
    import pyarrow.parquet as pq
    from src.feature_store import FeatureStore
    from src.modelling.data import load_dataset
    from src.source_dataset import load_manifest, read_table, use_manifest
    from src.source_prices import attach_bsp

    schema = set(pq.read_schema(FeatureStore(str(store_root) if store_root else None).featured_path).names)
    tracking = [c for c in tracking_columns(exclude) if c in schema]
    if not tracking:
        raise ValueError("None of the correction's tracking columns is in the feature store")
    frame, receipt = load_dataset(store_root, features=tracking, development=development, start=start, end=end,
                                  label_columns=[*ci.ENTITY_COLUMNS, *CARD_COLUMNS])
    forecasts = fh.read() if forecasts is None else forecasts
    if forecasts.empty:
        raise ValueError("No forecast history; seed it from a benchmark first")
    frame = _with_forecasts(frame, forecasts)
    manifest = receipt["dataset_manifest"]
    with use_manifest(load_manifest(manifest["id"], root=manifest.get("root"))):
        table = read_table("betfair", "timing", linked=True).rename(columns={"bsp": "bf_bsp"})
        prices = attach_bsp(frame[KEYS].copy(), table, cols=("bf_bsp", *ci.SNAPSHOTS))
    return frame, prices.reset_index(drop=True), tracking, receipt


SETTLED_COLUMNS = ("race_id", "horse_id", "horse_name", "race_date", "num_runners", "finish_position", "won", "odds",
                   *ci.ENTITY_COLUMNS, *CARD_COLUMNS)


def settled_frame(store_root=None, *, forecasts=None, cache=True):
    """The settled history for scoring a card: the same rows as ``assemble``, read the cheap way.

    ``assemble`` re-verifies the whole store against its sources and loads every tracking column
    for every past runner (minutes, and gigabytes). A card needs neither: its own features carry
    the tracking columns, and the live path has already checked the store's feature signature.
    This reads only identity, result, price and entity columns, and keeps the result in a file
    beside the forecast history until the store or the history changes.
    """
    from src.feature_store import FeatureStore
    from src.featured_dataset import fe_signature
    from src.forecast_evaluation import valid_race_mask
    from src.ingestion.storage import atomic_json, atomic_parquet, read_json
    from src.modelling.data import ordered_frame
    from src.source_dataset import load_manifest, read_table, use_manifest
    from src.source_prices import attach_bsp

    store = FeatureStore(str(store_root) if store_root else None)
    with store._access():
        meta = store.meta()
        if meta.get("fe_signature") != fe_signature() or not meta.get("dataset_manifest"):
            raise ValueError("Feature store is incompatible with the feature code; rebuild it first")
        forecasts = fh.read() if forecasts is None else forecasts
        if forecasts.empty:
            raise ValueError("No forecast history; seed it from a benchmark first")
        identity = {"fe_signature": meta["fe_signature"], "data_signature": meta.get("data_signature"),
                    "store_last_date": str(meta.get("last_date")), "manifest": meta["dataset_manifest"]["id"],
                    "store": str(store.root), "forecast_rows": int(len(forecasts)),
                    "forecast_last_date": str(forecasts.race_date.max().date()),
                    "columns": list(SETTLED_COLUMNS)}
        folder = fh.default_root() / "settled_cache"
        if cache and read_json(folder / "identity.json") == identity:
            return (pd.read_parquet(folder / "frame.parquet"), pd.read_parquet(folder / "prices.parquet"),
                    identity["store_last_date"])
        frame = pd.read_parquet(store.featured_path, columns=list(SETTLED_COLUMNS))
    manifest = meta["dataset_manifest"]
    with use_manifest(load_manifest(manifest["id"], root=manifest.get("root"))):
        frame = attach_bsp(frame, cols=("bsp",), overwrite=True)
        # The same row rules as src.modelling.data.load_dataset.
        rows = valid_race_mask(frame)
        bad = frame.loc[pd.to_numeric(frame.odds, errors="coerce").lt(1.01), "race_id"]
        rows &= ~frame.race_id.isin(bad)
        frame, _ = ordered_frame(frame.loc[rows])
        frame = _with_forecasts(frame, forecasts)
        table = read_table("betfair", "timing", linked=True).rename(columns={"bsp": "bf_bsp"})
        prices = attach_bsp(frame[KEYS].copy(), table, cols=("bf_bsp", *ci.SNAPSHOTS)).reset_index(drop=True)
    if cache:
        atomic_json(folder / "identity.json", {"building": True})
        atomic_parquet(frame, folder / "frame.parquet")
        atomic_parquet(prices, folder / "prices.parquet")
        atomic_json(folder / "identity.json", identity)
    return frame, prices, identity["store_last_date"]
