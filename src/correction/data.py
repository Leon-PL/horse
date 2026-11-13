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
    manifest = receipt["dataset_manifest"]
    with use_manifest(load_manifest(manifest["id"], root=manifest.get("root"))):
        table = read_table("betfair", "timing", linked=True).rename(columns={"bsp": "bf_bsp"})
        prices = attach_bsp(frame[KEYS].copy(), table, cols=("bf_bsp", *ci.SNAPSHOTS))
    return frame, prices.reset_index(drop=True), tracking, receipt
