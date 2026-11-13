"""The forecast history: the win model's out-of-sample forecast for every past runner.

The correction's inputs measure trainers, sires and horses against what the
model said about them before each race. That only means something if the
forecast was made without having seen the race, so this table accepts two
kinds of row and nothing else:

- walk-forward forecasts from a ``src.modelling benchmark`` (each quarter
  forecast by a model fitted on earlier quarters);
- forecasts from a fitted model for races dated after its training cutoff.

A forecast from a model that was trained on the race would look far too
accurate, and every record built on it would shrink towards nothing without
any error being raised. ``append`` therefore refuses races on or before the
stated training cutoff.

One history describes one recipe. When the win model's recipe or inputs
change, start a new history from a new benchmark (``seed(..., replace=True)``).
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.ingestion.storage import atomic_json, atomic_parquet, read_json, writer_lock

KEYS = ["race_id", "horse_id"]
COLUMNS = [*KEYS, "race_date", "probability", "source", "recipe_signature", "training_last_date"]


def default_root():
    return Path(config.DATA_DIR) / "processed" / "forecast_history"


def _root(root):
    return Path(root) if root is not None else default_root()


def _normalised(records):
    out = pd.DataFrame({
        "race_id": records.race_id.astype(str).to_numpy(),
        "horse_id": pd.to_numeric(records.horse_id, errors="raise").astype("int64").to_numpy(),
        "race_date": pd.to_datetime(records.race_date).dt.normalize().to_numpy(),
        "probability": records.probability.to_numpy(float),
        "source": records.source.astype(str).to_numpy(),
        "recipe_signature": records.recipe_signature.astype(str).to_numpy(),
        "training_last_date": pd.to_datetime(records.training_last_date).dt.strftime("%Y-%m-%d").to_numpy()})
    if out.duplicated(KEYS).any():
        raise ValueError("Forecast history rows must be unique per runner")
    p = out.probability.to_numpy()
    if not (np.isfinite(p) & (p > 0) & (p <= 1)).all():
        raise ValueError("Forecasts must be probabilities in (0, 1]")
    total = out.groupby("race_id").probability.sum()
    if (total - 1).abs().gt(1e-6).any():
        raise ValueError("Each race's forecasts must cover its whole field and sum to one")
    if (out.race_date <= pd.to_datetime(out.training_last_date)).any():
        raise ValueError("A forecast is dated on or before its model's training cutoff: not out of sample")
    return out.sort_values(["race_date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)


def read(root=None):
    path = _root(root) / "history.parquet"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_parquet(path)


def _write(frame, root, event):
    root = _root(root)
    atomic_parquet(frame[COLUMNS], root / "history.parquet")
    meta = read_json(root / "meta.json")
    log = [*meta.get("log", []), {**event, "at": datetime.now().isoformat(timespec="seconds")}][-200:]
    atomic_json(root / "meta.json", {
        "rows": int(len(frame)), "races": int(frame.race_id.nunique()),
        "first_date": str(frame.race_date.min().date()), "last_date": str(frame.race_date.max().date()),
        "recipe_signatures": sorted(frame.recipe_signature.unique()), "log": log})


def seed(benchmark, *, component=None, root=None, replace=False):
    """Start the history from a benchmark's walk-forward forecasts."""
    folder = Path(benchmark)
    records = pd.read_parquet(folder / "oof.parquet")
    components = sorted(records.component.unique())
    if component is None:
        if len(components) != 1:
            raise ValueError(f"The benchmark holds several components; name one of {components}")
        component = components[0]
    records = records.loc[records.component.eq(component)]
    if records.empty:
        raise ValueError(f"No forecasts for component {component}")
    records = records.assign(source="benchmark:" + records.fold.astype(str))
    frame = _normalised(records)
    root = _root(root)
    with writer_lock(root / ".lock") as acquired:
        if not acquired:
            raise RuntimeError("Another process is writing the forecast history")
        if (root / "history.parquet").exists() and not replace:
            raise ValueError("A forecast history already exists; pass replace=True to start a new one")
        _write(frame, root, {"event": "seed", "benchmark": str(folder.resolve()), "component": component,
                             "rows": int(len(frame))})
    return frame


def append(records, *, source, recipe_signature, training_last_date, root=None):
    """Add forecasts for races after ``training_last_date``; races already held are left as they are."""
    records = records.assign(source=source, recipe_signature=recipe_signature, training_last_date=training_last_date)
    fresh = _normalised(records)
    root = _root(root)
    with writer_lock(root / ".lock") as acquired:
        if not acquired:
            raise RuntimeError("Another process is writing the forecast history")
        held = read(root)
        if held.empty:
            raise ValueError("No forecast history to append to; seed it from a benchmark first")
        if set(held.recipe_signature) != {recipe_signature}:
            raise ValueError("These forecasts come from a different recipe than the history; seed a new history")
        fresh = fresh.loc[~fresh.race_id.isin(set(held.race_id))]
        if fresh.empty:
            return 0
        combined = pd.concat([held, fresh], ignore_index=True)
        combined = combined.sort_values(["race_date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)
        _write(combined, root, {"event": "append", "source": source, "rows": int(len(fresh)),
                                "first_date": str(fresh.race_date.min().date()),
                                "last_date": str(fresh.race_date.max().date())})
    return int(len(fresh))


def status(root=None):
    meta = read_json(_root(root) / "meta.json")
    return {k: v for k, v in meta.items() if k != "log"} if meta else {}
