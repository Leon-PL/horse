"""Learned profiles for trainers, jockeys and sires (experiment 060).

``build`` fits the profiles (``src/connection_profiles.py``) once per quarter
on races before it, minus the purge gap, over the store's development rows
(the golden hold-out is dropped before anything is fitted), and keeps the 13
profile inputs for the cached rows. ``confirm`` adds them to the production
inputs and fits the production recipe at native capacity on 2025Q3-2026Q2,
against the production arm on the same races. The profile settings (rank 4,
ridge, two-year half-life, context columns) are fixed in the module, not
tuned here.

    python -m scripts.connection_profiles_060 build
    python -m scripts.connection_profiles_060 confirm
    python -m scripts.connection_profiles_060 report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.model_ideas import (OUT, Cache, baseline, compare, predict_production, print_summary,  # noqa: E402
                                 production_inputs, run_arm, save_json, setup_logging)
from scripts.select_features import OUT as CACHE_DIR  # noqa: E402
from src.connection_profiles import feature_names, new_yard, quarterly_features  # noqa: E402
from src.feature_store import FeatureStore  # noqa: E402
from src.wf_eval import development_mask  # noqa: E402

FEATURES = OUT / "profiles_060.parquet"
COLUMNS = ["race_id", "race_date", "horse_id", "trainer", "jockey", "sire", "race_type", "surface", "race_name",
           "going", "race_class", "handicap", "age", "distance_furlongs", "num_runners", "horse_prev_races",
           "days_since_last_race", "odds", "won"]

logger = logging.getLogger("connection_profiles_060")


def market_residual(frame: pd.DataFrame) -> np.ndarray:
    """won minus the SP-implied chance, normalised per race; NaN for races without a full SP field."""
    inverse = 1 / pd.to_numeric(frame.odds, errors="coerce").where(lambda o: o > 1)
    complete = inverse.notna().groupby(frame.race_id).transform("all")
    total = inverse.groupby(frame.race_id).transform("sum")
    winners = frame.won.groupby(frame.race_id).transform("sum")
    p = (inverse / total).where(complete & winners.eq(1))
    return (frame.won - p).to_numpy(np.float64)


def build(_args) -> None:
    store = pq.read_table(FeatureStore().featured_path, columns=COLUMNS).to_pandas()
    store["store_row"] = np.arange(len(store))
    store = store.loc[development_mask(store.race_date)].copy()
    store["race_date"] = pd.to_datetime(store.race_date)
    store = store.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)
    store["new_yard"] = new_yard(store)
    residual = market_residual(store)
    logger.info("%d development runners, %.1f%% with a market residual", len(store),
                100 * np.isfinite(residual).mean())
    features = quarterly_features(store, residual, log=logger.info)
    features["store_row"] = store.store_row.to_numpy()
    rows = pd.read_parquet(CACHE_DIR / "targets.parquet", columns=["store_row"])
    cached = rows.merge(features, on="store_row", how="left", validate="one_to_one")
    OUT.mkdir(parents=True, exist_ok=True)
    cached.drop(columns="store_row").to_parquet(FEATURES, index=False)
    names = feature_names()
    summary = {"rows": int(len(cached)), "missing_share": float(cached[names[0]].isna().mean()),
               "describe": cached[names].describe().round(4).to_dict()}
    save_json(OUT / "profiles_build_060.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "describe"}, indent=2))
    print(cached[names].describe().T.to_string())


def confirm(args) -> None:
    cache = Cache()
    names = production_inputs(cache)
    extra = pd.read_parquet(FEATURES)[feature_names()].to_numpy(np.float32)
    if len(extra) != len(cache.targets.won):
        raise SystemExit("Profile inputs do not line up with the cache; rerun build.")
    baseline(cache, "confirm")
    run_arm(cache, "confirm", "profiles",
            lambda train, test, capacity: predict_production(cache, names, train, test, capacity=capacity,
                                                             extra=extra))
    report(args)


def report(_args) -> dict:
    summary = compare("confirm", ["production", "profiles"], [("profiles", "production")])
    save_json(OUT / "summary_060.json", summary)
    print_summary(summary, "060 connection profiles")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["build", "confirm", "report"])
    args = parser.parse_args()
    setup_logging()
    {"build": build, "confirm": confirm, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
