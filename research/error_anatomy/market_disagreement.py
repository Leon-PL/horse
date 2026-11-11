"""What predicts where the market will disagree with the model?

    python -m research.error_anatomy.market_disagreement diagnose [--threads 6]

The gap between the closing market and the model, ``log(market p) - log(model p)``
centred within the race, is known for every runner and has no outcome noise. If
a set of pre-race inputs predicts that gap on later races, those inputs hold
public information the generalist is not using.

``diagnose`` fits a plain LightGBM regression of the gap on each input family
using 2022Q3-2025Q2 and reports out-of-sample R-squared on 2025Q3-2026Q2, with
the strongest inputs. The families:

- ``own``: the generalist's own 193 inputs (anything found here is under-fitting);
- ``store``: every other permitted numeric column in the feature store;
- ``errors``: the entity-residual inputs from ``entity_residuals.py``;
- ``all``: everything together.

No forecast is changed here; it is a map of where to look.
"""
import argparse

import numpy as np
import pandas as pd

from research.error_anatomy import entity_residuals as er
from research.error_anatomy.frame import OUT
from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s
from src.ingestion.storage import atomic_json

KEYS = s.KEYS
TRAIN_END = "2025Q2"
PARAMS = {"objective": "regression", "learning_rate": .05, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": .5, "bagging_fraction": .8, "bagging_freq": 1, "lambda_l2": 10., "max_bin": 63,
          "verbose": -1, "seed": 7, "force_row_wise": True}
ROUNDS = 300


def centred_gap(frame, market="p_market", model="p_single"):
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.log(frame[market].to_numpy(float)) - np.log(frame[model].to_numpy(float))
    gap = pd.Series(np.where(np.isfinite(gap), gap, np.nan), index=frame.index)
    return (gap - gap.groupby(frame.race_id).transform("mean")).clip(-3, 3)


def store_numeric_columns():
    import pyarrow.parquet as pq
    import pyarrow.types as pt
    schema = pq.read_schema(sc.store_file())
    numeric = [f.name for f in schema if pt.is_integer(f.type) or pt.is_floating(f.type) or pt.is_boolean(f.type)]
    return [c for c in sc.allowed(numeric) if c not in (*KEYS, "num_runners")]


def read_store_float32(columns, start, chunk=150):
    """Store columns as float32, read a block at a time to keep memory down."""
    parts = []
    for i in range(0, len(columns), chunk):
        block = sc.read_store(columns[i:i + chunk], start=start).drop(columns="race_date")
        block[columns[i:i + chunk]] = block[columns[i:i + chunk]].astype("float32")
        parts.append(block if not parts else block.drop(columns=KEYS))
        if len(parts) > 1 and not parts[0][KEYS[1]].equals(block[KEYS[1]]):
            raise ValueError("Store blocks are not aligned")
    return pd.concat(parts, axis=1)


def diagnose(threads):
    import lightgbm as lgb
    frame = er.long_frame()
    frame["quarter"] = frame.race_date.dt.to_period("Q").astype(str)
    frame["gap"] = centred_gap(frame)
    own = list(sc.spec_features("full-063"))
    own_set = set(own)
    store = [c for c in store_numeric_columns() if c not in own_set]
    start = frame.race_date.min()
    own_values = pd.read_parquet(s.STUDY / "input.parquet", columns=[*KEYS, *own], filters=[("race_date", ">=", start)])
    own_values.attrs = {}
    own_values[own] = own_values[own].astype("float32")
    errors = pd.read_parquet(OUT / "long_inputs.parquet")
    error_columns = er.columns_of(errors)
    for values in (own_values, read_store_float32(store, start), errors):
        # A few inputs (field size, days since last run) are already in the frame as context.
        frame = s.attach(frame, values.drop(columns=[c for c in values if c in frame and c not in KEYS]))
    known = frame.gap.notna().to_numpy()
    train = known & (frame.quarter <= TRAIN_END).to_numpy()
    test = known & frame.quarter.isin(s.SCORED).to_numpy()
    y = frame.gap.to_numpy(float)
    families = {"own": own, "store": store, "errors": error_columns, "all": [*own, *store, *error_columns]}
    report = {"train_runners": int(train.sum()), "test_runners": int(test.sum()),
              "gap_sd_test": float(y[test].std()), "inputs": {k: len(v) for k, v in families.items()}}
    for name, columns in families.items():
        x = frame[columns].to_numpy(dtype=np.float32, na_value=np.nan)
        booster = lgb.train({**PARAMS, "num_threads": threads}, lgb.Dataset(x[train], label=y[train]),
                            num_boost_round=ROUNDS)
        predicted = booster.predict(x[test])
        r2 = 1 - np.mean((y[test] - predicted) ** 2) / np.var(y[test])
        gain = booster.feature_importance("gain")
        top = [(columns[i], round(float(gain[i] / gain.sum()), 4)) for i in np.argsort(-gain)[:25]]
        by_quarter = {q: float(1 - np.mean((y[test & (frame.quarter == q).to_numpy()]
                                            - booster.predict(x[test & (frame.quarter == q).to_numpy()])) ** 2)
                               / np.var(y[test & (frame.quarter == q).to_numpy()])) for q in s.SCORED}
        report[name] = {"r2": float(r2), "r2_by_quarter": by_quarter, "top_inputs": top}
        print(f"{name:<7} {len(columns):>5} inputs  out-of-sample R2 {r2:+.4f}  "
              f"{ {q: round(v, 4) for q, v in by_quarter.items()} }", flush=True)
        print("   ", top[:12], flush=True)
        atomic_json(OUT / "market_disagreement.json", report)
        del x


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["diagnose"])
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    diagnose(args.threads)


if __name__ == "__main__":
    main()
