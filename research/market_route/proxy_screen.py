"""Do public proxies for private information predict the 08:00 to 11:00 price move?

    python -m research.market_route.proxy_screen [--threads 6]

Every family here was judged priced against BSP. The 08:00 exchange price is a
weaker opponent, so the question is repeated against it: does the family
predict the part of a runner's 08:00 to 11:00 move (and 08:00 to BSP move) that
the fundamentals forecast does not explain?

Frame: races fully traded at 08:00 and 11:00 with no withdrawal after 08:00
(``moves_0800_1100.parquet``, written by the exploratory decomposition).
Target: the race-centred log move minus its least-squares fit on the forecast's
disagreement with the 08:00 price. Each family is given to a small LightGBM
with the 08:00 price as a base input, on purged expanding quarters, and judged
by its out-of-fold gain over the base alone, against a twin whose family
columns are permuted within the quarter. Exploratory, development data only.
"""
import argparse
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from research.error_anatomy import sale_inputs
from research.meta_learner import screen_candidates as sc
from src import wf_eval
from src.ingestion.storage import atomic_json
from src.model import EXCLUDE_COLUMNS

OUT = Path("data/research/market_route_20261111")
KEYS = ["race_id", "horse_id"]
BASE = ["log_p08", "log_p08_centred"]
PARAMS = {"objective": "regression", "learning_rate": .05, "num_leaves": 15, "min_data_in_leaf": 300,
          "feature_fraction": .8, "bagging_fraction": .8, "bagging_freq": 1, "lambda_l2": 20., "verbose": -1}
ROUNDS = 200
PATTERNS = {
    "pedigree": r"^sire_|^damsire_|^dam_",
    "first-string": r"^trainer_first_string",
    "first-time equipment": r"^first_time_|^has_headgear$|^headgear_off$|^any_first_time_equip$",
    "wind surgery": r"^wind_surgery|^wind_ops_prior$|^first_wind_surgery$|^prev_run_wind_surgery$",
    "trainer recent form": r"^trainer_(wins|runs|places|wr|pr|hot)_?.*(14d|30d)",
    "absence and health reports": r"^days_since_last_race$|^days_bucket_fresh$|^freshness_relative_score$|^bha_days_since",
}
RIDERS = {"conditional": "conditional jockeys", "apprentice": "apprentice", "amateur": "amateur"}


def families():
    names = pq.ParquetFile(sc.store_file()).schema_arrow.names
    return {family: [n for n in names if re.search(pattern, n) and n not in EXCLUDE_COLUMNS]
            for family, pattern in PATTERNS.items()}


def load():
    frame = pd.read_parquet(OUT / "moves_0800_1100.parquet")
    groups = families()
    wanted = sorted({c for columns in groups.values() for c in columns})
    store = sc.read_store([*wanted, "prior_form_runs", "days_since_last_run", "race_name"],
                          start=frame.race_date.min()).drop(columns="race_date")
    frame = frame.merge(store, on=KEYS, how="left", validate="one_to_one")
    sales, sale_columns = sale_inputs.load(frame)
    frame = frame.merge(sales[[*KEYS, *sale_columns]], on=KEYS, how="left", validate="one_to_one")
    groups["sale price"] = sale_columns
    race = frame.race_id
    frame["log_p08"] = np.log(frame.p08)
    frame["log_p08_centred"] = frame.log_p08 - frame.log_p08.groupby(race).transform("mean")
    total = np.log(frame.pb / frame.p08)
    total = total - total.groupby(race).transform("mean")
    slope = (total * frame.gap).sum() / (frame.gap ** 2).sum()
    frame["res_close"] = total - slope * frame.gap
    frame["quarter"] = frame.race_date.dt.to_period("Q").astype(str)
    name = frame.race_name.astype(str).str.lower()
    for label, text in RIDERS.items():
        frame[f"riders_{label}"] = name.str.contains(text)
    return frame.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True), groups


def oof(frame, columns, target, threads):
    def run(train, test):
        model = lgb.train({**PARAMS, "num_threads": threads, "seed": 7}, lgb.Dataset(train[columns], train[target]),
                          ROUNDS)
        return model.predict(test[columns])
    folds = wf_eval.expanding_period_folds(frame.race_date, freq="Q", min_train_periods=4)
    return wf_eval.oof_predict_expanding(frame, fit_predict=run, folds=folds)


def permuted(frame, columns, seed=11):
    """The family's rows moved to random runners within the same quarter; base inputs stay put."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for _, index in frame.groupby("quarter").indices.items():
        out.iloc[index, [out.columns.get_loc(c) for c in columns]] = frame.iloc[rng.permutation(index)][columns].to_numpy()
    return out


def gain(frame, prediction, base, target, mask):
    """Share of the target's variance explained beyond the base-only model, on *mask*."""
    rows = mask & np.isfinite(prediction) & np.isfinite(base)
    y = frame.loc[rows, target]
    return float((((y - base[rows]) ** 2).sum() - ((y - prediction[rows]) ** 2).sum()) / (y ** 2).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    frame, groups = load()
    runs = frame.prior_form_runs
    segments = {"all": pd.Series(True, index=frame.index), "unraced": runs.eq(0), "one run": runs.eq(1),
                "2-4 runs": runs.between(2, 4), "5+ runs": runs.ge(5), "off 120+ days": frame.days_since_last_run.gt(120)}
    print(f"{frame.race_id.nunique()} races, {len(frame)} runners; segments "
          + ", ".join(f"{k} {int(v.sum())}" for k, v in segments.items()), flush=True)
    report = {}
    for target, label in (("res", "08:00 to 11:00"), ("res_close", "08:00 to BSP")):
        base = oof(frame, BASE, target, args.threads)
        print(f"\n== move {label} not explained by the forecast; variance explained beyond the 08:00 price, "
              f"real / permuted twin (percent)", flush=True)
        print(f"{'family':28s} {'inputs':>6s}  " + "  ".join(f"{k:>15s}" for k in segments), flush=True)
        for family, columns in groups.items():
            real = oof(frame, [*BASE, *columns], target, args.threads)
            twin = oof(permuted(frame, columns), [*BASE, *columns], target, args.threads)
            cells = {k: [100 * gain(frame, real, base, target, m), 100 * gain(frame, twin, base, target, m)]
                     for k, m in segments.items()}
            report[f"{label}|{family}"] = {"inputs": len(columns), **cells}
            print(f"{family:28s} {len(columns):6d}  " + "  ".join(f"{a:+7.2f}/{b:+6.2f}" for a, b in cells.values()),
                  flush=True)
    # Races restricted to inexperienced riders: how the forecast and the early prices fare there.
    print("\n== races restricted by rider: distance to BSP per 1,000 runners", flush=True)
    kinds = {"open": ~frame[[f"riders_{k}" for k in RIDERS]].any(axis=1), **{k: frame[f"riders_{k}"] for k in RIDERS}}
    won = frame.merge(pd.read_parquet(OUT / "predictions_no_withdrawal_0800.parquet", columns=[*KEYS, "won"]),
                      on=KEYS, how="left").won
    for kind, mask in kinds.items():
        row = {"races": int(frame.loc[mask, "race_id"].nunique())}
        for column in ("pm", "p08", "p11"):
            row[f"distance_{column}"] = float(1000 * (frame.pb * np.log(frame.pb / frame[column]))[mask].mean())
        winners = mask & won.eq(1)
        for column in ("pm", "p08", "p11", "pb"):
            row[f"loss_{column}"] = float(-np.log(frame.loc[winners, column]).mean())
        report[f"riders|{kind}"] = row
        print(f"{kind:12s} races {row['races']:6d}  forecast {row['distance_pm']:6.2f}  08:00 {row['distance_p08']:6.2f}  "
              f"11:00 {row['distance_p11']:6.2f}  | winner log loss: forecast {row['loss_pm']:.3f}  08:00 "
              f"{row['loss_p08']:.3f}  11:00 {row['loss_p11']:.3f}  BSP {row['loss_pb']:.3f}", flush=True)
    atomic_json(OUT / "proxy_screen.json", report)


if __name__ == "__main__":
    main()
