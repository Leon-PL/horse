"""Walk-forward ranking comparison: GBM vs logistic regression vs the market.

ACTION_PLAN 3a and 4b. The plan's diagnosis rests on 646 test races, where
the production 2,000-tree LightGBM ranked *worse* than plain logistic
regression on the same features (top-1 0.268 vs 0.292) - about 1.3 sigma.
This re-measures it on every development quarter, and tests whether the
production configuration is the cause by adding a loosened GBM (plan 4b).
Class weighting is not a variable: the LightGBM win model is already
unweighted (ledger 13).

All models train on the same rows (everything before the fold, minus the
purge, within the development period) with the production recency weights,
and are scored on the same test races: those with a complete Betfair T-5
market, so the market can be compared on identical races. Early stopping
uses the latest 15% of training dates; no model is refit afterwards, so
every variant is treated identically.

Metrics per race, compared pairwise with race-bootstrap CIs:
top-1 (winner ranked first), top-3, and race log loss after per-race
normalisation (calibration cannot change a within-race ranking, so ranking
metrics need no calibration step).

Usage::

    python scripts/wf_ranking_compare.py
    python scripts/wf_ranking_compare.py --max-folds 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from src.betfair import attach_bsp
from src.feature_store import FeatureStore
from src.model import RacePredictor, compute_recency_sample_weights, get_feature_columns
from src.wf_eval import bootstrap_roi_ci, development_mask, expanding_period_folds

DIAG = ["race_id", "race_date", "won", "track", "off_time", "horse_name", "num_runners"]
INNER_VALID_FRAC = 0.15
ES_ROUNDS = int(getattr(config, "EARLY_STOPPING_ROUNDS", 80)) or 80

_cp = config.CLASSIFIER_PARAMS
GBM_VARIANTS = {
    # The production win classifier's hyperparameters (config.CLASSIFIER_PARAMS).
    "gbm_prod": dict(n_estimators=_cp["n_estimators"], learning_rate=_cp["learning_rate"],
                     max_depth=_cp["max_depth"], num_leaves=_cp.get("num_leaves", 31),
                     subsample=_cp["subsample"], colsample_bytree=_cp["colsample_bytree"],
                     min_child_samples=_cp["min_child_samples"], reg_alpha=_cp["reg_alpha"],
                     reg_lambda=_cp["reg_lambda"]),
    # Plan 4b: a much less damped configuration.
    "gbm_loose": dict(n_estimators=3000, learning_rate=0.05, max_depth=-1, num_leaves=63,
                      subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                      reg_alpha=0.0, reg_lambda=1.0),
}


def load_frame() -> pd.DataFrame:
    store = FeatureStore()
    available = set(pq.read_schema(store.featured_path).names)
    allow = [line.strip() for line in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
             if line.strip() and not line.lstrip().startswith("#")]
    cols = [c for c in sorted(set(allow) | set(DIAG)) if c in available]
    df = pd.read_parquet(store.featured_path, columns=cols)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int)
    df = attach_bsp(df, cols=("ltp_300s",))
    entry = pd.to_numeric(df["ltp_300s"], errors="coerce")
    g = df.groupby("race_id")
    df["scorable"] = ((entry > 1).groupby(df["race_id"]).transform("all")
                      & (g["won"].transform("sum") == 1) & (g["won"].transform("size") >= 2))
    implied = (1.0 / entry).where(df["scorable"])
    df["mkt"] = implied / implied.groupby(df["race_id"]).transform("sum")
    df = df[development_mask(df["race_date"])]
    # horse_name, not storage order, inside a race: rows may sit in finish order.
    return df.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)


def _fit_gbm(params: dict, train: pd.DataFrame, features: list[str], weights: np.ndarray):
    dates = train["race_date"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    is_val = dates > np.quantile(dates, 1 - INNER_VALID_FRAC)
    model = lgb.LGBMClassifier(objective="binary", subsample_freq=1, n_jobs=-1, verbose=-1,
                               random_state=config.RANDOM_SEED, **params)
    model.fit(train.loc[~is_val, features], train.loc[~is_val, "won"],
              sample_weight=weights[~is_val],
              eval_set=[(train.loc[is_val, features], train.loc[is_val, "won"])],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    return model


def _per_race(test: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Per-race scores. Ties share credit: breaking them by row order would
    leak the result wherever rows sit in finishing order, and tied top prices
    (co-favourites) are common in the market."""
    frame = pd.DataFrame({"race": test["race_id"].to_numpy(), "won": test["won"].to_numpy(),
                          "p": probs})
    frame["p"] = frame["p"] / frame.groupby("race")["p"].transform("sum")
    grouped = frame.groupby("race")["p"]
    is_top = frame["p"] == grouped.transform("max")
    frame["top1"] = np.where(is_top, 1.0 / is_top.groupby(frame["race"]).transform("sum"), 0.0)
    frame["rank"] = grouped.rank(ascending=False, method="average")
    w = frame[frame["won"] == 1]
    return pd.DataFrame({"race": w["race"].to_numpy(), "top1": w["top1"].to_numpy(),
                         "top3": (w["rank"] <= 3).to_numpy(float),
                         "ll": -np.log(np.clip(w["p"].to_numpy(), 1e-12, None))})


def run(max_folds: int | None = None, n_boot: int = 2000, last_folds: int | None = None) -> dict:
    df = load_frame()
    features = [c for c in get_feature_columns(df) if c not in ("ltp_300s", "mkt", "scorable")]
    folds = expanding_period_folds(df["race_date"], freq="Q", min_train_periods=8)[: max_folds or None]
    if last_folds:
        folds = folds[-last_folds:]
    burn_in = df["race_date"].min() + pd.DateOffset(months=int(getattr(config, "BURN_IN_MONTHS", 8)))
    print(f"rows {len(df):,} features {len(features)} folds {len(folds)} burn-in to {burn_in.date()}",
          flush=True)
    names = [*GBM_VARIANTS, "logreg", "market"]
    per_model = {n: [] for n in names}
    per_fold = []
    for fold in folds:
        train = df[fold.train & (df["race_date"] >= burn_in).to_numpy()]
        test = df[fold.test & df["scorable"].to_numpy()]
        if test.empty:
            continue
        weights = compute_recency_sample_weights(train["race_date"])
        row = {"fold": fold.label, "test_races": int(test["race_id"].nunique())}
        for name, params in GBM_VARIANTS.items():
            model = _fit_gbm(params, train, features, weights)
            per_model[name].append(_per_race(test, model.predict_proba(test[features])[:, 1]))
            row[f"{name}_trees"] = int(model.best_iteration_ or params["n_estimators"])
        lr = RacePredictor._fit_linear_baseline(train[features], train["won"], sample_weight=weights)
        per_model["logreg"].append(_per_race(test, RacePredictor._baseline_predict_proba(lr, test[features])))
        per_model["market"].append(_per_race(test, test["mkt"].to_numpy()))
        for n in names:
            row[f"{n}_top1"] = round(float(per_model[n][-1]["top1"].mean()), 4)
        per_fold.append(row)
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    pooled = {n: pd.concat(per_model[n], ignore_index=True) for n in names}
    report = {"races": int(len(pooled["market"])), "per_fold": per_fold, "models": {}, "pairs": {}}
    for n in names:
        p = pooled[n]
        report["models"][n] = {m: round(float(p[m].mean()), 5) for m in ("top1", "top3", "ll")}
    pairs = [("gbm_prod", "logreg"), ("gbm_loose", "gbm_prod"), ("gbm_loose", "logreg"),
             ("gbm_prod", "market"), ("logreg", "market")]
    for a, b in pairs:
        report["pairs"][f"{a} - {b}"] = {
            m: bootstrap_roi_ci(pooled[a][m].to_numpy() - pooled[b][m].to_numpy(),
                                clusters=pooled[a]["race"].to_numpy(), n_boot=n_boot, scale=1.0)
            for m in ("top1", "ll")}
    return report


def render(r: dict) -> str:
    lines = [f"\n=== ranking comparison, {r['races']:,} development races ===",
             f"{'model':10s} {'top-1':>7s} {'top-3':>7s} {'race LL':>8s}"]
    for n, m in r["models"].items():
        lines.append(f"{n:10s} {m['top1']:7.4f} {m['top3']:7.4f} {m['ll']:8.4f}")
    lines.append("\npaired differences (race-bootstrap 95% CI):")
    for pair, ms in r["pairs"].items():
        t, ll = ms["top1"], ms["ll"]
        lines.append(f"  {pair:22s} top-1 {t['mean']:+.4f} [{t['ci_low']:+.4f},{t['ci_high']:+.4f}]"
                     f"   LL {ll['mean']:+.4f} [{ll['ci_low']:+.4f},{ll['ci_high']:+.4f}]")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--last-folds", type=int, default=None,
                    help="score only the most recent N development quarters")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    logging.disable(logging.INFO)
    report = run(args.max_folds, args.n_boot, last_folds=args.last_folds)
    print(render(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
