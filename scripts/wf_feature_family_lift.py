"""Does a feature family make the production model better? Walk-forward A/B.

Triage (``scripts/triage_features.py``) asks whether a family carries
information the *market* lacks. This asks the other question an operator
needs before allowlisting it: does the production GBM rank and price races
better with the family than without? Both arms share everything - training
rows, recency weights, early stopping, test races (those with a complete T-5
market) - and differ only in the feature list. Per-race top-1 and race log
loss are compared pairwise with race-bootstrap CIs; the market is shown on
the same races for scale.

The family comes from the store (``--prefix``) or from a parquet keyed on
``race_id`` + ``horse_name`` (``--extra``, for candidates not yet in the
store).

Usage::

    python scripts/wf_feature_family_lift.py --prefix cp_ --last-folds 8
    python scripts/wf_feature_family_lift.py --prefix cp_ --extra cand.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.feature_store import FeatureStore
from src.model import compute_recency_sample_weights, get_feature_columns
from src.wf_eval import bootstrap_roi_ci, expanding_period_folds
import wf_ranking_compare as wrc  # noqa: E402

import config


def load(prefix: str, extra: str | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Base frame plus the family's columns. The family is always selected by
    *prefix*, so an ``--extra`` file's labels (``won``) can never become features."""
    df = wrc.load_frame()
    base = [c for c in get_feature_columns(df) if c not in ("ltp_300s", "mkt", "scorable")]
    path = extra or FeatureStore().featured_path
    fam = [c for c in pq.read_schema(path).names if c.startswith(prefix) and c not in base]
    cols = pd.read_parquet(path, columns=["race_id", "horse_name", *fam])
    fam = [c for c in fam if pd.api.types.is_numeric_dtype(cols[c])]
    cols = cols[["race_id", "horse_name", *fam]]
    cols["race_id"] = cols["race_id"].astype(str)
    df["race_id"] = df["race_id"].astype(str)
    n = len(df)
    df = df.merge(cols.drop_duplicates(["race_id", "horse_name"]), on=["race_id", "horse_name"],
                  how="left")
    assert len(df) == n, "family join duplicated rows"
    return df, base, fam


def run(prefix, extra, last_folds: int, n_boot: int) -> dict:
    df, base, fam = load(prefix, extra)
    folds = expanding_period_folds(df["race_date"], freq="Q", min_train_periods=8)
    folds = folds[-last_folds:]
    burn_in = df["race_date"].min() + pd.DateOffset(months=int(getattr(config, "BURN_IN_MONTHS", 8)))
    print(f"rows {len(df):,}; base {len(base)} features; family {len(fam)}: {fam}", flush=True)
    params = wrc.GBM_VARIANTS["gbm_prod"]
    arms = {"base": base, "with_family": base + fam}
    per = {k: [] for k in (*arms, "market")}
    for fold in folds:
        train = df[fold.train & (df["race_date"] >= burn_in).to_numpy()]
        test = df[fold.test & df["scorable"].to_numpy()]
        if test.empty:
            continue
        weights = compute_recency_sample_weights(train["race_date"])
        row = {"fold": fold.label, "races": int(test["race_id"].nunique())}
        for name, feats in arms.items():
            model = wrc._fit_gbm(params, train, feats, weights)
            per[name].append(wrc._per_race(test, model.predict_proba(test[feats])[:, 1]))
            row[f"{name}_ll"] = round(float(per[name][-1]["ll"].mean()), 4)
        per["market"].append(wrc._per_race(test, test["mkt"].to_numpy()))
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    pooled = {k: pd.concat(v, ignore_index=True) for k, v in per.items()}
    report = {"family": fam, "races": int(len(pooled["market"])), "models": {}, "pairs": {}}
    for k, p in pooled.items():
        report["models"][k] = {m: round(float(p[m].mean()), 5) for m in ("top1", "top3", "ll")}
    for a, b in [("with_family", "base"), ("base", "market"), ("with_family", "market")]:
        report["pairs"][f"{a} - {b}"] = {
            m: bootstrap_roi_ci(pooled[a][m].to_numpy() - pooled[b][m].to_numpy(),
                                clusters=pooled[a]["race"].to_numpy(), n_boot=n_boot, scale=1.0)
            for m in ("top1", "ll")}
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", required=True, help="family columns start with this")
    ap.add_argument("--extra", default=None,
                    help="parquet keyed on race_id + horse_name to take the family from "
                         "instead of the store")
    ap.add_argument("--last-folds", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    r = run(args.prefix, args.extra, args.last_folds, args.n_boot)
    print(f"\n=== family lift, {r['races']:,} development races ===")
    for k, m in r["models"].items():
        print(f"{k:12s} top-1 {m['top1']:.4f}  top-3 {m['top3']:.4f}  race LL {m['ll']:.4f}")
    for pair, ms in r["pairs"].items():
        t, ll = ms["top1"], ms["ll"]
        print(f"  {pair:24s} top-1 {t['mean']:+.4f} [{t['ci_low']:+.4f},{t['ci_high']:+.4f}]"
              f"   LL {ll['mean']:+.4f} [{ll['ci_low']:+.4f},{ll['ci_high']:+.4f}]")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
