"""A/B the get_feature_columns dtype fix on the saved featured snapshot.

Same parquet, two feature lists: the old int64/float64/int32/float32
allow-list vs the fixed numeric-dtype filter. No-market protocol,
same two purged folds as fe_sweep.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, ndcg_score

from src.app_helpers import _drop_market_feature_columns
from src.model import EXCLUDE_COLUMNS, get_feature_columns

featured = pd.read_parquet("data/runs/20260611_000627/featured_races.parquet")
featured, dropped = _drop_market_feature_columns(featured)
print(f"dropped {len(dropped)} market cols")


def old_feature_columns(df):
    return [
        c for c in df.columns
        if c not in EXCLUDE_COLUMNS and df[c].dtype in ["int64", "float64", "int32", "float32"]
    ]


def quick_walk_forward(df, feature_cols):
    df = df[df["finish_position"] > 0].copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)

    race_ids = df["race_id"].drop_duplicates().values
    race_dates = df.drop_duplicates("race_id")["race_date"].values
    n_races = len(race_ids)

    fold_metrics = []
    for lo, hi in [(0.70, 0.85), (0.85, 1.0)]:
        ev_beg, ev_end = int(n_races * lo), int(n_races * hi)
        purge_cut = race_dates[ev_beg] - np.timedelta64(7, "D")
        tr_races = set(race_ids[:ev_beg][race_dates[:ev_beg] <= purge_cut])
        ev_races = set(race_ids[ev_beg:ev_end])
        tr_mask = df["race_id"].isin(tr_races).values
        ev_mask = df["race_id"].isin(ev_races).values

        X_tr = df.loc[tr_mask, feature_cols].values.astype(np.float32)
        y_tr = (df.loc[tr_mask, "finish_position"] == 1).astype(int).values
        X_ev = df.loc[ev_mask, feature_cols].values.astype(np.float32)

        model = LGBMClassifier(
            n_estimators=400, learning_rate=0.06, num_leaves=63,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.7,
            n_jobs=-1, verbose=-1, random_state=42,
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_ev)[:, 1]

        ev_df = df.loc[ev_mask, ["race_id", "finish_position"]].copy()
        ev_df["prob"] = probs
        ndcg1, top1, total = [], 0, 0
        for _, grp in ev_df.groupby("race_id", sort=False):
            fp = grp["finish_position"].values
            p = grp["prob"].values
            if len(grp) < 2 or p.max() == p.min():
                continue
            rel = np.clip(8.0 - fp, 0, 7)
            total += 1
            ndcg1.append(ndcg_score([rel], [p], k=1))
            if fp[np.argmax(p)] == 1:
                top1 += 1
        won = (ev_df["finish_position"] == 1).astype(int).values
        fold_metrics.append({
            "ndcg1": float(np.mean(ndcg1)),
            "top1": top1 / total,
            "logloss": float(log_loss(won, np.clip(probs, 1e-12, 1 - 1e-12))),
        })
    return {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}


for name, cols in [
    ("old dtype filter", old_feature_columns(featured)),
    ("fixed dtype filter", get_feature_columns(featured)),
]:
    m = quick_walk_forward(featured, cols)
    print(f"{name:20s} n_features={len(cols):4d} "
          f"ndcg1={m['ndcg1']:.4f} top1={m['top1']:.4f} logloss={m['logloss']:.4f}")
