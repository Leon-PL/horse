"""A/B: conditional-logit (per-race softmax) LightGBM vs binary classifier.

The conditional-logit / Plackett-Luce(top-1) objective models
P(win) = softmax(score) within each race, which is the canonical model
for mutually-exclusive outcomes. Gradients are simple:
    grad_i = p_i - y_i        (p = within-race softmax of raw scores)
    hess_i = p_i * (1 - p_i)  (diagonal approximation)

Protocol: same two purged folds and features as fe_sweep, run twice —
no-market features and with-market features. Uses the new full-rebuild
snapshot (pedigree + TrueSkill).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, ndcg_score

from src.app_helpers import _drop_market_feature_columns
from src.model import get_feature_columns

RUN = "20260612_172827"
featured_full = pd.read_parquet(f"data/runs/{RUN}/featured_races.parquet")


def softmax_by_group(scores, group_ids):
    s = pd.Series(scores)
    g = pd.Series(group_ids)
    mx = s.groupby(g, sort=False).transform("max")
    ex = np.exp(np.clip(s - mx, -60, 0))
    return (ex / ex.groupby(g, sort=False).transform("sum")).to_numpy()


def make_softmax_objective(group_ids):
    def objective(preds, train_data):
        y = train_data.get_label()
        p = softmax_by_group(preds, group_ids)
        grad = p - y
        hess = np.maximum(p * (1.0 - p), 1e-6)
        return grad, hess
    return objective


PARAMS = dict(
    learning_rate=0.06, num_leaves=63, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.7, n_jobs=-1, verbose=-1, seed=42,
)


def run_eval(df, feature_cols, mode):
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
        y_tr = (df.loc[tr_mask, "finish_position"] == 1).astype(float).values
        X_ev = df.loc[ev_mask, feature_cols].values.astype(np.float32)
        g_tr = df.loc[tr_mask, "race_id"].values
        g_ev = df.loc[ev_mask, "race_id"].values

        if mode == "binary":
            model = lgb.LGBMClassifier(n_estimators=400, **{k: v for k, v in PARAMS.items() if k != "seed"},
                                       random_state=42)
            model.fit(X_tr, y_tr.astype(int))
            raw = model.predict_proba(X_ev)[:, 1]
            probs = pd.Series(raw).pipe(
                lambda s: s / s.groupby(pd.Series(g_ev), sort=False).transform("sum")
            ).to_numpy()
            raw_probs = raw  # un-normalised, for logloss comparability
        else:  # conditional logit
            train_set = lgb.Dataset(X_tr, label=y_tr)
            booster = lgb.train(
                {**PARAMS, "objective": make_softmax_objective(g_tr)},
                train_set, num_boost_round=400,
            )
            scores = booster.predict(X_ev)
            probs = softmax_by_group(scores, g_ev)
            raw_probs = probs

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


for label, drop_market in [("no-market", True), ("with-market", False)]:
    df = featured_full
    if drop_market:
        df, _ = _drop_market_feature_columns(featured_full.copy())
    cols = get_feature_columns(df)
    for mode in ["binary", "condlogit"]:
        m = run_eval(df, cols, mode)
        print(f"{label:12s} {mode:10s} ndcg1={m['ndcg1']:.4f} top1={m['top1']:.4f} "
              f"logloss={m['logloss']:.4f}", flush=True)
