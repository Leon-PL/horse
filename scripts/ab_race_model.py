"""Stage-3 A/B: race-aware model vs the current per-horse-binary-then-normalize.

Same form features, same data, one held-out-by-date split. Compares:
  * CLASSIFIER  — LGBMClassifier on `won`, win_prob = p / per-race sum (current).
  * RANKER      — LGBMRanker (lambdarank) grouped by race, win_prob = softmax(score).
  * MARKET      — opening-odds implied prob (reference ceiling), where available.

Metrics per test race (races with exactly one winner, >=2 runners):
  * top1_hit  — fraction of races whose top-ranked runner won (discrimination).
  * winner_rank — mean predicted rank of the actual winner (lower better).
  * logloss   — -mean log(win_prob assigned to the actual winner) (calibration+disc).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pyarrow.parquet as pq
import lightgbm as lgb
import config
from src.feature_store import FeatureStore
from src.model import get_feature_columns
from src.place_market_model import PLACE_PARAMS

DIAG = ["race_id", "race_date", "won", "opening_odds"]


def _per_race_metrics(te: pd.DataFrame, score_col: str) -> dict:
    """te has race_id, won, and a per-race win_prob in score_col (sums to 1/race)."""
    rows = []
    for _, g in te.groupby("race_id", sort=False):
        if len(g) < 2 or int(g["won"].sum()) != 1:
            continue
        order = g[score_col].to_numpy()
        won = g["won"].to_numpy()
        ranks = (-order).argsort().argsort() + 1          # 1 = highest score
        wr = int(ranks[won == 1][0])
        p_winner = float(g[score_col].to_numpy()[won == 1][0])
        rows.append((wr, p_winner, len(g)))
    wr = np.array([r[0] for r in rows]); pw = np.array([r[1] for r in rows])
    return {
        "races": len(rows),
        "top1_hit": round(float((wr == 1).mean()), 4),
        "winner_rank": round(float(wr.mean()), 3),
        "logloss": round(float(-np.log(np.clip(pw, 1e-12, 1)).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2025-04-25")
    args = ap.parse_args()

    s = FeatureStore(); have = set(pq.read_schema(s.featured_path).names)
    allow = [l.strip() for l in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
             if l.strip() and not l.lstrip().startswith("#")]
    cols = [c for c in sorted(set(allow) | set(DIAG)) if c in have]
    df = pd.read_parquet(s.featured_path, columns=cols, engine="pyarrow")
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int)
    feats = [c for c in get_feature_columns(df) if c in df.columns]
    print(f"rows {len(df):,}  features {len(feats)}  cutoff {args.cutoff}", flush=True)

    tr = df[df["race_date"] < args.cutoff].copy()
    te = df[df["race_date"] >= args.cutoff].copy()
    # group-contiguous ordering for the ranker
    tr = tr.sort_values("race_id", kind="stable")
    tr_groups = tr.groupby("race_id", sort=False).size().to_numpy()
    print(f"train {len(tr):,} / test {len(te):,}", flush=True)

    # 1) CLASSIFIER (current)
    clf = lgb.LGBMClassifier(**PLACE_PARAMS).fit(tr[feats], tr["won"])
    te["p_clf"] = clf.predict_proba(te[feats])[:, 1]
    te["wp_clf"] = te["p_clf"] / te.groupby("race_id")["p_clf"].transform("sum")
    print("classifier trained", flush=True)

    # 2) RANKER (lambdarank)
    rnk_params = {k: v for k, v in PLACE_PARAMS.items() if k != "objective"}
    rnk = lgb.LGBMRanker(objective="lambdarank", **rnk_params)
    rnk.fit(tr[feats], tr["won"], group=tr_groups)
    te["s_rnk"] = rnk.predict(te[feats])
    te["_ex"] = te.groupby("race_id")["s_rnk"].transform(lambda x: np.exp(x - x.max()))
    te["wp_rnk"] = te["_ex"] / te.groupby("race_id")["_ex"].transform("sum")
    print("ranker trained", flush=True)

    # 3) MARKET baseline (opening odds implied), where present
    if "opening_odds" in te.columns:
        ip = 1.0 / pd.to_numeric(te["opening_odds"], errors="coerce")
        te["wp_mkt"] = ip / ip.groupby(te["race_id"]).transform("sum")

    print("\n=== A/B (held-out", args.cutoff, "+) ===")
    print("CLASSIFIER (current):", _per_race_metrics(te, "wp_clf"))
    print("RANKER (lambdarank) :", _per_race_metrics(te, "wp_rnk"))
    if "wp_mkt" in te.columns:
        mk = te.dropna(subset=["wp_mkt"])
        print("MARKET (open odds)  :", _per_race_metrics(mk, "wp_mkt"))


if __name__ == "__main__":
    main()
