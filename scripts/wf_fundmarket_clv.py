"""Walk-forward: does Fund+Market beat the market across time, and is it bankable?

Pre-off market signal = Betfair ltp_300s (~5 min out → tradeable). Expanding
quarterly folds on the rebuilt store. Each fold trains a Fund+Market win model
(form + pre-off market features) and forward-tests the next quarter:

  PREDICTIVE  — top-1 hit & logloss of FUND+MARKET vs MARKET-RAW (ltp_300s implied).
  TRADEABLE   — back value picks (fund+market win_prob > market implied) at ltp_300s;
                CLV vs BSP (mean ltp_300s/bsp; >1 = beat the close) and ROI@entry
                (Betfair win, 2% commission). Pooled + per-fold.
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pyarrow.parquet as pq
import lightgbm as lgb
import config
from src.feature_store import FeatureStore
from src.model import get_feature_columns
from src.place_market_model import PLACE_PARAMS
from src.betfair import attach_bsp

DIAG = ["race_id", "race_date", "won", "track", "off_time", "horse_name"]
COMM = 0.02


def _pred_metrics(te, col):
    rows = []
    for _, g in te.groupby("race_id", sort=False):
        if len(g) < 2 or int(g["won"].sum()) != 1:
            continue
        won = g["won"].to_numpy(); order = g[col].to_numpy()
        ranks = (-order).argsort().argsort() + 1
        rows.append((int(ranks[won == 1][0]), float(order[won == 1][0])))
    wr = np.array([r[0] for r in rows]); pw = np.array([r[1] for r in rows])
    return {"races": len(rows), "top1": round(float((wr == 1).mean()), 4),
            "logloss": round(float(-np.log(np.clip(pw, 1e-12, 1)).mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", type=float, default=0.0, help="back when fund_prob/mkt_imp-1 > edge")
    ap.add_argument("--min-train-quarters", type=int, default=8)
    args = ap.parse_args()

    s = FeatureStore(); have = set(pq.read_schema(s.featured_path).names)
    allow = [l.strip() for l in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
             if l.strip() and not l.lstrip().startswith("#")]
    cols = [c for c in sorted(set(allow) | set(DIAG)) if c in have]
    df = pd.read_parquet(s.featured_path, columns=cols, engine="pyarrow")
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int)
    df = attach_bsp(df, cols=("bsp", "ltp_300s"))
    p300 = pd.to_numeric(df["ltp_300s"], errors="coerce"); bsp = pd.to_numeric(df["bsp"], errors="coerce")
    df = df[(p300 > 1.0) & (bsp > 1.0)].copy()
    # pre-off market features (from ltp_300s) + per-race implied prob
    o = pd.to_numeric(df["ltp_300s"], errors="coerce"); ip = 1.0 / o
    df["mkt_imp"] = ip / ip.groupby(df["race_id"]).transform("sum")
    df["mkt_logodds"] = np.log(o.clip(lower=1.01))
    df["mkt_oddsrank"] = o.groupby(df["race_id"]).rank()
    mkt = ["mkt_imp", "mkt_logodds", "mkt_oddsrank"]
    form = [c for c in get_feature_columns(df) if c in df.columns and c not in ("bsp", "ltp_300s") + tuple(mkt)]
    df["_q"] = df["race_date"].dt.to_period("Q")
    print(f"rows(ltp_300s&bsp) {len(df):,}  form {len(form)}  edge {args.edge}", flush=True)

    quarters = sorted(df["_q"].unique())
    pf_pred_fm, pf_pred_mk = [], []
    clv_chunks, pnl_chunks, pnl_bsp_chunks, per_fold = [], [], [], []
    for q in quarters[args.min_train_quarters:]:
        tr = df[df["_q"] < q]; te = df[df["_q"] == q].copy()
        if len(tr) < 1000 or te.empty:
            continue
        m = lgb.LGBMClassifier(**PLACE_PARAMS).fit(tr[form + mkt], tr["won"])
        te["_p"] = m.predict_proba(te[form + mkt])[:, 1]
        te["wp_fm"] = te["_p"] / te.groupby("race_id")["_p"].transform("sum")
        pf_pred_fm.append(_pred_metrics(te, "wp_fm")); pf_pred_mk.append(_pred_metrics(te, "mkt_imp"))
        # tradeable: value vs the pre-off implied price
        edge = te["wp_fm"] / te["mkt_imp"].clip(lower=1e-9) - 1.0
        sel = te[(edge > args.edge) & te["ltp_300s"].between(1.5, 50)]
        if not sel.empty:
            entry = pd.to_numeric(sel["ltp_300s"], errors="coerce").to_numpy()
            close = pd.to_numeric(sel["bsp"], errors="coerce").to_numpy()
            won = sel["won"].to_numpy().astype(bool)
            clv = entry / close
            pnl = np.where(won, (entry - 1.0) * (1 - COMM), -1.0)       # ROI@entry (ltp_300s)
            pnl_bsp = np.where(won, (close - 1.0) * (1 - COMM), -1.0)   # ROI@BSP — artifact-proof
            clv_chunks.append(clv); pnl_chunks.append(pnl); pnl_bsp_chunks.append(pnl_bsp)
            per_fold.append({"q": str(q), "bets": int(len(sel)),
                             "mean_clv": round(float(clv.mean()), 4),
                             "roi_entry": round(float(pnl.mean() * 100), 2),
                             "roi_bsp": round(float(pnl_bsp.mean() * 100), 2)})
        print(f"fold {q} done", flush=True)

    def _pool(ms):
        return {k: round(float(np.mean([m[k] for m in ms])), 4) for k in ("top1", "logloss")}
    clv = np.concatenate(clv_chunks); pnl = np.concatenate(pnl_chunks)
    pnl_bsp = np.concatenate(pnl_bsp_chunks); n = len(pnl)
    rng = np.random.default_rng(0)
    def _ci(a, scale=100):
        b = a[rng.integers(0, n, size=(2000, n))].mean(axis=1) * scale
        return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
    out = {
        "predictive": {"FUND+MARKET": _pool(pf_pred_fm), "MARKET-RAW": _pool(pf_pred_mk)},
        "tradeable": {
            "n_bets": int(n), "folds": len(per_fold),
            "mean_clv": round(float(clv.mean()), 4), "clv_ci": _ci(clv, 1),
            "roi_entry": round(float(pnl.mean() * 100), 2), "roi_entry_ci": _ci(pnl),
            "roi_bsp": round(float(pnl_bsp.mean() * 100), 2), "roi_bsp_ci": _ci(pnl_bsp),
            "folds_bsp_positive": int(sum(f["roi_bsp"] > 0 for f in per_fold)),
        },
        "per_fold": per_fold,
    }
    print("\n=== PREDICTIVE (pooled across folds) ===")
    print("  FUND+MARKET:", out["predictive"]["FUND+MARKET"])
    print("  MARKET-RAW :", out["predictive"]["MARKET-RAW"])
    t = out["tradeable"]
    print("\n=== TRADEABLE (value picks @ ltp_300s, 2% comm) ===")
    print(f"  bets={t['n_bets']:,} folds={t['folds']}  mean_CLV={t['mean_clv']} CI{t['clv_ci']}")
    print(f"  ROI@entry(ltp_300s) = {t['roi_entry']}% CI{t['roi_entry_ci']}   <- inflated by stale 300s price")
    print(f"  ROI@BSP (artifact-proof) = {t['roi_bsp']}% CI{t['roi_bsp_ci']}  folds+ {t['folds_bsp_positive']}/{t['folds']}")
    for f in per_fold:
        print(f"   {f['q']}: {f['bets']:5} bets  CLV {f['mean_clv']:.4f}  ROI@entry {f['roi_entry']:+.2f}%  ROI@BSP {f['roi_bsp']:+.2f}%")
    print("\nJSON:", json.dumps(out))


if __name__ == "__main__":
    main()
