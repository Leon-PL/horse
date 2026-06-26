"""Stage-3 decisive test: does the fundamental model add anything on top of the market?

Same held-out split, restricted to runners with an opening price (so every model
is judged on the same races). Trains:
  * FUNDAMENTAL   — classifier on form features only.
  * MARKET-RAW    — opening-odds implied prob (no model), per-race normalised.
  * MARKET-MODEL  — classifier on market features only (calibrated market).
  * FUND + MARKET — classifier on form + market features.

The test: does FUND+MARKET beat MARKET-MODEL? If yes, our features carry signal
the market hasn't fully priced. If ~equal, they add nothing the market lacks.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pyarrow.parquet as pq
import lightgbm as lgb
import config
from src.feature_store import FeatureStore
from src.model import get_feature_columns
from src.place_market_model import PLACE_PARAMS
from src.betfair import attach_bsp

# Use Betfair BSP as the market signal: ~80% coverage back to 2020 (opening_odds
# is recent-only) and the sharpest benchmark — "do fundamentals add to BSP?".
DIAG = ["race_id", "race_date", "won", "track", "off_time", "horse_name"]
MKT_PRICE = "bsp"


def _metrics(te, col):
    rows = []
    for _, g in te.groupby("race_id", sort=False):
        if len(g) < 2 or int(g["won"].sum()) != 1:
            continue
        order = g[col].to_numpy(); won = g["won"].to_numpy()
        ranks = (-order).argsort().argsort() + 1
        rows.append((int(ranks[won == 1][0]), float(g[col].to_numpy()[won == 1][0])))
    wr = np.array([r[0] for r in rows]); pw = np.array([r[1] for r in rows])
    return {"races": len(rows), "top1_hit": round(float((wr == 1).mean()), 4),
            "winner_rank": round(float(wr.mean()), 3),
            "logloss": round(float(-np.log(np.clip(pw, 1e-12, 1)).mean()), 4)}


def _mkt_feats(df):
    o = pd.to_numeric(df[MKT_PRICE], errors="coerce")
    ip = 1.0 / o
    df["mkt_imp"] = ip / ip.groupby(df["race_id"]).transform("sum")
    df["mkt_logodds"] = np.log(o.clip(lower=1.01))
    df["mkt_oddsrank"] = o.groupby(df["race_id"]).rank()
    return ["mkt_imp", "mkt_logodds", "mkt_oddsrank"]


def _wp(model, te, feats, col):
    te[col + "_p"] = model.predict_proba(te[feats])[:, 1]
    te[col] = te[col + "_p"] / te.groupby("race_id")[col + "_p"].transform("sum")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cutoff", default="2025-04-25")
    args = ap.parse_args()
    s = FeatureStore(); have = set(pq.read_schema(s.featured_path).names)
    allow = [l.strip() for l in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
             if l.strip() and not l.lstrip().startswith("#")]
    cols = [c for c in sorted(set(allow) | set(DIAG)) if c in have]
    df = pd.read_parquet(s.featured_path, columns=cols, engine="pyarrow")
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int)
    df = attach_bsp(df, cols=(MKT_PRICE,))
    df = df[pd.to_numeric(df[MKT_PRICE], errors="coerce") > 1.0].copy()  # same races for all
    form = [c for c in get_feature_columns(df) if c in df.columns and c != MKT_PRICE]
    mkt = _mkt_feats(df)
    print(f"rows(with {MKT_PRICE}) {len(df):,}  form {len(form)}  cutoff {args.cutoff}", flush=True)

    tr = df[df["race_date"] < args.cutoff].copy()
    te = df[df["race_date"] >= args.cutoff].copy()
    print(f"train {len(tr):,} / test {len(te):,}", flush=True)

    fund = lgb.LGBMClassifier(**PLACE_PARAMS).fit(tr[form], tr["won"]); _wp(fund, te, form, "wp_fund")
    print("fundamental trained", flush=True)
    mktm = lgb.LGBMClassifier(**PLACE_PARAMS).fit(tr[mkt], tr["won"]); _wp(mktm, te, mkt, "wp_mktm")
    print("market-model trained", flush=True)
    both = lgb.LGBMClassifier(**PLACE_PARAMS).fit(tr[form + mkt], tr["won"]); _wp(both, te, form + mkt, "wp_both")
    print("fund+market trained", flush=True)

    print("\n=== DOES FUNDAMENTAL ADD TO MARKET? (held-out", args.cutoff, "+) ===")
    print("FUNDAMENTAL   :", _metrics(te, "wp_fund"))
    print("MARKET-RAW    :", _metrics(te, "mkt_imp"))
    print("MARKET-MODEL  :", _metrics(te, "wp_mktm"))
    print("FUND + MARKET :", _metrics(te, "wp_both"))


if __name__ == "__main__":
    main()
