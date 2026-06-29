"""Triage: does public sale price carry signal the market underprices?

Hypothesis: sale price proxies the private stable expectation that beats us on
debutants/lightly-raced horses. But the sale price is PUBLIC, so the market may
already price it. Decisive tests (priced subset only; NO imputation of the rest):

  1) corr(log_sale_price, market residual won-mkt_imp) overall AND on DEBUTANTS
     (horse_prev_races==0) — the hypothesised sweet spot. ~0 => already priced.
  2) DEBUT AUC: does adding log_sale_price to a debut win-model beat market-only?
  3) Fund+Market value picks ROI@BSP + CLV, baseline vs +sale features, with the
     ROI@BSP-vs-CLV leak tripwire.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import config
from src.feature_store import FeatureStore
from src.model import get_feature_columns
from src.place_market_model import PLACE_PARAMS
from src.betfair import attach_bsp

CUT = "2025-04-25"; COMM = 0.02
DIAG = ["race_id", "race_date", "won", "horse_id", "horse_name", "track",
        "off_time", "horse_prev_races"]
NEW = ["log_sale_price", "has_sale_price"]


def roi_bsp(s):
    b = pd.to_numeric(s["bsp"], errors="coerce").to_numpy(); w = s["won"].to_numpy().astype(bool)
    return float(np.where(w, (b - 1) * (1 - COMM), -1.0).mean()) * 100


def mean_clv(s):
    e = pd.to_numeric(s["ltp_300s"], errors="coerce").to_numpy(); b = pd.to_numeric(s["bsp"], errors="coerce").to_numpy()
    return float(np.nanmean(e / b))


def main():
    s = FeatureStore(); have = set(pq.read_schema(s.featured_path).names)
    allow = [l.strip() for l in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
             if l.strip() and not l.lstrip().startswith("#")]
    cols = [c for c in sorted(set(allow) | set(DIAG)) if c in have]
    df = pd.read_parquet(s.featured_path, columns=cols)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int)

    price = pd.read_parquet("data/processed/horse_sale_price.parquet")
    price["horse_id"] = price["horse_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    df = df.merge(price, on="horse_id", how="left")
    df["has_sale_price"] = df["sale_price_guineas"].notna().astype(int)
    df["log_sale_price"] = np.log(pd.to_numeric(df["sale_price_guineas"], errors="coerce"))
    # NO imputation: leave log NaN where unpriced; LightGBM handles NaN natively.
    print(f"rows {len(df):,}  with sale price {df['has_sale_price'].mean()*100:.1f}%")

    df = attach_bsp(df, cols=("bsp", "ltp_300s"))
    p = pd.to_numeric(df["ltp_300s"], errors="coerce"); b = pd.to_numeric(df["bsp"], errors="coerce")
    df = df[(p > 1) & (b > 1)].copy()
    o = pd.to_numeric(df["ltp_300s"], errors="coerce"); ip = 1.0 / o
    df["mkt_imp"] = ip / ip.groupby(df["race_id"]).transform("sum")
    df["mkt_logodds"] = np.log(o.clip(lower=1.01)); df["mkt_oddsrank"] = o.groupby(df["race_id"]).rank()
    mkt = ["mkt_imp", "mkt_logodds", "mkt_oddsrank"]
    form = [c for c in get_feature_columns(df) if c in df.columns
            and c not in ("bsp", "ltp_300s") + tuple(mkt) + tuple(NEW)]
    prev = pd.to_numeric(df["horse_prev_races"], errors="coerce").fillna(99)

    # ---- (1) orthogonality vs market residual ----
    resid = df["won"] - df["mkt_imp"]
    priced = df["has_sale_price"] == 1
    deb = priced & (prev == 0)
    def corr(mask):
        x = df.loc[mask, "log_sale_price"]; y = resid[mask]
        m = x.notna() & y.notna()
        return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 50 else float("nan")
    print("\n=== orthogonality: corr(log_sale_price, won - mkt_imp) ===")
    print(f"  all priced rows  n={int(priced.sum()):,}   corr={corr(priced):+.4f}")
    print(f"  DEBUTANTS priced n={int(deb.sum()):,}   corr={corr(deb):+.4f}   <- the hypothesised edge")
    # does price predict won at all (sanity, not vs market)?
    mm = priced & df["log_sale_price"].notna()
    print(f"  sanity AUC(log_price -> won) all priced: {roc_auc_score(df.loc[mm,'won'], df.loc[mm,'log_sale_price']):.3f}")

    # ---- (2) DEBUT model: market vs market+price ----
    dtr = df[deb & (df["race_date"] < CUT)]; dte = df[deb & (df["race_date"] >= CUT)]
    if len(dte) > 300:
        Xtr_m = dtr[["mkt_imp"]]; Xte_m = dte[["mkt_imp"]]
        for label, extra in [("market only", []), ("market + log_price", ["log_sale_price"])]:
            feats = ["mkt_imp"] + extra
            m = lgb.LGBMClassifier(**{**PLACE_PARAMS, "n_estimators": 300}).fit(dtr[feats], dtr["won"])
            auc = roc_auc_score(dte["won"], m.predict_proba(dte[feats])[:, 1])
            print(f"  DEBUT win-AUC ({label}): {auc:.4f}")

    # ---- (3) full Fund+Market value picks: baseline vs +sale ----
    tr = df[df["race_date"] < CUT]; te = df[df["race_date"] >= CUT].copy()
    def run(feats, label):
        m = lgb.LGBMClassifier(**PLACE_PARAMS).fit(tr[feats], tr["won"])
        te["_p"] = m.predict_proba(te[feats])[:, 1]
        te["wp"] = te["_p"] / te.groupby("race_id")["_p"].transform("sum")
        te["edge"] = te["wp"] / te["mkt_imp"].clip(lower=1e-9) - 1.0
        band = pd.to_numeric(te["ltp_300s"], errors="coerce").between(1.5, 50)
        vp = te[(te["edge"] > 0) & band]
        print(f"  {label:24} value n={len(vp):,}  ROI@BSP {roi_bsp(vp):+.2f}%  CLV {mean_clv(vp):.4f}")
        return m
    print("\n=== Fund+Market value picks (leak tripwire: big ROI@BSP + flat CLV == leak) ===")
    run(form + mkt, "BASELINE")
    withf = run(form + mkt + NEW, "WITH sale price")
    imp = pd.Series(withf.booster_.feature_importance("gain"), index=form + mkt + NEW)
    for c in NEW:
        print(f"    {c:16} gain={imp[c]:.0f}  rank {int((imp>imp[c]).sum())+1}/{len(imp)}")


if __name__ == "__main__":
    main()
