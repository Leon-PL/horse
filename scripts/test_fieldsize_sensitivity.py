"""Prototype + triage: horse-specific field-size sensitivity features.

Hypothesis: some horses run worse than usual when many horses are running, in a
way the market underprices. We already feed the GENERIC field-size effect
(num_runners, base_win_rate=1/n, draw_x_field_size, small/large_field, pace
traffic). What's missing is a HORSE-SPECIFIC term: does THIS horse historically
underperform (relative to field) as the field grows?

Build (race-safe, strictly excludes the current race; horse appears once per race
so per-horse expanding sums shifted within the horse group are leak-free):

  perf            = 1 - (finish_pos-1)/(n-1)        in [0,1], higher=better,
                    field-size-normalised so mid-pack ~0.5 regardless of n.
  fs_slope        = expanding OLS slope of perf on field size over the horse's
                    PRIOR races, shrunk by n/(n+K). >0 => does BETTER in bigger
                    fields; <0 => worse in bigger fields (the effect asked about).
  fs_delta        = shrunk (mean perf in prior big-field runs - mean perf overall).
  closer_x_field  = pace_closer_flag * (num_runners/20)  -- traffic hurts closers;
                    a less market-legible cut than raw sensitivity.

Triage (cheap, decisive BEFORE any full retrain):
  1) corr of each feature with the MARKET RESIDUAL (won - mkt_imp). If ~0 the
     market already prices it (or it's noise) -> won't add CLV.
  2) Fund+Market value picks: ROI@BSP, mean CLV, gain importance, WITH vs WITHOUT
     the new features; plus the large-field (n>=14) subset where the effect should
     concentrate.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pyarrow.parquet as pq
import lightgbm as lgb
import config
from src.feature_store import FeatureStore
from src.model import get_feature_columns
from src.place_market_model import PLACE_PARAMS
from src.betfair import attach_bsp

CUT = "2025-04-25"; COMM = 0.02; K = 8.0
DIAG = ["race_id", "race_date", "won", "horse_id", "horse_name", "track",
        "off_time", "finish_position", "num_runners", "pace_closer_flag"]
NEW = ["fs_slope", "fs_delta", "closer_x_field"]


def roi_bsp(sub):
    b = pd.to_numeric(sub["bsp"], errors="coerce").to_numpy(); w = sub["won"].to_numpy().astype(bool)
    return float(np.where(w, (b - 1) * (1 - COMM), -1.0).mean()) * 100


def mean_clv(sub):  # ltp_300s / bsp ; >1 beats the close
    e = pd.to_numeric(sub["ltp_300s"], errors="coerce").to_numpy()
    b = pd.to_numeric(sub["bsp"], errors="coerce").to_numpy()
    return float(np.nanmean(e / b))


def main():
    s = FeatureStore(); have = set(pq.read_schema(s.featured_path).names)
    allow = [l.strip() for l in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
             if l.strip() and not l.lstrip().startswith("#")]
    cols = [c for c in sorted(set(allow) | set(DIAG)) if c in have]
    df = pd.read_parquet(s.featured_path, columns=cols)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.sort_values(["race_date", "race_id"]).reset_index(drop=True)
    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int)

    # ---- race-safe per-horse field-size sensitivity ----
    n = pd.to_numeric(df["num_runners"], errors="coerce")
    fp = pd.to_numeric(df["finish_position"], errors="coerce")
    perf = 1.0 - (fp - 1.0) / (n - 1.0).clip(lower=1)
    perf = perf.where((fp >= 1) & (n >= 2))
    ok = perf.notna()
    x = n.astype(float).where(ok)            # only finished races contribute
    big = (n >= 12).astype(float).where(ok)
    tmp = pd.DataFrame({
        "horse_id": df["horse_id"], "_x": x.fillna(0), "_y": perf.fillna(0),
        "_xy": (x * perf).fillna(0), "_xx": (x * x).fillna(0),
        "_by": (big * perf).fillna(0), "_b": big.fillna(0), "_ok": ok.astype(float),
    })
    g = tmp.groupby("horse_id", sort=False)
    def sh(col):
        # race-safe prior sum: grouped cumsum MINUS the current row's own value
        # == sum over the horse's strictly-prior races. No global .shift(1)
        # (which would leak across horses — CLAUDE.md hard rule #1).
        return (g[col].cumsum() - tmp[col]).clip(lower=0)
    cnt = sh("_ok"); sx = sh("_x"); sy = sh("_y"); sxy = sh("_xy"); sxx = sh("_xx")
    sby = sh("_by"); sb = sh("_b")
    cnt_safe = cnt.clip(lower=1)
    xbar = sx / cnt_safe; ybar = sy / cnt_safe
    cov = sxy / cnt_safe - xbar * ybar
    var = (sxx / cnt_safe - xbar * xbar).clip(lower=1e-6)
    slope_raw = cov / var
    shrink = cnt / (cnt + K)
    df["fs_slope"] = (slope_raw * shrink).where(cnt >= 3, 0.0).fillna(0.0)
    big_mean = (sby / sb.clip(lower=1))
    df["fs_delta"] = ((big_mean - ybar) * (sb / (sb + K))).where(sb >= 2, 0.0).fillna(0.0)
    df["closer_x_field"] = pd.to_numeric(df["pace_closer_flag"], errors="coerce").fillna(0) * (n / 20.0)

    # ---- market ----
    df = attach_bsp(df, cols=("bsp", "ltp_300s"))
    p = pd.to_numeric(df["ltp_300s"], errors="coerce"); b = pd.to_numeric(df["bsp"], errors="coerce")
    df = df[(p > 1) & (b > 1)].copy()
    o = pd.to_numeric(df["ltp_300s"], errors="coerce"); ip = 1.0 / o
    df["mkt_imp"] = ip / ip.groupby(df["race_id"]).transform("sum")
    df["mkt_logodds"] = np.log(o.clip(lower=1.01)); df["mkt_oddsrank"] = o.groupby(df["race_id"]).rank()
    mkt = ["mkt_imp", "mkt_logodds", "mkt_oddsrank"]
    form = [c for c in get_feature_columns(df) if c in df.columns
            and c not in ("bsp", "ltp_300s") + tuple(mkt) + tuple(NEW)]

    # ---- (1) orthogonality: corr with market residual ----
    resid = df["won"] - df["mkt_imp"]
    print("=== coverage / corr with MARKET RESIDUAL (won - mkt_imp) ===")
    print(f"  rows {len(df):,}   big-field(n>=12) {int((n.reindex(df.index)>=12).sum()):,}")
    for c in NEW:
        v = df[c]
        nz = float((v != 0).mean())
        cc = float(np.corrcoef(v, resid)[0, 1]) if v.std() > 0 else 0.0
        print(f"  {c:16} nonzero={nz:5.1%}  corr_resid={cc:+.4f}")

    tr = df[df["race_date"] < CUT]; te = df[df["race_date"] >= CUT].copy()

    def run(feats, label):
        m = lgb.LGBMClassifier(**PLACE_PARAMS).fit(tr[feats], tr["won"])
        te["_p"] = m.predict_proba(te[feats])[:, 1]
        te["wp"] = te["_p"] / te.groupby("race_id")["_p"].transform("sum")
        te["edge"] = te["wp"] / te["mkt_imp"].clip(lower=1e-9) - 1.0
        band = pd.to_numeric(te["ltp_300s"], errors="coerce").between(1.5, 50)
        vp = te[(te["edge"] > 0) & band]
        nr = pd.to_numeric(te["num_runners"], errors="coerce")
        vp_big = te[(te["edge"] > 0) & band & (nr >= 14)]
        print(f"\n--- {label} ({len(feats)} feats) ---")
        print(f"  value picks n={len(vp):,}  ROI@BSP {roi_bsp(vp):+.2f}%  meanCLV {mean_clv(vp):.4f}")
        print(f"  large-field(n>=14) picks n={len(vp_big):,}  ROI@BSP {roi_bsp(vp_big):+.2f}%  meanCLV {mean_clv(vp_big):.4f}")
        return m

    base = run(form + mkt, "BASELINE Fund+Market")
    withf = run(form + mkt + NEW, "WITH field-size feats")
    imp = pd.Series(withf.booster_.feature_importance("gain"), index=form + mkt + NEW)
    print("\n=== gain importance of NEW feats (context: mkt_imp ~ {:.0f}) ===".format(imp["mkt_imp"]))
    for c in NEW:
        print(f"  {c:16} gain={imp[c]:.0f}   rank {int((imp > imp[c]).sum())+1}/{len(imp)}")


if __name__ == "__main__":
    main()
