"""Audit the Fund+Market BSP edge: is it real signal or a subtle leak?

Single split. Trains Fund+Market, confirms ROI@BSP of the value picks, then
attributes it: gain importance, and SHAP contributions averaged over the VALUE
PICKS (what pushes win_prob above the market). Plus a debut-row leakage check
(a horse with no prior races must not have informative 'history' features).
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
from src.bet_settlement import settle_win_exchange_bets

CUT = "2025-04-25"
DIAG = ["race_id", "race_date", "won", "track", "off_time", "horse_name", "horse_prev_races"]


def roi_bsp(sub):
    b = pd.to_numeric(sub["bsp"], errors="coerce").to_numpy(); w = sub["won"].to_numpy().astype(bool)
    return float(settle_win_exchange_bets(b, w).mean()) * 100


def main():
    s = FeatureStore(); have = set(pq.read_schema(s.featured_path).names)
    allow = [l.strip() for l in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
             if l.strip() and not l.lstrip().startswith("#")]
    cols = [c for c in sorted(set(allow) | set(DIAG)) if c in have]
    df = pd.read_parquet(s.featured_path, columns=cols)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int)
    df = attach_bsp(df, cols=("bsp", "ltp_300s"))
    p = pd.to_numeric(df["ltp_300s"], errors="coerce"); b = pd.to_numeric(df["bsp"], errors="coerce")
    df = df[(p > 1) & (b > 1)].copy()
    o = pd.to_numeric(df["ltp_300s"], errors="coerce"); ip = 1.0 / o
    df["mkt_imp"] = ip / ip.groupby(df["race_id"]).transform("sum")
    df["mkt_logodds"] = np.log(o.clip(lower=1.01)); df["mkt_oddsrank"] = o.groupby(df["race_id"]).rank()
    mkt = ["mkt_imp", "mkt_logodds", "mkt_oddsrank"]
    form = [c for c in get_feature_columns(df) if c in df.columns and c not in ("bsp", "ltp_300s") + tuple(mkt)]
    if "--ablate-trainer" in sys.argv:
        drop = [c for c in form if any(k in c.lower() for k in ["trainer", "jt_", "stable", "same_trainer"])]
        form = [c for c in form if c not in drop]
        print(f"ABLATING {len(drop)} trainer/jt/stable features")
    feats = form + mkt

    tr = df[df["race_date"] < CUT]; te = df[df["race_date"] >= CUT].copy()
    m = lgb.LGBMClassifier(**PLACE_PARAMS).fit(tr[feats], tr["won"])
    te["_p"] = m.predict_proba(te[feats])[:, 1]
    te["wp"] = te["_p"] / te.groupby("race_id")["_p"].transform("sum")
    te["edge"] = te["wp"] / te["mkt_imp"].clip(lower=1e-9) - 1.0
    vp = te[(te["edge"] > 0) & pd.to_numeric(te["ltp_300s"], errors="coerce").between(1.5, 50)]
    print(f"train {len(tr):,} / test {len(te):,}  value-picks {len(vp):,}")
    print(f"ROI@BSP value picks (vs ltp_300s ref): {roi_bsp(vp):+.2f}%   | bet-all@BSP: {roi_bsp(te[pd.to_numeric(te.ltp_300s,errors='coerce').between(1.5,50)]):+.2f}%")

    # --- reference-mismatch diagnostic: is the model actually better than BSP? ---
    ib = 1.0 / pd.to_numeric(te["bsp"], errors="coerce"); te["bsp_imp"] = ib / ib.groupby(te["race_id"]).transform("sum")
    def _m(col):
        rows = []
        for _, g in te.groupby("race_id", sort=False):
            if len(g) < 2 or int(g["won"].sum()) != 1: continue
            order = g[col].to_numpy(); won = g["won"].to_numpy()
            rows.append((int((-order).argsort().argsort()[won == 1][0]) + 1, float(order[won == 1][0])))
        wr = np.array([r[0] for r in rows]); pw = np.array([r[1] for r in rows])
        return f"top1={ (wr==1).mean():.4f} logloss={-np.log(np.clip(pw,1e-12,1)).mean():.4f}"
    print(f"  PREDICTIVE  model: {_m('wp')}")
    print(f"  PREDICTIVE  ltp300:{_m('mkt_imp')}")
    print(f"  PREDICTIVE  BSP:   {_m('bsp_imp')}   <- if BSP beats model, model can't truly beat BSP")
    selb = te[(te["wp"] > te["bsp_imp"]) & pd.to_numeric(te.ltp_300s, errors="coerce").between(1.5, 50)]
    print(f"  value vs BSP-ref, settle@BSP: ROI {roi_bsp(selb):+.2f}%  n={len(selb):,}  "
          f"<- ~-2% here means the +7% was the ltp_300s-vs-BSP reference mismatch")

    # gain importance
    imp = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=feats).sort_values(ascending=False)
    print("\n=== top-15 GAIN importance ===")
    for f in imp.head(15).index: print(f"  {f:34} {imp[f]:.0f}{'  [MKT]' if f in mkt else ''}")

    # SHAP contributions averaged over the value picks (what pushed wp UP)
    contrib = m.booster_.predict(vp[feats], pred_contrib=True)[:, :-1]
    mc = pd.Series(contrib.mean(axis=0), index=feats).sort_values(ascending=False)
    print("\n=== features pushing win_prob UP on value picks (mean SHAP) ===")
    for f in mc.head(12).index: print(f"  {f:34} {mc[f]:+.4f}{'  [MKT]' if f in mkt else ''}")

    # stablemate-leak test: does the edge concentrate in multi-runner-trainer rows?
    if "same_trainer_runners" in te.columns:
        st = pd.to_numeric(vp["same_trainer_runners"], errors="coerce").fillna(1)
        solo = vp[st <= 1]; multi = vp[st > 1]
        teband = te[pd.to_numeric(te.ltp_300s, errors="coerce").between(1.5, 50)]
        sto = pd.to_numeric(teband["same_trainer_runners"], errors="coerce").fillna(1)
        print("\n=== STABLEMATE-LEAK TEST (value picks ROI@BSP) ===")
        print(f"  solo trainer (==1 runner): n={len(solo):,}  ROI@BSP {roi_bsp(solo):+.2f}%  "
              f"(pool {roi_bsp(teband[sto<=1]):+.2f}%)")
        print(f"  multi-runner trainer (>1): n={len(multi):,}  ROI@BSP {roi_bsp(multi):+.2f}%  "
              f"(pool {roi_bsp(teband[sto>1]):+.2f}%)")

    # debut leakage check
    if "horse_prev_races" in te.columns:
        deb = te[pd.to_numeric(te["horse_prev_races"], errors="coerce").fillna(0) == 0]
        if len(deb) > 200:
            from sklearn.metrics import roc_auc_score
            try:
                auc_m = roc_auc_score(deb["won"], deb["wp"]); auc_k = roc_auc_score(deb["won"], deb["mkt_imp"])
                print(f"\n=== DEBUT rows (no prior races): n={len(deb):,} ===")
                print(f"  model wp AUC vs won: {auc_m:.3f}   market AUC: {auc_k:.3f}  "
                      f"(model >> market on debuts would imply leaked 'history')")
            except Exception as e:
                print("debut auc failed:", e)


if __name__ == "__main__":
    main()
