"""Triage: do the intent-signal families carry anything the market misses?

Families (all engineered into the rebuilt store):
  WS      — wind_surgery_run / wind_ops_prior / first_wind_surgery /
            prev_run_wind_surgery  (declared wind op = condition info)
  TRAVEL  — trainer_travel_km / _known / _far / trainer_runners_at_meeting /
            trainer_lone_long_trip (yard->track distance, lone-runner intent)
  BOOKING — trainer_first_string_by_jockey / jockey_elo_vs_best_stablemate /
            jockey_booking_vs_market / trainer_runner_rank_by_jockey

Protocol (same as scripts/test_sale_price_signal.py, the family-triage template):
  1) Market-residual analysis: mean(won - mkt_imp) on each flag subset, corr
     for continuous features. ~0 => the market already prices it.
  2) Fund+Market value picks, baseline vs +family: ROI@BSP + CLV, with the
     ROI@BSP-vs-CLV leak tripwire (big ROI jump + flat CLV == leak/noise).
  3) LightGBM gain ranks for the new columns (are they even used?).
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

CUT = "2025-04-25"; DIAG = ["race_id", "race_date", "won", "horse_id", "horse_name", "track",
        "off_time", "horse_prev_races", "race_type"]

FAMILIES = {
    "WS": ["wind_surgery_run", "wind_ops_prior", "first_wind_surgery",
           "prev_run_wind_surgery"],
    "TRAVEL": ["trainer_travel_km", "trainer_travel_known", "trainer_travel_far",
               "trainer_runners_at_meeting", "trainer_lone_long_trip"],
    "BOOKING": ["trainer_first_string_by_jockey", "jockey_elo_vs_best_stablemate",
                "jockey_booking_vs_market", "trainer_runner_rank_by_jockey"],
}
NEW = [c for cols in FAMILIES.values() for c in cols]

# (feature, is_binary_flag, subset label) for the residual scan
RESID_SPECS = [
    ("wind_surgery_run", True, "first run after wind op"),
    ("first_wind_surgery", True, "FIRST ever wind op"),
    ("prev_run_wind_surgery", True, "second run after op"),
    ("trainer_travel_far", True, "trip > 250 km"),
    ("trainer_lone_long_trip", True, "lone runner, long trip"),
    ("trainer_first_string_by_jockey", True, "stable's best-ridden runner"),
    ("trainer_travel_km", False, ""),
    ("jockey_booking_vs_market", False, ""),
    ("jockey_elo_vs_best_stablemate", False, ""),
]


def roi_bsp(s):
    b = pd.to_numeric(s["bsp"], errors="coerce").to_numpy()
    w = s["won"].to_numpy().astype(bool)
    return float(settle_win_exchange_bets(b, w).mean()) * 100


def mean_clv(s):
    e = pd.to_numeric(s["ltp_300s"], errors="coerce").to_numpy()
    b = pd.to_numeric(s["bsp"], errors="coerce").to_numpy()
    return float(np.nanmean(e / b))


def main():
    s = FeatureStore(); have = set(pq.read_schema(s.featured_path).names)
    missing = [c for c in NEW if c not in have]
    if missing:
        raise SystemExit(f"store lacks {missing} — rebuild first")
    allow = [l.strip() for l in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
             if l.strip() and not l.lstrip().startswith("#")]
    cols = [c for c in sorted(set(allow) | set(DIAG) | set(NEW)) if c in have]
    df = pd.read_parquet(s.featured_path, columns=cols)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int)

    df = attach_bsp(df, cols=("bsp", "ltp_300s"))
    p = pd.to_numeric(df["ltp_300s"], errors="coerce")
    b = pd.to_numeric(df["bsp"], errors="coerce")
    df = df[(p > 1) & (b > 1)].copy()
    o = pd.to_numeric(df["ltp_300s"], errors="coerce"); ip = 1.0 / o
    df["mkt_imp"] = ip / ip.groupby(df["race_id"]).transform("sum")
    df["mkt_logodds"] = np.log(o.clip(lower=1.01))
    df["mkt_oddsrank"] = o.groupby(df["race_id"]).rank()
    mkt = ["mkt_imp", "mkt_logodds", "mkt_oddsrank"]
    form = [c for c in get_feature_columns(df) if c in df.columns
            and c not in ("bsp", "ltp_300s") + tuple(mkt) + tuple(NEW)]
    print(f"rows {len(df):,}  form feats {len(form)}  cutoff {CUT}")

    # ---- (1) market-residual scan ----
    resid = (df["won"] - df["mkt_imp"]).astype(float)
    print("\n=== market residual: does the market misprice these? ===")
    print("(binary: mean resid on flag==1 with t-stat; continuous: corr)")
    for feat, is_flag, label in RESID_SPECS:
        x = pd.to_numeric(df[feat], errors="coerce")
        if is_flag:
            m = x == 1
            n = int(m.sum())
            if n < 50:
                print(f"  {feat:32} n={n:>7,}  (too few)")
                continue
            r = resid[m]
            t = r.mean() / (r.std() / np.sqrt(n)) if n > 1 else float("nan")
            print(f"  {feat:32} n={n:>7,}  mean_resid={r.mean():+.4f}  t={t:+.2f}"
                  f"   [{label}]")
        else:
            mm = x.notna() & resid.notna()
            c = float(np.corrcoef(x[mm], resid[mm])[0, 1]) if mm.sum() > 50 else float("nan")
            print(f"  {feat:32} n={int(mm.sum()):>7,}  corr={c:+.4f}")

    # WS runners: does the market price them differently at all? (sanity)
    ws = pd.to_numeric(df["wind_surgery_run"], errors="coerce") == 1
    print(f"\n  WS runners: win rate {df.loc[ws,'won'].mean()*100:.2f}% vs "
          f"field {df['won'].mean()*100:.2f}%; "
          f"mean mkt_imp {df.loc[ws,'mkt_imp'].mean():.4f} vs {df['mkt_imp'].mean():.4f}")

    # ---- (2) Fund+Market value picks: baseline vs +families ----
    tr = df[df["race_date"] < CUT]; te = df[df["race_date"] >= CUT].copy()
    print(f"\ntrain {len(tr):,} / test {len(te):,}")

    def run(feats, label):
        m = lgb.LGBMClassifier(**PLACE_PARAMS).fit(tr[feats], tr["won"])
        te["_p"] = m.predict_proba(te[feats])[:, 1]
        te["wp"] = te["_p"] / te.groupby("race_id")["_p"].transform("sum")
        te["edge"] = te["wp"] / te["mkt_imp"].clip(lower=1e-9) - 1.0
        band = pd.to_numeric(te["ltp_300s"], errors="coerce").between(1.5, 50)
        vp = te[(te["edge"] > 0) & band]
        print(f"  {label:26} value n={len(vp):,}  ROI@BSP {roi_bsp(vp):+.2f}%  "
              f"CLV {mean_clv(vp):.4f}")
        return m

    print("\n=== Fund+Market value picks (tripwire: ROI jump + flat CLV == leak) ===")
    run(form + mkt, "BASELINE")
    for fam, fcols in FAMILIES.items():
        run(form + mkt + fcols, f"+{fam}")
    mall = run(form + mkt + NEW, "+ALL THREE")

    # ---- (3) gain ranks in the all-in model ----
    feats_all = form + mkt + NEW
    imp = pd.Series(mall.booster_.feature_importance("gain"), index=feats_all)
    print("\n=== gain ranks of new features (of "
          f"{len(feats_all)} features) ===")
    for c in NEW:
        print(f"  {c:32} gain={imp[c]:>10.0f}  rank {int((imp > imp[c]).sum()) + 1}")


if __name__ == "__main__":
    main()
