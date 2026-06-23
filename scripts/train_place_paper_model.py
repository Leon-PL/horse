"""Train + save the form + pre-off win-market PLACE model used for paper trading.

Reuses the walk-forward-validated setup (form allowlist features + 6 win-market
features built by src.paper_trading.build_market_features; target = placed i.e.
finish <= Betfair numberOfWinners; trained on matched rows only). Win-market
features come from the BF pre-off win price (ltp_300s) at train time and the live
MB win price at inference — same recipe. Saves model + exact feature order.
"""
import os, joblib, numpy as np, pandas as pd, config, pyarrow.parquet as pq
import lightgbm as lgb, scripts.model_lab as ml
from src.feature_store import FeatureStore
from src.model import get_feature_columns
from src.betfair import attach_bsp, bf_track_key, bf_horse_key
from src.utils import normalise_off_time_key
from src.paper_trading import build_market_features, MARKET_FEATS

OUT_DIR = os.path.join(config.PROCESSED_DATA_DIR, "place_paper_model")
os.makedirs(OUT_DIR, exist_ok=True)

s = FeatureStore(); schema = set(pq.read_schema(s.featured_path).names)
allow = [ln.strip() for ln in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
         if ln.strip() and not ln.lstrip().startswith("#")]
DIAG = ["race_id", "race_date", "off_time", "track", "horse_name", "finish_position", "opening_odds"]
cols = [c for c in sorted(set(allow) | set(DIAG)) if c in schema]
df = pd.read_parquet(s.featured_path, columns=cols, engine="pyarrow")
df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
# win pre-off price (BF ltp_300s) for the market features
df = attach_bsp(df, cols=("ltp_300s",)); df["win_ltp300_raw"] = pd.to_numeric(df["ltp_300s"], errors="coerce")
# place places-paid (Betfair numberOfWinners) for the target
pb = pd.read_parquet("data/processed/betfair_place_bsp.parquet",
                     columns=["race_date", "track", "off_key", "horse_key", "bf_places_paid"])
d10 = lambda x: pd.to_datetime(x, errors="coerce").dt.strftime("%Y-%m-%d")
L = pd.DataFrame({"race_date": d10(df["race_date"]), "track_key": df["track"].map(bf_track_key),
                  "off_key": df["off_time"].map(normalise_off_time_key), "horse_key": df["horse_name"].map(bf_horse_key)})
pb["race_date"] = d10(pb["race_date"]); pb["track_key"] = pb["track"].map(bf_track_key)
R = pb[["race_date", "track_key", "off_key", "horse_key", "bf_places_paid"]].drop_duplicates(
    ["race_date", "track_key", "off_key", "horse_key"])
df["bf_places_paid"] = pd.to_numeric(L.merge(R, on=["race_date", "track_key", "off_key", "horse_key"], how="left")["bf_places_paid"].to_numpy(), errors="coerce")

# market features (race_id present in store)
mf = build_market_features(df, "win_ltp300_raw", "opening_odds")
for c in MARKET_FEATS:
    df[c] = mf[c].to_numpy()
form = [c for c in get_feature_columns(df) if c in df.columns]
feature_cols = form + MARKET_FEATS

train = df[df["bf_places_paid"].notna()].copy()
fp = pd.to_numeric(train["finish_position"], errors="coerce")
y = ((fp > 0) & (fp <= train["bf_places_paid"])).astype(int)
print(f"training rows {len(train):,}  features {len(feature_cols)} (form {len(form)} + market {len(MARKET_FEATS)})", flush=True)
model = lgb.LGBMClassifier(**ml.BASE_PARAMS).fit(train[feature_cols], y)

joblib.dump(model, os.path.join(OUT_DIR, "model.joblib"))
joblib.dump({"feature_cols": feature_cols, "form": form, "market": MARKET_FEATS,
             "trained_rows": int(len(train)), "trained_at": pd.Timestamp.now().isoformat()},
            os.path.join(OUT_DIR, "meta.joblib"))
print(f"saved -> {OUT_DIR}  (model.joblib, meta.joblib)", flush=True)
