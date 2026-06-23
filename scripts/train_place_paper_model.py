"""Train + save the form (+ pre-off win-market) PLACE model for paper trading.

Thin wrapper over src.place_market_model (the shared, validated training path
also used by the training app). Trains on all matched rows and saves the model +
exact feature order to data/processed/place_paper_model/ for scripts/paper_trade.py.

  python scripts/train_place_paper_model.py                 # form + win-market (default)
  python scripts/train_place_paper_model.py --form-only     # odds-free form model
"""
import argparse, os, sys, joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, pyarrow.parquet as pq
import config
from src.feature_store import FeatureStore
from src.place_market_model import train_and_evaluate

OUT_DIR = os.path.join(config.PROCESSED_DATA_DIR, "place_paper_model")
os.makedirs(OUT_DIR, exist_ok=True)
# Project to allowlist + the diag/key/eval columns to stay memory-bounded (the
# store is ~1100 cols); get_feature_columns then selects the form features.
DIAG = ["race_id", "race_date", "off_time", "track", "horse_name",
        "finish_position", "opening_odds", "race_type", "num_runners"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form-only", action="store_true", help="odds-free form model (no win-market features)")
    ap.add_argument("--ev", type=float, default=0.10)
    args = ap.parse_args()

    s = FeatureStore()
    schema = set(pq.read_schema(s.featured_path).names)
    allow = [ln.strip() for ln in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
             if ln.strip() and not ln.lstrip().startswith("#")]
    cols = [c for c in sorted(set(allow) | set(DIAG)) if c in schema]
    featured = pd.read_parquet(s.featured_path, columns=cols, engine="pyarrow")
    model, meta, evals = train_and_evaluate(
        featured, use_market_features=not args.form_only, ev_threshold=args.ev)

    joblib.dump(model, os.path.join(OUT_DIR, "model.joblib"))
    joblib.dump(meta, os.path.join(OUT_DIR, "meta.joblib"))
    print(f"held-out eval: {evals}", flush=True)
    print(f"saved -> {OUT_DIR}  (model.joblib, meta.joblib)  "
          f"[{len(meta['feature_cols'])} feats, {meta['trained_rows']:,} rows]", flush=True)


if __name__ == "__main__":
    main()
