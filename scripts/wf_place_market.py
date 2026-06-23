"""Walk-forward the place-market strategy on the feature store.

Expanding quarterly folds, +EV @ Betfair place BSP, 2% commission. Reports the
pooled across-time ROI + bootstrap CI and per-fold ROIs — the robust read.

  python scripts/wf_place_market.py                # form + win-market
  python scripts/wf_place_market.py --form-only
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, pyarrow.parquet as pq
import config
from src.feature_store import FeatureStore
from src.place_market_model import walk_forward_place_market

DIAG = ["race_id", "race_date", "off_time", "track", "horse_name",
        "finish_position", "opening_odds", "race_type", "num_runners"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form-only", action="store_true")
    ap.add_argument("--ev", type=float, default=0.10)
    ap.add_argument("--min-train-quarters", type=int, default=8)
    args = ap.parse_args()

    s = FeatureStore()
    schema = set(pq.read_schema(s.featured_path).names)
    allow = [ln.strip() for ln in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
             if ln.strip() and not ln.lstrip().startswith("#")]
    cols = [c for c in sorted(set(allow) | set(DIAG)) if c in schema]
    featured = pd.read_parquet(s.featured_path, columns=cols, engine="pyarrow")

    kw = dict(use_market_features=not args.form_only, ev_threshold=args.ev,
              min_train_quarters=args.min_train_quarters)
    overall = walk_forward_place_market(featured, **kw)
    jumps = walk_forward_place_market(featured, jumps_only=True, **kw)

    def _summ(tag, r):
        print(f"\n=== {tag} ===")
        print(f"folds={r['n_folds']} ({r['folds_positive']} positive)  bets={r['n_bets']:,}  "
              f"ROI={r['roi_pct']}%  CI[{r['ci_low']}, {r['ci_high']}]  pnl={r['total_pnl']}")
        for f in r.get("per_fold", []):
            print(f"   {f['quarter']}: {f['n_bets']:5} bets  {f['roi_pct']:+.2f}%")

    _summ(f"ALL  ({'form+market' if not args.form_only else 'form-only'})", overall)
    _summ("JUMPS", jumps)
    print("\nJSON:", json.dumps({"all": overall, "jumps": jumps}))


if __name__ == "__main__":
    main()
