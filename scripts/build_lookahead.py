"""Build the lookahead feature cache for explicit dates (incl. today).

Uses the same validated, bounded feature build as the picks app. Dates must
follow the store's settled history; changed cards/history invalidate the cache.

  python scripts/build_lookahead.py                 # today + tomorrow
  python scripts/build_lookahead.py --date 2026-06-23
  python scripts/build_lookahead.py --days-ahead 3  # today .. +3
"""
import argparse, os, sys, logging
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingestion.pipeline import get_scraped_racecards
from src.feature_store import FeatureStore
from src.live_prediction import features_for_live_cards

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_lookahead")


def build_for_date(store: FeatureStore, date_str: str) -> int:
    cards = get_scraped_racecards(date_str=date_str)
    if cards is None or cards.empty:
        log.info("No racecards for %s", date_str)
        return 0
    featured = features_for_live_cards(cards, date_str, store=store)
    log.info("Cached %d featured runners for %s", len(featured), date_str)
    return len(featured)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="single date YYYY-MM-DD")
    ap.add_argument("--days-ahead", type=int, default=1, help="today .. today+N (ignored if --date)")
    args = ap.parse_args()

    if args.date:
        dates = [args.date]
    else:
        today = date.today()
        dates = [(today + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(0, args.days_ahead + 1)]

    store = FeatureStore()
    if not store.exists():
        raise SystemExit("FeatureStore is empty; build it first with "
                         "`python -m scripts.build_feature_store_batched`.")
    total = {d: build_for_date(store, d) for d in dates}
    print("Built lookahead cache:", total)


if __name__ == "__main__":
    main()
