"""Build the lookahead feature cache for explicit dates (incl. today).

Uses the validated, bounded live feature build in ``src.live_prediction``. Dates
must follow the store's settled history; changed cards/history invalidate the cache.

Before building, date-stamped live-feature, lookahead and racecard cache entries
older than ``config.CACHE_TTL_DAYS`` are deleted (``src.utils.cleanup_stale_caches``,
which the removed picks app ran on start-up). Past racecards cannot always be
re-scraped, so pass ``--no-cleanup`` to keep them.

  python scripts/build_lookahead.py                 # today + tomorrow
  python scripts/build_lookahead.py --date 2026-06-23
  python scripts/build_lookahead.py --days-ahead 3  # today .. +3
  python scripts/build_lookahead.py --no-cleanup    # keep old cache entries
"""
import argparse, os, sys, logging
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingestion.pipeline import get_scraped_racecards
from src.feature_store import FeatureStore
from src.live_prediction import features_for_live_cards
from src.utils import cleanup_stale_caches

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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="single date YYYY-MM-DD")
    ap.add_argument("--days-ahead", type=int, default=1, help="today .. today+N (ignored if --date)")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="keep date-stamped cache entries older than config.CACHE_TTL_DAYS")
    args = ap.parse_args(argv)

    if args.date:
        dates = [args.date]
    else:
        today = date.today()
        dates = [(today + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(0, args.days_ahead + 1)]

    # Housekeeping runs before the build, so nothing written below is removed,
    # and never blocks it.
    if not args.no_cleanup:
        try:
            cleanup_stale_caches()
        except Exception as exc:
            log.warning("Stale-cache cleanup failed: %s", exc)

    store = FeatureStore()
    if not store.exists():
        raise SystemExit("FeatureStore is empty; build it first with "
                         "`python -m scripts.build_feature_store_batched`.")
    total = {d: build_for_date(store, d) for d in dates}
    print("Built lookahead cache:", total)


if __name__ == "__main__":
    main()
