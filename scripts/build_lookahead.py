"""Build the lookahead feature cache for explicit dates (incl. today).

The picks-app's build_lookahead_cache only covers tomorrow onward (offset 1+);
paper trading needs *today's* races too. This CLI builds the same cache the
app/paper-trader reads (src.live_prediction lookahead_cache), for any dates.

  python scripts/build_lookahead.py                 # today + tomorrow
  python scripts/build_lookahead.py --date 2026-06-23
  python scripts/build_lookahead.py --days-ahead 3  # today .. +3
"""
import argparse, os, sys, logging
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import config
from src.data_collector import get_scraped_racecards
from src.data_processor import process_data
from src.live_prediction import (
    feature_engineer_with_history_core, save_lookahead_cache, cards_signature,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_lookahead")


def _load_history() -> pd.DataFrame:
    pq = os.path.join(config.PROCESSED_DATA_DIR, "processed_races.parquet")
    csv = os.path.join(config.PROCESSED_DATA_DIR, "processed_races.csv")
    path = pq if os.path.exists(pq) else csv
    if not os.path.exists(path):
        raise SystemExit(f"No processed history at {pq} / {csv} — run the pipeline first.")
    log.info("Loading processed history: %s", path)
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def build_for_date(history: pd.DataFrame, date_str: str) -> int:
    cards = get_scraped_racecards(date_str=date_str)
    if cards is None or cards.empty:
        log.info("No racecards for %s", date_str)
        return 0
    cards = cards.copy()
    cards["won"] = 0
    cards["finish_position"] = 0
    cards["finish_time_secs"] = 0.0
    cards["lengths_behind"] = float("nan")
    proc = process_data(df=cards, save=False)
    featured = feature_engineer_with_history_core(history, proc)
    save_lookahead_cache(date_str, featured, cards_signature(cards))
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

    history = _load_history()
    total = {d: build_for_date(history, d) for d in dates}
    print("Built lookahead cache:", total)


if __name__ == "__main__":
    main()
