"""Place-market paper trading CLI (Matchbook live odds).

  # Generate today's selections (logs to data/processed/paper_trades.csv):
  python scripts/paper_trade.py --generate --hours-ahead 24
  # See what MB market names look like (confirm the PLACE market name live):
  python scripts/paper_trade.py --debug-markets
  # Settle finished races against scraped results:
  python scripts/paper_trade.py --settle --date 2026-06-23

Selections need today's form-featured runners — built by the picks app's
lookahead cache (src.live_prediction.build_lookahead_cache). This script loads
that cache; build it first if missing.
"""
import argparse, os, sys, logging
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.matchbook_client import MatchbookClient
from src.matchbook_signals import event_track_and_off_time, event_date_str, _best_price
from src.live_prediction import load_lookahead_cache
from src.paper_trading import score_and_select, log_selections, settle, DEFAULT_LEDGER
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("paper_trade")
MODEL_DIR = os.path.join(config.PROCESSED_DATA_DIR, "place_paper_model")


def _markets_for_event(client, event):
    mk = event.get("markets")
    if mk:
        return mk
    return client.get_event_markets(event["id"]).get("markets", [])


def _find_market(markets, *names):
    for m in markets:
        if str(m.get("name") or "").upper() in names:
            return m
    return None


def matchbook_prices(client, *, hours_ahead=24, debug=False) -> pd.DataFrame:
    """Per-runner live MB win + place prices for upcoming GB/IE races."""
    resp = client.get_horse_racing_events(hours_ahead=hours_ahead, include_prices=True, per_page=50)
    events = resp.get("events", []) if isinstance(resp, dict) else []
    rows, seen_names = [], set()
    for ev in events:
        markets = _markets_for_event(client, ev)
        if debug:
            for m in markets:
                seen_names.add(str(m.get("name")))
            continue
        track, off = event_track_and_off_time(ev)
        date = event_date_str(ev)
        win = _find_market(markets, "WIN")
        place = next((m for m in markets if "PLACE" in str(m.get("name") or "").upper()), None)
        if win is None or place is None:
            continue
        places_paid = place.get("number-of-winners") or place.get("places")
        win_px = {str(r.get("name")): _best_price(r.get("prices") or [], {"back", "win"})[0]
                  for r in win.get("runners") or []}
        for r in place.get("runners") or []:
            name = str(r.get("name"))
            pb, pa = _best_price(r.get("prices") or [], {"back", "win"})
            if pb is None:
                continue
            rows.append({
                "race_date": date, "track": track, "off_time": off, "horse_name": name,
                "mb_win_odds": win_px.get(name), "mb_place_odds": pb, "mb_place_available": pa,
                "places_paid": places_paid,
            })
    if debug:
        print("MB market names seen:", sorted(seen_names))
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    log.info("MB prices: %d runners across %d races", len(df), df.groupby(["track", "off_time"]).ngroups if not df.empty else 0)
    return df


def generate(args):
    client = MatchbookClient()
    client.login(config.MATCHBOOK_USERNAME, config.MATCHBOOK_PASSWORD)
    mb = matchbook_prices(client, hours_ahead=args.hours_ahead, debug=args.debug_markets)
    if args.debug_markets:
        return
    if mb.empty:
        log.warning("No MB win+place prices found."); return
    # places_paid fallback if MB didn't expose it
    if mb["places_paid"].isna().all():
        log.warning("MB did not expose places-paid; falling back to field-size formula at settle.")
    date = sorted(mb["race_date"].dropna().unique())[0] if mb["race_date"].notna().any() else None
    featured = load_lookahead_cache(date) if date else None
    if featured is None or featured.empty:
        log.error("No lookahead featured cache for %s — build it in the picks app first.", date); return
    model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    meta = joblib.load(os.path.join(MODEL_DIR, "meta.joblib"))
    sel = score_and_select(featured, mb, model, meta["feature_cols"],
                           ev_threshold=args.ev, stake=args.stake, jumps_only=args.jumps_only)
    if sel.empty:
        log.info("No +EV selections (ev>%.2f).", args.ev); return
    added = log_selections(sel, args.ledger)
    print(f"\n{len(sel)} selections (ev>{args.ev}, {added} new to ledger):")
    cols = ["off_time", "track", "horse_name", "p_place", "mb_place_odds", "mb_place_available", "ev"]
    print(sel[[c for c in cols if c in sel.columns]].to_string(index=False))


def do_settle(args):
    from src.data_collector import scrape_todays_results
    res = scrape_todays_results(date_str=args.date)
    if res is None or res.empty:
        log.warning("No results for %s.", args.date); return
    summary = settle(res, ledger_path=args.ledger)
    print("Settlement:", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--debug-markets", action="store_true")
    ap.add_argument("--date", default=None)
    ap.add_argument("--hours-ahead", type=int, default=24)
    ap.add_argument("--ev", type=float, default=0.10)
    ap.add_argument("--stake", type=float, default=1.0)
    ap.add_argument("--jumps-only", action="store_true")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    args = ap.parse_args()
    if args.generate or args.debug_markets:
        generate(args)
    elif args.settle:
        do_settle(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
