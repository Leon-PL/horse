"""The Matchbook paper record of the 11:00 rule: forecast, paper bets, settlement and summary.

No bet is ever placed. Each day:

1. ``eleven``: the 11:00 Matchbook snapshot (and the latest one by 08:30 for the 08:00 move), the day's card
   forecast by the win model and the Matchbook 11:00 correction (``src.matchbook.model``), then the registered rule
   (research 107): ``value_bet_selection`` at base threshold 0.02 against the race-normalised 11:00 share, races off
   after 11:00 with every card runner linked and backable, best back price at most 51. The paper bet is taken at
   the best Matchbook back price at 11:00; a ``stake`` fill is walked down the three quoted levels.
2. ``settle``: Betfair's public daily BSP file (published the next day) gives the close and the result; the
   Matchbook close is the last stored quote before the off. CLV is taken / BSP.

   ``field.parquet`` keeps SP and result for every forecast runner, settled apart from the bets.

Files: ``data/matchbook/paper/<date>/{forecast,bets,links}.parquet``, ``run.json``, ``settled.parquet``,
``field.parquet``.
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.bet_settlement import settle_win_exchange_bets, value_bet_selection
from src.ingestion.storage import atomic_json, atomic_parquet
from src.matchbook import feed, link, record
from src.matchbook import model as mm
from src.utils import canonical_track_key, normalise_off_time_key

ROOT = record.ROOT
PAPER = ROOT / "paper"
CONFIG = ROOT / "config.json"
TIGHT = 0.8   # a race counts as tight when this share of its field is quoted within 25% back-to-lay
RULE = {"threshold": 0.02, "max_odds": 51.0, "stake": 10.0, "clock": "11:00"}
BSP_URL = "https://promo.betfair.com/betfairsp/prices/dwbfprices{region}win{ddmmyyyy}.csv"


def config():
    if not CONFIG.exists():
        raise SystemExit(f"No {CONFIG}: set the win model run and the correction folder (python -m src.matchbook configure)")
    return json.loads(CONFIG.read_text())


def day_dir(day):
    return PAPER / str(pd.Timestamp(day).date())


def fill(row, stake):
    """Average price and amount for ``stake`` walked down the three back levels."""
    left, paid, got = stake, 0., 0.
    for level in (1, 2, 3):
        odds, size = row[f"back{level}"], row[f"back{level}_size"]
        if not (odds > 1 and size > 0) or left <= 0:
            continue
        take = min(left, size)
        paid, got, left = paid + take, got + take * odds, left - take
    return (got / paid if paid else np.nan), paid


def _book_prices(card, book):
    """Matchbook quotes for linked card rows."""
    quotes = book.drop_duplicates("runner_id").set_index("runner_id")
    cols = ["back1", "back1_size", "back2", "back2_size", "back3", "back3_size", "lay1", "lay1_size", "withdrawn",
            "market_volume", "runner_volume"]
    out = quotes.reindex(card.runner_id.astype(object))[cols].reset_index(drop=True)
    return out


def prepare(day=None):
    """Everything that does not need the 11:00 price, done ahead so 11:00 itself takes seconds.

    Scrapes the card, featurises it, forecasts it with the win model and builds the correction's other
    inputs (cold caches after a store refresh made this take up to ~27 minutes). Saves
    ``prepared.parquet`` (keys, names, track, off time, the win-model forecast and those inputs).
    """
    from src.correction import serve
    from src.ingestion.pipeline import get_scraped_racecards
    from src.live_prediction import features_for_live_cards
    from src.run_store import load_run_predictor

    day = str(pd.Timestamp(day).date()) if day else str(record.uk_now().date())
    cfg = config()
    out = day_dir(day)
    started = record.uk_now()
    cards = get_scraped_racecards(date_str=day, force_refresh=True)
    if cards is None or cards.empty:
        atomic_json(out / "prepared.json", {"date": day, "status": "no racecards", "at": str(started)})
        return None
    featured = features_for_live_cards(cards, day)
    captured = {}

    def capture(part, values, columns):
        captured["values"], captured["columns"] = values.reset_index(drop=True).copy(), list(columns)
        blank = pd.DataFrame(np.nan, index=range(len(values)), columns=list(mm.PRICE_INPUTS), dtype="float32")
        return pd.concat([values.reset_index(drop=True), blank], axis=1), [*columns, *mm.PRICE_INPUTS]

    forecast, lineage = serve.forecast(featured, load_run_predictor(cfg["win_run"]), mm.load(cfg["correction"]),
                                       extra=capture)
    keys = ["race_id", "horse_id"]
    values = captured["values"]
    context = cards[[*keys, "track", "off_time"]].astype({k: forecast[k].dtype for k in keys})
    frame = forecast[[*keys, "race_date", "p_base", "horse_name"]].merge(context, on=keys, how="left",
                                                                         validate="one_to_one")
    frame = frame.merge(values.astype({k: frame[k].dtype for k in keys}), on=keys, how="left", validate="one_to_one")
    atomic_parquet(frame, out / "prepared.parquet")
    atomic_json(out / "prepared.json", {"date": day, "status": "prepared", "columns": captured["columns"],
                                        "card_races": int(frame.race_id.nunique()), "runners": int(len(frame)),
                                        "started": str(started), "finished": str(record.uk_now()), **lineage})
    return frame


def eleven(day=None, *, snapshot=True):
    """The 11:00 forecast and paper bets for ``day`` (UK date, default today); prepares first if needed."""
    from src.correction import model as cm

    day = str(pd.Timestamp(day).date()) if day else str(record.uk_now().date())
    cfg = config()
    out = day_dir(day)
    if not (out / "prepared.parquet").exists() and prepare(day) is None:
        atomic_json(out / "run.json", {"date": day, "status": "no racecards"})
        return None
    prep = pd.read_parquet(out / "prepared.parquet")
    meta = json.loads((out / "prepared.json").read_text())
    if snapshot:
        path, book = record.snapshot("1100")
    else:
        book = record.at(day, "11:10")
        if not book.empty and book.taken_at.iloc[0].tz_convert(feed.LONDON).strftime("%H:%M") < "10:55":
            book = book.iloc[:0]
    book = book.loc[book.race_date.eq(day)].reset_index(drop=True) if not book.empty else book
    early = record.at(day, "08:30")
    if book.empty:
        atomic_json(out / "run.json", {"date": day, "status": "no 11:00 Matchbook snapshot or races"})
        return None
    taken_at = str(book.taken_at.iloc[0].tz_convert(feed.LONDON))

    linked, report = link.link(prep[["race_id", "horse_id", "horse_name", "track", "off_time"]], book)
    withdrawn = linked.runner_id.map(book.drop_duplicates("runner_id").set_index("runner_id").withdrawn)
    keep = ~withdrawn.fillna(False).astype(bool).to_numpy()
    dropped = int((~keep).sum())          # withdrawn on Matchbook since the card was prepared
    prep, linked = prep.loc[keep].reset_index(drop=True), linked.loc[keep].reset_index(drop=True)
    quotes = _book_prices(linked, book)
    price_1100 = feed.reference_price(quotes.back1, quotes.lay1)
    early_price = np.full(len(linked), np.nan)
    if not early.empty:
        eq = early.drop_duplicates("runner_id").set_index("runner_id").reindex(linked.runner_id.astype(object))
        early_price = feed.reference_price(eq.back1.to_numpy(), eq.lay1.to_numpy())
    off_after = linked.off_time.map(normalise_off_time_key).gt(RULE["clock"]).to_numpy()
    price_1100 = np.where(off_after, price_1100, np.nan)

    model = mm.load(cfg["correction"])
    columns = meta["columns"]
    if model.columns != [*columns, *mm.PRICE_INPUTS]:
        raise ValueError("The prepared card was built for a different correction")
    values = pd.concat([prep[columns].reset_index(drop=True),
                        mm.price_inputs(prep.race_id, prep.p_base, price_1100, early_price)], axis=1)
    adjustment = model.adjustment(values)
    keys = ["race_id", "horse_id"]
    frame = prep[[*keys, "race_date", "p_base", "horse_name", "track", "off_time"]].copy()
    frame["adjustment"] = adjustment
    frame["probability"] = cm.corrected(frame.p_base.to_numpy(float), adjustment, frame.race_id.to_numpy())
    frame[["event_id", "runner_id"]] = linked[["event_id", "runner_id"]].to_numpy()
    frame = pd.concat([frame, quotes.reset_index(drop=True)], axis=1)
    frame["price_1100"], frame["price_0800"] = price_1100, early_price
    lineage = {k: meta.get(k) for k in ("history_last_date", "store_last_date", "correction_as_of", "win_model_as_of")}
    cards = prep
    race = frame.race_id.to_numpy()
    frame["p"] = frame.probability / frame.probability.groupby(race).transform("sum")
    # A race off before the forecast is ready could not have been bet at 11:00 prices.
    ready = max(RULE["clock"], (record.uk_now() + pd.Timedelta(minutes=2)).strftime("%H:%M"))
    whole = (frame.runner_id.notna() & frame.back1.gt(1) & frame.price_1100.gt(1)).groupby(race).transform("all")
    whole &= frame.off_time.map(normalise_off_time_key).gt(ready)
    frame["eligible_race"] = whole.to_numpy()
    # Share of the field quoted within 25% back-to-lay; thin books make the 11:00 share unreliable.
    tight = (frame.lay1 <= frame.back1 * 1.25) & frame.back1.gt(1)
    frame["race_tight"] = tight.groupby(race).transform("mean").to_numpy()
    inverse = (1 / frame.price_1100).where(frame.eligible_race)
    frame["share"] = inverse / inverse.groupby(race).transform("sum")
    taken = frame.back1.where(frame.eligible_race).to_numpy(float)
    band = (taken > 1) & (taken <= RULE["max_odds"])
    pick = value_bet_selection(frame.p.fillna(0).to_numpy(), np.where(band, taken, 2.), RULE["threshold"],
                               implied_prob=frame.share.fillna(1).to_numpy())["mask"] & band
    frame["bet"] = pick
    frame["edge"] = frame.p - frame.share
    filled = [fill(r, RULE["stake"]) if b else (np.nan, 0.) for b, (_, r) in zip(pick, frame.iterrows())]
    frame["fill_price"] = [f[0] for f in filled]
    frame["fill_stake"] = [f[1] for f in filled]
    frame["race_date"] = day
    frame["snapshot_1100"] = taken_at
    atomic_parquet(frame, out / "forecast.parquet")
    bets = frame.loc[frame.bet].reset_index(drop=True)
    atomic_parquet(bets, out / "bets.parquet")
    atomic_parquet(report, out / "links.parquet")
    run = {"date": day, "status": "forecast", "snapshot_1100": taken_at, "bettable_after": ready,
           "snapshot_0800": str(early.taken_at.iloc[0].tz_convert(feed.LONDON)) if not early.empty else None,
           "card_races": int(cards.race_id.nunique()), "matchbook_races": int(book.event_id.nunique()),
           "prepared_at": meta.get("finished"), "withdrawn_since_prepared": dropped,
           "races_linked": int((report.status == "matched").sum()) if not report.empty else 0,
           "eligible_races": int(frame.loc[frame.eligible_race].race_id.nunique()), "bets": int(len(bets)),
           "tight_eligible_races": int(frame.loc[frame.eligible_race & frame.race_tight.ge(TIGHT)].race_id.nunique()),
           "rule": RULE, "win_run": cfg["win_run"], "correction": cfg["correction"], **lineage,
           "made_at": str(record.uk_now())}
    atomic_json(out / "run.json", run)
    return run


# --------------------------------------------------------------------------
# Settlement


def betfair_day(day):
    """Betfair's public BSP rows for races on ``day`` (UK and Irish win markets), or empty if not yet published."""
    stamp = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%d%m%Y")
    frames = []
    for region in ("uk", "ire"):
        response = requests.get(BSP_URL.format(region=region, ddmmyyyy=stamp), timeout=30,
                                headers={"User-Agent": feed.USER_AGENT})
        if response.status_code == 404 or not response.content.lstrip().lower().startswith(b"event_id"):
            continue
        response.raise_for_status()
        f = pd.read_csv(io.BytesIO(response.content), dtype=str)
        f.columns = [c.strip().lower() for c in f.columns]
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    f = pd.concat(frames, ignore_index=True)
    when = pd.to_datetime(f.event_dt.str.strip(), format="%d-%m-%Y %H:%M", errors="coerce")
    f = f.loc[when.dt.date.astype(str).eq(str(pd.Timestamp(day).date()))].copy()
    f["off_time"] = when.dt.strftime("%H:%M")
    f["track"] = f.menu_hint.str.replace(r"^\s*[A-Z]{2,3}\s*/\s*", "", regex=True).str.replace(
        r"\s+\d{1,2}(st|nd|rd|th)\s+\w{3}\s*$", "", regex=True).str.strip()
    f["horse_name"] = f.selection_name.str.replace(r"^\s*\d+\.\s*", "", regex=True).str.strip()
    f["won"] = pd.to_numeric(f.win_lose, errors="coerce")
    for column in ("bsp", "ppwap", "morningwap", "morningtradedvol", "pptradedvol"):
        f[column] = pd.to_numeric(f[column], errors="coerce")
    return f.reset_index(drop=True)


def settle(day):
    """Settle ``day``'s paper bets at Betfair SP results; returns the settled frame or None if not yet possible."""
    out = day_dir(day)
    if not (out / "bets.parquet").exists():
        return None
    bets = pd.read_parquet(out / "bets.parquet")
    bf = betfair_day(day)
    if bf.empty:
        return None
    rows = bets.rename(columns={"runner_id": "mb_runner_id", "event_id": "mb_event_id"})
    bf_card = bf.rename(columns={"selection_id": "runner_id", "event_id": "bf_event_id"})
    bf_card["event_id"] = bf_card.track.map(canonical_track_key) + "|" + bf_card.off_time
    linked, _ = link.link(rows[["race_id", "horse_id", "horse_name", "track", "off_time"]], bf_card)
    lookup = bf_card.set_index("runner_id")
    rows["bsp"] = linked.runner_id.map(lookup.bsp).to_numpy()
    rows["won"] = linked.runner_id.map(lookup.won).to_numpy()
    rows["bf_morning_volume"] = linked.runner_id.map(lookup.morningtradedvol).to_numpy()
    closes = record.closes(day)
    if not closes.empty:
        last = closes.set_index("runner_id")
        rows["mb_close_back"] = rows.mb_runner_id.map(last.back1)
        rows["mb_close_lay"] = rows.mb_runner_id.map(last.lay1)
        rows["mb_close_at"] = rows.mb_runner_id.map(last.taken_at)
        rows["mb_withdrawn"] = rows.mb_runner_id.map(last.withdrawn)
    ran = rows.bsp.gt(1) & rows.won.notna()
    rows["status"] = np.where(ran, "settled", "void or unmatched")
    rows["pnl_unit"] = np.where(ran, settle_win_exchange_bets(rows.back1.fillna(2), rows.won.fillna(0)), 0.)
    rows["pnl_fill"] = np.where(ran & rows.fill_stake.gt(0),
                                settle_win_exchange_bets(rows.fill_price.fillna(2), rows.won.fillna(0), 1.)
                                * rows.fill_stake, 0.)
    rows["clv"] = np.where(ran, rows.back1 / rows.bsp, np.nan)
    atomic_parquet(rows, out / "settled.parquet")
    run = json.loads((out / "run.json").read_text())
    run.update({"status": "settled", "settled_bets": int(ran.sum()), "settled_at": str(record.uk_now())})
    # The whole field is settled apart from the bets, so a fault there can never change a bet's settlement.
    try:
        field = settle_field(day, bf_card, closes)
        run["settled_field_runners"] = int(field.bsp.gt(1).sum()) if field is not None else 0
    except Exception as exc:   # noqa: BLE001 - recorded, and the bets above stand
        run["field_error"] = repr(exc)
    atomic_json(out / "run.json", run)
    return rows


FIELD_COLUMNS = ("race_id", "horse_id", "race_date", "track", "off_time", "horse_name", "p_base", "p", "share", "edge",
                 "back1", "back1_size", "lay1", "price_0800", "price_1100", "eligible_race", "race_tight", "bet",
                 "runner_id")


def settle_field(day, bf_card, closes=None):
    """Betfair SP, result and Matchbook close for every forecast runner of ``day`` (``field.parquet``).

    The bets are only a few runners a day; the whole field gives the forecast's daily log loss and its edge against
    the market on every runner, which the summary reads as ``field``.
    """
    out = day_dir(day)
    if not (out / "forecast.parquet").exists():
        return None
    forecast = pd.read_parquet(out / "forecast.parquet")
    rows = forecast[[c for c in FIELD_COLUMNS if c in forecast]].rename(columns={"runner_id": "mb_runner_id"})
    linked, _ = link.link(rows[["race_id", "horse_id", "horse_name", "track", "off_time"]], bf_card)
    lookup = bf_card.drop_duplicates("runner_id").set_index("runner_id")
    rows["bsp"] = linked.runner_id.map(lookup.bsp).to_numpy()
    rows["won"] = linked.runner_id.map(lookup.won).to_numpy()
    rows["bf_morning_volume"] = linked.runner_id.map(lookup.morningtradedvol).to_numpy()
    if closes is not None and not closes.empty:
        last = closes.drop_duplicates("runner_id", keep="last").set_index("runner_id")
        rows["mb_close_back"] = rows.mb_runner_id.map(last.back1)
        rows["mb_close_lay"] = rows.mb_runner_id.map(last.lay1)
    atomic_parquet(rows, out / "field.parquet")
    return rows


def _race_log_loss(won, prob, race):
    """Mean over races of −log(probability given to the winner); ``prob`` normalised within each race."""
    p = pd.Series(np.asarray(prob, dtype=float))
    race = pd.Series(np.asarray(race))
    p = p / p.groupby(race).transform("sum")
    winners = np.asarray(won, dtype=float) == 1
    return float(-np.log(p[winners].clip(lower=1e-9)).mean())


def field_summary():
    """The forecast on every runner of eligible, fully settled races: log loss against both markets and its edge.

    ``log_loss`` compares the win model, the corrected forecast, the 11:00 Matchbook share and the Betfair SP book
    on the same races (lower is better). ``edge_result_corr`` is the correlation of (forecast − 11:00 share) with
    (won − SP-implied probability): above zero means the forecast knows something the close does not. ``clv_by_edge``
    reads taken / BSP for runners the forecast rated above and below the 11:00 share.
    """
    from src.clv import summarise_clv
    frames = [pd.read_parquet(p) for p in sorted(PAPER.glob("*/field.parquet"))]
    if not frames:
        return {"races": 0}
    f = pd.concat(frames, ignore_index=True)
    f = f.loc[f.eligible_race.astype(bool)] if "eligible_race" in f else f
    race = f.race_id
    complete = (f.bsp.gt(1) & f.won.notna()).groupby(race).transform("all")
    one_winner = f.won.eq(1).groupby(race).transform("sum").eq(1)
    f = f.loc[complete & one_winner].reset_index(drop=True)
    if f.empty:
        return {"races": 0}
    race = f.race_id
    market = (1 / f.bsp) / (1 / f.bsp).groupby(race).transform("sum")
    model = f.p_base / f.p_base.groupby(race).transform("sum")
    losses = {"win_model": _race_log_loss(f.won, model, race), "forecast": _race_log_loss(f.won, f.p, race),
              "matchbook_1100": _race_log_loss(f.won, f.share, race), "betfair_sp": _race_log_loss(f.won, market, race)}
    clv = f.back1 / f.bsp
    edge = f.p - f.share
    by_day = []
    for d, g in f.groupby("race_date"):
        r = g.race_id
        m = (1 / g.bsp) / (1 / g.bsp).groupby(r).transform("sum")
        by_day.append({"date": str(d), "races": int(r.nunique()),
                       "forecast": _race_log_loss(g.won, g.p, r), "betfair_sp": _race_log_loss(g.won, m, r)})
    return {"days": int(f.race_date.nunique()), "races": int(race.nunique()), "runners": int(len(f)),
            "log_loss": losses, "forecast_minus_sp": losses["forecast"] - losses["betfair_sp"],
            "edge_result_corr": float(np.corrcoef(edge, f.won - market)[0, 1]) if len(f) > 2 else None,
            "clv_by_edge": {"forecast_above_share": summarise_clv(clv[edge > 0]),
                            "forecast_below_share": summarise_clv(clv[edge <= 0]),
                            "bets": summarise_clv(clv[f.bet.astype(bool)])},
            "by_day": by_day}


def unsettled_days():
    return sorted(p.name for p in PAPER.glob("*") if (p / "bets.parquet").exists() and not (p / "settled.parquet").exists())


def summary(boot=2000, seed=0):
    """Every settled paper bet so far: CLV, ROI and fills."""
    from src.clv import summarise_clv
    from src.wf_eval import bootstrap_roi_ci
    frames = [pd.read_parquet(p) for p in sorted(PAPER.glob("*/settled.parquet"))]
    days = sorted(p.parent.name for p in PAPER.glob("*/run.json"))
    field = field_summary()
    if not frames:
        return {"days_recorded": len(days), "settled_bets": 0, "field": field}
    f = pd.concat(frames, ignore_index=True)
    f = f.loc[f.status.eq("settled")].reset_index(drop=True)
    if f.empty:
        return {"days_recorded": len(days), "settled_bets": 0, "field": field}
    clv = summarise_clv(f.clv.to_numpy())
    log = np.log(f.clv.to_numpy())
    by_day = pd.DataFrame({"l": log, "d": f.race_date}).groupby("d").l.agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    draws = [np.exp(by_day["sum"].to_numpy()[i].sum() / by_day["size"].to_numpy()[i].sum())
             for i in (rng.integers(0, len(by_day), len(by_day)) for _ in range(boot))]
    roi = bootstrap_roi_ci(f.pnl_unit.to_numpy(), clusters=f.race_id.to_numpy()) if len(f) > 1 else None
    filled = f.fill_stake.sum()
    tight = f.loc[f.race_tight.ge(TIGHT)] if "race_tight" in f else f.iloc[:0]
    tight_clv = float(np.exp(np.log(tight.clv).mean())) if len(tight) else None
    return {"tight_races": {"threshold": TIGHT, "bets": int(len(tight)), "geometric_mean_clv": tight_clv,
                            "pnl_unit": float(tight.pnl_unit.sum()) if len(tight) else 0.},
            "days_recorded": len(days), "settled_days": int(f.race_date.nunique()), "settled_bets": int(len(f)),
            "wins": int(f.won.sum()), "geometric_mean_clv": float(np.exp(log.mean())),
            "geo_clv_ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "median_clv": clv["median_clv"], "pct_beat_close": clv["pct_beat_close"],
            "roi_unit": roi, "pnl_unit_stakes": float(f.pnl_unit.sum()),
            "stake_target": RULE["stake"], "stake_filled": float(filled),
            "fill_rate": float(filled / (RULE["stake"] * len(f))), "pnl_filled": float(f.pnl_fill.sum()),
            "median_back_size": float(f.back1_size.median()), "field": field}
