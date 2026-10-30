"""Observe current public feeds and issue separate independent/market forecasts."""
from __future__ import annotations

from datetime import timedelta
import io
import json
import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.betfair import bf_track_key
from src.forward_store import numeric_id, publish_json, timestamp, utcnow
from src.matchbook_signals import clean_runner_name, event_track_and_off_time
from src.utils import normalise_horse_key, normalise_off_time_key


def uk_ire_event(event):
    return any(tag.get("type") == "COUNTRY" and tag.get("name") in
               {"UK", "Ireland", "United Kingdom", "GB", "IRE"}
               for tag in event.get("meta-tags", []))


def discover_events(study, *, client=None, hours=24, max_pages=5, clock=utcnow):
    """Bounded public discovery, preserving the first schedule before capture."""
    from src.matchbook_client import MatchbookClient
    import config
    if not 1 <= hours <= 48 or not 1 <= max_pages <= 10:
        raise ValueError("Discovery must use 1–48 hours and 1–10 pages")
    study.verify()
    client = client or MatchbookClient()
    start = timestamp(clock())
    events, seen, offset, total, pages = [], set(), 0, None, 0
    for _ in range(max_pages):
        payload = client.get_events(sport_ids=config.MATCHBOOK_HORSE_RACING_SPORT_ID,
            after=int(start.timestamp()), before=int((start+timedelta(hours=hours)).timestamp()),
            per_page=50, offset=offset, include_prices=False, price_depth=1, currency="GBP")
        received = timestamp(clock())
        rows = payload.get("events", [])
        pages += 1
        total = payload.get("total", total)
        new_ids = 0
        for event in rows:
            identifier = numeric_id(event["id"])
            if identifier in seen:
                continue
            seen.add(identifier)
            new_ids += 1
            if not uk_ire_event(event):
                continue
            first = study.observe_event(event, received)
            events.append({"id": identifier, "name": event.get("name"), "start": event["start"],
                "status": event.get("status"), "in-running-flag": event.get("in-running-flag"),
                "first_start": first["first_start"], "first_seen_at": first["first_seen_at"]})
        offset += len(rows)
        if not rows or not new_ids or (total is not None and offset >= int(total)):
            break
    receipt = {"requested_at": start.isoformat(), "received_at": timestamp(clock()).isoformat(),
        "pages": pages, "returned_events": offset, "reported_total": total,
        "truncated": total is None or offset < int(total), "events": events,
        "scope": "Discovery only; prices must be refreshed when issuing a forecast"}
    from uuid import uuid4
    publish_json(study.path / "discovery" / f"{uuid4().hex}.json", receipt)
    return receipt


def market_forecast(event, market, observed_at, protocol):
    """Complete current field, one declared quote side, no price substitution."""
    observed = timestamp(observed_at)
    if not uk_ire_event(event):
        raise ValueError("outside_uk_ireland")
    if (event.get("status") != "open" or event.get("in-running-flag") is not False
            or market.get("status") != "open" or market.get("in-running-flag") is not False
            or market.get("live") is True):
        raise ValueError("market_not_open_preplay")
    if (str(market.get("name", "")).upper() != "WIN" or type(market.get("number-of-winners")) is not int
            or market.get("number-of-winners") != 1
            or market.get("type") != "multirunner" or market.get("product") != "EXCHANGE"
            or str(market.get("event-id")) != numeric_id(event["id"])):
        raise ValueError("not_single_winner_event_market")
    rows, identifiers = [], set()
    for runner in market.get("runners", []):
        identifier = numeric_id(runner["id"])
        if identifier in identifiers:
            raise ValueError("duplicate_market_runner_id")
        identifiers.add(identifier)
        if runner.get("withdrawn") is True:
            continue
        if runner.get("withdrawn") is not False or runner.get("status") != "open":
            raise ValueError("runner_state_unknown_or_suspended")
        changed = timestamp(runner.get("last-price-update-time"))
        age = (observed-changed).total_seconds()
        if age < -protocol["clock_skew_tolerance_seconds"] or age > protocol["max_runner_price_age_seconds"]:
            raise ValueError("stale_or_future_runner_price")
        quotes = []
        for price in runner.get("prices", []):
            if price.get("side") != "back":
                continue
            if price.get("currency") != "GBP" or price.get("odds-type") != "DECIMAL":
                raise ValueError("unexpected_price_currency_or_type")
            odds, amount = float(price.get("odds")), float(price.get("available-amount"))
            if not math.isfinite(odds) or odds <= 1 or not math.isfinite(amount) or amount < 0:
                raise ValueError("invalid_back_quote")
            if amount >= protocol["minimum_available_gbp"]:
                quotes.append((odds, amount))
        if not quotes:
            raise ValueError("incomplete_back_prices")
        odds, amount = max(quotes)
        name = clean_runner_name(runner.get("name"))
        if str(runner.get("market-id")) != numeric_id(market["id"]):
            raise ValueError("runner_market_mismatch")
        rows.append({"runner_id": numeric_id(runner["id"]), "market_runner_name": runner.get("name"),
            "horse_key": normalise_horse_key(name), "horse_name": name,
            "back_odds": odds, "back_available_gbp": amount,
            "price_updated_at": changed.isoformat(), "price_age_seconds": max(0.0, age),
            "source_clock_lead_seconds": max(0.0, -age)})
    result = pd.DataFrame(rows)
    if (len(result) < 2 or result.runner_id.duplicated().any() or result.horse_key.duplicated().any()
            or result.horse_key.eq("").any()):
        raise ValueError("incomplete_or_ambiguous_market_field")
    inverse = 1/result.back_odds.to_numpy(float)
    result["market_probability"] = inverse/inverse.sum()
    return result


def card_identity(cards, event):
    required = {"race_id", "horse_name", "race_date", "track", "off_time", "num_runners"}
    if not required <= set(cards) or cards.empty:
        raise ValueError("missing_racecard_fields")
    if cards.race_id.isna().any() or cards.race_id.nunique() != 1:
        raise ValueError("ambiguous_racecard_id")
    if any(pd.to_numeric(cards.get(name, pd.Series(0, index=cards.index)), errors="coerce").ne(0).any()
           for name in ("won", "finish_position")):
        raise ValueError("racecard_contains_outcomes")
    track, off = event_track_and_off_time(event)
    date = timestamp(event["start"]).date().isoformat()
    if (not pd.to_datetime(cards.race_date).dt.strftime("%Y-%m-%d").eq(date).all()
            or not cards.track.map(bf_track_key).eq(bf_track_key(track)).all()
            or not cards.off_time.map(normalise_off_time_key).eq(normalise_off_time_key(off)).all()):
        raise ValueError("racecard_event_identity_mismatch")
    if not pd.to_numeric(cards.num_runners, errors="coerce").eq(len(cards)).all():
        raise ValueError("racecard_declared_field_incomplete")
    keys = cards.horse_name.map(normalise_horse_key)
    if len(cards) < 2 or keys.eq("").any() or keys.duplicated().any():
        raise ValueError("ambiguous_racecard_runners")
    return keys


def eligible_reasons(protocol, first, event, *, card_at, market_at, finished_at, market_rows,
                     paired, history_last):
    finished = timestamp(finished_at)
    start = timestamp(first["first_start"])
    decision = start-timedelta(seconds=protocol["horizon_seconds"])
    opening = decision-timedelta(seconds=protocol["capture_window_seconds"])
    reasons = []
    if finished.astimezone(ZoneInfo("Europe/London")).date().isoformat() < protocol["first_eligible_date"]:
        reasons.append("before_registered_start_date")
    if timestamp(first["first_seen_at"]) > opening:
        reasons.append("schedule_first_seen_too_late")
    if timestamp(event["start"]) != start:
        reasons.append("schedule_changed")
    if not opening <= finished <= decision:
        reasons.append("outside_five_minute_window")
    if not paired:
        reasons.append("paired_forecast_unavailable")
    for name, observed, limit in (("racecard", card_at, protocol["max_card_age_seconds"]),
                                  ("market", market_at, protocol["max_market_age_seconds"])):
        if observed is None or not 0 <= (finished-timestamp(observed)).total_seconds() <= limit:
            reasons.append(f"{name}_observation_stale_or_missing")
        elif timestamp(observed) > decision:
            reasons.append(f"{name}_observed_after_cutoff")
    if market_rows is not None and any(timestamp(v) > decision for v in market_rows.price_updated_at):
        reasons.append("price_updated_after_cutoff")
    if history_last is None or not 1 <= (start.date()-pd.Timestamp(history_last).date()).days <= protocol["max_history_lag_days"]:
        reasons.append("history_not_current")
    return reasons


def fetch_racecard(event):
    """Fetch one current card, preserving decoded source data and receipt times."""
    from src.data_scraper import SportingLifeScraper

    class ObservedScraper(SportingLifeScraper):
        payload = None
        def _fetch_next_data(self, url):
            result = super()._fetch_next_data(url)
            self.payload = result
            return result

    date = timestamp(event["start"]).date().isoformat()
    track, off = event_track_and_off_time(event)
    scraper = ObservedScraper()
    links = scraper.get_racecard_urls(date, uk_only=True)
    matches = [row for row in links if bf_track_key(row["track"]) == bf_track_key(track)
               and normalise_off_time_key(row["time"]) == normalise_off_time_key(off)]
    if len(matches) != 1:
        raise ValueError("racecard_listing_identity_not_unique")
    started = utcnow()
    rows = scraper.scrape_racecard(matches[0]["url"], date)
    received = utcnow()
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError("racecard_unavailable")
    # Same CSV type contract used by the current serving racecard cache.
    canonical = pd.read_csv(io.StringIO(raw.to_csv(index=False)))
    return canonical, {"url": matches[0]["url"], "started_at": started.isoformat(),
        "received_at": received.isoformat(), "listing": matches[0], "payload": scraper.payload}


def _load_model(study):
    from src.model import RacePredictor
    model = RacePredictor()
    model.load(str(study.path / "model.joblib"))
    if model.market_anchor is not None:
        raise ValueError("Pinned forecast is not independent of current prices")
    return model


def capture_event(study, event_id, *, client=None, card_fetcher=None, clock=utcnow,
                  wait_for_issue_window=False, sleeper=None, monotonic=None):
    """One manual current observation; retries remain separate immutable attempts."""
    from src.matchbook_client import MatchbookClient
    from src.feature_store import FeatureStore, _stringify_objects
    from src.live_prediction import cards_signature, feature_store_signature, features_for_live_cards
    from src.forward_timing import plan_window, wait_for_window

    study.verify()
    event_id = numeric_id(event_id)
    if study.selected(event_id):
        raise ValueError("An eligible forecast for this event is already recorded")
    capture_id, folder = study.begin_attempt()
    client = client or MatchbookClient()
    errors, market_rows, cards, features, forecasts = {}, None, None, None, None
    card_at = market_at = history_last = first = event = market_id = None
    started = timestamp(clock())
    try:
        event = client.get_event(event_id)
        if numeric_id(event.get("id")) != event_id or not uk_ire_event(event):
            raise ValueError("event_identity_or_country_mismatch")
        event_received = timestamp(clock())
        first = study.observe_event(event, event_received)
        publish_json(folder / "first_observation.json", first)
        publish_json(folder / "event.json", {"received_at": event_received.isoformat(), "payload": event})
        if timestamp(clock()) >= min(timestamp(event["start"]), timestamp(first["first_start"])):
            raise ValueError("scheduled_start_already_passed")
        timing_plan = None
        if wait_for_issue_window:
            timing_plan = plan_window(study.protocol, first, event, now=clock())
            publish_json(folder / "timing_intent.json", timing_plan)
        try:
            cards, card_source = (card_fetcher or fetch_racecard)(event)
            card_at = timestamp(card_source["received_at"])
            if card_at > timestamp(clock()) or timestamp(card_source["started_at"]) < started:
                raise ValueError("racecard_receipt_outside_capture")
            publish_json(folder / "racecard_source.json", card_source)
            _stringify_objects(cards).to_parquet(folder / "racecard.parquet", index=False)
            card_identity(cards, event)
            store = FeatureStore()
            history_meta = store.meta()
            history_last = history_meta.get("last_date")
            history_signature = feature_store_signature(store)
            features = features_for_live_cards(cards, timestamp(event["start"]).date().isoformat(), store=store, use_cache=False)
            model = _load_model(study)
            record_places = bool(study.protocol.get("record_place_probabilities",False))
            extra = ["race_type","handicap","num_runners"] if record_places else []
            input_columns = list(dict.fromkeys(model.feature_cols + ["race_id", "horse_name", "race_date", "odds"]+extra))
            inputs = features[input_columns].sort_values(["race_id", "horse_name"], kind="stable")
            _stringify_objects(inputs).to_parquet(folder / "model_inputs.parquet", index=False)
            columns = ["race_id","horse_name","win_probability"]+(["place_probability"] if record_places else [])
            predicted = model.predict_races(inputs, win_only=not record_places)[columns]
            if record_places:
                refinement = getattr(model,"place_refinement",None)
                try:
                    model.place_refinement = None
                    legacy = model.predict_races(inputs,win_only=False)[["race_id","horse_name","win_probability","place_probability"]]
                finally:
                    model.place_refinement = refinement
                predicted = predicted.merge(legacy,on=["race_id","horse_name"],suffixes=("","_legacy"),validate="one_to_one")
                if not np.array_equal(predicted.win_probability,predicted.win_probability_legacy):
                    raise ValueError("Place control changed win probabilities")
                predicted = predicted.drop(columns="win_probability_legacy").rename(columns={
                    "place_probability":"model_place_probability","place_probability_legacy":"legacy_place_probability"})
                from src.each_way import get_ew_terms
                hcap = bool(inputs.handicap.iloc[0])
                terms = get_ew_terms(len(inputs),is_handicap=hcap)
                predicted["forecast_places"] = min(len(inputs),terms.places_paid if terms.eligible else 3)
                for name in ("model_place_probability","legacy_place_probability"):
                    values = predicted[name].to_numpy()
                    if (not np.isfinite(values).all() or ((values<0)|(values>1)).any()
                            or (values+1e-12<predicted.win_probability.to_numpy()).any()):
                        raise ValueError("Invalid recorded place probability")
            predicted["horse_key"] = predicted.horse_name.map(normalise_horse_key)
            if (len(predicted) != len(cards) or predicted.horse_key.duplicated().any()
                    or set(predicted.horse_key) != set(cards.horse_name.map(normalise_horse_key))
                    or not np.isfinite(predicted.win_probability).all()
                    or not predicted.win_probability.between(0, 1).all()
                    or not np.isclose(predicted.win_probability.sum(), 1, atol=1e-12, rtol=0)):
                raise ValueError("independent_probability_field_invalid")
            if timing_plan is not None:
                kwargs = {"clock": clock}
                if sleeper is not None:
                    kwargs["sleeper"] = sleeper
                if monotonic is not None:
                    kwargs["monotonic"] = monotonic
                try:
                    timing_receipt = wait_for_window(timing_plan, **kwargs)
                except Exception as exc:
                    publish_json(folder / "timing_receipt.json", {"failed_at": timestamp(clock()).isoformat(),
                        "error": str(exc)[:1000]})
                    raise
                publish_json(folder / "timing_receipt.json", timing_receipt)
            # Native history loading can outlast the observation-age limit.
            # Re-read the card after inference: only an EXACT unchanged input
            # can establish current availability of the already-computed p.
            refreshed, refreshed_source = (card_fetcher or fetch_racecard)(event)
            refreshed_at = timestamp(refreshed_source["received_at"])
            if (not card_at <= timestamp(refreshed_source["started_at"]) <= refreshed_at <= timestamp(clock())):
                raise ValueError("racecard_recheck_receipt_outside_capture")
            publish_json(folder / "racecard_refresh_source.json", refreshed_source)
            _stringify_objects(refreshed).to_parquet(folder / "racecard_refresh.parquet", index=False)
            card_identity(refreshed, event)
            if (list(cards.columns) != list(refreshed.columns)
                    or not cards.dtypes.equals(refreshed.dtypes)
                    or cards_signature(cards) != cards_signature(refreshed)):
                raise ValueError("racecard_changed_during_forecast")
            card_at = refreshed_at
            if feature_store_signature(store) != history_signature:
                raise ValueError("history_changed_during_forecast")
            publish_json(folder / "feature_vintage.json", {"signature": history_signature, "meta": history_meta})
            forecasts = predicted.rename(columns={"win_probability": "model_probability"})
            forecasts.to_parquet(folder / "independent_forecast.parquet", index=False)
        except Exception as exc:
            errors["independent"] = str(exc)[:1000]

        # Prices are observed after the potentially slower feature computation.
        # Refresh event state too; discovery prices cannot be reused as current.
        event = client.get_event(event_id)
        if numeric_id(event.get("id")) != event_id:
            raise ValueError("refreshed_event_identity_mismatch")
        market_started = timestamp(clock())
        payload = client.get_event_markets(event_id, include_prices=True, price_depth=3,
            per_page=20, currency="GBP", names="WIN")
        market_at = timestamp(clock())
        publish_json(folder / "market_source.json", {"started_at": market_started.isoformat(),
            "received_at": market_at.isoformat(), "event": event, "payload": payload})
        try:
            markets = [m for m in payload.get("markets", []) if str(m.get("name", "")).upper() == "WIN"]
            if len(markets) != 1:
                raise ValueError("win_market_missing_or_ambiguous")
            market_rows = market_forecast(event, markets[0], market_at, study.protocol)
            market_id = numeric_id(markets[0]["id"])
            market_rows.to_parquet(folder / "market_forecast.parquet", index=False)
            if forecasts is not None:
                if set(forecasts.horse_key) != set(market_rows.horse_key):
                    raise ValueError("racecard_market_field_mismatch")
                forecasts = forecasts.merge(market_rows.drop(columns="horse_name"), on="horse_key", validate="one_to_one")
            else:
                forecasts = market_rows.copy()
                forecasts["model_probability"] = np.nan
        except Exception as exc:
            errors["market"] = str(exc)[:1000]
            # An incomplete market field never gets renormalised or filled.
            if forecasts is not None:
                forecasts["market_probability"] = np.nan
    except Exception as exc:
        errors["capture"] = str(exc)[:1000]
    if forecasts is not None:
        forecasts.to_parquet(folder / "forecasts.parquet", index=False)
    study.verify()
    finished = timestamp(clock())
    paired = bool(forecasts is not None and not errors and "market_probability" in forecasts
                  and forecasts.model_probability.notna().all() and forecasts.market_probability.notna().all())
    try:
        reasons = (eligible_reasons(study.protocol, first, event, card_at=card_at, market_at=market_at,
            finished_at=finished, market_rows=market_rows, paired=paired, history_last=history_last)
            if first and event else ["event_unavailable"])
    except (ValueError, TypeError, KeyError):
        reasons = ["event_schedule_unusable"]
    if first and finished >= timestamp(first["first_start"]):
        reasons.append("committed_after_scheduled_start")
    reasons.extend(f"{name}:{value}" for name, value in errors.items())
    value = {"event_id": event_id, "market_id": market_id,
        "capture_mode": "timed" if wait_for_issue_window else "immediate",
        "started_at": started.isoformat(), "issued_at": finished.isoformat(),
        "model_id": study.protocol["model_id"], "model_sha256": study.protocol["model_sha256"],
        "paired_available": paired, "benchmark_eligible": not reasons,
        "reasons": reasons, "errors": errors, "history_last_date": history_last,
        "first_start": first["first_start"] if first else None,
        "first_seen_at": first["first_seen_at"] if first else None,
        "card_observed_at": card_at.isoformat() if card_at else None,
        "market_observed_at": market_at.isoformat() if market_at else None,
        "reference": study.protocol["reference"],
        "runners": len(forecasts) if forecasts is not None else 0,
        "scope": study.protocol["assessment_scope"]}
    return study.finish_attempt(capture_id, value, clock=clock)
