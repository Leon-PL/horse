"""Parsers for Betfair Historical BASIC ``mcm`` market streams.

Each ``.bz2`` holds one market's JSON-lines stream, the same feed the Exchange API
emits. A market is parsed purely from its own ``marketDefinition``
(``eventName``/``marketTime``), never its archive path, because one folder can mix
tracks and days. The main Historical API collector replays these parsers over
both automated downloads and consolidated existing archives.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from src.betfair import bf_horse_key
from src.utils import normalise_off_time_key, normalise_track_key

KEEP_COUNTRIES = {"GB", "IE"}
# "Nottingham 1st Jun" -> "Nottingham"; "Chelmsford City 1st Jun" -> "Chelmsford City".
# Month optional (malformed "Newton Abbot 16th"); leading space optional too
# (older data fuses them: "Cork1st Oct" -> "Cork"). No course name ends in a
# digit+ordinal, so this is safe.
TRACK_DATE_RE = re.compile(r"\s*\d{1,2}(?:st|nd|rd|th)(?:\s+\w{3,})?\s*$", re.IGNORECASE)
COUNTRY_SUFFIX_RE = re.compile(r"\s*\([A-Z]{2,3}\)\s*$")
CLOTH_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")


def _track_from_event(event_name: str) -> str:
    return TRACK_DATE_RE.sub("", event_name or "").strip()


def _clean_horse(name: str) -> str:
    name = CLOTH_PREFIX_RE.sub("", name or "")
    name = COUNTRY_SUFFIX_RE.sub("", name)
    return name.strip()


def _iso_to_epoch_ms(iso: str) -> int | None:
    if not iso:
        return None
    try:
        dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return int(dt.timestamp() * 1000)


def _uk_clock_ms(market_time: str, hour: int, minute: int, day_offset: int = 0) -> int | None:
    """Epoch ms of HH:MM UK local time on the race's date (marketTime is UTC),
    shifted by *day_offset* days (-1 = the evening before)."""
    if not market_time or len(market_time) < 10:
        return None
    from datetime import timedelta
    from zoneinfo import ZoneInfo
    try:
        day = datetime.strptime(market_time[:10], "%Y-%m-%d") + timedelta(days=day_offset)
        local = day.replace(hour=hour, minute=minute, tzinfo=ZoneInfo("Europe/London"))
    except ValueError:
        return None
    return int(local.timestamp() * 1000)


#: Reduction factors below this (%) may be waived by Betfair; unverified, so
#: the waived version is recorded alongside, never instead of, the full one.
RF_WAIVER_PCT = 2.5


def _timing_fields(events: list[tuple[int, float]], cuts: dict[str, int | None],
                   removals: list[tuple[int, float]]) -> dict:
    """How tradeable each early LTP was, for the execution-timing test.

    Per cut: ``age_<cut>`` (seconds since the runner's LTP last changed; BASIC
    publishes changes only, so this bounds staleness from above),
    ``nchg_<cut>`` (LTP changes so far, an activity proxy known at the cut),
    and the non-runner reduction that a back bet matched at the cut would
    suffer: ``rf_after_<cut>`` multiplies the price for every runner removed
    after the cut (Betfair reduces matched odds by the removed runner's
    reduction factor; BSP is unaffected), ``rf_stale_<cut>`` for removals
    between the last trade and the cut (the LTP predates the reformed market).
    Every factor is applied, including those under the 2.5% Betfair may waive:
    the conservative reading.
    """
    out: dict = {}
    for name, cut in cuts.items():
        last_pt = None
        n = 0
        prev = last = None
        if cut is not None:
            for pt, ltp in events:
                if pt > cut:
                    break
                last_pt = pt
                n += 1
                if ltp != last:
                    prev, last = last, ltp
        out[f"age_{name}"] = (cut - last_pt) / 1000.0 if last_pt is not None else None
        out[f"nchg_{name}"] = n
        # The LTP before the last change: a fall means a backer just took the
        # price (so it was there to back); a rise means a layer may have
        # matched at the top of the spread (bid-ask bounce).
        out[f"prev_ltp_{name}"] = prev
        after = stale = after25 = stale25 = 1.0
        n_after = 0
        if cut is not None:
            for rm_ms, rf in removals:
                keep25 = rf >= RF_WAIVER_PCT
                if rm_ms > cut:
                    after *= 1.0 - rf / 100.0
                    after25 *= (1.0 - rf / 100.0) if keep25 else 1.0
                    n_after += 1
                elif last_pt is not None and rm_ms > last_pt:
                    stale *= 1.0 - rf / 100.0
                    stale25 *= (1.0 - rf / 100.0) if keep25 else 1.0
        out[f"rf_after_{name}"] = after
        out[f"rf_stale_{name}"] = stale
        # The same, if Betfair waives factors below 2.5% (unverified).
        out[f"rf25_after_{name}"] = after25
        out[f"rf25_stale_{name}"] = stale25
        out[f"n_removed_after_{name}"] = n_after
    return out


def parse_market(stream: bytes) -> list[dict] | None:
    """Parse one market's mcm stream -> list of per-runner rows (or None to skip)."""
    final_def: dict | None = None
    names: dict[int, str] = {}
    # runner_id -> list[(pt_ms, ltp)] kept only as a running "last trade" tracker
    last_ltp: dict[int, float] = {}
    ltp_60: dict[int, float] = {}
    ltp_300: dict[int, float] = {}
    ltp_pre: dict[int, float] = {}
    ltp_3600: dict[int, float] = {}
    ltp_1100: dict[int, float] = {}
    off_ms: int | None = None

    # First pass to find off-time (marketTime) — appears in every marketDefinition.
    # We parse line-by-line, updating definition and prices together; because the
    # marketTime is present from the first definition, the windowed cutoffs work.
    lines = stream.split(b"\n")
    parsed = []
    for raw in lines:
        if not raw.strip():
            continue
        try:
            parsed.append(json.loads(raw))
        except json.JSONDecodeError:
            continue

    # Establish off-time and the final settled definition.
    for d in parsed:
        for mc in d.get("mc", []):
            md = mc.get("marketDefinition")
            if md:
                final_def = md
                if off_ms is None:
                    off_ms = _iso_to_epoch_ms(md.get("marketTime", ""))

    if not final_def:
        return None
    if final_def.get("marketType") != "WIN":
        return None
    if final_def.get("countryCode") not in KEEP_COUNTRIES:
        return None
    if final_def.get("eventTypeId") != "7":  # horse racing
        return None

    for r in final_def.get("runners", []):
        names[r["id"]] = r.get("name", "")

    cut_60 = (off_ms - 60_000) if off_ms else None
    cut_300 = (off_ms - 300_000) if off_ms else None
    # Early decision prices for the morning-money test (ledger 14): one hour
    # before the off, and 11:00 UK time on race day - the end of Betfair's
    # "morning" period, inferred from where the morning share of pre-play
    # volume collapses. A bet struck on full-morning information can take the
    # 11:00 price, never the morning VWAP (which includes earlier prices).
    cut_3600 = (off_ms - 3_600_000) if off_ms else None
    cut_1100 = _uk_clock_ms(final_def.get("marketTime", ""), 11, 0)
    # Execution-timing and ground-forecast tests (EXPLORATION_PLAN B, C): the
    # evening before (18:00 UK) and race-day 08:00 UK, after the morning
    # going reports.
    cut_eve = _uk_clock_ms(final_def.get("marketTime", ""), 18, 0, day_offset=-1)
    cut_0800 = _uk_clock_ms(final_def.get("marketTime", ""), 8, 0)
    ltp_eve: dict[int, float] = {}
    ltp_0800: dict[int, float] = {}
    events: dict[int, list[tuple[int, float]]] = {}
    first_pt = next((d.get("pt") for d in parsed if d.get("pt") is not None), None)

    for d in parsed:
        pt = d.get("pt")
        for mc in d.get("mc", []):
            for rc in mc.get("rc", []):
                if "ltp" not in rc:
                    continue
                rid = rc["id"]
                ltp = rc["ltp"]
                last_ltp[rid] = ltp
                if pt is not None:
                    events.setdefault(rid, []).append((pt, ltp))
                if cut_eve is not None and pt is not None and pt <= cut_eve:
                    ltp_eve[rid] = ltp
                if cut_0800 is not None and pt is not None and pt <= cut_0800:
                    ltp_0800[rid] = ltp
                if off_ms is not None and pt is not None and pt <= off_ms:
                    ltp_pre[rid] = ltp
                if cut_60 is not None and pt is not None and pt <= cut_60:
                    ltp_60[rid] = ltp
                if cut_300 is not None and pt is not None and pt <= cut_300:
                    ltp_300[rid] = ltp
                if cut_3600 is not None and pt is not None and pt <= cut_3600:
                    ltp_3600[rid] = ltp
                if cut_1100 is not None and pt is not None and pt <= cut_1100:
                    ltp_1100[rid] = ltp

    event_name = final_def.get("eventName", "")
    track = _track_from_event(event_name)
    market_time = final_def.get("marketTime", "")
    race_date = market_time[:10] if market_time else ""
    off_hhmm = market_time[11:16] if len(market_time) >= 16 else ""
    n_active = sum(
        1 for r in final_def.get("runners", []) if r.get("status") in ("WINNER", "LOSER")
    )
    removals = sorted(
        (ms, float(r.get("adjustmentFactor") or 0.0))
        for r in final_def.get("runners", [])
        if r.get("status") == "REMOVED"
        for ms in [_iso_to_epoch_ms(r.get("removalDate", ""))]
        if ms is not None
    )
    cuts = {"eve1800": cut_eve, "0800": cut_0800, "1100": cut_1100,
            "3600s": cut_3600, "300s": cut_300}
    market_ids = {mc.get("id") for d in parsed for mc in d.get("mc", []) if mc.get("id")}
    # The book as it stood at each cut includes runners withdrawn later, which
    # never appear as rows. Without them an early overround (or "favourite")
    # computed over the final field would encode the future withdrawal.
    removed_book: dict = {}
    for name, cut in cuts.items():
        prices = []
        for r in final_def.get("runners", []):
            if r.get("status") != "REMOVED":
                continue
            rm = _iso_to_epoch_ms(r.get("removalDate", ""))
            if cut is None or rm is None or rm <= cut:
                continue
            at_cut = [ltp for pt, ltp in events.get(r["id"], []) if pt <= cut]
            if at_cut and at_cut[-1]:
                prices.append(float(at_cut[-1]))
        removed_book[f"book_removed_after_{name}"] = sum(1.0 / x for x in prices if x > 1)
        removed_book[f"min_removed_ltp_{name}"] = min(prices) if prices else None

    rows = []
    for r in final_def.get("runners", []):
        status = r.get("status")
        if status == "REMOVED":
            continue
        rid = r["id"]
        raw_name = names.get(rid, r.get("name", ""))
        horse = _clean_horse(raw_name)
        timing = _timing_fields(events.get(rid, []), cuts, removals)
        rows.append(
            {
                "race_date": race_date,
                "off_time": off_hhmm,
                "track": track,
                "horse_name": horse,
                "bf_runner_id": rid,
                "bsp": r.get("bsp"),
                "ltp_preoff": ltp_pre.get(rid),
                "ltp_60s": ltp_60.get(rid),
                "ltp_300s": ltp_300.get(rid),
                "ltp_3600s": ltp_3600.get(rid),
                "ltp_1100": ltp_1100.get(rid),
                "ltp_last": last_ltp.get(rid),
                "won": 1 if status == "WINNER" else 0,
                "n_runners": n_active,
                "market_id": final_def.get("eventId", ""),
                "track_key": normalise_track_key(track),
                "off_key": normalise_off_time_key(off_hhmm),
                "horse_key": bf_horse_key(horse),
                **timing,
                **removed_book,
                "ltp_eve1800": ltp_eve.get(rid),
                "ltp_0800": ltp_0800.get(rid),
                "off_ms": off_ms,
                "mkt_first_pt": first_pt,
                "bf_market_id": next(iter(market_ids)) if len(market_ids) == 1 else None,
            }
        )
    return rows


def parse_place_market(stream: bytes) -> list[dict] | None:
    """Parse one market's mcm stream -> place rows, or None if not a PLACE market.

    Captures the place BSP (settlement price) AND the pre-off last-traded place
    price (``place_ltp_300s`` = 5 min before the off, ``place_ltp_preoff`` = at
    the off). The pre-off prices are what you'd actually *select* on live, so the
    BSP-peek bias is removed by selecting on them and settling at BSP.
    """
    parsed = []
    for raw in stream.split(b"\n"):
        if not raw.strip():
            continue
        try:
            parsed.append(json.loads(raw))
        except json.JSONDecodeError:
            continue

    final_def = None
    off_ms = None
    for d in parsed:
        for mc in d.get("mc", []):
            md = mc.get("marketDefinition")
            if md:
                final_def = md
                if off_ms is None:
                    off_ms = _iso_to_epoch_ms(md.get("marketTime", ""))
    if not final_def:
        return None
    if final_def.get("marketType") != "PLACE":
        return None
    if final_def.get("countryCode") not in KEEP_COUNTRIES:
        return None
    if final_def.get("eventTypeId") != "7":
        return None

    # Pre-off last-traded-price tracking (mirrors the win extractor).
    last_ltp: dict[int, float] = {}
    ltp_pre: dict[int, float] = {}
    ltp_300: dict[int, float] = {}
    cut_300 = (off_ms - 300_000) if off_ms else None
    for d in parsed:
        pt = d.get("pt")
        for mc in d.get("mc", []):
            for rc in mc.get("rc", []):
                if "ltp" not in rc:
                    continue
                rid = rc["id"]
                last_ltp[rid] = rc["ltp"]
                if off_ms is not None and pt is not None and pt <= off_ms:
                    ltp_pre[rid] = rc["ltp"]
                if cut_300 is not None and pt is not None and pt <= cut_300:
                    ltp_300[rid] = rc["ltp"]

    places_paid = final_def.get("numberOfWinners")
    event_name = final_def.get("eventName", "")
    track = _track_from_event(event_name)
    market_time = final_def.get("marketTime", "")
    race_date = market_time[:10] if market_time else ""
    off_hhmm = market_time[11:16] if len(market_time) >= 16 else ""

    rows = []
    for r in final_def.get("runners", []):
        status = r.get("status")
        if status not in ("WINNER", "LOSER"):
            continue
        rid = r["id"]
        horse = _clean_horse(r.get("name", ""))
        rows.append({
            "race_date": race_date,
            "off_time": off_hhmm,
            "track": track,
            "horse_name": horse,
            "bf_runner_id": rid,
            "place_bsp": r.get("bsp"),
            "place_ltp_300s": ltp_300.get(rid),
            "place_ltp_preoff": ltp_pre.get(rid),
            "place_ltp_last": last_ltp.get(rid),
            "bf_places_paid": places_paid,
            "placed_bf": 1 if status == "WINNER" else 0,
            "track_key": normalise_track_key(track),
            "off_key": normalise_off_time_key(off_hhmm),
            "horse_key": bf_horse_key(horse),
        })
    return rows or None
