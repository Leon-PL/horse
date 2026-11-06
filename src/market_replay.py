"""Replay a historical BASIC stream using only information available at a cutoff.

This is a forecast reference from last traded prices, not an executable quote.
Final definitions supply outcomes only: they cannot remove runners, rename
horses, change the scheduled cutoff or supply a missing price at that cutoff.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.ingestion.betfair_streams import _clean_horse, _iso_to_epoch_ms, _track_from_event
from src.betfair import bf_horse_key, bf_track_key
from src.utils import normalise_off_time_key


class ReplayRejected(ValueError):
    """A whole market failed the declared replay contract."""


def split_market_streams(stream: bytes):
    """Demultiplex event files; never let one market's definition price another.

    Yield (initial definition, stream) for every GB/IE WIN market. The archive
    includes both single-market and event-level files, regardless of extension.
    Other sports/market types are discarded before buffering their updates.
    """
    buffers, definitions, seen = {}, {}, set()
    for raw in stream.splitlines():
        if not raw.strip():
            continue
        try:
            message = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ReplayRejected("malformed_json") from exc
        for change in message.get("mc", []):
            mid = change.get("id")
            if not mid:
                raise ReplayRejected("missing_market_id")
            if mid not in seen:
                seen.add(mid)
                md = change.get("marketDefinition", {})
                if not md:
                    raise ReplayRejected("missing_initial_definition")
                if (md.get("marketType") == "WIN" and md.get("countryCode") in {"GB", "IE"}
                        and str(md.get("eventTypeId")) == "7"):
                    definitions[mid], buffers[mid] = md, []
            if mid in buffers:
                buffers[mid].append(json.dumps({"pt": message.get("pt"), "mc": [change]}).encode())
    for mid, chunks in buffers.items():
        yield definitions[mid], b"\n".join(chunks)


def replay_market(stream: bytes, *, seconds_before=300) -> list[dict]:
    """Freeze the first published schedule; include every active runner at T-5.

    Definitions contain full runner lists in these BASIC archives. Missing or
    inconsistent lists are rejected. A later withdrawal stays in the forecast
    and receives a separate outcome flag; it is never disguised as a DNF.
    """
    if not isinstance(seconds_before, int) or isinstance(seconds_before, bool) or seconds_before < 1:
        raise ValueError("seconds_before must be a positive integer")
    initial = snapshot = final = None
    cutoff = schedule = market_id = observation = definition_at = previous = None
    prices, price_at = {}, {}
    for raw in stream.splitlines():
        if not raw.strip():
            continue
        try:
            message = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise ReplayRejected("malformed_json") from exc
        pt = message.get("pt")
        if not isinstance(pt, int) or (previous is not None and pt < previous):
            raise ReplayRejected("missing_or_nonmonotonic_timestamp")
        previous = pt
        for change in message.get("mc", []):
            md = change.get("marketDefinition")
            if initial is None:
                if not md:
                    raise ReplayRejected("missing_initial_definition")
                if (md.get("marketType") != "WIN" or md.get("countryCode") not in {"GB", "IE"}
                        or str(md.get("eventTypeId")) != "7"):
                    raise ReplayRejected("outside_gb_ie_win")
                schedule = _iso_to_epoch_ms(md.get("marketTime", ""))
                if schedule is None:
                    raise ReplayRejected("missing_initial_schedule")
                cutoff = schedule - seconds_before * 1000
                if pt > cutoff:
                    raise ReplayRejected("archive_begins_after_decision")
                market_id, initial = change.get("id"), dict(md)
                if not market_id:
                    raise ReplayRejected("missing_market_id")
            if change.get("id") != market_id:
                raise ReplayRejected("multiple_markets_in_stream")
            if md:
                final = {**(final or {}), **md}
            if pt > cutoff:
                continue
            observation = pt
            if md:
                snapshot = {**(snapshot or {}), **md}
                definition_at = pt
            for runner in change.get("rc", []):
                if "ltp" in runner:
                    try:
                        prices[runner["id"]] = float(runner["ltp"])
                    except (TypeError, ValueError, KeyError) as exc:
                        raise ReplayRejected("invalid_price_update") from exc
                    price_at[runner["id"]] = pt
    if snapshot is None:
        raise ReplayRejected("no_snapshot")
    if snapshot.get("status") != "OPEN" or snapshot.get("inPlay") is not False:
        raise ReplayRejected("not_open_preplay_at_decision")
    runners = snapshot.get("runners", [])
    ids = [r.get("id") for r in runners]
    if not ids or None in ids or len(set(ids)) != len(ids):
        raise ReplayRejected("ambiguous_runner_ids")
    if any(r.get("status") not in {"ACTIVE", "REMOVED"} for r in runners):
        raise ReplayRejected("unknown_runner_status_at_decision")
    active = [r for r in runners if r["status"] == "ACTIVE"]
    if len(active) < 2 or snapshot.get("numberOfActiveRunners") != len(active):
        raise ReplayRejected("incomplete_active_field")
    if any(not math.isfinite(prices.get(r["id"], math.nan)) or prices.get(r["id"], 0) <= 1
           for r in active):
        raise ReplayRejected("missing_or_invalid_active_price")
    final_ids = [r.get("id") for r in (final or {}).get("runners", [])]
    if len(set(final_ids)) != len(final_ids):
        raise ReplayRejected("ambiguous_settlement_ids")
    settled = {r["id"]: r.get("status") for r in (final or {}).get("runners", [])}
    winners = [rid for rid, status in settled.items() if status == "WINNER"]
    if ((final or {}).get("status") != "CLOSED" or len(winners) != 1
            or winners[0] not in {r["id"] for r in active}
            or any(settled.get(r["id"]) not in {"WINNER", "LOSER", "REMOVED"} for r in active)):
        raise ReplayRejected("not_single_winner_settled")
    # A runner becoming active after the decision changes the event itself.
    if any(status in {"WINNER", "LOSER"} and rid not in {r["id"] for r in active}
           for rid, status in settled.items()):
        raise ReplayRejected("starter_added_after_decision")
    stamp = datetime.fromtimestamp(schedule / 1000, tz=timezone.utc)
    local = stamp.astimezone(ZoneInfo("Europe/London"))
    track = _track_from_event(snapshot.get("eventName", ""))
    if not track:
        raise ReplayRejected("missing_snapshot_track")
    keys = [bf_horse_key(_clean_horse(r.get("name", ""))) for r in active]
    if "" in keys or len(set(keys)) != len(keys):
        raise ReplayRejected("ambiguous_snapshot_names")
    book = sum(1 / prices[r["id"]] for r in active)
    return [{
        "market_id": market_id, "selection_id": r["id"],
        "race_date": local.date().isoformat(), "track": track, "track_key": bf_track_key(track),
        "off_key": normalise_off_time_key(stamp.strftime("%H:%M")),
        "off_key_local": normalise_off_time_key(local.strftime("%H:%M")),
        "horse_name": _clean_horse(r.get("name", "")), "horse_key": key,
        "scheduled_start_ms": schedule, "decision_ms": cutoff,
        "observed_ms": observation, "definition_ms": definition_at,
        "price_changed_ms": price_at[r["id"]], "active_runners": len(active),
        "market_price": prices[r["id"]], "market_probability": 1 / prices[r["id"]] / book,
        "settlement_status": settled[r["id"]], "won": int(settled[r["id"]] == "WINNER"),
        "withdrawn_after_decision": settled[r["id"]] == "REMOVED",
    } for r, key in zip(active, keys)]
