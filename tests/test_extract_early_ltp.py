"""Early decision prices from the BASIC stream must use UK clock time.

ltp_1100 is the price a bettor acting on full-morning information can take
(ledger 14). In summer 11:00 UK is 10:00 UTC; a UTC slip would hand the
bettor a price from an hour after the morning closed.
"""

import json
from datetime import datetime, timezone

from src.ingestion.betfair_streams import _uk_clock_ms, parse_market


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp() * 1000)


def test_uk_clock_handles_summer_and_winter():
    assert _uk_clock_ms("2025-07-05T14:35:00.000Z", 11, 0) == _ms("2025-07-05T10:00:00")
    assert _uk_clock_ms("2025-01-11T14:35:00.000Z", 11, 0) == _ms("2025-01-11T11:00:00")


def test_parse_market_records_early_cuts():
    definition = {"marketTime": "2025-07-05T14:35:00.000Z", "marketType": "WIN",
                  "countryCode": "GB", "eventTypeId": "7", "eventName": "Sandown 5th Jul",
                  "runners": [{"id": 1, "name": "Alpha", "status": "WINNER", "bsp": 4.2}]}
    lines = [{"pt": _ms("2025-07-05T08:00:00"), "mc": [{"marketDefinition": definition}]}]
    for when, price in (("09:59", 6.0), ("10:01", 5.5), ("13:30", 5.0), ("14:29", 4.6)):
        lines.append({"pt": _ms(f"2025-07-05T{when}:00"), "mc": [{"rc": [{"id": 1, "ltp": price}]}]})
    rows = parse_market("\n".join(json.dumps(x) for x in lines).encode())
    row = rows[0]
    assert row["ltp_1100"] == 6.0      # last trade at or before 10:00 UTC (11:00 BST)
    assert row["ltp_3600s"] == 5.0     # last trade at or before 13:35 UTC
    assert row["ltp_300s"] == 4.6


def test_timing_fields_staleness_activity_and_reduction_factors():
    """EXPLORATION_PLAN B: an early LTP is only a strike price if it is fresh
    and survives later non-runners (Betfair reduces matched odds; BSP is not)."""
    definition = {"marketTime": "2025-07-05T14:35:00.000Z", "marketType": "WIN",
                  "countryCode": "GB", "eventTypeId": "7", "eventName": "Sandown 5th Jul",
                  "runners": [
                      {"id": 1, "name": "Alpha", "status": "WINNER", "bsp": 4.2},
                      {"id": 2, "name": "Beta", "status": "REMOVED", "adjustmentFactor": 20.0,
                       "removalDate": "2025-07-05T12:00:00.000Z"},
                      {"id": 3, "name": "Gamma", "status": "REMOVED", "adjustmentFactor": 10.0,
                       "removalDate": "2025-07-05T09:30:00.000Z"},
                  ]}
    lines = [{"pt": _ms("2025-07-04T16:00:00"), "mc": [{"id": "1.5", "marketDefinition": definition}]}]
    for when, price in (("2025-07-04T16:30", 7.0), ("2025-07-05T09:00", 6.0),
                        ("2025-07-05T13:30", 5.0), ("2025-07-05T14:29", 4.6)):
        lines.append({"pt": _ms(f"{when}:00"), "mc": [{"id": "1.5", "rc": [{"id": 1, "ltp": price}]}]})
    row = parse_market("\n".join(json.dumps(x) for x in lines).encode())[0]
    assert row["ltp_eve1800"] == 7.0          # 18:00 BST the day before = 17:00 UTC
    assert row["ltp_0800"] == 7.0             # 08:00 BST: the evening trade still stands
    assert row["age_1100"] == 3600.0          # last change 09:00 UTC, cut 10:00 UTC
    assert row["nchg_1100"] == 2
    assert row["prev_ltp_1100"] == 7.0 and row["prev_ltp_300s"] == 5.0
    # A bet matched at 11:00 BST is cut by Beta (removed 12:00 UTC) but not
    # Gamma (removed before the cut) ...
    assert row["rf_after_1100"] == 0.8 and row["n_removed_after_1100"] == 1
    # ... and its LTP (09:00 UTC) predates Gamma's removal (09:30 UTC).
    assert abs(row["rf_stale_1100"] - 0.9) < 1e-12
    assert row["rf_after_300s"] == 1.0 and row["rf_stale_300s"] == 1.0
    assert row["rf25_after_1100"] == 0.8 and abs(row["rf25_stale_1100"] - 0.9) < 1e-12
    assert row["bf_market_id"] == "1.5"


def test_waived_reduction_variant_skips_small_factors():
    from src.ingestion.betfair_streams import _timing_fields
    out = _timing_fields([(100, 5.0)], {"c": 200}, [(300, 2.0), (400, 10.0)])
    assert abs(out["rf_after_c"] - 0.98 * 0.9) < 1e-12
    assert abs(out["rf25_after_c"] - 0.9) < 1e-12


def test_removed_runners_count_in_the_early_book():
    """A runner withdrawn after the cut was in the book at the cut."""
    definition = {"marketTime": "2025-01-11T14:35:00.000Z", "marketType": "WIN",
                  "countryCode": "GB", "eventTypeId": "7", "eventName": "Sandown 11th Jan",
                  "runners": [
                      {"id": 1, "name": "Alpha", "status": "WINNER", "bsp": 2.5},
                      {"id": 2, "name": "Beta", "status": "REMOVED", "adjustmentFactor": 40.0,
                       "removalDate": "2025-01-11T13:00:00.000Z"},
                  ]}
    lines = [{"pt": _ms("2025-01-11T08:00:00"), "mc": [{"id": "1.6", "marketDefinition": definition}]},
             {"pt": _ms("2025-01-11T10:30:00"), "mc": [{"id": "1.6", "rc": [{"id": 1, "ltp": 4.0},
                                                                            {"id": 2, "ltp": 2.0}]}]}]
    rows = parse_market("\n".join(json.dumps(x) for x in lines).encode())
    assert len(rows) == 1
    row = rows[0]
    assert row["book_removed_after_1100"] == 0.5 and row["min_removed_ltp_1100"] == 2.0
    assert row["book_removed_after_300s"] == 0.0 and row["min_removed_ltp_300s"] is None
    assert abs(row["rf_after_1100"] - 0.6) < 1e-12
