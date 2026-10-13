"""Early decision prices from the BASIC stream must use UK clock time.

ltp_1100 is the price a bettor acting on full-morning information can take
(ledger 14). In summer 11:00 UK is 10:00 UTC; a UTC slip would hand the
bettor a price from an hour after the morning closed.
"""

import json
from datetime import datetime, timezone

from scripts.extract_betfair_bsp import _uk_clock_ms, parse_market


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
