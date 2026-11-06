# ATR direct collector benchmark — 17 October 2026

Outcome: the existing direct collector was blocked on its first request.
No races were collected. Throughput and a historical backfill duration could
not be established. The test stopped immediately without retries or parallel
requests.

- Planned sample: 50 races, starting with the previously browser-verified
  Cheltenham and Wolverhampton races, followed by 48 races spread across the
  local results database's January 2025–September 2026 interval.
- Attempted: Cheltenham, 11 March 2025, 13:20.
- Existing transport selected: curl_cffi.
- Configured minimum request gap: 2 seconds, plus the collector's jitter.
- First request duration: 0.314 seconds; race attempt: 0.613 seconds.
- Collector response: status 429, empty body, classified blocked.
- Important ambiguity: the collector converts an HTTP 200 Client Challenge
  response into status 429 and discards the body. Consequently this record
  cannot distinguish a real server 429 from that normalized challenge outcome.
- Complete races: zero. No sectionals endpoints were requested.

The same race loaded successfully in a normal browser in the earlier sample.
That demonstrates a difference between access methods, not reliable batch
access. Repeated attempts or a larger target list would not resolve the
immediate blocker, so no further requests were made.

## Implication for the project

The current collector is not demonstrated ready for a thousands-race backfill
from this environment. Its code comments claiming that browser-like requests
pass the challenge did not hold in this test. Request latency cannot be used
as successful-race throughput, and no completion-time estimate is justified.

The earlier browser test establishes small-scale extraction only. A sustained
browser benchmark would be a separate experiment; its throughput remains
unmeasured. For a dependable backfill, compare supported bulk historical
access with the engineering and maintenance cost of browser collection.

## Artifacts and isolation

The benchmark runner is scripts/benchmark_atr_collection.py. The detailed
attempt and request records are in
data/research/atr_benchmark_2026-10-17/summary.json; its planned sample is in
planned_sample.json beside it. Production caches, skip-lists, source collector,
model inputs and model settings were not modified. The benchmark uses a
read-only SQLite connection to select candidate races.
