# Betfair import speed — 6 November 2026

The importer retains both GB/IE WIN and PLACE markets and every existing parsed
field. Routine refreshes and historical catch-up use the same optimized path.
Full raw BASIC payloads remain available for replay.

## Changes

- Group parsed observations in durable queue transactions: up to 128 markets or
  two seconds of completed work, instead of a separate commit for every market.
- Flush complete new raw files before accepting their queue rows. Already retained
  files and consolidated archive payloads are reused without another raw copy.
- Keep at most twice the worker count in flight, plus the checkpoint buffer.
- Update progress every five seconds using incremental counters; recount committed
  state at batch boundaries. Cached discovery windows avoid repeated queue scans.
- Count imported markets only after a successful commit. Interrupted checkpoints
  leave queue entries replayable. Stop scheduling new work after login failure.

SQLite durability settings, strict native validation, parser outputs, request
limits and existing-price preservation are unchanged. Four workers remain active;
continuous catch-up uses 5,000-market batches to spread selection/publication costs.

## Measurements

`data/audit/betfair_performance_20261106/` contains the source/queue backup,
before/after comparisons, test results, live qualification and launch receipt.

The controlled comparison uses 16 real retained markets: nine WIN and seven PLACE.
The old source snapshot and new writer receive identical payloads in disposable
queues under the normal Windows account. Production databases are read-only.

| Work | Time |
|---|---:|
| Old individual metadata saves | 7.827 seconds |
| New grouped durable metadata save | 1.492 seconds |
| New grouped save including full raw retention | 2.285 seconds |

Metadata checkpointing was **5.24 times faster** in this comparison. Parsed
observations, queue state, payload hashes and byte counts matched exactly. The
raw-retention variant also verified every compressed source byte. A preceding
trial measured 19.24 seconds individually versus 1.51 seconds in one transaction;
storage contention makes the absolute timings variable.

The bounded production check imported 256 archived markets in 118.263 seconds
(2.16 markets/second, including pending selection and checkpointing). It reported
zero failures and published all four canonical price caches. Including previously
staged work, the queue reached 2,179 published markets with 163,107 still pending.
Publishing those caches took approximately another three minutes; that cost is
outside `import_seconds` and is shared across larger continuous batches.

A separate read-only selection probe found 489,650 archived market IDs in 38.61
seconds; queue selection itself took 0.39 seconds and iterating candidate rows
another 0.57 seconds. This archive-availability scan remains a batch overhead.
It is not a per-market cost. Consolidation continued during the measurements.

After restart, a 98.39-second window imported 783 more markets: **7.96 markets per
second**, with zero failures. An earlier window measured about six per second.
The queue then had 161,564 pending, 1,543 durably staged and 2,179 published markets.
The audit retains both progress snapshots and `live_rate.json`; collector pins
match the running worker. This larger-batch sample excludes initial selection
and publication, and measures local replay rather than network downloads.

At six to eight markets per second, pending imports alone represent roughly six
to eight hours. Allow **one to two days provisionally** for the full backfill:
archive completion, API gaps, publication and competing disk work remain variable.
Use continuing live progress rather than treating the local rate as a completion
promise. Both markets and all source data remain included.

## Verification and operation

The focused history/archive suite passed **76 tests**, with the restricted-profile
Windows encryption case skipped. That encryption path was separately verified
under the normal Windows account during connection setup; the production check
also used the saved account successfully. New checks exercise transaction rollback,
exact resume from retained raw files, duplicate checkpoint rejection, login failure
and mixed success/retry progress counts. Targeted lint and whitespace checks pass.

The continuous worker was restarted at 22:07 UTC on 6 November with the current
collector pins, the original full date range (2020-09-15–2026-10-31), four workers
and 5,000-market batches. Read `status.json` and `backfill_status.json` under
`data/betfair/history/` for current committed progress. Launch and later rate
snapshots are retained in the performance audit directory. The app health check
passed. The independent ATR and archive consolidation workers continued; all four
ATR collector hashes still match their manifest.

Temporary test queues were removed after verification. The reusable diagnostic is
`scripts/benchmark_betfair_ingestion.py`; use a trusted source snapshot with
`--baseline-source` for a future before/after comparison. Audit receipts and the
consistent pre-change queue backup are intentionally retained.
