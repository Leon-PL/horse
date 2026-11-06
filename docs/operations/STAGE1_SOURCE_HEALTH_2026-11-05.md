# Stage 1 follow-up: targeted recovery and source-health reporting

**Snapshot at 07:47 UTC.** The later [BSP/ATR repair report](BSP_AND_ATR_2026-11-05.md)
records the current endpoint repair, 30 further recovered races and the remaining
new-day index access limitation at 09:18 UTC.

The source coordinator now preserves other providers' coverage during a
targeted refresh. A busy feature store is reported as busy rather than missing
its signature. ATR cache and confirmed-absence writes now publish atomically.

The 07:20–07:25 UTC retry also recovered **496 BSP prices** and **11 ATR rows**.
All existing canonical records, cached observations, prices and timed trades
were preserved. ATR then blocked the next field, so its remaining queue stays
explicitly incomplete.

This follows the [identity repair](STAGE1_IDENTITY_2026-11-05.md) and
[original collection audit](STAGE1_DATA_PIPELINE_AUDIT_2026-11-04.md).

## One recovery interface, separate run and coverage records

Previously, `refresh_sources(..., sources=("atr",))` replaced `status.json`
with an ATR-only report. The readiness check then lost the previous SL, RTV,
BSP and FX results. Checking those sources again was unnecessary work.

The coordinator now writes:

- `data/source_refresh/last_run.json`: the exact requested operation, selected
  sources, elapsed time and outcomes; this remains the API's return value.
- `data/source_refresh/status.json`: combined health for all five required
  sources, retaining each source's `checked_at` and requested date range.

Untouched providers keep their original timestamps. The aggregate `checked_at`
is the oldest retained check; `updated_at` and `latest_run` identify the latest
operation. There is no combined elapsed-time claim for checks run at different
times. Counts describe their provider's recorded assessment range.

A successful subset does not erase another source's failure. Unknown providers
remain unverified. An older explicit backfill retains newer coverage and records
its outcome under `last_attempt`; the requested run still reports its own
failure. Pure SL historical recovery continues to use `backlog_status.json`.

After the installed ATR-only retry, all five provider results remain present.
RTV/FX retain their earlier checks around 00:29 UTC, while the latest run is
07:25 UTC. This does not claim that those untouched sources were checked again.

## Accurate feature-store diagnostics

The generation check previously caught every metadata-read exception and
treated the store as unsigned. A legitimate exclusive lock therefore produced
an instruction to rebuild an already signed history.

The check now reports `store_state="busy"` with a retry message, or
`store_state="unreadable"` with the access error. Genuine unsigned stores retain
their existing handling. Unverifiable models remain blocked from ordinary
promotion. Readiness exposes the store-access state separately from the model's
generation match. An actual lock/release regression test verifies that the
same signed store becomes readable again after release.

## Recovery results

| Source | Installed outcome |
|---|---|
| Sporting Life | 48 pages checked in 26.039s; no factual changes; one recent dormant field remains |
| BSP | 496 prices added; all 496 canonical runners / 47 finished fields on 4 November have BSP |
| BSP preservation | All 818,601 previous rows, price values and timed trades retained; total 819,097 rows |
| ATR | One race / 11 rows added; total 232,149 rows; all 232,138 previous rows unchanged |
| ATR identity check | All 11 names match the canonical 4 November Musselburgh 12:05 field |
| ATR follow-up | A 50-race budget stopped at the first blocked field: Nottingham, 4 November, 14:20; zero further races fetched |
| ATR queue | **3,966 pending**; retry after **08:25:46 UTC on 5 November** |

The three-source run took **67.659 seconds**. Its scope differs from earlier
14-day refreshes, so this is not a before/after speed comparison. The follow-up
ATR batch took 40.463 seconds including its normal cooldown/retry.
Blocked responses did not enter the confirmed-no-data list.

Canonical factual revision remains **2314**. The results CSV, feature-store
metadata and production pointer remain byte-for-byte unchanged. This pass
performed no model fit or promotion.

## Interrupted writes preserve accepted source data

ATR previously wrote directly over its accepted Parquet and confirmed-no-data
JSON files. The writer now uses the shared atomic publisher. Parquet is written
to a temporary file, its footer and row count checked, and its contents flushed
before replacement. JSON uses the existing flushed temporary-file publisher.
Failed writes remove temporary files and retain the previous accepted file.

Tests cover an exception after a partial write, a silently shortened Parquet
and an interrupted skip-list write. A separate private round trip of all
**232,149 cached ATR rows** preserves every value and dtype; the accepted cache's
content hash remains unchanged by that test.

## Feature-history boundary and remaining work

The saved feature history still contains 757,580 runners through 4 November.
Its canonical input-data check passes, but it now needs a feature refresh:

- The ATR input cache contains newly collected observations.
- The collector shares a fingerprinted module with ATR feature joins. The
  publication-only edit therefore changes current FE signature to
  `13fb7f7dd1e187e0`; the saved store carries `1355eccb0838ad9b`.

The 07:47 UTC deep readiness check reports both reasons correctly. The model
`20261104_235300_444862` still belongs to `bc357c18a5e5a261` and requires an
evaluated refit after compatible features are built. Source recovery should be
batched before the next history rebuild to avoid repeated full-history work.

A useful next boundary cleanup is separating ATR collection/publication from
its pure feature-join module. Cache-content signatures should continue to
invalidate changed input data; collection-only edits should not force feature
recomputation. The larger RTV learned-view boundary and optional pedigree queue
remain in the earlier audit. No full-history rebuild was run in this pass.

## Verification

The final source/publication/diagnostic regression run passed **139 tests**,
with one existing pandas concatenation warning. A separate serving/registry
run passed **52 tests**. These groups overlap. Source and test lint plus the
working-tree whitespace check pass.

Evidence is in `data/audit/stage1_followup_20261105/`: `before.json`, database and
cache backups, `recovery.json`, `atr_batch_001.json`, `verification.json`,
`atomic_roundtrip.json`, `final_tests.txt`, `contract_tests.txt` and
`readiness.json`. The prior full-suite baseline limitations remain documented
in the identity report.
