# ATR rich-data rebuild and shutdown recovery — 5 November 2026

**Current update, 6 November, 12:08 UTC:** the main collector and worker now
check five performance tabs, adding Jumps Data and Distance Ran. The reviewed
upgrade preserved 33,918 runner payloads and queued 4,224 previously processed
fields for the missing endpoints, reusing retained raw HTML. The queue now has
per-tab coverage and a source navigation drift check. Final verification covers
272 distinct checks. Following final reporting qualification at 12:30, current
worker PID 16432, launcher 6748, pacing unchanged at two seconds.
Read [the five-tab operating guide](../archive/operations/ATR_TAB_COVERAGE_2026-11-06.md) for current
scope, pins, missingness rules and evidence. Older three-tab coverage, pin lists
and throughput figures below describe the earlier generation.

**6 November, 10:36 UTC:** a controlled pause allowed correction of legacy
efficiency feature semantics. The parser is byte-identical and all 31,041
checkpointed rows were preserved; the worker resumed at two-second pacing.
The guard now pins the **collector and worker**, excluding `atr_features.py`,
which raw collection does not execute. A regression test confirms feature edits
do not stop collection while parser/worker edits still do. Feature calculations
remain guarded by the normal feature signature. All 178 relevant checks passed.
The [efficiency/consistency usage audit](../archive/research/ATR_EFFICIENCY_AND_USAGE_2026-11-06.md)
and `data/audit/atr_efficiency_20261106/` record the change, queue backup and resume.

**6 November recovery, 07:16 UTC:** a feature-performance edit to shared ATR
modules triggered the code-change safeguard at 00:02. The worker had preserved
**2,267 completed fields / 19,062 runner records**. Review confirmed source
extraction is unchanged (excluding the separate feature-merge helpers), and the
backfill worker itself is byte-identical. All **80 scraper/recovery tests passed**;
SQLite integrity is `ok`. A consistent queue backup precedes the audited adoption
of reviewed hashes. The resumed worker (PID 10864) accepted a new field and raised
the count to 19,068, confirming progress after restart. Pacing remains two seconds
plus jitter. Queue state, retries, retained source responses and the old accepted
cache were preserved. Evidence is in `data/audit/atr_resume_20261106/`.

New-day discovery still receives client challenges and uses its existing cooldown;
known-ID sectional collection continues. This recovery did not change extraction,
feature calculations or fitted models.

**18:57 performance update:** the live worker now uses a **2-second** base gap
(plus jitter). The measured 20-race trial ran at **414 races/hour**, versus a
15-minute baseline of **332/hour**: about **25% higher active throughput**.
All three tabs and the same data checks are retained. This suggests about six
days of uninterrupted successful collection for the full remaining scope,
excluding access blocks; it is not a completion guarantee.

A single HTTP 503 occurred during the trial. Its cause cannot be inferred from
that sample. The exact endpoint subsequently returned healthy HTTP 200 data
twice. The original policy paused every race for an hour after that one fault;
this is now corrected. Temporary 500/502/503/504 or network failures briefly
back off globally (initially 60 seconds, increasing on consecutive failures)
and defer the individual race for longer (initially five minutes, increasing
on repeat failures). Other races can proceed after the short global delay.
Genuine throttling, forbidden responses and bot challenges retain their long
backoff. Server `Retry-After` values are honoured by the active HTTP transport.

The operational change was qualified with 108 relevant regressions and a
41-test focused follow-up (overlap), plus an AST comparison proving that
parsing, field validation and storage functions were unchanged. Existing
records were retained and the controlled runtime fingerprint update is in
`retry_policy_deployment.json`. The failed race was then successfully saved.
Current FE signature is **f91cd05955e05228**. The worker resumed at 18:55 with
PID **1340** and continues in the background; current launch options are in
`data/atr_cache/rebuild_v2/launch_options.json`.

The corrected collector and a fresh full-history backfill are installed.
The worker restarted at **17:56 UTC** after the computer shut down. At 17:57
it had advanced to **46 resolved fields / 381 saved runner records**, proving
that it resumed from the durable checkpoint. Counts continue to change in
`data/atr_cache/rebuild_v2/status.json`.

## Corrected extraction

The public times, tools and stride tabs now retain:

- Split times, aligned cumulative times, pace flags and per-section Front
  markers; missing cumulative values retain their position in the array.
- Consecutive sectional speeds separately from top speed, instantaneous final
  speed and 0–30 mph acceleration **seconds**. Previously the latter could be
  mistaken for closing speed. Label-aware feature derivation also protects
  reads of older labelled caches.
- Stride length **and frequency** curves, their independent markers and source
  peak-average reference lines. Charts are identified by their labels.
- Individual finishing-speed percentage, efficiency score/grade, energy
  distribution, optimum profiles, comments, placings, draw and native IDs.
- Race-level early pace, race finishing-speed percentage, opening section
  distance and the published pace-bar colour stops.
- Labelled extra metrics and complete chart attributes, plus retained HTML,
  actual HTTP status and response hashes for offline diagnosis/reparsing.

Every added raw outcome column belongs to `ALL_ATR_COLS` and is removed by
the existing historical feature boundary. No new predictor family or model
fit is part of this work.

## Identity and source coverage

Native IDs from the old cache are **lookup hints**, not accepted telemetry.
The rebuilt records are parsed from source HTML and matched to canonical
runner identities. Unexpected runners, duplicate identities and contradictory
source dates/off-times are rejected. Dated silk filenames supply an additional
date/local-off-time cross-check where the source exposes them.

The pilot established that ATR omits some nonfinishers from its sectional
tables. The collector keeps the available runners and records expected/provided
runner counts and missing names. Missing nonfinishers are labelled
`finishers_only`; other tracking omissions are labelled `partial_tracking`.
A subset requires independent source date/off-time evidence. This preserves
usable observations without inventing missing telemetry or discarding a whole
valid race. Source omissions remain visible as `data_partial` jobs.

An unrecognised successful HTML page is retryable, never permanent no-data.
Blocked later tabs cannot publish an incompletely downloaded field.

## A genuinely fresh reconstruction

The queue covers **62,432 UK races from 15 September 2020 through 4 November
2026**. The canonical snapshot contains 80,536 races; 18,104 races at the
existing known-unsupported Irish venues are excluded. The prior cache supplies
**28,186 native-ID hints**. No old parsed observations or legacy no-data
decisions were imported into the new dataset. The 379 older absence decisions
are therefore reconsidered within the rebuild's eligible scope.

The old **232,397-row** accepted cache remains separate and byte-identical to
its backup. A replacement should be installed after coverage and identity
qualification; an empty or partial rebuild must not replace useful history.
Backups and qualification evidence are in `data/audit/atr_rebuild_20261105/`.

The worker uses SQLite transactions to commit each race. Daily Parquet files
are derived checkpoints, not the authoritative progress record. They export
in small batches, avoiding repeated rewrites of the entire historical cache.
Normal ATR refreshes recognise the active worker and report its progress
instead of starting another collector.

## Retry and continuation behaviour

The process now uses a two-second minimum request gap plus jitter. A challenged
HTML discovery page defers uncached-ID discovery for an hour while collection
can continue for known IDs. A block on the sectional endpoints triggers global
backoff. Unknown layouts are retained for diagnosis and retried later rather
than classified as absent. Races younger than 48 hours remain pending.

The worker keeps running and heartbeats during cooldowns. It stops if its
pinned collector code changes, so different parser generations cannot silently
mix. This is a local background process: shutdown stops execution. Starting
the worker again resumes its committed queue; it does not start over.

```powershell
.venv\Scripts\python.exe -m src.ingestion.atr_rebuild --request-delay 2
Get-Content data/atr_cache/rebuild_v2/status.json
```

Create `data/atr_cache/rebuild_v2/STOP` for an orderly stop. Remove that explicit
stop marker before resuming. Do not reset the queue or blindly update pinned
hashes after editing the collector; requalify affected source records first.

The earlier roughly seven-hour estimate concerned only the 3,936 recent gaps.
This full reconstruction is about sixteen times larger and uses a more
conservative request delay. It is a multi-day job; uncached-day challenges
prevent a reliable completion promise.

## Recovering missing IDs from saved pages

When the HTML cooldown expires, the worker gives eligible daily results lookups
first use of that window. Its status records the most recent automatic attempt,
including the date, outcome and number of IDs added.

If ATR challenges the automatic lookup, save a dated results or replay archive
page as HTML after it loads in a browser. Import one or more files through the
same command. The running worker picks up new receipts within about a minute.

```powershell
.venv\Scripts\python.exe -m src.ingestion.atr_ids --import-results 'C:\path\to\results.html' 'C:\path\to\replays.html'
```

Replay pages supply IDs through the replay controls that accompany each racecard
link. A replay import fills gaps alongside an existing results receipt. The
worker keeps conflicting IDs for review and validates runner identity when it
collects the performance tabs.

## Verified installation and recovery

- **152 relevant tests pass**, including current-page fixtures, units, source
  identity, legitimate tracking gaps, interrupted publication, retries,
  exclusive collection, resume and feature-generation coverage. Lint passes.
- The qualified pilot accepted **31 fields / 253 source runner records**.
  Thirty fields were reparsed from retained source HTML; a historical
  Doncaster field was downloaded through the corrected endpoints.
- The first background run reached **37 resolved fields / 295 records** before
  shutdown. Recovery checked both queue and raw-response SQLite databases:
  `PRAGMA quick_check` returned `ok`, and collector hashes still matched.
- Consistent SQLite backups and previous logs were retained under
  `data/audit/atr_rebuild_20261105/resume_20261105_175547/`. Both derived daily
  exports were regenerated from committed rows before restart.
- Restarted worker PID at 17:56: **17748**. Its subsequent heartbeat and rising
  row count verify actual progress after the launching shell returned.
- The accepted cache SHA-256 remains
  `0a09320eadc18e94e8e038852492e35a368f938da421da989359476f85d54f07`.
  No old accepted observations were lost during the shutdown or restart.

Evidence: `tests_qualified.txt`, `pilot_qualification.json`,
`latest_recovery.json`, `resume_launch.json` and `resume_source_health.json`
in the audit directory. The source-health coordinator now reports the running
fresh backfill; other providers retain their own previous check times.

The initial FE signature after the extraction corrections was **7ee6994aa3026400**;
the later operational change is recorded above. No feature
build, training run or promotion was performed by this collection task.
