# Current handoff — 6 November 2026

Start with [README](README.md), the [project structure](docs/architecture/PROJECT_STRUCTURE.md),
and [project inventory](docs/PROJECT_INVENTORY.md). Earlier handoff notes are
preserved in [the archive](docs/archive/CONTINUATION_before_cleanup_2026-11-05.md).

## Collection cleanup

**Temporary-code cleanup, 6 November:** completed diagnosis, migration, restart
and qualification helpers from the recent collection work have been removed.
Audit results, code snapshots and database recovery checkpoints remain in place.
Duplicate ATR HTML now uses the canonical fixtures; the reusable collection and
research commands are unchanged. The cleanup inventory is recorded under
`data/audit/session_cleanup_20261106/`. The active ATR worker was left running.

Real-source collection now lives in `src/ingestion/`. The entry point is
`python -m src.ingestion --sync 14`. The older collection names and commands were
removed on 6 November; see the
[consolidation report](docs/operations/PRE_FEATURE_CONSOLIDATION_2026-11-06.md).
The synthetic race generator, sample collection API and unused collector shim
are removed. Processing no longer writes an unused global Parquet by default.
See the [cleanup report](docs/operations/PRE_FEATURE_CLEANUP_2026-11-05.md).

## Betfair historical downloads

The Historical Data API downloader is implemented in `src/ingestion/betfair_history.py`.
It retains full BASIC WIN/PLACE files, resumes its SQLite queue and merges missing
observations into the existing price caches. Normal source refreshes opt in once
an account is connected. **Data → Historical exchange data** supplies certificate
download, account connection and catch-up controls. Login settings use Windows
account encryption; certificate login does not require a paid Live App Key.

The local certificate/key pair is prepared at `data/betfair/auth/client.crt` and
`client.key`. The public certificate still needs uploading to the user's Betfair
account, followed by login through the app. No account credentials were available;
an authenticated online transfer has not yet been verified. Local archived market
compatibility and encryption under the normal Windows user were verified.
See [setup details](docs/operations/BETFAIR_HISTORY_AUTOMATION.md).

## ATR backfill

**New-tab feature derivations, 6 November:** `src/research/atr_new_tabs_features.py`
implements 11 completed-run descriptors, **17 prior predictors and four quality
inputs** from Jumps Data and Distance Ran. The read-only generator is
`scripts/derive_atr_rich_features.py --family new-tabs`. Its frozen output under
`data/research/atr_new_tabs_20261106/` matches 4,298 source runners to 67,233
canonical appearances, with predictors on 8,294 targets. Grade/course-par inputs
respect dated revisions as well as the 48-hour assumption; nonfinishers and
untracked races remain missing slots. Repeatability/change support is still very
sparse. Fifty-seven checks passed, plus exact replay from frozen inputs. These
are unselected research features, not ordinary model inputs; no fit/build was
run. The four pinned collector files were preserved and the worker continued.
See [feature definitions and coverage](docs/research/ATR_NEW_TAB_FEATURES_2026-11-06.md).

**Five-tab integration, 6 November; latest resumption 12:30 UTC:** the main scraper and historical
worker now collect **times, tools, stride, jumpsdata and distanceran**. The
broader audit found Distance Ran as well as the previously missed Jumps Data.
An hourly source-page inventory flags unknown tabs; blocked pages remain visibly
unverified. The seven pre-race tabs were separately inventoried and are not
claimed as historical snapshots collected by this result backfill.

The controlled upgrade preserved all **33,918 runner payloads** and all 62,432
jobs, reopening **4,224 completed fields** for the missing endpoints. Per-tab
receipts track real signal counts, missing runners and errors. Retained HTML is
replayed before missing tabs are fetched. **272 distinct checks passed**; live
qualification matched 13 runners and preserved existing core values. Worker
PID **16432**, launcher **6748**, still at two-second pacing. No feature build
or model fit was run; new observations are in the rebuilding generation.
At 12:31 the main queue had 87 runner records with jumps measurements and 825
with distance, including the three jumps rows from the user's Chepstow example.
Legacy receipts without signal counts are explicitly `legacy_received` until
rechecked. Source signal detection and tab coverage share one definition.

**Current raw-worker pins:** `ingestion/attheraces.py`, `ingestion/atr_rebuild.py`,
`ingestion/atr_tabs.py` and `ingestion/atr_jumps.py`. Stop and review before
editing these four files. Earlier two-file pin lists below are superseded;
`atr_features.py` remains outside this guard. Evidence and the consistent backup
are in `data/audit/atr_tab_coverage_20261106/`. See [current tab coverage and catch-up](docs/operations/ATR_TAB_COVERAGE_2026-11-06.md).

**Efficiency correction, 6 November (10:36 UTC):** native efficiency is an error
score. `atr_features.py` now uses lower-is-better, retains valid scores above 40
and masks unrated zero placeholders without profiles. An audit found 12,929
otherwise valid values above the old cap among 30,562 source rows. The grade,
profile irregularity, cadence rhythm and prior-five-run consistency now have
separate research derivations. All 178 relevant checks passed; no feature-store
rebuild or model fit was run. See [the usage audit](docs/research/ATR_EFFICIENCY_AND_USAGE_2026-11-06.md).

The collector paused cleanly at 31,041 rows and resumed from that checkpoint;
worker PID **19156**, launcher **9628**. No raw observations were changed or reset.
**The worker now pins only `attheraces_scraper.py` and `atr_rebuild.py`.** It does
not execute feature derivation, so `atr_features.py` was removed from that guard.
Feature calculations remain protected by the normal FE signature. Collector/queue
edits still stop the worker. Evidence and the consistent queue backup are under
`data/audit/atr_efficiency_20261106/`; source parsing is byte-identical.

**New ATR research candidates, 6 November:** seven richer per-run measurements
and their prior-only last-three-race medians are implemented in
`src/research/atr_rich_features.py`, outside the normal recipe. The pilot matched
29,184 fresh ATR records to canonical IDs and derived prior inputs using all
757,580 canonical appearances, with a 48-hour availability assumption. Seventeen
formula/identity/timing tests passed. Outputs are in `data/research/atr_rich_20261106/`.
No predictive evaluation, ordinary feature build or model fit was run. Worker
files were preserved; the backfill continued. See
[the feature report](docs/research/ATR_RICH_FEATURES_2026-11-06.md).

**6 November, 07:16 UTC:** resumed after the worker stopped at 00:02 when shared
ATR files were edited for feature-building performance. The source-extraction
AST is unchanged apart from feature-merge helpers, and the worker is byte-identical.
Eighty scraper/recovery tests passed; SQLite integrity is `ok`. A consistent queue
backup and audited fingerprint adoption preserved all 19,062 previously saved rows.
Restarted worker PID **10864** (launcher 1704), still at a two-second base gap.
The first new accepted field increased the saved count to 19,068. New-day discovery
still encounters challenges; collection continues for known native IDs. Evidence:
`data/audit/atr_resume_20261106/`. No extraction code, feature build or fitted model
was changed by this recovery. Use the live status file for subsequent counts.

The full backfill continues with `--request-delay 2` plus jitter. Its collector
and worker files are pinned. Do not reset the queue or replace
the accepted cache before qualification.

- Progress: `data/atr_cache/rebuild_v2/status.json`.
- Durable queue and collected rows: `data/atr_cache/rebuild_v2/rebuild.sqlite`.
- Runtime options: `data/atr_cache/rebuild_v2/launch_options.json`.
- Details: [ATR rebuild](docs/operations/ATR_REBUILD_2026-11-05.md).

Use the project Python, `.venv\Scripts\python.exe` (3.14). The global Python
has a different dependency stack. Check the status file before starting any
worker; an existing worker owns the queue.

## Feature/model state

No feature rebuild, model fit or promotion is part of this cleanup. The installed
feature store and model already needed refresh/refitting after the ATR repairs.
The feature signature also records the updated dependency declarations. Run
`scripts/project_status.py` for actual compatibility before using predictions;
do not copy old fingerprint values into metadata to bypass the checks.

## Repository state

The working tree contains substantial earlier source and research changes.
They were retained. The cleanup's before-files, move manifests, removal receipt
and verification outputs are under `data/audit/pre_feature_cleanup_20261105/`.
