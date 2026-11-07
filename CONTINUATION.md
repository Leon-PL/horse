# Current handoff — 7 November 2026

Start with [README](README.md), the [project structure](docs/architecture/PROJECT_STRUCTURE.md),
and [project inventory](docs/PROJECT_INVENTORY.md). Earlier handoff notes are
preserved in [the archive](docs/archive/CONTINUATION_before_cleanup_2026-11-05.md).

## Layout cleanup, 7 November

- **No UI.** The Streamlit apps (`picks_app.py`, `training_app.py`), their helper
  modules and the app-only Matchbook/paper-trade/autotune modules were removed,
  with `streamlit` and `plotly` (23 packages left `uv.lock`; `.venv` re-synced).
  Use `train.py`, `src.run_store` and `scripts/build_lookahead.py`.
- **Research is separate.** 150 research scripts, 25 research-only `src`
  modules plus `src/research/`, and 52 research test files were removed;
  `scripts/` now holds 13 operating commands. New work goes in
  [research/](research/README.md), which pipeline code never imports
  (`tests/test_layout.py`). Everything removed is in git at `26d1b6f`, except
  `benchmark_atr_queue.py`, copied to `data/archive/removed_research_2026-11-07/`.
  No saved model references a removed module.
- **Unevaluated ATR candidates removed with it:** the new-tab and rich ATR
  families below (`src/research/atr_*_features.py`,
  `scripts/derive_atr_rich_features.py`) had no predictive evaluation yet. Their
  frozen outputs stay in `data/research/`; restore the code into `research/` from
  `26d1b6f` to continue.
- **Builder in `src/`, research names gone.** The store build moved from
  `scripts/build_feature_store_batched.py` (now only its command line) into
  `src/store_build.py` (`build_store`), and `src/` no longer imports `scripts/`
  (`tests/test_layout.py`). `prior_market_research.py` became
  `prior_market_features.py` and `audited_surface_features.py` became
  `surface_features.py`. These change `fe_signature` (now `72cca8f88f9f9ff6`)
  but no feature value; the store needed a rebuild already.

## Collection cleanup

**BHA feature candidates, 7 November:** normal FE now appends 161 numeric BHA
history candidates through `bha_features.add_bha_features`; `bha_events` handles
source-backed horse attribution. No model trials or selection fits were run.
The current derivation produced 462,785 observation rows, 275 withdrawals and
73,695 current horse states under `data/processed/bha_candidates/`.
Feature selection can use `candidate_feature_columns` to discover candidates
before the existing production allowlist. The full production store has not been
rebuilt. Code and BHA-input signatures now force the normal refresh when needed.
**Latest user decision, 7 November: use a two-day assumption.**
`config.BHA_FEATURE_AVAILABILITY_POLICY = "race_plus_48h"` makes report and
withdrawal features eligible 48 hours after their source event. This supersedes
the earlier capture-only choice. Actual source capture times remain unchanged;
original publication and later amendment times are unverified. The selected
snapshot's latest captured text is therefore used retrospectively under this
explicit assumption. Same-race and future-start exclusions still apply. See
[definitions](docs/research/BHA_FEATURE_CANDIDATES_2026-11-07.md) and
`data/audit/bha_two_day_20261107/`. Historical candidate rows are materialised by
`python -m src.bha_features --historical`; the full production store still needs
its normal refresh. The completed two-day regeneration contains 758,304 runner
rows; 493,672 have eligible report history in their latest five starts. All 40
timing/signature checks passed. Collection continues independently.

**BHA regular ingestion, 7 November:** `bha` is now in the normal source plan.
`ingestion.bha_stewards` owns a durable SQLite day/revision/document queue;
`ingestion.bha_reports` retains all report fields and extracts full PDFs. Accepted
tables are `bha/reports`, `bha/documents` and `bha/race_notes`, with source-native
identities and actual observation times. The 8,463-report CSV is preserved.
The first coordinated run added four reports/full PDFs for 6 November and 29
race-note blocks; new model features have not been enabled. Full historical
catch-up uses the same collector; check `data/bha/status.json` and its worker
lock before starting anything. Read [the operating guide](docs/operations/BHA_REGULAR_INGESTION_2026-11-07.md)
and `data/audit/bha_regular_ingestion_20261107/` for evidence. Its normalisation
benchmark improved 12.3 times; cached native refresh used zero HTTP requests.
The full source coordinator also performs other-provider publication, so those
native timings are not an end-to-end ingestion benchmark.

**Shared source publication, 7 November:** `src/source_contracts.py`,
`src/source_dataset.py` and `src/ingestion/publication.py` provide the common
accepted-data boundary. Ordinary processing/builds freeze source versions;
training records its manifest including Betfair. ATR promotion reads completed
worker jobs without editing its pinned code or original cache. Source
fingerprints cover link/timing sidecars, and ATR history respects its assumed
48-hour availability delay with explicit support and age. This changes feature
definitions: rebuild/refit compatibility is still required. The selected model
is unchanged. See [operations and qualification](docs/operations/SOURCE_PUBLICATION_2026-11-07.md).
Use `python -m src.ingestion --publish-sources` for local adoption and
`--source-status` for accepted-data quality, separately from worker progress.

**Fallback paths removed, 7 November (later):** processing, features, prices and CLV now read
sources only through a selected dataset. The legacy routes are gone: the name-joined ATR merge,
RTV snapshot name joins, native-cache path fallbacks, `compute_clv(bsp_path=...)` and the
unversioned "direct-input" frames. A frame without a dataset ID uses the last committed dataset
or stops; a store without a recorded dataset rebuilds. On the frozen qualified dataset the
production outputs (processed frame, 1,233-column store, carried state, prices by ID and by
name, CLV) were identical before and after. Tests now run against an explicit fixture dataset
(`tests/conftest.py`). Merged with the layout cleanup and the BHA history
features above. BHA now reads through the same helpers: it honours a dataset's own `root` and
stops when no dataset is selected, instead of returning empty history (on the committed dataset
it selects the same files as before). `fe_signature()` is now `5eaf59189052d233`; the rebuild and
refit already owed above cover it. The installed store records no dataset, so live prediction refuses
it until that rebuild. See [the qualification](docs/operations/SOURCE_PUBLICATION_2026-11-07.md#fallback-paths-removed).

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
The ATR worker now runs as `python -m src.ingestion.atr_rebuild --request-delay 2`.
The production store was rebuilt on 6 November at `fe_signature()` `457c83789e6b29ab` and data
signature `7b99cce04504a09d`. The promoted model predates it and still needs a refit.
The published history is now `data/raw/race_results.parquet`, at the types declared in
`RESULTS_SNAPSHOT_COLUMNS`. The old CSV is in `data/backups/race_results_csv_20261106/`. The
rebuild moved `weather_precip_prev3` to the database's exact values and eight feature columns
slightly, because running totals now resume exactly across chunks; see the
[follow-up report](docs/operations/RESULTS_SNAPSHOT_AND_TEST_FIXES_2026-11-06.md).

**ATR comments, 7 November:** a runner with no ATR comment now gives missing comment tags, not
zeros, so the `atr_cmt_*` rates skip that run. This moves `fe_signature()` to `33491388b5ad5e6e`.
The store was deliberately not rebuilt and still holds the zeros, and the production input
`atr_cmt_front_rate_5_vs_field` was trained on them, so apply the rebuild and a refit together.

**Old build route retired, 7 November:** `featured_dataset.get_featured_dataset` and
`FeatureStore.append` are removed. Training refreshes the store only through
`src.store_refresh.get_training_dataset`, and the empty-store and pre-flip errors now name
`python -m scripts.build_feature_store_batched`. The append, parity, transaction and
live-feature tests run against `append_current_store`. This moves `fe_signature()` again, to
`60990991f5a30931`, superseding the value above. The store was deliberately not rebuilt while
scraping continues, so `project_status.py` reports it stale and the next training run will
rebuild it (about 14 minutes). Do that rebuild and the refit together.

**ATR speed and IDs, 7 November:** the worker now loads saved daily indices, persists all
exact IDs from each discovered day immediately, uses indexed queue selection, groups exports,
and recovers unexported fields after shutdown. Eleven duplicated IDs affected races an hour
apart; the reviewed plan clears 13 conflicting hints while preserving every saved runner.
Browser-read source receipts add 63 previously missing IDs across 15 September 2020 and
4 November 2026. The normal browser works, but direct HTTP and isolated automated browsers
remain challenged; no unattended browser dependency was deployed. Known-ID collection can
continue while discovery respects its cooldown. Remaining missing dates are not yet recovered.
Read [the current ID and speed report](docs/operations/ATR_SPEED_AND_IDS_2026-11-07.md) and
`data/audit/atr_speed_ids_20261107/adoption.json`. Use live status and qualification receipts
for progress and throughput. All five performance tabs and parsing rules remain unchanged.
The reviewed adoption preserved all 43,857 saved rows exactly, removed every duplicate ID,
and left 34,197 missing IDs. The ten-race live check added 117 runners in 121.63 seconds
(about 296 races/hour), with no failures, reaching 43,974 rows before continuous restart.
This is a short sample, not a full-backfill ETA. Runtime launch details are saved in the
same audit directory and in `data/atr_cache/rebuild_v2/launch_options.json`.

The synthetic race generator, sample collection API and unused collector shim
are removed. Processing no longer writes an unused global Parquet by default.
See the [cleanup report](docs/operations/PRE_FEATURE_CLEANUP_2026-11-05.md).

## Betfair historical downloads

**Runner identity recovery, 6 November:** the 22:50 publication failure is the
Gowran Park Sierra Nevada pair (2022-06-12, 15:50). Matching now uses exchange
selection IDs within the race, with unique-name fallback. PLACE retains IDs and
the publisher upgrades older staged/published parses from exact retained raw
revisions. A repeated winner alias and an unidentified legacy PLACE observation
are preserved under `history/identity_review/` before replacement. Real replay
produces all 15 runners in each cache, with distinct prices for both Sierra Nevadas.
Ambiguous name-only model joins stay unknown; SL IDs are not exchange IDs.
See [recovery and evidence](docs/operations/BETFAIR_RUNNER_IDENTITY_2026-11-06.md)
and `data/audit/betfair_identity_20261106/` for publication/restart receipts. The
backfill also pins `src/betfair.py`, `src/runner_identity.py` and ingestion `storage.py`;
do not edit its seven collector/matcher/storage dependencies while it runs. All
4,988 stopped markets published (17,157 total); older PLACE parses gained IDs.
A transient Windows progress-file lock also stopped consolidation; atomic writes
now retry that lock before failing safely, and consolidation resumes its checkpoint.
ATR's separate pins are maintained by the reviewed ATR deployments; the 7 November ID
module brings that set to five. Betfair's seven collector pins are unaffected.

**Import speed, 6 November:** both WIN and PLACE are retained. Grouped durable
checkpoints, bounded parallel work and lightweight progress counters replace
per-market saves/counts. Identical real-market output was verified; the controlled
writer comparison was 5.24 times faster. A real 256-market batch imported in
118.263 seconds, with zero failures, and all four price caches published, bringing
the queue to 2,179 published markets. The continuous 5,000-market worker restarted
at 22:07 UTC with current pins; a subsequent 98-second local replay sample achieved
7.96 markets/second with zero failures. Allow one to two days provisionally for
the full backfill, including network gaps and archive work. Archive scans and publication remain
batch overheads; do not use the writer benchmark as the total backfill ETA.
See [measurements and verification](docs/operations/BETFAIR_IMPORT_SPEED_2026-11-06.md)
and `data/audit/betfair_performance_20261106/`. Temporary test queues were removed;
the profiler `scripts/benchmark_betfair_ingestion.py` was retired on 7 November (commit `26d1b6f`). ATR pins match.

**Consolidated main path, 6 November:** the default source plan now uses `betfair`
for full Historical BASIC WIN/PLACE data. Public BSP fetching and both separate
manual extraction commands are removed. Its 4,472 raw files and parsed volume/VWAP
research dataset are frozen at `data/research/betfair_public_legacy/`; existing
research readers were redirected. A real main-coordinator run completed all 92
markets for 31 October with no failures. The 17 original tar files (10.10 GB) are
being deduplicated into `data/betfair/history/archive_payloads.sqlite`; each is
removed only after verification and a durable checkpoint. The resumable historical
worker covers 2020-09-15–2026-10-31, reuses retained payloads while consolidation
continues, and then downloads remaining gaps. Use the live `archive_import_status`,
`backfill_status` and `status` JSON files under `data/betfair/history/`; do not infer
completion from launch. See [consolidation details](docs/operations/BETFAIR_CONSOLIDATION_2026-11-06.md).

The old `2020.tar` has an incomplete final member. Consolidation now preserves
damaged originals in `data/betfair/history/quarantine/`, records the issue, and
continues with intact archives; the API fills missing markets. Read-back hash
checks still precede deletion of verified originals. Recovery checks cover
truncated sources, corrupt nonzero headers and legitimate interleaved zero padding.

The Historical Data API downloader is implemented in `src/ingestion/betfair_history.py`.
It retains full BASIC WIN/PLACE files, resumes its SQLite queue and merges missing
observations into the existing price caches. Normal source refreshes opt in once
an account is connected. The Data-page setup controls were removed with the
Streamlit apps on 2026-11-07; the setup guide now gives the Python and CLI
routes for connection and catch-up. Login settings use Windows
account encryption; certificate login does not require a paid Live App Key.

**Connected and verified, 6 November:** the user uploaded the public certificate
and connected the account. Windows-encrypted settings are saved locally; normal
source refreshes can now fetch historical markets. Live checks confirmed BASIC
access from July 2015 through October 2026. Fixed GET for `GetMyData` (POST returned
405), accepted the current `/xds_nfs/edp_processed/BASIC/` file layout, and added
specific connection diagnostics. Five real markets from 31 October downloaded,
validated and published, with zero failures. The main refresh subsequently completed
the remaining 87 markets from that day. Conflicting accepted BSP values were
preserved for review. Initial setup evidence: `data/audit/betfair_connection_20261106/`;
main-path qualification: `data/audit/betfair_consolidation_20261106/`.
The certificate/key and encrypted connection remain in `data/betfair/auth/`.
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
