# Collection cleanup and project organization — 5 November 2026

## Changes

The application no longer contains `data_collector_sample.py`, its invented
horse/jockey pools or its random race generator. The redundant `data_collector`
interface and `source="sample"` mode were removed. Real callers now use the
collection coordinator or Sporting Life collector directly. Unsupported source
plans are rejected before any files or source status are written.

Six implementations now live under `src/ingestion/`: source coordination,
auxiliary refreshes, raw/atomic storage, RTV generation storage, Sporting Life
and Racing TV. `python -m src.ingestion` is the normal collection command.
Small aliases preserve the previous import paths and supported CLIs, with the
same module objects and global state. The unused `collect_scraped_data` wrapper
was removed. Source refresh still precedes feature construction.

Racecard caching now uses the configured data directory rather than deriving a
project root from the collector's file location. The BSP receipt continues to
hash its real dependency files after the move. The feature-dependency guard
now traverses nested packages and `from src import ...` imports, and explicitly
classifies collection modules. No feature calculations were changed.

`process_data` no longer writes `data/processed/processed_races.parquet` by
default. That unused export contained five runner records and was unrelated
to the maintained historical store. Explicit `save=True` remains available for
callers that deliberately need an export. Normal feature and prediction callers
already use processing in memory.

## Removed stored data

The cleanup removed **2,007,710,539 bytes (about 2.0 GB)** across 61 verified
targets:

- Obsolete June global processing/feature caches with no current readers.
- Rebuildable `model_lab_slim*.parquet` research caches. Recreate them with
  `scripts/model_lab.py --build-slim-store` when needed.
- The unused five-runner processing export, a standalone offline-test CSV,
  old CatBoost diagnostic output and bytecode for the deleted generator.
- Temporary test databases, fake source stores and copied feature stores
  under explicitly identified `test_tmp*` / `pytest_*` directories.

Targets were resolved and checked inside the workspace before deletion; linked
paths were rejected. Old test folders belonged to different Windows execution
identities, so two passes were needed. The second pass removed all 45 remaining
targets; no permission-blocked folders remain in this cleanup's target list.
Each pass has a path/size/reason manifest and removal receipt.

Canonical results, source exports and retained raw responses remain. BSP,
weather, stewards, sales and other source datasets still have processing or
research consumers; absence from the currently selected predictor list alone
does not make them redundant. Saved runs and qualification evidence remain.
Test fixtures and the real all-weather surface classifier also remain.

## ATR preservation

`attheraces_scraper.py`, `atr_features.py` and `atr_rebuild.py` are byte-for-byte
unchanged from the pre-cleanup copies. Their running worker pins those hashes.
The worker continued collecting during the move; its SQLite queue, raw payloads,
daily exports and earlier accepted cache were preserved. No worker restart,
queue reset or replacement-cache publication was needed.

## Organization

Documentation is grouped into `architecture`, `operations`, `research` and
`archive`. Relative links were updated. The root README, project outline and
handoff now describe current responsibilities and commands; their longer
previous versions are archived. The [structure review](../architecture/PROJECT_STRUCTURE.md)
explains the four stages, storage ownership and retained compatibility paths.
The [script guide](../../scripts/README.md) groups maintenance and research tools.

Feature/model modules and research scripts retain stable import paths because
saved models and recorded experiments use them. They are indexed by responsibility;
there is no duplicate implementation in the collection compatibility paths.

## Verification and evidence

**307 distinct tests passed** across the two final runs (291 and 23 tests,
with seven shared cases). Targeted lint passed; all 433 Python files parsed;
local documentation links resolved. The structural comparison verified 143
unchanged definitions in the moved code, with only the documented dependency
path/cache path changes and unused-wrapper removal.

Focused checks cover shared import identity, configured cache locations,
unsupported-source rejection, the package-aware feature dependency graph,
source recovery, database synchronization, ATR parsing/publication/rebuild,
RTV features, serving and readiness. Existing collection CLIs and the training
CLI were checked using `--help`, without source requests or model fitting.

The final test totals and checks are recorded in
`data/audit/pre_feature_cleanup_20261105/verification.xml` and `verification_summary.json`.
Before-files, moves, the removal manifests and ATR preservation hashes are in
the same audit directory. No feature rebuild, fit or promotion was run; the
previously stale store/model still need the normal refresh/refit process.
