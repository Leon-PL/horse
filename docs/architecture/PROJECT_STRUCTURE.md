# Project structure

Reviewed and organized on 5 November 2026; collection consolidated on 6 November.
The application is one four-stage
pipeline with a separate research toolkit. Stage one has a dedicated package;
the other stages retain stable module names used by saved models and experiments.

```text
horse/
  picks_app.py, training_app.py, train.py    user entry points
  config.py                               shared settings and paths
  src/
    ingestion/                            collection and source publication
      __main__.py                         coordinated collection command
      pipeline.py                         source plan, coverage, identity links
      results.py                          resumable Sporting Life results ledger
      sporting_life.py                    Sporting Life pages and racecards
      pedigree.py                         horse profile collection
      racing_tv.py                        RTV raw payloads and parsing
      rtv_observations.py                 RTV run, card and text materialisation
      auxiliary.py                        bounded ATR/Betfair refresh operations
      attheraces.py                       ATR requests, parsing and normal backfill
      atr_rebuild.py                      durable ATR backfill worker
      atr_tabs.py                         ATR contract: tabs, columns, cache reader
      atr_jumps.py                        ATR jumps data
      betfair_archives.py                 verified deduplication and replay of existing downloads
      betfair_streams.py                  BASIC market-stream parsers
      betfair_history.py                  full historical market downloads and import
      betfair_credentials.py              Windows-encrypted account connection
      betfair_history_ui.py               Data-page connection and catch-up controls
      fx.py                               ECB reference rates
      weather.py, bha_stewards.py         explicit research collectors
      storage.py                          locks, raw payloads, atomic writes
      rtv_storage.py                      accepted RTV generations and partitions
    database.py, identity_schema.py        canonical result store and published snapshot
    betfair.py                            shared price matching
    data_processor.py                     load, clean and encode accepted history
    feature_*.py, fe_*.py, ratings.py       feature definitions and store
    training.py, model.py, run_store.py    fitting and saved-run registry
    live_prediction.py                    current-card prediction inputs
  scripts/                                maintenance and research commands
  tests/                                  fixtures and regression checks
  feature_sets/                           versioned model input selections
  reference/                              versioned corrections and lookup rules
  docs/
    architecture/, operations/              current structure and maintenance
    research/, archive/                     evidence and earlier plans
  data/                                   local source, derived and research data
  models/                                 selected-run pointer and model artifacts
  _research_snapshots/                     isolated historical implementations
```

## Boundaries

| Stage | Owner | Output and rule |
|---|---|---|
| 1. Collect and aggregate | `ingestion.pipeline.refresh_sources` | Canonical SL records plus independently retained RTV, ATR, BSP and FX observations. No sample-data mode. |
| Hand-off | `database.publish_results_snapshot` | `data/raw/race_results.parquet`, every column at the type declared in `RESULTS_SNAPSHOT_COLUMNS`, with a receipt naming the database revision. |
| Prepare for features | `data_processor.process_data` | Reads that snapshot, or `load_database_history` for a check of the live database, joins pedigree, cleans and encodes. Clean runner frame in memory; nothing is written. |
| 2. Build features | `store_refresh` / `FeatureStore` / `featured_dataset` | Historical features and incremental state. `store_refresh` appends new dates or calls the batched builder; `featured_dataset` fingerprints the code and accepted source content. |
| 3. Train and evaluate | `training`, `model`, evaluation modules, `run_store` | Candidate models and measured results; explicit selected-run pointer. |
| 4. Predict | `live_prediction`, `picks_app` | Today's cards joined to compatible historical state. |

Collection can run without building features or fitting a model. The coordinator
collects canonical results before identity joins, then runs independent providers
with bounded concurrency. It publishes coverage and failures rather than silently
substituting generated runners or replacing successful generations with failures.

The rule is one owner per step: every regular collector is in `src/ingestion/`,
storage is in `src/database.py`, and cleaning is in `src/data_processor.py`.
`scripts/` holds the commands that run those collectors over local archives; the
application imports nothing from a collection script.

ATR follows the same rule. `src/ingestion/atr_tabs.py` is its contract: the columns,
the cache location and the cache reader, which collection and feature building both
use. The join into feature building, `merge_atr_sectionals`, is in `src/atr_features.py`,
so editing the join does not stop the worker. The worker's queue and raw response
store are separate from the accepted feature cache.

`src/trainer_locations.py`, `scripts/fetch_bha_going.py` and
`scripts/fetch_forecast_previous_runs.py` are one-off research collectors outside the
normal plan.

## Compatibility

The old `source_pipeline`, `source_auxiliary`, `source_storage`, `rtv_storage`,
`data_scraper`, `rtv_scraper`, `attheraces_scraper` and `atr_rebuild` names were removed
on 6 November, along with the extra commands that reached the same refresh.
`python -m src.ingestion` is the collection command, `python -m src.ingestion.racing_tv`
holds Racing TV's own tools, and `python -m src.ingestion.atr_rebuild` runs the ATR
backfill worker.

The feature dependency check now follows packages and `from src import ...`
imports. Collection modules are explicitly classified as source producers;
their accepted observations are covered by source/data fingerprints. The ATR
join and its column contract remain in the feature-code signature.

## Stored data

| Location | Role | Retention |
|---|---|---|
| `data/races.db` | Canonical results, identities, source queues and pedigree | Keep; repairs and exports flow through its writer |
| `data/raw/race_results.parquet` + source receipt | Published processing input | Keep synchronized with the database |
| `data/sl_cache`, `data/rtv_cache`, `data/atr_cache` | Raw responses, source observations, manifests and checkpoints | Keep for replay, coverage and recovery |
| `data/atr_cache/rebuild_v2` | Running fresh ATR backfill | Keep; never reset during ordinary cleanup |
| `data/processed/feature_store` | Derived history and state | Keep; rebuild through the feature owner |
| Other source Parquets in `data/processed` | BSP, sales, going and research inputs | Keep while referenced by collection, evaluation or research |
| `data/runs`, `models`, `data/audit`, `data/research` | Models and recorded evidence | Keep reproducibility and qualification records |
| Test temporary folders, unused global processing caches | Disposable or rebuildable outputs | Removed with a path/size/reason manifest |

Raw and derived source observations share some historical folder names. They
are documented by ownership rather than moved while consumers or backfills are
active. There is no benefit in forcing downloads or invalidating a qualified
dataset solely to rename its directory.

## Evaluation of the structure

The main problem was overlapping collection interfaces, scattered source
infrastructure and current instructions mixed with old reports. The dedicated
ingestion package, single collection command and grouped documentation resolve
those navigation problems.

The flat feature/model modules and research scripts contain cross-imports and
saved-model class paths. Bulk renaming them would add migration risk without
improving collection. Their ownership is indexed in the [inventory](../PROJECT_INVENTORY.md)
and [script guide](../../scripts/README.md). Reorganize an individual family when
its implementation changes, with compatibility and signature checks alongside it.

Unused optional research data is distinguishable from obsolete cache data:
weather, stewards, sales and public market files have research consumers even
when absent from the selected model. They remain available. Artificial race
generation has been removed from the application; isolated test fixtures and
the real all-weather surface classification remain necessary.
