# Project structure

Reviewed and organized on 5 November 2026. The application is one four-stage
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
      sporting_life.py                    real results and racecards
      racing_tv.py                        RTV raw payloads and parsing
      auxiliary.py                        bounded ATR/BSP refresh operations
      storage.py                          locks, raw payloads, atomic writes
      rtv_storage.py                      accepted RTV generations and partitions
      betfair_history.py                  full historical market downloads and import
      betfair_credentials.py              Windows-encrypted account connection
      betfair_history_ui.py               Data-page connection and catch-up controls
    database.py, identity_schema.py        canonical result store
    attheraces_scraper.py, atr_rebuild.py   ATR collector and durable backfill
    pedigree_backfill.py, betfair.py       shared source-specific integrations
    data_processor.py                     clean accepted history
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
| Prepare for features | `data_processor.process_data` | Clean runner frame in memory; no redundant global export by default. |
| 2. Build features | `featured_dataset` / `FeatureStore` | Historical features, incremental state and signatures. Accepted source content is fingerprinted. |
| 3. Train and evaluate | `training`, `model`, evaluation modules, `run_store` | Candidate models and measured results; explicit selected-run pointer. |
| 4. Predict | `live_prediction`, `picks_app` | Today's cards joined to compatible historical state. |

Collection can run without building features or fitting a model. The coordinator
collects canonical results before identity joins, then runs independent providers
with bounded concurrency. It publishes coverage and failures rather than silently
substituting generated runners or replacing successful generations with failures.

ATR's existing module combines collection with the sectional join consumed by
feature building. Its file and the resumable worker stay at their stable paths:
the active backfill pins them. Its queue and raw response store are separate from
the accepted feature cache. Pedigree and price integrations also have consumers
outside collection, so they retain their shared module paths.

## Compatibility

The old `source_pipeline`, `source_auxiliary`, `source_storage`, `rtv_storage`,
`data_scraper` and `rtv_scraper` modules contain only compatibility aliases to
the implementations above. Both import paths resolve to the same module object,
so patched globals, caches and locks are shared. Existing supported commands
still work. New collection code should import `src.ingestion` modules.

The feature dependency check now follows packages and `from src import ...`
imports. Collection modules are explicitly classified as source producers;
their accepted observations are covered by source/data fingerprints. The ATR
join remains in the feature-code signature.

## Stored data

| Location | Role | Retention |
|---|---|---|
| `data/races.db` | Canonical results, identities, source queues and pedigree | Keep; repairs and exports flow through its writer |
| `data/raw/race_results.csv` + source receipt | Published processing input | Keep synchronized with the database |
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
those navigation problems. Small compatibility modules carry no duplicate logic.

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
