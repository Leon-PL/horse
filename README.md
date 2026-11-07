# Horse racing probabilities

A local application that collects UK and Irish racing data, builds historical
features, trains LightGBM win/place models and presents race probabilities.

The project follows four stages: **collect and aggregate → build features →
train and evaluate → predict**. See the [project structure](docs/architecture/PROJECT_STRUCTURE.md)
for ownership and data flow, the [inventory](docs/PROJECT_INVENTORY.md) for tools,
and [CONTINUATION](CONTINUATION.md) for the current handoff.

## Current state

The synthetic race generator and sample collection interface have been removed.
Real-source collection lives in `src/ingestion/`, with one command:
`python -m src.ingestion`. Storage is in `src/database.py` and cleaning is in
`src/data_processor.py`. [Consolidation report](docs/operations/PRE_FEATURE_CONSOLIDATION_2026-11-06.md).
The database still exports `data/raw/race_results.parquet`, with declared types.
Ordinary processing, feature builds and training now select immutable accepted
source versions through a dataset manifest under `data/sources/`. Results and
pedigree are captured together; ATR promotion, RTV, FX and evaluation prices use
the same publication contract. [Data flow](Racing%20Data%20Flow.md) and
[publication operations](docs/operations/SOURCE_PUBLICATION_2026-11-07.md).

Full Betfair historical market downloads are the main refresh's Betfair source
(`--sources betfair`), including win/place BSP, timed prices and market changes.
The account is connected under **Data → Historical exchange data**. Existing
downloads are consolidated and replayed locally; missing markets are fetched
automatically. Public BSP fetching and separate manual extraction commands are
retired. The public volume/VWAP research dataset is frozen under
`data/research/betfair_public_legacy/`. Historical markets arrive after settlement;
the collector conservatively waits six days before requesting them.
[Setup and access requirements](docs/operations/BETFAIR_HISTORY_AUTOMATION.md).
[Consolidation and active catch-up](docs/operations/BETFAIR_CONSOLIDATION_2026-11-06.md).

The fresh ATR historical backfill is running independently at a two-second base
request interval plus jitter. Progress is in
`data/atr_cache/rebuild_v2/status.json`. It resumes from its SQLite checkpoint;
do not start a second worker or delete that directory.
[Backfill operation and qualification](docs/operations/ATR_REBUILD_2026-11-05.md).

The installed feature store and model need refresh/refitting after the source
and identity repairs. This cleanup does not fit or promote a replacement model.
Check actual compatibility before using forecasts:

```powershell
.venv\Scripts\python.exe scripts/project_status.py
```

## Collect and aggregate

```powershell
.venv\Scripts\python.exe -m src.ingestion --sync 14 --results-only
```

This refreshes Sporting Life results, Racing TV observations, bounded ATR
recovery, full Betfair history, currency references and BHA stewards' reports.
BHA includes full PDFs, going/rails, non-runner reasons and race notes; see
[BHA collection and catch-up](docs/operations/BHA_REGULAR_INGESTION_2026-11-07.md).
The refresh retains raw evidence,
checks identities and publishes source coverage. Independent providers run
with bounded concurrency after canonical results are available. A partial
report returns a nonzero exit code.

`--enrich` drains a bounded pedigree queue; `--weather` explicitly collects
research weather. `--start` and `--end` select an historical range.
`--sl-backlog-max-races` bounds older result recovery and `--atr-max-races`
bounds the normal ATR queue. The separate full ATR backfill uses its own durable
queue. Use `--help` for the complete options.

Source maintenance is also available on the **Data** page of the maintenance
app. See the [source operating notes](docs/operations/STAGE1_IMPLEMENTATION_2026-11-04.md)
and [BSP/ATR provenance](docs/operations/BSP_AND_ATR_2026-11-05.md).

## Build features, train and evaluate

Refresh the historical dataset without fitting a model:

```powershell
.venv\Scripts\python.exe -c "from src.store_refresh import get_training_dataset; get_training_dataset()"
```

The store appends valid newer dates and rebuilds when earlier observations or
feature definitions change. A full rebuild can take substantial time. Feature
signatures and source fingerprints prevent silent reuse of incompatible history.
Processing prepares data in memory instead of writing an unused global export.

Fit and evaluate a candidate from current local data:

```powershell
.venv\Scripts\python.exe train.py --no-refresh --threads 2
```

Review scores, dates and feature compatibility with `src.run_store`
(`list_runs`, `feature_generation_status`), then explicitly promote a chosen run
with `train.py --promote` or `src.run_store.promote(run_id)`. Saving a candidate does not replace the selected model.
`train.py --fit-as-of YYYY-MM-DD` fits through a settled cutoff without a held-out
score; `--validation-report` reuses and checks an earlier evaluated recipe.

## Forecasts and saved runs

The Streamlit apps (`picks_app.py`, `training_app.py`) were removed on
2026-11-07; there is no UI. `scripts/build_lookahead.py` prepares dated live
features and `src.live_prediction` scores them against the promoted run.
Saved runs are listed, renamed, deleted and pruned through `src.run_store`
(`list_runs`, `rename_run`, `delete_run`, `prune_runs`). Serving requires
compatible data, feature history and fitted model weights.

## Research and development

Read the [negative-results ledger](docs/NEGATIVE_RESULTS.md),
[research progress](docs/MARKET_PERFORMANCE_PROGRESS.md) and
[evaluation protocol](docs/operations/FORECAST_BENCHMARK_PROTOCOL.md) before
repeating experiments. Numbered reports are in [docs/research](docs/research/);
earlier plans and handoffs are in [docs/archive](docs/archive/).

Pipeline code is in `src/` and operating commands in `scripts/`. New experiments,
screens and audits go in [research/](research/README.md), which pipeline code
never imports. The earlier research code was retired on 2026-11-07 and is in
git at commit `26d1b6f`.

Use Python 3.14 from the project `.venv`. `pyproject.toml` owns dependencies,
`uv.lock` locks them, and `requirements.lock` exports the core runtime. The
development environment has separate tooling:

```powershell
.\scripts\tooling.ps1 setup
.\scripts\tooling.ps1 test
.\scripts\tooling.ps1 validate
.\scripts\tooling.ps1 lint
```

See [development tools](docs/operations/DEVELOPMENT_TOOLS.md) and the
[script guide](scripts/README.md). Local `data/` and `models/` are excluded from
Git; dependency installation does not recreate the installed data or model.
Read [CLAUDE.md](CLAUDE.md) before changing feature or settlement calculations.

The [previous README](docs/archive/README_before_cleanup_2026-11-05.md) retains
the detailed installation and experiment chronology.
