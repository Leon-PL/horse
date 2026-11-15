**French and UAE races, 14 November:** [collection and rating-only use](operations/FOREIGN_RACES.md).
`src/ingestion/foreign_form.py` collects complete fields; `src/ratings.py` sweeps them without adding rows.

# Project inventory

**Shared source publication, 7 November:** [contract, operations and qualification](operations/SOURCE_PUBLICATION_2026-11-07.md).
`src/source_contracts.py` owns identity/timing/quality rules; `src/source_dataset.py`
publishes immutable versions and selects manifests; `src/ingestion/publication.py`
adapts native stores and promotes validated ATR work. `src/source_health.py`
separates attempts from consumer readiness, and `src/source_prices.py` selects
evaluation prices. Use the normal collection CLI's `--publish-sources` and
`--source-status` options; builds can replay a recorded `--dataset-manifest`.

**ATR speed and IDs, 7 November:** [repair, preservation and access report](operations/ATR_SPEED_AND_IDS_2026-11-07.md).
`src/ingestion/atr_ids.py` records exact daily ID evidence and audits legacy hints.
The historical worker recovers saved indices, selects jobs through partial indices
and batches crash-safe daily exports. `scripts/benchmark_atr_queue.py` measured local
selection with exact old/new results (retired with the research code on 2026-11-07;
a copy is in `data/archive/removed_research_2026-11-07/`). Browser-readable pages still challenge the
unattended client; the report distinguishes recovered IDs from remaining access gaps.

**Store build speed, 6 November:** [timings and equivalence report](operations/STORE_BUILD_SPEED_2026-11-06.md).
A full store build takes 14.1 minutes, down from 92, with every feature kept and
the store byte-identical to the old build. On long histories it runs in three
processes. `compare_feature_stores.py` (retired 2026-11-07, commit `26d1b6f`)
proved two stores match. One trial build differed in a single carried value, and
the cause was not established; the report says what was ruled out.

**Betfair historical automation, 6 November:** [setup guide](operations/BETFAIR_HISTORY_AUTOMATION.md).
Certificate login (connected once; the Data page was removed on 2026-11-07) automatically fetches the full
BASIC market streams through the Historical Data API. Raw files, resumable queue
and timed-price imports live in `src/ingestion/betfair_history.py`; saved connection
details use Windows account encryption. The account is connected and the full
WIN/PLACE feed is part of the main source plan. Public BSP collection and manual
extraction commands are retired; unique volume/VWAP history is retained for research.
Imports use grouped durable checkpoints and bounded parallel work. The
`benchmark_betfair_ingestion.py` profiler that measured them was retired on
2026-11-07 (commit `26d1b6f`).

**Collection cleanup, 5 November:** real-source implementations are grouped in
[`src/ingestion`](../src/ingestion/), with `python -m src.ingestion` as the
collection command. The fake-data generator and unused collector interface
were removed. See the [current structure](architecture/PROJECT_STRUCTURE.md),
[cleanup report](operations/PRE_FEATURE_CLEANUP_2026-11-05.md),
[consolidation report](operations/PRE_FEATURE_CONSOLIDATION_2026-11-06.md),
[results snapshot report](operations/RESULTS_SNAPSHOT_AND_TEST_FIXES_2026-11-06.md), and
[documentation index](README.md). Earlier reports below describe their dated checkpoints.

**ATR reconstruction, 5 November:** [qualified collector and full backfill](operations/ATR_REBUILD_2026-11-05.md).
`src/ingestion/atr_rebuild.py` commits verified fields to a fresh SQLite queue and exports
daily Parquets. Progress is in `data/atr_cache/rebuild_v2/status.json`. The worker
resumed after shutdown; the old accepted cache remains preserved pending
replacement qualification.

**BSP/ATR, 5 November (09:18 UTC):** [source provenance and repair](operations/BSP_AND_ATR_2026-11-05.md).
ATR uses the current Razor endpoints, saved per-day native race IDs and retained
raw responses. Thirty races / 248 records were recovered, with every earlier
cache value preserved. New-day ID discovery still encounters challenges; the
normal collector retains 3,936 pending races and its retry cooldown.

**Source health, 5 November:** [targeted recovery and atomic publication](operations/STAGE1_SOURCE_HEALTH_2026-11-05.md).
`last_run.json` records the requested source plan; `status.json` retains combined
coverage and individual check times. ATR publication is atomic, busy-store
diagnostics are explicit, BSP covers 4 November and ATR recovery remains partial.
The changed inputs/collector fingerprint require a subsequent feature refresh.

**Runner identity, 5 November:** [migration and preservation report](operations/STAGE1_IDENTITY_2026-11-05.md).
Canonical storage, repair writers, rating joins and settlement now retain
different source horse IDs sharing a display name. The final historical SL
identity case is resolved; compatible full histories are rebuilt and installed.

**Source collection, 4 November:** [implementation and operating notes](operations/STAGE1_IMPLEMENTATION_2026-11-04.md)
document the unified source plan, revision-aware results publication, resumable
ATR recovery, safe RTV generations, raw SL retention and offline replay. Normal
collection uses `src.ingestion.pipeline.refresh_sources`; its report is visible in
the Data page and readiness check.

[Historical queue recovery](operations/STAGE1_BACKLOG_2026-11-04.md) documents
`--sl-backlog-max-races`, the Data page's Older result recovery control,
source-response fingerprints and the original horse-identity finding. Its last
historical review case was resolved by the 5 November migration above.

**Racing TV features and repairs, 30 October:** [source pipeline](../src/ingestion/pipeline.py)
connects collection, comparison and numeric observations to training and apps. [RTV features](../src/rtv_features.py) compute 26 historical and
four timestamped pre-race inputs; the deployed selection remains 154 after an
uncertain fixed trial. [Repairs and current operation](research/RTV_FEATURES_REPAIRS_044.md).

The [README](../README.md) is the operating guide. This inventory identifies
current entry points, shared implementation, research tools and historical
records. The038 probability, source-context and recorder additions were reviewed on **29 October 2026**;
the wider inventory was reviewed on28 October.

**Tooling additions, 29 October:** [development tools](operations/DEVELOPMENT_TOOLS.md)
records the isolated uv environment, Pandera audit, Hypothesis tests and DuckDB
measurement. Entry point: [tooling.ps1](../scripts/tooling.ps1).
The added [data contracts](../src/data_contracts.py),
[validation command](../scripts/validate_data_contracts.py) and
[query benchmark](../scripts/benchmark_data_queries.py) are opt-in development
tools outside the pinned forecast implementation. Dependency ownership is now
[pyproject.toml](../pyproject.toml) and [uv.lock](../uv.lock).

**Layout since 2026-11-07:** `src/` holds pipeline code only, `scripts/` holds
the operating commands listed below, and new experiments go in
[`research/`](../research/README.md), which core code never imports. The research
scripts, research-only modules and their tests were removed; see
[Retired research code](#retired-research-code).

## Entry points and local artifacts

| Item | Role |
|---|---|
| [train.py](../train.py) | CLI wrapper around the training core. The Streamlit apps (`picks_app.py`, `training_app.py`) were removed on 2026-11-07. |
| [config.py](../config.py) | Shared paths and defaults; recipe/feature settings participate in fingerprints. |
| [feature_sets/win.txt](../feature_sets/win.txt) | Versioned normal input selection. |
| [feature_sets/research_win_with_atr.txt](../feature_sets/research_win_with_atr.txt) | Separate sectional-family research selection. |
| [feature_sets/win_rtv_derived.txt](../feature_sets/win_rtv_derived.txt) | win.txt plus the RTV-derived family and Timeform card; the 046 native-check candidate, not the normal selection. |
| [reference/](../reference) | Versioned race-kind corrections, venue currency rules and dated exchange rates used by normal features. |
| `data/races.db` | Local results database; canonical sporting records. |
| `data/raw/` | Exported source snapshots consumed by processing; database repairs must be reflected in the export. |
| `data/processed/feature_store/` | Historical processed/features/state files and provenance. |
| `data/processed/betfair_bsp.parquet` | Normal historical exchange-price teaching cache. |
| `data/runs/` | Candidate models, metadata and recorded evaluations. |
| `models/production.json` | Explicit selected-run pointer. |
| `data/forecasts/forward/` | Pinned studies and immutable issued-forecast evidence. |
| `data/audit/` | Local research outputs, manifests, source snapshots and qualification receipts. |
| `_research_snapshots/` | Separate historical research implementations; not the normal training entry point. |
| [tests/](../tests) | Correctness, parity, provenance and evaluation guards. |

The local data/model folders are excluded from Git. A source checkout alone
cannot reproduce the installed state. Read the runtime report for current
dates and compatibility rather than inferring them from a file's timestamp.

## Maintenance commands

Use the README for ordinary history refresh, training and app startup.
The following are additional deliberate operations, run from the project root.

Explicit full feature rebuild, without model fitting:

```powershell
.venv\Scripts\python.exe -m scripts.build_feature_store_batched --chunk-rows 60000
```

Normal training already uses that bounded builder when a rebuild is necessary.
A full rebuild is unnecessary for an ordinary valid append of later dates.

Export an evaluated run's recipe, then fit it through a settled cutoff:

```powershell
.venv\Scripts\python.exe scripts/create_validation_report.py --run RUN_ID --output data/validation_reports/RUN_ID.json
.venv\Scripts\python.exe train.py --no-refresh --fit-as-of YYYY-MM-DD --validation-report data/validation_reports/RUN_ID.json --threads 2
```

Replace the placeholders with an evaluated run and settled cutoff. Recipe
compatibility is enforced; an older incompatible certificate may be refused.
The deployment refit has no new held-out score. Neither command promotes.

The forward recorder, its pinned studies and the scripts that created,
recorded, assessed or transitioned them were removed on 30 October 2026
(experimentation phase, no live model). They remain in git history.

## Source-module map

All source modules present at this inventory are grouped below. This is a responsibility map, not an instruction to move or delete files.

| Area | Modules | Responsibility |
|---|---|---|
| Collection and publication | [pipeline](../src/ingestion/pipeline.py), [auxiliary](../src/ingestion/auxiliary.py), [storage](../src/ingestion/storage.py), [RTV storage](../src/ingestion/rtv_storage.py) | Coordinated real-source collection, coverage, retained responses and atomic publication. |
| Results and racecards | [results ledger](../src/ingestion/results.py), [Sporting Life](../src/ingestion/sporting_life.py), [database](../src/database.py), [data_processor](../src/data_processor.py) | Collect, store, export and clean source records. Sporting Life commentary and verdicts go to the append-only `runner_text_observations` table (racecard and result copies, with the time seen), never to `results`. |
| Auxiliary sources | [pedigree](../src/ingestion/pedigree.py), [At The Races](../src/ingestion/attheraces.py), [Racing TV](../src/ingestion/racing_tv.py), [RTV observations](../src/ingestion/rtv_observations.py), [exchange rates](../src/ingestion/fx.py), [full Betfair history](../src/ingestion/betfair_history.py), [market streams](../src/ingestion/betfair_streams.py), [weather](../src/ingestion/weather.py) | Optional enrichment. Weather is stored as raw data and does not supply a current feature stage. The Racing TV scraper ([guide](operations/RTV_SCRAPER.md)) collects results, racecards and RaceiQ data. `src.rtv_features` reads its Timeform card and run history; `ingestion.pipeline.load_matched_text` gives its text (Timeform comment and verdict, pace and draw notes, close-ups) per SL runner for research. [rtv_text](../src/rtv_text.py) turns the Timeform comment and verdict into the comment score (`rtv_text_logit`, `rtv_text_prob`, `rtv_verdict_rank`; production inputs since 056). Quarterly walk-forward models, each with a scale calibrated on its latest training races, live in `data/rtv_cache/text_models/format_2`; history scores in `feature_text_scores.parquet` record their format, and a format change rescores all history. Runners with no text score as missing. Live cards are scored from the pre-race snapshot. `ingestion.rtv_observations` keeps them current with the rest of the RTV cache. |
| Canonical feature definitions | [feature_engineer](../src/feature_engineer.py), [feature_semantics](../src/feature_semantics.py), [result_margins](../src/result_margins.py), [result_status](../src/result_status.py), [surface_features](../src/surface_features.py), [prior_market_features](../src/prior_market_features.py), [prize_currency_features](../src/prize_currency_features.py), [verified_race_metadata](../src/verified_race_metadata.py), [track_config](../src/track_config.py), [atr_features](../src/atr_features.py) | Normal calculations and source corrections. `surface_features.pedigree_surface_history` and `prize_currency_features.native_prize_history` are reference calculations used only by their tests, not feature stages. |
| Historical state and storage | [ratings](../src/ratings.py), [trueskill_kernel](../src/trueskill_kernel.py), [fe_incremental](../src/fe_incremental.py), [feature_store](../src/feature_store.py), [featured_dataset](../src/featured_dataset.py), [store_refresh](../src/store_refresh.py), [store_build](../src/store_build.py), [feature_inputs](../src/feature_inputs.py), [rating_reuse](../src/rating_reuse.py), [parquet_io](../src/parquet_io.py) | Historical state, transactional files, provenance and bounded refresh/build support. `store_refresh` is the one refresh route; it appends new dates or calls `store_build.build_store` for the bounded-memory full rebuild. `featured_dataset` holds the signatures it checks. |
| Training and validation | [model](../src/model.py), [race_objectives](../src/race_objectives.py), [training](../src/training.py), [probability_calibration](../src/probability_calibration.py), [deployment_validation](../src/deployment_validation.py) | Shared training core, optional tuning/calibration and deployment-recipe checks. |
| Model selection and serving | [run_store](../src/run_store.py), [live_prediction](../src/live_prediction.py) | Saved runs, explicit production pointer and current-card feature generation. |
| Scoring and shared rules | [wf_eval](../src/wf_eval.py), [feature_selection](../src/feature_selection.py), [forecast_evaluation](../src/forecast_evaluation.py), [bet_analysis](../src/bet_analysis.py), [backtester](../src/backtester.py), [clv](../src/clv.py), [bet_settlement](../src/bet_settlement.py), [each_way](../src/each_way.py), [quality_control](../src/quality_control.py), [utils](../src/utils.py) | Purged research evaluation, walk-forward input selection, complete-field scoring, settlement, identity joins and app-facing simulation. The research and app backtest interfaces have distinct jobs. |
| Market data | [betfair](../src/betfair.py), [source_prices](../src/source_prices.py) | Historical teaching/evaluation prices. The app-only Matchbook client/signals, paper-trade store and autotune modules were removed with the apps on 2026-11-07. |
| Package support | [__init__](../src/__init__.py) | Package marker. The Streamlit UI helpers (`app_helpers`) were removed with the apps on 2026-11-07. |
| Result repair libraries | [verified_result_repair](../src/verified_result_repair.py), [result_amendment_repair](../src/result_amendment_repair.py) | Transactional repairs of source-verified fields and placings, with original-row receipts and rollback. Nothing calls them automatically; they are kept as data-integrity tooling. |
| BHA stewards | [Collector](../src/ingestion/bha_stewards.py), [document parser](../src/ingestion/bha_reports.py), [operating guide](operations/BHA_REGULAR_INGESTION_2026-11-07.md) | Regular source `bha`; retained JSON/PDFs, durable days/revisions, full document sections and native race notes. Published through accepted-source contracts; historical catch-up uses the same collector. New feature use remains a separate evaluation. |
| BHA feature candidates | [Attribution](../src/bha_events.py), [history features](../src/bha_features.py), [definitions](research/BHA_FEATURE_CANDIDATES_2026-11-07.md) | 161 numeric candidates in normal FE; prior histories, coverage and withdrawal features under the explicit 48-hour availability assumption, with recorded capture times retained. Available to full candidate selection; no individual models trained or production allowlist changes. |
| Research weather collector | [weather](../src/ingestion/weather.py) | No current feature stage reads it; race-level weather also has a source publication adapter. |

`prior_market_features.py` and `surface_features.py` were called
`prior_market_research.py` and `audited_surface_features.py` until 2026-11-07.

## Commands (scripts/)

Run from the project root with `.venv\Scripts\python.exe`. These are deliberate
actions, not a batch; repairs and backfills can rewrite history and trigger a
full feature rebuild.

| File | Purpose and scope |
|---|---|
| [project_status.py](../scripts/project_status.py) | Read-only readiness check across database, features, runs and production. Full check compares database contents; `--quick` skips that comparison. |
| [build_feature_store_batched.py](../scripts/build_feature_store_batched.py) | Explicit full history rebuild with bounded memory and a staged commit: a command over `src.store_build.build_store`, which normal training also calls through `src/store_refresh.py`. Adds a separate store root, validated rating reuse and dataset replay. |
| [build_lookahead.py](../scripts/build_lookahead.py) | Live feature cache for today's and upcoming cards through the shared live feature path. First deletes date-stamped caches older than `CACHE_TTL_DAYS` (`--no-cleanup` keeps them). |
| [create_validation_report.py](../scripts/create_validation_report.py) | Export an evaluated run's recipe certificate for a later deployment refit (`train.py --fit-as-of ... --validation-report`). |
| [backfill_missing_races.py](../scripts/backfill_missing_races.py) | Collect missing races identified against exchange data; writes results data and republishes the results snapshot. |
| [reconcile_result_fields.py](../scripts/reconcile_result_fields.py) | Source reconciliation that can repair canonical result fields; inspect its planned changes and evidence. |
| [repair_overfull_fields.py](../scripts/repair_overfull_fields.py) | Remove non-runners stored as runners (more rows than `num_runners`). Re-scrapes each race and replaces the validated field, or deletes only rows the page marks NONRUNNER/WITHDRAWN; backs up to data/backups first, then republishes `data/raw/race_results.parquet`. Dry run unless `--apply`. |
| [backfill_rtv_text.py](../scripts/backfill_rtv_text.py) | Rebuild the Timeform comment-score history: collects RTV card text from 2020-09, fits any missing quarterly text models (`--refit` refits all) and rescores every runner. Normal RTV refreshes do this incrementally. |
| [build_connection_aliases.py](../scripts/build_connection_aliases.py) | Rebuild `reference/connection_aliases.json`, which gives each jockey and trainer one name across Sporting Life renamings. Changing the file changes `fe_signature`. Report only unless `--write`. |
| [validate_data_contracts.py](../scripts/validate_data_contracts.py) | Read-only Pandera audit of settled fields using the existing eligibility rules (`tooling.ps1 validate`). |
| [benchmark_data_queries.py](../scripts/benchmark_data_queries.py) | Compare equivalent filtered Parquet reads in bounded workers (`tooling.ps1 benchmark`); fits no model. |
| [setup_betfair_history.ps1](../scripts/setup_betfair_history.ps1) | `-CreateCertificate` creates the local Betfair login certificate. |
| [tooling.ps1](../scripts/tooling.ps1) | `venv` rebuilds `.venv` from `uv.lock`; `setup` builds the tooling env; `lint`, `test`, `validate`, `benchmark`. |

Pedigree backfill is `python -m src.ingestion --enrich`; Betfair catch-up is
`python -m src.ingestion.betfair_history --start YYYY-MM-DD --drain`.

## Retired research code

On 2026-11-07 the research and audit code was removed so that `src/` and
`scripts/` hold only what the pipeline runs: 150 research scripts (benchmarks,
screens, audits, diagnostics, experiment controllers and one-off installers),
25 research-only modules plus the `src/research` package, and 52 test files that
covered only them. No saved model references a removed module.

- Every removed file except one is in git at commit `26d1b6f`:
  `git show 26d1b6f:scripts/<name>.py`, or
  `git checkout 26d1b6f -- scripts/<name>.py` to bring one back.
- `scripts/benchmark_atr_queue.py` was never committed; its copy is in
  `data/archive/removed_research_2026-11-07/`.
- Results stay in [docs/research/](research/), the
  [negative-results ledger](NEGATIVE_RESULTS.md) and `data/research/`,
  `data/audit/`. Their script and module names refer to the commit above.
- New experiments go in [`research/`](../research/README.md).

## Documentation map

The operating guide owns current procedures. Runtime state comes from
`scripts/project_status.py`, `models/production.json`, run metadata and the
selected study's protocol. Reports preserve the state and measurements at their
own checkpoints; their old model IDs are not instructions to switch production.

| Role | Documents |
|---|---|
| Start here | [README](../README.md), this inventory |
| Development correctness rules | [CLAUDE](../CLAUDE.md) |
| Session handoff | [CONTINUATION](../CONTINUATION.md); transient working context, not a second operating guide |
| Research standards | [Forecast benchmark protocol](operations/FORECAST_BENCHMARK_PROTOCOL.md), [forward recording contract](research/FORWARD_FORECASTS_006.md), [manual timing contract](research/TIMED_RECORDING_008.md); older example study IDs remain historical |
| Evidence index and prior failures | [Implementation progress](MARKET_PERFORMANCE_PROGRESS.md), [negative-results ledger](NEGATIVE_RESULTS.md) |
| Latest installed correction evidence at this review | [026 canonical corrections](research/CANONICAL_FEATURE_REPAIRS_026.md), [027 teaching-data refresh](research/BSP_TEACHER_REFRESH_027.md) |
| Historical architecture/design | [Project outline](archive/PROJECT_OUTLINE_before_cleanup_2026-11-05.md), [FE taxonomy](architecture/FE_TAXONOMY.md), [FE refactor plan](architecture/FE_REFACTOR_PLAN.md) |
| Historical plans and briefs | [Action plan](archive/ACTION_PLAN.md), [roadmap](archive/ROADMAP.md), [exploration plan](archive/EXPLORATION_PLAN.md), [next-agent brief](archive/NEXT_AGENT_BRIEF.md), [market-performance plan](archive/MARKET_PERFORMANCE_PLAN.md), [prediction-improvement plan](research/PREDICTION_IMPROVEMENT_PLAN_014.md), [feature-repair plan](research/FEATURE_REPAIR_PLAN_018.md) |
| Historical session records | [Session log](archive/SESSION_LOG.md), [archived handoff](archive/CONTINUATION_20261028T144650281Z.md) |

### Checkpoint reports

These retain their original evidence, scope and conclusions. They are not
current operating procedures or an implicit work queue.

- Collection feasibility: [ATR feasibility](operations/ATR_FEASIBILITY_2026-10-17.md),
  [ATR benchmark](operations/ATR_BENCHMARK_2026-10-17.md),
  [sample observations](atr_feasibility_sample_2026-10-17.json),
  [ATR scraper audit](operations/ATR_SCRAPER_AUDIT_2026-10-29.md) and
  [Racing TV scraper audit](operations/RTV_SCRAPER_AUDIT_2026-10-29.md).
- Market comparisons: [001 baseline](research/MARKET_COMPARISON_001.md),
  [002 sectionals](research/MARKET_COMPARISON_002.md),
  [003 race objective](research/MARKET_COMPARISON_003.md),
  [004 refit cadence](research/MARKET_COMPARISON_004.md).
- Deployment and planning: [005 fitting](research/DEPLOYMENT_FITTING_005.md),
  [007 confirmation feasibility and earlier pilot](research/CONFIRMATION_FEASIBILITY_007.md).
- Probability experiments: [009 pace](research/MODEL_PREDICTIONS_009.md),
  [010 calibration](research/MODEL_CALIBRATION_010.md),
  [011 tail smoothing](research/MODEL_TAIL_SMOOTHING_011.md),
  [012 history](research/MODEL_HISTORY_012.md),
  [013 blend](research/MODEL_HISTORY_BLEND_013.md).
- Data/features/models: [014 source findings](research/DATA_REPAIR_FINDINGS_014.md),
  [014 margin repair](research/MODEL_MARGIN_REPAIR_014.md),
  [015/016 history and teaching](research/MODEL_HISTORY_TEACHER_015_016.md),
  [017 feature review](research/FEATURE_REVIEW_017.md),
  [018 feature review](research/FEATURE_REVIEW_018.md),
  [019 calibration choice](research/MODEL_CALIBRATION_CHOICE_019.md),
  [020 prize currency](research/PRIZE_CURRENCY_020.md),
  [021/022 capacity](research/MODEL_CAPACITY_021_022.md),
  [024 remaining gap](research/REMAINING_MARKET_GAP_024.md),
  [025 race kinds](research/RACE_TYPE_AUDIT_025.md).
- [Project audit](archive/PROJECT_AUDIT.md): the earlier probability and pipeline checkpoint.

## Consolidation boundaries

The 2026-11-07 pass removed the Streamlit apps and the research code. It did
not move pipeline files, alter defaults, rebuild data, fit or promote models.

The next code consolidation should address these concrete boundaries:

1. **One historical-feature refresh implementation. Done 2026-11-07.**
   `src.store_refresh.get_training_dataset` is the only route. The older
   `src.featured_dataset.get_featured_dataset` and `FeatureStore.append` were
   removed; the build-vs-append, full-build parity, transaction and
   live-feature tests now run against `append_current_store`.
   `FeatureStore.build` remains as the in-memory reference those tests use.
2. **Shared implementations belong behind command wrappers. Done 2026-11-07.**
   The store builder moved from `scripts/build_feature_store_batched.py` into
   `src/store_build.py`, so `src/` no longer imports `scripts/`
   (`tests/test_layout.py` checks this). The Betfair parsers moved this way on
   6 November. Both moves, and the two renamed feature modules, changed
   `fe_signature`.
3. **Research is separate. Done 2026-11-07.** Research code was removed from
   `src/`, `scripts/` and `tests/`; new work goes in `research/`, which core
   code never imports (`tests/test_layout.py`).
4. **Plan fingerprint transitions.**
   `src.featured_dataset.fe_signature` hashes registered source paths and
   each source's AST (comments, docstrings and formatting excluded since
   30 October 2026). Moving a registered source or editing its code
   invalidates store/model compatibility even when the intended calculations
   are unchanged; prove matching outputs before re-stamping instead of
   rebuilding. The forward-study source pins were removed with the recorder.
5. **Split large modules after ownership is clear.**
   Separate responsibilities inside `feature_engineer.py` and `model.py`
   incrementally. Keep date-strict history, complete-field validation,
   calibration boundaries and identical live/training behavior under their
   existing tests. Preserve saved-model loading compatibility.

Research walk-forward primitives and app-facing betting simulation have
distinct purposes; consolidate shared calculations without silently changing
their evaluation contracts.


## Rating systems045 (30 October 2026)

`src/ratings.py` holds the repaired Elo, margin Elo, Glicko-1/2, TrueSkill and
value-added jockey/trainer ratings; its settings are in the `config.py` rating
block. The retired `scripts/screen_ratings.py` (commit `26d1b6f`) scored each
rating as a standalone within-race logit and on top of SP.
[Changes, settings and results](research/RATING_SYSTEMS_045.md).

## Further research039–040 (29October2026)

The unselected prototypes `src/connection_market_research.py` and
`src/recovery_pattern_research.py`, their preparations and
`scripts/screen_frozen_win_candidate.py` were retired on 2026-11-07 (commit
`26d1b6f`). Both stopped after their single small win screen failed the
established-handicap proper-score gate.
[Results and limits](research/ESTABLISHED_CONTEXT_039_040.md); sealed receipts are in
`data/audit/connection_form_039/` and `data/audit/recovery_pattern_040/`.
