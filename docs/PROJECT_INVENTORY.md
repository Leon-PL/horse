# Project inventory

**Store build speed, 6 November:** [timings and equivalence report](operations/STORE_BUILD_SPEED_2026-11-06.md).
A full store build takes 14.1 minutes, down from 92, with every feature kept and
the store byte-identical to the old build. On long histories it runs in three
processes. [compare_feature_stores.py](../scripts/compare_feature_stores.py)
proves two stores match. One trial build differed in a single carried value, and
the cause was not established; the report says what was ruled out.

**Betfair historical automation, 5 November:** [setup guide](operations/BETFAIR_HISTORY_AUTOMATION.md).
The Data page can connect certificate login and automatically fetch the full
BASIC market streams through the Historical Data API. Raw files, resumable queue
and timed-price imports live in `src/ingestion/betfair_history.py`; saved connection
details use Windows account encryption. Account setup is still required before
the first authenticated transfer. Existing public BSP collection remains active.

**Collection cleanup, 5 November:** real-source implementations are grouped in
[`src/ingestion`](../src/ingestion/), with `python -m src.ingestion` as the
collection command. The fake-data generator and unused collector interface
were removed. See the [current structure](architecture/PROJECT_STRUCTURE.md),
[cleanup report](operations/PRE_FEATURE_CLEANUP_2026-11-05.md), and
[documentation index](README.md). Earlier reports below describe their dated checkpoints.

**ATR reconstruction, 5 November:** [qualified collector and full backfill](operations/ATR_REBUILD_2026-11-05.md).
`src/atr_rebuild.py` commits verified fields to a fresh SQLite queue and exports
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

**Racing TV features and repairs, 30 October:** [source pipeline](../src/source_pipeline.py)
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

Classification describes a file's role, not whether it is safe to rerun.
Research code can import other scripts, write canonical data or install models.
Keep completed evidence and inspect dependencies before retiring anything.

## Entry points and local artifacts

| Item | Role |
|---|---|
| [picks_app.py](../picks_app.py) | Forecasting, manual forward records and exchange connectivity. |
| [training_app.py](../training_app.py) | Data maintenance, candidate training, analysis and registry. |
| [train.py](../train.py) | CLI wrapper around the same training core as the app. |
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
| Collection and publication | [pipeline](../src/ingestion/pipeline.py), [auxiliary](../src/ingestion/auxiliary.py), [storage](../src/ingestion/storage.py), [RTV storage](../src/ingestion/rtv_storage.py) | Coordinated real-source collection, coverage, retained responses and atomic publication. Legacy source-module imports are compatibility aliases. |
| Results and racecards | [Sporting Life](../src/ingestion/sporting_life.py), [database](../src/database.py), [data_processor](../src/data_processor.py) | Collect, store, export and clean source records. Sporting Life commentary and verdicts go to the append-only `runner_text_observations` table (racecard and result copies, with the time seen), never to `results`. |
| Auxiliary sources | [pedigree_backfill](../src/pedigree_backfill.py), [attheraces_scraper](../src/attheraces_scraper.py), [rtv_scraper](../src/rtv_scraper.py), [weather](../src/weather.py) | Optional enrichment. Weather is stored as raw data and does not supply a current feature stage. The Racing TV scraper ([guide](operations/RTV_SCRAPER.md)) collects results, racecards and RaceiQ data. `src.rtv_features` reads its Timeform card and run history; `source_pipeline.load_matched_text` gives its text (Timeform comment and verdict, pace and draw notes, close-ups) per SL runner for research. [rtv_text](../src/rtv_text.py) turns the Timeform comment and verdict into the comment score (`rtv_text_logit`, `rtv_text_prob`, `rtv_verdict_rank`; production inputs since 056). Quarterly walk-forward models live in `data/rtv_cache/text_models`, history scores in `feature_text_scores.parquet`, and live cards are scored from the pre-race snapshot. `rtv_feature_collection` keeps them current with the rest of the RTV cache. |
| Canonical feature definitions | [feature_engineer](../src/feature_engineer.py), [feature_semantics](../src/feature_semantics.py), [result_margins](../src/result_margins.py), [result_status](../src/result_status.py), [audited_surface_features](../src/audited_surface_features.py), [prize_currency_features](../src/prize_currency_features.py), [verified_race_metadata](../src/verified_race_metadata.py), [track_config](../src/track_config.py), [atr_features](../src/atr_features.py) | Normal calculations and source corrections. Surface/currency modules also contain separate research helpers; their research origin does not make their canonical functions optional. |
| Historical state and storage | [ratings](../src/ratings.py), [trueskill_kernel](../src/trueskill_kernel.py), [fe_incremental](../src/fe_incremental.py), [feature_store](../src/feature_store.py), [featured_dataset](../src/featured_dataset.py), [store_refresh](../src/store_refresh.py), [feature_inputs](../src/feature_inputs.py), [rating_reuse](../src/rating_reuse.py), [parquet_io](../src/parquet_io.py), [fe_taxonomy](../src/fe_taxonomy.py) | Historical state, transactional files, provenance and bounded refresh/build support. Two refresh interfaces currently coexist. |
| Training and validation | [model](../src/model.py), [race_objectives](../src/race_objectives.py), [training](../src/training.py), [autotune](../src/autotune.py), [probability_calibration](../src/probability_calibration.py), [deployment_validation](../src/deployment_validation.py) | Shared training core, optional tuning/calibration and deployment-recipe checks. |
| Model selection and serving | [run_store](../src/run_store.py), [live_prediction](../src/live_prediction.py) | Saved runs, explicit production pointer and current-card feature generation. |
| Scoring and shared rules | [wf_eval](../src/wf_eval.py), [feature_selection](../src/feature_selection.py), [forecast_evaluation](../src/forecast_evaluation.py), [bet_analysis](../src/bet_analysis.py), [backtester](../src/backtester.py), [clv](../src/clv.py), [bet_settlement](../src/bet_settlement.py), [each_way](../src/each_way.py), [quality_control](../src/quality_control.py), [utils](../src/utils.py) | Purged research evaluation, walk-forward input selection, complete-field scoring, settlement, identity joins and app-facing simulation. The research and app backtest interfaces have distinct jobs. |
| Market data and analysis | [betfair](../src/betfair.py), [matchbook_client](../src/matchbook_client.py), [matchbook_signals](../src/matchbook_signals.py), [paper_trade_store](../src/paper_trade_store.py), [market_replay](../src/market_replay.py), [forecast_benchmark](../src/forecast_benchmark.py) | Historical teaching/evaluation prices, exchange connectivity, paper records, archive replay and benchmark alignment. |
| App/support interfaces | [app_helpers](../src/app_helpers.py), [__init__](../src/__init__.py) | UI helpers. Application synthetic-data support has been removed. |
| Research or unwired feature tools | [audited_class_features](../src/audited_class_features.py), [history_blend](../src/history_blend.py), [missing_history_snapshot](../src/missing_history_snapshot.py), [public_history_features](../src/public_history_features.py), [race_probability_model](../src/race_probability_model.py), [sibling_features](../src/sibling_features.py), [tail_calibration](../src/tail_calibration.py), [tail_smoothing](../src/tail_smoothing.py), [trainer_locations](../src/trainer_locations.py), [bha_stewards_scraper](../src/bha_stewards_scraper.py), [thurstone](../src/thurstone.py), [luck_weights](../src/luck_weights.py), [connection_profiles](../src/connection_profiles.py) | Separate experimental predictors, projections, mixtures and unwired collection tools. Presence under src does not mean inclusion in normal predictions. |

## Operating tools

### Recent feature work and installation

038 installs a qualified recent place-probability layer while retaining the033
win probabilities and both final tree estimators. Normal future training fits
the same causal calibration recipe in `probability_calibration.py`. The separate
`recent_probability.py` prototype preserves the original development experiment.
`performance_context.py` is an unselected research family; normal ingestion now
preserves its source observations without selecting new tree inputs.
See [the six-issue review and evidence](research/PREDICTIVE_REVIEW_038.md).

| File | Role |
|---|---|
| [benchmark_recent_probability.py](../scripts/benchmark_recent_probability.py), [read_probability_refinement.py](../scripts/read_probability_refinement.py) | Completed fixed90-day development comparison and reporting-only cohort repair. Preserve their frozen source capsule and two fitted checkpoints. |
| [adopt_recent_place.py](../scripts/adopt_recent_place.py) | Completed historical replay and guarded current calibration/promotion using unchanged final heads. Only installation_v4 has the verified current population; do not resume earlier attempts. |
| [collect_performance_context.py](../scripts/collect_performance_context.py), [verify_performance_context.py](../scripts/verify_performance_context.py) | Bounded factual-source collection and causal history verification. Context observations are separate from canonical result/feature columns. |

034/035 research is described in [the market-history review](research/MARKET_HISTORY_REVIEW_034_035.md).
`src/market_transfer_research.py` and `src/market_finish_surprise_research.py`
are separate prototypes, not normal inputs.034 failed its native established-
handicap Brier/precision gate;035 stopped after its target screen failed.
Their preparation/screen scripts reuse the existing small-fit
harness and the same corrected154 baseline. `check_feature_candidate_native.py`
provides a registered win-only check against saved forecasts, without refitting
the baseline or place head. The market-assisted036 check also remains research-
only: `market_correction_research.py` and `check_market_assisted.py` reuse the
same historical model without new tree fits; validation selects raw market.
See [036 results](research/MARKET_ASSISTED_REFRESH_036.md).

The ordinary collector now preserves structured nonfinish reasons (037).
`audit_nonfinish_source.py` checks391 starters in48 cached results against the
before-parser snapshot;37 reasons are recovered with all other values identical.
See [the repair and its limits](research/NONFINISH_SOURCE_037.md). Current model weights
and the154-input feature generation remain033; forward source pins were renewed.

The033 inputs are part of ordinary feature engineering and the default154
selection. The helper's original research filename does not make them optional.
The031 performance and032 trip proposals remain outside normal forecasts.

| File | Role |
|---|---|
| [prior_market_research.py](../src/prior_market_research.py) | Normal bounded history calculations called by feature_engineer; prices and opposition from strictly earlier same-scale starts. Included in the ordinary feature fingerprint. |
| [handicap_performance_research.py](../src/handicap_performance_research.py), [trip_suitability_research.py](../src/trip_suitability_research.py) | Unselected031/032 research helpers, preserved for reproducibility. |
| [quick_win_feature_screen.py](../scripts/quick_win_feature_screen.py) | Small win-only screen; reuses exact numeric baseline fits and can join audited extra inputs without replacing old columns. No automatic qualification or promotion. |
| [check_prior_market_native.py](../scripts/check_prior_market_native.py) | Completed033 native win check against frozen029 forecasts. Its original032 sources are sealed evidence; do not rerun against the changed canonical source. |
| [prepare_prior_market_backfill.py](../scripts/prepare_prior_market_backfill.py), [install_prior_market_history.py](../scripts/install_prior_market_history.py) | Completed additive033 adoption, guarded by source proofs and exact old-column comparisons. These are one-generation migration tools. |
| [rebuild_race_type_repairs.py](../scripts/rebuild_race_type_repairs.py) | Completed032 source repair rebuild with guarded reuse of unchanged rating components. |
| [audit_store_input_delta.py](../scripts/audit_store_input_delta.py) | Read-only bounded comparison of ordered feature generations; writes an audit report. |
| [integrate_canonical_repairs.py](../scripts/integrate_canonical_repairs.py) | Explicit case/parent qualification and current-data installation. Frozen manifests, exact component checkpoints and separate historical validation prevent duplicate fitting and mismatched promotion. |

See [033 results](research/PRIOR_MARKET_HISTORY_033.md) for the selected feature generation,
[038 results](research/PREDICTIVE_REVIEW_038.md) for the installed probability layer and
CONTINUATION.md for the current operating state. Preserve completed audit capsules;
their source snapshots and old input files support the recorded comparisons.

### Shared operating tools

These support readiness and forecasts. Caching can write local files; the readiness check is read-only.

| File | Purpose and scope |
|---|---|
| [build_lookahead.py](../scripts/build_lookahead.py) | Optional forecast-cache preparation using current cards and the shared live feature path. Fetches cards and writes cache files. |
| [project_status.py](../scripts/project_status.py) | Read-only readiness check across database, features, runs and production. Full check compares database contents; --quick skips that comparison. |

## Maintenance and source-data tools

These are deliberate maintenance actions, not commands to run as a batch. Backfills can rewrite history and trigger a full feature rebuild.

| File | Purpose and scope |
|---|---|
| [backfill_draw.py](../scripts/backfill_draw.py) | Targeted database repair for starting stalls; writes results data. |
| [backfill_missing_races.py](../scripts/backfill_missing_races.py) | Collect missing races identified against exchange data; writes results data. |
| [backfill_pedigree.py](../scripts/backfill_pedigree.py) | Collect missing horse pedigree; writes source data. |
| [backfill_rtv_text.py](../scripts/backfill_rtv_text.py) | Rebuild the Timeform comment-score history: collects RTV card text from 2020-09, fits any missing quarterly text models (`--refit` refits all) and rescores every runner. Normal RTV refreshes do this incrementally. |
| [backfill_wind_surgery.py](../scripts/backfill_wind_surgery.py) | Targeted wind-surgery source backfill; writes results data. Presence does not imply a selected prediction input. |
| [build_connection_aliases.py](../scripts/build_connection_aliases.py) | Rebuild `reference/connection_aliases.json`, which gives each jockey and trainer one name across Sporting Life renamings. Links Racing TV person IDs to stored rides; names whose IDs overlap in time, or with clashing initials, are left alone. Changing the file changes `fe_signature`. Report only unless `--write`. |
| [build_feature_store_batched.py](../scripts/build_feature_store_batched.py) | Explicit full history rebuild with bounded memory and a staged commit. Its _build function is also called by normal training. |
| [compare_feature_stores.py](../scripts/compare_feature_stores.py) | Prove two feature stores match: every column, row order and carried state. Reads only. |
| [create_validation_report.py](../scripts/create_validation_report.py) | Export an evaluated run's recipe certificate for a later deployment refit; writes a new report. |
| [extract_betfair_bsp.py](../scripts/extract_betfair_bsp.py) | Extract win BSP and timed observations from local BASIC archives. Also exports helpers imported by src.market_replay. |
| [extract_betfair_place_bsp.py](../scripts/extract_betfair_place_bsp.py) | Extract place BSP from local BASIC archives; writes a derived research/data artifact. |
| [fetch_betfair_sp_public.py](../scripts/fetch_betfair_sp_public.py) | Download public daily exchange starting-price files. Refreshing the canonical teaching cache is a separate audited step. |
| [fetch_bha_going.py](../scripts/fetch_bha_going.py) | Collect going, rail and stalls reports; collection does not establish that the model uses these fields. |
| [repair_overfull_fields.py](../scripts/repair_overfull_fields.py) | Remove non-runners stored as runners (more rows than `num_runners`). Re-scrapes each race and replaces the validated field, or deletes only rows the page marks NONRUNNER/WITHDRAWN; backs up to data/backups first, then re-exports `data/raw/race_results.csv` from the DB. Dry run unless `--apply`. |
| [verify_live_upgrade.py](../scripts/verify_live_upgrade.py) | Operational qualification of the installed artifact, including a public card; writes a new receipt, without fitting or promotion. |

## Research comparisons and screens

General comparison tools and experiment-specific implementations are kept together here as research. Names such as benchmark or screen do not imply fresh data, current defaults, or promotion eligibility.

| File | Purpose and scope |
|---|---|
| [assess_forecast_parity.py](../scripts/assess_forecast_parity.py) | Descriptive parity bounds for frozen forecasts; never select on this report. |
| [assess_independent_market.py](../scripts/assess_independent_market.py) | Describe frozen independent feature forecasts against their same-time market. |
| [benchmark_atr_collection.py](../scripts/benchmark_atr_collection.py) | Bounded live ATR benchmark; writes only a separate research directory. |
| [benchmark_calibration.py](../scripts/benchmark_calibration.py) | Compare calibration on strictly chronological, disjoint race windows. |
| [benchmark_forecasts.py](../scripts/benchmark_forecasts.py) | Frozen chronological comparison of independent and same-time market forecasts. |
| [benchmark_history_blend.py](../scripts/benchmark_history_blend.py) | Experiment 013 context mixture following 012; exploratory, not fresh confirmation. |
| [benchmark_history_features.py](../scripts/benchmark_history_features.py) | Experiment 012 history-family comparison, tied to its frozen baseline and inputs. |
| [benchmark_independent_features.py](../scripts/benchmark_independent_features.py) | Registered, purged comparison of two independent probability feature sets. |
| [benchmark_margin_repair.py](../scripts/benchmark_margin_repair.py) | Experiment 014 historical-margin comparison against its preserved reference. |
| [benchmark_model_repair.py](../scripts/benchmark_model_repair.py) | Configurable, resumable native model comparison. Reusable harness structure; inspect its registration and frozen inputs. |
| [benchmark_race_objective.py](../scripts/benchmark_race_objective.py) | One registered comparison of two race-level recipes and frozen binary forecasts. |
| [benchmark_refit_cadence.py](../scripts/benchmark_refit_cadence.py) | Registered monthly versus quarterly refitting, with the forecast recipe fixed. |
| [benchmark_surface_repair.py](../scripts/benchmark_surface_repair.py) | Experiment 017 surface comparison against its completed 016 reference. |
| [benchmark_tail_calibration.py](../scripts/benchmark_tail_calibration.py) | Experiment 010 fixed-tree calibration comparison. |
| [benchmark_tail_smoothing.py](../scripts/benchmark_tail_smoothing.py) | Experiment 011 relative probability-floor comparison on saved out-of-sample forecasts. |
| [compare_probability_runs.py](../scripts/compare_probability_runs.py) | Replay two saved models on identical complete historical race fields. |
| [fe_sweep.py](../scripts/fe_sweep.py) | Historical feature-constant sweep; BASE_RUN refers to a removed snapshot. Update and re-register before any new use. |
| [holdout_sibling_test.py](../scripts/holdout_sibling_test.py) | Historical reserved-period sibling-feature test; those inspected dates are not a fresh holdout. |
| [model_lab.py](../scripts/model_lab.py) | Fast train+eval harness for improving the (odds-free) win model. |
| [replay_market_snapshots.py](../scripts/replay_market_snapshots.py) | Create immutable T-5 forecast research inputs from local Betfair archives. |
| [screen_calibration_choice.py](../scripts/screen_calibration_choice.py) | Experiment 019 calibration-family screen using training-history splits. |
| [screen_feature_batch.py](../scripts/screen_feature_batch.py) | Experiment 018 registered feature screen after audits. |
| [screen_model_capacity.py](../scripts/screen_model_capacity.py) | Experiment 021 fixed capacity screen. |
| [screen_prize_currency.py](../scripts/screen_prize_currency.py) | Experiment 020 currency-feature screen. |
| [screen_ratings.py](../scripts/screen_ratings.py) | Experiment 045 rating-system screen: each rating as a standalone within-race logit, alone and on top of SP; `--set` overrides settings, `--stored` scores the store, `--systems pl` a research-only Plackett–Luce rating. |
| [race_profile_screen.py](../scripts/race_profile_screen.py) | Race-profile screen: how predictable duration, tempo, finishing shape and leader bias are before the off, by nested input layer (RTV, development period). |
| [race_profile_fit.py](../scripts/race_profile_fit.py) | Runner x projected-race-profile families (trip on duration, stamina reach, shape fit, style x shape) and the store pace family, scored over the T-5 market with size-matched noise baselines. |
| [rtv_derived_screen.py](../scripts/rtv_derived_screen.py) | RTV-derived candidates the pipeline does not build (sectional-time figures, pace-adjusted form, within-race stride/jump metrics, Timeform card, pace hint), scored over the T-5 market and over a market-free form proxy against size-matched noise. |
| [ablate_rtv_derived.py](../scripts/ablate_rtv_derived.py) | Four-quarter quick LightGBM ablation (quick_win_feature_screen recipe) of the Timeform card and RTV-derived families on the promoted inputs; receipts in data/audit/rtv_derived_ablation. |
| [validate_rtv_derived_native.py](../scripts/validate_rtv_derived_native.py) | Native win-recipe fits with and without win_rtv_derived.txt for 2026Q1 and 2026Q2 (development only), scored paired and against the SP; resumable, receipts in data/audit/rtv_derived_native. |
| [impl_review_variants.py](../scripts/impl_review_variants.py), [impl_review_variants_b.py](../scripts/impl_review_variants_b.py) | Experiment 047: rebuild repaired versions of promoted inputs from the store (round 1: relative transforms, cold starts, form positions, prize/draw; round 2: OR stand-ins, shrunk connection rates, rank shares), after reproducing the originals exactly. |
| [ablate_impl_review.py](../scripts/ablate_impl_review.py), [ablate_impl_review_b.py](../scripts/ablate_impl_review_b.py) | Experiment 047: four-quarter quick LightGBM ablation of those repairs, replacing inputs in place; receipts in data/audit/feature_impl_review_047{,b}. |
| [select_features.py](../scripts/select_features.py) | Experiment 048: chooses the win allowlist by the model's own out-of-sample race log loss. Stages are `prepare` (development matrix cache), `eliminate` (recursive SHAP elimination over purged quarters 2024Q1–2025Q2, stopped by the one-SE rule) and `confirm` (selected vs `win.txt` vs all on 2025Q3–2026Q2, at screen or native capacity). Primitives are in `src/feature_selection.py`; receipts are in data/research/feature_selection_048. |
| [text_model.py](../scripts/text_model.py) | Experiment 052: walk-forward race-wise softmax text model on the Timeform comment and verdict (RTV cards), whose out-of-fold score is added to the 048 inputs on the 049 harness. Small confirmed gain (-0.0029 race LL, 4/4 quarters); receipts in data/research/text_052. Also holds the reusable `fit_scores`, `ExtraCache` and `fit_arm`. Its text primitives (tokens, TF-IDF, race softmax) now live in `src/rtv_text.py`, which the production stage shares. |
| [rtv_card_extras.py](../scripts/rtv_card_extras.py) | Experiment 053: unused RTV card data (badges, pundit tips, draw comment, owner history) and a second text model that adds pace notes and earlier close-ups, on the same harness; receipts in data/research/rtv_extras_053. |
| [screen_controllable_055.py](../scripts/screen_controllable_055.py) | Experiment 055 follow-up: sizes fixes for the 054 model's worst misses on the 049 `general` arm's out-of-fold predictions. The fixes are a debutant shrink, the trainer/sire debut record, the off-DB last run with its position, and discipline carry-over. Each is fitted as a race-wise logit adjustment on earlier quarters and scored on the next. All are within noise; receipts in data/research/controllable_055. |
| [specialist_models.py](../scripts/specialist_models.py) | Experiment 049: hurdle-only and Wolverhampton-only win models against the general model and the market, on the 048 inputs and cache (2025Q3–2026Q2, native capacity). It includes an out-of-sample market+model blend. Refuted; receipts are in data/research/specialist_049. |
| [model_ideas.py](../scripts/model_ideas.py) | Shared harness for experiments 057–060 (docs/research/MODEL_IDEAS_057_060.md): the 048/049 cache and production recipe, resumable per-quarter arms on the development (screen) and confirmation (native) quarters, and paired date-clustered comparisons. `baseline [--dev]` fits the production arm. |
| [exploded_softmax_057.py](../scripts/exploded_softmax_057.py) | Experiment 057: race softmax and exploded (rank-ordered) softmax objectives in place of production's per-runner cross-entropy. Stage weights are chosen on the development quarters and registered before confirmation. Confirmed (−0.0046, 4 of 4 quarters). |
| [exploded_depth_061.py](../scripts/exploded_depth_061.py) | Experiment 061, a follow-up to 057: development screen of finishing-order depth (top three against every finisher) and per-stage flattening (temperatures). Every finisher ties with the top three; flattening makes the model over-confident. 057 stands. |
| [catboost_pairwise_062.py](../scripts/catboost_pairwise_062.py) | Experiment 062: CatBoost `PairLogit` on pairs built from 057's three stages. CatBoost is installed beside the locked environment in data/tooling/catboost (the command is in the script). The results-only fast arm is much worse; the soft-target arm ties 057 and adds little to the 057 + figure blend. Not adopted. |
| [feature_structure_063.py](../scripts/feature_structure_063.py) | Experiment 063 (docs/research/FEATURES_AND_STACKING_063.md): development screen of interactions, race context, finer race kinds, domain-shaped inputs, target market share, sub-model targets, subject models and meta learners. Reads the frozen 2 November cache with its store check bypassed on purpose. Its registered confirmation found 64 extra inputs with a 0.75 market share 0.00628 better than 057. Not installed. |
| [thurstone_058.py](../scripts/thurstone_058.py) | Experiment 058: mean performance figure (pounds behind the winner) and spread, turned into win chances by a Thurstone race model. The report has an out-of-sample blend with production and spread/calibration by debutant, new-yard and 11:00→BSP move. Worse alone; adds information in a blend. |
| [luck_weights_059.py](../scripts/luck_weights_059.py) | Experiment 059: trouble severity per runner from Racing TV close-ups and BHA stewards' events; luck-adjusted training weights screened and registered on development quarters. Not confirmed. |
| [connection_profiles_060.py](../scripts/connection_profiles_060.py) | Experiment 060: quarterly learned trainer, jockey and sire profiles against SP, added as 13 inputs. Not confirmed; the profiles are priced. |
| [run_model_ideas.sh](../scripts/run_model_ideas.sh) | Runs experiments 057–060 end to end, one stage at a time, logging to data/logs/model_ideas_057_060.log. Every stage resumes from saved folds. |
| [triage_features.py](../scripts/triage_features.py) | Stage-1 triage: does a candidate feature carry information the market lacks? |
| [validate_model_capacity.py](../scripts/validate_model_capacity.py) | Experiment 022 wider validation of the selected capacity change. |
| [wf_benter_combination.py](../scripts/wf_benter_combination.py) | Walk-forward re-validation of the Benter market-combination edge. |
| [wf_fundmarket_clv.py](../scripts/wf_fundmarket_clv.py) | Walk-forward: does Fund+Market beat the market across time, and is it bankable? |
| [wf_market_offset.py](../scripts/wf_market_offset.py) | Walk-forward test of ACTION_PLAN 4c: learn only what the market misses. |
| [wf_ranking_compare.py](../scripts/wf_ranking_compare.py) | Walk-forward ranking comparison: GBM vs logistic regression vs the market. |

## Research audits and diagnostics

These investigate inputs, causality, coverage or errors. Some collect data or write projections/reports. Their frozen paths and scopes need inspection before reuse.

| File | Purpose and scope |
|---|---|
| [audit_atr_invariance.py](../scripts/audit_atr_invariance.py) | Gate an ATR family trial on date/future invariance, without rebuilding data. |
| [audit_cached_race_codes.py](../scripts/audit_cached_race_codes.py) | Cross-check local metadata against already cached public performances. |
| [audit_canonical_inputs.py](../scripts/audit_canonical_inputs.py) | Confirm that the corrected store changes input values without losing runners. |
| [audit_established_horse_errors.py](../scripts/audit_established_horse_errors.py) | Research diagnostic of established-horse errors using frozen out-of-sample forecasts. |
| [audit_feature_contracts.py](../scripts/audit_feature_contracts.py) | Outcome-free semantic checks of selected features and overseas-history gaps. |
| [audit_field_timing.py](../scripts/audit_field_timing.py) | Are the scraped per-race fields as-of-race, or as-of-scrape? |
| [audit_glicko_inactivity.py](../scripts/audit_glicko_inactivity.py) | Measure the verified Glicko time-accounting error without fitting a learner. |
| [audit_history_invariance.py](../scripts/audit_history_invariance.py) | Date/row/result invariance of the existing history-support family for012. |
| [audit_labels.py](../scripts/audit_labels.py) | Are the labels right? Cross-check the results DB against Betfair. |
| [audit_missing_history.py](../scripts/audit_missing_history.py) | Validate the Cheltenham dated-history pilot without fitting or viewing returns. |
| [audit_nonfinisher_history.py](../scripts/audit_nonfinisher_history.py) | Check that non-completions survive storage as actual historical starts. |
| [audit_pace_invariance.py](../scripts/audit_pace_invariance.py) | Check the existing pace/setup family before its independent model trial. |
| [audit_prediction_inputs.py](../scripts/audit_prediction_inputs.py) | Audit all selected independent inputs against temporal and price changes. |
| [audit_prize_currency.py](../scripts/audit_prize_currency.py) | Verify purse units at every stored venue from early, middle and late races. |
| [audit_public_history_coverage.py](../scripts/audit_public_history_coverage.py) | Build a separate dated feature projection after the fixed public pilot ends. |
| [audit_repaired_store.py](../scripts/audit_repaired_store.py) | Review every selected input after rebuilding, before screening a model. |
| [audit_result_integrity.py](../scripts/audit_result_integrity.py) | Measure source-margin, identity, label and market-teacher integrity. |
| [audit_selected_feature_semantics.py](../scripts/audit_selected_feature_semantics.py) | Audit the selected model inputs and their source semantics, without fitting. |
| [audit_selection_rule.py](../scripts/audit_selection_rule.py) | Where does the value filter lose money? |
| [audit_source_recency.py](../scripts/audit_source_recency.py) | Check whether raw pre-race recency agrees with observed earlier starts. |
| [betfair_db_audit.py](../scripts/betfair_db_audit.py) | Two-directional integrity check between Betfair and ``data/races.db``. |
| [diagnose_disagreement.py](../scripts/diagnose_disagreement.py) | Model-vs-market disagreement diagnostic. |
| [diagnose_feature_screen.py](../scripts/diagnose_feature_screen.py) | Describe the frozen 018 screen's market gap, coverage and probability tails. |
| [diagnose_remaining_market_gap.py](../scripts/diagnose_remaining_market_gap.py) | Describe remaining losses using frozen022 forecasts; never fit or select a model. |
| [diagnose_tail_tradeoff.py](../scripts/diagnose_tail_tradeoff.py) | Post-result sensitivity: earlier best log-loss floor without the Brier veto. |
| [explain_established_horse_errors.py](../scripts/explain_established_horse_errors.py) | Explain saved established-horse cases using the preserved historical learner; no new fitting. |
| [feature_audit.py](../scripts/feature_audit.py) | Feature-quality audit over the feature store. |

## Experiment support and historical workflows

These include completed migration controllers, cohort collectors, simulations and exact-fit helpers. Several can modify source files, canonical data, production models or running jobs; this is not a list of routine maintenance commands.

| File | Purpose and scope |
|---|---|
| [build_missing_history_projection.py](../scripts/build_missing_history_projection.py) | Compose the Flat and Cheltenham public-history pilots for one research test. |
| [build_surface_repair_projection.py](../scripts/build_surface_repair_projection.py) | Freeze a causal research projection of the audited surface repair. |
| [canonical_training_cache.py](../scripts/canonical_training_cache.py) | Importable exact-fit cache helper used by the 026/027 controllers; retained shared code, not a standalone operating command. |
| [clv_report.py](../scripts/clv_report.py) | Historical CLV/exchange-settlement analysis; references local run and price artifacts. |
| [collect_missing_history_results.py](../scripts/collect_missing_history_results.py) | Extend the dated-results pilot to jumps and Cheltenham missing-history horses. |
| [collect_prior_horse_results.py](../scripts/collect_prior_horse_results.py) | Cache public prior results for a fixed, outcome-free missing-history cohort. |
| [execution_timing.py](../scripts/execution_timing.py) | Ledger 20: execution timing - when to strike, not what to back. |
| [execution_timing_diagnostics.py](../scripts/execution_timing_diagnostics.py) | Exploratory diagnostics for ledger 20's descriptive lead. NOT a test. |
| [export_feature_repair_batch.py](../scripts/export_feature_repair_batch.py) | Save reviewable candidate/native diffs without replacing installed sources. |
| [fetch_forecast_previous_runs.py](../scripts/fetch_forecast_previous_runs.py) | Fetch archived day-ahead weather forecasts for every course (EXPLORATION_PLAN C). |
| [finish_feature_batch.py](../scripts/finish_feature_batch.py) | Checkpoint continuation for the historical repair/screen batch; review its frozen state before use. |
| [finish_surface_screen.py](../scripts/finish_surface_screen.py) | Historical surface-screen job controller, including process/queue stopping behavior; not a routine diagnostic. |
| [ground_forecast.py](../scripts/ground_forecast.py) | EXPLORATION_PLAN C: do early prices under-react to forecast ground changes? |
| [integrate_canonical_repairs.py](../scripts/integrate_canonical_repairs.py) | Completed 026 correction/deployment controller. Reproduction requires preserved teacher inputs; do not rerun against changed defaults. |
| [microstructure_clv.py](../scripts/microstructure_clv.py) | Economic follow-up to the microstructure screen: is the timing signal bankable? |
| [microstructure_screen.py](../scripts/microstructure_screen.py) | Screen market microstructure for information the price has not absorbed. |
| [morning_mm_feasibility.py](../scripts/morning_mm_feasibility.py) | EXPLORATION_PLAN E, phase 1: could a passive morning market-maker earn the spread? |
| [morning_money_segments.py](../scripts/morning_money_segments.py) | Ledger 19: is the 11:00 morning-money value concentrated in debutants and favourites? |
| [reconcile_result_fields.py](../scripts/reconcile_result_fields.py) | Historical source reconciliation that can repair canonical result fields; inspect its planned changes and evidence. |
| [refit_bsp_teacher.py](../scripts/refit_bsp_teacher.py) | Completed 027 teaching-data installation and win refit controller; writes canonical data and production artifacts. |
| [run_one_feature_screen.py](../scripts/run_one_feature_screen.py) | Execute one registered fold using its frozen experiment implementation. |
| [smoke_prediction_repair.py](../scripts/smoke_prediction_repair.py) | Qualify a prospective repair against complete public cards; experiment support, not deployment. |

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

The current documentation pass changes navigation and descriptions. It does
not move executable files, alter defaults, rebuild data, fit/promote models or
create a new forecast study.

The next code consolidation should address these concrete boundaries:

1. **One historical-feature refresh implementation.**
   `src.store_refresh.get_training_dataset` is the normal CLI/app route.
   `src.featured_dataset.get_featured_dataset` still contains an older
   refresh/rebuild decision path and is exercised by its own tests.
   Preserve full-build, append, transaction and live-feature parity while
   unifying the underlying behavior.
2. **Shared implementations belong behind command wrappers.**
   `src.store_refresh` imports `scripts.build_feature_store_batched._build`;
   `src.market_replay` imports helpers from `scripts.extract_betfair_bsp`.
   These dependencies prevent treating the whole scripts directory as an
   archive. Extract shared implementation first and retain compatible wrappers.
3. **Review research dependencies before archiving.**
   Example: `canonical_training_cache` is imported by both correction/refit
   controllers; comparison scripts also share scoring/alignment functions.
   Completed work does not make every supporting function disposable.
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

The two Streamlit processes serve different workloads: forecasts and heavy
maintenance/research. A clearer navigation model does not require combining
them. Likewise, research walk-forward primitives and app-facing betting
simulation have distinct purposes; consolidate shared calculations without
silently changing their evaluation contracts.


## Rating systems045 (30 October 2026)

`src/ratings.py` holds the repaired Elo, margin Elo, Glicko-1/2, TrueSkill and
value-added jockey/trainer ratings; its settings are in the `config.py` rating
block. `scripts/screen_ratings.py` scores each rating as a standalone
within-race logit and on top of SP; `--set` overrides a setting.
[Changes, settings and results](research/RATING_SYSTEMS_045.md).

## Further research039–040 (29October2026)

`src/connection_market_research.py` and `src/recovery_pattern_research.py` are
unselected prototypes. Their fixed preparations are
`scripts/prepare_connection_form.py` and `prepare_recovery_pattern.py`. Both
stopped after their single small win screen failed the established-handicap
proper-score gate. `scripts/screen_frozen_win_candidate.py` reuses the sealed
original win source and cached baseline, checking all numeric arguments and
exact saved forecast replay before one candidate fit. It does not alter normal
training or bypass current model/store generation rules.
[Results and limits](research/ESTABLISHED_CONTEXT_039_040.md); sealed receipts are in
`data/audit/connection_form_039/` and `data/audit/recovery_pattern_040/`.
