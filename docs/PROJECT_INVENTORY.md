# Module map

Generated from each module's docstring on 16 November 2026. Commands are in
[scripts/README.md](../scripts/README.md); the operating guide is the [README](../README.md).

## Collection

| Module | Purpose |
|---|---|
| `src/ingestion/__main__.py` | Coordinated real-source collection CLI. |
| `src/ingestion/atr_ids.py` | ATR identity receipts and batched discovery; telemetry is never inferred. |
| `src/ingestion/atr_jumps.py` | Collect ATR's separate Jumps Data tab; independent of sectional rebuilding. |
| `src/ingestion/atr_rebuild.py` | Fresh, resumable ATR reconstruction. Accepted history is never overwritten. |
| `src/ingestion/atr_tabs.py` | The ATR source contract and extra tab parsers, without features. |
| `src/ingestion/attheraces.py` | = Fetches per-runner sectional, stride, jumping and distance-travelled data from At The Races (``attheraces.com``). It retains per-furlong split times, running  |
| `src/ingestion/auxiliary.py` | Bounded ATR and full Betfair historical refreshes for the source coordinator. |
| `src/ingestion/betfair_archives.py` | Consolidate existing downloads into verified, deduplicated raw payloads. |
| `src/ingestion/betfair_credentials.py` | Windows-account encrypted login settings for unattended historical downloads. |
| `src/ingestion/betfair_history.py` | Resumable downloads of entitled Betfair Historical market files. |
| `src/ingestion/betfair_streams.py` | Parsers for Betfair Historical BASIC ``mcm`` market streams. |
| `src/ingestion/bha_reports.py` | Retain every BHA report field and extract full documents, without features. |
| `src/ingestion/bha_stewards.py` | Paced, resumable BHA reports and full PDFs, outside feature building. |
| `src/ingestion/foreign_form.py` | Collect complete French and UAE race results from Sporting Life. |
| `src/ingestion/fx.py` | European Central Bank reference rates for prize money. |
| `src/ingestion/pedigree.py` | Sporting Life pedigree collection and Racing TV foaling-date enrichment. |
| `src/ingestion/pipeline.py` | SL canonical results plus independently stored Racing TV observations. |
| `src/ingestion/publication.py` | Publish native source stores through the common accepted-data contract. |
| `src/ingestion/racing_tv.py` | = Collects everything Racing TV publishes about GB and IE races, from the JSON API its website calls (``https://api.racingtv.com/racing``). See ``docs/operation |
| `src/ingestion/results.py` | Sporting Life results: the resumable listing and race-page collector. |
| `src/ingestion/rtv_observations.py` | Materialise numeric RTV observations; keep network/collection out of FE. |
| `src/ingestion/rtv_storage.py` | Published Racing TV generations and compatible historical partitions. |
| `src/ingestion/sporting_life.py` | === Scrapes real UK horse racing data from Sporting Life (sportinglife.com). |
| `src/ingestion/storage.py` | Small storage primitives shared by collectors; no feature transformations. |
| `src/ingestion/weather.py` | ==== Fetches historical weather for racecourse locations via the free Open-Meteo Archive API and caches results in a local SQLite database. |

## Pipeline core (src/)

| Module | Purpose |
|---|---|
| `src/atr_features.py` | Pure per-run feature derivations from At The Races sectionals. |
| `src/bet_analysis.py` | Runs the three-strategy betting simulation (top pick / win value / each-way value) over a held-out test set and produces the bets, PnL curves, per-strategy stat |
| `src/bet_settlement.py` | == Single source of truth for the betting-strategy rules shared by: |
| `src/betfair.py` | == |
| `src/clv.py` | ==== |
| `src/data_processor.py` | = Cleans and preprocesses raw horse racing data for feature engineering. |
| `src/database.py` | = SQLite-backed persistent storage for scraped race results: the schema, reads, writes and the published ``data/raw/race_results.parquet``. |
| `src/each_way.py` | Identifies value in each-way betting by comparing the model's predicted place probability to the implied place probability from standard EW terms. |
| `src/established_form_features.py` | Prior-day discipline experience and comparable handicap-mark histories. |
| `src/fe_cache.py` | Saved results of history computations that depend only on immutable inputs. |
| `src/fe_incremental.py` | Incremental state primitives for build-once / append-forever feature engineering (taxonomy class **C**; see docs/architecture/FE_TAXONOMY.md). |
| `src/feature_context.py` | Private working state for one feature execution, including nested scopes. |
| `src/feature_engineer.py` | = Creates predictive features from processed horse racing data. |
| `src/feature_inputs.py` | Fingerprints of the selected dataset's ATR and Racing TV tables. |
| `src/feature_schema.py` | Shared categorical domains and required numeric feature validation. |
| `src/feature_selection.py` | Walk-forward feature selection for the win model (experiment 048). |
| `src/feature_semantics.py` | Canonical parsing/validity contracts used by normal training and live cards. |
| `src/feature_store.py` | Persisted, append-only feature store. |
| `src/featured_dataset.py` | Signatures and metadata stamps that keep the feature store current. |
| `src/forecast_evaluation.py` | Shared, strict probability scoring for complete, single-winner races. |
| `src/foreign_form.py` | Read accepted French and UAE races (complete fields) for history calculations. |
| `src/identity_schema.py` | Atomic source-table key migrations; no changes to accepted result values. |
| `src/live_prediction.py` | Featurise an unsettled card against the store's carried history (features_for_live_cards). |
| `src/model.py` | === Win and Place classifiers for horse race prediction. |
| `src/parquet_io.py` | Bounded parquet consolidation; no feature construction or imputation. |
| `src/prior_market_features.py` | Prior-day market assessments; never current-race prices as predictors. |
| `src/prize_currency_features.py` | Canonical GBP purse inputs plus a research oracle for the native histories. |
| `src/probability_calibration.py` | Race-level temperature calibration shared by training and benchmarks. |
| `src/race_objectives.py` | Race-structured LightGBM objectives (experiments 057, 058; 057 is in production). |
| `src/rating_reuse.py` | Verified reuse of rating sweeps when only non-rating history inputs change. |
| `src/ratings.py` | Dynamic strength ratings for horses (Elo, margin Elo, Glicko-1/2, TrueSkill) and value-added ratings for jockeys and trainers, computed from head-to-head race r |
| `src/result_margins.py` | Convert the source's adjacent finishing gaps to distances behind the winner. |
| `src/result_status.py` | Restore the distinction between settled non-finishers and pending cards. |
| `src/rtv_features.py` | Quality-filtered prior RTV performance and timestamped pre-race context. |
| `src/rtv_text.py` | Timeform comment score: a race-wise text model refitted every quarter (052). |
| `src/run_store.py` | === Each training run is saved as a folder under ``data/runs/<run_id>/`` containing: |
| `src/runner_identity.py` | Canonical Sporting Life runner identities; other sources need a crosswalk. |
| `src/source_contracts.py` | Shared identity, timing and quality rules for accepted source observations. |
| `src/source_dataset.py` | Immutable source publication and a build-wide dataset manifest. |
| `src/source_health.py` | Source attempts, publication delays and consumer readiness are separate. |
| `src/source_prices.py` | Evaluation prices from the selected dataset, attached by saved runner links. |
| `src/store_build.py` | Memory-bounded history build, shared by normal training and explicit rebuilds. |
| `src/store_columns.py` | Numeric feature columns the feature store keeps; every other numeric feature is dropped at the end of ``engineer_features``. Generated by ``research/feature_pru |
| `src/store_refresh.py` | Training-store refresh without loading two copies of every historical feature. |
| `src/surface_features.py` | Canonical surface classification, plus a reference pedigree-surface calculation. |
| `src/track_config.py` | = Physical characteristics of UK & Ireland racecourses. |
| `src/trueskill_kernel.py` | Compiled TrueSkill sweep for races rated as free-for-alls. |
| `src/utils.py` | == Helper functions used across the horse racing prediction application. |
| `src/verified_race_metadata.py` | Immutable, source-confirmed corrections to published historical metadata. |
| `src/wf_eval.py` | Shared walk-forward evaluation primitives — the measuring instrument. |

## Win-model recipes and fits

| Module | Purpose |
|---|---|
| `src/modelling/__main__.py` | Supported modelling operations: benchmark, prepare/select features, and fit candidates. |
| `src/modelling/candidate.py` | Lean candidate artifacts in the existing run registry, without promotion. |
| `src/modelling/catalog.py` | Model-facing family/kind metadata; explicitly not a column formula catalog. |
| `src/modelling/components.py` | Explicit component dispatch shared by evaluation and candidate fitting. |
| `src/modelling/data.py` | Complete fields and explicit projected reads from a compatible feature store. |
| `src/modelling/evaluation.py` | Shared chronological component predictions and nested pool evaluation. |
| `src/modelling/performance.py` | Complete-field performance/spread expert with chronological residual targets. |
| `src/modelling/pool.py` | A small conditional-logit pool with complete-field availability and fallback. |
| `src/modelling/selection.py` | Supported recursive selection using the actual win adapter and later confirmation. |
| `src/modelling/spec.py` | Immutable, serialisable model recipes; defaults are resolved once. |
| `src/modelling/win.py` | Win-only adapter over the existing objectives, weights and calibration. |

## Forecast correction

| Module | Purpose |
|---|---|
| `src/correction/__main__.py` | Forecast correction commands. |
| `src/correction/data.py` | Assemble the correction's history frame from the feature store, the forecast history and prices. |
| `src/correction/history.py` | The forecast history: the win model's out-of-sample forecast for every past runner. |
| `src/correction/inputs.py` | Inputs of the forecast correction: how the model's earlier forecasts turned out. |
| `src/correction/model.py` | The correction model: a race-softmax LightGBM fitted from the win model's forecast as offset. |
| `src/correction/serve.py` | Corrected forecasts for an unsettled card. |

## Matchbook paper record

| Module | Purpose |
|---|---|
| `src/matchbook/__main__.py` | Matchbook paper record of the 11:00 rule (read-only: nothing here can place a bet). |
| `src/matchbook/feed.py` | Matchbook's public price feed for British and Irish horse racing: read-only, no account. |
| `src/matchbook/link.py` | Match Matchbook runners to card runners by track, UK off time and horse name. |
| `src/matchbook/model.py` | The correction with the race's own 11:00 price: the forecast behind the Matchbook paper record. |
| `src/matchbook/paper.py` | The Matchbook paper record of the 11:00 rule: forecast, paper bets, settlement and summary. |
| `src/matchbook/record.py` | Matchbook price snapshots on disk. |

## Workbench (local UI)

| Module | Purpose |
|---|---|
| `src/workbench/data.py` | Bounded, read-only queries for the ingestion workbench. |
| `src/workbench/evaluation.py` | Read historical evaluation artifacts and summarise their recorded bets. |
| `src/workbench/feature_catalog.py` | Read-only feature-set/model-role configuration and honest definition metadata. |
| `src/workbench/features.py` | Bounded artifact inspection for Features. Browsing never computes features. |
| `src/workbench/jobs.py` | Durable project jobs; existing owners retain locks and policies. |
| `src/workbench/modelling.py` | Read saved modelling evidence and submit jobs to the existing modelling CLI. |
| `src/workbench/paper.py` | Workbench view of the Matchbook paper record (read-only; ``src.matchbook`` writes it). |
| `src/workbench/review.py` | Persistent review decisions and validated correction proposals. |
| `src/workbench/server.py` | Loopback-only HTTP API and static client for Racing Workbench. |
| `src/workbench/worker.py` | Execute one persisted workbench command outside the HTTP process. |
