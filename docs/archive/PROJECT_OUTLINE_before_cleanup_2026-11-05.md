# Horse Racing Machine Learning & Betting System

**Historical architecture reference.** Start with the [operating guide](../../README.md)
and [project inventory](../PROJECT_INVENTORY.md) for current procedures and
dependencies. Tables and status statements below reflect earlier checkpoints;
the old table A is not an exhaustive list of code affecting today's pipeline.
**Project Handover & Context Document**

This document serves as a high-level overview of the project architecture, current status, and critical context for any AI agent or developer continuing work on the codebase.

## 1. Project Overview
The project is a comprehensive Python-based machine learning pipeline and betting automation system for horse racing. It features data scraping, feature engineering, model training (LightGBM by default; XGBoost/CatBoost selectable), backtesting, and live Matchbook API integration, all orchestrated through a Streamlit UI.

**Core Tech Stack**
*   **UI Application:** Streamlit, split into two processes — `picks_app.py` (lean serving) and `training_app.py` (heavy training/analysis)
*   **Modeling:** LightGBM (default framework, incl. focal-loss classifiers), XGBoost, CatBoost, Scikit-learn (Joblib for persistence)
*   **Hyperparameter Tuning:** Optuna
*   **Data Processing:** Pandas, Numba (for high-performance functions)
*   **Integrations:** Matchbook Betting Exchange API
*   **Tests:** Pytest (`tests/`) — settlement maths, each-way rules, merge-key normalisers

## 2. Directory Structure & System Status

### 2.0 The map — read this first

The repository is **one pipeline plus a research toolkit**. Nothing else is
live. If a module is not in table A, it cannot affect a bet.

#### A. The pipeline — the only path that produces a selection

| # | Stage | Modules | Produces |
|---|---|---|---|
| 1 | **Collect** | `data_scraper` (Sporting Life), `data_collector`, `database.sync_database`; side feeds `pedigree_backfill`, `attheraces_scraper` (TPD sectionals), `weather` (raw columns into the DB only), `betfair` (BSP/LTP) | `data/races.db` |
| 2 | **Process** | `data_processor.process_data` | canonical runner frame |
| 3 | **Engineer** | `feature_engineer.engineer_features` + `ratings`, `atr_features`, `track_config`, `fe_incremental` | ~1,100 feature columns |
| 4 | **Store** | `feature_store.FeatureStore` behind `featured_dataset.get_featured_dataset` | `data/processed/feature_store/` |
| 5 | **Train** | `training.train_run` → `model.RacePredictor` (+ `autotune`, `probability_calibration`) | win + place classifiers |
| 6 | **Evaluate** | `bet_analysis`, `backtester`, `clv`, `each_way`, `quality_control`; **rules in `bet_settlement`** | calibration / ROI / CLV |
| 7 | **Register** | `run_store.save_run` → `run_store.promote` | `data/runs/`, `models/production.json` |
| 8 | **Serve** | `picks_app.py` → `live_prediction` → `FeatureStore.featurize_live`; `matchbook_client` / `matchbook_signals` / `paper_trade_store` for live prices and paper bets | today's picks |

**Entry points:** `train.py` (CLI train), `training_app.py` (train/tune/backtest/promote
UI), `picks_app.py` (serving UI), `scripts/project_status.py` (**run this first** —
it verifies stages 1→4→5→7 agree, per CLAUDE.md hard rule 5).

Shared infrastructure that every stage depends on: `utils` (the merge keys —
hard rule 3), `bet_settlement` (the bet rules — hard rule 2), `featured_dataset`
(`fe_signature` — hard rule 4), `wf_eval` (walk-forward primitives — hard rule 6).

#### B. The research toolkit — measures the pipeline, never feeds it

| Tool | Answers |
|---|---|
| `scripts/project_status.py` | do database → store → runs → promoted model agree? |
| `scripts/triage_features.py` | is this feature orthogonal to the market? (partial correlation + BH q) |
| `scripts/audit_selection_rule.py` | does this filter beat *its own universe*, not just the whole band? |
| `scripts/audit_labels.py` | do our `won`/placed labels agree with Betfair's settlement? |
| `scripts/audit_field_timing.py` | is every scraped per-race field as-of-race, not as-of-scrape? (the `sex` leak, ledger 15) |
| `scripts/betfair_db_audit.py` with `--bsp data/processed/betfair_sp_public.parquet` | is the DB missing races? (checked against Betfair's public census of every GB/IE race) |
| `scripts/wf_market_offset.py`, `scripts/wf_ranking_compare.py` | plan 4c (market as offset) and plans 3a/4b (GBM vs logistic vs market), purged walk-forward |
| `scripts/microstructure_screen.py`, `scripts/microstructure_clv.py` | does morning money carry information beyond the close; is the timing edge bankable? (ledger 14) |
| `scripts/clv_report.py`, `scripts/wf_benter_combination.py`, `scripts/wf_fundmarket_clv.py` | CLV at BSP and market combination, on purged walk-forward folds |
| `scripts/model_lab.py` | fast single-fit model comparison (~1 min per idea) |
| `scripts/feature_audit.py`, `scripts/diagnose_disagreement.py` | burn-in curves; where the model disagrees with the market |
| `scripts/benchmark_calibration.py`, `scripts/fe_sweep.py` | calibration methods; FE constants |
| `scripts/backfill_*.py`, `scripts/extract_betfair_*.py`, `scripts/betfair_db_audit.py`, `scripts/build_*.py` | data repair, extraction and store builds |

`scripts/fe_sweep.py` still points `BASE_RUN` at a run snapshot that no longer
exists; change the id before using it.

#### C. What was tried and refuted — do not rebuild these

Every row was built, measured and deleted. The measurement lives in
`docs/NEGATIVE_RESULTS.md`; the code is in git history. **Read the ledger before
proposing a new feature or data source** — most plausible ideas are already in it.

| Deleted 2026-10-13 | Was | Verdict |
|---|---|---|
| `rtv_scraper` + `add_rtv_features` | Racing TV RACEiQ sectionals | superseded by ATR (broader coverage, richer record); stage was already unwired. The scraper was rebuilt on 2026-10-29 as a collector; `add_rtv_features` stays deleted |
| `stewards_features`, `ihrb_daily_enquiries_scraper` | stewards-report trouble signals | derived trouble beat the report text, and both were tiny (ledger 7 + 2b) |
| `par_time_features` | going-corrected par-time speed figures | most discriminative new family yet CLV-flat → redundant (ledger 4) |
| `tattersalls_sales_scraper` | public sale price as a debutant proxy | market prices it fully (residual corr 0.0000) (ledger 5) |
| `place_market_model`, `paper_trading` | +EV exchange-place strategy | the +15% was a stale-store artifact; on correct features +1.5% with a CI spanning zero, going negative (ledger 1) |
| `strategy_calibrator` | grid search over betting parameters on the test set | unused, and the method is the overfitting trap this project keeps paying for |
| 17 one-off scripts | sale-price / intent-signal / field-size triages, `train_600d*`, `ab_*`, leak probes, a GPU benchmark | each answered its question; results in the ledger |

Still **unwired by design**: `sibling_features` (dam-produce form) — priced
overall, with one unconfirmed debutant lead awaiting a CLV test (ledger 2h).
Wiring it in requires registering it in `_FE_SOURCE_FILES`.

### 2.1 User Interface (`picks_app.py`, `training_app.py` & `config.py`)
*   **Status:** Functional, actively maintained.
*   **Description:** The UI is split into two Streamlit processes so the day-to-day serving app stays fast (the old monolithic `app.py` was retired). Pages are selected via a sidebar radio, so only the active page's section executes per rerun. The cached run-store wrappers, chart builders and diagnostic panels live in `src/app_helpers.py`.
    *   **`picks_app.py` (serving)** — loads only the production-promoted model via the registry pointer (`src/run_store.load_production_predictor` / `production.json`); it never loads the full feature store at startup. Pages: `💰 Today's Picks` (high-value predictions matched to today's races, plus manual custom-race entry) and `🔌 Matchbook API` (Matchbook connectivity harness).
    *   **`training_app.py` (training/analysis)** — heavy, run on a separate port. A thin Streamlit shell over the callable `src/training` modules. Pages: `📦 Data` (feature-store inspect/refresh), `🤖 Train` (configure frameworks/hyperparameters, run `train_run`, review calibration/ROI/CLV evals), `🗂️ Registry` (browse saved runs, promote one to production).
*   **Recent Changes:** Training pipeline rebuilt as a callable core (`src/training.train_run`) with a lean run registry (no per-run featured/processed parquet snapshots — reproduce from the store) and a production pointer for promotion. Serving and training were split into two apps. H2O AutoML and FLAML were previously removed from the UI and dependencies. Sub-model frameworks are selectable per role (`config.SUB_MODEL_FRAMEWORKS`); the current defaults are LightGBM for both win and place classifiers.

### 2.2 Data Pipeline & Scraping (`src/data_*.py`, `src/database.py`, `src/pedigree_backfill.py`)
*   **Status:** Stable / In Production.
*   **Description:** Extracts race results and live racecards. Includes caching logic (`data/raw/`, `data/racecards_cache/`) to limit network usage and API bans.
*   **Notable modules:** `database.py` manages historical data access; `attheraces_scraper.py` fetches At The Races / TPD sectionals (needs `curl_cffi` or the system `curl` for the Fastly challenge) and is the only sectional source. Also present: `betfair.py` (BSP/LTP joins) and `weather.py` (the sync writes raw weather into the DB; since 2026-10-13 no feature stage reads it). `trainer_locations.py` and `bha_stewards_scraper.py` are unwired data tools: the trainer-travel stage that used them was removed on 2026-10-13 (ledger 13).
*   **Retired scrapers** (deleted 2026-10-13, recoverable from git): IHRB/BHA stewards reports and Tattersalls sale prices (both refuted — see `docs/NEGATIVE_RESULTS.md`). The Racing TV scraper deleted on the same day was rebuilt on 2026-10-29 as a collector only (`src/rtv_scraper.py`, not wired into features; see `docs/operations/RTV_SCRAPER.md`). `data/rtv_cache/` is its store and holds the only Irish sectional data — do not delete it.
*   **Pedigree:** the results/racecard APIs never include breeding, so the `sire`/`dam`/`damsire` columns in `results` are always null at scrape time. `src/pedigree_backfill.py` fills a `horse_pedigree` table from Sporting Life horse *profile* pages (keyed by horse_id; pedigree stored by NAME because the profile JSON's nested ids are buggy). `data_processor.process_data` joins it in; `sync_database` fetches profiles for first-time runners automatically. Full historical backfill completed 2026-06-12 (70k+ horses).

### 2.3 Feature Engineering & Ratings (`src/feature_engineer.py`, `src/ratings.py`, `src/data_processor.py`)
*   **Status:** Active Development / Needs careful handling.
*   **Description:** Transforms raw horse racing outcomes into lag features, rolling statistics, and rating systems.
*   **Rating systems** (`src/ratings.py`): adaptive-K Elo (horse/jockey/trainer + per-dimension variants + margin Elo), Glicko-1 (horse; rating deviation inflates with layoffs) and TrueSkill (n-way free-for-all mu/sigma; `TRUESKILL_ENABLED`, on by default since the 2026-06-12 sweep). Also implemented but **off after losing their sweeps**: jockey/trainer Glicko, per-dimension Glicko, margin-weighted Glicko, Glicko-2 volatility, collateral form (`COLLATERAL_FORM`). The generic engine is `_glicko_pass` (arbitrary entity keying; same-entity runners never play each other).
*   **FE constants** (`ELO_K_BASE=16`, `MARGIN_ELO_SCALE=3`, `TE_EWMA_HALF_LIFE_RACES=20`) were chosen by walk-forward sweep (`scripts/fe_sweep.py`, 2026-06-11) — re-sweep when meaningful new features land.
*   **CRITICAL Context:**
    *   **Pandas GroupBy Pitfall**: *Never* use `df.groupby(key)[col].cummax().shift(1)`. The `.shift(1)` behaves globally in pandas, causing catastrophic data leakage across different horses/races. Always use `transform(lambda x: x.cummax().shift(1))` instead. An AST-based lint (`tests/test_leakage_patterns.py`) guards this and the leak-equivalent spellings (`.transform("cummax").shift(1)`, `.rolling(3).mean().shift(1)`) across `src/`, `scripts/` and the apps.
    *   **Elo Nulls / Non-Finishers Handling**: Custom logic exists for how non-finishers (falls, unseated riders) are penalized in historical rating datasets.
    *   **Scrape-time columns are post-race:** `horse_runs`/`horse_wins`/`horse_places` from the API describe the horse AFTER the race; `*_freq` counts span the whole dataset (future included). Both are explicitly in `EXCLUDE_COLUMNS` — never feed them to a model.
    *   **Same-day visibility (date-strict rule):** jockeys/trainers run many races per day, but at prediction time the day's earlier results don't exist. Every jockey/trainer/combo-keyed history feature must therefore be **date-strict** — all of an entity's runs on one day share the value at its first run of the day. Implemented via `_entity_day_start_view` (feature_engineer) and `_day_start_gather` (ratings). Guarded by `tests/test_live_parity.py`, which runs the training path and the live path (`feature_engineer_with_history_core`) on the same data and requires identical features — run it after ANY new entity-history feature.
    *   **Sectional metrics are sparse by design** (jump metrics don't exist for flat races); they are left as NaN with a `has_atr_history` flag — do not reintroduce global median fills. The source is At The Races / TPD (`src/attheraces_scraper.py` → `add_atr_features`); the older RacingTV stage was deleted on 2026-10-13.

### 2.4 Modeling & Tuning (`train.py`, `src/model.py`, `src/autotune.py`)
*   **Status:** Central to project. Focus on tree-based ensembles.
*   **Description:** `src/model.py` defines `RacePredictor` (formerly `TripleEnsemblePredictor` — an alias is kept so old pickles load): two independently trained, Platt+isotonic-calibrated classifiers (Win and Place) that drive all betting, plus logistic-regression + market (overround-normalised 1/odds) baselines reported in metrics as sanity references. (A LambdaRank ranker was removed in 2026-06 — it was diagnostics-only, never blended into win probabilities, and cost ~⅓ of training time; the win classifier is still evaluated *as a ranker* via `_evaluate_as_ranker` for NDCG/top-1.) Optuna is heavily utilized for tuning. The test-set betting simulation (`analyse_test_set`) lives in `src/bet_analysis.py`; `src/model.py` re-exports it for backwards compatibility.
*   **Market anchor (Benter combination):** after calibration, win probabilities are combined with the market via `softmax(alpha*log(p_model) + beta*log(p_market))` per race, with (alpha, beta) MLE-fitted on OOF predictions (`config.MARKET_ANCHOR`). The fitted **alpha is the scoreboard for fundamental-model progress**: it first went positive (+0.079) in run `20260612_172827` after the 2026-06-11/12 feature work. Races with incomplete odds fall back to unanchored probabilities.
*   **Negative results worth not repeating:** a conditional-logit / Plackett-Luce (grouped-softmax) LightGBM objective lost to the plain binary classifier on both no-market and with-market protocols (2026-06-12, script since deleted); the dtype-filter fix that admitted ~160 uint8 one-hot/flag columns measured neutral.
*   **Checkpoints:** Models are saved straight to `models/` (e.g., `horse_race_model.joblib`, `ensemble_models.joblib`). All runs stream metrics to `data/runs/`.

### 2.5 Backtesting & Strategy execution (`src/backtester.py`, `src/wf_eval.py`, `src/matchbook_client.py`)
*   **Status:** Functional, but often requires iterative refinements based on exchange behavior.
*   **Description:** Includes `backtester.py` for historical simulation, computing PnL, and ROI. `matchbook_signals.py` translates model probabilities into actual market actions against Matchbook APIs. Contains logic for Each-Way betting (`src/each_way.py`) and paper trading (`src/paper_trade_store.py`).
*   **CRITICAL Context:** All bet-selection and settlement rules (dynamic value threshold, each-way odds band, EW PnL maths) live in **`src/bet_settlement.py`** — the single source of truth shared by `analyse_test_set`, the walk-forward backtester, and the live Today's Picks settlement. Never inline these rules at a call site; if the backtest and live settlement use different formulas, every reported ROI becomes untrustworthy. Known intentional difference: the backtester computes value edge against race-normalised (overround-corrected) implied probabilities, while `analyse_test_set` uses raw `1/odds` (passed explicitly via the `implied_prob` argument).

    There are **two walk-forward implementations** and they are not interchangeable: `src/backtester.walk_forward_validation` is the app-facing simulation (per-fold model fits, bet settlement, the Backtest tab), while `src/wf_eval.py` holds the research primitives (purged expanding folds, out-of-fold second-stage fitting, race-clustered bootstrap) that CLAUDE.md hard rule 6 requires for any new measurement. Use `wf_eval` for research; a hand-rolled fold loop has already produced a biased result that closed a real lead for three months.

## 3. Important Context for Agents

1.  **Strictly Python/Streamlit:** Rely on standard python practices. The project is managed with standard `pip` (`requirements.txt`).
2.  **Performance Constraints:** Features are computed over thousands of races. Numba and optimized pandas operations are favored over loops.
3.  **Data Leakage Prevention:** Always ensure that calculations representing a horse's past performance *strictly* exclude the current race.
5.  **Environment:** Handled via a local `.venv` on Windows (`c:\Users\Leonp\Documents\horse\.venv`).

## 4. Current state & where to go next

**Check the pipeline's state before anything else:**

```
.venv/Scripts/python.exe scripts/project_status.py
```

It reports all four links that must agree — results database → feature store →
trained model → promoted model — and says plainly whether the system can serve
a pick. Exit code 0 means yes. By default it also recomputes the data
signature from the database (about a minute; `--quick` skips it), checks the
store against the database's latest date, and verifies the ATR-cache
fingerprint - the three ways a store went stale with its code unchanged.

**State on 2026-10-13** (see `docs/archive/SESSION_LOG.md` for the detail): database
to 2026-10-12 after a full-history relist restored 543 dropped races; store
rebuilt at `63d88ff721a02632` after fixing the `sex` look-ahead leak (ledger
15); a model retrained on that store is the promoted one (check the status
report - a store rebuild always invalidates the promoted model until it is
retrained).

**Operational rule learned the hard way:** after a sync that adds rows dated
*before* the store's last date (the sync re-lists the last seven days, so a
late-resolving race can do this), rebuild with
`scripts/build_feature_store_batched.py` before training.
`get_featured_dataset` would otherwise detect the data change and attempt its
in-memory full rebuild, which peaks past 16 GB on this machine.

**Where the edge question stands:** decision gate 4 has fired. The
market-offset model (plan 4c) and the free microstructure data (plan 5a) both
fail; see `docs/archive/ROADMAP.md` for the options.

### 4.1 The honest position on edge

**No edge has survived honest out-of-sample validation.** The model does not
beat Betfair SP. Read **[docs/NEGATIVE_RESULTS.md](../NEGATIVE_RESULTS.md)**
before proposing anything — it records what was built, measured and refuted,
including several ideas that first appeared to work:

- the "+15% place-market edge" was a **stale feature store**;
- a "+56.8% ROI" was the **`trainer_freq` leak**;
- the Benter market combination is **negative, measured twice** (−2.64% then
  −3.55% on a corrected harness, both CIs excluding zero, both losing to
  simply backing everything in the band);
- sale price, wind surgery, trainer travel, jockey booking, stewards reports
  and par-time speed figures are each **fully priced or redundant**.

The pattern is consistent: the feature store is saturated for *public form*,
and the market's advantage is *private* information (stable confidence on
favourites and debutants) that no public source proxies.

### 4.2 Ranked next steps

**Superseded on 2026-10-13 by [docs/archive/ROADMAP.md](ROADMAP.md):** items 1-3
below are done or measured (the instrument is restored; microstructure in its
free form and the market-offset model both fail, gate 4). The list is kept as
it stood, for the record.

The full version, with success criteria, costs and decision gates, is
**[docs/archive/ACTION_PLAN.md](ACTION_PLAN.md)** — start there. Summary:

1. **Rebuild the measuring instrument first.** Refresh results, rebuild the
   store, retrain, promote — then `project_status.py` goes green. Until then
   every number produced is untrustworthy, which is how two phantom edges
   happened.
2. **Market microstructure** — the one untapped orthogonal signal. Traded
   volume and ladder depth from Betfair historic PRO files (we currently hold
   only the price trajectory `ltp_*`/`bsp`). Where money enters and how late is
   the closest available proxy for the private information that beats us.
3. **Race-aware modelling** — conditional logit / softmax over the field, or
   learning-to-rank, instead of a per-horse binary classifier plus
   normalisation; and place probabilities derived from win probabilities
   (Harville) rather than a second classifier. Note a Plackett-Luce objective
   already lost to the plain binary classifier once (2026-06-12).
4. **Prune the feature set.** 151 of ~1178 stored columns are allowlisted;
   adding the unused 173 produced no CLV. Out-of-sample ablation under a locked
   walk-forward would likely shrink the set without loss.
5. **Matchbook automation** — robustness of live trading rules, API rate
   limits, liquidity estimation. Only worth effort *after* CLV > 0.

Shared fold/CI/stacking primitives now live in **`src/wf_eval.py`**; use them
instead of hand-rolling a fold loop. Settlement maths lives in
`src/bet_settlement.py` (hard rule #2) and every harness now calls it.

### 4.3 Known dead code / loose ends

Cleared on 2026-10-13 — 32 files and ≈5,200 lines removed; see §2.0 table C and
`docs/NEGATIVE_RESULTS.md` 12 for the item-by-item justification. What remains:

- **`src/sibling_features.py` is unwired by design.** It is screened against the
  database directly so it can be judged before costing a rebuild. Wiring it in
  means registering it in `_FE_SOURCE_FILES` (hard rule 4).
- **The weather and trainer-travel FE stages were removed on 2026-10-13**
  after a parity check (ledger 13): the store no longer calls Open-Meteo at
  build time, and `weather.py`, `trainer_locations.py` and
  `bha_stewards_scraper.py` no longer sit in `_FE_SOURCE_FILES`.
- **`scripts/fe_sweep.py` points `BASE_RUN` at a run snapshot that no longer
  exists.** Change the id before using it.
- **2 of 8 saved runs carry no feature signature**, so
  `run_store.feature_generation_status` reports `unknown` for them.
- **`data/rtv_cache/` is in use again** by the rebuilt Racing TV scraper (2026-10-29); the old `rtv_metrics.parquet` inside it is kept.
- **Three single functions are unreferenced and were deliberately kept**, because
  a reference count is not a reason to strip a module's API:
  `bet_settlement.settle_place_exchange_bets` (the vectorised twin of a formula
  hard rule 2 wants centralised), `feature_engineer._cumulative_plain` (part of
  the incremental-append machinery a later FE phase needs) and
  `utils.print_race_prediction`. The last two sit in `_FE_SOURCE_FILES`, so
  removing them costs a store rebuild. `sibling_features.add_sibling_features`
  was removed on 2026-11-03 after the sibling lead failed on the hold-out
  (ledger 2h), along with the focal-loss classes no saved run still pickles.

### 4.4 Guards you should not weaken

These exist because each one corresponds to a bug that actually shipped:

| Guard | Protects against |
|---|---|
| `tests/test_leakage_patterns.py` | grouped-then-globally-shifted history features (AST-based; covers `src/`, `scripts/` and the apps) |
| `tests/test_fe_signature_coverage.py` | an FE dependency not hashed into `fe_signature` → silently stale features |
| `tests/test_feature_generation_contract.py` | promoting/serving a model against a feature generation it was not trained on |
| `tests/test_live_parity.py` | training-path vs live-path feature divergence |
| `tests/test_project_status.py` | the status report claiming "ready" when it is not |
| `tests/test_wf_eval.py` | folds without an embargo, second stages fitted in-sample, CIs that ignore race clustering |
| `tests/test_benter_harness.py` | a research harness biased against the hypothesis it is testing |
| `tests/test_birth_sex.py`, `scripts/audit_field_timing.py` | a raw field recorded at scrape time leaking each horse's future (no feature-level guard can see it) |
| `wf_eval.development_mask` (used by every research harness, triage and the app backtester) | spending the golden hold-out before a final, single test |
