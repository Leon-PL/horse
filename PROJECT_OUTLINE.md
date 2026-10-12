# Horse Racing Machine Learning & Betting System
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

### 2.1 User Interface (`picks_app.py`, `training_app.py` & `config.py`)
*   **Status:** Functional, actively maintained.
*   **Description:** The UI is split into two Streamlit processes so the day-to-day serving app stays fast (the old monolithic `app.py` was retired). Pages are selected via a sidebar radio, so only the active page's section executes per rerun. The cached run-store wrappers, chart builders and diagnostic panels live in `src/app_helpers.py`.
    *   **`picks_app.py` (serving)** — loads only the production-promoted model via the registry pointer (`src/run_store.load_production_predictor` / `production.json`); it never loads the full feature store at startup. Pages: `💰 Today's Picks` (high-value predictions matched to today's races, plus manual custom-race entry) and `🔌 Matchbook API` (Matchbook connectivity harness).
    *   **`training_app.py` (training/analysis)** — heavy, run on a separate port. A thin Streamlit shell over the callable `src/training` modules. Pages: `📦 Data` (feature-store inspect/refresh), `🤖 Train` (configure frameworks/hyperparameters, run `train_run`, review calibration/ROI/CLV evals), `🗂️ Registry` (browse saved runs, promote one to production).
*   **Recent Changes:** Training pipeline rebuilt as a callable core (`src/training.train_run`) with a lean run registry (no per-run featured/processed parquet snapshots — reproduce from the store) and a production pointer for promotion. Serving and training were split into two apps. H2O AutoML and FLAML were previously removed from the UI and dependencies. Sub-model frameworks are selectable per role (`config.SUB_MODEL_FRAMEWORKS`); the current defaults are LightGBM for both win and place classifiers.

### 2.2 Data Pipeline & Scraping (`src/data_*.py`, `src/database.py`, `src/rtv_scraper.py`, `src/pedigree_backfill.py`)
*   **Status:** Stable / In Production.
*   **Description:** Extracts race results and live racecards. Includes caching logic (`data/raw/`, `data/racecards_cache/`) to limit network usage and API bans.
*   **Notable modules:** `database.py` manages historical data access; `attheraces_scraper.py` fetches At The Races / TPD sectionals (needs `curl_cffi` or the system `curl` for the Fastly challenge) and is the live sectional source; `rtv_scraper.py` (Racing TV) is retired. Also present: `bha_stewards_scraper.py` / `ihrb_daily_enquiries_scraper.py` (stewards reports), `tattersalls_sales_scraper.py` (sale prices), `betfair.py` (BSP/LTP joins), `weather.py`, `trainer_locations.py`.
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
    *   **Sectional metrics are sparse by design** (jump metrics don't exist for flat races); they are left as NaN with a `has_atr_history` flag — do not reintroduce global median fills. The source is now At The Races / TPD (`src/attheraces_scraper.py` → `add_atr_features`); the older RacingTV stage (`add_rtv_features`) is retired and no longer in the pipeline.

### 2.4 Modeling & Tuning (`train.py`, `src/model.py`, `src/autotune.py`)
*   **Status:** Central to project. Focus on tree-based ensembles.
*   **Description:** `src/model.py` defines `RacePredictor` (formerly `TripleEnsemblePredictor` — an alias is kept so old pickles load): two independently trained, Platt+isotonic-calibrated classifiers (Win and Place) that drive all betting, plus logistic-regression + market (overround-normalised 1/odds) baselines reported in metrics as sanity references. (A LambdaRank ranker was removed in 2026-06 — it was diagnostics-only, never blended into win probabilities, and cost ~⅓ of training time; the win classifier is still evaluated *as a ranker* via `_evaluate_as_ranker` for NDCG/top-1.) Optuna is heavily utilized for tuning. The test-set betting simulation (`analyse_test_set`) lives in `src/bet_analysis.py`; `src/model.py` re-exports it for backwards compatibility.
*   **Market anchor (Benter combination):** after calibration, win probabilities are combined with the market via `softmax(alpha*log(p_model) + beta*log(p_market))` per race, with (alpha, beta) MLE-fitted on OOF predictions (`config.MARKET_ANCHOR`). The fitted **alpha is the scoreboard for fundamental-model progress**: it first went positive (+0.079) in run `20260612_172827` after the 2026-06-11/12 feature work. Races with incomplete odds fall back to unanchored probabilities.
*   **Negative results worth not repeating:** a conditional-logit / Plackett-Luce (grouped-softmax) LightGBM objective lost to the plain binary classifier on both no-market and with-market protocols (`scripts/_ab_pl_objective.py`, 2026-06-12); the dtype-filter fix that admitted ~160 uint8 one-hot/flag columns measured neutral.
*   **Checkpoints:** Models are saved straight to `models/` (e.g., `horse_race_model.joblib`, `ensemble_models.joblib`). All runs stream metrics to `data/runs/`.

### 2.5 Backtesting & Strategy execution (`src/backtester.py`, `src/strategy_calibrator.py`, `src/matchbook_client.py`)
*   **Status:** Functional, but often requires iterative refinements based on exchange behavior.
*   **Description:** Includes `backtester.py` for historical simulation, computing PnL, and ROI. `matchbook_signals.py` translates model probabilities into actual market actions against Matchbook APIs. Contains logic for Each-Way betting (`src/each_way.py`) and paper trading (`src/paper_trade_store.py`).
*   **CRITICAL Context:** All bet-selection and settlement rules (dynamic value threshold, each-way odds band, EW PnL maths) live in **`src/bet_settlement.py`** — the single source of truth shared by `analyse_test_set`, the walk-forward backtester, and the live Today's Picks settlement. Never inline these rules at a call site; if the backtest and live settlement use different formulas, every reported ROI becomes untrustworthy. Known intentional difference: the backtester computes value edge against race-normalised (overround-corrected) implied probabilities, while `analyse_test_set` and the strategy calibrator use raw `1/odds` (passed explicitly via the `implied_prob` argument).

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
a pick. Exit code 0 means yes. As of 2026-10-12 it reports **not ready**: the
results database and feature store are ~114 days behind, the store was built by
older FE code than is now on disk, and **no model is promoted to production**.
The newest saved run (`20260623_223148`) predates both the `trainer_freq` leak
fix and the store rebuild, so it should not be promoted.

### 4.1 The honest position on edge

**No edge has survived honest out-of-sample validation.** The model does not
beat Betfair SP. Read **[docs/NEGATIVE_RESULTS.md](docs/NEGATIVE_RESULTS.md)**
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

The full version, with success criteria, costs and decision gates, is
**[docs/ACTION_PLAN.md](docs/ACTION_PLAN.md)** — start there. Summary:

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

- `add_rtv_features` is retired (superseded by `add_atr_features`) but still
  defined in `feature_engineer.py`; `src/rtv_scraper.py` remains for its cache.
- `src/par_time_features.py` and `src/stewards_features.py` are tested but
  **never called by the pipeline** — both were triaged as redundant. They are
  still hashed into `fe_signature`, so editing them forces a store rebuild.
- `models/` holds March-2026 artifacts (`ranker_model.joblib`,
  `rank_ensemble_models.joblib`, `ensemble_models.joblib`) from the removed
  LambdaRank path. Safe to delete.

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
