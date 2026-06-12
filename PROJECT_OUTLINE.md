# Horse Racing Machine Learning & Betting System
**Project Handover & Context Document**

This document serves as a high-level overview of the project architecture, current status, and critical context for any AI agent or developer continuing work on the codebase.

## 1. Project Overview
The project is a comprehensive Python-based machine learning pipeline and betting automation system for horse racing. It features data scraping, feature engineering, model training (XGBoost/CatBoost), backtesting, and live Matchbook API integration, all orchestrated through a Streamlit UI.

**Core Tech Stack**
*   **UI Application:** Streamlit (`app.py`)
*   **Modeling:** LightGBM (default framework, incl. focal-loss classifiers), XGBoost, CatBoost, Scikit-learn (Joblib for persistence)
*   **Hyperparameter Tuning:** Optuna
*   **Data Processing:** Pandas, Numba (for high-performance functions)
*   **Integrations:** Matchbook Betting Exchange API
*   **Tests:** Pytest (`tests/`) — settlement maths, each-way rules, merge-key normalisers

## 2. Directory Structure & System Status

### 2.1 User Interface (`app.py` & `config.py`)
*   **Status:** Functional, actively maintained.
*   **Description:** The main entry point is `app.py`; pages are selected via a sidebar radio, so only the active page's section executes per rerun. The cached run-store wrappers, chart builders and diagnostic panels live in `src/app_helpers.py`. The page sections:
    *   `🎓 Train & Tune`: Core pipeline execution.
    *   `🧭 Autotune`: Hyperparameter optimization wrapper.
    *   `🧪 Experiments`: Run tracking and MLFlow-style experiment viewing.
    *   `🔮 Predict`: Serving predictions for specific race cards.
    *   `💰 Today's Picks`: High-value predictions matched to today's races.
    *   `🔌 Matchbook API`: Live connection to Matchbook for pulling odds and placing bets.
    *   `🔎 Shortcomings`: Analytical tools for model weaknesses.
*   **Recent Changes:** H2O AutoML and FLAML were cleanly removed from the UI and dependencies to simplify the modeling pipeline. Sub-model frameworks are selectable per role (`config.SUB_MODEL_FRAMEWORKS`); the current defaults are LightGBM for both win and place classifiers.

### 2.2 Data Pipeline & Scraping (`src/data_*.py`, `src/database.py`, `src/rtv_scraper.py`, `src/pedigree_backfill.py`)
*   **Status:** Stable / In Production.
*   **Description:** Extracts race results and live racecards. Includes caching logic (`data/raw/`, `data/racecards_cache/`) to limit network usage and API bans.
*   **Notable modules:** `rtv_scraper.py` handles Racing TV data; `database.py` manages historical data access.
*   **Pedigree:** the results/racecard APIs never include breeding, so the `sire`/`dam`/`damsire` columns in `results` are always null at scrape time. `src/pedigree_backfill.py` fills a `horse_pedigree` table from Sporting Life horse *profile* pages (keyed by horse_id; pedigree stored by NAME because the profile JSON's nested ids are buggy). `data_processor.process_data` joins it in; `sync_database` fetches profiles for first-time runners automatically. Full historical backfill completed 2026-06-12 (70k+ horses).
*   **Opening odds:** result pages carry the opening price in `bet_movements` ("op 8/11"); the scraper stores it as `opening_odds` (analysis-only column for market-drift / closing-line-value work — never a model feature). `scripts/backfill_opening_odds.py` UPDATEs it onto already-stored rows.

### 2.3 Feature Engineering & Ratings (`src/feature_engineer.py`, `src/ratings.py`, `src/data_processor.py`)
*   **Status:** Active Development / Needs careful handling.
*   **Description:** Transforms raw horse racing outcomes into lag features, rolling statistics, and rating systems.
*   **Rating systems** (`src/ratings.py`): adaptive-K Elo (horse/jockey/trainer + per-dimension variants + margin Elo), Glicko-1 (horse; rating deviation inflates with layoffs) and TrueSkill (n-way free-for-all mu/sigma; `TRUESKILL_ENABLED`, on by default since the 2026-06-12 sweep). Also implemented but **off after losing their sweeps**: jockey/trainer Glicko, per-dimension Glicko, margin-weighted Glicko, Glicko-2 volatility, collateral form (`COLLATERAL_FORM`). The generic engine is `_glicko_pass` (arbitrary entity keying; same-entity runners never play each other).
*   **FE constants** (`ELO_K_BASE=16`, `MARGIN_ELO_SCALE=3`, `TE_EWMA_HALF_LIFE_RACES=20`) were chosen by walk-forward sweep (`scripts/fe_sweep.py`, 2026-06-11) — re-sweep when meaningful new features land.
*   **CRITICAL Context:** 
    *   **Pandas GroupBy Pitfall**: *Never* use `df.groupby(key)[col].cummax().shift(1)`. The `.shift(1)` behaves globally in pandas, causing catastrophic data leakage across different horses/races. Always use `transform(lambda x: x.cummax().shift(1))` instead. A static lint test (`tests/test_leakage_patterns.py`) guards this pattern.
    *   **Elo Nulls / Non-Finishers Handling**: Custom logic exists for how non-finishers (falls, unseated riders) are penalized in historical rating datasets.
    *   **Scrape-time columns are post-race:** `horse_runs`/`horse_wins`/`horse_places` from the API describe the horse AFTER the race; `*_freq` counts span the whole dataset (future included). Both are explicitly in `EXCLUDE_COLUMNS` — never feed them to a model.
    *   **Same-day visibility (date-strict rule):** jockeys/trainers run many races per day, but at prediction time the day's earlier results don't exist. Every jockey/trainer/combo-keyed history feature must therefore be **date-strict** — all of an entity's runs on one day share the value at its first run of the day. Implemented via `_entity_day_start_view` (feature_engineer) and `_day_start_gather` (ratings). Guarded by `tests/test_live_parity.py`, which runs the training path and the live path (`feature_engineer_with_history_core`) on the same data and requires identical features — run it after ANY new entity-history feature.
    *   **RTV metrics are sparse by design** (jump metrics don't exist for flat races); they are left as NaN with a `has_rtv_history` flag — do not reintroduce global median fills.

### 2.4 Modeling & Tuning (`train.py`, `src/model.py`, `src/autotune.py`)
*   **Status:** Central to project. Focus on tree-based ensembles.
*   **Description:** `src/model.py` defines `RacePredictor` (formerly `TripleEnsemblePredictor` — an alias is kept so old pickles load): two independently trained, Platt+isotonic-calibrated classifiers (Win and Place) that drive all betting, an opt-in LambdaRank ranker (`config.TRAIN_RANKER`, default off) trained for diagnostics only (not blended into win probabilities), and logistic-regression + market (overround-normalised 1/odds) baselines reported in metrics as sanity references. Optuna is heavily utilized for tuning. The test-set betting simulation (`analyse_test_set`) lives in `src/bet_analysis.py`; `src/model.py` re-exports it for backwards compatibility.
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

## 4. Potential Areas for Improvement
*   **Each-way edge validation:** test-set simulations repeatedly show EW ROI > +100% on ~130-bet samples — the only strategy consistent with a small model edge, since it exploits mechanical 1/4-odds place terms. A monthly walk-forward (`scripts/wf_each_way.py`) exists to confirm/refute across folds before real money.
*   **Closing-line value:** `opening_odds` is now collected and backfilled; compare model-flagged bets' opening vs SP prices to measure whether the model beats the *early* market (the realistic edge, vs out-informing SP).
*   **Pedigree feature depth:** breeding data only became real on 2026-06-12; sire/damsire aptitude features (going/trip/all-weather) are most valuable in low-information races (2yos, maidens).
*   **Matchbook Automation:** Enhancing the robustness of live trading rules, handling API rate limits, and improving liquidity estimations strings.
*   **UI Responsiveness:** Caching more heavy Streamlit functions to keep UI fast while computing inference.