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
*   **Description:** The main entry point is `app.py`. It is divided into several tabs:
    *   `🎓 Train & Tune`: Core pipeline execution.
    *   `🧭 Autotune`: Hyperparameter optimization wrapper.
    *   `🧪 Experiments`: Run tracking and MLFlow-style experiment viewing.
    *   `🔮 Predict`: Serving predictions for specific race cards.
    *   `💰 Today's Picks`: High-value predictions matched to today's races.
    *   `🔌 Matchbook API`: Live connection to Matchbook for pulling odds and placing bets.
    *   `🔎 Shortcomings`: Analytical tools for model weaknesses.
*   **Recent Changes:** H2O AutoML and FLAML were cleanly removed from the UI and dependencies to simplify the modeling pipeline. Sub-model frameworks are selectable per role (`config.SUB_MODEL_FRAMEWORKS`); the current defaults are LightGBM for both win and place classifiers.

### 2.2 Data Pipeline & Scraping (`src/data_*.py`, `src/database.py`, `src/rtv_scraper.py`)
*   **Status:** Stable / In Production.
*   **Description:** Extracts race results and live racecards. Includes caching logic (`data/raw/`, `data/racecards_cache/`) to limit network usage and API bans.
*   **Notable modules:** `rtv_scraper.py` handles Racing TV data; `database.py` manages historical data access. 

### 2.3 Feature Engineering & Ratings (`src/feature_engineer.py`, `src/ratings.py`, `src/data_processor.py`)
*   **Status:** Active Development / Needs careful handling.
*   **Description:** Transforms raw horse racing outcomes into lag features, rolling statistics, and Elo ratings. 
*   **CRITICAL Context:** 
    *   **Pandas GroupBy Pitfall**: *Never* use `df.groupby(key)[col].cummax().shift(1)`. The `.shift(1)` behaves globally in pandas, causing catastrophic data leakage across different horses/races. Always use `transform(lambda x: x.cummax().shift(1))` instead.
    *   **Elo Nulls / Non-Finishers Handling**: Custom logic exists for how non-finishers (falls, unseated riders) are penalized in historical rating datasets.

### 2.4 Modeling & Tuning (`train.py`, `src/model.py`, `src/autotune.py`)
*   **Status:** Central to project. Focus on tree-based ensembles.
*   **Description:** The pipeline trains custom ensemble models, rankers, and multi-stage architectures. Optuna is heavily utilized for continuous performance tuning.
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
*   **Feature Engineering Depth:** Continued iteration on track conditions, jockey trajectories, and weather effects (`src/weather.py`).
*   **Matchbook Automation:** Enhancing the robustness of live trading rules, handling API rate limits, and improving liquidity estimations strings.
*   **UI Responsiveness:** Caching more heavy Streamlit functions to keep UI fast while computing inference.