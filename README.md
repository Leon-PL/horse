# 🏇 Horse Racing Prediction System

An ML-powered application that predicts UK & Ireland horse race outcomes using **real data scraped from Sporting Life**. Built with Python, XGBoost, LightGBM, and Streamlit.

The flagship model is a **6-model ensemble** combining Learning-to-Rank, regression, classification, and pairwise comparison — with Optuna-learned blending weights, race-coherent temperature probability calibration, and per-sub-model framework selection (XGBoost or LightGBM).

> ⚠️ **Disclaimer**: This application is for educational and entertainment purposes only. It is not financial advice. Gambling involves risk — always bet responsibly.

---

## Features

- **Real Data — No Accounts or Keys Needed** — Scrapes live racecards and historical results directly from [Sporting Life](https://www.sportinglife.com), including SP odds, jockey, trainer, form, official ratings, and more
- **120+ Predictive Features** — Horse form, jockey/trainer stats, jockey-trainer combos, C&D winners, class changes, headgear signals, track/going/distance preferences, speed figures, market indicators, field quality metrics
- **Elo Ratings** — Dynamic head-to-head strength ratings for horses, jockeys, and trainers with adaptive K-factors (if A beats B, and B beats C → A is rated above C)
- **6-Model Ensemble** — LTR Ranker + LB Regressor + Win Classifier + Pairwise Classifier + Place Classifier + Norm-Pos Regressor, blended with Optuna-learned weights
- **Temperature Probability Calibration** — Out-of-sample race-coherent probability scaling fitted on purged CV folds
- **Framework Selection** — Each sub-model can independently use XGBoost or LightGBM — configurable via UI, CLI, or `config.py`
- **Brier Score Optimisation** — All ensemble weight learning targets Brier Score for value betting alignment
- **Learning-to-Rank** — XGBRanker and LGBMRanker rank horses *within each race* (eliminates the class imbalance problem)
- **Early Stopping** — All sub-models use validation-fold early stopping (50 rounds) to prevent overfitting
- **Interactive Web UI** — Streamlit dashboard with 7 pages: Train & Tune, Experiments, Predict, Today's Picks, Data Explorer, Model Insights, Guide
- **Today's Races** — Predict live UK & Ireland racecards with win probabilities, real odds, and value indicators
- **Value Betting Detection** — Dynamic odds-adjusted threshold highlights horses where the model's probability exceeds the market's implied probability
- **Walk-Forward Backtesting** — Expanding-window validation with P&L simulation, equity curves, and per-fold metrics
- **Historical Database** — SQLite-backed persistent storage; only scrapes new days you're missing, so retraining is fast
- **Auto-Tuning** — Optuna hyperparameter search with configurable trial count
- **Experiment Tracking** — Save, compare, and replay training runs
- **Sample Data Fallback** — Synthetic data generator for quick offline testing

---

## Project Structure

```
horse/
├── app.py                          # Streamlit web application (7 pages)
├── train.py                        # Command-line training script
├── trial_run.py                    # Quick trial: scrape today + predict
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_scraper.py             # Web scraper (Sporting Life)
│   ├── data_collector.py           # Unified data collector interface
│   ├── data_collector_real.py      # Real data via Sporting Life scraper
│   ├── data_collector_sample.py    # Synthetic data generator (fallback)
│   ├── database.py                 # Historical SQLite database
│   ├── data_processor.py           # Data cleaning & preprocessing
│   ├── feature_engineer.py         # Feature engineering pipeline (120+ features)
│   ├── ratings.py                  # Elo rating system (horse/jockey/trainer)
│   ├── model.py                    # ML models, training, evaluation, tuning
│   ├── backtester.py               # Walk-forward validation & P&L simulation
│   ├── run_store.py                # Experiment persistence (run snapshots)
│   └── utils.py                    # Utility functions
├── data/
│   ├── races.db                    # Historical SQLite database
│   ├── raw/                        # Raw collected data (CSV)
│   └── processed/                  # Processed & feature-engineered data
├── models/                         # Saved trained models
└── runs/                           # Experiment run snapshots
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd horse
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# RECOMMENDED — 6-model ensemble with database source
python train.py --model triple_ensemble --source database --days-back 90

# First run scrapes all 90 days; subsequent runs only fetch new days
python train.py --model triple_ensemble --source database --days-back 90

# Train with all sub-models using LightGBM
python train.py --model triple_ensemble --source database --days-back 90 \
  --frameworks ltr=lgbm,regressor=lgbm,classifier=lgbm,pairwise=lgbm,place=lgbm,norm_pos=lgbm

# Mix frameworks: LGBM ranker + XGB classifiers/regressors (default)
python train.py --model triple_ensemble --source sample --races 2000

# Override specific sub-models
python train.py --model triple_ensemble --source sample \
  --frameworks classifier=lgbm,pairwise=lgbm

# Train + run walk-forward backtest
python train.py --model triple_ensemble --source database --days-back 90 --backtest

# Train with sample (synthetic) data — instant, no network needed
python train.py --source sample --races 3000

# Standalone Learning-to-Rank models
python train.py --source sample --races 2000 --model xgb_ranker
python train.py --source sample --races 2000 --model lgbm_ranker
python train.py --source sample --races 2000 --model rank_ensemble

# Retrain on already-downloaded data
python train.py --skip-collection --model triple_ensemble
```

### 3. Launch the Web App

```bash
streamlit run app.py
```

This opens an interactive dashboard with 7 pages:

| Page | Description |
|------|-------------|
| **🎓 Train & Tune** | Scrape data, configure frameworks & hyperparameters, train the ensemble from the UI |
| **🧪 Experiments** | Compare saved training runs side-by-side |
| **🔮 Predict** | Paste a race URL or enter horses manually for prediction |
| **💰 Today's Picks** | Auto-scrape current UK racecards with live odds and value picks |
| **📊 Data Explorer** | Charts and statistics from the training dataset |
| **📈 Model Insights** | Feature importance, SHAP explanations, overfit diagnostics, backtesting with P&L charts |
| **📖 Guide** | In-app documentation |

### 4. Quick Trial Run

```bash
# Scrape today's racecards and predict every race (uses saved model)
python trial_run.py
```

---

## How It Works

### Data Pipeline

```
Sporting Life  →  Scraper  →  Processor  →  Feature Engineer  →  6-Model Ensemble
(sportinglife.com)  (__NEXT_DATA__ JSON)    (cleaning, encoding)   (120+ features)     (blended probabilities)
```

1. **Collection** — The scraper fetches UK & Ireland race results from Sporting Life. Each page embeds full race data as `__NEXT_DATA__` JSON (no JavaScript execution required). For each race it extracts every runner's finishing position, SP odds, jockey, trainer, form, weight, age, draw, official rating, and lifetime stats.

2. **Processing** — Cleans data types, parses form strings into numeric features, handles missing values, encodes categoricals, and extracts temporal features.

3. **Feature Engineering** — Creates 120+ features per runner across 16 categories:

| Category | Example Features | Description |
|----------|-----------------|-------------|
| Horse Form | `horse_avg_pos_3/5/10`, `horse_wins_3/5/10` | Rolling performance over last N races |
| Form String | `form_last_pos`, `form_avg`, `form_wins_count` | Parsed from form figures (e.g. "2142-31") |
| Horse History | `horse_win_rate`, `horse_prev_races` | Lifetime cumulative statistics |
| Consistency | `horse_pos_consistency`, `horse_best/worst_pos_5` | How reliable the horse is |
| Freshness | `days_since_last_race` | Days since last run |
| Jockey | `jockey_win_rate`, `jockey_place_rate` | Jockey track record in dataset |
| Trainer | `trainer_win_rate`, `trainer_place_rate` | Trainer statistics |
| Jockey–Trainer | `jt_win_rate`, `jt_place_rate`, `jt_prev_runs` | Partnership synergy stats |
| Jockey–Track | `jockey_track_win_rate` | Jockey performance at specific venues |
| C&D Winner | `horse_cd_winner`, `horse_cd_win_rate` | Won at same course **and** distance before |
| Course Winner | `horse_course_winner` | Won at this track before |
| Class Change | `class_change`, `class_dropped`, `class_raised` | Class movement between runs |
| Headgear | `has_headgear`, `first_time_headgear`, `headgear_off` | Equipment signals (blinkers, visor, etc.) |
| Speed Figures | `speed_best`, `speed_recent`, `speed_rolling_avg` | Performance speed ratings |
| Surface | `surface_type` | Track surface features |
| Elo Rating | `horse_elo`, `horse_elo_vs_field`, `horse_elo_rank` | Dynamic head-to-head strength (transitive) |
| Jockey Elo | `jockey_elo`, `jockey_elo_vs_field` | Jockey strength propagated through results |
| Combined Elo | `combined_elo`, `combined_elo_vs_field` | Horse + 0.3×Jockey composite strength |
| Track Pref | `horse_track_win_rate` | Performance at specific venues |
| Going Pref | `horse_going_win_rate` | Performance on ground conditions |
| Distance Pref | `horse_dist_win_rate` | Performance at race distances |
| Market | `implied_prob`, `odds_rank`, `is_favourite` | SP odds-based features |
| Race Context | `draw_pct`, `weight_vs_field`, `field_size` | Race conditions |
| Field Quality | `field_avg_elo`, `field_strength` | Strength of competition |
| Interactions | Cross-feature combination features | Non-linear interactions |

4. **Model Training** — The 6-model ensemble trains in two phases:
  - **Phase 1 (Weight Learning):** Sub-models train on purged expanding-window CV folds. A meta-learner/weighting layer is learned from OOF predictions targeting **Brier Score**, then a race-level temperature calibrator is fitted on OOF probabilities.
   - **Phase 2 (Final Models):** All sub-models retrain on 100% of training data using the locked-in weights.

### The 6-Model Ensemble

The `TripleEnsemblePredictor` combines six diverse sub-models, each capturing a different aspect of race prediction:

| # | Sub-Model | Task | Framework | What It Learns |
|---|-----------|------|-----------|----------------|
| 1 | **LTR Ranker** | Learning-to-Rank | LGBM (default) | Within-race ordering via `lambdarank` — no class imbalance |
| 2 | **LB Regressor** | Regression | XGB (default) | Lengths behind the winner — continuous distance measure |
| 3 | **Win Classifier** | Binary classification | XGB (default) | Win probability with class-weighted loss |
| 4 | **Pairwise Classifier** | Binary classification | XGB (default) | "Is horse A better than horse B?" on feature differences |
| 5 | **Place Classifier** | Binary classification | XGB (default) | Top-3 finish probability |
| 6 | **Norm-Pos Regressor** | Regression | XGB (default) | Normalised finishing position (0=win, 1=last) |

**Default ensemble weights** (before Optuna tuning): LTR=0.30, Reg=0.15, Clf=0.15, Pair=0.15, Place=0.10, NormPos=0.15

**Blending:** Each sub-model's raw scores are min-max normalised per race, then combined using learned weights/meta-learner. The blended scores pass through a softmax layer with global + learned calibration temperature to produce final calibrated win probabilities.

**Framework selection:** Each sub-model can independently use XGBoost or LightGBM. Change defaults in `config.py` under `SUB_MODEL_FRAMEWORKS`, or override via the Streamlit UI or `--frameworks` CLI flag.

### Other Models

| Model | Description |
|-------|-------------|
| **XGB Ranker** | Standalone XGBoost `rank:pairwise` |
| **LGBM Ranker** | Standalone LightGBM `lambdarank` |
| **Rank Ensemble** | 50/50 blend of XGB + LGBM rankers |

### Prediction Output

For each horse in a race, the model outputs:

- **Win Probability** — Calibrated probability of winning (0–100%)
- **Predicted Rank** — Horses ranked by calibrated probability
- **Value Score** — `model_prob − implied_prob` — positive = potential value bet ⭐

### Walk-Forward Backtesting

Standard train/test splits can overestimate real-world performance. Walk-forward validation simulates live deployment:

```
Fold 1:  Train [Month 1–2]  →  Test [Month 3]
Fold 2:  Train [Month 1–3]  →  Test [Month 4]
Fold 3:  Train [Month 1–4]  →  Test [Month 5]
...
```

Each fold trains a fresh model on all data *before* the test period, then simulates betting on the test month with two strategies:

| Strategy | Rule | What it measures |
|---|---|---|
| **Top Pick** | Bet the model's #1 pick in every race | Can the model find winners? |
| **Value** | Bet when `model_prob > implied_prob` (dynamic odds-adjusted threshold) | Can the model find mis-priced horses? |

The backtest tracks cumulative P&L, strike rate, ROI, and Brier Score per fold.

```bash
# Run backtest after training
python train.py --model triple_ensemble --source database --days-back 90 --backtest

# Standalone backtester with custom settings
python -m src.backtester --model triple_ensemble --min-train-months 3

# Backtest with a standalone ranker
python -m src.backtester --model xgb_ranker
python -m src.backtester --model lgbm_ranker
```

### Elo Rating System

Traditional per-horse win-rate features can't capture **transitive strength**. If Horse A beats Horse B, and Horse B beats Horse C, then A should be rated above C — even if they never met.

The system maintains dynamic Elo ratings (similar to chess) for **horses**, **jockeys**, and **trainers**:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$

After each race, every pair of finishers is compared and ratings are updated. The K-factor is normalised by field size to prevent big-field races from causing wild swings.

**Features generated:**

| Feature | Description |
|---|---|
| `horse_elo` | Horse's rating entering the race |
| `horse_elo_vs_field` | Rating relative to race average |
| `horse_elo_rank` | Rating rank within the race (1 = highest) |
| `jockey_elo` / `trainer_elo` | Jockey & trainer strength |
| `combined_elo` | Horse + 0.3×Jockey composite |
| `horse_elo_delta` | Rating change after this race |

> Ratings are computed **chronologically** — each race only uses ratings from *before* that race (no look-ahead leakage).

---

## Data Source

### Sporting Life Web Scraper

The scraper extracts data from [sportinglife.com](https://www.sportinglife.com) — a well-known UK racing information site. No accounts, keys, or sign-ups are needed.

**What it scrapes:**

| Data Point | Results | Racecards |
|-----------|---------|-----------|
| Horse name, ID, age, sex | ✅ | ✅ |
| Jockey & trainer | ✅ | ✅ |
| SP / forecast odds | ✅ | ✅ |
| Form figures | ✅ | ✅ |
| Weight (stone-lbs) | ✅ | ✅ |
| Official rating | ✅ | ✅ |
| Headgear (blinkers, visor…) | ✅ | ✅ |
| Days since last run | ✅ | ✅ |
| Lifetime runs / wins / places | ✅ | ✅ |
| Finishing position | ✅ | — |
| Lengths behind winner | ✅ | — |
| Going, distance, class, prize money | ✅ | ✅ |

**How it works:**

Sporting Life is a Next.js application that embeds the full page data in a `<script id="__NEXT_DATA__">` tag. The scraper makes a simple HTTP GET request and parses this JSON — no browser automation or JavaScript execution is required.

**Rate limiting:** The scraper waits 1.5 seconds between requests to be respectful to the server.

### Direct Scraper Usage

```bash
# Scrape results + racecards
python -m src.data_scraper --mode both --days-back 7

# Results only
python -m src.data_scraper --mode results --days-back 14

# Racecards only (today)
python -m src.data_scraper --mode racecards

# Include international tracks
python -m src.data_scraper --mode results --days-back 3 --all-tracks
```

### Historical Database

The system uses a local SQLite database (`data/races.db`) to store every scraped result. On subsequent runs, only **missing days** are fetched — so a 90-day retrain that previously took hours now takes seconds if you trained yesterday.

```bash
# Sync the database (scrape only missing days in the last 90-day window)
python -m src.database --sync 90

# View database statistics
python -m src.database --stats

# Export the full database to CSV
python -m src.database --export data/all_results.csv
```

| First run (empty DB) | Second run (next day) |
|---|---|
| Scrapes all 90 days (~1–2 hours) | Scrapes only 1 new day (~2 min) |
| Stores everything in SQLite | Appends new results, skips duplicates |

---

## Configuration

Edit `config.py` to adjust:

- **Data paths** — Where data, models, and experiment runs are stored
- **Scraper settings** — Request delay, user agent
- **Database** — SQLite path (`data/races.db`), managed automatically
- **Feature settings** — Rolling window sizes, form calculation periods
- **Sub-model frameworks** — `SUB_MODEL_FRAMEWORKS` dict: set each of the 6 sub-models to `"xgb"` or `"lgbm"`
- **Per-sub-model hyperparameters** — Independent parameter dicts for LTR, Regressor, Classifier, Pairwise, Place, and Norm-Pos sub-models
- **Ranker hyperparameters** — XGBoost and LightGBM ranker defaults
- **Softmax temperature** — Controls sharpness of probability output (`<1` = sharper, `>1` = softer)
- **Elo settings** — Base K-factor, minimum K, decay rate per run
- **Training settings** — Test/train split ratio, random seed

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data manipulation | pandas, NumPy |
| ML models | XGBoost, LightGBM, scikit-learn |
| Probability calibration | Race-level temperature scaling |
| Hyperparameter tuning | Optuna |
| Web dashboard | Streamlit |
| Visualizations | Plotly |
| Web scraping | Requests, BeautifulSoup, Selenium |
| Historical database | SQLite (built-in) |
| Model persistence | joblib |

Requires **Python 3.10+**.

---

## License

This project is for educational purposes. Use responsibly.
