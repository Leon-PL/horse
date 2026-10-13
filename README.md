# Horse racing prediction system

Python tools for collecting UK and Ireland race results, building chronological features, training win and place classifiers, and evaluating later races. The main objective is reliable out-of-sample probabilities. Betting simulations are research outputs; the project has not established a profitable betting strategy.

## What actually runs

- **Data:** Sporting Life results and racecards in SQLite, with optional pedigree, sectional, weather, and exchange-price enrichment.
- **Features:** a persisted feature store with historical ratings, form, race context, and date-strict entity statistics. The versioned `feature_sets/win.txt` selects the default model inputs.
- **Models:** separate win and place classifiers. LightGBM is the default; XGBoost and CatBoost are supported. `TripleEnsemblePredictor` remains an alias for loading older artifacts, not a six-model ensemble.
- **Calibration:** Platt plus isotonic remains the default. Race-temperature calibration is an experimental option for new win models. Each model artifact records its fitted method.
- **Evaluation:** chronological holdouts and purged walk-forward folds, with Brier score, log loss, ranking metrics, and betting diagnostics. `src/wf_eval.py` provides shared research evaluation helpers.
- **Serving:** a lightweight picks app loads the explicitly promoted run. Promotion and live features are checked against their recorded feature-generation signatures.

Start with [the pipeline map](PROJECT_OUTLINE.md) (§2.0) for what is live. [The research ledger](docs/NEGATIVE_RESULTS.md) records every hypothesis measured and refuted, with numbers; [the action plan](docs/ACTION_PLAN.md) explains why the model loses to the market and what is left to try.

## Install and inspect

Use a Python version supported by the packages in `requirements.txt`. The existing workspace uses Python 3.14. A virtual environment created elsewhere should be recreated if its base interpreter paths no longer exist.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/project_status.py
```

`project_status.py` checks the database, feature store, saved runs, and production pointer. A stale feature store or an absent promoted model means the serving pipeline is not ready. `data/` and `models/` are local artifacts and are not shipped with a fresh checkout.

## Build and train

For a large history, use the memory-bounded feature builder. It processes whole dates and carries historical state across chunks:

```powershell
.venv\Scripts\python.exe -m scripts.build_feature_store_batched --chunk-rows 120000
.venv\Scripts\python.exe train.py --no-refresh --window-days 600
```

The feature builder reads the existing database. To collect recent missing results first, use the training app's data controls or:

```powershell
.venv\Scripts\python.exe -m src.database --sync 14
```

The callable entry point is `src.training.train_run(TrainingConfig(...))`. The CLI and training app share it. Examples:

```powershell
# Train from current local data, without scraping
.venv\Scripts\python.exe train.py --no-refresh

# Include a walk-forward evaluation using the selected model settings
.venv\Scripts\python.exe train.py --no-refresh --window-days 600 --backtest

# Compare the simpler calibration candidate
.venv\Scripts\python.exe train.py --no-refresh --win-calibration race_temperature

# Override the model framework per role
.venv\Scripts\python.exe train.py --no-refresh --frameworks classifier=lgbm,place=cat
```

The old `--source`, `--model`, `--races`, `--days-back`, and `--skip-collection` training examples no longer describe this CLI. Run `train.py --help` for the supported arguments.

Training saves a run; it does not promote one unless `--promote` is requested. Review the held-out metrics and feature signatures before promoting through the registry. `--blend-market-odds` uses odds in a post-model market blend; historical starting prices do not establish what was available when a live bet would have been placed.

## Applications

```powershell
.venv\Scripts\python.exe -m streamlit run picks_app.py
.venv\Scripts\python.exe -m streamlit run training_app.py --server.port 8502
```

`picks_app.py` serves predictions and the Matchbook connectivity page. `training_app.py` provides data, training, tuning, backtesting, insights, diagnostics, and registry pages. Matchbook credentials are optional and belong in an untracked `.env` file; data collection and offline model evaluation do not require them.

The serving app reloads when the production run changes. Cached race features require matching cards, history, and feature-generation metadata. A changed field is recomputed against stored history. Historical cards must be evaluated against history that ends before their date.

## Validation and research

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
```

The tests cover race/date leakage, full versus incremental feature parity, calibration, model persistence, cache validity, promotion, data integrity, and shared settlement rules.

A standalone calibration benchmark keeps model fitting, early stopping, calibration, and evaluation in separate date ranges with purge gaps:

```powershell
.venv\Scripts\python.exe scripts/benchmark_calibration.py `
  --features data/processed/feature_store/featured.parquet `
  --source-note "Full rebuilt store" `
  --output data/audit/calibration_benchmark.json
```

Use that source note only after a full rebuild has actually completed. The audit's partial feature repair is explicitly diagnostic and cannot validate a production model. Candidate selection uses the earlier development periods before the last confirmation period is evaluated.

Assess calibrated predictions using Brier score and log loss together. ECE depends on binning and can look good for a model that barely distinguishes runners. Treat financial returns and closing-line value as separate questions. Compare models on identical complete races and keep final evaluation data out of feature selection, tuning, and calibration fitting.

## Code map

| Area | Main files |
|---|---|
| Collection and cleaning | `src/data_scraper.py`, `src/database.py`, `src/data_processor.py` |
| Features and historical state | `src/feature_engineer.py`, `src/ratings.py`, `src/fe_incremental.py`, `src/feature_store.py` |
| Cache and provenance | `src/featured_dataset.py`, `src/live_prediction.py` |
| Models and calibration | `src/model.py`, `src/probability_calibration.py` |
| Training and evaluation | `src/training.py`, `src/backtester.py`, `src/wf_eval.py` |
| Run registry | `src/run_store.py` |
| Settlement and market joins | `src/bet_settlement.py`, `src/each_way.py`, `src/betfair.py`, `src/clv.py` |

Historical research scripts and optional unwired feature modules are retained for reproducibility. Their existence does not mean that they participate in the current training pipeline.
