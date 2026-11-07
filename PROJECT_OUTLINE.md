# Project outline

The project has four stages:

1. **Collect and aggregate:** `src/ingestion/` coordinates Sporting Life,
   Racing TV, ATR, Betfair BSP and currency references. `src/database.py`
   owns canonical runner records; other sources retain their own identities,
   raw payloads and publication checkpoints.
2. **Build features:** `src/data_processor.py` prepares accepted observations;
   `src/featured_dataset.py` and `src/feature_store.py` own the historical build.
   Feature calculations use earlier races and timestamped pre-race observations.
3. **Train and evaluate:** `src/training.py`, the LightGBM model, calibration
   and evaluation modules produce candidate runs. `src/run_store.py` records
   results and the explicit selected-run pointer.
4. **Predict:** `src/live_prediction.py` uses today's cards and compatible
   historical features. There is no UI: the Streamlit apps were removed on
   2026-11-07.

See the [directory and dependency map](docs/architecture/PROJECT_STRUCTURE.md)
for code ownership, the [operating guide](README.md) for commands, and the
[inventory](docs/PROJECT_INVENTORY.md) for individual tools.

The [older outline](docs/archive/PROJECT_OUTLINE_before_cleanup_2026-11-05.md)
is retained as historical context. Its retired framework and workflow descriptions
do not describe the current application.
