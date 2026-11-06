# Commands

Run commands from the project root with `.venv\Scripts\python.exe`. For example,
`python -m src.ingestion --sync 14` refreshes the normal real-source plan.
Use `--help` before deliberate backfills, repairs or model installation.

| Workflow | Commands |
|---|---|
| Check installed state | `project_status.py`, `validate_data_contracts.py` |
| Collect history | `python -m src.ingestion`; `fetch_betfair_sp_public.py`, `backfill_pedigree.py`, `backfill_rtv_text.py` for explicit specialist jobs |
| Resume ATR five-tab backfill | `python -m src.ingestion.atr_rebuild --request-delay 2`; first check its status so a second worker is not launched; see [tab coverage](../docs/operations/ATR_TAB_COVERAGE_2026-11-06.md) |
| Inspect one ATR jumps result | `python -m src.ingestion.atr_jumps --help`; explicit native race ID and racecard URL; keeps its observation separate from the sectional cache |
| Automatic Betfair archives | Set up **Data → Historical exchange data**, or use `python -m src.ingestion.betfair_history --start YYYY-MM-DD`; `setup_betfair_history.ps1 -CreateCertificate` prepares the local certificate |
| Recover or repair results | `backfill_missing_races.py`, `reconcile_result_fields.py`, `repair_overfull_fields.py`; each publishes the CSV through `publish_after_repair`. Inspect their scope and audit outputs |
| Build historical features | `build_feature_store_batched.py` |
| Prepare dated live features | `build_lookahead.py` |
| Train normal recipe | Root `train.py` or `training_app.py` |
| Research input selection | `select_features.py`, `triage_features.py`, `screen_*.py` |
| New ATR research candidates | `derive_atr_rich_features.py`; writes coverage and candidate Parquets separately from the normal feature store |
| ATR jumping and distance features | `derive_atr_rich_features.py --family new-tabs`; exports prior predictors plus frozen replay inputs; see [definitions](../docs/research/ATR_NEW_TAB_FEATURES_2026-11-06.md) |
| Model experiments | `model_lab.py`, `model_ideas.py`, numbered experiment scripts |
| Audit and compare | `audit_*.py`, `benchmark_*.py`, `compare_*.py`, `verify_*.py` |
| Local dependency tooling | `tooling.ps1`; see [development tools](../docs/operations/DEVELOPMENT_TOOLS.md) |

The [project inventory](../docs/PROJECT_INVENTORY.md) describes individual tools;
[research reports](../docs/research/) record completed experiments. Some research
commands intentionally write or install data/models. Presence here is not an
instruction to rerun them or include their outputs in normal predictions.

`model_lab_slim*.parquet` caches were cleared during cleanup. Recreate a current
slim cache explicitly with `model_lab.py --build-slim-store` before using that
harness. These caches are distinct from the maintained historical feature store.
