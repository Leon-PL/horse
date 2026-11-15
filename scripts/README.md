# Commands

Operating commands. Run from the project root with `.venv\Scripts\python.exe`;
use `--help` before a deliberate backfill or repair.

| Workflow | Command |
|---|---|
| Daily paper record (scheduled 07:50) | `python -m src.matchbook day --refresh`; also `prepare`, `eleven`, `settle`, `summary`; see [MATCHBOOK_PAPER.md](../docs/operations/MATCHBOOK_PAPER.md) |
| Check installed state | `scripts/project_status.py` |
| Open the local UI | `scripts/workbench.py`, then http://127.0.0.1:8765 |
| Collect and publish sources | `python -m src.ingestion --sync 14 --results-only`; `--publish-sources`, `--source-status`, `--enrich` |
| ATR historical backfill | `python -m src.ingestion.atr_rebuild --request-delay 1.5`; check that no worker is already running |
| BHA report catch-up | `python -m src.ingestion.bha_stewards --start YYYY-MM-DD --end YYYY-MM-DD --drain --publish` |
| Betfair historical archives | `python -m src.ingestion.betfair_history --start YYYY-MM-DD --drain`; `setup_betfair_history.ps1 -CreateCertificate` once |
| Refresh the feature store | `python -c "from src.store_refresh import get_training_dataset; get_training_dataset()"`; `build_feature_store_batched.py` for an explicit or replayed build |
| Win model | `python -m src.modelling benchmark / prepare / select / admit / fit`; see [MODELLING.md](../docs/operations/MODELLING.md) |
| Forecast correction | `python -m src.correction seed / evaluate / fit / extend / forecast / status` |
| Tooling | `tooling.ps1 venv`, `setup`, `lint`, `test` |
