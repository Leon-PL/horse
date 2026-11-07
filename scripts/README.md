# Commands

This folder holds operating commands only. Research, audits and benchmarks live
in [research/](../research/README.md); the earlier ones were retired on
2026-11-07 and are in git at commit `26d1b6f`.

Source publication is part of the collection command: use
`python -m src.ingestion --publish-sources` for local accepted-data adoption and
ATR promotion, and `--source-status` for accepted-version quality. The batched
builder accepts `--dataset-manifest ID` for verified replay into a candidate store.
See [publication operations](../docs/operations/SOURCE_PUBLICATION_2026-11-07.md).

Run commands from the project root with `.venv\Scripts\python.exe`. For example,
`python -m src.ingestion --sync 14` refreshes the normal real-source plan.
Use `--help` before deliberate backfills or repairs.

| Workflow | Commands |
|---|---|
| Check installed state | `project_status.py`, `validate_data_contracts.py` |
| Collect history | `python -m src.ingestion`; `--enrich` drains the pedigree queue; `backfill_rtv_text.py` rebuilds the Timeform comment-score history |
| BHA stewards and full reports | Included in normal ingestion as `bha`. Historical catch-up: `python -m src.ingestion.bha_stewards --start YYYY-MM-DD --end YYYY-MM-DD --drain --publish`; `--status` reads progress; see [collection guide](../docs/operations/BHA_REGULAR_INGESTION_2026-11-07.md) |
| Derive BHA candidates without training | `python -m src.bha_features --historical`; materialises source-backed observations, dated runner candidates and current horse-state candidates, also wired into normal FE; see [definitions](../docs/research/BHA_FEATURE_CANDIDATES_2026-11-07.md) |
| Resume ATR five-tab backfill | `python -m src.ingestion.atr_rebuild --request-delay 2`; first check its status so a second worker is not launched; see [tab coverage](../docs/operations/ATR_TAB_COVERAGE_2026-11-06.md) |
| Audit ATR native IDs | `python -m src.ingestion.atr_ids --out data/audit/atr_id_audit.json`; reads production data without changing it; see [ID repair and speed](../docs/operations/ATR_SPEED_AND_IDS_2026-11-07.md) |
| Inspect one ATR jumps result | `python -m src.ingestion.atr_jumps --help`; explicit native race ID and racecard URL; keeps its observation separate from the sectional cache |
| Automatic Betfair archives | Connect once with `betfair_history.connect_account` (see the [setup guide](../docs/operations/BETFAIR_HISTORY_AUTOMATION.md)), then `python -m src.ingestion.betfair_history --start YYYY-MM-DD --drain --max-files 5000`; `setup_betfair_history.ps1 -CreateCertificate` prepares the local certificate |
| Consolidate older Betfair downloads | `python -m src.ingestion.betfair_archives --remove-verified`; retains raw members once and removes each original only after verification |
| Recover or repair results | `backfill_missing_races.py`, `reconcile_result_fields.py`, `repair_overfull_fields.py`; each publishes the results snapshot through `publish_after_repair`. Inspect their scope and audit outputs |
| Build historical features | `build_feature_store_batched.py` (the build itself is `src/store_build.py`, which training also uses); `build_connection_aliases.py` rebuilds the jockey/trainer alias file the features read (`--write` to save) |
| Prepare dated live features | `build_lookahead.py` (first deletes caches older than 30 days; `--no-cleanup` keeps them) |
| Train normal recipe | Root `train.py`; `create_validation_report.py` exports a run's recipe for a `--fit-as-of` refit |
| Local dependency tooling | `tooling.ps1` (`venv`, `setup`, `lint`, `test`, `validate`, `benchmark` — the last runs `benchmark_data_queries.py`); see [development tools](../docs/operations/DEVELOPMENT_TOOLS.md) |

The [project inventory](../docs/PROJECT_INVENTORY.md) describes each command;
[research reports](../docs/research/) record completed experiments.
