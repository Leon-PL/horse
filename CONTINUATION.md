# Current handoff — 6 November 2026

Start with [README](README.md), the [project structure](docs/architecture/PROJECT_STRUCTURE.md),
and [project inventory](docs/PROJECT_INVENTORY.md). Earlier handoff notes are
preserved in [the archive](docs/archive/CONTINUATION_before_cleanup_2026-11-05.md).

## Collection cleanup

Real-source collection now lives in `src/ingestion/`. The entry point is
`python -m src.ingestion --sync 14`; existing `src.source_pipeline`,
`src.data_scraper` and `src.rtv_scraper` commands remain compatible aliases.
The synthetic race generator, sample collection API and unused collector shim
are removed. Processing no longer writes an unused global Parquet by default.
See the [cleanup report](docs/operations/PRE_FEATURE_CLEANUP_2026-11-05.md).

## Betfair historical downloads

The Historical Data API downloader is implemented in `src/ingestion/betfair_history.py`.
It retains full BASIC WIN/PLACE files, resumes its SQLite queue and merges missing
observations into the existing price caches. Normal source refreshes opt in once
an account is connected. **Data → Historical exchange data** supplies certificate
download, account connection and catch-up controls. Login settings use Windows
account encryption; certificate login does not require a paid Live App Key.

The local certificate/key pair is prepared at `data/betfair/auth/client.crt` and
`client.key`. The public certificate still needs uploading to the user's Betfair
account, followed by login through the app. No account credentials were available;
an authenticated online transfer has not yet been verified. Local archived market
compatibility and encryption under the normal Windows user were verified.
See [setup details](docs/operations/BETFAIR_HISTORY_AUTOMATION.md).

## ATR backfill

**6 November, 07:16 UTC:** resumed after the worker stopped at 00:02 when shared
ATR files were edited for feature-building performance. The source-extraction
AST is unchanged apart from feature-merge helpers, and the worker is byte-identical.
Eighty scraper/recovery tests passed; SQLite integrity is `ok`. A consistent queue
backup and audited fingerprint adoption preserved all 19,062 previously saved rows.
Restarted worker PID **10864** (launcher 1704), still at a two-second base gap.
The first new accepted field increased the saved count to 19,068. New-day discovery
still encounters challenges; collection continues for known native IDs. Evidence:
`data/audit/atr_resume_20261106/`. No extraction code, feature build or fitted model
was changed by this recovery. Use the live status file for subsequent counts.

The full backfill continues with `--request-delay 2` plus jitter. Its three
pinned implementation files were preserved. Do not reset the queue or replace
the accepted cache before qualification.

- Progress: `data/atr_cache/rebuild_v2/status.json`.
- Durable queue and collected rows: `data/atr_cache/rebuild_v2/rebuild.sqlite`.
- Runtime options: `data/atr_cache/rebuild_v2/launch_options.json`.
- Details: [ATR rebuild](docs/operations/ATR_REBUILD_2026-11-05.md).

Use the project Python, `.venv\Scripts\python.exe` (3.14). The global Python
has a different dependency stack. Check the status file before starting any
worker; an existing worker owns the queue.

## Feature/model state

No feature rebuild, model fit or promotion is part of this cleanup. The installed
feature store and model already needed refresh/refitting after the ATR repairs.
The feature signature also records the updated dependency declarations. Run
`scripts/project_status.py` for actual compatibility before using predictions;
do not copy old fingerprint values into metadata to bypass the checks.

## Repository state

The working tree contains substantial earlier source and research changes.
They were retained. The cleanup's before-files, move manifests, removal receipt
and verification outputs are under `data/audit/pre_feature_cleanup_20261105/`.
