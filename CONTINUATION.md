# Continuation

Current handoff. Earlier handoffs are in `docs/archive/`.

## State (16 November 2026)

- **Live:** the Matchbook paper record of the 11:00 rule runs daily from the
  scheduled task `HorseMatchbookPaper` (07:50; wakes the PC from sleep, not from
  shutdown; the user must be logged in). Config: `data/matchbook/config.json`
  (win run `20261114_122027_967139`, correction `data/matchbook/model`).
  Results: workbench **Paper record** page, or `python -m src.matchbook summary`.
- **Store:** fe_signature `3457b041a4c180f1`, 338 numeric features kept
  (`src/store_columns.py`, 0.82 GB); full rebuild ~15 min.
- **Cleanup (16 November):** old training/backtest path, retired research,
  old feature sets and dated docs removed or archived. All recoverable at tag
  `pre-cleanup-2026-11`. About 136 GB of old audit/research data and 32 old
  saved runs were deleted (not recoverable); 45 empty locked temp folders remain
  under `data/audit`. A second pass the same evening deleted ~0.75 GB more unused
  data (`data/backups`, `weather_forecast`, `bha_going`, the frozen public
  Betfair copy, one-off CSVs) and two unused result-repair modules.
- **Git (16 November):** `main` was reset to the live line; research 065's FE
  consolidation (previously only on `main`) is at tag
  `archive/fe-consolidation-065`. `main`, the live branch and the archive tags
  are pushed to origin.
- **Paper record, whole field:** settlement now also writes `field.parquet`
  (SP and result for every runner) and `summary()` has a `field` section (log
  loss of win model, forecast, 11:00 Matchbook book and BSP; edge vs result;
  CLV by edge). The workbench Paper page shows every runner's prices and
  forecasts.
- **BHA:** no feature has read BHA data since research 109 (15 November); from
  16 November it is not a default sync source (code and data kept).

## Open

1. Let the paper record run; judge it by geometric and median CLV against BSP
   and by how often the 11:00 quote could have been filled.
2. Fills and capacity (research 111, `python -m research.fills_111.run`): Matchbook depth at 11:00 is thin in some races; the
   record keeps `race_tight` so tight-book races can be read separately.
3. **Research 110** (`research/retrain_110/chain.py`, status in
   `data/research/retrain_110/status.json`): benchmark, history run and
   serving run (`20261116_174805_943504`, as of 2026-11-15) are done. The
   extend step failed on an empty window (fixed in `src/modelling/data.py`);
   resume with `.venv/Scripts/python.exe -m research.retrain_110.chain` for the
   correction refit and the two rehearsals, then compare `rehearsal_new.json`
   with `rehearsal_live.json`. Switching `data/matchbook/config.json` is the
   user's decision. Background: win model retrain is owed after the French/UAE rating-only change
   (2026-11-15); it also clears project_status.py's "promoted model stale"
   flag (a signature mismatch from research 109, not a value change). Refit the
   correction after it, then point
   `data/matchbook/config.json` at both.
4. Real betting needs `MATCHBOOK_USERNAME` / `MATCHBOOK_PASSWORD` in the
   gitignored `.env`, and code that does not exist yet.
