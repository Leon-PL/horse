# Continuation

Current handoff. Earlier handoffs are in `docs/archive/`.

## State (16 November 2026)

- **Live:** the Matchbook paper record of the 11:00 rule runs daily from the
  scheduled task `HorseMatchbookPaper` (07:50; wakes the PC from sleep, not from
  shutdown; the user must be logged in). Config: `data/matchbook/config.json`
  (win run `20261114_122027_967139`, correction `data/matchbook/model`).
  Results: workbench **Paper record** page, or `python -m src.matchbook summary`.
- **Store:** fe_signature `02a313577e1de9b3`, 518 numeric features kept
  (`src/store_columns.py`); full rebuild ~15 min.
- **Cleanup (16 November):** old training/backtest path, retired research,
  old feature sets and dated docs removed or archived. All recoverable at tag
  `pre-cleanup-2026-11`.

## Open

1. Let the paper record run; judge it by geometric and median CLV against BSP
   and by how often the 11:00 quote could have been filled.
2. Fills and capacity: Matchbook depth at 11:00 is thin in some races; the
   record keeps `race_tight` so tight-book races can be read separately.
3. Win model retrain is owed after the French/UAE rating-only change
   (2026-11-15); refit the correction after it, then point
   `data/matchbook/config.json` at both.
4. Real betting needs `MATCHBOOK_USERNAME` / `MATCHBOOK_PASSWORD` in the
   gitignored `.env`, and code that does not exist yet.
