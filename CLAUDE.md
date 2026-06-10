# CLAUDE.md

Read **[PROJECT_OUTLINE.md](PROJECT_OUTLINE.md)** first — it is the full
architecture and handover document. The rules below are the ones that
cause real damage when violated.

## Hard rules

1. **Data-leakage:** any feature describing a horse's past must strictly
   exclude the current race. Never use
   `df.groupby(key)[col].cummax().shift(1)` — the `.shift(1)` is global and
   leaks across horses. Use `transform(lambda x: x.cummax().shift(1))`.
2. **Bet rules live in `src/bet_settlement.py`.** Value thresholds, the
   each-way odds band, and settlement PnL maths are shared by the test-set
   simulation, the walk-forward backtester, and live settlement. Never
   inline them — backtest and live results must use identical formulas.
3. **Merge keys live in `src/utils.py`** (`normalise_horse_key`,
   `normalise_track_key`, `normalise_off_time_key`). Use them for any
   horse/track/off-time join.

## Practical notes

- Environment: local `.venv` on Windows; plain `pip` with
  `requirements.txt` (loose) and `requirements.lock` (pinned).
- Run tests with `.venv/Scripts/python.exe -m pytest tests/ -q`.
- Default model framework is **LightGBM** (see
  `config.SUB_MODEL_FRAMEWORKS`), not XGBoost/CatBoost.
- `data/` and `models/` are gitignored and large; run snapshots are
  pruned from the Experiments tab (Delete Runs → Disk Usage & Pruning).
- Performance: features are computed over hundreds of thousands of rows —
  prefer Numba/vectorised pandas over Python loops.
