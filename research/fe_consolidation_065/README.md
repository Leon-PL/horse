# Feature-engineering consolidation (065)

2026-11-07. Follows a review of the feature build that found 693 of its
1,142 eligible inputs in no feature set (another review, not committed here,
is `docs/research/FEATURE_ENGINEERING_CONSOLIDATION_REVIEW_2026-11-07.md` in
the main checkout).

Each step was checked by building the 2026 slice (103,128 rows) from the
raw sources before and after it and comparing every column bit for bit.

| Step | Change | Check |
|---|---|---|
| 1 | Schema guards: declared one-hot domains (`data_processor.INDICATOR_DOMAINS`), and a feature list that names a missing column now raises | 1,392 shared columns identical; adds `surface_Fibresand` and `surface_Sand` (declared, absent from 2026) |
| 2 | Removed 465 output columns that no recipe in `feature_sets/` names, with the code that made only them ([removed_columns.txt](removed_columns.txt)); race-relative transforms are now a declared table (`src/features/relative.py`) | the 929 kept columns identical |
| 3 | Split `src/feature_engineer.py` (4,938 lines) into `src/features/` family modules; the engine keeps only stage order | all 929 columns identical |
| 4 | Semantic fixes (see below) | full build: 921 of 926 columns identical; the 5 stored weather readings keep 11,727 missing values that were filled before; `trainer_freq` and its two relatives equal in value (now float); the 3 unused counts gone |

Step 4 changes:

- `trainer_freq` is counted in the trainer stage on the carried counts state,
  in event order. It used to be a `cumcount` over the processed frame, made
  three ways: in processing, again in the history-concatenating live path, and
  for a card alone by a catch-up that read `processed.parquet`. The horse,
  jockey and track counts were dropped: nothing used them.
- The Elo sweep no longer deletes the caller's `_event_dt`. Every stage after
  it used to fall back to day order in a full build, and the store append
  deleted `_event_dt` on purpose to match; both now use date plus off time.
- `dist_category` has one definition (`src.features.common.dist_category`),
  used by the horse stage and the store's standalone rating sweep. The event
  sort key also has one definition (`ratings.py` imported a copy).
- Stored weather readings are no longer filled with UK averages.
- `_horse_key` requires a complete `horse_id`; it used to fall back to names.

On a full build the `trainer_freq` and `_event_dt` fixes change nothing (the
processed frame was already in event order, and no kept column after the Elo
sweep depends on order within a day). They remove the separate live
catch-up and the special case that kept the store append in step with the
Elo side effect.

Seven older recipes (`handicap_compact_030`, `handicap_performance_031`,
`market_finish_surprise_035`, `market_transfer_034`, `prior_market_033`,
`research_win_with_atr`, `trip_suitability_032`) already named columns the
build had stopped producing before this work. With the step-1 guard,
training on them raises instead of silently dropping those names.

## Files

- `slice_build.py`: builds the slice from a data root (`--data-root`,
  `--start`, `--end`, `--output`).
- `compare.py`: column-by-column parquet comparison (restored from 60c4873's
  parent tooling); `--columns` limits it to a list.
- Outputs: `data/research/fe_consolidation_065/`.
