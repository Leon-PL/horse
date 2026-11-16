# Faster store and live cards: BHA out, unused columns pruned, saved source history (research 109)

The Matchbook paper record has to featurise a card and forecast it between the
morning refresh and 11:00. Profiling showed three costs that the feature store's
carried state did not cover:

- **BHA stage:** 37 s per live card and 42% of build time. None of its columns
  are read by the win model or the correction, and the family is refuted in
  `docs/NEGATIVE_RESULTS.md` (stewards reports).
- **ATR and Racing TV stages:** they rebuilt their whole source history in every
  process. On each live card that cost 27 s for ATR, 12 s for Racing TV and 44 s
  for the Racing TV signature, and it was repeated in every build worker.
- **Column count:** the store held 1,412 columns, while the models read 285. The
  win model reads 87; the correction reads tracking, entity and card columns.

Code: `research/feature_pruning_109/` (`stage_map.py`, `build_profile.py`,
`keep_list.py`, `fingerprint.py`, `rebuild.py`).

## Changes

1. **BHA features removed from `engineer_features`.** Their signature has also
   left the store's metadata and refresh checks. The BHA collector and its
   modules remain.
2. **`src/store_columns.py`** keeps 518 numeric features and drops the rest at
   the end of `engineer_features`.
   - A feature is kept if any of these names it: a feature-set file, a saved
     run's inputs, the forecast correction, or pipeline code that reads the
     finished store.
   - All 44 raw rating columns are also kept, because `src/rating_reuse.py`
     reads them back to skip the rating sweep. The first keep list dropped 24
     of them, and `tests/test_rating_reuse.py` caught it.
   - Tests that check computed-but-unstored columns run inside
     `store_columns.unpruned()`.
   - Keys, labels, metadata and text are never dropped.
   - To restore a dropped column, add it to a feature-set file, regenerate with
     `keep_list.py`, and rebuild.
3. **`src/fe_cache.py`** saves the ATR run history, the Racing TV run history
   and the source signatures, keyed by the immutable input versions plus a hash
   of the computing module's source. Every later process reads them back.

All three modules are registered in `_FE_SOURCE_FILES`.

## Proof that values did not change

The baseline store and a live card (cached 16 Nov racecard) were fingerprinted
column by column. The store was then rebuilt twice with the new code, pinned to
the same dataset publication (`8c61db28…`), and fingerprinted after each build.
The first build used the 495-column keep list; the second used the final
518-column list, which adds the rating columns.

| Comparison | Columns | Store | Live card |
|---|---:|---|---|
| Baseline vs build 1, columns the models read | 285 | 283 identical; `atr_turn_of_foot_avg_5_vs_field`, `_z_in_race` differ | all identical |
| Baseline vs build 2 (final), columns the models read | 285 | 284 identical; `atr_early_pace_avg_5` differs | all identical |
| Baseline vs build 2, all kept columns | 518 | 517 identical; `atr_early_pace_avg_5` differs | all identical |
| Build 1 vs build 2, same code | 584 | those three columns plus `odds_vs_elo_rank` differ | (card file re-scraped in between) |

**The differing columns vary from build to build, not with the code.**

- They differ between two builds of the same code and data, and each one
  matches the baseline in one of the two builds.
- The final store's ATR windows equal a fresh recompute from the saved
  history for every row.
- A freshly computed ATR history equals the saved one, apart from None vs NaN
  in two columns of foreign rows, which no feature reads.
- Reordering rows within a race moves a race mean by up to 2e-15, which is
  enough to change a hash.

This matches the run-to-run variation already seen in this builder (an
unexplained 0.5 count in a trial build). The earlier stores were replaced
before their values could be read, so the size of the differences is not
measured. The live card, built by both the old and the new code from the same
card file, was identical on every kept column.

Nothing needs refitting.

## Timings

| Step | Before | After |
|---|---:|---:|
| Full store rebuild | ~26–30 min | **15–16.5 min** (894 s and 988 s, 13 chunks; ratings ~514 s in a parallel worker) |
| Store file | 2.9 GB, 1,412 columns | **1.4 GB, 605 columns** |
| Live card features (`features_for_live_cards`) | BHA 37 s + ATR 27 s + RTV 12 s + RTV signature 44 s, plus other stages | **42 s total** |
| Matchbook `prepare` (warm) | 232 s | **174 s** |
| Matchbook `eleven` | 1.1 s | 1.3 s |

Most of the remaining `prepare` time is the correction's input build (about
72 s). Carried state already covers most of the build, so a partial rebuild
from the earliest changed date was not pursued.

Tests now write saved histories to a temporary folder (`tests/conftest.py`).
During this work they had pushed the real ATR history out of
`data/processed/fe_cache`, which cost one recompute.

## Second cut (16 November, after the project cleanup)

The cleanup removed 32 old saved runs and every feature-set file except the
live model's. Regenerating the keep list from what is left (the live 87 inputs,
the correction's columns and the 44 raw ratings) keeps **338** numeric features
(from 518). Rebuilt pinned to the same dataset publication:

| Check | Result |
|---|---|
| Rebuild time | 883 s (14.7 min) |
| Store file | 0.82 GB, 425 columns (from 1.4 GB, 605) |
| Baseline vs this build, all 338 kept columns | identical in the store and on the live card |
| Matchbook `prepare` / `eleven` | 225 s (first run after rebuild) / 2.1 s |

fe_signature is now `3457b041a4c180f1`.
