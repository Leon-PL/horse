# French and UAE races

Horses move between Britain/Ireland and France or the UAE. Their overseas runs
belong in their history, and a rating needs the opponents they met there.

## What is collected

`src/ingestion/foreign_form.py` reads Sporting Life's French and UAE result
listings and keeps **every runner of every flat and jumps race**. Trotting and
Arabian races are excluded. Sporting Life horse IDs are the same IDs the
domestic results use, so a raider's French runs join its UK/Irish career
without matching names.

Runners are not filtered to horses already known here. Doing that would give a
historical training row a raider's overseas form that a live card for the same
horse could not have, and a rating update needs the whole field.

Racing TV was dropped as a second feed. Across about 30 sampled days it added
one Abu Dhabi meeting and one Enghien meeting. It could also only identify
runners by matching them to horses already seen here.

Store: `data/foreign_form/races_v2.parquet` (one row per runner, upserted by
race). A horse that appears in two races on one day withholds **both races**:
partial fields never enter a rating sweep.

## How they are used

- **History only.** Foreign races never become training, evaluation or
  prediction rows.
- **Form and trip** (`feature_engineer._with_foreign_starts`): a horse's
  foreign runs count as prior starts.
- **Ratings** (`ratings.compute_elo_features`, `compute_glicko_features`): the
  foreign races are put into the chronological sweep at the end of their race
  day and removed again before the frame is returned. Horses that never race
  here are rated too, because they are a raider's opponents.
- The rating state records how far it has swept the foreign races
  (`foreign_races.through`). A chunked or appended store gives exactly the
  same ratings as one sweep (`tests/test_foreign_form.py`).
- Results are released the day after the race (`available_date`), so a race
  is never seen by a card on its own day.

Betfair has French WIN/PLACE markets in the archive. The importer still keeps
only GB/IE, so foreign races are not trainable yet.

## Commands

The daily `python -m src.ingestion --sync N` now includes the `foreign` source.
For a range:

```
python -m src.ingestion --foreign-form --start 2026-11-01 --end 2026-11-12
python -m src.ingestion --publish-foreign-form
```

A published change to the foreign races changes `foreign_form_signature`, and
the feature store rebuilds (`foreign_form_history_changed`).
