# Results snapshot and test fixes — 6 November 2026

**Status: done and committed on `claude/pre-feature-consolidation-dfd05d`. It is not
on the main checkout's branch and nothing was pushed. No production store was rebuilt
and no model was trained or promoted. The ATR worker was not touched.**

This follows the [consolidation report](PRE_FEATURE_CONSOLIDATION_2026-11-06.md), which
left four failing tests and recommended replacing the CSV hand-off. Both are finished.

## What changes for the next rebuild

- **The processed data does not change.** The data signature stays `d94b262598f1663c`.
- **Eight feature columns change slightly, because of the append fix.** Seven move by
  rounding error only. One, `horse_avg_position_rank_pct_in_race`, changes in 1,137 of
  757,580 rows, and no feature set uses it. Details are under "Measured on real builds".
- `fe_signature()` is now `457c83789e6b29ab`. The store was already due one rebuild of
  about 14 minutes, and this work adds no second one.

## The four failing tests

| Test | Cause | Fix |
|---|---|---|
| `test_append_matches_full_rebuild` | A real bug, described below | `src/fe_incremental.py` |
| `test_every_added_atr_column_is_invariant_to_future_and_current_day_results` | The test was out of date. There was no look-ahead: the audit passes on all 206 columns | The fixture now builds its baseline as the study froze it |
| `test_apply_backs_up_all_columns_and_preserves_remaining_rows` | The fixture's table had no `horse_id`, which the real table has | The fixture now has it |
| `test_interrupted_delete_rolls_back_and_retains_backup` | Same as above | Same as above |

The ATR test compared the study's candidate list with today's `feature_sets/win.txt`.
That file has changed since the study, so 10 non-ATR columns looked like additions. The
registered study froze its baseline as the candidate's non-ATR columns, and that is what
the fixture now does.

## The append bug

pandas adds up a running total with Kahan compensation, which keeps a small correction
alongside the total so rounding errors do not build up. `src/fe_incremental.py`
reproduces this exactly within one frame. Across frames it did not: a later chunk or an
append restarted the correction at zero and added the old total on afterwards.

So two horses with equal histories could end up with totals that differ in the last
bit. A rank then treats a true tie as a non-tie. In the failing test, two horses both on
0.49 ranked 0.5 and 0.667 instead of sharing 0.583.

Totals now resume with their correction, so resuming gives the same bits as one pass.
The correction is stored beside each total in the carried state, as
`{column}::compensation`, and only where it is not zero. Whole-number metrics have none.
A store saved before this change still loads, and its corrections read as zero.

This mattered beyond appends, because the normal store build works in chunks through the
same code. A store's values therefore depended on where its chunks happened to fall.

### Measured on real builds

Three full builds ran on a frozen copy of the data (757,580 rows, 1,231 columns), in a
scratch folder. No production file was used.

| Comparison | Result |
|---|---|
| Old code against new code, same chunks | 1,223 columns bit-identical, 7 differ in rounding only, 1 differs by more |
| Processed data, old against new | All 97 columns identical |
| New code, 60,000-row chunks against 45,000-row chunks | 1,221 columns bit-identical |
| Running totals and corrections across those two builds | All 230 identical |
| Carried state size | 617 MB to 618 MB; 66,165 corrections against 14.3 million entries |

The columns that moved between old and new code:

| Column | Rows changed | Largest absolute change | In a feature set? |
|---|---|---|---|
| `horse_avg_position_rank_pct_in_race` | 1,137 | 0.25 | No |
| `horse_avg_position_z_in_race` | 395,950 | 2.3e-14 | No |
| `horse_avg_position_vs_best_in_race` | 282,592 | 4.4e-16 | Yes |
| `horse_avg_position_vs_field` | 229,242 | 4.4e-16 | No |
| `horse_form_relative_score` | 225,003 | 3.9e-15 | No |
| `horse_avg_position` | 183,391 | 4.4e-16 | No |
| `horse_avg_prize_contested` | 61,998 | 8.7e-11 (3.6e-15 relative) | Yes |
| `prize_money_change_vs_avg` | 6,326 | 3.6e-15 | Yes |

The 10 columns that still differ between chunk sizes are all `atr_speed_score` and
`atr_stride_x_gallop` and their relatives. Those two composites are normalised over
whatever rows they are given, which is why they are
[quarantined](../MARKET_PERFORMANCE_PROGRESS.md), and no feature set uses them.

In the carried state, 302 of 322 keys are identical across the two chunk sizes. The other
20 are time-window lists that differ only in which expired rows were pruned when a chunk
ended. Rows outside a window cannot affect a later value.

## The Parquet hand-off

`data/raw/race_results.csv` is replaced by `data/raw/race_results.parquet`. Racecard
caches stay CSV, and `python -m src.database --export` still writes a CSV for anyone who
wants one.

Every column now has a declared type in `RESULTS_SNAPSHOT_COLUMNS` in `src/database.py`.
The same function, `results_snapshot_frame`, types the rows on both routes: when they are
published, and when the readiness check reads the database directly. So the two routes
cannot drift apart, and `load_database_history` no longer goes through CSV text.

| What can go wrong | With the CSV | Now |
|---|---|---|
| A text value lands in `race_class` or an ID column | The whole column silently becomes text | Publication stops, names the column and keeps the last good file |
| A new database column appears | It flows through with an inferred type | Publication stops until the column is declared |
| A column is empty | Its type depended on the values | Its type is declared |
| Missing text | NaN or None, depending on the route | Always NaN |

### What the CSV trip was doing

Measured on all 757,580 rows, column by column, to find what had to be reproduced on
purpose:

- **Blank text became missing.** That was the only text change, in `headgear` (466,468
  rows), `race_class` (198,508), `form` (52,397), `finish_pos_label` (6,238), `going`
  (66) and `jockey` (2). `_snapshot_text` now does this explicitly.
- **Numbers kept every bit, with one exception.** `weather_precip_prev3` differs in
  62,189 rows, by at most 7e-15.

That exception needs a decision. The column sums rainfall readings, so the database holds
values like `0.30000000000000004`, and pandas' fast CSV float parser is not exact. The
history has therefore always held the parser's slightly different result. No rounding rule
reproduces it: 57,569 rows still differ at any precision from 2 to 12 places.

To keep every value unchanged, `_snapshot_csv_parsed_real` keeps that parser step for this
one column. It is the only CSV parsing left in the hand-off. The alternative is to use the
database's value, which is one word to change in `RESULTS_SNAPSHOT_COLUMNS` and is cleaner,
but it changes the data signature. If that is wanted, do it before the next rebuild, not
after.

`opening_odds` also came to light. The real table has it, but a fresh database did not,
because collection stopped long ago. `init_db` now adds it, which does nothing to the real
database and makes a new one match.

`sire`, `dam` and `damsire` are declared as text. The CSV read them as all-empty float
columns, because pedigree is filled in later. This does not reach the processed data.

### Proof

Run on frozen revision 2,314, against the fingerprint taken before any of this work:

| Check | Result |
|---|---|
| Data signature, published route | `d94b262598f1663c`, identical |
| Data signature, database route | `d94b262598f1663c`, identical |
| Processed data: rows, columns, order, dtypes, every column hash | Identical (757,580 rows, 97 columns) |
| SHA-256 of `processed.parquet` built from it | Identical, `5f92703406a86cad...` |
| Raw frame against `read_csv`, value by value, including the missing marker | Identical except the three declared text columns above |
| Database route against published route, strictly | No differences |
| Three cached racecards | Identical |

### Speed and size

Measured on a quiet machine.

| | CSV | Parquet |
|---|---|---|
| File size | 212 MB | 25 MB |
| Read the file | 7.5 s | 3.7 s |
| Publish a changed revision | Not re-measured | About 67 s, of which 52 s is reading SQLite from a cold disk |
| Refresh with nothing new | Under a second | 0.1 s, through the same receipt |

## Tests

The full suite ran with 1,908 passed, 11 failed and 3 skipped. The 11 are all in
`test_data_integrity.py` and need a real `data/races.db`, which a worktree does not have.
They fail the same way before this work. No other test fails.

Against the run before this work (1,883 passed, 15 failed, 3 skipped), the four tests
above now pass, 21 tests are new, and no test that exists in both runs got worse. Two
were renamed to match what they now cover:

- `test_deep_probe_fingerprints_database_even_if_csv_is_stale` now ends
  `_snapshot_is_stale`. Its fixture supplies a complete row, because the check rejects
  partial ones.
- `test_interrupted_snapshot_publish_preserves_previous_whole_file` is now
  `test_interrupted_csv_write_preserves_previous_whole_file`, because it tests the CSV
  writer racecards still use. `test_interrupted_publication_keeps_the_last_file` covers
  the Parquet publish.

New tests are in `tests/test_results_snapshot.py`, plus bit-exact resume tests in
`tests/test_fe_incremental.py`. The sync test fixtures name races with letters, so they
declare `race_id` as text for their own run. The real declaration is numeric, and
`tests/test_results_snapshot.py` tests it.

## Landing this

This branch has not been merged, because landing it has three consequences:

1. **It stops the ATR worker.** `src/ingestion/attheraces.py` is a watched file, and its
   `backfill_atr_sectionals` command read the CSV. The worker never calls that command, so
   the change does not affect collection, but the worker stops on any edit to a watched
   file. It needs the same audited handover as before: stop, back up, adopt the new
   fingerprints, restart. That took about two minutes last time, with no queue change.
2. **One publish is needed before processing can run**, because no Parquet exists yet.
   `load_raw_data` says so if it is missed. Any refresh does this, or with no network:

   ```bash
   .venv/Scripts/python.exe -c "from src.database import publish_results_snapshot as p; print(p())"
   ```

3. **`data/raw/race_results.csv` is then unused and can be deleted.** Nothing removes it
   automatically.

## Reproducing

The scratch stores, code snapshots and comparison reports served only this measurement.
To repeat the store comparison, build twice with
`scripts/build_feature_store_batched.py --store-root` and compare the results with
`scripts/compare_feature_stores.py`. Its state check reports a difference on the new
`::compensation` keys when one side is an older store, and on time-window pruning when
the chunk sizes differ.
