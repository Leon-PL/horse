# Pre-feature consolidation — 6 November 2026

**Later Betfair update:** the data flow map below is current. Public and manual collection
entries elsewhere in this snapshot were superseded by the
[unified full-history path](BETFAIR_CONSOLIDATION_2026-11-06.md).

**Status: done, including At The Races (ATR). The work is committed and is on the
main checkout's branch, `checkpoint/eval-harness-and-guards`. Nothing was pushed. No
store was rebuilt and no model was trained or promoted. The ATR worker was stopped,
moved and restarted, with its queue unchanged.**

This covers everything that runs before feature building: collecting from each
source, storing and cleaning the data, and handing it to the feature store. It
builds on the [5 November cleanup](PRE_FEATURE_CLEANUP_2026-11-05.md).

The user chose Option B, one owner for each step, from the map and proposal this
report started as. Option C, which would have moved work across stage boundaries,
was not done, because each of its changes risks altering feature values or the
column types the store is keyed on.

## The rule now

**Every regular collector is in `src/ingestion/`, storage is in `src/database.py`,
and cleaning is in `src/data_processor.py`.** There are no exceptions and no
compatibility aliases left.

## How data flows now

```text
python -m src.ingestion, the Data page and train.py all call
ingestion/pipeline.py: refresh_sources

SOURCE                  COLLECTED BY                        STORED IN
Sporting Life results   ingestion/results.py                data/races.db
                        + ingestion/sporting_life.py        data/sl_cache/raw.sqlite
Pedigree (--enrich)     ingestion/pedigree.py               races.db: horse_pedigree
Weather (--weather)     ingestion/weather.py                races.db: results.weather_*
Racing TV               ingestion/racing_tv.py              data/rtv_cache/ (generations)
                        + ingestion/rtv_observations.py     fitted par and text models too
Betfair full history    ingestion/auxiliary.py              data/betfair/history/
                        + ingestion/betfair_history.py      data/processed/betfair_*.parquet
                        + ingestion/betfair_streams.py
Exchange rates          ingestion/fx.py                     reference/ecb_gbp_per_eur.csv
At The Races, normal    ingestion/auxiliary.py              data/atr_cache/atr_sectionals.parquet
                        + ingestion/attheraces.py
At The Races, backfill  ingestion/atr_rebuild.py            data/atr_cache/rebuild_v2/
                        + ingestion/atr_tabs.py, atr_jumps.py

HAND-OFF
races.db -> database.publish_results_snapshot -> data/raw/race_results.parquet + receipt
         -> data_processor.process_data: load snapshot -> add pedigree -> clean -> sort
            -> time and category columns -> in-memory frame -> feature store

BETFAIR
Routine refresh  auxiliary.refresh_betfair, the "betfair" source. Needs the saved connection.
                 Asks for dates up to six days ago; Betfair publishes five days after settlement.
Catch-up         python -m src.ingestion.betfair_history --drain, or the Data page button.
Old tar files    python -m src.ingestion.betfair_archives --remove-verified stores every
                 member once in archive_payloads.sqlite, then removes each verified tar.

Refresh and catch-up run the same path, betfair_history.sync_history:

  list WIN and PLACE market files -> queue.sqlite
  -> find each market: raw/BASIC/ file, then archive_payloads.sqlite, then download
     (while the tar files are being consolidated, catch-up skips the download step)
  -> betfair_streams parsers, with date, country, market type and settlement checks
  -> parsed rows saved in queue.sqlite
  -> publish: fills gaps and adds runners, and never replaces an accepted price
     WIN    betfair_bsp, betfair_bsp_early, betfair_timing (.parquet)
     PLACE  betfair_place_bsp (.parquet)

JOINED LATER, NOT THROUGH THE HAND-OFF
ATR       read inside feature building by atr_features.merge_atr_sectionals
Racing TV read inside feature building by rtv_features and rtv_text
Betfair   betfair_bsp.parquet read at training and evaluation by model.py and clv.py,
          never as a feature input; research scripts read the other three caches

FROZEN, NO LONGER COLLECTED
data/research/betfair_public_legacy/  the old public files and their volume and VWAP data,
                                      read only by research and field-repair scripts
```

## Before and after

| | Before | After |
|---|---|---|
| Sporting Life sync loop | Inside `src/database.py` (1,181 lines) | `src/ingestion/results.py` (366 lines); `database.py` is now 841 lines |
| Places holding collectors | Five: `src/ingestion/`, `src/`, a feature module, `scripts/`, the database module | One |
| Production code importing collection scripts | `auxiliary.py`, `betfair_history.py`, `market_replay.py` | None |
| Compatibility aliases | Six | None |
| Commands for the same refresh | Four | One: `python -m src.ingestion` |
| Ways to read history for processing | CSV reader plus a hand-built copy in `project_status.py` | `load_raw_data`, or `load_database_history` for the live database |
| Repair tools | Three ways to publish | All through `publish_after_repair` |
| Editing the ATR join | Stops the worker, because it shared a file with the collector | Does not stop the worker |

| Moved | From | To |
|---|---|---|
| Sync loop and its ledger logic | `src/database.py` | `src/ingestion/results.py` |
| Pedigree collection | `src/pedigree_backfill.py` | `src/ingestion/pedigree.py` |
| Pedigree table and reading | `src/pedigree_backfill.py` | `src/database.py` |
| Pedigree join | `src/pedigree_backfill.py` | `src/data_processor.py` |
| ECB rate download | `src/prize_currency_features.py` | `src/ingestion/fx.py` |
| Racing TV materialisation | `src/rtv_feature_collection.py` | `src/ingestion/rtv_observations.py` |
| Weather | `src/weather.py` | `src/ingestion/weather.py` |
| BHA stewards | `src/bha_stewards_scraper.py` | `src/ingestion/bha_stewards.py` |
| Win and place stream parsers | `scripts/extract_betfair_*.py` | `src/ingestion/betfair_streams.py` |
| Public BSP download and parsing | `scripts/fetch_betfair_sp_public.py` | `src/ingestion/betfair_public.py` |
| ATR collector | `src/attheraces_scraper.py` | `src/ingestion/attheraces.py` |
| ATR backfill worker | `src/atr_rebuild.py` | `src/ingestion/atr_rebuild.py` |
| ATR columns, cache location and cache reader | `src/attheraces_scraper.py` | `src/ingestion/atr_tabs.py` |
| ATR join | `src/attheraces_scraper.py` | `src/atr_features.py` |

The three Betfair scripts remain as thin commands. Racing TV's own tools are now
`python -m src.ingestion.racing_tv`.

## What was removed

- **25 finished one-off scripts, 2,802 lines.** They were tied to the 042 and 044
  repairs and the race-type rebuild. The user's "Yes" to that question was read as
  approval to delete them. They stay in git history once this is committed, and can be
  restored until then with `git checkout HEAD -- <file>`.
- **Aliases:** `source_pipeline`, `source_auxiliary`, `source_storage`, `rtv_storage`,
  `data_scraper`, `rtv_scraper`.
- **Unused code:** `load_matched_history`, the `sync_database` wrapper in the
  pipeline, `generation_for_publish`, `_publish`, `scrape_results`, the Sporting Life
  command line, the `return_report` argument, and the `save` argument of
  `process_data` with its legacy export.
- **Duplicates:** one `result_rows` reader now serves `canonical_rows` and
  `canonical_fields`, and Racing TV uses the shared `atomic_parquet`.

## Behaviour that did change

Nothing below changes a feature value. These are the changes someone could notice.

1. `python -m src.database --sync` and its `--results-only` option, `-m src.data_scraper`,
   `-m src.source_pipeline` and `-m src.rtv_scraper` no longer exist. `--audit-history`
   now only queues the dates and tells you to run
   `python -m src.ingestion --sync DAYS --sources sl`.
2. The Sporting Life command line had a `--mode racecards` option. Cards still come
   from the picks app and the Data page through `get_scraped_racecards`.
3. `repair_overfull_fields.py` now updates the CSV receipt, and
   `reconcile_result_fields.py` now publishes after deleting rows. Before, the second
   left the CSV out of date until the next refresh.
4. The Betfair public-price receipt hashes its source file, which moved, so the next
   refresh re-parses those files once. The result is unchanged.
5. The Racing TV time par is cached by the bytes of `src/rtv_features.py`, and that
   file's imports changed, so the next materialisation refits it once. The fixed
   window gives an identical par (see below).
6. Racing TV parquet writes gain the shared row-count check and file sync.
7. `weather.py` found its folder from its own file location, which would have been
   wrong after the move. It now uses `config.DATA_DIR`, the same place in practice.
8. Log lines from the sync now appear under `src.ingestion.results`.
9. The ATR commands are now `python -m src.ingestion.atr_rebuild` and
   `python -m src.ingestion.attheraces`. The old module paths no longer exist.
10. ATR join log lines now appear under `src.atr_features`.

## Proof that nothing changed

The non-ATR comparisons ran on one frozen copy: database revision 2,314, its CSV and
three cached racecards. Unchanged code ran twice first, to show each check gives the same
answer every time. Both runs matched exactly.

| Check | Compared | Result |
|---|---|---|
| Processing from the CSV | 757,580 rows, 97 columns, data signature `d94b262598f1663c`, parquet hash | Identical |
| Processing from the database | Every column | Identical |
| `load_database_history` against the old `project_status.py` route | Data signature, every column, every data type | Identical |
| Three cached racecards | 389, 478 and 270 rows, each with its data signature | Identical |
| Offline replay of the Sporting Life sync, 28 October to 4 November | 311 stored race pages with the network forbidden: all 18 database tables except timestamp columns, the published CSV, revision, new rows, pending races, final state | Identical |
| Exchange rates, offline | Output file, number of requests, report | Identical |
| Betfair parsers on the `2026-06.tar` sample | Win 225 rows, place 1,357 rows | Identical |
| Racing TV time par | Old code and new code on 14,572 races, plus both cached pars | Identical, largest difference 0.0 |
| ATR parsing | 2,732 stored races replayed through old and new code with no network: outcomes, 22,718 rows and tab receipts | Identical, no race differs |
| ATR join | 69 columns on 757,580 rows, 232,283 of them with ATR data, year by year | Identical |
| Commands | 15 commands run with `--help` | All exit 0 |
| Imports | 1,330 `src` and `scripts` imports, read statically | None unresolved |

Every definition was also compared by syntax tree, before and after, wherever it moved:

| | Definitions |
|---|---|
| Unchanged in place | 2,385 |
| Moved, unchanged | 74 |
| Moved and edited | 11 |
| Edited in place | 46 |
| Removed | 8 |
| Added | 4 |

All 57 edits were read by hand. They are repointed imports and file paths, module
prefixes needed after a move, removal of the `save` and `return_report` arguments, and
the behaviour changes listed above. The file paths rewritten in research scripts name
files whose current code those scripts hash, so the new paths are correct.

The ATR move was compared separately: 2,492 definitions unchanged in place, 83 moved
unchanged, 6 moved and edited, 7 edited in place, none removed and none added. Every
ATR collection edit is an import path, plus the worker's list of pinned files.

**Tests.** The final run was on the finished code and started from the same state as
the baseline.

| | Passed | Failed | Skipped |
|---|---|---|---|
| Before | 1,781 | 15 | 3 |
| After | 1,785 | 15 | 3 |

- No test that exists in both runs changed result.
- The same 15 tests fail in both runs, with the same error message.
- The extra passes come from test changes: 9 tests were retired or renamed, and 13
  were added.

The ATR move had its own before-and-after run, because the merged ATR work added
tests. It went from 1,881 passed, 15 failed and 3 skipped to 1,883, 15 and 3. The same
15 tests fail with the same messages, and no test that exists in both runs changed
result. One retired test and three new ones account for the difference.

The 15 failures already existed:

- **11 in `test_data_integrity.py`** need a real `data/races.db`, and a worktree has
  none. They fail with "no such table: results" on both sides.
- **`test_append_matches_full_rebuild`** reports that
  `horse_avg_position_rank_pct_in_race` differs on 4 of 360 rows between an appended
  store and a full rebuild.
- **`test_every_added_atr_column_is_invariant_to_future_and_current_day_results`**
  fails its invariance audit.
- **2 in `test_result_field_reconciliation.py`** fail with "no such column: horse_id".

An interim run showed 1,795 passed and 4 failed. That figure was misleading: an empty
database left by the first run let the 11 data tests pass without checking anything.
That file was removed before the final run, and the table above is the true comparison.

**What these checks do not cover.**

- Nothing was collected from a live website, so the proof is offline replay, not a
  live refresh. The first real refresh is the remaining check.
- The Betfair check is one month's sample, not all history.
- Weather, BHA stewards and `scripts/backfill_pedigree.py` were not run. The first two
  were moved with only the path and import edits shown in the comparison. The third has
  no help option, and running it would change data.
- The ATR replay covers races the worker has already finished, so it proves the parsers
  on stored responses, not ATR's live requests. The restarted worker covers those.

Golden hold-out races were only hashed by these checks. Nothing was measured on them.

## Feature signature

`fe_signature()` changes from `e411196de4871a7d` to `96f8dbe1ce9de8f0` because hashed
modules changed, though the processed data and the ATR join are identical. The store
on disk was built by `1355eccb0838ad9b`, so it was already out of date for other
reasons, and the promoted model already needed a refit. The next rebuild takes about
14 minutes, and this work adds no second one.

`src/ingestion/atr_tabs.py` now counts towards the signature, because it holds the
columns the join uses. `src/attheraces_scraper.py` no longer does, so an edit to the ATR
collector no longer forces a rebuild.

## How the ATR move was done

The main checkout had uncommitted ATR work, so that was committed first, as `3474f1b`,
and merged into this branch. The move itself is `9f7daf0`. The worker kept collecting
through the replay and both test runs, and was only stopped once they had passed.

1. `data/atr_cache/rebuild_v2/STOP` was created at 16:28:35 UTC. The worker finished its
   race and exited in about seven seconds, with state `stopped`.
2. The queue was backed up to `data/audit/atr_consolidation_20261106/queue_before_move.sqlite`,
   and the backup passed its integrity check.
3. The branch was fast-forwarded onto `checkpoint/eval-harness-and-guards` in the main
   checkout.
4. The new file fingerprints were adopted in the queue and the manifest. Every other
   queue value was hashed before and after, and the digest did not change
   (`a6f85557ff2e7a5c`). No row was reset or reopened: 62,432 jobs, 33,932 runners and
   312,160 tab records.
5. The stop marker was removed and one worker was started with
   `python -m src.ingestion.atr_rebuild --request-delay 2`. Launcher PID 13356, worker
   PID 14328. Use the live status file, not these PIDs, once they go stale.

The worker was collecting again about a minute and a half after it stopped. Finished
races rose from 3,048 to 3,059 in its first minute, with state `running`, and exactly
one worker is running.

Saved rows are not the measure of progress at the moment. The worker is re-collecting
races it already holds, under the five-tab contract, so saved rows stay at 33,932 while
finished races rise.

The one-off helpers from earlier resumes were removed by the cleanup at 14:04, so a new
one was written from their records. It and the proof files are in
`data/audit/atr_consolidation_20261106/`.

## Follow-ups outside this work

- `src/trainer_locations.py`, `scripts/fetch_bha_going.py` and
  `scripts/fetch_forecast_previous_runs.py` still sit outside `src/ingestion/`. They are
  one-off research collectors, not part of the normal plan.
- The four tests that failed on the unchanged code are fixed, and the CSV hand-off above
  has since been replaced by Parquet. See the
  [follow-up report](RESULTS_SNAPSHOT_AND_TEST_FIXES_2026-11-06.md). The proof tables
  above describe the CSV hand-off as it stood when this report was written.
