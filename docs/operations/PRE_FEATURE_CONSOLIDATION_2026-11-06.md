# Pre-feature consolidation — 6 November 2026

**Status: done except for the At The Races (ATR) files, which wait on other work
being committed. Nothing is committed yet. No store was rebuilt and no model was
trained or promoted.**

This covers everything that runs before feature building: collecting from each
source, storing and cleaning the data, and handing it to the feature store. It
builds on the [5 November cleanup](PRE_FEATURE_CLEANUP_2026-11-05.md).

The user chose Option B, one owner for each step, from the map and proposal this
report started as. Option C, which would have moved work across stage boundaries,
was not done, because each of its changes risks altering feature values or the
column types the store is keyed on.

## The rule now

**Every regular collector is in `src/ingestion/`, storage is in `src/database.py`,
and cleaning is in `src/data_processor.py`.** ATR is the one exception until its
files move.

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
Betfair public prices   ingestion/auxiliary.py              data/processed/betfair_bsp.parquet
                        + ingestion/betfair_public.py
Betfair full history    ingestion/betfair_history.py        data/betfair/, price caches
                        + ingestion/betfair_streams.py
Exchange rates          ingestion/fx.py                     reference/ecb_gbp_per_eur.csv
At The Races, normal    ingestion/auxiliary.py              data/atr_cache/atr_sectionals.parquet
                        + attheraces_scraper.py (not moved)
At The Races, backfill  atr_rebuild.py (not moved)          data/atr_cache/rebuild_v2/

HAND-OFF
races.db -> database.publish_results_snapshot -> data/raw/race_results.csv + receipt
         -> data_processor.process_data: load CSV -> add pedigree -> clean -> sort
            -> time and category columns -> in-memory frame -> feature store

JOINED LATER, NOT THROUGH THE HAND-OFF
ATR       read inside feature building by attheraces_scraper.merge_atr_sectionals
Racing TV read inside feature building by rtv_features and rtv_text
Betfair   read at training and evaluation by model.py and clv.py
```

## Before and after

| | Before | After |
|---|---|---|
| Sporting Life sync loop | Inside `src/database.py` (1,181 lines) | `src/ingestion/results.py` (366 lines); `database.py` is now 841 lines |
| Places holding collectors | Five: `src/ingestion/`, `src/`, a feature module, `scripts/`, the database module | One, plus ATR |
| Production code importing collection scripts | `auxiliary.py`, `betfair_history.py`, `market_replay.py` | None |
| Compatibility aliases | Six | One, `src.source_storage`, for the ATR files only |
| Commands for the same refresh | Four | One: `python -m src.ingestion` |
| Ways to read history for processing | CSV reader plus a hand-built copy in `project_status.py` | `load_raw_data`, or `load_database_history` for the live database |
| Repair tools | Three ways to publish | All through `publish_after_repair` |

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

The three Betfair scripts remain as thin commands. Racing TV's own tools are now
`python -m src.ingestion.racing_tv`.

## What was removed

- **25 finished one-off scripts, 2,802 lines.** They were tied to the 042 and 044
  repairs and the race-type rebuild. The user's "Yes" to that question was read as
  approval to delete them. They stay in git history once this is committed, and can be
  restored until then with `git checkout HEAD -- <file>`.
- **Aliases:** `source_pipeline`, `source_auxiliary`, `rtv_storage`, `data_scraper`,
  `rtv_scraper`.
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

## Proof that nothing changed

All comparisons ran on one frozen copy: database revision 2,314, its CSV and three
cached racecards. Unchanged code ran twice first, to show each check gives the same
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
- ATR files were not touched, so they were not tested beyond the existing test suite.

Golden hold-out races were only hashed by these checks. Nothing was measured on them.

## Feature signature

`fe_signature()` changes from `e411196de4871a7d` to `2870d125d10da827` because hashed
modules changed, though the processed data is identical. The store on disk was built
by `1355eccb0838ad9b`, so it is already out of date for other reasons, and the
promoted model already needs a refit. If this lands before the next rebuild it costs
nothing extra. If a rebuild happens first, it forces one more of about 14 minutes.

## ATR: what is left

The user agreed that the ATR worker may stop as long as it is resumed. The files have
not moved yet for a different reason: the main checkout has uncommitted ATR work,
including `src/ingestion/atr_tabs.py` and `atr_jumps.py`, and this branch does not
contain it. Moving the committed versions here would collide with that work.

So the work is in two steps.

**Step 1, this branch, can land with the worker running.** It touches none of the four
files the worker watches. A preview merge of the seven files both sides changed gave
no conflicts: `CONTINUATION.md`, `docs/README.md`, `scripts/README.md`,
`src/featured_dataset.py`, `src/ingestion/auxiliary.py`,
`tests/test_atr_current_routes.py` and `tests/test_atr_rebuild.py`.

**Step 2, the ATR move, needs the main checkout's ATR work committed first.**

1. Commit that work and bring it into this branch.
2. Split `src/attheraces_scraper.py`. The collector goes to `src/ingestion/attheraces.py`.
   The cache reader, `merge_atr_sectionals`, `atr_merge_scope` and the column lists stay
   on the feature side, because feature building needs them.
3. Move `src/atr_rebuild.py` to `src/ingestion/` and remove `src.source_storage`.
4. Stop the worker in an orderly way with `data/atr_cache/rebuild_v2/STOP`, and let it
   finish its current race.
5. Land the change, then resume by the recorded procedure: back up the queue, adopt the
   new code fingerprints in an audited step, restart with
   `python -m src.ingestion.atr_rebuild --request-delay 2`, and confirm saved rows rise.
   The queue is never reset. Fingerprint adoption is justified only by showing the
   collection code is unchanged, as this report does for the other collectors.

The one-off helper scripts used for the earlier resumes were removed from `data/audit/`
by a separate cleanup at 14:04 on 6 November (`session_cleanup_20261106/`). Their
records remain, including `atr_resume_20261106/adoption.json`, `review.json` and
`launch.json`, which show what each step did. Step 2 needs a new adoption helper
written from those records. This does not affect step 1.

## Follow-ups outside this work

- `prepare_prize_references` says it refreshes rates, but it has only read them since
  the stage-one audit. It is harmless, and the docstring is wrong.
- `src/trainer_locations.py`, `scripts/fetch_bha_going.py` and
  `scripts/fetch_forecast_previous_runs.py` still sit outside `src/ingestion/`. They are
  one-off research collectors, not part of the normal plan.
- Four tests fail on the unchanged code for reasons other than a missing database (see
  the test section above). `test_append_matches_full_rebuild` matters most, because it
  says an appended store can differ from a full rebuild.
