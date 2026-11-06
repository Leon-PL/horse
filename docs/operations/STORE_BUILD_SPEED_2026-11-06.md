# Feature store build speed — 6 November 2026

A full store build now takes **14.1 minutes, down from 92**, with no feature
removed. The rebuilt store is byte-identical to the one the old code builds. The
production store was not touched and nothing was retrained. Any later store
rebuild uses this code; the changed `fe_signature` makes the next training run
rebuild once.

One earlier trial build differed from the baseline in a single carried value and
the cause was not established. See [the difference that appeared once](#the-difference-that-appeared-once).

## Timing

Both runs used the same frozen data (757,580 rows, 1,231 columns, 13 chunks) on
this machine with no compile cache. The ATR collector ran throughout; no other
heavy job did. The last install's own log, on 755,309 rows, shows 5,000 seconds.

| Part of the build | Before (s) | After (s) | Change |
|---|---:|---:|---|
| Whole build | **5,521** | **845** | 6.5 times faster |
| `process_data` | 82 | 53 | Repeated parsing and key work runs once per distinct value |
| Ratings | 1,137 | 395 in a helper; the build waited 45 | Compiled TrueSkill, rated ahead of FE |
| Feature engineering, summed over processes | 3,618 | 918 | Rewrites below; 40% of this runs in a second process |
| Writing chunks | 127 | In a helper, not waited for | Streams into the final file |
| Consolidating chunks | 291 | 0 | Every chunk now has one schema |
| Saving state | 113 | 74 | |
| Input fingerprints | 54 | 61, beside the first chunks | Runs on a thread while the first ratings run |

The chunk loop ended at 715 seconds. The remaining 130 seconds finish the last
chunk, bring the helper's state back and save it.

| FE stage | Before (s) | After (s) | Faster by |
|---|---:|---:|---:|
| `add_atr_features` | 1,315 | 104 | 12.6× |
| `add_horse_features` | 343 | 42 | 8.2× |
| `add_supplementary_features` | 318 | 59 | 5.4× |
| `add_speed_figure_features` | 262 | 25 | 10.5× |
| `add_target_encoded_features` | 260 | 63 | 4.2× |
| `add_rtv_features` | 123 | 81 | 1.5× |
| `add_opposition_strength_features` | 115 | 13 | 8.9× |
| `add_established_form_features` | 115 | 63 | 1.8× |
| `add_jockey_features` | 101 | 15 | 6.5× |
| `add_trainer_features` | 84 | 24 | 3.5× |
| `add_race_relative_signal_features` | 84 | 100 | Slower |
| `add_prior_market_features` | 58 | 42 | 1.4× |
| Every other stage | 440 | 288 | 1.5× |

FE time per chunk used to grow from 107 to 388 seconds at constant chunk size. It
now stays between 35 and 48 seconds in the main process. Both before and after
figures are sums across chunks taken from `FE_PROFILE=1` logs.

## What changed

- **Carried state:** the seeded helpers did work proportional to every entity seen
  so far. They now do work proportional to the current chunk
  ([fe_incremental](../../src/fe_incremental.py), the `_cumulative`, `_rolling`,
  `_lag`, `_time_window`, `_ffill_prev` and `_expanding_prev` paths in
  [feature_engineer](../../src/feature_engineer.py)).
- **Three processes:** on 100,000 rows or more, one process rates ahead, the main
  process runs FE through `add_atr_features`, and a third runs the rest and writes
  the store ([build script](../../scripts/build_feature_store_batched.py)). Each
  carries its own state. If settings differ from `config.py`, or a helper cannot
  start, the build stays in one process. `_add_second_half_state` refuses to merge
  state where both halves hold the same entity.
- **Ratings:** TrueSkill moved to a compiled kernel,
  [trueskill_kernel](../../src/trueskill_kernel.py), about 65 seconds to 1 second
  per chunk. It is registered in `_FE_SOURCE_FILES`.
- **ATR:** run features are compiled, 16.8 to 1.3 seconds per chunk, and the
  sectional cache is joined on keys and read once per build.
- **Wide frames:** `drop` and copying `concat` rewrote every block of a frame with
  over a thousand columns. `_without` and `_with_columns` avoid that.
- **Store writing:** chunks stream into one file through `RollingParquetWriter`.
  The three columns whose type used to change between chunks are now always float,
  which the final store already held.
- **End of the build:** the helper's state comes back through a file, because the
  pipe took about four minutes. The processed rows and input fingerprints are
  written beside the first chunks. Fingerprints taken before FE reads the caches
  also mean a cache changed mid-build now shows up as a mismatch.

## Equivalence

[`scripts/compare_feature_stores.py`](../../scripts/compare_feature_stores.py)
compared the final build with the baseline. The tolerance was 1e-9, and nothing
needed it.

| Check | Result |
|---|---|
| `featured.parquet` | 757,580 rows, same order; 1,231 of 1,231 columns exact, 0 within tolerance, 0 different; same types and column order |
| `processed.parquet` | 757,580 rows; 97 of 97 columns exact |
| `ratings_state.pkl`, `counts_state.pkl` | 0 differences, compared with no tolerance |
| `featured.parquet`, SHA-256 | Same as both baseline builds: `71efdbf50ab109f0…` |
| `processed.parquet`, SHA-256 | Same as the baseline: `5f92703406a86cad…` |
| `meta.json` | Differs only in `built_at` and `fe_signature` |

Both baseline builds also match each other. Hashing two builds takes under a
minute, so it is the quickest check that two builds agree.

**Against the production store:** it has the same 757,580 rows in the same order.
All 97 processed columns and the ratings state match, and 1,029 of 1,231 feature
columns are exact. The other 202 are 180 `atr_` columns and 22 pace features built
on them. That store was built early on 5 November, before that day's ATR repair,
and its `atr_signature` differs. The old code on the frozen data produces the new
store exactly, so these differences predate this work. The next rebuild brings the
production store up to date.

## The difference that appeared once

Trial build `rehearsal2` differed from the baseline in five columns and nowhere
else: `trainer_te_norm_pos_365d` and its four race-relative forms. Six runs by one
trainer changed directly, between 9 September 2024 and 1 June 2025. The largest
change was 0.0117.

- **What went wrong:** the windowed count for one race, 800699 on 2 June 2024, was
  carried as 0.5 instead of 1.0. Every affected value fits that exactly, and the
  matching sum was correct.
- **Where:** after the frame. `going`, which uses the same weight column for the
  same race, was unaffected, so the fault lies in the trainer's aggregate or its
  carried entry.
- **Repeats:** it did not come back. Five later builds of this pipeline are
  byte-identical to the baseline, one of them under heavy load like the original.
- **Stray writes:** a bounds-checked run of the compiled kernels found no
  out-of-bounds access. That covers ratings and FE on one late chunk.

**The cause is not established.** The machine has no error-correcting memory, and a
one-bit change turns 1.0 into 0.5. Against that, the race's own position value is
exactly 0.5, which is an awkward coincidence and could point to an intermittent
software fault. No mechanism for one was found. A memory test would settle part
of this; it needs a restart, so it was not run. The first baseline run also died
silently at 21:07 on 5 November. Windows logged no crash record, and it may be
unrelated.

For a store behind a decisive result, build it twice and compare the hashes. That
takes about 15 minutes.

## Tests

`scripts/tooling.ps1 test -TestPath tests`: **1,803 passed, 5 failed.** None of the
five comes from this work. Four fail with the same message on the code from before
the merge. The fifth only reads the race database.

| Test | Cause |
|---|---|
| `test_data_integrity::test_unique_horse_per_race` | Race 688054 has two different horses named Sierra Nevada; the test predates horse-ID identity |
| `test_result_field_reconciliation` (2 tests) | The fixture's table has no `horse_id` column |
| `test_feature_store::test_append_matches_full_rebuild` | Append and full rebuild disagree on `horse_avg_position_rank_pct_in_race`, 4 of 360 rows |
| `test_independent_feature_benchmark::test_every_added_atr_column…` | `check_eligibility` rejects the registration; cause not investigated |

Six new test files cover the compiled kernels against their originals, the helpers
against one process, streaming, and the comparison tool. The comparison tool's
state check had been lost in an earlier edit, and its new test now covers it.

## What did not get faster

- **`add_race_relative_signal_features`:** 84 to 100 seconds. It was not rewritten
  and now absorbs some consolidation. `add_jockey_intent_features` is about level.
- **Elo:** still pure Python. It costs little now because it runs in the helper.
- **Saving the counts state:** 64 seconds for a 617 MB pickle, most of it file
  writing. The format was kept so that stores stay compatible.
- **Compiling:** the first build after an FE edit recompiles that file's kernels,
  up to about 50 seconds. The figures above include this cost.
- **The hardware:** an eight-thread FX-8300 and one hard disk. Two simultaneous
  builds would probably page, because one build holds about 9 GB across three processes
  (estimated, not measured; the main process peaked at 4.3 GB against 5.6 GB
  before).

Moving `add_atr_features` to the second process would balance the two FE halves
and might save about a minute. It was left out to keep this change smaller.

## Follow-ups

- Investigate `test_append_matches_full_rebuild`, which is a real disagreement
  between the append and rebuild paths, and the other four failures above.
- `_tg_wins` is one state key used by two stages. Their entities are disjoint, so
  values are right, but it should be two keys.
- Dropping the unused features remains a separate step. It was out of scope here.

## Reproducing

Logs, patches and the comparison reports are in
`data/audit/store_speedup_20261105/`. The final comparison is
`logs/equivalence_final.json`. `run_build.py` there is the script that ran each
build from a frozen copy of the data. The stores, frozen data and code snapshots in
that folder served only this comparison and can be removed; to repeat it, build
twice and compare the results. Files replaced by the merge were backed up to
`data/backups/store_speedup_20261105/before_merge/`.

```powershell
.venv\Scripts\python.exe scripts\compare_feature_stores.py LEFT_ROOT RIGHT_ROOT --report compare.json
```
