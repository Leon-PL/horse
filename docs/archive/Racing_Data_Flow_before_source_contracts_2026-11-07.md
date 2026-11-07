Sources to feature store · store built 6 November 2026

# Racing Data Flow

The middle of the pipeline is in good shape. Once data reaches the results database, it follows one path with one owner at each step and a store that commits as a unit. The weak spots are at the edges, and the worst are in At The Races (ATR).

ATR data stops in June 2026, and the races since then are blocked by a bot check on ATR's results pages. The fresh ATR rebuild has no way into features yet. Missing ATR comments were also being recorded as zero in one production input. That is now fixed in code and reaches the store at the next rebuild.

757,580 runners, 80,536 races, 15 September 2020 to 4 November 2026. Figures describe the store as built on 6 November, read from the code and the live store.

## The map as it stands

Swipe the map sideways to see all of it.

Solid lines work today. Red dashed lines are stopped, paused or missing. Pedigree comes from Sporting Life horse pages and, like weather, is an optional step inside the Sporting Life collection. Upcoming racecards go through the same stages, seeded from the two state files.

## Seven sources, three routes

108 of 129 inputs

### Through the database

Sporting Life, pedigree and weather reach `races.db`, then one typed snapshot. ECB rates join in `process_data`. This route is the backbone and gets the most checks.

21 of 129 inputs

### Side caches, read mid-build

ATR gives 5 inputs and Racing TV gives 16. Feature stages open these caches directly, so they never pass through the snapshot. Each has its own staleness signature instead.

0 of 129 inputs

### Beside the store

Betfair never enters features. No feature code reads a Betfair file. Prices join at training and evaluation only, for CLV and profit. The prior-market features use the results `odds` column.

## Step by step

| Step | Owner | What happens | Result |
| --- | --- | --- | --- |
| Collect | `src/ingestion/pipeline.py` | Sporting Life runs first. ECB, Betfair, Racing TV and ATR then run in parallel. One source failing does not stop the others. | Source stores |
| Publish snapshot | `src/database.py` | Writes every declared column with its declared type, plus a source record. | 44 columns |
| Process | `src/data_processor.py` | Adds pedigree, cleans, converts euro prizes, sorts by race time, adds time columns, encodes categories. | 97 columns, 46 MB |
| Build features | `scripts/build_feature_store_batched.py`, `src/feature_engineer.py` | 13 chunks over 2,223 dates. Ratings run ahead in a worker. The second half of the stages runs in a finisher, starting at the ATR stage. | 1,231 columns. 16 min 47 s with collectors running, about 14 min quiet |
| Commit store | `src/feature_store.py` | Five files land together, stamped with signatures for code, data, ATR and Racing TV. | featured 2.87 GB, counts state 618 MB, ratings state 42 MB |
| Keep current | `src/store_refresh.py` | Appends new dates. Any change at or before the store's last date in results, pedigree, ATR or Racing TV forces a full rebuild. | Append or full build |
| Train | `src/training.py`, `src/model.py` | A feature list selects inputs. Betfair prices attach here. Races from 11 June 2026 stay held out. | 129 inputs in production |

## Each source and how it joins

| Source | Joins on | If a key is ambiguous | Coverage |
| --- | --- | --- | --- |
| Sporting Life | Its own race and horse IDs | Not applicable | All 757,580 runners |
| Pedigree | Horse ID | Not applicable | 99.3% to 100% by year |
| Weather (Open-Meteo) | Stored on the race | Not applicable | 97% to 100%, then 87% in 2026 |
| ECB rates | Latest rate before race date | Build fails if no earlier rate exists | Not measured |
| At The Races | Date, track, off time, horse name | Those rows get no ATR data | 232,291 runners, 30.7% |
| Racing TV | Date, track, UTC time, horse name, then Sporting Life IDs | No join for those rows | Tracking input 89% to 93% from 2024, none before 2023 |
| Betfair (at training) | Date, track, off time, horse name | Those rows get no price | 757,292 runners, 99.96% |
| Betfair (publishing) | Same name keys | **Stops all publication** | 148,129 files waiting when last checked |

## What survives each step

Snapshot44

After processing97

Feature store1,231

In any feature list449

Production model129

Nothing is lost by accident: every snapshot column reaches processing, every processed column reaches the store, and no column is fully empty. The other side of that is size. The 782 columns in no feature list hold 63% of the stored bytes.

## How far back each source reaches

Share of runners in each year. Darker means better covered.

| Year | Runners | Pedigree | Weather | ATR data | ATR history input | Racing TV tracking | Betfair BSP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 42,333 | 100% | 100% | none | none | none | 100% |
| 2021 | 128,532 | 100% | 100% | none | none | none | 100% |
| 2022 | 121,261 | 100% | 97% | none | none | none | 100% |
| 2023 | 121,195 | 100% | 98% | 25% | 17% | 49% | 100% |
| 2024 | 122,309 | 100% | 98% | 68% | 67% | 89% | 100% |
| 2025 | 119,546 | 100% | 98% | 68% | 69% | 92% | 100% |
| 2026 | 102,404 | 99% | 87% | 36% | 63% | 93% | 100% |

ATR data is the cache's rows divided by runners. The history input, `atr_energy_eff_avg_5`, uses past runs, so it stays filled for a while after new data stops. It can't tell you the data is getting old.

### The ATR gap, month by month

71%

54%

18%

No rows\
for 4 months

OctNovDecJanFebMarAprMayJunJulAugSepOctNov

ATR cache rows as a share of all runners, October 2025 to November 2026. Bars are drawn on a 0 to 100% scale. November covers only 1 to 4 November.

Monthly numbers

| Month | Runners | ATR rows | Share |
| --- | --- | --- | --- |
| 2025-10 | 12,013 | 7,885 | 66% |
| 2025-11 | 9,401 | 6,234 | 66% |
| 2025-12 | 8,721 | 5,875 | 67% |
| 2026-01 | 7,653 | 5,336 | 70% |
| 2026-02 | 6,793 | 4,805 | 71% |
| 2026-03 | 8,426 | 5,918 | 70% |
| 2026-04 | 10,461 | 6,970 | 67% |
| 2026-05 | 12,352 | 8,054 | 65% |
| 2026-06 | 10,873 | 5,821 | 54% |
| 2026-07 | 11,008 | 0 | 0% |
| 2026-08 | 10,347 | 0 | 0% |
| 2026-09 | 11,198 | 0 | 0% |
| 2026-10 | 11,829 | 0 | 0% |
| 2026-11 | 1,464 | 259 | 18% |

## What works well

- **One owner per step.** Each step has a single place in the code, and the consolidation proved output identical.
- **Typed hand-off.** Every snapshot column has a declared type, so one stray value can't flip a column the way it could with the CSV.
- **Failures stay contained.** Sporting Life runs first and the other sources are isolated from each other.
- **The store is all or nothing.** Five files commit together, and signatures catch stale code and stale data.
- **Chunk size doesn't change results.** The running-total drift between chunks is fixed. The two quarantined ATR composites are the known exception, and no feature list uses them.
- **Strong joins where it counts.** Betfair matches 99.96% of runners and pedigree is 99.3% or better in every year.

## What needs attention, in order

1

### ATR data is out of date and the rebuild can't reach features

High

The ATR feature cache has no rows for July to October 2026 and only 259 for November. Normal refresh is paused while the rebuild worker runs, and it's capped at 50 races per run anyway.

The worker has finished 5,292 of 62,432 races. Another 22,893 have a known ATR race ID, and at about 250 races an hour they should finish in about 4 days. The remaining 34,247 have no ATR ID, and that includes nearly every race from late June 2026. Getting an ID means loading an ATR results page, and ATR answers those pages with a bot check. The worker tries once an hour and is turned away, so the gap will not close on its own. The rest of the blocked group is mostly races from before August 2023.

Nothing moves the worker's output into the feature cache, so even finished races help no model. The worker also dies whenever the Claude app closes, and its status file can still say it's running.

Recommendation: build the step that promotes finished worker races into the feature cache, and make it safe to run at any time. Run the worker outside the Claude app. Treat the blocked races as an access decision, such as a licensed feed, because no code change here should get past that check.

Cost: design and code, then one store rebuild each time you promote. The proposed task for the promotion step is already waiting for you.

2

### A missing ATR comment was recorded as zero

Fixed in code

This was the one confirmed data fault reaching the production model. A missing comment became empty text, and empty text scored 0 on every comment tag. So "we have no comment" looked the same as "the comment didn't say this", and rates built from it looked real.

`atr_cmt_front_rate_5_vs_field`, a production input, in the current store

| Year | Looks filled | Of those, exactly 0 |
| --- | --- | --- |
| 2020 | 58% | 100% |
| 2021 | 90% | 100% |
| 2022 | 92% | 100% |
| 2023 | 93% | 75% |
| 2024 | 93% | 20% |
| 2025 | 93% | 17% |
| 2026 | 93% | 20% |

There are no ATR comments before August 2023, so those rows should be blank. Missing comments now stay missing, and the rates skip those runs. I ran the fixed ATR stage on all 757,580 runners and compared all 216 ATR columns with the store, without writing anything:

- Across the 48 comment columns, 20.9 million made-up values go away and no column gains a value. Median fill falls from 90% to 33%.
- 4.1 million real values move, because zeros no longer drag them down.
- The production input is now blank for 2020 to 2022 and 63% to 69% filled from 2024. Exact zeros fall to 3% to 4% of filled rows.
- The four other ATR production inputs don't change. The only other differences are the two known composites, `atr_speed_score` and `atr_stride_x_gallop`, and their derived columns. Those depend on chunk size, and this check ran as one chunk. No feature list uses them.

Status: fixed in code, not yet in the store. The next rebuild applies it, and the model needs a refit at the same time, because it was trained on the zeros.

Cost: one 14-minute rebuild and one model fit, whenever you choose. The effect on accuracy isn't known yet. `atr_speed_score` has a similar zero fault in chunks with no ATR data. It was left alone because nothing uses it.

3

### Three sources handle a name clash in different ways

Medium

ATR, Racing TV and the Betfair join at training all leave ambiguous rows unjoined. Betfair publishing stops entirely. One clash, two horses named Sierra Nevada in race 688054, is holding back 148,129 waiting files. This doesn't block feature builds, because Betfair doesn't feed them. It does hold back fresher prices for training and evaluation.

Recommendation: use one rule everywhere. Set the ambiguous rows aside, report them, and carry on. Another agent has the Betfair fix, so this applies once that lands.

Cost: small once the Betfair fix is in. A store rebuild only if the ATR or Racing TV joins change.

4

### The old build route is still in the code

Medium

`get_featured_dataset` to `_rebuild` to `FeatureStore.build` has no production callers, only 9 test files. But error messages at `src/feature_store.py:434` and `:481`, and in `build_lookahead.py:45`, tell you to use it. Its peak memory passes 16 GB. Staleness logic also lives in more than one place because of it.

Recommendation: point those messages at the batched build now, then retire the old route and move its tests over.

Cost: moderate. Moving registered code can change `fe_signature` and force one rebuild.

5

### Late data forces a full rebuild

Medium

Any change at or before the store's last date in results, pedigree, ATR or Racing TV rebuilds everything. The data signature also covers columns no model uses: the rainfall fix changed it even though no feature list reads weather. Promoting ATR races will trigger this often.

Recommendation: record how often it happens before changing anything. At 14 minutes it may be acceptable, and loosening the signature risks the stale-feature fault that produced the phantom place edge.

Cost: a small log line.

6

### Smaller items

Low

| Item | Recommendation |
| --- | --- |
| **Weather.** Collected, carried through every stage, gaps filled with UK averages, used by no feature list. Coverage fell to 87% in 2026. | Test it once properly, or stop carrying it. |
| **782 unused columns.** They hold 63% of stored bytes. | Keep for now. They have research value, and the build time saved by removing them hasn't been measured. |
| **Feature-list drift.** Research lists 030 to 035 name 6 to 11 columns no longer in the store. | Fix a list when it's next used. |
| **`src/feature_engineer.py`.** 5,008 lines with stages that depend on order. | Leave it. No fault here traces back to its size. |

## Suggested order of work

1. Comment zeros: fixed in code. Apply them at the next rebuild and refit together.
2. Build ATR promotion before the known-ID races finish, around 10 November, and decide what to do about the 34,247 blocked races.
3. Apply one join rule once the Betfair fix lands.
4. Repoint the error messages and retire the old build route.
5. Log how often full rebuilds happen, then decide about weather.

The promoted model (`20261104_235300_444862`) already predates the current store, so none of this reaches production until you ask for a refit.

## Not measured

The accuracy effect of the comment zeros. How often full rebuilds happen. The build time that removing unused columns would save. ECB rate coverage. Why ATR tops out near two-thirds of runners: earlier work found its sectionals cover Great Britain only, which would explain it, but I didn't re-check that here. The route totals of 108, 21 and 0 inputs are grouped by column name prefix, so treat them as a close guide and not exact lineage. This page could not be viewed in a browser before publishing, so tell me if anything looks wrong on your screen.