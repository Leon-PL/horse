# ATR sectional scraper audit — 29 October 2026

Question: is `src/attheraces_scraper.py` retrieving the data correctly, is
coverage good, is missing data real, and what does the site offer that is not
collected? Measured against `data/races.db` (754,842 rows, 2020-09-15 to
2026-10-28) and `data/atr_cache/atr_sectionals.parquet` (232,138 rows, last
written 2026-07-04). Site checks used a normal browser session (small samples
only); nothing in the cache, skip-list or feature store was modified.

## Verdict

The scraper parses correctly and, inside the dates it covers, coverage is
near-complete. The weak ATR results come from three other places, in order of
likely impact:

1. **Feature construction**, not the data: most derived metrics measure race
   conditions rather than the horse, one is sign-inverted and half-blanked.
2. **Time coverage**: nothing before 2023-08-10 or after 2026-06-20, so 48% of
   training rows have no ATR history at all.
3. **Uncollected fields**: distance ran (ground covered), top/final speed,
   time-to-30mph, and positional data derivable from what is already cached.

## 1. Correctness — passes

- Redcar 2026-06-20 15:22 (local), all six runners: cached split times,
  running totals, the three sectional speeds, finishing speed %, efficiency
  score and peak stride length/frequency match the live page exactly.
- Official position agrees with the results DB for 99.76% of runners (547
  disagreements, e.g. Ayr 2026-06-19 20:00 where the DB has a different 3rd;
  worth a look by `audit_result_amendments.py`, since removed; it is in git history).
- Implied speed from final time vs distance is plausible for all but 54 rows
  (11 races with truncated or "Not Available" section markers).

**Bug (minor):** the BST fallback. Stored off-times are UTC; the scraper tries
UTC+1 first, then the raw time. When the correct race has no sectionals, the
raw-time request lands on the race an hour earlier. 13 cache races hold another
race's runners (e.g. key Ayr 2026-05-20 14:12 holds the 13:12 field). The merge
on horse name rejects them, so no feature is contaminated, but the fallback
should be removed or validated against the DB field.

## 2. Coverage inside the cache window (2023-08-10 .. 2026-06-20) — good

| Level | Coverage |
|---|---|
| GB races | 28,155 / 28,552 = 98.6% (Flat 98.5%, Hurdle 98.9%, Chase 98.6%) |
| Finishers in covered races | 99.89% |
| Non-finishers | 0% — genuine, TPD publishes no sectionals for them |
| Stride data | 55% of races — genuine (Hamilton 2026-05-27 has no stride on the site) |
| Irish races | 0% — skipped by design; ATR has no TPD data for them |

Missing races, classified:

- **363 on the no-data skip-list: real.** 212 are Chelmsford City from
  2023-08 to mid-January 2024 (zero coverage, then full from 2024-02;
  Chelmsford 2023-11-16 17:30 verified empty on the site). Royal Ascot
  2026-06-19 17:35 verified empty.
- **34 neither cached nor skip-listed: scraper gaps.** 20 on 2023-08-10 (the
  backfill stopped mid-day), 6 at Wolverhampton 2023-12-26 (the site *has*
  data — 12 runners verified), 8 at Hexham/Fakenham in May 2026.

## 3. Time coverage — the large data gap

- **Before 2023-08-10:** 29,913 GB races never requested. The backfill started
  2026-06-26, newest first, and ended at 2023-08-10. A random sample of 38 races
  from 2021 to July 2023 found sectionals on ATR for **15 (39%)**: Ascot,
  Doncaster, Fontwell, Ripon, Plumpton, Worcester, Brighton, Newcastle,
  Southwell, Hereford and Nottingham had data; Newmarket, Sandown, Goodwood,
  Kempton, Cheltenham, Newbury, Thirsk, Ayr, Perth, Market Rasen, Taunton,
  Salisbury, Beverley, Wetherby and Hamilton did not. So roughly 11–12k older
  races are recoverable; pre-2023 TPD coverage is partial by course, not
  absent.
- **After 2026-06-20:** 3,749 GB races not collected. Nothing schedules ATR
  updates, so live and hold-out features roll over a history frozen at
  20 June.
- **Effect on the model:** 364,944 of 754,842 DB rows (48%) predate the cache,
  so every ATR feature is NaN there. Only 34% of all rows have at least one
  prior ATR run. For GB rows from 2024 onwards it is ~90%. A model trained on
  the full history learns ATR from about half its rows.

## 4. Feature construction — the likely main reason ATR looks weak

Measured on the 232k cached runs with `src/atr_features.compute_atr_run_features`:

| Per-run metric | Non-null | Share of variance *between* races |
|---|---|---|
| `atr_early_pace` | 99.9% | **0.99** |
| `atr_max_speed` | 99.9% | **0.98** |
| `atr_turn_of_foot` | 99.9% | 0.90 |
| `atr_closing_speed` | 99.9% | 0.86 |
| `atr_finishing_speed_pct` | 99.9% | 0.68 |
| `atr_efficiency_score` | **55.0%** | 0.57 |
| `atr_closing_in_hand` | 36.2% | 0.95 |

- **Raw mph speeds encode the race, not the horse.** 98–99% of the variance in
  early and top speed is between races (distance, going, surface, code).
  A horse's rolling average mainly records which races it ran in, which the
  model already knows. The horse-specific part (1–2%) is buried.
  `atr_early_pace` is also the "Start to *X*f" segment, whose length varies
  from 2f (5f sprint) to about 4f (2m4f hurdle) and includes the standing
  start.
- **Efficiency score is inverted and half-deleted.** It is an error score:
  median 0.47 for grade A++, 5.4 A, 11.5 B, 21.7 C, 42.4 D, 72.7 E, 147 F (win
  rate falls from 18.2% to 6.9% down the grades). `RUN_METRICS` marks it
  higher-is-better, so `atr_efficiency_score_best_3` takes the *worst* run.
  `_PLAUSIBLE` caps it at 40, which blanks 43.6% of values (every E and F,
  half of D), so the feature exists mainly for runs that went well.
- **"Adjusted placing" is sometimes the saddle-cloth number.** In 12.4% of
  races (rising to ~25% in 2025Q3 and 2026Q2) `data-adjplacing` equals the
  cloth number for every runner, the site's fallback when no adjusted order
  exists. `atr_upgrade_clean` (official − adjusted) is noise in those races.
- **Comment tagger false positives:** a bare "behind" tags a run as held up
  (2.2% of runs, including "always behind"); "never nearer" tags it as eased
  (0.8%), which then gates `atr_closing_in_hand`.

The ledger's 2b/25 results were measured on these constructions, so they test
this feature build, not the information in sectionals.

## 5. On the site but not collected

- **Distance ran** (`/ajax/sectionals*/{id}/distanceran`, labelled beta): each
  runner's ground covered and its difference from the race median in lengths.
  Chester 2026-06-13 14:10 ranges from −2.43 to +1.60 lengths. `distanceran` is
  listed in `_AJAX_KINDS` but `_fetch_one` never requests it. Ground lost
  wide is a classic hidden-form signal.
- **Newer endpoint `/ajax/sectionals-razor/`** (what the page now links to;
  the old `/ajax/sectionals/` still answers): the tools fragment adds **top
  speed**, **final speed crossing the line** and **time to reach 30 mph** (a
  measured break/start), absent from the old fragment and the cache.
- **Already cached but unused:** running totals at every marker give each
  horse's position and time behind the leader at each furlong. That is a
  direct run-style and pace measure, better than raw segment mph. Per-section
  FRONT/REAR flags are on the page too.
- **Stride frequency per furlong** (only per-furlong stride *length* is
  parsed; peak frequency is kept).
- **Race-level**: early-pace indicator and race finish % (derivable from
  runner data).
- **Racing TV cache** (`data/rtv_cache/rtv_metrics.parquet`, 350k rows,
  2023-06 to 2026-06, includes Irish courses such as Dundalk, Curragh,
  Leopardstown and Punchestown): unused since the RTV stage was removed. It is
  the only sectional source here that covers Irish races. See the
  [Racing TV scraper audit](RTV_SCRAPER_AUDIT_2026-10-29.md).

## 6. Access

Tested today with one request per transport: curl_cffi is challenged (reported
as 429) and the curl binary receives the 3 KB "Client Challenge" stub. In June
the curl binary passed. A normal browser passes. The earlier feasibility note
records that robots.txt disallows `/ajax/`, which every sectional fragment
uses. Resuming collection (post-June races, the pre-2023 partial history,
distance ran) therefore needs an access decision — a licensed TPD feed, or a
deliberate choice about browser-based collection — rather than a code fix.

## Suggested order of work

1. Rebuild the ATR features from the existing cache: within-race normalised
   speeds (vs race median, or rank), positions and gaps from running totals,
   efficiency as log score with lower-is-better and no cap, guard
   `upgrade_clean` where adjusted placing equals cloth, tighten the two regexes.
   This is a materially changed hypothesis under ledger 25, so pre-register it
   and test once on the development folds, evaluating on rows with ATR history.
2. Fix the BST fallback and the 6 Wolverhampton 2023-12-26 races when
   collection resumes.
3. Decide on access; then distance ran, the razor tools fields, post-June
   races, and the ~39% of pre-August-2023 races that have data.
