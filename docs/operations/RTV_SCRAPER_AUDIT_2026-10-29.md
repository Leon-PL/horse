# Racing TV (RaceiQ) scraper audit — 29 October 2026

> **Update, same day:** the scraper was rebuilt as `src/rtv_scraper.py`
> (collector only, not wired into features). It now also collects results,
> racecards and per-furlong sectionals, and avoids the time-zone bug in
> section 3. See [the scraper guide](RTV_SCRAPER.md).

Companion to the [ATR scraper audit](ATR_SCRAPER_AUDIT_2026-10-29.md), with
the same questions: is the data right, is coverage good, is missing data real,
and what is on offer that was never collected. Measured against
`data/races.db` and `data/rtv_cache/rtv_metrics.parquet`. Site checks used a
normal browser on racingtv.com (a handful of result pages). Nothing in the
project was modified.

## Status: code removed, cache kept, source still live

- **Pipeline:** `add_rtv_features` left the stage list on 2026-06-28, when
  `add_atr_features` replaced it. There was no head-to-head comparison: RTV
  never got its own ledger verdict. Its only measurement was inside the June
  bundle A/B of 173 unused features, which also included `rtv_fsp`,
  `rtv_speed_lost` and `rtv_top_speed`. That test was read on mean CLV, which
  CLAUDE.md now forbids.
- **Code:** `src/rtv_scraper.py` (759 lines) and `add_rtv_features` were
  deleted on 2026-10-13 in 10a1286 ("provably unreachable", ledger). Recover
  with `git show 10a1286^:src/rtv_scraper.py`.
- **Cache:** still on disk. 350,089 rows, 2023-06-10 to 2026-06-13, last
  written 2026-06-22. `docs/archive/PROJECT_OUTLINE_before_cleanup_2026-11-05.md` calls it "orphaned and safe to
  delete". **It is not:** it is the only Irish sectional data in the project
  (see below).
- **Source:** Racing TV still publishes RaceiQ (checked on races from
  29 October 2026). Its result pages load
  `api.racingtv.com/racing/iq-results/{date}/{track}/{hhmm}/comparison`, the
  same endpoint the scraper used, plus `/metrics` and `/sectional`, which it
  never fetched. The "retired 2026-10" wording in the ATR docstring refers to
  this project, not to Racing TV.

## What it collected

The `comparison` endpoint returns one value plus in-race rank for each metric:

| Code | Metrics |
|---|---|
| Flat | finishing speed % (`fsp`), top speed, 0–20 mph time (`acceleration`), stride length |
| Jumps | `fsp`, top speed, jump index (/10), lengths gained jumping, speed lost, entry speed |

## 1. Correctness — passes

On the 228,995 runners present in both caches:

| RTV metric | ATR counterpart | Agreement |
|---|---|---|
| `rtv_fsp` | `atr_finishing_speed_pct` | r = 0.978, median difference −0.02 |
| `rtv_top_speed` | max of ATR's three segment speeds | Spearman 0.946; r = 0.96 after dropping 181 ATR rows with junk segment speeds (110 mph, 90,000) |
| `rtv_stride_length` (m) | `atr_peak_stride_length` (ft) | r = 0.80, ratio 3.31 (m→ft is 3.28) |

Stored ranks match ranks recomputed from the values for 99.1% (fsp), 96.8%
(top speed) and 93.7% (acceleration) of runners. Value ranges are plausible
(1st–99th percentile: fsp 85–115, top speed 30–43 mph, stride 6.4–8.0 m). The
earlier ±15k fsp junk for non-finishers was fixed at parse in 91e7e65.

## 2. Coverage — better than the ledger says

The ledger's "things that are true" section says RTV coverage "starts in 2023
at selected tracks. Low coverage is expected". Measured in the cache window
(2023-06-10 to 2026-06-13):

| Scope | RTV | ATR, same window |
|---|---|---|
| GB races | **98.8%** (Flat 98.6, Hurdle 99.2, Chase 99.0) | 92.5% |
| Irish races | **79.6%**: 11% in 2023Q2, 47% in 2023Q4, 86–100% every quarter from 2024 | 0% |
| Finishers in covered races | 99.78% | 99.89% |

- GB overlap: 92.3% of races are in both, **6.5% (1,958) only in RTV**, 0.2%
  only in ATR, 1.0% in neither. 1,841 of the RTV-only races fall between
  2023-06-10 and 2023-08-09, before the ATR cache begins. Chelmsford City
  has no data in either source from Jun 2023 to Jan 2024, so that gap is real.
- **The start date is real.** RaceiQ begins on 10 June 2023: Haydock
  2023-06-10 has comparison data, while Thirsk 2023-06-09 and Ascot
  2023-05-13 have none. Unlike ATR, there is no older history to recover.
- **The end date is not.** Nothing after 2026-06-13: 3,978 GB and 1,264
  Irish races are uncollected.
- Metric completeness for finishers: fsp 96–97%, top speed 94–95%; Flat
  acceleration 86.6% and stride 92.8%; jumps jump index 95%, lengths gained,
  speed lost and entry speed 98%.

## 3. Missing or mismatched data, classified

- **10,926 cache rows match no DB runner.** 92.7% have no finish position and
  91.6% no metrics: non-runners listed by the API. They are harmless.
- **89 races contain the wrong race** (about 900 rows). The scraper tries UTC+1
  first in BST. On 22–25 October 2025, and on scattered other dates, the API's
  times were UTC, so the UTC+1 request returned the race an hour later. The
  name merge rejects these rows, so features are not contaminated, but those
  races lose their data. The results DB's off-times are correct: ATR's
  Chelmsford card for 23 October 2025 confirms them.
- **Design gaps that hide failures:** any non-200/404 response (rate limit,
  5xx) was treated as "no data". The incremental backfill only fetches dates
  after the newest cached date, so a failed race is never retried. In practice
  only three races are on the skip-list, but nothing would report a missed day.
- The request gap was 0.05 s. The API key sent with each request is the
  website's front-end key, hard-coded in the module and therefore in git
  history; the API host's terms were not checked.

## 4. Feature construction — same flaw as ATR

The deleted `add_rtv_features` rolled raw values over prior runs, as ATR does.
The share of each metric's variance that lies *between* races:

| Metric | Between races |
|---|---|
| `rtv_top_speed` | 0.96 |
| `rtv_entry_speed` | 0.80 |
| `rtv_stride_length` | 0.69 |
| `rtv_fsp` | 0.66 |
| `rtv_speed_lost` | 0.62 |
| `rtv_acceleration` | 0.49 |
| `rtv_jump_index` | 0.36 |
| `rtv_lengths_gained_jumping` | **0.02** |

Top speed and entry speed mostly describe the race. **Lengths gained jumping
is almost entirely horse-specific**, and so, to a lesser extent, is jump index.
These are the jumping measures ATR does not have (ATR offers only
comment-derived error tags). `rtv_speed_score` was z-scored over the full
sample, the same future-dependent pattern as `atr_speed_score`.

## 5. On the site but never collected

- **`/sectional`**: per-furlong time for every runner with its **race position
  and gap to the leader at each furlong** (e.g. "3rd | +0.24"), total time,
  FSP, and race-level **Time Index** (/10), meeting average and **time vs
  par**. Present in samples from Jan 2025, Aug 2025 and Oct 2026 (including
  Punchestown, so Ireland is covered); absent from Dec 2023 and Jun 2024
  samples. It therefore starts somewhere between mid-2024 and January 2025.
  Tracking glitches are visible (one runner at Bath 2026-10-29 shows
  adjacent splits of 18.06 s and 10.03 s), so it needs cleaning.
- **`/metrics`**: loaded by the page; content not inspected.

## 6. Access

Racing TV result pages load normally in a browser with no bot challenge, and
www robots.txt disallows only `/my-account` paths. The old scraper was not
run, and the API host's terms were not checked. Compared with ATR (bot
challenge on both transports; robots.txt disallows `/ajax/`), RTV looks like
the more workable route to TPD-style data, subject to a terms check.

## Recommendations

1. Keep `data/rtv_cache/`, and correct the `docs/archive/PROJECT_OUTLINE_before_cleanup_2026-11-05.md` "safe to
   delete" note and the ledger's "low coverage is expected" line.
2. Any sectional re-test (see the ATR audit) should include RTV's
   horse-specific jumping metrics and use within-race normalised speeds.
   Irish runners can draw on RTV where ATR has nothing.
3. If collection resumes: restore the scraper from git, replace the
   UTC+1-then-raw fallback with one conversion plus a runner-name check,
   distinguish blocked from not-found, and add the `/sectional` endpoint.
