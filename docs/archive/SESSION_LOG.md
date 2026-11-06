# Session log — long autonomous session from 2026-10-13

**Historical session record.** Preserve entries as written. Use the
[README](../../README.md) for current operation, [CONTINUATION](CONTINUATION_before_cleanup_2026-11-05.md)
for the latest handoff, and [implementation progress](../MARKET_PERFORMANCE_PROGRESS.md)
for the evidence index.

Running state for the session driven by [NEXT_AGENT_BRIEF.md](NEXT_AGENT_BRIEF.md).
Newest entries at the bottom. Measurements go in
[NEGATIVE_RESULTS.md](../NEGATIVE_RESULTS.md); this file only tracks where the
work is, so it survives a context compaction.

## 2026-10-13

**Start state:** `project_status.py` exit 1 — store STALE (stamped
`724ce1afe9ad7611`, code `3e3b9bbb5d79bacc`), no promoted model, DB to
2026-10-11.

1. **Concurrent session's 12 files committed as one unit** (`c59ad23`):
   resumable race-level DB sync, transactional feature-store commits + OS lock,
   optional race-temperature win calibration. HEAD already carried half of the
   calibration change set (training.py passed `win_calibration_method` into
   code that did not accept it). 528 pass in the working tree.
2. **`fe_signature` made line-ending invariant** (`9b3f645`). It hashed raw
   bytes; the tree is mixed LF/CRLF and git autocrlf rewrites files on any
   checkout, so one git operation could force a ~56 min rebuild. Measured:
   converting `feature_store.py` alone moved `3e3b9bbb…` → `e7f6193d…`.
   New signature `2d3192c11154211b` is identical in the main tree and a fresh
   worktree. **HEAD verified in a fresh worktree: 517 passed, 12 skipped.**
3. **Flat draw backfill is DONE, not pending.** `draw_backfill_done` covers
   46,791 races through 2026-06-13; the 3,645 later flat races were scraped by
   the fixed scraper. Check: per-race Spearman(draw, weight) in flat handicaps
   is ~0 in every month 2020-09..2026-10 (saddle cloths would give ~-1).
4. **Found: the old sync truncated busy days at 50 races.** 105 dates have
   exactly 50 races (vs 53 at 49, 24 at 51). 2020-21: 50 such days and *zero*
   days above 50; 2026: 25 and 5. 2022-25 (a different, uncapped collection
   path): 30-38 days/yr above 50. The old sync also skipped any race page
   whose fetch raised, while the date counted as present. Added
   `database.queue_history_audit` + `--audit-history` to relist every legacy
   date through the new resumable ledger. DB backed up to
   `data/backups/races_2026-10-13_pre_audit.db` first.

**Next:** 14-day sync (smoke test, running) → full history audit → store
rebuild (~56 min) → Phase A stages 1-8.

### 2026-10-13 afternoon

5. **History audit done**: +543 races / 6,410 rows (Irish cards cut by the
   50-race cap, and skipped pages). 3,472 listed races yield nothing: sample of
   80 = all abandoned / dormant / void / Arabian / 404. Sync now closes those
   (`RaceNotRun`). DB: 755,706 rows to 2026-10-12.
6. **Weather + trainer-travel FE stages removed** after a parity check (10
   columns dropped, other 1,058 bit-identical on 106k rows). Store no longer
   calls Open-Meteo at build time. New fe_signature `e522ca1b86174217`.
7. **promote() now refuses unverifiable runs** (all 8 saved runs predate the
   trainer_freq leak fix; 6 unsigned).
8. **Batched build stamps the signature it started with**, aborting if FE code
   changes mid-build (how the old store got "stamped mid-edit").
9. **Public Betfair BSP files** downloaded and validated (4,440 files; BSP
   99.99% within 1% of BASIC; 95-100% DB coverage every quarter incl. the
   2020Q4/2021 gaps). `data/processed/betfair_sp_public.parquet`. Carries
   morning/pre-play VWAP and traded volume. Pre-off prices sit short of BSP
   (median morning VWAP/BSP 0.8925).
10. **Golden hold-out pre-registered**: races from 2026-06-11 (`3800c33`),
    because no Betfair-priced analysis has ever seen them. No T-5 price there
    (BASIC needed from the operator); morning protocol can be confirmed now.
11. Harnesses written + tested, not yet run: `wf_market_offset.py` (plan 4c,
    protocols t5/morning) and `wf_ranking_compare.py` (plans 3a/4b).
12. Store rebuild running (started 14:52:57, 7 chunks).

**Next:** rebuild done -> project_status -> train + promote (gate 1) ->
wf_ranking_compare (gate 2) -> wf_market_offset t5 + morning (gate 4 half) ->
5a-lite volume/continuation test -> golden hold-out once -> Phase C.

### Priors, written before any Phase B result (2026-10-13 ~14:00)

So a surprise can be recognised as one rather than rationalised afterwards:

- **3a (GBM vs logistic):** on ~40k races I expect the GBM to rank about the
  same as logistic regression (|top-1 gap| < 1pt), both ~5pt below the
  market. The 646-race "GBM worse" result was probably noise.
- **4b (loosened GBM):** a small change either way (< 0.5pt top-1).
- **4c (market offset):** a small log-loss gain over the market (delta race
  LL between -0.001 and -0.004 nats), consistent with Benter alpha ~+0.046;
  selections negative at BSP after 2% commission; median CLV < 1.
  P(passing ACTION_PLAN 7) < 5%.
- **5a-lite (volume / move continuation):** partial correlations ~0; the move
  continuation near the ledger-8 figure of 0.01. P(any test passing) ~15%,
  most of it mechanical (VWAP noise), which is what the liquid-only
  robustness rows are for.

New source found while waiting: the BHA API publishes per-fixture going
reports (`/bha/v1/fixtures/{year}/{id}/going`): stalls by distance, rail
movements, GoingStick with inside/centre/outside splits, back to 2020, GB
only. This is the plan-5a-bis collection path. Queued after the core results.

### Hold-out confirmatory tests, pre-registered (2026-10-13 ~16:00)

Written before any hold-out number exists. Each hypothesis gets ONE test on
the golden hold-out (races from 2026-06-11); with ~4 such tests planned this
session, the pass bar is one-sided p < 0.01.

- **H1, sibling lead (ledger 2h).** Population: hold-out runners with <= 1
  prior UK/IRE start and sibling history, in races with a complete BSP market.
  x = `sibling_best_or_vs_field`; y = won - p_BSP; both within-race demeaned,
  price basis (p, logit p, p^2) on p_BSP partialled out, race-clustered
  bootstrap SE. Prediction from discovery: partial corr < 0 (discovery
  -0.0641, t = -3.9, n = 3,746, vs the T-5 price). Pass: one-sided p < 0.01.
  Expected n ~3,000, SE ~0.018.
- **H2, atr_closing_in_hand (ledger 2b): not tested.** The ATR cache ends
  2026-06-20, so the hold-out holds 10 days of sectionals. Even after a
  2.5-hour backfill (~35k runners, SE ~0.0053) power at the discovered
  +0.0105 is about 50%, and the ledger already rates that size economically
  negligible. Recorded as untestable rather than spending the hold-out on it.
- **H3/H4:** the plan-4c candidate(s), fixed from development folds only, on
  the morning protocol (the only one with hold-out entry prices).

### 2026-10-13 late afternoon

13. **H1 (sibling lead) FAILED on the hold-out**: partial -0.016, t -0.89,
    p 0.19, n 3,711. Closed; ledger 2h.
14. **Census check**: DB holds every non-Arabian Betfair race bar 3 Sporting
    Life omissions. Fixed Aint/List/Nott abbreviations in betfair.py.
15. **Serving guards**: features_for_live_cards refuses unsigned stores, stores
    behind the DB, and history ending >4 days before the card.
16. **project_status** now recomputes data_signature (deep), checks store vs
    DB date, and verifies a new ATR-cache fingerprint (`src/feature_inputs.py`;
    the sectional cache was read inside FE outside both signatures).
17. **BHA going reports**: 37 months (2020-09..2023-09) downloaded; the API
    then returned 429 for every request even at 5 s spacing - a long-window
    quota. Resume later: `python scripts/fetch_bha_going.py --download --delay 5`.
18. Store rebuild consolidating (staging `.commit-qvb8rtac`).

**Next:** stamp_atr (scratchpad) -> project_status deep -> train+promote ->
microstructure screen -> wf_ranking_compare -> wf_market_offset t5/morning.

### 2026-10-13 evening

19. **Store rebuilt and verified**: 755,706 rows to 2026-10-12, fe
    `e522ca1b86174217`, data `b8676e9b1dcd6adf`, ATR stamp `ac2f21f41f9535c8`
    (cache untouched since 2026-07-04, before the build). project_status: DB,
    store (code AND data) and runs OK; only "no promoted model" blocks.
20. **Microstructure (free 5a)**: morning money predicts later moves and the
    result beyond the morning price, but NOTHING beyond BSP (T2 +0.0004, T6
    +0.0007). T-5 timing edge: top-excess-volume CLV 0.9953 vs band 0.9809 -
    real, < 1.0, fails. Morning-VWAP "1.0187" is look-ahead (flagged before
    the run). Honest 11:00-price test pending: `extract_betfair_bsp.py` now
    emits `ltp_1100`/`ltp_3600s`; re-extraction running to
    `data/processed/betfair_bsp_early.parquet`, then
    `python scripts/microstructure_clv.py` (protocol m11).
21. **Matchbook harness works** (read-only check: login, 71 events/12h,
    markets with prices). No balance/account calls made.
22. Production training running (`train.py --no-refresh --promote`).

### 2026-10-13 night

23. **Production run `20261013_162640` trained and promoted; project_status
    went fully green (gate 1)** - then invalidated, correctly, by item 25.
24. **Plan 4c first run: +11.62% at BSP with CLV < 1 - tripwire fired.**
25. **Found the leak: `sex` is scrape-time** (encodes each horse's future
    career; impossible-coded runners +21.5% model-free). Fixed in `e689d98`
    (birth sex only), store rebuilt at `63d88ff721a02632`. Without the leak
    plan 4c's race log loss gain is 0.0000; on the clean store -0.00010
    [-0.00040, +0.00020], all rules' CIs span zero. Official rating verified
    race-day. Ledger 15. New guard: `scripts/audit_field_timing.py`.
26. **Honest early-price timing tests** (`ltp_1100`/`ltp_3600s` extracted):
    following morning money at 11:00 reaches break-even (CLV 1.0000, CI
    touches 1.0; ROI +0.85% CI spans 0), gone by T-60. Fails ACTION_PLAN 7.
27. **Canonical Betfair parquet was missing 99,917 runners** (2020Q3-Q4,
    2021Q3-Q4: two tars never extracted). Fixed.
28. **BHA going reports complete**: 9,311 fixtures, 100% GB meeting join.
    Draw-signal feature work deferred (low prior, days of parsing).
29. **ATR triage re-confirmed on the clean store** (20/21 priced);
    **closing-in-hand closed** (offset model with it alone: 0.0000).
30. **Gate 4 has fired.** Roadmap drafted in `docs/archive/ROADMAP.md`.
31. Running: `wf_ranking_compare.py --last-folds 8` (gate 2) and the clean
    production retrain (`train.py --promote`).

**Next:** ranking result -> roadmap's last line -> live end-to-end check on
the new model -> final status check -> report.

### 2026-10-13 late night - session close

32. **3a/4b (ledger 16)**: GBM ties logistic on top-1 (+0.36 pt, CI spans 0),
    better on log loss; loosening hurts; market +5 pt. **Gate 2 fired.**
33. **Clean production run `20261013_220519` promoted; project_status green
    on all four links** (deep check, exit 0).
34. **Serving bug fixed** (ledger 13): fresh racecard scrapes carried text
    horse_ids, missing the store's integer-keyed history - 25/149 model
    features wrong on 68-98% of runners on every first load and odds refresh.
    Verified end to end through a forced fresh scrape.
35. **Roadmap committed** (`docs/archive/ROADMAP.md`). Gates 1 passed, 2 and 4 fired.

Open, for whoever continues: BASIC data June-Oct 2026 (operator) for any T-5
hold-out test; stalls/rail feature work (data collected); GoingStick fetch;
move the BHA aliases into src/utils at the next rebuild needed anyway.

## 2026-10-14 evening - EXPLORATION_PLAN session

**Start state:** `project_status.py` ready (fe `221fcd4db0fc5631`, run
`20261014_105017` promoted); tree clean; BHA rail backfill running (not a store
build).

36. **Idea B set up.** Found a bias the earlier timing tests (ledgers 14, 19)
    missed: a back bet matched before a non-runner is withdrawn has its odds cut
    by the reduction factor; BSP is not. `extract_betfair_bsp.py` now records,
    per cut (18:00 eve, 08:00, 11:00, T-60, T-5), LTP age, LTP change count,
    later reduction factors, and withdrawn runners' book share; `--jobs` runs
    tars in parallel. Writing to `data/processed/betfair_timing.parquet` (the
    canonical BSP parquet is untouched). Tick ladder + reduction helper in
    `bet_settlement`. Ledger 20 pre-registered.
37. **Idea A step 0 - blocked on the operator (terms of service).** Sporting
    Life result pages carry every Tote dividend since 2020 (`tote_win`,
    `place_win`, `exacta_win`, `trifecta`, `swingers`, `straight_forecast`
    (CSF), `tricast`, `place_pot`/`quad_pot` with pool and winning stakes;
    `deductions`; checked 2020-09-19, 2022-03-05, 2025-10-04) but no
    exacta/trifecta pool sizes. **Its terms say "screen scraping is expressly
    prohibited"** - the plan's Racing Post rule applies, so no 80k-page
    backfill without the operator's say-so (the main pipeline already scrapes
    it; that inconsistency is the operator's to resolve). At The Races has one
    results page per day with dividends (~2,200 requests), robots.txt allows
    `/results/` (but disallows `/ajax/`); its terms page and a results page
    both returned HTTP 429, terms unread. Placepot takeout measured from one
    page: pot x winning stakes / pool = 72.9% returned (~27% deduction).
38. **Idea C data check:** Open-Meteo Previous Runs API has 24-48 h-ahead
    precipitation forecasts from January 2024 for most models (GFS 2 m
    temperature from 2021-03); the Historical Forecast API stitches the first
    hours of each run (too short a lead). So C has ~2.4 development years.
39. **Timing parquet built** (`data/processed/betfair_timing.parquet`, 746,812
    runners, prices identical to the canonical file). 21.7% of runners have a
    later withdrawal that would cut an 11:00 strike; 11:00 prices are inflated
    almost exactly by that factor (mean log(11:00/BSP) +0.41 where later
    reductions exceed 30%, -0.13 after applying them, vs -0.12 with none).
40. **Idea B dry run (n_boot 50, final run pending):** fails the pre-registered
    bar in every universe (G +0.10 / +0.15 / +0.16 CLV points vs 0.5).
    **Ledgers 14 and 19 overturned by the reductions**: m11 median CLV 1.0000
    -> 0.9849, ROI +0.85% -> -1.84% (dry-run CI; official [-3.75, +0.12]); H2 1.0122 -> 0.9963.
    Descriptive lead: the policy's early strikes alone have median CLV 1.016
    [1.008, 1.024], but two ticks of spread erase the U1 gain. Re-extracting
    with a waived-under-2.5% reduction variant before the official run.
41. **Idea D stopped at step 0** (ledger 21): no official European breeze-up
    times 2020-2026.
42. **Idea E phase 1** (ledger 22): break-even ~0.7 ticks (balanced flow),
    1.2-1.7 ticks against informed flow; GBP 303k/day morning volume. Phase 1
    does not stop it; phase 2 needs the operator.
43. **Idea C pre-registered** (ledger 23); forecasts for 80/86 courses fetched
    (Open-Meteo hourly cap; resumable).
44. **Idea B final (ledger 20): fails** - G +0.096/+0.145/+0.158 vs 0.5.
    Ledger 14/19 correction holds with reductions >= 2.5% only (0.9856,
    0.9968); corrections noted in ledgers 14, 19, ROADMAP, ACTION_PLAN.
    Exploratory: the lead is all 11:00 strikes (median 1.049 after a tick);
    not stale, survives removing every morning field (LightGBM 1.0435), and
    survives on strikes whose last LTP change was a fall (1.0336). Frozen
    rule written into ledger 20; needs ~600 unseen strikes or an order-book
    recording.
45. **Idea C (ledger 23): fails** - partial +0.0060 (p 0.041) < 0.01; economic
    rule median CLV 0.90. All 86 course forecasts cached.
46. **Idea A (ledger 24)** recorded as blocked on the operator.
47. ROADMAP addendum section 7 written.

## 2026-10-15

48. BHA race backfill restarted (detached, from the `feature/course-profile`
    worktree; log `data/bha_going/races_backfill.err.log`).
49. `data/betfair/data (x).tar` renamed by content (2020.tar, 2021.tar,
    2020-08to2021-04.tar, 2021-08to2022-01.tar, 2026-05to06.tar); they are the
    only source of 2020-2021, not duplicates. Operator added BASIC for
    2026-07-01..2026-10-10 (`2026-07to10.tar`), appended to
    `betfair_timing.parquet` (783,476 runners; ~98% of DB races Jul-Oct).
    **Hold-out gap: 2026-06-11..06-30 (775 races) still has no BASIC data.**
    The canonical `betfair_bsp.parquet` is unchanged (still to 2026-06-10).
50. Operator added BASIC for June 2026 (`2026-06.tar`): appended to
    `betfair_timing.parquet` (791,026 runners). **Hold-out BASIC now complete
    2026-06-11..2026-10-10**: 11-30 June 100% of DB races, Jul-Oct ~98%;
    43,795 hold-out runners, 99.6% with an 11:00 price. Not yet used for any
    test (the frozen 11:00 rule needs ~600 strikes; ~4 months gives ~150).

## 2026-10-25

51. **Project audit integrated** ([report](PROJECT_AUDIT.md)). Independent
    local starter-field evidence repaired 3,297 result fields and removed
    6,445 extra rows, preserving true non-finishers. Original affected rows
    were backed up before applying one fingerprint-checked transaction.
    The rebuilt store has 749,261 runners / 79,664 races through 2026-10-12,
    FE `221fcd4db0fc5631`, data `f2a120a477c117a1`. Twenty-two fields remain
    unresolved. Ingestion now validates exact field counts and replaces
    verified fields atomically, without losing separate enrichment.
52. **New production: `20261025_193016_331639`**, replacing and preserving
    `20261014_105017`. The trainer now defaults to a 90-day calendar holdout
    and refits all eligible rows at the early-stopped tree count. New run:
    149 features, 629,311 fitting rows through 2026-07-08, seven-day purge,
    holdout 2026-07-15..2026-10-12, seed 42, LightGBM win/place, default
    Platt/isotonic, no market blend. On the same 3,471 complete races / 32,357
    runners (947 non-finishers), Brier 0.0863669 -> 0.0860888, binary log loss
    0.2990108 -> 0.2967461, race log loss 1.8970388 -> 1.8773931. The fixed
    gate required all three scores to avoid worsening. Paired race log-loss
    delta CI [-0.0336752, -0.0088206]; race-averaged Brier CI crosses zero.
    Dates were already inspected elsewhere: **retrospective evidence**, not
    untouched confirmation. The closing-price reference remains stronger.
53. **Final verification: 686 tests passed.** Actual LightGBM, XGBoost and
    CatBoost fits verify full refits and holdout exclusion. Saved win/place
    forecasts match public inference on 32 shuffled fields; a further replay
    through the promoted loader passed on eight fields. Final deep status
    verifies database/store/code/model consistency, with a 13-day age warning.
    Exact local source archives, runtime versions, comparison forecasts and
    the promotion decision are retained under `data/audit/20261025/`.
54. **Market plan implementation began** ([first comparison](../research/MARKET_COMPARISON_001.md),
    [progress](../MARKET_PERFORMANCE_PROGRESS.md)). Added strict shared probability
    scoring, faithful timestamped T-5 field/price replay, all-history
    `fit_as_of` with a shared win-only research path, and a frozen chronological
    market-ladder benchmark. On 3,380 identical 2026Q3 races /31,235 runners,
    independent race log loss 1.873924 trails market 1.759440; Brier 0.086853
    versus 0.083297. Pre-test validation gave the independent forecast zero
    blend weight; its small market correction adds no demonstrated gain.
    These are retrospective LTP-reference results, not fresh confirmation or
    executable-price returns. Production is unchanged. Full suite 726 passed;
    61 further focused checks passed after diagnostic corrections. Two unused
    sectional composites and their eight descendants have future-data
    dependence and are excluded from the next, not-yet-run feature trial.

## 2026-10-26

55. **Independent sectional trial completed; candidate rejected**
    ([report](../research/MARKET_COMPARISON_002.md)). Before fitting, 206 eligible additions
    passed actual feature-function temporal checks; no equivalent earlier
    independent full-family proper-score comparison was recovered. Six fixed
    historical fits compared 149 versus 355 inputs on 2025Q3, 2025Q4 and
    2026Q1. On 9,356 races / 85,863 runners, race log loss worsened 1.872045
    -> 1.876713, paired delta +0.004668, 95% interval [+0.000852, +0.009229];
    date-cluster sensitivity agrees. Every quarter worsened; pooled runner
    Brier and binary log loss also worsened. Retain 149 inputs; no subset
    search or production change. Secondary T-5 context on 8,942 identical
    fields gives baseline/candidate/market race log loss 1.868796 / 1.872976 /
    1.749283; neither independent forecast approaches parity, including the
    fixed freshness subset. Local replay overlaps agree; all exclusions,
    fitted artifacts, source snapshots and paired forecasts are preserved.
56. **Feature-selection guard extended after the frozen experiment.** The ten
    future-dependent sectional composites/descendants are now excluded even
    with all-feature selection or an explicit allowlist. Regression cases
    reproduced both holes before the fix. Production and both trial recipes
    already omitted them, so their probabilities and feature values do not
    change. A win-only inference path also avoids unused place/betting work
    while reproducing full-path win probabilities exactly. Final regression:
    737 passed, 24 warnings. A 64-field / 635-runner production replay has
    saved win-probability error 1.67e-16 and full versus win-only error zero.
    Deep status passes; data are 14 days old after midnight. Original fit,
    assessment and post-guard implementation sources are preserved separately.
57. **Bounded race-objective trial completed; neither challenger advances**
    ([report](../research/MARKET_COMPARISON_003.md)). Recovered the older scratch comparison
    from Git; it removed actual non-finishers and lacked the repaired protocol.
    Six fixed challenger fits used the same 149 inputs and binary forecasts
    on 9,356 races / 85,863 runners. Conditional linear logit loses all proper
    scores (race LL 1.890830 versus 1.872045). Race-softmax trees improve Brier
    .08754346 -> .08736141 in every quarter, and mean race LL to 1.871035, but
    primary LL delta interval [-.005101,+.003276] crosses zero; date sensitivity
    agrees. Preserve the secondary gain as unpromoted research evidence;
    do not shift the gate or start a hyperparameter sweep. On 8,942 matched
    fields tree/market LL 1.867684 / 1.749283; no parity. Full suite 742 passed,
    24 warnings; nine real model artifacts replay shuffled forecasts to
    1.11e-16. Final deep status passes with the existing 14-day data-age warning.
    No production, canonical data, feature values or default feature recipe
    changed. Registered choices, prior source, all models, paired forecasts,
    source archives and checksums are retained.
58. **Monthly versus quarterly refitting completed; primary benefit unresolved**
    ([report](../research/MARKET_COMPARISON_004.md)). All six new native fits completed before
    score inspection; three first months share their exact quarterly forecasts.
    Same 149 inputs and nine-month population: 9,356 races / 85,863 runners.
    Race LL 1.87204489 -> 1.87097477 improves in all quarters, but difference
    interval [-.00436065,+.00347671] crosses zero; date sensitivity agrees.
    Brier .08754346 -> .08746770 improves in every quarter, with a negative
    pooled one-sided upper95 bound. Preserve the secondary evidence; the fixed
    primary gate remains unmet. Mean parameter age 52.46 -> 22.21 days.
    Matched monthly/market LL 1.86755050 / 1.74928256 closes only 1.04% of the
    market gap at the point estimate. Freshness sensitivity also fails parity.
    Stop this cadence comparison without a sweep or production change.
    Full suite 745 passed (24 warnings); nine saved artifacts replay 576
    shuffled fields / 5,307 runners to 1.43e-14. Saved calibration cutoffs,
    feature lists, methods, fitting cutoffs and exact shared-month identity
    pass. A reporting-only change makes project status show the actual saved
    parameter cutoff separately from feature freshness (32 focused checks
    passed). Deep status passes, with data 14 days old and production parameters
    fitted through July 8. Canonical data, feature values and production stay
    unchanged; source/evidence/registration hashes and all forecasts are retained.
59. **Current-data deployment fitting completed and activated**
    ([report](../research/DEPLOYMENT_FITTING_005.md)). Shared training/CLI/app now support
    explicit settled cutoffs and inherit a fingerprinted evaluated recipe.
    New refits save their actual fitting/calibration dates and empty test
    metrics, linking the earlier evaluation separately. Results-only refresh
    avoids optional unbounded backfills. Preserved and verified all five old
    store files, raw CSV, production pointer and a consistent database backup.
    The first append exposed full-matrix memory duplication; it was stopped
    before any staged commit and all old file hashes matched. New streamed
    consolidation shares one writer with the batched builder, preserving
    native calculations, processed columns and transactional state. Native
    parity/rollback tests pass; all 749,261 historical rows across 1,163 columns
    match exactly. Added 4,473 validated rows; store 753,734 runners / 80,135
    races through Oct25, FE221fcd4db0fc5631 / data f3259ac47f8249b0. Sixteen
    recent inconsistent fields stay pending. Full suite 769 passed,108 warnings;
    35 focused checks passed. Run **20261026_140950_205296** fits 665,252 eligible
    runners / 71,207 races through Oct19, with purged training-only calibration.
    Complete-field policy excludes 162 invalid/degenerate fields / 1,549 input
    rows; true DNFs stay. Original/save-load win and place probabilities match
    exactly on 64 fields / 574 runners, including after production activation.
    Prior **20261025_193016_331639** remains available for rollback. This is the
    selected base recipe's operational refit; no new proper-score, forward-test
    or market-parity claim. Uncertain research challengers remain unpromoted.

60. **Forward baseline forecast recorder006 complete** (2026-10-26). Added a
    shared manual CLI and Forecast record page using the
    actual pinned model, fresh one-race Sporting Life cards and refreshed
    Matchbook WIN GBP best-back prices. Source, recipe, runtime, first schedule,
    quote age, complete fields and actual publication are recorded and checked.
    Exclusive attempt folders preserve all available evidence; first eligible
    pairs cannot be replaced. Missing prices and field changes never produce
    a partial renormalized pair. Native live inference took about two minutes;
    the first nine-runner Southwell capture correctly failed card freshness.
    Preserved that study, added an exact fresh-card recheck after inference and
    registered a revised study. A nine-runner Kempton capture passed freshness.
    Both saved model inputs and raw prices replay with zero probability error.
    Today's pairs precede the October27 registered start and are monitoring
    only. Shared assessment preserves result snapshots, DNFs and exclusions;
    UI saved-record/assessment/retired-study smoke checks pass. Full regression
    793 passed,108 warnings,583.03s; all24 new checks pass. The interim native
    test/store-lock conflict is retained in its log; the final serial run
    passes. All41 sealed005 evidence files and13 archived inputs verify;
    current model/feature hashes match, database schema and every row in all
    six tables match the preserved backup. Both live studies and final sources
    are archived with checksums in data/audit/forward_forecasts/006. Active
    study20261026_baseline_v2 starts eligible recording October27. No new training, research
    winner, production change, automation or market-parity declaration.

61. **Confirmation feasibility and fixed baseline pilot007** (2026-10-26).
    Added a reproducible, exclusive variance-planning tool and publication-only
    pilot readiness checker. No training or model selection. Frozen estimates
    use retained independent predictions on inspected 2025Q3/2025Q4/2026Q1 and
    the unpromoted correction on inspected 2026Q3, with pooled/quarter and fixed
    freshness sensitivity. Preserve runner weighting and race/date dependence;
    four required gates target 80% joint power under the normal approximation.
    If truly equal in mean with archived variance, estimates are 1,215,235
    independent races versus 2,244 correction races; correction observed-mean
    scenario 3,093. Independent historical mean outside margins has no
    high-power parity sample. No assertion about the current refit's scores or
    transfer from Betfair LTP to Matchbook. Registered a baseline companion
    pilot before October 27: 84 calendar days, 1,000 eligible published pairs,
    60 distinct race dates, then seven days for labels. Outcome-free stopping,
    complete last-day inclusion, immutable plan/source anchor, no early parity
    claim or promotion. Earliest possible readout January 26, 2027, conditional
    on counts. UI shows progress 0/1000 and 0/60. All 62 focused checks pass in
    25.12s, including27 new tests; UI progress and saved 9-runner display pass.
    Existing793-test full baseline remains the prior regression receipt; it
    was not rerun for these isolated additions. Evidence under
    data/audit/confirmation/007_feasibility. Packages001–006 and all original
    forecasts stay immutable; no scheduled collection or external messages.

62. **Bounded manual timed recording008 complete** (2026-10-26). Added optional
    --at-window and UI Record in eligible window.
    Compute native probabilities first, then bounded wait, fresh exact-card
    recheck and current market read. Target opening+2seconds; at most600seconds
    waiting, sleep steps at most1second, monotonic/UTC divergence and deadline
    guards. Timed eligible pairs require intent and receipt evidence; actual
    publication/field/history/price rules unchanged. New baseline_v3, same
    production20261026_140950_205296 and original pilot policy/startOctober27.
    Prior v1/v2 kept, original records and plans not changed. New v3 action and
    retired v2 nine-runner table pass UI checks. Live20:30Southwell six-runner
    attempt inferred by20:22:13, waited138.54seconds, published20:24:33.448130
    within window. Correctly rejected model pair after odds changed on three
    runners; preserved fresh market vector, exact replay error0. Date also
    precedes registered start. UI shows market-only, model unavailable and
    pair count0. All prior006/007 sealed evidence and original study files
    verify; original model and all forecast/pilot contract fields match.
    Initial35focused checks pass. Final full suite833passed,108warnings,490.45s,
    including13new timing/recorder tests and27planning/readout tests from007.
    Single-field native replay from the refreshed card changes none of149
    selected inputs; reconstructed win probabilities differ by0. Diagnostic
    only, no general odds-invariance claim, original rejected record remains
    model-unavailable. Paired scores remain conditional on unchanged cards.
    Broader native odds-dependency verification is the next coverage task
    before considering a new input policy. Active studybaseline_v3 retains
    the original October27pilot conditions. Source/runtime/recording snapshots
    and final checksum verification are in008_timing. Quick status passes;
    feature code/parameters/data generation unchanged, no new deep data scan.
    No scheduled collector, orders, model
    fitting, promotion, data refresh or forecasting-lift claim.

63. **Actual model prediction work009 complete** (2026-10-26). User redirected
    work from infrastructure to forecasting. Registered and trained149-input
    baseline versus176-input pace/setup candidate, six native fits on three
    later historical quarters. Temporal feature audit passed all27additions;
    no FE rebuild or serving change. All9,356races/85,863runners scored. Race
    LL1.87204489→1.87310002, Brier.08754346→.08752490, binary loss worsens;
    LL improves only one quarter. Primary interval crosses0, candidate rejected.
    No favourable partial/subset selection. Market assessment8,942fields:
    baseline/candidate/market LL1.86879634/1.86965468/1.74928256. Saved models
    replay64shuffled fields per model within1e−12;15focused checks pass.
    Single worst underestimation exceeds net loss deterioration, motivating
    separate training-only tail/calibration research rather than dropping an
    outcome. Retained all artifacts, registrations and source/input hashes.
    About82.7minutes of fitting/evaluation. Production remains unchanged.

64. **Direct calibration experiment010 complete** (2026-10-26). Reused all six
    saved009 models, preserving trees/maps/features. Fit one coefficient on
    Jul1–Sep24 OOS forecasts, validate Oct1–Dec22, freeze before Jan–Mar and
    Apr–May scoring.149-input model selects82.867%Platt; pace diagnostic selects
    zero under the earlier Brier constraint. Primary2,591races: winnerLL
    1.81799380→1.81818510, Brier.09050170→.09046597; primary confidence interval
    crosses0. Transfer2,439races with fixedDecember trees: LL1.90437727→1.90275709,
    also uncertain. No near-zero winning forecast in either later period;
    no claim that these scores validate rare-tail protection. Severe individual
    underestimates improve, but no primary lift or market parity. Exact native,
    retained009 and overlay probability replays have zero difference. All18
    focused calibration/scoring tests pass; independent read-only verification
    validates19sealed files, source archive, models/features and market manifests.
    225.33seconds without tree fitting. Research-only module/script/overlays;
    production20261026_140950_205296, all009artifacts and activebaseline_v3 study
    stay unchanged. See MODEL_CALIBRATION_010.md and model-010logs. Every period
    is inspected retrospective development, not fresh confirmation. Continue
    actual prediction work; a narrow positive tail smoother is a distinct
    hypothesis, and calibration alone has not closed the market gap.

65. **Narrow tail smoothing011 complete** (2026-10-26). New research-only
    relative-floor method below1%, capped at25%Platt; seven registered strengths,
    earlierQ3fit/Q4validation with existing proper-score guards, fixed009trees
    and010saved probability stages.149learns0.1 but Q4LLworsens0.00002891;
    176fails strict fittingBrierguard. Both select zero, later forecasts exactly
    unchanged. Separate post-result diagnostic records earlier fitLLminimum
    without Brier veto,0.25both. Actual known Naas pace forecast rises1e-9→
    0.000971833 (validation example, not fresh evidence).149Q1deltaLL−0.00005136,
    uncertain, transfer+0.00002678;176Q1+0.00012877 with positive interval,
    transfergainuncertain. Tiny Brier increases alone are not material damage,
    but removing the veto still gives no reliable overall gain. All32focused
    tests pass, including14new checks. Verify011 validates11primary+4diagnostic
    artifacts, source archives/CRC, source/input/model/study hashes and exact
    replay0. All001–010artifacts, production20261026_140950_205296 and activev3
    study unchanged. No new tree fitting/data refresh. Primary run8.88seconds
    using cached010forecasts; sessions95555/99147 collected exit0. Main result
    and secondary diagnostic separately frozen, no retrospective promotion or
    rescue of009. Docs MODEL_TAIL_SMOOTHING_011.md, ledger30, progress andREADME.
    Close this floor/blend line; next original-plan target is history-depth/
    missing-history/estimate-uncertainty information after prior-artifact review.

66. **History-family012 completed across PC shutdown** (2026-10-27). Trained
    three165-input models, reusing exact verified009149-input references.
    Initial17-column/166-input attempt stopped before saving/scoring after an
    outcome-free audit found duplicate availability flags. All records kept;
    compact16-column registration frozen before fits. Seven native temporal/
    live-card controls pass; initial NaN-card fixture failure preserved and
    corrected to existing numeric-zero serving contract. Q3/Q4LL improve;
    Q1worsens. Pooled9,356races LL1.87204489→1.87133338,95%delta crosses0.
    Brier/binary point gains; no advancement. Matched market gap closes only
    0.38%. External-form776-field diagnostic motivates separate013, no slice
    rescue. All fits finished/saved before shutdown; morning checks recovered
    completed results without retraining. Corrected stale initial running-status
    claim after checking saved files. Verify012 exit0:17evidence files and
    64shuffled fields per model, maxerror2.004e-14<1e-12; all baseline forecasts
    exact.48focused tests pass. Two additional baseline alias pairs recorded,
    no baseline/production changes. Core comparison45.39minutes; trees
    1986/1647/1977. Docs MODEL_HISTORY_012.md.

67. **Exploratory history-context ensemble013 complete** (2026-10-27). Reused
    saved149/165forecasts; two coefficients fit onQ3 and validatedQ4, no new
    trees. Frozen weights0.494305(no external form)/1.0(contains external form).
    PrimaryQ1LL1.81799380→1.81767994; transferApril–May1.90437727→1.90248792,
    both95%intervals cross0. Brier/binary point estimates improve, market remains
    ahead. Context lead includes inspected012Q1outcomes, so no fresh confirmation
    or promotion claim and no reversal of012. Fit/validation periods never
    pooled into test evidence. New transfer165inference uses sameDecember22
    artifact, baseline matches010exactly,8invalid fields excluded as before.
    All66focused tests pass(8new);9sealed files/source archive verify; overlay
    replay0 and64shuffled native transfer fields maxerror1.11e-16. Runtime39.50s;
    sessions54078/40612 collected exit0, no jobs remain. All001–012sealed work,
    production20261026_140950_205296 and activebaseline_v3 study unchanged.
    Ledger wording now distinguishes uncertain results from refutation and
    baseline representability from a guarantee of future performance. No further
    subgroup/weight searches on these same outcomes. Docs MODEL_HISTORY_BLEND_013.
