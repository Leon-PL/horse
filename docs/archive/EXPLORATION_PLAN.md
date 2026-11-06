# Exploration plan: five unconventional ideas (2026-10-14)

**Historical experiment plan, 14 October 2026.** Its proposed sequence is retained
for interpreting the research, rather than as a current work queue. Start with
the [operating guide](../../README.md) and [inventory](../PROJECT_INVENTORY.md).

*For an autonomous agent session. The operator chose these five from a longer
list after the project's standard route - beating the closing price with a
form model built from public data - was measured and closed (ACTION_PLAN 7,
gates 2 and 4; ROADMAP). Each idea below tries a different route: use the
market as an input, change when a bet is struck, or find information the
market does not see first.*

Read before starting: [CLAUDE.md](../../CLAUDE.md),
[NEXT_AGENT_BRIEF.md](NEXT_AGENT_BRIEF.md), and
[NEGATIVE_RESULTS.md](../NEGATIVE_RESULTS.md) sections 0, 2, 5, 8, 13-19. Section
19 is the most recent template for how a test is pre-registered and reported.

## 0. Ground rules

These are the project's rules plus what has gone wrong in the past. None is
optional.

1. **Check the instrument first.** `.venv/Scripts/python.exe scripts/project_status.py`
   must say `ready`. If it does not, fix that before anything else.
2. **Never commit on `main`.** Work on `checkpoint/eval-harness-and-guards` or
   a branch off it. End commit messages with the attribution lines the
   harness gives you.
3. **Look for a concurrent session before editing.** Run `git status` and
   list running `python` processes. If a feature-store build
   (`build_feature_store_batched`) is running, do not touch any file in
   `_FE_SOURCE_FILES` (`src/featured_dataset.py`): the build aborts if the
   feature signature changes mid-build. Do research in `scripts/`, which is
   not hashed, or in a separate `git worktree`.
4. **Pre-register every test in the ledger and commit it before running it**:
   hypothesis, data, rule, matched baseline, pass bar. Then record the result
   under it, pass or fail. Exploratory looks are allowed, but only on the
   discovery split defined for each idea, and they must be labelled as such.
5. **Development data only, then the golden hold-out once.** Races from
   2026-06-11 are the hold-out (`config.GOLDEN_HOLDOUT_START`;
   `src.wf_eval.development_mask` / `holdout_mask`). A hypothesis gets one
   hold-out test, after it passes on development data, never before.
6. **Measure honestly.**
   - Race-clustered confidence intervals: `bootstrap_roi_ci`,
     `bootstrap_median_ci` in `src/wf_eval.py`.
   - CLV from `src.clv.summarise_clv` / `wf_eval.clv_summary`: median,
     geometric mean, beat-close rate. Never the mean ratio alone.
   - Compare a rule against the same universe it bets in.
   - Correct for multiple tests (Bonferroni or BH).
   - Settle with the rules in `src/bet_settlement.py` (2% exchange
     commission).
7. **The tripwire.** High ROI with CLV at or below 1.0 has meant a leak or
   artefact four times out of four (ledgers 1, 3, 15, 18). If you see it,
   stop and find the cause before reporting anything.
8. **No real money.** Never place a bet or an exchange order. Paper only,
   unless the operator explicitly authorises it in writing for a specific,
   limited purpose.
9. **Respect data sources.** Read terms of service before scraping. Racing
   Post forbids it: do not scrape it. Go slowly (seconds between requests),
   cache raw responses under `data/`, make scrapers resumable, and back off on
   HTTP 429. The BHA API, for example, locks you out for 20+ minutes after
   ~490 quick requests.
10. **Free data unless the operator says otherwise.** Any spend (Betfair live
    app key £299, ADVANCED/PRO data, paid sale or weather data) is an
    operator decision: stop and ask.
11. **Write down every measurement** in NEGATIVE_RESULTS.md, and keep a
    running line per step in SESSION_LOG.md.

## 1. What is on disk

| data | where | coverage and caveats |
|---|---|---|
| Results (SP in `odds`, `finish_position` 0 = non-finisher) | `data/races.db`, table `results` | 2020-09-15 to now, GB/IRE. `form` is the **pre-race** form string (an empty string means a debutant). `horse_runs`/`horse_wins`/`horse_places` are scrape-time career totals: never use them raw (FE overwrites them with prior counts). |
| Betfair BSP + late prices (`bsp`, `ltp_300s`, `ltp_60s`, `ltp_preoff`) | `data/processed/betfair_bsp.parquet` | 2020-2026-06-10. Join with `src.betfair.attach_bsp`, which raises on column clashes. |
| Early prices (`ltp_1100` = last trade at or before 11:00 UK, `ltp_3600s` = T-60) | `data/processed/betfair_bsp_early.parquet` | 2020-01-01 to 2026-06-10, 96.4% of runners. |
| Raw Betfair BASIC streams (1-minute LTP, no ladder, no volume) | `data/betfair/*.tar` | 2020-2026-06. New cut times go into `scripts/extract_betfair_bsp.py` (see `cut_1100`, `cut_3600`). |
| Betfair public BSP files (BSP, morning/pre-play traded volume, `morningwap`, pre-play max/min) | `data/processed/betfair_sp_public.parquet` (`scripts/fetch_betfair_sp_public.py`) | Includes the hold-out. `morningwap` averages the whole morning, so scoring a whole-morning signal at it is look-ahead (ledger 14). |
| Place BSP | `data/processed/betfair_place_bsp.parquet` | 2020-2026-06-10 |
| Feature store (1,161 columns) | `data/processed/feature_store/featured.parquet` | Read columns with pyarrow. Never hold it open while a build is committing. |
| BHA going reports (stalls, rail text) + per-race rail changes | `data/processed/bha_going.parquet`, `bha_races.parquet` (`src/bha_reports.py`) | GB only. The rail backfill was running on 2026-10-14. |
| Observed weather | `weather_*` columns in `results`, `data/weather_cache.db` | Open-Meteo **archive** (what happened), so it is look-ahead for any forecast test. |
| Course coordinates | `src/track_config.py` (`lat`, `lon`) | all 86 tracks |
| Pedigree (sire, dam, damsire by name) | `horse_pedigree` table; `src/pedigree_backfill.py` | 70k+ horses |
| Old Tattersalls sale scraper (4DCGI API, prices) | deleted; recover with `git show 10a1286^:src/tattersalls_sales_scraper.py` | ledger 5 |

Tools to reuse rather than rewrite:
- `src/wf_eval.py`: folds, bootstraps, the masks.
- `scripts/triage_features.py`: market-residual partial correlation.
- `scripts/microstructure_screen.py` (`load()`) and `scripts/microstructure_clv.py`: the morning-money frame and early-price protocols.
- `scripts/morning_money_segments.py`: the ledger 19 report format.
- `scripts/wf_market_offset.py` (`--extra`, `--only-prefix`) and `scripts/wf_feature_family_lift.py`: model-level tests.

## 2. Order, effort and stopping rules

| order | idea | new data needed | effort | can the hold-out test it? | stop when |
|---|---|---|---|---|---|
| 1 | **B. Execution timing** (operator's #3) | none | 1-2 days | only with Betfair BASIC files from 2026-06-11 (operator) | the gain over always-BSP, after a spread haircut, is under 0.5 CLV points or its interval includes 0 |
| 2 | **A. Tote exotics priced from BSP** (#1) | Tote dividends (scrape) | 3-6 days | yes: public BSP covers the hold-out | after takeout, no rule has a discovery-split ROI interval clear of 0, or validation fails |
| 3 | **C. Forecast ground changes, bet early** (#4) | archived forecasts; an evening-before price from the tars | 3-5 days | only with BASIC from 2026-06-11 | no continuation signal (partial vs the later move < 0.01, or not significant) |
| 4 | **D. Breeze-up times for debutants** (#5) | sale breeze times (source discovery first) | 3-6 days | yes: 2026 juvenile season, BSP market | fewer than ~1,500 matched debut runs, or triage partial < 0.01 |
| 5 | **E. Morning market-making, feasibility only** (#8) | none for phase 1 | 1-2 days | n/a (prospective) | required spread capture exceeds a plausible 1-2 ticks, or capacity is trivial |

Do them in this order unless the operator says otherwise. B and E reuse what
is already loaded; A has the best prior; C and D need data discovery first.
Time-box each. A failure, recorded honestly, is a complete result.

---

## 3. Idea A - Tote exotics priced from Betfair (operator's #1)

**Why it might work here.** This project cannot beat the closing price, but
it can *use* it. Betfair's starting price gives near-fair single-race win
probabilities for free (ledger 16: the market beats the model by five points
of top-1). Combination pools (exacta, trifecta, swinger, Placepot) are filled
largely by recreational money and priced by the pool itself. Bill Benter's
documented profits came from exotic pools priced off a sharp win estimate.
The hurdle is the takeout, typically far higher on exotics than on the
exchange's 2%.

**Step 0 - data source (stop if this fails).**
- Find historical UK/IRE Tote dividends for 2020-09 onwards: win, place,
  exacta, trifecta, swinger, and Placepot/Quadpot (with pool sizes if
  published). Candidates: Sporting Life results pages (`src/data_scraper.py`
  parses them but stores no dividends - inspect the raw page or JSON for a
  returns or Tote block), At The Races results pages (already scraped for
  sectionals with `curl_cffi`), and the Tote's own results pages. Not Racing
  Post.
- Record units (per £1 or per 50p), dead-heat and non-runner rules, whether a
  pool was commingled internationally, and the published deductions per pool
  type.
- Build `scripts/fetch_tote_dividends.py` (resumable, cached, slow) writing
  `data/processed/tote_dividends.parquet`, keyed to `race_id` through the
  existing merge keys (`src/utils.py`).

**Method.**
1. Win probabilities per runner from BSP, overround-normalised. Selection
   must use information available when the pool closes; BSP is only known at
   the off, so also run everything with `ltp_300s` as the selection price,
   which is the honest version.
2. Combination probabilities. Plain Harville
   (`p_i * p_j / (1 - p_i)`, and the trifecta analogue) is known to overrate
   favourites finishing second and third. Fit a Henery or Stern discount on
   the discovery split, using observed 2nd/3rd frequencies. Place BSP can
   help calibrate.
3. **Every rule's ROI can be computed exactly from winners' dividends.** For a
   rule that selects combination set `S_r` in race `r` using pre-race
   information only:

   ```
   ROI = sum_r D_r * 1{winning combo in S_r} / sum_r |S_r| - 1
   ```

   Only the winning combination's dividend is ever observed, and this
   formula needs nothing else.
4. Explore on the **discovery split only (2020-09 to 2023-12)**. Look at ROI
   by model-probability band, by structure (favourite-led vs outsider-led
   exactas, outsider in the second leg), by field size, by flat/jumps, by
   pool size, and at "value index" `D * p_model` against takeout.
5. Pre-register at most 3 rules in the ledger, then test them on the
   **validation split (2024-01 to 2026-06-10)**, Bonferroni-corrected. Only a
   validation pass earns the hold-out test (2026-06-11 onwards; BSP from the
   public files).

**Pass bar.** ROI above 0 after deductions, with a race-clustered interval
(Bonferroni-adjusted) excluding 0 on validation. More than half of quarters
positive. Then the hold-out, once.

**Traps.**
- Your own stake moves the dividend: the observed dividend already includes
  all money. Estimate pool sizes and cap stakes, or a wide-coverage rule will
  look better than it could ever be.
- Dividends are after deductions; do not subtract them twice.
- Handle dead heats, non-runners and reserves.
- Can a small UK bettor actually bet these pools at those terms? Minimums,
  rebates (large syndicates get them; you will not), commingling.
- The computer straight forecast (CSF) and tricast are bookmaker formulas
  based on SP, not pools. Report them separately; bookmakers pay them, but
  accounts get restricted.

**Deliverables.** The fetch script with tests for parsing and joins; a ledger
section; a one-paragraph verdict.

---

## 4. Idea B - Execution timing: when to strike, not what to back (operator's #3)

**Why it might work here.** Ledger 14 showed that morning money predicts the
later price move strongly (partial correlation +0.169 with the move from
morning to BSP), even though it predicts nothing about the result beyond BSP.
Ledger 19 found that when the money backs the 11:00 favourite, the median bet
beats BSP (1.0122). This does not need a selection edge. It is a cost
reduction for any bet already worth making: take the early price when the
money is coming in, wait for BSP when it is leaving.

**Data.** All on disk:
- `scripts/microstructure_screen.load()`: development period, complete
  markets, `excess_vol`, `morningwap`, BSP.
- `betfair_bsp_early.parquet`: `ltp_1100`, `ltp_3600s`.
- `betfair_bsp.parquet`: `ltp_300s`.

**Method.**
1. Decision points: 11:00 UK (races off at or after 11:30, as in
   `microstructure_clv._with_early_price`), T-60, T-5. Target:
   `y_t = log(price_t / BSP)`. Positive means striking at time `t` beat the
   close.
2. Features must be known at `t`. Excess morning volume is only known at
   about 11:00, so use it at 11:00 or later, never earlier. Also: price level,
   field size, race type, minutes to the off, the overround at `t` (from LTPs),
   and the price change since an earlier cut. Never use the result or BSP.
3. Fit `y_t` walk-forward (`wf_eval.expanding_period_folds`, quarterly,
   purged). Start with a linear model; use LightGBM only if it clearly beats
   linear out of sample.
4. Policy, fixed before scoring: strike at `t` if predicted `y_t` is above a
   threshold (pre-register, for example 0.02), otherwise take BSP.
5. Evaluate on three fixed universes:
   - every runner priced 1.5-50;
   - 11:00 favourites;
   - the morning-money picks of ledger 14.

   Metric: mean `log(price_taken / BSP)` against always-BSP (and against
   always-`t`), with race-clustered intervals. Also report the share struck
   early, and the morning volume available (median about £345 per runner:
   capacity).
6. **Haircut.** An LTP is a traded price, not a guaranteed back price.
   Re-score with a one-tick and a 1% haircut on every early strike. A result
   that survives only without the haircut is not a result.

**Pass bar.** A gain over always-BSP of at least 0.5 CLV points, with the
interval excluding 0, after the one-tick haircut, on the walk-forward. The
hold-out needs BASIC files from 2026-06-11, which only the operator can
download.

**Traps.**
- `morningwap` is never a decision price.
- The 11:00 cut is UK local time; race off-times in the database are UTC.
- BASIC rounds BSP to the LTP tick grid, so ties pile up at short prices.
  Read the geometric mean and the beat-close rate as well as the median
  (ledger 14).

---

## 5. Idea C - Forecast ground changes, bet early (operator's #4)

**Hypothesis.** When weather forecasts imply the going will change between an
early decision point (the evening before, or race-day morning) and the race,
early prices under-react for horses with strong ground preferences. The
market updates once the official going changes, which is a timing edge of the
same shape as ledger 14's.

**Step 0 - data (stop if any fails).**
1. **Forecasts as they were issued** before the decision time. Never analysis
   or observations: the `weather_*` columns and `src/weather.py` use the
   Open-Meteo archive, which records what happened and is look-ahead here.
   Candidates: Open-Meteo's historical-forecast and previous-runs APIs. Check
   which years they cover and what lead times they offer; you need forecasts
   issued at least 12-24 hours ahead, for each course's `lat`/`lon`.
2. **An evening-before price.** Add a cut (for example 18:00 UK the day
   before) to `scripts/extract_betfair_bsp.py` next to `cut_1100` and
   re-extract to a new column. Check coverage: some markets open late.
3. **A prior-only ground preference per horse.** Build it from market-relative
   performance on soft or heavy versus good-to-firm or firm, shrunk. The
   `cp_*` affinity machinery on branch `feature/course-profile`
   (`add_course_profile_features`) shows the pattern. The store's
   `horse_track_going_*` and `going_specialist_edge` are alternatives. Keep
   it prior-only (hard rule 1).
4. **The going at decision time** is not stored historically; forecast
   rainfall between decision and race stands in for "expected change". The
   BHA going text carries update timestamps for GB.

**Method.**
- Shift index: forecast rain (mm) from decision to off at the course, and for
  drying, forecast temperature and wind with no rain.
- Interaction: `x = preference * shift`, centred within each race.
- Continuation test (primary): does `x` predict `log(price_decision / BSP)`
  beyond the price level? Use triage-style partial correlation, race-clustered.
- Economic test: back the top-`x` runner in each race at the decision price
  if `x` is above a pre-registered threshold. Report in ledger 19's format.
- Turf races only; all-weather "going" barely moves.

**Pass bar.** Continuation partial at least 0.01 and significant. The
economic rule must clear the ACTION_PLAN 7 bar: median CLV above 1 with the
interval excluding 1, geometric mean above 1, ROI above 0. The hold-out needs
BASIC files from 2026-06-11.

---

## 6. Idea D - Breeze-up times for two-year-old debutants (operator's #5)

**Why.** Debutants are where the market's lead over the model is largest
(ledgers 2d and 8): the model squeezes them into a 14x probability range
where the market uses 54x. Breeze-up sales publish measured furlong times
(sometimes stride data) for the two-year-olds they sell, which is a measured
speed figure rather than a price.

**Prior: low.** Sale price was fully priced, even on debutants (ledger 5:
0.005), and buyers price the breeze. Run this cheaply and stop early.

**Step 0 - sources (stop if coverage is thin).**
- Which sales published official times for 2020-2026, in what form, and
  under what terms: Tattersalls Craven and Guineas breeze-ups, Goffs UK
  (Doncaster), Tattersalls Ireland (Ascot), Arqana, Goffs Ireland.
- The deleted Tattersalls scraper (section 1) shows how to reach that
  company's lot data.
- Lots are **unnamed at sale**. Match on sire + dam + foaling year
  (`horse_pedigree`; foaling year = race year - age). Report the match rate.

**Test.**
- Power first: how many matched debut runs exist? About 1,500 are needed to
  see a partial correlation near 0.03.
- Then triage on two-year-old debut runs that have a time: within-race
  partial correlation of the time (adjusted for the distance timed and, if
  published, the sale and ground) with `won - p(T-5)`.
- If it passes, test it in the walk-forward offset model
  (`wf_market_offset.py --extra file --only-prefix`).
- Hold-out: the 2026 juvenile season, with `won - p(BSP)` from the public
  files.

**Pass bar.** Partial at least 0.01 with BH q at most 0.05 on development
data; a non-zero offset-model gain; then the hold-out once.

---

## 7. Idea E - Morning market-making, feasibility only (operator's #8)

**Hypothesis.** Morning Betfair markets are thin and wide, so posting back
and lay offers around a fair price could earn the spread.

**The known obstacle.** Morning money is informed: it predicts results better
than the morning price does (ledger 14, T3). A market maker trades against
that money, so passive fills tend to land on the side that later proves
wrong. Following the money at 11:00 broke even against BSP (median CLV
1.0000, ledger 14), so taking the other side loses about that much *before*
any spread is earned.

**Phase 1 (free data, this session).**
- **Markout.** For the informed side (high excess morning volume) and the
  rest, the expected value at BSP of having filled the morning trade at
  `morningwap` or `ltp_1100`, per £ matched. A first-order estimate is fine;
  say clearly what BASIC cannot show (no ladder, no queue).
- **Spread needed.** The break-even spread, in ticks by price band (Betfair's
  tick ladder), that pays for adverse selection plus commission.
- **Capacity.** Morning matched volume per runner and per day (public files)
  bounds what could ever be earned.
- **Output.** A go/no-go memo: break-even spread versus a plausible 1-2
  ticks, capacity in £ per day, and what phase 2 would cost.

**Phase 2 (operator decision only).** A live app key, Stream API recording
of the order book for 4-8 weeks, and a paper simulation with queue position.
Never post a real order.

---

## 8. Reporting

For each idea, in order:
1. A ledger section: pre-registration (committed first), data notes, result
   tables in ledger 19's format, verdict.
2. One line per step in SESSION_LOG.md.
3. Scripts in `scripts/`, with unit tests in `tests/` for any reusable
   parser or join.
4. Nothing enters the feature pipeline (`src/feature_engineer.py`) unless it
   has passed. If something does pass, wiring it in triggers a store rebuild,
   retrain and promote, and `project_status.py` must be green afterwards.

Final message to the operator: one line per idea (passed / failed / stopped
at step 0, and why), what is blocked on them (BASIC files from 2026-06-11,
any paid access), and the recommended next step. A passed hold-out is the
only thing that justifies real money, and only by the operator's decision.

## 9. State when this was written

- Production run `20261014_105017` (non-finishers kept, ledger 18);
  `project_status` green; feature signature `221fcd4db0fc5631`.
- Branch `feature/course-profile` (stalls, rail, course types; ledger 17,
  all priced) waits to be merged. Rebuild the store once the BHA rail
  backfill finishes: `ls data/bha_going/races | wc -l` against about 8,736
  fixtures. Then run `scripts/fetch_bha_going.py --build`,
  `scripts/build_feature_store_batched.py`, retrain and promote. A trial
  merge conflicts only in NEGATIVE_RESULTS.md: keep both sections, 17 before
  18.
