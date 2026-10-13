# Session log — long autonomous session from 2026-10-13

Running state for the session driven by [NEXT_AGENT_BRIEF.md](NEXT_AGENT_BRIEF.md).
Newest entries at the bottom. Measurements go in
[NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md); this file only tracks where the
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
30. **Gate 4 has fired.** Roadmap drafted in `docs/ROADMAP.md`.
31. Running: `wf_ranking_compare.py --last-folds 8` (gate 2) and the clean
    production retrain (`train.py --promote`).

**Next:** ranking result -> roadmap's last line -> live end-to-end check on
the new model -> final status check -> report.
