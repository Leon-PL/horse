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
12. Store rebuild running (started ~13:20, 7 chunks).

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
