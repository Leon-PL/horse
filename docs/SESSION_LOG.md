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
