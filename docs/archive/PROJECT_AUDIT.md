# Project audit: probability quality and pipeline integrity

**Historical audit checkpoint, 25 October 2026.** Dated status notes below retain
their original context. Current operating instructions are in the [README](../../README.md),
with later evidence in [implementation progress](../MARKET_PERFORMANCE_PROGRESS.md).

**27October2026 status:** this report describes the25October checkpoint.
The installed model is now `20261027_165656_616818`; current results and open
work are in [market-performance progress](../MARKET_PERFORMANCE_PROGRESS.md) and
[the018 feature review](../research/FEATURE_REVIEW_018.md). The production reference and
test count below are retained as historical evidence for that checkpoint.

Updated 2026-10-25. The requested objective is reliable, calibrated win
probabilities. This report covers the audit begun on 2026-10-11 and the final
integration review. Work from concurrent sessions is identified where relevant;
the detailed research history remains in [NEGATIVE_RESULTS.md](../NEGATIVE_RESULTS.md)
and [SESSION_LOG.md](SESSION_LOG.md).

Production now uses **`20261025_193016_331639`**, after it passed the fixed
shared-race comparison and saved-artifact checks. On 3,471 recent races,
runner Brier improved **0.32%** and race log loss improved **1.04%** relative
to the previous production model. The current full regression suite passed
**686 tests**. These are retrospective results; future outcomes are still
needed for independent confirmation.

## Findings and repairs

| Area | Failure and resulting repair |
|---|---|
| Historical features | Draw and pedigree statistics could include current-race or same-day results. Histories now respect date boundaries. Future-wide imputations of odds, ratings, weight, draw and speed were removed; unknown values remain unknown or use available race information. |
| Feature selection | Outcome and settlement columns could enter a model through an allowlist. Exclusions now take precedence. An explicitly missing, empty or unmatched allowlist fails instead of silently selecting every feature. The default allowlist is versioned in `feature_sets/win.txt`. |
| Calibration | Refitting could pass predictions through a stale isotonic map before fitting its replacement. Calibration state is reset; fitting and inference share the same transformation. Race-temperature calibration is available as an experimental alternative and is saved with each model. Legacy artifacts retain their original interpretation. |
| Race grouping | Batch prediction could normalize interleaved race rows together. Predictions now preserve race membership; tests cover ordering and probability sums. |
| Evaluation settings | Walk-forward evaluation could ignore training configuration or mutate global parameters. Frameworks, seeds, calibration, market blending and strategy settings are now forwarded explicitly; fast-fold overrides are local. |
| Place evaluation | Live predictions constrained place marginals to the field's places paid, while historical reports used unadjusted marginals. Holdout, OOF diagnostics and fold predictions now share the serving adjustment; a deliberately miscalibrated case verifies parity. |
| Ranking reports | Equal-score races were omitted, letting uncertainty change the scored population. They now count, with expected credit under a uniform tie-break. Partial ties receive the same treatment, independent of row order. |
| Missing prices | Top-pick simulations could settle a runner at an invalid price. Such picks are skipped without substituting a different horse. Initial losses now count toward maximum drawdown. |
| Collection | Failed or truncated races could be treated as complete, with a daily scrape cap silently losing races. Persistent manifests retain retries, completed inserts commit atomically, and recent dates are revisited. Empty/abandoned listings remain pending when the source cannot distinguish them from failure. |
| Feature storage | An interruption could leave rows, historical state and metadata from different generations. All five store files are staged, flushed, journalled and committed under a process lock; interrupted commits recover before use. Metadata stamping now shares that lock and uses atomic replacement. |
| Provenance and serving | Code/data changes and model promotion could leave stale cached predictions. Signatures cover feature dependencies and historical input values. Serving uses the promoted model, checks feature compatibility, and invalidates caches on promotion. The promoted artifact cannot be pruned or deleted accidentally. |
| Run registration | Concurrent training jobs could copy another job's global model into their run. New training saves its own predictor directly into an exclusively allocated run directory. Metadata is published after the artifacts; incomplete runs are excluded from pruning. Provenance is captured before fitting. |
| Status reporting | A failed check or unknown model provenance could still produce a successful “ready” verdict. These cases now block readiness. A modest data-age warning remains distinct from failed verification. |
| Source verification | The deep status check fingerprinted the CSV while claiming to check the database. It now uses current database contents through the canonical CSV parsing contract, detecting repairs without treating TEXT/integer ID representations as different data. |
| Final fitting | Early stopping chose the tree count but its smaller training subset was also used for the saved estimator. Both models now refit all eligible training rows at the selected capacity. |
| Model recency | A percentage of six years of rows created a holdout longer than a year. New CLI/app runs use a configurable 90-day calendar holdout, with the purge preserved. The legacy percentage option remains explicit. |

Concurrent sessions also removed substantial unused feature/research code,
corrected scrape-time `sex` leakage, repaired string/integer horse-ID mismatches
in live history joins, and fixed the exclusion of non-finishers from training
and evaluation. These changes are documented in the ledger, especially entries
13, 15 and 18. The live system is two classifiers, for win and place, rather
than the obsolete ensemble described in earlier documentation.

## What the measurements establish

The clean calibration benchmark used 149 selected features from the fully
rebuilt store. Each fold separated model fitting, early stopping, calibration
and evaluation by date, with seven-day gaps. It retained complete single-winner
races, including non-finishers as losers. Three evaluation windows covered
9,416 races and 85,729 runners from 2026-01-16 through 2026-10-12.

The first two windows selected **raw normalized probabilities** by their
combined Brier score. Race temperature had been the candidate suggested by the
earlier, partially repaired snapshot; that earlier result was diagnostic only.

Latest window, 2026-07-15 through 2026-10-12, 3,336 races / 31,112 runners:

| Metric, lower is better | Platt + isotonic | Race temperature | Raw normalized |
|---|---:|---:|---:|
| Runner Brier score | 0.0860683 | 0.0860982 | 0.0861122 |
| Binary log loss | 0.2972910 | 0.2968655 | 0.2969281 |
| Race log loss | 1.8836909 | 1.8794162 | 1.8799026 |
| ECE, ten fixed bins | 0.0031304 | 0.0022447 | 0.0030036 |

Temperature lowered race log loss in all three windows. In the latest window,
the paired whole-race bootstrap difference was -0.004275, with a 95% interval
of **[-0.015066, +0.001717]**. Its runner Brier score was slightly worse.
The race-averaged Brier difference was +0.0000912, interval
[-0.0000265, +0.0002176]; this averages races equally, unlike the table's runner
average. Neither interval establishes a latest-period improvement.

**Decision: retain Platt + isotonic as the default.** The simpler alternatives
are useful candidates, but these results do not justify claiming a confirmed
gain or changing the promoted model's calibration. ECE alone cannot establish
forecast quality: a weak model can be well calibrated while failing to
distinguish likely winners.

The report's original “confirmation” label meant separated within that
benchmark run. It does **not** mean untouched across the project: other
experiments and production reports have already inspected these dates, and
the second development fold overlapped the repository's reserved period
beginning 2026-06-11. Treat the existing comparison as retrospective evidence.
The benchmark now excludes the reserved period by default and requires an
explicit option to inspect it. A future claim needs a frozen method and new
outcomes that have not already informed decisions.

Local evidence:

- `data/audit/20261011/clean_calibration_benchmark.json` and `.log` — full rebuild comparison.
- `data/audit/20261011/calibration_benchmark.json` — earlier partial repair; not production validation.
- `data/audit/20261011/final-rebuild.log` — completed transactional rebuild.
- `data/audit/20261011/field_integrity_review.json` — final field-metadata inspection.

These large/local artifacts live under gitignored `data/`; the benchmark and
regression tests are versioned.

## Rebuilt store and verified model on 15 October

The protected rebuild completed with **755,706 runners**, through
**2026-10-12**, feature signature **`221fcd4db0fc5631`** and data signature
**`d3970f5407e44d88`**. It completed all nine chunks and committed the complete
generation. No provenance was manually restamped to bypass rebuilding.

The subsequent session trained and promoted **`20261014_105017`**, retaining
non-finishers and the default calibration. The final review verified that this
run matches the rebuilt store; it did not repeat that training or replace the
production pointer unnecessarily. On 2026-10-15 the deep status check found
matching feature code, data and model, with a three-day results-age warning.

Its saved holdout report gives Brier **0.086612**, binary log loss **0.2992**
and top-pick accuracy **29.38%**. The corresponding market report gives
**0.082544**, **0.2816** and **34.53%**. These are descriptive historical
diagnostics, not new untouched confirmation or a live betting guarantee.
The saved ranking report omitted two uniform-probability races; the corrected
ranking code counts them. Brier and log loss already included all runners.

The DNF repair is particularly important. Ledger 18 evaluates the old and new
models on the same 14,331-race field: race log loss improves modestly from
1.8920 to 1.8896. More importantly, simply removing the known non-finishers
from evaluation had turned an old model's negative value-bet return into a
positive one. Scores calculated on that filtered population are invalid for
live predictions. The apparent large betting edge disappeared when all actual
runners were included.

## Field reconciliation and retrain on 25 October

The original mismatch investigation was extended using the existing local
Betfair public archive. A repair required a unique race/event join, unique
normalized horse keys, an entire reference field contained in the result
rows, exact agreement with the reported field size, and matching winners.
Ambiguity caused abstention. Genuine non-finishers stayed whenever the
independent source recorded them as starters; finish position alone was not
used to infer participation.

**3,297 races were repaired, removing 6,445 extra rows.** The corrected results
contain **749,261 rows across 79,664 races** through 2026-10-12. Original rows
for every affected race were backed up before committing one transaction;
fingerprints prevented applying a plan to changed data. All repaired races
retain their recorded field size, which the independent archive corroborates.
The full CSV was then exported atomically. There are **22 unresolved fields**.

Evidence is in `data/audit/20261025/field_reconciliation/`: `plan.json`,
`original_races.json`, `verification.json`, and `unresolved_races.csv`.
`scripts/reconcile_result_fields.py` defaults to a dry run and abstains from
conflicting evidence. Ingestion now rejects both underfull and overfull
responses and can replace a complete race, including obsolete stored rows,
without losing unrelated enrichment. Failures preserve the old transaction
and CSV snapshot. Database handles close explicitly, with one idle handle
retained during sync to avoid checkpointing after every race.

The full rebuild completed with **749,261 rows**, feature signature
**`221fcd4db0fc5631`** and data signature **`f2a120a477c117a1`**. The deep
status check verified the store against the current database. The results
remain dated through **2026-10-12**, thirteen days before this review.

Candidate **`20261025_193016_331639`** is saved with 149 features and the
default Platt + isotonic calibration, seed 42, full history and no market
blend. Both selected estimators refit **629,311 eligible training rows**,
through **2026-07-08**: 1,897 win trees and 2,989 place trees. Its 90-day
holdout runs **2026-07-15 through 2026-10-12**, after the seven-day purge.
The comparison rule was fixed before fitting: evaluate both saved models on
identical complete single-winner fields, no earlier than either model's
holdout start; do not replace the baseline if Brier, binary log loss, or
race log loss worsens. Report paired race-level uncertainty and treat this
as retrospective diagnostics, because these historical dates were inspected
earlier.

Both saved models were replayed on **3,471 races / 32,357 runners**, including
**947 non-finishers**, from 2026-07-15 through 2026-10-12. Three invalid or
ambiguous fields (33 rows) were excluded in their entirety. All remaining
fields had valid closing prices, so the reference below uses exactly the
same population. Lower scores are better:

| Metric | Previous production | New production | Closing-price reference |
|---|---:|---:|---:|
| Runner Brier | 0.0863669 | **0.0860888** | 0.0820915 |
| Binary log loss | 0.2990108 | **0.2967461** | 0.2804954 |
| Race log loss | 1.8970388 | **1.8773931** | 1.7517307 |
| ECE, ten fixed bins | 0.0014782 | **0.0011182** | 0.0047485 |

The paired whole-race bootstrap gives a race log-loss difference of
**-0.0196457**, 95% interval **[-0.0336752, -0.0088206]**. This supports an
improvement on this historical period. The race-averaged Brier difference
is **-0.0002816**, interval **[-0.0006474, +0.0000851]**; the smaller Brier
gain remains uncertain. This interval weights races equally, while the
table's runner Brier weights each runner equally. Binary log loss improves
0.76% by its point estimate.

**Decision: promote the new candidate.** All three pre-specified probability
scores improved, and serialization/serving checks passed. The production
pointer changed from `20261014_105017` to `20261025_193016_331639` on
2026-10-25; the previous run remains available for rollback. No calibration
method change or market blend was introduced. This comparison measures the
combined data and training repairs; it does not isolate their individual
contributions.

The closing-price reference remains stronger on proper probability scores.
Its odds are historical starting prices, not evidence of prices available
earlier. Better marginal calibration alone does not make the new model more
informative than that reference, and no betting edge is established.

Evidence in `data/audit/20261025/`: `probability_run_comparison.json` and
`.parquet`, `candidate_comparison_context.json`, `promotion_decision.json`,
and `candidate_artifact_verification.json`.

## Remaining limits and next work

1. **Resolve the remaining 22 fields from primary result records.** The first
   audit found 3,319 mismatched races and differences at place-payment bands
   in 721 races. The independent comparison established that most mismatches
   were extra result rows, rather than a count that should simply be increased.
   The 3,297 repairs above address those cases. Conflicting counts, winners or
   incomplete reference coverage remain listed explicitly; do not manufacture
   labels or delete genuine starters to make the counts agree. The clean run
   comparison excludes entire fields that still fail validation. The trainer
   does not yet apply that evaluator's strict complete-field/single-winner
   filter; changing the treatment of ambiguous history and genuine deadheats
   needs a separate chronological comparison.
2. **Do not reuse an inspected holdout as fresh confirmation.** Freeze the
   next candidate and its decision rule before collecting the next evaluation
   period. The repository's older “golden” label no longer establishes that a
   period is unseen for every hypothesis.
3. **Match the final training procedure when judging deployment candidates.**
   The isolated calibration benchmark and production trainer use different
   history windows and weighting. Production calibration uses predictions
   from disposable training-slab models; OOF metrics can also reuse the rows
   used to fit calibration. Evaluate the final artifact on later complete
   races, not on calibrator fitting scores.
4. **Direct file readers bypass store transactions.** Consumers using
   `FeatureStore` receive locking and recovery. Ad hoc research that reads its
   parquet directly must run against a stationary store or an explicit
   snapshot. A rebuild needs disk space for both generations.
5. **Older ingestion dates are not automatically re-audited by the new retry
   manifest.** Recent dates are revisited; legacy history requires a targeted
   completeness audit. Source ambiguity keeps abandoned/empty races retryable.
6. **Keep price and probability metrics distinct.** Some legacy fields named
   `clv` hold model probability times price, an expected-return multiple.
   Actual price-to-closing-price analysis lives in `src/clv.py`. Later research
   also corrected early-price comparisons for non-runner reductions; use the
   latest ledger conclusions, not earlier positive summaries.
7. **Do not prune from a marginal screen alone.** Race constants cancel under
   common-logit softmax shifts, but not generally under binary sigmoid followed
   by race normalization; they may change confidence. Tree interactions also
   need direct testing. The broad cancellation claim in ledger 2g has been
   qualified without changing its measured screening figures.

## Verification evidence

- Final full regression run on the current code: **686 passed**, 16 warnings,
  in 8 minutes 31 seconds. Log: `data/audit/20261025/final-full-suite.log`.
- This includes actual LightGBM, XGBoost and CatBoost fits. Each framework
  verifies that both final models refit all eligible rows, exclude holdout
  rows, and reproduce the place probabilities reported for evaluation.
  Native test diagnostics use a temporary directory.
- Comparison checks reject negative, fractional and infinite finish positions
  across an entire field, retain genuine non-finishers, and verify shared
  population alignment. Earlier targeted runs are superseded by the final
  suite; their counts are not added to it.
- Real corrected database and CSV: **749,261 processed rows each**, identical
  data fingerprint **`f2a120a477c117a1`**, recorded in
  `data/audit/20261025/snapshot_contract.json`.
- Reloading the candidate reproduced its saved holdout win and place
  probabilities on **32 shuffled race fields / 290 runners**. Maximum errors
  were **1.11e-16** for win and **2.22e-16** for place, and win probabilities
  summed to one per race. Evidence: `candidate_artifact_verification.json`.
- After promotion, replay through the production loader passed on eight
  heldout fields / 88 runners, with maximum win-probability error 1.11e-16
  and matching feature provenance. Evidence: `production-smoke.json`.
- Final deep status passed after promotion: the database, current feature
  code, rebuilt data and new production artifact agree. Both the deep and
  quick checks returned **ready**, retaining the explicit 13-day data-age
  warning. Evidence: `final-dataset-status.log` and
  `final-production-quick-status.log`.
- Syntax compilation and whitespace checks passed.

Exact local source archives and SHA-256 manifests are retained in
`data/audit/20261025/candidate_source_snapshot.zip` and
`comparison_source_snapshot.zip`. These include the uncommitted audit changes;
the recorded Git revision alone does not reproduce this working tree.
`runtime_versions.json` records the Python and package versions used.

The repairs produce a more trustworthy prediction and evaluation pipeline
and a modest measured improvement over the prior production model. There is
no demonstrated reliable betting edge, and the fundamental model still
trails the closing-price reference on proper probability scores.
