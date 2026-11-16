# Forecast benchmark protocol

Implementation begun 2026-10-25, following `MARKET_PERFORMANCE_PLAN.md`.
Primary outcome: calibrated, discriminating win probabilities. Profitability
and the ability to execute a quoted price require separate evidence.

## Market and field contract

- Horizon: five minutes before the **first published scheduled start** in the
  local archive. Later schedule changes cannot move this decision backwards.
- Market universe: GB/IE horse racing WIN, open and not in play at decision,
  at least two active runners, subsequently settled with exactly one winner.
- Replay messages in timestamp order. Separate interleaved markets by market
  ID before updating definitions or prices. Reject malformed/incomplete data.
- The forecast includes every active runner known at the decision, including
  horses withdrawn later. Use only prices and names published by the cutoff.
  Carry forward change-only LTP updates; retain update and observation times.
- Final definitions supply outcomes only. Later removals get a separate flag;
  they are neither quietly discarded nor treated as genuine non-finishers.
- Reference probabilities are reciprocal LTP normalised across the entire
  decision field. LTP is a forecast reference, not an executable back price.
- Write a new immutable research snapshot and source hashes. Do not modify
  canonical BSP/timing files, results, features, or production pointers.

The current feature store describes final starters. Paired comparison therefore
requires identical complete fields at prediction and settlement, a unique
date/track/off-time/horse match and matching outcomes. Exclude the **whole
race** on a discrepancy and report coverage. This is a retrospective
comparison conditional on no field change, not an operational as-issued
racecard backtest. Historical off-time keys use UTC; local clock time is
retained separately. Do not choose whichever clock gives a better result.
Other late racecard changes, such as jockey or going updates, also lack a
complete historical publication-time log. Date-strict features do not prove
that every racecard value was already available at T-5. Forward confirmation
must save the actual racecard and feature inputs at the decision time.

## Training and evaluation contract

`RacePredictor.fit_as_of` fits through an explicit settled date. It uses the
existing native estimators, recency weights, chronological early stopping,
full refit and training-only calibration. Research can omit place fits.
Its artifact retains cutoff/calibration metadata and **no claimed holdout
scores**. A deployment artifact must link separately validated update rules;
it cannot inherit scores measured on data now included in its fitting rows.
The archived slab field `base_training_last_date` is the eligible boundary
before the optional inner early-stopping split; it is a conservative upper
bound for the disposable model's fitted rows. Calibration dates are recorded
separately. The final estimator's training cutoff is exact.

`python -m src.modelling benchmark` uses the shared purged quarterly fold helper
(the older `scripts/benchmark_forecasts.py` is at tag `pre-cleanup-2026-11`).
For each predeclared experiment:

1. Register the feature/data/source/price hashes, parameters, cutoffs, allowed
   corrections and stop rule before fitting. Cache fits only within that
   exact manifest. Freeze a source archive including uncommitted files.
2. Fit the historical independent recipe before the correction-calibration
   period, with the project purge gap. Predict later calibration races without
   fitting their outcomes.
3. Fit market corrections on the earlier 65% of calibration dates. Leave a
   purge gap before later calibration validation. Require at least 50 races
   in both portions.
4. M0 is the same-time market. M1 fits a single market temperature. M2 uses
   the existing log-probability combiner, bounded to model coefficient [0,1]
   and market coefficient [.5,1.5], with shrinkage .25/.5/1 towards M0.
   Choose on **calibration validation** only: lower race log loss with Brier
   no worse than M0, otherwise use M0. Do not refit the correction on test data.
5. Refit the independent recipe through the purged test boundary, using all
   eligible history. Evaluate all four frozen forecasts on the same fields.
   Test outcomes cannot select correction weights or fallback behaviour.

Race log loss is primary. Report runner-weighted Brier/binary log loss,
race-averaged Brier, ECE and tie-aware winner accuracy. Use paired whole-race
bootstrap intervals and date-cluster sensitivity. Slices by race type,
handicap, field size, minimum experience and market favourite strength are
diagnostics, not retrospective selection rules.

The reserved holdout and purge are excluded by default. An explicit override
labels the run retrospective. Existing dates have already been examined by
the project and cannot become universally untouched again. No correction
representation, training score or validation fallback guarantees future
non-worsening performance. The market baseline itself is market-informed;
matching it does not establish independent forecasting skill.

## First registered experiment

`data/audit/market_performance/001_t5_2026Q3/manifest.json`:
149 existing features, hard labels, current LightGBM recipe, at most 2,000
trees, seed 42, two worker threads. Calibration forecasts come from a fit
through 2026-04-25; corrections use May/June data. The final independent fit
ends 2026-06-24 and the test period is July–September 2026. Dates are explicitly
retrospective. One recipe and the declared market ladder only; no repeated
search against that quarter. Production is unchanged by this experiment.

Archive coverage actually ends on 2026-10-10, despite the requested upper
bound of 2026-10-12. Keep missing dates separate from field exclusions. The
supplementary `freshness_sensitivity_registration.json` fixes one diagnostic
before test scoring: every runner's LTP changed within 300 seconds and the
market was observed within 90 seconds. Apply the same frozen forecasts;
do not choose or refit a correction on that subset.

Next independent feature-family comparisons must select on earlier
development folds. Do not use the above quarter repeatedly to choose new
features. Fresh confirmation begins after the recipe/update rule is locked.

The first comparison is complete in `MARKET_COMPARISON_001.md`. Its original
source archive and report are retained. Experience diagnostics were corrected
after scoring to use `prior_form_runs`; see `parity_assessment.json` for the
corrected slices. This change does not alter any forecast or overall score.
Current diagnostic code therefore differs from the frozen original manifest;
use its source archive for exact reproduction, rather than changing the
manifest in place to resume under different code.
