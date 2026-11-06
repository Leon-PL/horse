# Prediction improvement programme — 27 October 2026

**Historical programme, 27 October 2026.** The requested sequence and scope below
record that phase of work. Current operation is in the [README](../../README.md);
subsequent outcomes are indexed in [implementation progress](../MARKET_PERFORMANCE_PROGRESS.md).

Objective: reduce out-of-sample probability error and the gap to the market.
Maintain both an independent forecast and a forecast allowed to use prices.
The user authorises changes to data, features, models and production defaults.

## Order of work

1. **Repair result margins, then measure the effect.** Sporting Life's
   `finish_distance` is the gap to the preceding finisher. Our stored
   `lengths_behind` contains this gap, although downstream code treats it as
   distance behind the winner. In 75,302 of 80,135 races it decreases as
   finishing position increases. This corrupts margin Elo, close-finish form
   and margin-based speed proxies. Preserve the original gap; derive an
   explicit cumulative margin. Handle missing gaps, dead heats, incomplete
   results and non-finishers without inventing distances. Rebuild histories
   before fitting; never feed changed features to an old model as a fair test.
2. **Audit the actual selected inputs.** Check current-result, future-result,
   future-append, input-order and current-price invariance. Quantify missing
   values, duplicate aliases, identity continuity, age progression, field
   completeness and label validity on the real data. Distinguish an actual
   defect from an unlikely edge case; preserve an audit receipt.
3. **Repair market-teacher targets.** Partial BSP coverage currently produces
   a mixture of two separately normalised distributions whose field total
   can exceed one. Choose complete BSP, else complete SP, else no teacher,
   for the whole field. Reject ambiguous price joins. Measure actual impact.
4. **Compare one corrected independent model against the frozen reference.**
   Keep the existing 149 inputs, native learner, seed, recency weighting,
   calibration and training cutoffs. Evaluate all 2025Q3, 2025Q4 and 2026Q1
   fields, with the original inputs and predictions preserved. Primary score
   is race log loss against actual winners, with race/date uncertainty and
   Brier checks. All these dates are retrospective research, not fresh tests.
   A correction to data semantics does not itself prove a predictive gain.
5. **Test market teaching after its targets are valid.** One fixed 50/50
   winner/market target comparison is justified if no equivalent reliable
   earlier result is recovered. Inference remains independent of prices.
   Also retain the same-time market as the honest market-informed baseline;
   returning market probabilities is not evidence of independent skill.
6. **Fill demonstrated information gaps.** Existing winning-time data has
   near-complete coverage, but the active 'speed' inputs use margins. Recover
   the earlier time-based experiment before spending more effort: lack of
   residual betting edge does not establish lack of value to an independent
   model. Public official performance figures and external career histories
   are possible later inputs, subject to identity, coverage and publication
   dates. Today's revised ratings must not be inserted into past forecasts.
   A pilot has now verified missing French performances on public Sporting
   Life horse profiles. A fixed cohort of 2,193 older Flat horses has been
   cached separately (43,592 dated records). Seven date-strict history predictors are implemented
   in `src/public_history_features.py`, with identity, duplicate, non-runner,
   double-counting and temporal tests. These remain outside the active model
   until coverage and a paired prediction comparison justify adding them.
7. **Adopt demonstrated improvements together.** Version the features and
   model together, preserve the old production run, verify saved-model replay
   and live-card parity, and document the measured remaining market gap.
   Do not leave production silently mixing old models and corrected histories.
8. **Then consider learner and computation changes.** A direct race-level
   probability objective is a distinct modelling question from the previously
   tested post-fit softmax calibration. Test it once on the repaired inputs,
   with the same chronological controls. Remove duplicate allowlist aliases
   and skip unneeded feature families only after verifying their effect on
   predictions: feature subsampling means deleting a duplicate can change
   fitting. Keep a final, previously unexamined period for confirmation after
   selecting a recipe on the development data.

## Evaluation discipline

- Freeze the recipe before reading candidate outcomes; finish all three
  quarters. Save each model and field prediction so a shutdown loses no
  completed fit.
- Reuse reference predictions only after identity/label and native replay
  checks. Compare the same complete fields and exclude the same invalid
  races. Keep market comparisons on identical covered fields.
- Report point changes and uncertainty. A non-significant gain is uncertain,
  not a proven gain or proof that the idea cannot work.
- Keep earlier negative-result archives intact. Do not spend another cycle
  on pace/history blends or unchanged public-data families already tested.
- Correctness repairs can proceed without waiting for a long forward study;
  forecast changes require explicit evidence and feature/model compatibility.

## Public sources checked

- Sporting Life example result: https://www.sportinglife.com/racing/results/2020-09-15/yarmouth/588358/british-ebf-premier-fillies-handicap
- BHA distance definitions: https://www.britishhorseracing.com/about/faqs/
- BHA ratings and downloadable current figures: https://www.britishhorseracing.com/regulation/official-ratings/ratings-database/
- BHA explanation of retrospective figure revisions: https://www.britishhorseracing.com/regulation/adjusting-handicap-rating/

The source result shows Dusk a neck behind the winner and Oh It's Saucepot a
head behind Dusk. The third horse's total is approximately 0.5 lengths,
whereas the project stores 0.2 as its distance behind the winner.

## Completed results and active comparisons

014 completed all three quarters. Log loss improved 1.872045 -> 1.867804,
Brier .08754346 -> .08741422 on 9,356 races. All quarters improved; pooled
race- and date-cluster confidence intervals exclude zero. The same-field
market gap closed by about3.47%; substantial further improvement is needed.
See MODEL_MARGIN_REPAIR_014.md for methods, uncertainty and remaining tail errors.

A second confirmed data defect: finish_pos_label was discarded on SQLite
write/export. This hid45,097 position-zero runners from result histories.
The completed015 rebuild conservatively reconstructs45,005 DNFs in complete
settled fields;92 incomplete-field outcomes remain unknown. Previous-start
counts change on141,607 later rows, by as much as17starts. Database persistence,
legacy reconstruction and live pending-card resets are repaired and tested.

015 compares the corrected histories plus two completion inputs with the
original149-input model and completed014 result. 016 is the one fixed50/50
market-teacher test promised above, using those same151 inputs. All3quarters
are required for both; the final quarter also validates the place head.
Recipes and source inputs are frozen before fitting. Neither includes the
public-history prototype. Neither is automatically promoted on a single-quarter
result. Adoption must also improve on the best completed014 model.


## Faster feature exploration (user direction,27October2026)

The original allthreequarter requirement remains the historical protocol for
completed014–016. It is not the default cost for every new feature idea.
Audit source semantics and validity first; screen promising families in one
chronological window with matched cheaper baseline/candidate models (initial
budget:two-year training rows,500trees,learning rate.025,win head only). The
underlying historical features still use all available earlier starts. Preserve
purging/calibration. Cache a matching cheap baseline. Interpret these as screens,
then use broader confirmation only on shortlisted changes.

017 was already running a full native first fold when the user raised compute
cost. Preserve that completed model/prediction checkpoint and stop its later
queued quarters. Its amendment was recorded before the first result existed.
The original frozen source/manifest remain unchanged; one period cannot pass
the original allthreequarter advancement gate. Use scripts/run_one_feature_screen.py
for later single registered folds rather than invoking an all-period runner.
The selected016 full-history production refit continues independently.
