# Pace and race-shape model comparison009

**Decision: retain the current model.** Adding the 27 existing pace/setup
features did not establish a reliable improvement in win-probability quality.
All six models finished training and were evaluated on later races.

| On 9,356 races / 85,863 runners; lower is better | Current 149-input recipe | Pace-enhanced 176-input recipe |
|---|---:|---:|
| Race log loss, primary | 1.87204489 | 1.87310002 |
| Runner Brier | 0.08754346 | 0.08752490 |
| Binary log loss | 0.30088334 | 0.30098498 |
| Winner accuracy | 29.03% | 29.25% |

Candidate minus baseline race log loss is **+0.00105513**, with a 95% paired
interval **[−0.00190667,+0.00513894]**. Date-cluster sensitivity agrees. The
small Brier improvement is also uncertain: its one-sided 95% upper paired
difference remains positive. Higher winner accuracy and lower ECE do not
override the registered probability-score criteria.

| Historical test period | Baseline race LL | Candidate race LL | Difference |
|---|---:|---:|---:|
| 2025Q3 | 1.92063259 | 1.92032464 | −0.00030795 |
| 2025Q4 | 1.86228139 | 1.86445858 | +0.00217719 |
| 2026Q1 | 1.81799380 | 1.81950698 | +0.00151317 |

The candidate improves primary loss in only one of three quarters. It fails
the predeclared advancement rule and is not promoted. This closes this
unchanged whole-family addition under the tested training recipe.

## What was trained

Both variants use the same native LightGBM training, training-only calibration,
180-day recency weighting, complete fields, 2,000-tree capacity limit and seed.
Six fits ran sequentially with two worker threads; measured fitting/evaluation
time was 82.7 minutes. Baseline and candidate were both retrained using only
history preceding each test period. No place model or betting strategy was
fitted. Current-race prices are excluded from both predictors.

The added family describes historical run style, likely leader/closer roles,
field pace pressure and course leader bias. It differs from the 206 ATR columns
previously tested: none of these 27 was in that earlier feature candidate.
Before training, the actual feature functions passed date-prefix, future-data,
current-day-result and row-permutation checks, with positive/negative controls.
Runners with missing pace history remain in the evaluation.

The quarters are already inspected retrospective development periods. This
is a controlled later-race comparison, not fresh forward confirmation.
There was one fixed feature addition and no test-driven subset or parameter
search. Production parameters, feature generation and live studies are unchanged.

## Market comparison

On 8,942 identical complete market fields, baseline race log loss is 1.86879634,
candidate 1.86965468 and market 1.74928256. The candidate widens the point-estimate
market gap slightly. Both independent models remain behind the market.

The predeclared fresh-price subset retains 7,719 races. Candidate race log loss
is slightly lower there (1.86759818 vs 1.86806567), but remains far above the
market's 1.75239937. This secondary diagnostic does not overturn the full-sample
advancement rule. The reference is historical T−5 Betfair BASIC last-traded
price, conditional on identical complete fields; it is not an executable-price
or betting-profit comparison.

## What the errors suggest next

An exploratory decomposition retains all 9,356 races. Total additional candidate
log loss is 9.8718; its single largest worsening contributes 13.4261. The
candidate improves individual race loss on 4,711 races and worsens it on 4,645.
Both models assign probabilities near 10⁻⁹ to some eventual winners.

This does not justify deleting an outlier or declaring the candidate better.
It identifies a useful next modelling question: whether training-only
calibration and probability regularization can avoid extreme underestimation
without damaging other forecasts. Any such method must be selected on earlier
training/validation predictions and tested as a separate comparison. These
post-result diagnostics are not an untouched validation result.

## Verification

All six saved models replay shuffled complete fields (64 per period/model) to
within 1e−12 of their recorded predictions; maximum errors are near floating-point
precision. The retained baseline also reproduces the earlier trial to a maximum
difference of 1.7e−12 over 85,863 runners. An initial historical-refit diagnostic
used an unnecessarily strict 1e−12 threshold; its failure log is retained. Saved
artifact replay retains the original 1e−12 tolerance; the separate historical
refit comparison records its actual error and 1e−10 tolerance.

Fifteen relevant scoring/feature-benchmark checks pass. No new full-suite run
is claimed for this isolated research script/metadata change. Evidence, fitted
models, predictions, registration and input/source hashes are retained under
`data/audit/model_improvement/009_pace_family/`.
