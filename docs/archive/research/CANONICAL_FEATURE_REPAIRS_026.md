# One corrected training pipeline

The user's 28 October clarification makes verified correctness repairs part of
normal training, even when a retrospective comparison is slightly weaker.
Earlier experiments remain evidence; their old feature definitions are no
longer the default. The installed model must be replaced with a model fitted
on the corrected histories before it can use the new generation.

Both `train.py` and the training app call `src.training.train_run`. They use
the same `process_data` and feature functions. Rebuilds now use the bounded
history builder automatically, including when initiated by ordinary training.
No research switch is required to obtain the corrections below.

| Verified correction | Normal implementation |
|---|---|
| Going descriptions and explicit unknown conditions | `data_processor` → `feature_semantics.normalise_going` |
| Synthetic surfaces, including Allweather, Polytrack and Fibresand | Existing pedigree history uses the audited surface predicate |
| Invented movements from imputed marks or different rating scales | Observed-mark flags and compatible-scale comparisons |
| Unknown class treated as Class 0/Class 5 | Missing class and class-adjusted performance remain unknown |
| Impossible or imputed weights used as measured changes | Observation flags and valid-weight histories |
| Four-runner top-three baseline | Prior probability 0.75 |
| Stalls missing in Flat/bumpers | Draw interactions unavailable where no stall exists |
| Unknown results counted as actual starts | Observed-start counters |
| Long/missing-result histories approximated by a short averaging tail | Exact persisted exponential averages |
| Confirmed erroneous historical race kinds | Sixteen source-verified event corrections applied to whole fields, with date/course checks |
| GBP and EUR purses averaged together | Common GBP values, strictly earlier ECB observations, original evidence retained |
| Place floors increased the expected number placed | Bounds included inside the existing fixed-slot normalisation |
| Ordinary training admitted incomplete or ambiguous target fields | Shared whole-field validation in normal and deployment training; genuine non-finishers retained |

The historical race-kind and venue-currency reference files are ordinary
versioned project inputs under `reference/`. Training does not depend on a
research snapshot directory. Original purse amounts are excluded from model
inputs so raw mixed units cannot accidentally return in all-feature training.
Exchange-rate updates also invalidate affected live feature caches.

The additional coverage predictors, dated overseas-form predictors, a new
bumper indicator, alternative calibration and feature pruning remain research
designs. The selected input list stays at 151 for this correction batch. Their
future tests must use the corrected normal pipeline as the reference.

The place correction retains the existing assumption that every available
place is filled. Modelling races with too few finishers is a separate design
question. It is not silently changed in this batch.

Before the rebuild, 77 focused correctness checks and137 broader integration
checks passed. All 151 selected inputs passed prefix/future-data, same-day
outcome, permutation and price-change checks. Two further tests verify that
completed native learner fits are reused only with identical data, labels,
weights, calibration split, feature order and recipe.
The completed rebuild contains all 754,497 runners through 27 October 2026.
A further 38 training, deployment, artifact and restart checks passed after
applying the same whole-field requirement to ordinary training.

Execution: rebuild all historical features and affected ratings once; make
one chronological comparison on the existing 2026Q1 development fields; then
use the ordinary training core for a full win/place refit through the settled
cutoff. The previous model and histories are recoverable comparison artifacts.
The comparison reports gains or losses honestly and does not reinstate a
verified error because it helped a score. No reserved later-period scores or
new feature-design sweep are part of this phase.

Evidence and resumable checkpoints: `data/audit/canonical_repairs_026/`.

The completed Q1 comparison covers 2,591 races and 22,824 runners, with both
models fitted through 22 December 2025. The reference is the previously
installed recipe at its full capacity, including its original place model.

| Measure | Previous pipeline | Corrected pipeline |
|---|---:|---:|
| Race log loss | 1.803081 | 1.802182 |
| Win Brier score | 0.090093 | 0.090014 |
| Winner accuracy | 30.30% | 30.75% |
| Place Brier score | 0.1746 | 0.1746 |

The paired race-log-loss change is −0.000899, with a 95% interval of
−0.004433 to +0.002555. The small gain is uncertain. On the same 2,469 fields
with frozen market prices, log loss is 1.790390 → 1.788627 against the
market's 1.686613. That closes about 1.7% of the previous gap as a point
estimate in this familiar development period; the market remains ahead.

Winners assigned at most 1% remain three, including one below 0.1%. The worst
winner probability moves slightly lower, 0.0674% → 0.0630%. Losing runners
assigned at least 50% fall from 105 to 83, while winners assigned at most 5%
increase from 175 to 193. The tail comparison is mixed. These results do not
veto verified repairs or establish general improvement or market parity.

Historical validation run: `20261028_123247_415677`. The ordinary full refit
through 27 October was installed as `20261028_134020_942529`, with freshly
fitted win and place components on 667,873 runners and 71,477 races. Deep
readiness and native saved-artifact checks passed. The completion receipt
preserves its exact sources, inputs and results before the subsequent
teacher-data extension.

The final rebuild also exposed a source-coverage gap: the sectional cache has
232,138 runner records from 10 August 2023 through 20 June 2026, with no later
records. Consequently none of the 32,531 runners in the final July–October
batch matched a current-race sectional record. Existing features may still
use their older available form. The selected 151 inputs contain margin-based
performance figures rather than direct sectional metrics. The cache gap is
therefore relevant to possible future feature expansion. A single ordinary
public-page request returned a client challenge; newer data availability was
not confirmed and the cache stayed unchanged. Checking supported availability
precedes any proposed refresh. These inputs stay frozen for this correction
refit.
