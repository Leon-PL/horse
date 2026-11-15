# First ensemble benchmark, 9 November 2026

This benchmark compares the current 129-input win recipe, the compact 063
recipe with eight rating gaps, and the full 063 recipe with 64 extra inputs.
Both 063 recipes use a 75% historical market target for the win stage. The
129-input recipe retains its 50% share. All three use the same accepted
source snapshot, feature store, fold owner, training capacity and complete
race population.

## Reconstructed inputs

`research/meta_learner/features_063.py` reconstructs the archived 063 inputs
from `26d1b6f:scripts/feature_structure_063.py`. The full recipe includes eight
rating gaps, twenty signal gaps, eight context-weighted form measures and
twenty-eight race context inputs. The preparation uses canonical runner keys,
accepted raw going descriptions, corrected beaten lengths and pre-race Elo.
Unobserved marks and weights stay missing. The historical Elo-to-pounds
coefficient uses earlier dates only, with its reference fixed at the start of
2023. Context form uses strictly earlier runs, including when multiple runs
fall on the same date. These changes make this a current-data reproduction.

The study uses a separate store under
`data/processed/modelling_feature_store_20261109`. Registration, verified
projected inputs, fold receipts and results live under
`data/research/meta_learner/first_ensemble_20261109`.
The driver is `python -m research.meta_learner.first_ensemble`. Its stages
are `prepare`, `screen`, `compare`, `text` and `leeway`. Use the study folder
as `--output` and supply the separate store with `--store-root` for preparation.
The preparation reads the resolved baseline recipe from the earlier native
registration and the existing performance/spread and text specifications.

## Comparison

1. Screen the three generalists at the existing 400-tree capacity on
   2024Q3–2025Q2. The paired date-bootstrap interval must favour the challenger
   and it must win at least three quarters. Prefer compact unless full clears
   the same gate against it.
2. On 2025Q3–2026Q2, compare the selected single recipe, its three-seed
   probability average, and its fixed-weight pool with performance/spread.
   Use 2025Q1 and 2025Q2 forecasts to initialise the pool, then expand using
   earlier out-of-sample forecasts. Score the original recipe over the same
   later races when an alternative was selected. The performance pool must
   beat both the single model and the repeat-fit average to clear the gate.
3. Compare four text configurations with and without legacy text columns and
   with and without the explicit text model. Remove derived copies of the
   legacy score too. If performance/spread clears the screening gate, apply the
   same text-column choice to both numeric heads. Report every configuration
   against the repeat-fit average as a common control.
4. Screen handicap leeway as the first new specialist. The learner predicts
   the margin-based performance left after subtracting the pre-race official
   mark at the weights. Restore the known offset for race probabilities.
   Score it only where every runner in the handicap has an observed mark and
   weight, and use the base forecast for other fields. Use the strongest
   eligible base from the preceding comparisons. When the repeat-fit average
   is the base, build its earlier forecasts and test the explicit text model
   on that average as well. These comparisons are exploratory because the
   same periods informed the base choice.

Seven-day purges apply to the fitted stages. These periods have informed
earlier research. A screen result is a shortlist for further evaluation.

## Generalist screen

The rebuilt store contains 758,304 runner rows through 6 November. The
verified development projection contains 700,639 runners across 74,427 races
through 3 June. The screen compares 12,947 matched races and 120,809 runners.

| Recipe | Race log loss | Change vs 129 inputs [95% interval] | Quarters better |
|---|---:|---:|---:|
| Original 129 | 1.849419 | — | — |
| Compact 063 | 1.844916 | −0.004503 [−0.006562, −0.002597] | 4/4 |
| Full 063 | **1.842696** | **−0.006723 [−0.009089, −0.004620]** | 4/4 |

Full 063 beats compact by −0.002220 [−0.003947, −0.000631] and improves all
four quarters. It advances under the registered rule. Its runner Brier score
improves from 0.085219 to 0.085063 against the 129-input recipe, while the
ten-band calibration error rises from 0.001362 to 0.001959. The recipe choice
and forecasts are recorded in `screen.json` and the fold cache.

## Later ensemble comparison

The four later quarters contain 11,905 matched races and 109,631 runners.
The final quarter ends on 3 June at the development cutoff.

| Forecast | Race log loss | Change vs single full 063 [95% interval] | Quarters better |
|---|---:|---:|---:|
| Original 129 | 1.830892 | +0.004909 [+0.002654, +0.007048] | 0/4 |
| Full 063, single fit | 1.825983 | — | — |
| Full 063, three-seed average | **1.824713** | **−0.001270 [−0.002129, −0.000426]** | 4/4 |
| Full 063 with performance/spread | 1.824841 | −0.001142 [−0.001646, −0.000598] | 4/4 |

The single full 063 recipe beats the original in all four later quarters.
The performance pool assigns its second component 11.4–12.2% weight. It
scores +0.000128 [−0.000704, +0.001005] relative to the three-seed average
and wins two quarters. The three-seed average sets the reference for the next
comparison. Its runner Brier score is 0.085767 versus 0.085973 for the
original. Ten-band calibration error is 0.003114 versus 0.002094. The matched
results and fold weights are in `compare_screen.json`.

## Text placement

The explicit Timeform model covers 11,893 of the 11,905 later races. The
ablation removes `rtv_text_logit`, `rtv_text_prob`, `rtv_verdict_rank` and the
two 063 field-relative copies of `rtv_text_logit`.

| Full 063 text configuration | Race log loss | Change vs legacy inputs [95% interval] | Quarters better |
|---|---:|---:|---:|
| Legacy inputs | **1.825983** | — | — |
| Legacy inputs and explicit text model | 1.826081 | +0.000098 [−0.000101, +0.000302] | 2/4 |
| No legacy inputs | 1.828472 | +0.002489 [+0.000603, +0.004698] | 0/4 |
| Explicit text model without legacy inputs | 1.828020 | +0.002037 [+0.000173, +0.004048] | 0/4 |

The explicit model improves the ablated generalist by −0.000452 [−0.001050,
+0.000078], with three quarters better. Its pool weight rises from 1.2–4.1%
alongside the legacy inputs to 7.1–9.0% after removal. The results favour
keeping the legacy inputs and suggest overlap with the explicit model. The
four configurations and common-control comparisons are in
`text_comparison_screen.json`.

## Handicap leeway and the repeat-fit reference

The three-seed full 063 average retained the legacy inputs and served as the
reference. Both specialists were pooled separately with that forecast.

| Forecast | Race log loss | Change vs three-seed average [95% interval] | Quarters better |
|---|---:|---:|---:|
| Three-seed average | **1.824713** | — | — |
| Average with explicit text | 1.824798 | +0.000085 [−0.000060, +0.000220] | 1/4 |
| Average with handicap leeway | 1.824745 | +0.000032 [−0.000107, +0.000166] | 2/4 |

Handicap leeway covers 7,619 races (64.0%). Within those fields the change is
+0.000049 [−0.000168, +0.000259]. Its weight falls from 5.3% to 2.2% over
the scored quarters. The text model's weight falls from 3.0% to under 0.1%.
The blend fits and supported-field results are in `leeway_screen.json`.

## Screening decision

Advance full 063 with its legacy text inputs to the native-capacity comparison.
Carry forward the original 129-input control, a single full 063 fit and the
three-seed full 063 average. Keep performance/spread, explicit text and handicap
leeway in research at the tested settings. Record the gain and calibration of
each native fit along with its fitting and scoring cost, then lock the selected
recipe and evaluate it on fresh decision-time forecasts.

The study compares recipes at 400 trees. Its 063 recipes change both input
columns and the training target, so the comparison estimates their combined
effect. The retrospective folds inform the shortlist. Serving qualification
also needs the reconstructed inputs in the canonical feature pipeline and
verification of training and prediction parity.

## Native-capacity quick check

The follow-up used the first completed native control fold, 2025Q3, and
fitted a single full 063 model on its exact training and evaluation splits.
Both fits used four threads. Their training and calibration cutoffs were
24 June 2025. The comparison covers 3,535 races and 32,980 runners.

| Forecast | Race log loss | Runner Brier | Calibration error | Trees | Fit | Batch forecast |
|---|---:|---:|---:|---:|---:|---:|
| Original 129 | 1.868388 | 0.085657 | 0.003900 | 3,380 | 25.5 min | 23.8 s |
| Full 063, single fit | **1.861449** | **0.085491** | **0.003091** | 3,734 | 36.1 min | 37.2 s |

The paired change is −0.006938 [−0.010451, −0.003126], which clears the
registered pilot rule. Batch forecast time starts with prepared inputs and
includes field validation. The three-seed comparison remains the four-quarter
400-tree screen above.

The single full 063 fit, seed 42 with the legacy text inputs, is locked as
the prospective challenger against the original 129-input control. The lock
specifies four calendar quarters with a common purged refit before each
quarter. It begins after decision-time capture and canonical input and
prediction parity are ready. It uses forecasts from five minutes before the
first published scheduled start. The `native_quick_2025Q3` subdirectory
contains the pilot registration, fit receipt, report and
`prospective_candidate.json` with the exact recipes and protocol.

## Verification

The focused modelling, layout and chronology suite passed 34 tests. A smoke
check also verified six-quarter pooling, paired scoring, missing-expert fallback
and the handicap-leeway fit on small complete fields. The completed forecasts
were checked for matching digests, training cutoffs, complete-field coverage,
and exact fallback to the repeat-fit reference where either expert was missing.
