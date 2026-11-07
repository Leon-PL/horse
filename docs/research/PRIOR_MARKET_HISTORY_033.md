# Prior market assessments 033

**Installed and verified, 29 October 2026.** Run `20261029_085433_011356`
uses 154 inputs and the confirmed source repairs. It improves historical win
and place probability scores, closing about 4.48% of the previous log-loss
gap on matched market fields. The market still leads. Full readiness and
native live checks pass. These are inspected development results; future
confirmation remains necessary.

The original151 selected predictors did not directly represent how strongly a
horse was fancied in earlier races. The selected154-input update adds dated
earlier market assessments and opposition strength. Market teaching already
informs the training target. These additional inputs passed a single native
win-model check and full win/place qualification. The matching current-data
model is now installed in the normal forecasting pipeline.

## Source and feature audit

On the saved951 complete established-handicap fields/7910 runners,7872 runners
(99.52%) have complete source-SP prices in all three previous local starts.
The remaining38 have two. This is a source-coverage finding, not a prediction
score. The proposed same-discipline and365-day restrictions reduce coverage.

Only complete historical SP books from valid settled fields are normalised.
Features use three strictly earlier same-scale starts, separately for Turf,
AW, Hurdle, Chase and Bumper. Current-day prices and outcomes cannot enter the
pre-race inputs. A missing complete book remains unknown; it advances the
three-start window. Nonstarters and pending cards do not. Inputs are:

- Mean past `log(field_size * normalised_SP_probability)`, relative to today's field.
- Mean past opponents' observed race-day official ratings, relative to today's field.
- Number of known price observations in the three-start window.

Opponent quality excludes the horse's own mark and requires three rated
rivals. Imputed marks do not count. Market opinion and official ratings stay
in separate units; no coefficient translating probabilities into rating points
was fitted or assumed. A preliminary Elo-scaled formulation was revised during
the input audit, before any fit or score: actual stored Elo ratings have a
much narrower spread than such a conversion would suggest. Its unfit artifacts
are preserved in `pre_elo_anchor_review/`.

Eight targeted tests cover current/future/same-day independence, complete-book
handling, imputed marks, separate scales, expiry, live seeding and unchanged
input state. On all678085 pre-April2026 rows, full-history and prefix/seeded
calculations agree. Price-history inputs exist for546348 rows and opposition
inputs for430608; missing values are preserved. Preparation took80.22s.

No current-race odds are required for forecasts. This is independent of
current prices, while intentionally using prices from completed earlier races.
Historical SP is not claimed to be an executable or timed exchange quote.

## Cheap first screen

Train24 September2023–23 September2025; test October–December2025. Same native
labels, teacher, weights and settings as030;400 trees and simple common race
normalisation. One new win fit, with the exact saved151-input baseline reused.
Total84.09s; no place or calibration fit, tuning or quarter sweep.

| Population | Baseline log loss | Proposal log loss | Baseline Brier | Proposal Brier |
| --- | ---: | ---: | ---: | ---: |
| All3233 fields/30081 runners |1.847941|1.845882|0.085482|0.085408|
| Established handicaps1071 fields/9777 runners |1.987342|1.983896|0.091748|0.091682|

Both primary scores improve in both populations. Overall log-loss delta
is-0.002059, date-cluster95% interval[-0.006409,+0.002707]. Target delta
is-0.003446, interval[-0.010763,+0.004206]. These uncertain point gains justify
a normal-capacity check; they do not establish a general improvement or market
parity. Dates were already inspected development, not a fresh holdout.

The relative past-price input ranks5th of154 by training split gain (3.38% of
total); opponent quality ranks36th. This confirms the learner uses the added
information. Training importance is not evidence of an out-of-sample gain.
The saved `quick_screen/feature_importance.csv` contains the full breakdown.

## Single native check

Completed29 October2026 at00:34UTC after the032 corrected history rebuild.
The candidate uses the ordinary corrected151 baseline inputs plus these three.
Only the native win head was fitted through22 December2025, then evaluated
on the same Q1development fields against saved029 native forecasts. No baseline
or place model was fitted for this check. Capacity selection, fitting and
calibration follow the normal win pipeline. Three estimator fits took1700.68s;
the selected final win estimator has3384 trees.

The advance rule, fixed before native scores: overall and established-handicap
race log loss lower, with neither runner Brier worse. Report the uncertainty;
do not retune if it fails. No automatic promotion follows. The candidate also
contains the mandatory52-event metadata corrections, so any native difference
is a combined update, not a pure attribution to the price family.

The advance rule passed without retuning:

| Q1 development population | Baseline log loss | Updated log loss | Baseline Brier | Updated Brier |
| --- | ---: | ---: | ---: | ---: |
| All2591 fields/22824 runners |1.802287|1.798224|0.09003383|0.08995795|
| Established handicaps993 fields/8385 runners |1.912089|1.905518|0.09773430|0.09756486|

The overall log-loss difference is-0.004062, with date-cluster95% interval
[-0.007837,-0.000379]. For established handicaps it is-0.006571, interval
[-0.012622,-0.000472]. Brier improvements have intervals crossing zero;
established-handicap binned calibration error increases slightly despite the
better primary probability scores. These are previously inspected development
dates, so the intervals do not account for the wider research selection.

On2469 identical fields with the market reference, baseline log loss is
1.789123, the update1.784530 and the market1.686613. The update closes about
4.48% of the previous log-loss gap as a point estimate. The market remains
clearly ahead. This is historical market comparison, not an executable-price
or betting-profit result. Removing all current prices on100 checked races
changes updated win probabilities by exactly zero.

The951 established-handicap fields with a market reference improve from
1.904043 to1.896616, versus the market's1.809162. The update reduces winner
log loss in521 of those951 fields. Among the16 previously selected worst
error cases, nine move closer to the market: four of eight missed winners
and five of eight overrated losers. The latter group's mean probability
falls from47.80% to46.62%; missed winners' mean rises only from9.68% to9.77%.
These descriptive cases show the severe errors remain. They were selected
earlier and are not a separate validation sample. The saved
`native_check/error_change_diagnostics.json` and corresponding CSVs use only
the completed forecasts; no extra model was fitted.

A further read-only tree-contribution replay covers all122 runners in those
16 fields and reproduces the recorded case forecasts within2.3e-16. Raw-score
contributions sum to the raw model output within1.6e-14. Files
`native_check/fixed_error_case_attributions.csv` and `.json` explain the raw
tree score, not probability-point changes or causal feature importance.

Two cases identify a specific information limitation for a later screen:
Arklow Lad had six Turf starts before his first all-weather race, and Brown
Boots had six Hurdle starts before his first Chase. Both therefore have zero
same-scale prior-price observations despite being locally established.
Their new price input is correctly missing under the selected definition;
it is not evidence that either horse is globally classified as unraced.
Testing transferable prior-price information with an explicit discipline/
surface change flag is a distinct proposal. Keep opposition rating scales
separate and examine the source history first. No such proposal was fitted
or added during this installation. Without Love had improved from twelfth
to third in her two previous races, while several overrated losers were
strongly fancied previously too. Historical prices alone cannot resolve all
current changes.

Across all993 established-handicap fields,377 of8385 runners (4.50%) have
zero same-scale365-day price support;273 fields contain at least one such
runner. Among the market-matched fields, the687 fields with every runner
covered improve log loss1.883431->1.873733; the264 fields containing a zero-
support runner improve1.957683->1.956164. This is a descriptive coverage
split, not evidence that pooling histories would cause an improvement.
Missing books and expired history can also produce zero support. See
`native_check/established_price_history_coverage.json` before designing a
transfer feature or treating a new-discipline horse as globally unfamiliar.

Exact estimator checkpoints include numeric training arguments, targets,
weights, feature order and complete learner semantics. If the family advances,
canonical qualification should reuse these win components when the full
matrix is identical. Place components and final current-data components remain
separate necessary fits. The original031/032 proposals remain unselected.

## Normal pipeline integration

The three inputs now run in ordinary feature engineering and are selected by
`feature_sets/win.txt`; future normal training needs no research switch.
The helper retains its original `src/prior_market_research.py` filename (renamed
`src/prior_market_features.py` on 2026-11-07), but
is now an actual dependency of the normal pipeline and its fingerprint.
Eight helper checks and eight integration/signature checks pass. The latter
include live-state seeding and genuinely missing card measurements.

The full754497-row history was extended without repeating unchanged feature
work. All1184 existing columns and their nulls are exactly preserved; only
the three independent inputs and their bounded history state were appended.
Every678085 pre-April row agrees with the earlier projection at model input
precision. A fresh ordinary calculation on2579 actual runners agrees on all
154 selected inputs. The stored history ends27 October2026 and has feature
signature `11bf1232d575eea6`.

Full qualification reuses the exact three native win fits after checking
learner source, settings, runtime, feature order, every numeric training
argument, target and weight. Place estimators and the final current-data
estimators require their own fits. No historical win refit or quarter sweep
is needed.

Qualification completed29 October at07:35UTC as run
`20261029_073522_043953`. It reproduces all native win probabilities exactly
(maximum difference0). Place Brier improves from0.17461220 to approximately
0.17425539, difference-0.00035680. The paired one-sided95% upper bound is
-0.00014626 by race and-0.00019308 by date. This is the same inspected Q1
development sample and does not provide fresh confirmation.

The ordinary current-data win/place update completed at 08:54 UTC and installed
run `20261029_085433_011356`. It fitted 667,873 eligible runners in 71,477 fields
through 27 October 2026, selecting 3,992 win / 2,991 place trees. The native
training run took 4,669.9 seconds. Its model fingerprint is
`a97c602aceddbc79da06a72d9f861f91a5e41bf6d686def224868d75f627fc15`.
The old 029 model remains recoverable; current source, history and weights
are aligned.

Yesterday's 38 source races / 345 runners were then collected and appended
through the ordinary carried-state feature path. History now contains 754,842
runners through **28 October**, data signature `554fd1046c28a932`; feature
signature remains `11bf1232d575eea6`. The original training-prefix signature
is `267f5d1582796e06`. No parameter refit followed this history refresh. All
345 appended prior-market input rows exactly match a calculation seeded from
the preserved 27 October state at model input precision.

Deep readiness passed at 09:11 GMT, including DB-content comparison and model
compatibility after the append. Native live verification passed at 09:14 GMT:
115 recent fields / 1,108 runners and Bath's 12-runner 12:42 card. Probability
ranges, win/place totals and floors, runner identities and single/batch equality
all pass; maximum single/batch difference is zero. This is operational evidence,
not an accuracy estimate on the model's recent training races or today's card.

Fresh pinned study
`20261029T091505Z_prior_market_033_20261029_085433_011356` is eligible from
30 October. It preserves the installed artifact and source, with no saved
observations or scheduled recorder. Older studies retain their original models.

Evidence: `data/audit/prior_market_feasibility_033/report.json`,
`feature_audit.json`, `quick_screen/report.json`, `native_check/report.json`,
`backfill/commit.json`, `integration_tests.log`, `installation/`,
`daily_history_refresh.json`, `daily_prior_market_parity.json`, `readiness.json`
and `final_status.json`. Live evidence is in `data/audit/live_upgrade_033/`.

## Training cost and scope

Each031/032/033 proposal first used one400-tree candidate fit and the same
saved baseline. The first two stopped there. Only033 advanced to normal
capacity. There was no hyperparameter search or quarter sweep.

| Stage | New native estimator fits | Reuse |
| --- | ---: | --- |
| Historical033 win check |3|Saved029 baseline predictions; no baseline fit|
| Full win/place qualification |3 place|All three historical033 win estimators|
| Final current-data update |6|Completed qualification; no repeated historical fit|

The three fits per head are capacity selection, a refit on all eligible rows,
and an earlier temporary estimator supplying disjoint later races for
calibration. These are components of one recipe, rather than separate
research candidates. Daily forecasts and valid history appends do not refit
these estimators. Current-data checkpoints preserve completed components if
the PC stops. Their presence does not mean a partly fitted model is installed.
