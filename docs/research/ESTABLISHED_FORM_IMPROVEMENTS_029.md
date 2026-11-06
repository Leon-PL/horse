# Established-horse form improvements029

Comparison completed on28 October2026. The proposed extra feature family did
not improve probability scores and is not selected. The normal rebuild covers
754,497 runners, with1,184 stored columns and fresh/state parity over134,178
runners. The matching win/place update is now installed as
`20261028_212411_292221`, trained through27 October. Deep readiness passed at
21:28GMT: database, feature history, training records and installed model all
agree, with no blockers. Live operational verification passed at21:31GMT.
The batch is complete; all receipts are linked below.

## Corrections that apply regardless of the experiment

The028 Glicko repair is included. This batch also removes the old handicap
context calculation that mixed Flat, hurdle and chase marks. Palamon's chase
rating had been compared with an earlier Flat placing, giving a misleading
33-point rise. The new history separates discipline and Flat natural/synthetic
surfaces, uses observed ratings, and leaves an unavailable comparison missing.
Winning/placed handicap anchors require a handicap in the same scale; an
unrated latest win does not revive an older winning mark. Career-high context
retains genuine prior observed marks across intervening unrated starts.

A separate source error affects all nine runners in the21 April2021 Catterick
handicap622382. Sporting Life's historical data still publishes marks including
6 for Mellys Flyer; the [independent race result](https://www.racingpost.com/results/10/catterick/2021-04-21/780429/)
records64. The canonical processor now applies explicitly sourced values from
`reference/verified_runner_ratings.json`, guarded by race ID, date, course,
horse ID and name. It preserves the raw source evidence, never reconstructs
missing digits and never imports RPR/TS or current profile ratings.

The other unusually low marks were not automatically removed. For example,
the [Southwell result on15 October2020](https://www.racingpost.com/results/394/southwell-aw/2020-10-15/767304)
independently confirms Dark Side Prince's rating10. A numerical outlier alone
does not establish a data error.

The ordinary processor and feature stages contain these corrections without
a research switch. The installed029 weights match the rebuilt generation;
no stale-generation checks have been bypassed.

## Fixed feature proposal

One compact family adds17 new-to-the-model historical/context inputs and the
existing Hurdle indicator, increasing151 inputs to169. The learner currently
retains Chase but omits Hurdle; the additional indicator and bumper flag let it
distinguish all four racing disciplines explicitly.

The family describes:

- Current bumper, novice and maiden/beginners-chase context.
- Experience, win/place rates, recent field-size-normalised finishing form,
  non-finishes and recency in today's discipline.
- Observed same-scale handicap marks versus prior wins, placings, career high
  and the last five handicap starts, with sample support.
- Declared weight versus the last same-scale handicap win and handicap
  experience. Declared weights are not assumed to contain jockey claims.

These are descriptors, not newly invented official performance ratings.
No claim is made that a lower mark necessarily means a stronger chance.
The [BHA's handicapping explanation](https://www.britishhorseracing.com/regulation/guide-to-handicapping/)
distinguishes handicap ratings from performance figures and shows separate
turf, all-weather, hurdle and steeplechase marks. The
[race-type glossary](https://www.britishhorseracing.com/regulation/glossary-of-race-types/)
explains novice and bumper context.

The history stage excludes all current-day results, records genuine non-finishes
as starts, and never advances for pending cards or nonstarters. It carries
bounded three-start form/five-handicap-mark windows and independent cumulative
counts. It uses the same stateful function for full builds, append and live
prediction, and leaves the caller's seed unchanged.

## Source and mathematical validation

The first conservative bumper rule recognised only explicit NH Flat/bumper
titles. A dated-source comparison found581 missed performances across354
fields: this provider calls many Irish bumpers simply 'Flat Race'. The revised
rule agrees with all24,747 matched public performances in17,905 fields,
including1,104 bumper performances. Actual Hurdle/Chase codes and later actual
'Handicap'/'Stakes' title words take precedence over advertising a bumper.
No current profile age, career totals, ratings or results enter the feature.

The initial baseline attempt was stopped during its first fit when this source
check returned. No completed component, candidate fit, forecast or score was
produced. Both earlier input-only registrations remain preserved; the final
comparison is registered from the rebuilt normal store under
`data/audit/established_form_029/native_canonical/`. This was a data-validation
amendment before any comparison outcome, not a search over Q1 scores.

The first preparation from the rebuilt store then omitted finishing-margin
metadata required even by native win-only training. It stopped before any
component fitted. That input and its sealed registration/source snapshot remain
under `native_canonical/pre_target_metadata_fix/`. The corrected preparation
shares the ordinary training projection and requires both finishing-margin
columns explicitly. This changes the caller's metadata projection only;
features, targets and the registered statistical recipe are unchanged.
An exact comparison confirms all181 pre-existing columns remain unchanged over
all678,085 projected runners; the receipt is
`native_canonical/projection_amendment_parity.json`.

The final targeted suites pass45 checks covering new history/state, source
identity guards, ordinary processor integration, unavailable values,
discipline/surface transfers, same-day/future blindness and signature coverage.
The39 additional rating, store, reuse and live-parity checks also pass:84
distinct relevant checks in total.

## One coherent rebuild and comparison

The rebuild uses the ordinary memory-bounded builder and recomputes all feature
stages and their state. It recomputes Glicko across the full history. It avoids
repeating unchanged Elo/TrueSkill sweeps only after proving their definitions
unchanged, verifying every original processed value apart from the explicit
rating corrections, and comparing37 unchanged raw rating columns with a fresh
native prefix. No old Glicko state is reused. The prior generation is preserved
through recoverable hard links; all five store files commit together.

The registered comparison used the same corrected151-input baseline and
169-input candidate, identical teacher observations, training through
22 December2025 and the same complete January–March2026 fields. It retains the
existing4,000-tree maximum, learner and calibration method, using two threads
and component checkpoints. It runs one development period rather than another
quarter-by-quarter sweep. The period has already been examined; it is not a
new untouched holdout. Reserved later-period performance is not evaluated.

The completed comparison and descriptive established-field breakdown are in
`data/audit/established_form_029/native_canonical/`. Objective corrections remain
in the normal pipeline. The added family did not pass its advancement rule;
the selected input set remains151.

The secondary descriptive breakdown reports the same established fields
against the market, retaining every handicap/conditions and
Flat/Bumper/Hurdle/Chase subgroup. It uses saved forecasts without additional
fitting; subgroup results are not separate feature-selection trials.

## Completed probability comparison and decision

Lower values are better. The full comparison contains2,591 races/22,824 runners,
trained only through22 December2025 and scored on January-March2026.

| Score | Repaired151 inputs | Proposed169 inputs |
| --- | ---: | ---: |
| Race log loss |1.802286512|1.805041094|
| Runner Brier |0.090033832|0.090103981|
| Winner accuracy |30.548%|30.323%|

Candidate-minus-baseline log loss is+0.002754583, with date-cluster95% interval
[-0.000219651,+0.005988717]. This is not evidence of a predictive gain.
Both log loss and Brier also worsen in the all-experienced cohort. A smaller
binned calibration error alone does not justify selecting poorer probability
forecasts. No further Q1 feature or parameter search was performed.

These rows each use identical complete fields for all three forecasts:

| Cohort | Races | Repaired baseline log loss | Proposed family | Market |
| --- | ---: | ---: | ---: | ---: |
| All market-matched fields |2469|1.789123|1.791494|1.686613|
| Every runner has5 local starts |1129|1.841198|1.842511|1.757557|
| Established handicaps |951|1.904043|1.904622|1.809162|
| Established conditions |178|1.505433|1.510673|1.481850|
| Established Flat |573|1.927255|1.924265|1.843342|
| Established Hurdle |177|1.902876|1.909580|1.801649|
| Established Chase |379|1.682285|1.687588|1.607270|

There are no bumper fields where every runner has at least five prior local
starts. The small Flat point improvement is uncertain (date-cluster interval
[-0.010146,+0.003464]) and does not support a new subgroup-specific selection.

Established handicaps account for95.55% of the established cohort's net
log-loss deficit to the market. Their gap is0.094882 per race (date-cluster95%
interval[0.071009,0.119643]); the established-conditions gap is0.023583 with an
interval crossing zero. The next error investigation should therefore focus
on familiar handicap runners and effective performance/weight context.
This describes the inspected development period, not a guaranteed future gain.

The repaired baseline is essentially unchanged from026's historical score.
The027 teaching-data refresh also changed inputs to learning since026, so that
older comparison cannot isolate this batch's correctness repairs. No predictive
improvement or market parity is claimed for those repairs. Their confirmed
errors stay corrected regardless of this diagnostic score.

## Training cost

The full workflow comprises15 native learner fits: six for the two win
comparison arms, three for the chosen historical place model, and six for the
current win/place update. Each head uses capacity selection, an all-row refit,
and an earlier temporary model that supplies later, unused-in-fitting races for
calibration. Historical baseline win components are reused exactly, rather
than fitted twice. Daily forecasting does not retrain the saved model.

Requiring both heads' historical validation before a win-focused deployment
adds substantial overhead. Future feature-only work should reuse an unchanged
baseline and stop after an unsuccessful comparison; it should not automatically
trigger place work or a production update. This batch also contains mandatory
input repairs, so the existing weights still need a matching update. No extra
experiments or quarter-by-quarter sweep were added after the negative result.

## Installed update

Production run `20261028_212411_292221` replaces027 with freshly fitted win and
place weights through27 October2026. It uses151 selected inputs,667,873 eligible
training runners in71,477 races after burn-in,3,725 win trees and2,993 place
trees. Model and store share feature signature `a1fc3c96b9af9e7b` and data
signature `e888a0c91c997047`. The full current update took4,057.8 seconds.

Separate historical validation is saved as run `20261028_201629_989153`.
Its win forecasts reproduce the chosen comparison exactly (maximum error0),
and its place Brier is0.1746121967. The current full-data weights carry no
additional held-out performance claim. Installation fingerprints and the
validation link are preserved under `data/audit/established_form_029/installation/`.

Deep readiness confirms current DB contents match the feature store and installed
weights. Native operational checks cover100 recent fields/955 runners plus
Bath's12-runner12:42 card for29 October. Single-race versus batch maximum
probability error is0. Win totals, place totals/floors, runner identities and
finite probabilities all pass. This is an operational check, not an accuracy
estimate on those already-trained recent races or tomorrow's card.

Receipts: `data/audit/established_form_029/final_status.json`, `readiness.json`
in that directory, and `data/audit/live_upgrade_029/report.json`. A fresh study
`20261028T213231Z_established_form_029_20261028_212411_292221` pins the installed
weights and source, eligible from29 October. No recorder or automation was
started; earlier pinned studies are preserved unchanged.
