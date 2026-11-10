# Jumps margin specialist: registered test, 9 November 2026

This registers one confirmatory test before any of its test races is scored.
Nothing below has been fitted or scored on a test race. Machine-readable copy:
`data/research/meta_learner/jumps_margin_test/registration.json`.

## Hypothesis

A correction learned from each horse's **ability in pounds** improves the race
log loss of the full 063 three-seed average **on jumps races**.

Ability in pounds is a Kalman filter per horse and code over earlier runs, using
weight carried minus pounds beaten (`research/meta_learner/ability.py`,
`ability_in_pounds`). The correction sees its 14 inputs on Hurdle and Chase
races only; Flat rows are blank, so Flat forecasts do not change.

## How it was found

The [cheap screens](CHEAP_SCREENS_2026-11-09.md) found it on development
quarters 2025Q3–2026Q2. On all races, ability in pounds beat three shuffled
twins (−0.00044, 4/4 quarters) but its net gain was within noise. Split by race
type after scoring, all of the gain was on jumps races. A jumps-only correction
then scored, on the three-seed average:

| Development, 4,449 jumps races | Change per jumps race [95% interval] | Quarters better |
|---|---:|---:|
| Jumps-only ability correction | −0.00174 [−0.00270, −0.00084] | 4/4 |

The segment was chosen after seeing these quarters, so that estimate is
optimistic. That is why it needs a test on races it has not seen.

Bounce inputs also helped only on jumps. Before comparing, the rule was fixed:
add them only if they improve on ability alone by at least 0.0001 with three of
four quarters better. They did not (+0.00001 [−0.00015, +0.00016], 2/4), so the
registered recipe is ability alone.

## Test races

Every Hurdle or Chase race from **11 June 2026** (`config.GOLDEN_HOLDOUT_START`),
in date order, with a complete settled field (at least two runners, one winner)
and base forecasts for every runner, **until 3,300 such races**.

- About 1,367 jumps races ran from 11 June to 6 November 2026 (the store's last date today).
  The rest are future races. At recent seasons' rates (about 600 jumps races in
  November, 550 in December, 400 in January, 450 in February) the 3,300th is
  expected in **early March 2027**.
- None of these races has been used for this hypothesis or its inputs. The
  June–November part was examined for other questions (reports 055–057), so it
  is not untouched in general. That examination looked at general model scores
  and worst misses, not jumps margins.
- The test is read **once**, when the store holds the 3,300th race's result.
  There is no interim look.

## Frozen components

All fitted once, before any test race is scored, and saved with hashes. None is
refitted during the test.

1. **Base.** The full 063 recipe (`first_ensemble_20261109/registration.json`,
   spec `full-063`, 193 inputs, 75% market target), screen capacity, seeds 42,
   43 and 44, each fitted on all development rows (`input.parquet`, through
   3 June 2026). The base forecast is the mean of the three seeds'
   probabilities.
2. **Correction.** The Tier 1 recipe (`screen_candidates.TIER1`): race-softmax
   LightGBM from the seed-42 offset, 3 leaves, at least 5,000 runners per leaf,
   L2 200, learning rate 0.02, 100 rounds, 80% bagging, averaged over boosting
   seeds 7, 8 and 9. It is fitted on all eight development quarters
   2024Q3–2026Q2 against the stored seed-42 out-of-fold forecasts, with the 14
   jumps-only ability inputs.
3. **Inputs.** `ability_in_pounds_jumps` with `ability.SETTINGS` as registered,
   computed from the store's results. Each test race sees only earlier races
   (leak tests in `research/meta_learner/test_ability.py`).

## Statistic and pass rule

For each test race, d = race log loss of (base log probability + correction,
renormalised within the race) minus race log loss of the base.

- **Primary:** the mean of d over the 3,300 races. One-sided test of mean d < 0,
  with the standard error clustered by race date. **Pass if one-sided p < 0.01**,
  the project's confirmatory bar. A date-clustered bootstrap interval is
  reported alongside.
- **Power:** the clustered standard deviation of d on development jumps races is
  0.0316. At the development effect (−0.00174) power is 80%; at half of it,
  23%. A fail therefore rules out an effect as large as the development
  estimate, not a smaller one.
- **Descriptive only:** d by month, by Hurdle and Chase, with the seed-42 base
  instead of the average, and on Flat races, where d must be exactly zero.

**If it passes,** the jumps correction joins the stack as a candidate for the
prospective serving confirmation, which needs decision-time inputs. **If it
fails,** the jumps margin specialist is dropped, and these races are not used
again for this hypothesis.

## What the test needs before it can be scored

1. **Commit** the code and this registration, and record the commit in the
   JSON. The JSON already stores digests of the fitting functions.
2. **Test-race inputs for the base.** `input.parquet` stops at 3 June 2026. Build
   rows for the test races with `load_dataset(development=False)` and the 063
   reconstruction (`features_063`, which needs the full history), then check
   that the development rows match `input.parquet`.
3. **Fit and save** the three base models on development rows (about 20 minutes
   at screen capacity) and the correction boosters (under a minute).
4. **A scoring script** that refuses to score before 3,300 qualifying races and
   refuses a second scoring, writing a receipt.
5. **Store refreshes** through early March 2027, so that results reach the
   store. Ability inputs need only results. The base model's five ATR inputs
   matter too: their coverage fell from about 70% in the development period to
   25–30% in September–October 2026. That weakens the base on those races but
   does not bias a paired comparison.

## Caveats

- The base is frozen at 3 June 2026 and is up to nine months old by the read.
  The correction is frozen with it. The test measures information, not a
  serving recipe.
- The development estimate carries selection bias, so the true effect is likely
  smaller than −0.00174. The test is powered for that size.
