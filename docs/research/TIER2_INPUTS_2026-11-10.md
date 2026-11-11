# Tier 2: ability and same-day inputs inside the generalist, 10 November 2026

Step 2 of the [meta-learner plan](../architecture/META_LEARNER_PLAN.md) asked
whether the inputs that survived the
[cheap screens](CHEAP_SCREENS_2026-11-09.md) help more inside the full 063
generalist than as corrections on top of it. The screens understate inputs
that restructure what the generalist already has (they recovered a tenth of the
063 refit gain), and ability in pounds is a rating-type input like the rating
gaps that drove 063, so a larger gain inside was plausible.

**Answer: no.** Inside the generalist the inputs are worth about what they are
worth on top, the result does not clear the gate, and it is noisier. Keep them
as corrections. Retrospective, on development quarters already used to choose
these inputs.

## What was run

`research/meta_learner/tier2_inputs.py`, arm `ability-sameday`: the registered
full 063 recipe (193 inputs, 75% market target, screen capacity) plus the 14
ability-in-pounds inputs (all races) and the 5 same-day bias inputs, 212 inputs
in all. Seeds 42, 43 and 44, purged quarterly refits for 2025Q3–2026Q2, twelve
fits, about six minutes each on five threads. The reference arm is the first
ensemble benchmark's cached three seeds: same rows (`input.parquet`), folds
and modelling code (the script refuses to run if either hash has moved). Its
scoring path reproduces the benchmark's 1.825983 and 1.824713 exactly.

**Gate, written into the script before any fit:** the arm's three-seed average
against the cached three-seed average, paired date-clustered 95% interval
below zero and at least three of four quarters better.

The tracking group (184 inputs) was not run. A 396-input fit needs more memory
than this machine had free (the 212-input fit peaked at 4.3 GB with 5.3 GB
free), and the arm below gives no reason to expect a different answer.

## Result

11,905 races, 109,631 runners.

| Comparison | Change [95% interval] | Quarters better |
|---|---:|---:|
| **Arm average against reference average (the gate)** | **−0.00050 [−0.00160, +0.00063]** | 3/4 — fails |
| Seed 42 against seed 42 | −0.00054 [−0.00203, +0.00111] | 2/4 |
| Seed 43 against seed 43 | −0.00059 [−0.00225, +0.00116] | 2/4 |
| Seed 44 against seed 44 | −0.00024 [−0.00195, +0.00148] | 2/4 |
| Same inputs as two summed corrections on the reference average | −0.00065 [−0.00138, +0.00020] | 3/4 |
| Inside against on top | +0.00016 [−0.00101, +0.00134] | 1/4 |
| Arm average with the same two corrections added again | +0.00026 [−0.00047, +0.00112] | 2/4 |
| Arm average plus tracking correction, against reference average plus tracking correction | −0.00051 [−0.00162, +0.00061] | 3/4 |

Per quarter, arm average against reference average: −0.00112, −0.00091,
+0.00087, −0.00050. Race log loss 1.824217 against 1.824713; ten-band
calibration error 0.0027 against 0.0031.

- The point estimate inside (−0.0005) matches the corrections on top
  (−0.0007). The refit did not unlock the larger gain that the rating gaps
  showed in 063.
- The refit absorbs the information: adding the same corrections again to the
  arm does nothing (+0.0003).
- The interval inside is half as wide again as on top (±0.0011 against
  ±0.0008), because every refit re-draws the generalist's own noise. A
  correction leaves the reference fixed.
- The tracking correction's value is unchanged on top of the arm.

## Exploratory: averaging all six fits

Not a registered comparison; computed from the forecasts above after the gate
was read.

| Average | Race log loss | Against the reference three-seed average | Quarters better |
|---|---:|---:|---:|
| Three reference fits and three arm fits | 1.823991 | −0.00072 [−0.00127, −0.00016] | 4/4 |
| The same, against the arm's three-seed average | | −0.00023 [−0.00079, +0.00033] | 3/4 |
| Two reference seeds against three (size of a seed) | | +0.00029 [−0.00015, +0.00071] | 0/4 |

Going from one fit to three was worth −0.00127, which puts the per-fit variance
term at about 0.0019 and the expected gain from three to six identical fits at
about −0.0003. The six-fit average gains −0.0007, so roughly −0.0004 beyond
plain seed averaging, the same size as the inputs' information measured the
other two ways. Three more reference seeds would separate the two effects; they
were not fitted.

## Decisions

- Ability in pounds and same-day bias stay as separate residual corrections.
  They do not move into the generalist's input list. This also keeps the
  locked prospective challenger (one full 063 fit) and the registered
  [jumps margin test](JUMPS_MARGIN_SPECIALIST_REGISTRATION_2026-11-09.md)
  unchanged.
- A refit is the wrong instrument for groups worth about 0.0005: three seeds
  per arm leave an interval of ±0.0011. Use the twin screens for information,
  and a refit only for a group expected to be worth 0.002 or more.
- The tracking arm is not run unless a machine with more memory is used.
- More fits in the average is the cheapest remaining gain in the generalist
  (about −0.0003 for three more), at the cost of fitting and serving time.

## Files

- `research/meta_learner/tier2_inputs.py` (`fit --arm NAME`, `report`), tests
  in `research/meta_learner/test_tier2_inputs.py`.
- `data/research/meta_learner/tier2_20261110/`: `ability-sameday_registration.json`
  (recipe, inputs, coverage, gate), `ability-sameday_seed{42,43,44}.parquet`,
  `report.json`, `inside_vs_on_top.json`, `six_fit_average.json`, the fit log
  and the fold cache.
