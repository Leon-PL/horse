# Cheap screens of meta-learner candidates, 9 November 2026

Recent experiments refitted the whole generalist for every idea: about five
minutes per quarter at screen capacity and 25–36 minutes at native capacity,
repeated over quarters, seeds and calibration slabs. This study adds a cheaper
first stage so that candidates from the
[meta-learner plan](../architecture/META_LEARNER_PLAN.md) can be ruled in or out
in minutes. Only survivors earn a refit.

These are retrospective screens on reused development quarters. A pass is a
shortlist entry, not evidence of a confirmed gain.

## Method

The [first ensemble benchmark](FIRST_ENSEMBLE_BENCHMARK_2026-11-09.md) saved
chronological out-of-fold forecasts of the full 063 generalist: seed 42 for
2024Q3–2026Q2 and seeds 43 and 44 for 2025Q1–2026Q2. The screens reuse them and
never refit the generalist. Each screen keeps the reference forecast fixed and
learns only a small correction from the candidate's own inputs.

- **Tier 0, offset logit.** A race conditional logit whose fixed offset is the
  reference's log probability, plus step coefficients per candidate input (four
  quantile steps and a missing indicator). Groups wider than 40 inputs get two
  steps and stronger shrinkage (L2 0.1 instead of 0.001). Seconds per screen.
- **Tier 1, residual boost against a twin.** Three LightGBM race-softmax
  corrections, averaged, each starting from the reference offset: 3 leaves, at
  least 5,000 runners per leaf, L2 200, learning rate 0.02, 100 rounds, 80% row
  bagging. The same fit is repeated on a **twin** in which the candidate's
  inputs are moved, as a block, to random rows of the same quarter. The twin
  keeps the inputs' distribution and coverage over time but breaks their link
  to the runner, so the candidate-minus-twin difference removes what the fit
  gains or loses from capacity alone. About a minute and a half per candidate.

Corrections are learned on the seed-42 forecasts, which cover more training
quarters, and applied to both the seed-42 forecast and the three-seed average.
The three-seed average is the bar. Every fit uses earlier quarters only, through
`src.wf_eval` folds with the seven-day purge. The scored quarters are
2025Q3–2026Q2 (11,905 races, 109,631 runners). Intervals are paired,
date-clustered bootstraps.

**Verdict rule, fixed before the candidates were run.** Advance if the Tier 0
interval on the average lies wholly below zero, or if Tier 1 beats its twin by
at least 0.0003 with three or more quarters better. Reject if both lower
bounds are above −0.0003. Otherwise unresolved.

## Cost

| Tier | What is fitted | Time per candidate |
|---|---|---|
| 0 | One small logit per scored quarter | Seconds |
| 1 | Three small LightGBM corrections per scored quarter, plus the same for the twin | About 1.5 minutes |
| 2 | Generalist refit with the candidate's inputs, repeat seeds in both arms | About an hour at screen capacity |
| 3 | Native capacity, prospective confirmation | Hours to months |

## Validation

### What did not work

The first harness used an early-stopped residual boost (15 leaves, at least
400 runners per leaf) that also saw the reference forecast and the 063 race
context, judged against a control with the same inputs minus the candidate.

- The control made the reference worse in all four quarters (+0.00066
  [+0.00005, +0.00127]). Early stopping chose 0 to 93 trees by noise.
- A fixed-round correction that saw only the reference also hurt (+0.00057).
- Two shuffled copies of the same inputs differed by 0.00058, as much as the
  effects being sought.
- Tier 0 on the 97-input 063 group at five bins overfitted (+0.00275, with its
  twin at +0.00476).

The cause is calibration drift. The temperature that best fits full 063 varies
from 0.956 to 1.057 between quarters (2025Q3 wanted 0.956, 2024Q4 wanted
1.057). A correction that can recalibrate learns the calibration of the
training quarters and applies it to a quarter that wanted the opposite. The
final instruments therefore see only the candidate's inputs, and Tier 1 is
judged against its twin.

### Known results replayed

Each check uses the final settings. "Known" is the earlier pooled or refitted
result on the same races.

| Check | Known | Tier 0 vs average | Tier 1 vs twin, average | Verdict |
|---|---|---:|---:|---|
| Performance/spread | Pool with the average: −0.00079 [−0.00121, −0.00036], 4/4 | −0.00096 [−0.00174, −0.00016], 4/4 | −0.00108 [−0.00180, −0.00033], 3/4 | Advance |
| Explicit text model | Pool with the average: +0.00008 [−0.00006, +0.00022], 1/4 | +0.00015 [−0.00015, +0.00044], 2/4 | +0.00017 [−0.00027, +0.00060], 1/4 | Reject |
| Handicap leeway | Pool with the average: +0.00003 [−0.00011, +0.00017], 2/4 | +0.00027 [−0.00019, +0.00072], 2/4 | +0.00027 [−0.00019, +0.00070], 1/4 | Reject |
| The 64 inputs full 063 added, on the original 129 | Refit: −0.00491 [−0.00705, −0.00265], 4/4 | −0.00017 [−0.00081, +0.00043], 2/4 | −0.00046 [−0.00110, +0.00018], 3/4 | Partial |

The twins cost between −0.00002 and +0.00015, so the fits no longer damage
the reference by themselves.

The screens agree with every pooled result. They recover only about a tenth
of the 063 refit gain. That gain combined the 64 inputs with the 75% market
share target, and the inputs mostly re-express information the original 129
already held: rating gaps, signal gaps and context weighting. A residual
screen measures **new information** well. It underestimates inputs that change
how the generalist uses what it already has, and those need a refit.

### Refit noise

One full 063 fit against another with a different seed, on the same races:

| Comparison | Change [95% interval] | Largest quarter swing |
|---|---:|---:|
| Seed 43 − seed 42 | −0.00026 [−0.00186, +0.00135] | 0.0023 |
| Seed 44 − seed 42 | −0.00013 [−0.00156, +0.00132] | 0.0023 |
| Seed 44 − seed 43 | +0.00013 [−0.00146, +0.00162] | 0.0013 |

A single-seed refit cannot resolve a change smaller than about 0.0015. Earlier
single-fit comparisons of small additions were inside this band. A Tier 2
refit therefore needs repeat seeds in both arms, about an hour per candidate
at screen capacity, and is reserved for candidates whose screen gain is large
or whose value lies in restructuring existing inputs.

## A new result: performance/spread on the three-seed average

The benchmark pooled performance/spread only with a single full 063 fit and
judged that pool against the three-seed average (+0.00013). Pooled with the
average itself, using the benchmark's log-pool combiner and cached forecasts
(`pool_average.py`, no new fits), it improves the average:

| Pool with the three-seed average | Change [95% interval] | Quarters better | Weight |
|---|---:|---:|---:|
| Performance/spread | **−0.00079 [−0.00121, −0.00036]** | 4/4 | 9.4–10.2% |
| Explicit text model | +0.00008 [−0.00006, +0.00022] | 1/4 | 3.0% falling |
| Handicap leeway | +0.00003 [−0.00011, +0.00017] | 2/4 | 5.3% falling |

The text and leeway rows reproduce the benchmark's figures exactly, which
confirms that the rebuilt average matches the stored one. Performance/spread is
the first stacked component to beat the repeat-fit reference in these
quarters. Averaging seeds and adding performance/spread overlap only partly:
together they come to about −0.0021 against a single fit. This is retrospective
and needs prospective confirmation alongside full 063.

## Results

Five input groups from the plan, none of them in full 063:

| Group | Inputs | Source |
|---|---:|---|
| Tracking | 184 | RTV RaceiQ and ATR sectional aggregates in the store that full 063 does not use (comment and Timeform text columns excluded) |
| Pedigree in the store | 26 | Sire, dam and dam-sire rates and surface, ground and trip edges not in full 063 |
| First string | 7 | `trainer_first_string_*`: which of a stable's runners is its first string by rating, Elo, odds or jockey |
| Relatives' form | 7 | New: runs, wins, places, finishing position and mean rating of the dam's other offspring on earlier dates (`candidate_features.relatives_form`) |
| Same-day bias | 5 | New: draw and running-style bias in races on the same card that started at least 30 minutes earlier, and its interaction with this runner's draw and style (`candidate_features.same_day_bias`) |

**Excluded columns.** The first tracking run had 194 inputs, ten of which
`src/model.py` lists in `EXCLUDE_COLUMNS`: the legacy composites
`atr_speed_score` and `atr_stride_x_gallop` and their field-relative versions,
blocked because an earlier build standardised them over the whole dataset. The
model spec refused them when the standalone specialist was built. Every store
group now drops `EXCLUDE_COLUMNS`, and the tracking figures below are from the
re-run with 184 inputs. Tracking was slightly stronger without them (−0.00077
against three twins, from −0.00069), so they were not the source of its gain.
The first-string row is from the original run, which included its excluded
odds-ranked columns; the "without odds" row is the valid one.

### First pass (one twin)

Change against the three-seed average; Tier 1 is reported against its twin.

| Group | Tier 0 | Tier 1 vs twin | Verdict |
|---|---:|---:|---|
| Tracking | +0.00001 [−0.00091, +0.00093], 2/4 | −0.00081 [−0.00137, −0.00026], 4/4 | Advance |
| Same-day bias | −0.00001 [−0.00057, +0.00055], 2/4 | −0.00037 [−0.00093, +0.00018], 3/4 | Advance |
| Relatives' form | +0.00082 [+0.00012, +0.00147], 1/4 | −0.00035 [−0.00082, +0.00011], 3/4 | Advance |
| Pedigree in the store | +0.00239 [+0.00093, +0.00387], 0/4 | −0.00032 [−0.00079, +0.00013], 3/4 | Advance |
| First string (with excluded odds columns) | −0.00054 [−0.00137, +0.00031], 3/4 | −0.00019 [−0.00052, +0.00018], 2/4 | Unresolved; invalid |
| First string without odds | −0.00028 [−0.00074, +0.00019], 3/4 | +0.00003 [−0.00008, +0.00013], 3/4 | Null |

The first-string group's strongest input ranks a stable's runners by `odds`,
which `src/model.py` excludes as market-derived; three of its seven columns
depend on it. Without them the group is null against its twin, so its apparent
signal was market information that a forecast made before the market forms
cannot use.

Tier 0 overfits groups of correlated, drifting inputs even at seven inputs
(relatives' form) and is uninformative for wide groups; for those, Tier 1
against its twin is the instrument to read.

### Survivors against three twins

The four survivors were re-screened against twins from three different
shuffles and against the mean of the three. "Net" is the correction's change
against the uncorrected average, which is what a stack would actually gain.

| Group | Net vs average | vs mean of three twins | Each twin | Reading |
|---|---:|---:|---|---|
| Tracking | −0.00051 [−0.00092, −0.00012], 4/4 | **−0.00077 [−0.00120, −0.00036], 4/4** | −0.00081, −0.00084, −0.00066 | Robust |
| Same-day bias | −0.00038 [−0.00086, +0.00011], 4/4 | **−0.00050 [−0.00100, −0.00000], 4/4** | −0.00037, −0.00076, −0.00038 | Robust, smaller |
| Pedigree in the store | −0.00008 [−0.00043, +0.00026], 1/4 | −0.00036 [−0.00074, +0.00002], 3/4 | −0.00032, −0.00042, −0.00033 | Information present, net gain about zero |
| Relatives' form | −0.00008 [−0.00038, +0.00021], 2/4 | −0.00017 [−0.00052, +0.00017], 2/4 | −0.00035, +0.00004, −0.00021 | Not robust; treat as null |
| All four in one correction | −0.00052 [−0.00090, −0.00017], 4/4 | −0.00074 [−0.00117, −0.00034], 4/4 | −0.00072, −0.00087, −0.00063 | About tracking alone |

The tracking group's strongest inputs are ATR efficiency, closing-in-hand and
turn-of-foot aggregates over the last three to five runs, field-relative
versions of them, and RTV stride frequency, exit-to-entry speed and speed lost.
They join strictly earlier runs that were published before the forecast time
(`src/atr_features.py`, backward as-of join without exact matches), so this is
not a timing leak. Their exclusion from full 063 fits the feature-crowding
argument in the plan: a model fitted on every year cannot hear a source that
starts in 2023 while strong rating features are available.

### Specialists: one correction, two, or a standalone model

`specialists.py` saved out-of-fold corrections for tracking alone, same-day
bias alone and both together for 2025Q1–2026Q2, so that a combiner could be
fitted on 2025Q1–Q2 before the first scored quarter. It also fitted a
standalone tracking model: the full 063 recipe (75% market-share target,
screen capacity) on the 199 permitted tracking inputs plus 22 shared context
inputs (race type, handicap, novice, maiden and bumper flags, class, distance,
going, surface, field size and age), trained only on rows from July 2023.

All rows start from the three-seed average pooled with performance/spread.

| Specialists on the pool | vs three-seed average | vs the joint correction | Quarters better than joint |
|---|---:|---:|---:|
| None (the pool) | −0.00079 [−0.00121, −0.00036] | +0.00062 [+0.00024, +0.00103] | 0/4 |
| Tracking correction only | −0.00130 [−0.00189, −0.00073] | +0.00011 [−0.00006, +0.00028] | 1/4 |
| Same-day correction only | −0.00117 [−0.00175, −0.00056] | +0.00024 [−0.00025, +0.00075] | 2/4 |
| One joint correction (189 inputs) | −0.00141 [−0.00198, −0.00086] | — | — |
| **Separate corrections, summed** | **−0.00168 [−0.00241, −0.00096]** | **−0.00027 [−0.00061, +0.00008]** | **4/4** |
| Separate corrections, weights fitted on earlier quarters | −0.00161 [−0.00223, −0.00099] | −0.00020 [−0.00043, +0.00002] | 4/4 |
| Standalone tracking model in the pool | −0.00078 [−0.00132, −0.00028] | +0.00063 [+0.00012, +0.00115] | 1/4 |
| Standalone model in the pool, plus the joint correction | −0.00139 [−0.00205, −0.00077] | +0.00002 [−0.00027, +0.00028] | 2/4 |

- **Separate corrections beat one joint correction.** Two corrections with
  their own trees capture both sources; in one heavily regularised correction
  the 184 tracking inputs crowd out the five same-day inputs, the same
  crowding that keeps tracking out of the generalist.
- **Summing needs no fitting.** Weights fitted on earlier quarters (unpenalised
  race logit) came out at 0.66–1.08 for tracking and 0.65–0.80 for same-day
  bias, rising towards one as more quarters became available, and scored
  slightly worse than plain unit weights. Each correction was already fitted
  as a correction to the same reference.
- **The standalone tracking model adds nothing.** Alone it scores 1.9705
  against 1.8247 for the average. The pool gives it 2.2–4.5% weight and the
  result does not move (−0.00078 against −0.00079 without it). A source that
  only partly describes a runner is useful as a correction to the generalist,
  not as a rival forecast.

The stack recipe is therefore: three-seed average, log-pooled with
performance/spread, plus the tracking and same-day corrections summed. Against
a single full 063 fit it is **−0.00295 [−0.00405, −0.00188]**, better in all
four quarters, more than twice the gain from averaging seeds (−0.00127).

**Caveats.** These are retrospective development quarters, already used to
choose the full 063 recipe and to compare six candidate groups and several
combiners here. The verdict rule was fixed before the candidates were run, but
the stack recipe was chosen on the same quarters, so its size is optimistic.
The corrections are learned on the seed-42 forecasts and applied unchanged to
the average and the pool.

### What was not run

No Tier 2 refit. The screens put tracking at about −0.0005 to −0.0008, inside
the ±0.0015 noise of a single-seed refit, so a refit test needs three seeds in
the candidate arm (twelve screen-capacity fits, about an hour; the reference
arm's three seeds are already cached). It answers a different question:
whether these inputs help more inside the generalist than as a correction on
top of it.

## New information: ability in pounds, relatives' ratings, going reasons

The plan's next step was three sources of information that are not in the
generalist. They were built in `research/meta_learner/ability.py` with
settings fixed before screening, and each has a leak test: changing the
results of a race and every later race leaves that race's inputs unchanged.

- **Ability in pounds (14 inputs).** A Kalman filter for each horse and code
  (Flat, jumps) over races in time order. A run's performance is weight
  carried minus pounds behind the winner (the performance model's scale:
  15 lb per length divided by furlongs, capped at 30 lb). Each race's level is
  estimated from the runners' prior abilities, so the scale follows official
  ratings, and a horse's first run starts from its official rating. Young or
  lightly raced horses (Flat age 3 or less, jumps 5 or less, or fewer than four
  runs) get process noise of 4 lb per run, exposed horses 1.5 lb, plus
  0.5 lb² per 30 days off. Jumps means race type Hurdle or Chase; the store's
  race type puts National Hunt Flat races under Flat. The inputs are the state before the race: ability, its uncertainty, runs,
  peak, last change, days since, distance below peak, and ability, ability at
  the weights, ability against the mark and peak relative to the field.
  Ability correlates 0.78 (Flat) and 0.85 (jumps) with the official rating.
- **Relatives' ratings (6 inputs).** For the dam's other offspring (keyed by
  dam and dam-sire) in the same code, on earlier dates only: how many there
  are, the mean and best of their peak abilities, the same relative to each
  sibling's trainer's average ability on earlier dates, and the mean relative
  to the field. This replaces the simple win and place rates screened above.
- **Going reasons (32 inputs).** BHA withdrawals for the going, joined to the
  going of the meeting they were made at (98% matched by course and date) and
  available 48 hours after the race, the store's policy. Six new inputs compare
  today's going with those goings: count, softest, firmest, same going, nearest
  distance and whether today's going is off the horse's usual going in the
  direction it was withdrawn from. The group adds the store's 26 BHA going and
  withdrawal columns; full 063 uses no BHA column. A second group has only the
  six new inputs.

### First pass

| Group | Inputs | Tier 0 | Tier 1 vs twin | Verdict |
|---|---:|---:|---:|---|
| Ability in pounds | 14 | +0.00117 [−0.00014, +0.00250], 1/4 | −0.00054 [−0.00120, +0.00017], 4/4 | Advance |
| Relatives' ratings | 6 | +0.00072 [+0.00013, +0.00134], 0/4 | +0.00002 [−0.00046, +0.00041], 1/4 | Unresolved: no information beyond the twin |
| Going reasons with store BHA columns | 32 | +0.00127 [+0.00029, +0.00211], 0/4 | +0.00020 [−0.00029, +0.00071], 2/4 | Reject |
| Going comparisons only | 6 | +0.00028 [−0.00009, +0.00063], 1/4 | Cannot be fitted | Null |

The going comparisons cover 3–5% of runners. The first training window
(2024Q3–2025Q2) has 4,896 runners with a going withdrawal, fewer than Tier 1's
minimum of 5,000 per leaf, so LightGBM can make no split. The recipe was not
relaxed for one group. Tier 0, which handles sparse inputs, is null. In the
wider group the strongest inputs were days since any withdrawal and health
withdrawals, not the going.

### Ability in pounds: three twins and the stack

| Ability correction | Change [95% interval] | Quarters better |
|---|---:|---:|
| Against the average (net) | −0.00028 [−0.00080, +0.00027] | 3/4 |
| Against the mean of three twins | −0.00044 [−0.00098, +0.00015] | 4/4 |
| Each twin | −0.00054, −0.00035, −0.00042 | |
| Added to the stack, against the stack | −0.00030 [−0.00082, +0.00025] | 3/4 |
| Added to the stack, against its twins on the stack | −0.00046 [−0.00100, +0.00013] | 4/4 |

The stack is the three-seed average log-pooled with performance/spread plus
the tracking and same-day corrections. Ability's correction is fitted
separately and summed onto it (`new_information.py`), as the specialists are.
Its strongest inputs in every quarter were ability at the weights relative to
the field, runs, uncertainty, rank in the field and days since the last run.

Split by race after scoring, the gain is all in jumps races:

| Races | Number | Against the stack | Against its twins |
|---|---:|---:|---:|
| Jumps | 4,449 | −0.00121 | −0.00129 |
| Flat | 7,456 | +0.00024 | +0.00004 |
| Every runner rated | 10,421 | −0.00047 | −0.00061 |
| Under half the runners rated | 1,484 | +0.00092 | +0.00061 |

**Exploratory: jumps only.** Chosen after seeing the split above, on the same
quarters, so it is not a pre-registered test. The same 14 inputs with Flat rows
blank, a separate correction summed onto the stack:

| | Change [95% interval] | Quarters better |
|---|---:|---:|
| Against the stack | **−0.00065 [−0.00100, −0.00029]** | 4/4 |
| Against its three twins | −0.00083 [−0.00120, −0.00041] | 4/4 |
| Stack plus jumps ability, against the three-seed average | −0.00233 [−0.00315, −0.00154] | 4/4 |
| Stack plus jumps ability, against a single fit | −0.00360 [−0.00472, −0.00249] | 4/4 |

Per quarter against the stack: −0.00051, −0.00097, −0.00033, −0.00078.

**Reading.** Ability in pounds measured from margins and weights adds
information to the generalist for jumps runners and none for Flat runners or
unexposed fields. Why jumps and not Flat has not been tested. The jumps-only
gain is about as large as the tracking correction's, but the segment was chosen
on the scored quarters.

Relatives' form is now null in two forms: simple rates and ratings with a
trainer adjustment. The effect the plan expected was small, and the screen
cannot find it. Going reasons carry nothing the generalist lacks, and the new
going comparisons are too sparse to matter even if they were informative.

## Forecast-level components: head-to-head, completion, improvement, bounce

The plan's untested targets (S10, S13, S14, S15). Recipes and gates were
written to `forecast_components_registration.json` before anything was
scored (`forecast_components.py register`).

- **Head-to-head (S10), a pooled component.** A LightGBM score on full 063's
  193 inputs, trained with a Bradley–Terry loss over every pair of runners in a
  race. Each pair's target is 75% the market's chance that one beats the other
  (Luce, from the BSP-then-SP teacher full 063 uses) and 25% which finished
  ahead (non-finishers behind every finisher, level with each other). Each
  runner carries about one unit of pair weight (recency weight divided by field
  size minus one). Win chances are a race softmax of the score times a scale
  fitted on the last 20% of training races by a separate earlier fit. Screen
  capacity, seed 42, out of fold for 2025Q1–2026Q2, pooled with the three-seed
  average and performance/spread with weights fitted on earlier quarters.
- **Completion (S13), screened as a correction.** The chance that a jumps
  runner fails to complete, fitted on full 063's inputs plus 20 completion and
  jumping-error history columns from the store, out of fold for every screening
  quarter. The screen inputs are that chance, the log chance of completing and
  its value relative to the field; Flat runners are blank. This tests whether
  a dedicated completion estimate adds information about winning that the
  factorised form would exploit.
- **Improvement (S14), screened as a correction.** From each run's residual
  against the horse's own prior ability (the ability filter): the mean residual
  of the sire's unexposed offspring (fewer than six runs) on earlier dates, the
  trainer's runners and unexposed runners over the previous 365 days, the
  horse's last three residuals, its run count and field-relative versions.
  Means are shrunk towards zero by 20 (sire) and 30 (trainer) runs.
- **Bounce (S15), screened as a correction.** The last run's residual, how far
  it raised the horse's peak, whether it was a career best, whether it won or
  was beaten under a length, its distance, days since, runs in the previous
  60 days, and the last residual when the horse returns within 21 days.

**Gates.** A pooled component joins the stack if the stack with it in the pool
beats the stack with a 95% interval below zero and at least three of four
quarters better. An input group follows the fixed verdict rule; a survivor is
re-checked against three twins and summed onto the stack, and joins if that
beats the stack (interval below zero, three of four quarters) and beats the
mean of its three twins by at least 0.0003. All races; a split by race type is
reported but is not a test.

### Corrections: completion, improvement, bounce

| Group | Inputs | Tier 0 | Tier 1 vs twin | Verdict |
|---|---:|---:|---:|---|
| Completion (S13) | 3 | +0.00032 [−0.00019, +0.00080], 1/4 | +0.00020 [−0.00012, +0.00053], 1/4 | Reject |
| Improvement (S14) | 7 | +0.00100 [+0.00010, +0.00189], 0/4 | +0.00015 [−0.00028, +0.00061], 1/4 | Reject |
| Bounce (S15) | 8 | +0.00041 [−0.00029, +0.00115], 2/4 | −0.00016 [−0.00060, +0.00028], 2/4 | Unresolved; does not advance |

The completion model itself works: out of fold its AUC for failing to complete
is 0.69–0.72 in every quarter from 2024Q3 to 2026Q2, with 12–18% of jumps
runners failing to complete. The generalist already prices that: the
factorised form has nothing to add.

Split by race type against the twin (reported, not a test): completion
+0.00002 on Flat and +0.00048 on jumps; improvement +0.00011 and +0.00021;
bounce +0.00015 on Flat and −0.00067 on jumps. Bounce, like ability in pounds,
helps only on jumps races, and both come from the same margin-based filter.
That is a hypothesis for a test registered in advance (a jumps-only
correction of ability and bounce inputs, scored on later data), not a result.

### Head-to-head in the pool

| Forecast, 2025Q3–2026Q2 | Race log loss | Against the three-seed average |
|---|---:|---:|
| Single full 063 fit | 1.82598 | +0.00127 [+0.00043, +0.00213], 0/4 |
| Head-to-head alone | 1.82796 | +0.00324 [+0.00173, +0.00491], 0/4 |
| Three-seed average | 1.82471 | — |
| Average pooled with head-to-head | 1.82469 | −0.00003 [−0.00008, +0.00002], 3/4 |
| Pool (average and performance/spread) | 1.82392 | −0.00079 [−0.00121, −0.00036], 4/4 |
| Pool with head-to-head added | 1.82392 | −0.00079, unchanged |
| Stack | 1.82304 | −0.00168 [−0.00241, −0.00096], 4/4 |
| Stack with head-to-head in the pool | 1.82304 | unchanged; fails the gate |

The pool gives head-to-head zero weight in every quarter (performance/spread
keeps 9–10%). The pairwise model on the same inputs and market-blended target
is 0.0020 worse than one exploded-softmax fit and carries nothing the average
lacks. This matches 062, where a CatBoost pairwise learner at best tied the
057 generalist. Its fitted scales were 1.12–1.15 in every quarter, so the
loss is not a calibration failure.

## Decisions

- Performance/spread moves from "research only" to the stack, pooled with the
  three-seed average.
- Tracking and same-day bias enter as two separate residual corrections,
  summed, on the pool.
- The standalone tracking model is dropped: it gets little pool weight and
  adds nothing.
- Pedigree in the store and relatives' form stay out: the information is
  either absorbed (pedigree) or not robust (relatives' form). Relatives' form
  should be retried only in rating points with trainer adjustment, as the plan
  describes, not as the simple rates screened here.
- First string is null once its odds-based columns are removed. A version on
  decision-time prices belongs to the market tier, not the generalist.
- Future specialists are screened as separate corrections and summed, not
  merged into one correction.
- The stack is a retrospective lead. It needs the same prospective
  confirmation as full 063, with locked recipes.
- Ability in pounds on jumps races is a candidate third correction for the
  stack, its jumps-only scope recorded as chosen after the fact. It joins the
  stack only after a locked recipe is confirmed prospectively. Ability on all
  races is not used: on Flat races it adds nothing.
- Relatives' form is dropped in both forms. Going reasons are dropped,
  together with the store's BHA withdrawal columns.
- The plan's untested targets are all null: head-to-head gets zero pool weight,
  and completion, improvement and bounce corrections do not beat their twins.
  None joins the stack. The jumps-only ability correction is registered as one
  confirmatory test on races it has not seen
  ([registration](JUMPS_MARGIN_SPECIALIST_REGISTRATION_2026-11-09.md)); bounce
  was left out because it added nothing on top of ability (+0.00001, 2/4).

## Code

- `research/meta_learner/screening.py`: harness (tiers, twins, folds, scoring).
- `research/meta_learner/candidate_features.py`: relatives' form and same-day
  bias, computed from store history.
- `research/meta_learner/screen_candidates.py`: stages `validate`,
  `candidates`, `followup` (three twins) and `stack`. Store groups drop
  `src.model.EXCLUDE_COLUMNS`.
- `research/meta_learner/pool_average.py`: pools of cached components with the
  three-seed average.
- `research/meta_learner/specialists.py`: separate corrections, the standalone
  tracking model and the specialist comparison.
- `research/meta_learner/ability.py`: ability in pounds (Numba Kalman filter,
  about 13 seconds for 758,000 runs), relatives' ratings and going-withdrawal
  comparisons.
- `research/meta_learner/forecast_components.py`: registration, the
  head-to-head (Bradley–Terry) component, the jumps completion model and the
  pooled comparison. `ability.py` also builds the improvement and bounce inputs.
- `research/meta_learner/new_information.py`: a group's correction and three
  twin corrections, summed onto the stack (`corrections`, `compare`).
  `screen_candidates.py followup --only NAME` re-checks one group against three
  twins.
- Tests: `research/meta_learner/test_screening.py`,
  `research/meta_learner/test_candidate_features.py`,
  `research/meta_learner/test_specialists.py`,
  `research/meta_learner/test_ability.py` (leak tests for the ability-based
  inputs and going comparisons), `research/meta_learner/test_forecast_components.py`
  (pairwise gradient against finite differences).
- Outputs: `data/research/meta_learner/screens_20261109/` (`validation.json`,
  `pool_average.json`, `followup.json`, `stack.json`, `specialists.json`,
  `specialist_corrections.parquet`, `tracking_specialist_oof.parquet`, one
  `*_summary.json` per candidate, `followup_ability-lb.json`,
  `new_information_<group>.json` and `.parquet`, and the logs
  `new_information.log`, `new_information_followup.log`,
  `going_withdrawn.log`, `ability_jumps.log`; `forecast_components_registration.json`,
  `head-to-head-screen_oof.parquet`, `completion_oof.parquet`,
  `forecast_components.json`, `head_to_head.log`, `components_screen.log`).
