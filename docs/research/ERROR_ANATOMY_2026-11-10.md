# Why the model is wrong, and inputs built from its own errors, 10 November 2026

This study started from the model's errors instead of from a list of feature
ideas, and re-checked the labels and inputs instead of assuming them. It found
that the generalist makes **persistent, attributable errors**: it misjudges the
same trainers, sires and individual horses again and again, and the market does
not. Inputs that measure those past errors are the strongest group screened so
far.

**Confirmed on unseen races, 11 November (section 7).** One registered read on
5,761 races from 11 June to 6 November 2026: seven corrections on a seed-42
fit give **−0.00910, z −4.8, one-sided p = 0.0000007**. The own-horse and
market-gap pair alone gives −0.00602 (z −4.5). The caveat paragraph below
describes the development results only.

**Update, 11 November (section 5).** Teaching the corrections from past
closing prices, over a longer window and with more capacity, takes the stack
to 1.81665, −0.0093 against one full 063 fit. A check against a
native-capacity forecast shows about four fifths of that carries over, and
that the own-horse and market-gap corrections carry over in full.

**Headline.** Four new residual corrections added to the 9 November stack, with
weights fitted on earlier quarters only: **−0.00363 [−0.00548, −0.00188],
4/4 quarters**, race log loss 1.81941 against 1.82304 (stack), 1.82471
(three-seed average) and 1.82598 (one full 063 fit). The stack's gain over the
three-seed average goes from −0.0017 to −0.0053.

Everything here is retrospective, on development quarters 2025Q3–2026Q2
(11,905 races) that earlier research has used, and the input families were
chosen after looking at errors in the same forecasts. It is a strong lead, not
a confirmed gain.

## 1. Checking the instrument

On the scored quarters (`research/error_anatomy/frame.py`):

- Every race has exactly one winner, rows equal the declared field size, no
  runner is duplicated, and `won` agrees with `finish_position` everywhere.
- The racecard `form` string is pre-race: its last figure matches the horse's
  previous finishing position in the database 96.9% of the time and the
  current race's position 11.8% of the time (chance).
- The card's days-since-last-run equals the database gap for 99.1% of runners.
- First-time runners that appear to carry an official rating do not: the
  store fills unrated runners with a race median and flags them
  (`has_official_rating = 0`; all 41,065 true debutants are flagged). Not a
  leak, and not a real rating.
- No input's coverage or scale breaks in the scored quarters. Two things worth
  knowing: `trainer_prev_runs` is a count since September 2020, so it drifts
  upward for every trainer (+0.86 standard deviations by 2026Q2) and means
  something different each year; and tracking inputs are absent before
  mid-2023, as already documented.

**One real gap in the data.** 6.4% of scored runners (16.6% of hurdlers, 6.4%
of chasers, 2.4% on the Flat) show more runs on their racecard than the
database holds for them: point-to-points, runs abroad and runs before
September 2020. They account for 12% of the model's divergence from the market,
and the model under-rates them (actual wins 6.5% above its forecast; 2.7% for
BSP). The store flags them (`hidden_form_runs`) but has no results for those
runs beyond the card's form figures.

## 2. Where the model loses

| Forecast, races with BSP | Race log loss |
|---|---:|
| Uniform | 2.1447 |
| Three-seed full 063 | 1.8245 |
| Starting price | 1.7524 |
| Betfair SP | 1.7446 |

The model recovers 80% of what the market knows. The remaining 0.080 is
spread almost evenly: 0.069–0.090 for Chase, Flat and Hurdle, handicaps and
non-handicaps, every class, every distance band and every quarter. Only
bumpers (0.133) and fields above 16 (0.113) stand out, and they are small. It
is not a weak segment.

**On the market's own timeline** (631 races with a full-field price at every
snapshot):

| Forecast | Race log loss |
|---|---:|
| Betfair, 18:00 the evening before | 1.6540 |
| **Model** | **1.6420** |
| Betfair, 08:00 | 1.6171 |
| Betfair, 11:00 | 1.6045 |
| Betfair, one hour before | 1.5977 |
| Betfair, five minutes before | 1.5839 |
| Betfair SP | 1.5736 |

The model is better than the evening-before exchange price and worse than the
08:00 price. On the 7,292 races with an 11:00 price, 0.041 of the gap exists by
11:00 and 0.034 arrives afterwards. So roughly half of the gap is information
that is in the price hours before the race, and half is late.

**By runner** (the market-to-model divergence, which sums to the gap):

| Runners | Share of runners | Share of divergence |
|---|---:|---:|
| First run in the database | 7.2% | 13.4% |
| One earlier run | 6.7% | 8.9% |
| Off more than 120 days | 11.4% | 16.7% |
| Card shows runs the database lacks | 6.4% | 11.9% |
| Five or more earlier runs | 68.8% | 58.3% |

Among the 235 winners the market rated at more than 2.7 times the model's
chance (31% of the whole gap), 24% were first-time runners against 5.5% of all
winners, and 21% were returning from more than 120 days off against 10.5%.
The model has least to go on exactly where the stable knows most. But most of
the gap still sits on ordinary exposed horses.

## 3. The errors persist by trainer, sire and horse

Split the eight out-of-fold quarters into two halves and compute, for each
entity with at least 150 runners in each half, the standardised sum of
`won − forecast`. If the forecast used everything knowable about the entity,
the two halves would be uncorrelated.

| Correlation between halves | Trainers (193) | Jockeys (224) | Sires (184) |
|---|---:|---:|---:|
| Model (seed 42) | 0.19 | 0.10 | 0.13 |
| Betfair SP | 0.09 | −0.01 | 0.05 |
| Starting price | 0.14 | 0.02 | 0.05 |

The model's errors by trainer and sire repeat; the market's mostly do not.
The store has no input of this kind. Its 55 trainer columns are strike rates,
counts, Elo and target-encoded positions, which mostly restate how good the
trainer's horses are. None measures the trainer against what was expected.

## 4. Inputs built from the model's past errors

`research/error_anatomy/entity_residuals.py`. For every earlier date with an
out-of-fold forecast, each run yields:

- **win residual:** `won − p`;
- **rivals-beaten residual:** the share of rivals the runner beat minus the
  share the forecast expected (pairwise chance `p_i / (p_i + p_j)`), which uses
  the whole finishing order;
- **market gap:** `log(market p) − log(model p)` at that earlier race's
  starting price (BSP, else SP). It has no outcome noise.

The inputs are shrunk means of those signals over runs on **strictly earlier
dates**, with a 365-day half-life, per key, plus the same relative to the
field. Tests change the results and prices of a date and confirm that date's
inputs do not move (`test_entity_residuals.py`).

Out-of-fold forecasts existed only from 2024Q3, so `extend_oof.py` fitted the
registered full 063 recipe (seed 42, screen capacity, purged quarters) for
2022Q3–2024Q2 as well. The inputs then have two years of memory before the
first correction is trained.

| Group | Keys | Inputs |
|---|---|---:|
| Own horse | The horse itself: its residuals and its market gap | 7 |
| Market gap | Trainer, jockey, sire, trainer and jockey, trainer by code, trainer by freshness (debut, back from 60+ days, or fit) | 12 |
| Residual | The same six keys, win and rivals-beaten residuals | 30 |
| Second wave (declared before the long screen was read) | Trainer by track, trainer by handicap, sire by surface, going and trip, dam's sire | 30 |

### Screens

The 9 November instrument and verdict rule, unchanged: Tier 1 corrections
fitted on earlier quarters, against a twin with the inputs moved to random
rows of the same quarter. Change against the three-seed average.

| Group | Net on the average | Against its twin | Quarters |
|---|---:|---:|---:|
| First definition (forecasts from 2024Q3 only, no decay): all six keys | −0.00067 [−0.00115, −0.00020] | −0.00103 [−0.00161, −0.00047] | 4/4 |
| — trainer only | | −0.00095 [−0.00153, −0.00036] | 4/4 |
| — sire only | | −0.00063 [−0.00113, −0.00014] | 4/4 |
| — jockey only | | +0.00024 [−0.00017, +0.00063] | 1/4 |
| **Own horse** | **−0.00216 [−0.00326, −0.00123]** | **−0.00225 [−0.00340, −0.00125]** | 4/4 |
| **Market gap** | **−0.00164 [−0.00258, −0.00063]** | −0.00162 [−0.00260, −0.00053] | 4/4 |
| **Residual, long memory** | **−0.00119 [−0.00189, −0.00052]** | −0.00129 [−0.00210, −0.00053] | 4/4 |
| Second wave | −0.00077 [−0.00144, −0.00012] | −0.00071 [−0.00150, +0.00006] | 3/4 |

For scale, the best group before this was tracking at −0.00051 net. Own horse
is also the first group to pass Tier 0 (−0.00156 [−0.00292, −0.00029], 4/4).
Lengthening the memory from one year to three nearly doubled the residual
group (−0.00067 to −0.00119). The strongest inputs are the horse's own market
gap, and the trainer-by-freshness keys: how a trainer's debutants and
returners have done against the model.

### On the stack

Each group is its own correction, summed onto the stack (three-seed average
pooled with performance/spread, plus tracking and same-day corrections), with
three twins each.

| Added to the stack | Against the stack | Against the mean of three twins |
|---|---:|---:|
| Own horse | −0.00224 [−0.00334, −0.00130], 4/4 | −0.00241, 4/4 |
| Market gap | −0.00171 [−0.00265, −0.00069], 4/4 | −0.00174, 4/4 |
| Residual | −0.00121 [−0.00192, −0.00054], 4/4 | −0.00126, 4/4 |
| Second wave | −0.00079 [−0.00146, −0.00015], 3/4 | −0.00084, 3/4 |
| All four, summed at full weight | −0.00343 [−0.00563, −0.00136], 4/4 | |
| **All four, weights fitted on earlier quarters** | **−0.00363 [−0.00548, −0.00188], 4/4** | |

The groups overlap, so the four together give less than the sum of each alone
(−0.0060). The fitted weights (race logit on earlier quarters only, starting
from 2025Q1–Q2) were 0.83–0.94 for own horse, 0.94–1.16 for market gap,
0.58–0.73 for residual and 0.47–0.75 for the second wave. Refitting weights
for the two existing corrections alone changes nothing (+0.00007).

| Forecast | Race log loss | Against one full 063 fit |
|---|---:|---:|
| One full 063 fit | 1.82598 | — |
| Three-seed average | 1.82471 | −0.00127 |
| 9 November stack | 1.82304 | −0.00295 |
| **Stack plus the four error corrections** | **1.81941** | **−0.00658** |

### Timing stress test

A result this size deserves suspicion, so the two strongest groups were
rebuilt from runs at least 7 and 30 days before the race. A same-day or
next-day result or price leaking into the inputs would collapse here.

| Group, net on the average | No lag | 7 days | 30 days |
|---|---:|---:|---:|
| Own horse | −0.00216 | −0.00217 | −0.00121 |
| Market gap | −0.00164 | −0.00167 | −0.00140 |

Nothing changes at seven days, and at thirty the own-horse group loses the
part carried by a horse's most recent run. That is persistence, not a leak.

## 5. Second round, 11 November: the market as teacher

### What predicts the market's disagreement

`market_disagreement.py` regresses the closing gap, `log(market p) − log(model p)`
centred within the race, on each input family (fit on 2022Q3–2025Q2, 361,346
runners; scored on 2025Q3–2026Q2). The gap's standard deviation is 0.54.

| Inputs | Number | Out-of-sample R² |
|---|---:|---:|
| The generalist's own inputs | 193 | 0.132 |
| Every other permitted store column | 1,176 | 0.138 |
| The error inputs of section 4 | 79 | 0.114 |
| All together | 1,448 | 0.206 |

- The rest of the store predicts the gap no better than the model's own
  inputs do, so it holds little the model lacks. The error inputs add to
  either.
- The own-inputs figure is not zero. Part of that is arithmetic (the model
  matches the market's probability on average, not its logarithm), but part
  is real: see the capacity finding below.

### Corrections taught by past prices

Every correction in the stack learned from winners only, one bit per race.
`soft_corrections.py` gives a correction the generalist's own teacher: 75% the
earlier race's closing price, 25% the result. Scoring is still on results, and
no price of a scored race is used. Four arms were declared before scoring.
Net change on the three-seed average; every row is 4/4 quarters unless marked.

| Group | Results, rows from 2024Q3 | Prices, same rows | Prices, rows from 2022Q3 | Prices, 2022Q3, more capacity |
|---|---:|---:|---:|---:|
| Own horse | −0.00216 | −0.00222 | −0.00230 | **−0.00256** |
| Market gap by entity | −0.00164 | −0.00194 | −0.00194 | **−0.00274** |
| Residual by entity | −0.00119 | −0.00091 | −0.00085 | **−0.00180** |
| Second wave | −0.00077 (3/4) | −0.00095 | −0.00081 | **−0.00134** |
| Tracking | −0.00051 | | −0.00035 | **−0.00136** |
| Same-day bias | −0.00038 | | −0.00030 | −0.00045 |
| Ability in pounds | −0.00028 (3/4) | | −0.00045 | −0.00092 |
| Sale price (new, below) | +0.00004 (1/4) | | −0.00015 (3/4) | −0.00019 (3/4) |
| The generalist's own inputs | −0.00037 | | −0.00032 (3/4) | −0.00151 |

"More capacity" is 7 leaves, at least 1,500 runners per leaf and 300 rounds,
against 3 leaves, 5,000 and 100. The price target makes the intervals about a
third narrower at the same recipe and lets the larger recipe work; every
larger-recipe row beats its twin by about its net figure.

### A capacity warning, and the check for it

The last row is not information. A correction that sees only the generalist's
own inputs gains −0.0015, so the 400-tree screen fits used as the reference in
all of these screens are under-fitted, and any correction might be repairing
that instead of adding something. `native_check.py` tests this on the one
native-capacity full 063 forecast that exists (2025Q3, 3,535 races, 3,734
trees): apply each correction to the screen forecast and to the native one.

| Correction (prices, 2022Q3, more capacity) | On the screen forecast | On the native forecast |
|---|---:|---:|
| Own horse | −0.00241 | −0.00229 |
| Market gap | −0.00196 | −0.00189 |
| Same-day bias | −0.00054 | −0.00051 |
| The generalist's own inputs | −0.00079 | −0.00036 |
| Tracking | −0.00066 | −0.00044 |
| Ability in pounds | −0.00051 | −0.00022 |

One quarter, so each row is noisy, but the pattern is clear. The own-horse,
market-gap and same-day corrections carry over in full: they are information.
The own-inputs correction loses half, and tracking and ability lose a third to
a half: that part is capacity. The 10 November result-taught corrections also
carry over in full (own horse −0.00188 on screen, −0.00186 on native).

**Consequence for the cheap screens.** A twin controls for what a fit gains
with unrelated inputs. It does not control for a real input that is correlated
with the generalist's own inputs and so can stand in for missing capacity.
Large-recipe screens should be read against a native-capacity reference where
one exists.

### The stack rebuilt

Six groups (own horse, market gap, residual, second wave, same-day bias,
tracking), each a price-taught, larger-recipe correction, combined on the pool
with weights fitted on earlier quarters. The own-inputs correction is left out.

| Forecast | Race log loss | Against one full 063 fit |
|---|---:|---:|
| One full 063 fit | 1.82598 | — |
| Three-seed average | 1.82471 | −0.00127 |
| 9 November stack | 1.82304 | −0.00295 |
| 10 November stack with four error corrections | 1.81941 | −0.00658 |
| **Rebuilt stack** | **1.81665** | **−0.00934 [−0.01200, −0.00686], 4/4** |

Against the 9 November stack it is −0.00639 [−0.00875, −0.00406], 4/4; per
quarter −0.0030, −0.0087, −0.0108, −0.0037. The gap to Betfair SP falls from
0.0799 to 0.0719.

On the native 2025Q3 forecast the same weighted correction gives −0.00233
against −0.00292 on the screen forecast, so about four fifths carries over.
The own-horse and market-gap pair alone gives −0.00355 [−0.00601, −0.00089] on
the native forecast (−0.00369 on screen), with the interval below zero on a
single quarter. That pair is the part to rely on.

### Second pass: what is left

`second_pass.py`, same races, divergence from BSP per 1,000 runners:

| Runners | Three-seed average | Rebuilt stack |
|---|---:|---:|
| First run in the database | 17.6 | 11.9 |
| Off more than 120 days | 13.8 | 8.7 |
| Card shows runs the database lacks | 17.6 | 14.3 |
| One earlier run | 12.5 | 12.5 |
| Five or more earlier runs | 8.0 | 8.0 |
| All | 9.4 | 8.8 |

The corrections fixed what a stable's record can tell: debutants and horses
back from a break. They did nothing for horses with one run or for exposed
horses, which are 69% of runners and now hold most of what remains. On the
market's timeline the rebuilt stack is still between the evening-before price
and the 08:00 price (1.6336 against 1.6540 and 1.6171 on 631 races).

### Sale price, judged by accuracy this time

The ledger dropped sale price because the market fully prices it. That is a
betting verdict; the model does not have the input and is weakest on
debutants. `sale_inputs.py` matches pre-racing sales (foal, yearling,
breeze-up; never horses-in-training) by sire, dam and foaling year, and adds
the sire's record at earlier sales. Debut win rates do rise with price, 5.6%
to 12.2% across quintiles. As a correction it is null in every arm (−0.0002 at
best, not better than its twin). The generalist already has the equivalent
through sire and trainer inputs. Sale price is dead for accuracy as well.

## 6. Has the form worked out? (11 November)

The second pass left exposed horses untouched, so the next input was aimed at
them. A rating system rates a race by its runners as they were on the day. A
handicapper also asks what those runners did next. Nothing in the store does.

`franking.py`: for a runner today and each of its last three races, take every
other runner in that race and its runs strictly after it and strictly before
today, and average their residuals against the model (rivals beaten, win,
market gap). Rivals that finished ahead of our horse and rivals behind it are
kept apart. 33 inputs. Tests confirm that changing today's or later results
leaves today's inputs unchanged.

| Franking, net on the three-seed average | Change | Against its twin | Quarters |
|---|---:|---:|---:|
| Results, rows from 2024Q3 | −0.00214 [−0.00305, −0.00123] | −0.00227 | 4/4 |
| Prices, rows from 2022Q3 | −0.00179 [−0.00242, −0.00119] | −0.00173 | 4/4 |
| Prices, 2022Q3, more capacity | −0.00265 [−0.00383, −0.00152] | −0.00254 | 4/4 |

It is as large as the own-horse group and mostly separate from it: in the
rehearsal below its fitted weight beside the other six corrections is 0.83–0.96.

## 7. The registered read on unseen races

`holdout.py` and `holdout_score.py`. The benchmark's prepared inputs were
rebuilt for every date to 6 November 2026 and checked against the benchmark's
own file: all 193 inputs equal on all 700,639 development rows. The standard
loader refuses the benchmark's store because feature code in the working tree
has changed since 9 November, so the frozen store was read directly; the
parity check is the evidence that this read is the same. Seed-42 forecasts for
2026Q2–Q4 came from the same purged quarterly refits.

**Rehearsal on development races only** (1 January to 3 June 2026, 5,136
races), to prove the scoring path before using the hold-out:

| Added to one seed-42 fit | Change | z (clustered by date) |
|---|---:|---:|
| Own horse and market gap | −0.00475 | −3.4 |
| All seven (own horse, market gap, residual, second wave, franking, same-day, tracking) | **−0.01146** | −5.6 |

Gap to Betfair SP on those races: 0.0803 to 0.0689. By race type the seven
give −0.0111 (Flat), −0.0143 (Hurdle), −0.0082 (Chase).

**Registered before the read** (`holdout/registration.json`, with code
hashes): primary, the seven-group stack against the seed-42 base on every
complete race from 11 June 2026 to the store's last date, pass at one-sided
p < 0.01 with the standard error clustered by date; secondary, the own-horse
and market-gap pair; corrections and weights fitted walk-forward on earlier
quarters only. `score` refuses a second run.

### Result: passes

Read once, 11 November (`holdout/receipt.json`). 5,761 complete races on 149
dates, 11 June to 6 November 2026. Where the new refit overlaps the
benchmark's cached forecasts they agree to 2e-14.

| Added to one seed-42 fit | Race log loss | Change | Standard error | z | One-sided p |
|---|---:|---:|---:|---:|---:|
| Nothing (base) | 1.83368 | — | | | |
| Own horse and market gap (secondary) | 1.82766 | −0.00602 | 0.00134 | −4.5 | 0.000004 |
| **All seven (primary)** | **1.82458** | **−0.00910** | 0.00189 | **−4.8** | **0.0000007** |

The primary passes its registered bar (p < 0.01) by a wide margin. The effect
is four fifths of the rehearsal's (−0.0115), the shrinkage expected from
choosing on development races.

| | Races | Pair | All seven |
|---|---:|---:|---:|
| 11–30 June 2026 | 774 | −0.0095 | −0.0148 |
| 2026Q3 | 3,524 | −0.0056 | −0.0057 |
| October to 6 November 2026 | 1,463 | −0.0053 | −0.0144 |
| Flat | 4,394 | −0.0073 | −0.0098 |
| Hurdle | 906 | −0.0001 | −0.0049 |
| Chase | 461 | −0.0053 | −0.0107 |

Gap to Betfair SP on the 5,673 priced races: 0.0863 for the base, 0.0807 with
the pair, 0.0778 with all seven, so a tenth of the gap is closed.

Each group alone at unit weight (descriptive):

| Group | Hold-out | z | Rehearsal |
|---|---:|---:|---:|
| Own horse | −0.0042 | −4.1 | −0.0029 |
| Market gap | −0.0029 | −3.8 | −0.0028 |
| Residual | −0.0020 | −3.6 | −0.0022 |
| Tracking | −0.0014 | −3.8 | −0.0015 |
| Second wave | −0.0012 | −2.2 | −0.0017 |
| Franking | −0.0013 | −1.6 | −0.0041 |
| Same-day bias | −0.0001 | −0.2 | −0.0005 |

- Own horse, market gap, residual and tracking hold at their development
  size or better.
- **Franking fell to a third** of its development size and is not significant
  alone. It was the newest group, found last and read on development once; it
  is the one most likely to have been flattered. Treat it as unconfirmed.
- **Same-day bias is nil** on these races.
- Hurdles gain least, and the pair gives nothing there. ATR tracking coverage
  was 25–30% in July to October 2026 against about 70% in development, which
  weakens the base and the tracking correction in this period.

What the read does and does not show. It confirms that the corrections add
information on unseen races, with every choice fixed beforehand. The base is a
single screen-capacity fit, so it does not measure the gain over a
native-capacity generalist; the 2025Q3 native check (section 5) says about
four fifths carries over, and all of the pair. These races were examined for
other questions in reports 055–057, though not for this one. This read is now
spent: these races must not be used again to tune these corrections.

## 8. Third pass: the disagreement still repeats, run by run (11 November)

`third_pass.py` asks what the corrected forecast still gets wrong.

**Where the divergence from BSP sits now** (per 1,000 runners, 61,017 runners
in races with a full-field 11:00 price):

| Earlier runs in the database | Runners | Three-seed average | Rebuilt stack | 11:00 price |
|---|---:|---:|---:|---:|
| None | 3,831 | 17.8 | 11.3 | 9.3 |
| One | 3,679 | 14.6 | 14.8 | 4.3 |
| Two to four | 10,391 | 11.6 | 9.9 | 4.3 |
| Five to fourteen | 20,458 | 10.2 | 9.6 | 5.8 |
| Fifteen or more | 22,658 | 6.3 | 7.0 | 4.6 |

Second-time-out runners are the worst served and the corrections did nothing
for them. Nothing in the debut run explains the gap (finishing position,
distance beaten, debut price: every correlation with the BSP gap is below
0.08). The 11:00 exchange price itself is unreliable for outsiders (it rates
debut flops 0.33 log units above the model and BSP then takes most of that
back), so the 11:00 column is a rough guide only.

**The disagreement with the market still repeats for the same horse.** The
correlation between a horse's race-centred gap, `log(BSP p / forecast p)`, and
its gap at its previous run:

| Forecast | All | 5–14 runs | 15+ runs |
|---|---:|---:|---:|
| Three-seed average | 0.255 | 0.280 | 0.241 |
| Rebuilt stack (section 5) | 0.180 | 0.222 | 0.200 |

The own-horse correction removed under a third of it. On the uncorrected
seed-42 forecasts from 2022Q3 the autocorrelation is 0.24, 0.16, 0.11, 0.09,
0.08 at one to five runs back: a fast-fading part on top of a slow one. The
section 4 inputs are one mean per signal with a 365-day half-life, which
cannot see the fast part. A straight-line fit of the next gap explains 6.4%
with those inputs and 9.2% with the last five gaps added.

Calibration is not the problem: a single sharpening exponent fitted on the
other three quarters is 0.98–1.01 and changes nothing.

### Horse state and recent stable form

`horse_state.py`, two groups declared before scoring:

- **Horse state** (35 inputs): the horse's last five race-centred gaps, last
  three win and rivals-beaten residuals, days back to each of the last three
  runs, run-weighted averages, and the same under today's code, surface and
  trip band.
- **Recent stable form** (56 inputs): the section 4 residual and market-gap
  means for trainer, jockey, trainer with jockey, and sire, with half-lives of
  30 and 90 days instead of 365.

Price-taught, larger recipe, net on the three-seed average, 11,905 races:

| Group | Change | Against its twin | Quarters | On the native 2025Q3 forecast |
|---|---:|---:|---:|---:|
| Own horse, section 5 (for comparison) | −0.00256 | | 4/4 | −0.00229 |
| **Horse state** | **−0.00666 [−0.00851, −0.00501]** | −0.00675 | 4/4 | **−0.00470 [−0.00795, −0.00170]** (−0.00521 on screen) |
| Horse state, gaps only (12 inputs) | −0.00549 [−0.00734, −0.00378] | −0.00559 | 4/4 | |
| Horse state, results only (10 inputs) | −0.00111 [−0.00191, −0.00034] | −0.00130 | 4/4 | |
| Horse state, plain lags only (14 inputs) | −0.00641 [−0.00820, −0.00474] | −0.00643 | 4/4 | |
| **Recent stable form** | **−0.00254 [−0.00367, −0.00143]** | −0.00247 | 4/4 | −0.00137 [−0.00310, +0.00047] (−0.00143 on screen) |

- Horse state is two and a half times the old own-horse group, and nine
  tenths of it carries over to the native-capacity forecast, so it is
  information and not capacity repair.
- Almost all of it is the run-by-run price history against the forecast.
  Averages and condition-specific means add little to plain lags.
- The generalist has only two market-history inputs
  (`horse_prior_market_logrel_3_vs_field`, `horse_prior_market_opposition_3_vs_field`),
  both three-run means. That is the gap these inputs fill.

**On the stack.** The six groups of section 5 plus these two, weights fitted
on earlier quarters:

| Forecast | Race log loss | Against the 9 November stack |
|---|---:|---:|
| Rebuilt stack, six groups | 1.81665 | −0.00639 |
| **With horse state and recent stable form** | **1.81307** | **−0.00996 [−0.01275, −0.00730], 4/4** |

Horse state takes a weight of 1.0–1.2 and the old own-horse correction goes
negative (−0.20 to −0.35), so it is replaced, not supplemented. Recent stable
form takes 0.32–0.42. On the new stack the run-to-run correlation of the gap
falls to 0.094 (0.10–0.12 for exposed horses, nil for second-time-out
runners).

### Second registered read on the unseen races

`holdout_state.py`. The hold-out races were read once already (section 7), and
that receipt was known when these groups were built. The groups themselves
were designed and screened on development quarters only. The registration
(`holdout/registration_state.json`) says so and halves the pass mark for the
two looks.

- **Statistic:** race log loss with nine weighted corrections (the seven of
  section 7 plus horse state and recent stable form) minus race log loss with
  the seven, on the same races, same base, same recipe.
- **Pass:** one-sided p < 0.005, standard error clustered by race date.
- **Checks built in:** the seven against the base must reproduce section 7's
  figure, and the corrections for the development part of 2026Q2 must match
  the rehearsal's.
- **Rehearsal** (development, 1 January to 3 June 2026, 5,136 races): nine
  against seven −0.00470, z −4.7; nine against base −0.01616; horse state
  alone −0.00831. The seven reproduced the first rehearsal exactly
  (−0.011458551990635).

**Result: the increment is not confirmed.** One read, 11 June to 6 November
2026, 5,761 races on 149 dates (`holdout/receipt_state.json`):

| Forecast | Race log loss | Change | z | One-sided p |
|---|---:|---:|---:|---:|
| Base, one seed-42 fit | 1.83368 | | | |
| Seven corrections (section 7) | 1.82458 | −0.00910 against base | −4.8 | 0.0000007 |
| Nine corrections | 1.82283 | −0.01085 against base | −4.8 | 0.0000008 |
| **Nine against seven (registered primary)** | | **−0.00175, se 0.00100** | **−1.7** | **0.040: fails the 0.005 mark** |

- Both built-in checks hold: the seven reproduce section 7 to 3e-18, and the
  development part of 2026Q2 matches the rehearsal exactly for all nine groups.
- The increment has the right sign in all three periods (−0.0022, −0.0006,
  −0.0043) and all three race types (Hurdle −0.0049, Flat −0.0012, Chase
  −0.0012), but it is 37% of the rehearsal's −0.0047 and its interval reaches
  zero.
- Registered descriptive rows, each new group alone at unit weight against
  the base: horse state −0.00652 (z −5.1), recent stable form −0.00298
  (z −3.7). Section 7's own-horse group alone was −0.0042 (z −4.1). So horse
  state is a real correction and probably a better own-horse correction than
  the one it replaces, but what it adds **on top of** the seven is not
  established.
- The gap to Betfair SP on 5,673 priced races: 0.0863 base, 0.0778 with
  seven, 0.0760 with nine.

Why the shrinkage is plausible: beside the seven, the fitted combination gives
the old own-horse correction a weight of −0.3 to −0.4 and horse state +1.1 to
+1.2. A difference of two correlated corrections is a fragile thing to
estimate, and the seven already hold the slow part of the same signal. The
development quarters were also where the group was designed. No further read
of these races will be taken for this question.

### Is the forecast needed, and did the money come late? (development only)

`market_history.py`, two groups declared before scoring, same recipe:

| Group | Change | Against its twin | Quarters | On native 2025Q3 |
|---|---:|---:|---:|---:|
| Past closing prices alone, no forecast (8 inputs) | −0.00032 [−0.00099, +0.00036] | −0.00045 | 2/4 | +0.00041 |
| Past late moves: 08:00, 11:00 and one hour to BSP (16 inputs) | −0.00159 [−0.00241, −0.00076] | −0.00149 | 3/4 | −0.00172 [−0.00317, −0.00040] |

- **The forecast is needed.** A horse's last five closing prices on their own
  add nothing: the generalist already knows roughly what price a horse has
  been. What it does not know is where that price stood against its own
  forecast. So these inputs cannot be had as ordinary feature-store columns;
  they need the stored out-of-fold forecast history.
- **Past late money is real but redundant.** Alone it gains −0.0016. Added to
  the eight-group stack it takes a negative weight (−0.4 to −1.2) and the
  stack gets slightly worse (1.81354 against 1.81307). The closing gap at
  earlier runs already contains it.

### One correction model instead of a weighted sum (development only)

Beside horse state the old own-horse correction takes a negative weight, and
the stack's increments shrank on unseen races. `joint_corrector.py` asks
whether a single correction that sees all 359 inputs of the eight groups does
better than eight small ones added with fitted weights. Same teacher, rows and
base; both on the three-seed average with no performance pool.

| Correction | Change, 11,905 races | Quarters | On the native 2025Q3 forecast |
|---|---:|---:|---:|
| Weighted sum of eight | −0.01073 [−0.01370, −0.00796] | 4/4 | −0.00419 [−0.00921, +0.00103] |
| One model, per-group recipe (7 leaves, 300 rounds) | −0.00858 [−0.01043, −0.00688] | 4/4 | −0.00573 [−0.00901, −0.00271] |
| One model, wider (15 leaves, 600 rounds, half the inputs per tree) | −0.01136 [−0.01361, −0.00936] | 4/4 | −0.00682 [−0.01077, −0.00309] |

The wider single model ties the weighted sum: −0.00062 [−0.00173, +0.00062]
against it, better in two quarters and worse in two. It is not a gain in
accuracy. It is the simpler thing to serve (one model, no weights, no
negative coefficient), its interval is a quarter narrower, and it held up
better on the one native-capacity quarter, so it is the form to carry forward
when this is built for serving. The small recipe is too small for 359 inputs.

## 9. Reading the inputs against the racecards (11 November)

A check of whether what the model is given matches what a racecard shows.

**Whole cards read by eye.** `card_audit.py` prints a race as a punter reads
it: each runner's card line, the model's and market's chance, the horse's
earlier runs from the results database and the raw values behind the main
inputs. On the races read, ratings, weights, weight and trip changes, days
off, course records and the prior-market input all agree with the record.
Three things that look wrong and are not:

- `speed_fig_*` is a margin-based figure where lower is better, so the worst
  horse has the highest value and `best_5` is a minimum.
- `lengths_behind` in the results is the gap to the horse in front; the
  features use the cumulative distance (`beaten_lengths`).
- An unrated run is stored as official rating 0 in the database but is
  missing in the store, so a handicap debutant's rating change is missing,
  not +65.

**Cached pre-race racecards against stored results** (`card_vs_result.py`, 18
card dates, 5,235 matched runners). Every large difference is the card being
an early snapshot, not the result being wrong:

| Field | Differ | What it is |
|---|---:|---|
| Field size | 39% | The card counts declared runners; the result has those that ran |
| Draw | 54% before 21 June, 0–8% after | The old card scraper stored cloth numbers. The results are right: of 29,733 Flat handicaps only 0.16% have draws in weight order |
| Going | 10% | The card has the going as declared; the result has it on the day |
| Weight | 9% | The card had 0 for 479 runners |
| Jockey | 2.2% | Changes of rider |
| Form, days off | 0.6% | The card was captured before the horse's latest run; the result has the race-day form |
| Career runs | 80% | The results' `horse_runs` counts the race itself. It is in the model's excluded list |
| Trainer, age, class, distance, surface | under 0.3% | |

So the training rows hold race-day values, which is right for a forecast made
at the off and comparable with BSP. A forecast made in the morning would see
the declared field and declared going instead; that difference is not
measured here. The trainer names are already normalised: only nine yards
change name in a way that restarts their history (8,583 runs, 1.1%).

**What the page has that the model does not.** The raw Sporting Life page
carries the owner, the jockey's claim and each horse's previous runs,
including runs the database lacks. None of that is kept historically (raw
pages are retained from 4 November 2026 only). A horse's profile page lists
its foreign runs with position, field size, price and comment.

### Form figures of the runs the database lacks

The card's form string shows how the horse finished in the runs the database
does not hold; the store gives the model only their number. On the scored
quarters, 7,004 runners (6.4%) have such runs:

| Best finish among the missing runs | Runners | Won | Stack expected | Betfair SP expected |
|---|---:|---:|---:|---:|
| Won | 2,887 | 499 | 443 | 462 |
| Second | 1,595 | 204 | 206 | 212 |
| Third | 840 | 79 | 87 | 83 |
| Fourth to sixth | 901 | 68 | 75 | 71 |

`hidden_form.py` turns those figures into ten inputs (wins, places, failures
to finish, best, mean and latest figure, against the field).

| Hidden form | Change | Against its twin | Quarters |
|---|---:|---:|---:|
| Prices, rows from 2022Q3 | −0.00057 [−0.00089, −0.00027] | −0.00059 | 4/4 |
| Prices, 2022Q3, more capacity | −0.00068 [−0.00113, −0.00027] | −0.00066 | 4/4 |
| Added to the eight-group stack | −0.00032 (1.81307 to 1.81276), weight 0.4–1.2 | | |
| On the native 2025Q3 forecast | −0.00020 [−0.00069, +0.00024] | | |

Small, consistent, and from data already held. It needs no forecast history,
so it belongs in the feature store as ordinary inputs. Development only; by
the section 8 lesson expect less on unseen races.

### The worst misses, read as cards, do not generalise

The 18 winners the eight-group stack most under-rated against Betfair SP
(horses with at least two earlier runs) repeat a handful of stories: a mark
well below a recent high after a run of poor efforts, a handicap debut after
quick runs at huge prices, a new stable after a long absence, first-time
headgear, a return to a course where the horse had run well. Several of these
are not among the 193 inputs, so each was measured on all 109,593 priced
runners of the scored quarters:

| Circumstance | Runners | Won | Stack expected | z |
|---|---:|---:|---:|---:|
| Handicap mark 8 lb or more below its six-run high | 13,014 | 1,014 | 1,021 | −0.2 |
| Handicap debut | 6,149 | 578 | 597 | −0.9 |
| Handicap debut, last three prices all 33/1 or longer | 1,234 | 57 | 51 | +0.9 |
| Different trainer from the last run | 4,030 | 353 | 345 | +0.5 |
| Off 200 days or more, same trainer | 5,150 | 533 | 547 | −0.7 |
| First-time headgear | 8,821 | 783 | 789 | −0.3 |
| First-time headgear and mark 8 lb down | 1,401 | 106 | 93 | +1.5 |
| Three poor runs, then back where it was placed before | 3,153 | 197 | 187 | +0.8 |
| Irish race, different trainer | 1,045 | 78 | 66 | +1.6 |

Nothing reaches two standard errors. The model already prices these
circumstances correctly on average; the worst misses are the gambled horses
among them, and the 11:00 price usually had not moved either (0.019 to 0.092,
0.038 to 0.163 in two of the cases read). Picking patterns out of the worst
misses selects on the outcome. This line is closed.

## 10. The corrections against full-capacity models (12 November)

Every figure above is against 400-tree screen fits, and one full-capacity
forecast existed. `native_folds.py` fits the seed-42 full 063 at native
capacity for the other three scored quarters (same purged refits, about 40
minutes each) and adds the saved corrections, unchanged, to both forecasts on
the same 11,905 races.

| Forecast | Race log loss |
|---|---:|
| Full 063, screen capacity, one seed | 1.82598 |
| Full 063, native capacity, one seed | 1.81937 (−0.00662 [−0.00812, −0.00514], 4/4) |
| Native with the weighted stack of eight | 1.81009 |
| **Native with the one joint correction model** | **1.80915** |

| Correction | On the screen forecast | On the native forecast | Carried over | Quarters |
|---|---:|---:|---:|---:|
| Weighted stack of eight | −0.01074 | −0.00928 [−0.01220, −0.00655] | 86% | 4/4 |
| **One joint model** | −0.01138 | **−0.01022 [−0.01243, −0.00825]** | 90% | 4/4 |
| Horse state alone | −0.00669 | −0.00604 [−0.00790, −0.00437] | 90% | 4/4 |
| Market gap alone | −0.00275 | −0.00243 [−0.00359, −0.00127] | 88% | 4/4 |
| Own horse (old) alone | −0.00259 | −0.00231 [−0.00381, −0.00094] | 89% | 4/4 |
| Recent stable form alone | −0.00255 | −0.00224 [−0.00338, −0.00113] | 88% | 4/4 |
| Residual alone | −0.00181 | −0.00163 [−0.00235, −0.00091] | 90% | 4/4 |
| Second wave alone | −0.00133 | −0.00124 [−0.00194, −0.00057] | 93% | 4/4 |
| Tracking alone | −0.00135 | −0.00112 [−0.00159, −0.00068] | 83% | 4/4 |
| Same-day bias alone | −0.00045 | −0.00042 [−0.00077, −0.00007] | 93% | 4/4 |

- **The corrections are not capacity repair.** Nine tenths of the gain
  survives on a model with nine times the trees, in every quarter, for every
  group. The native-capacity gain from the corrections (−0.010) is larger than
  the gain from native capacity itself (−0.007), and the two add.
- **The joint model is the better form at serving capacity**: larger, with a
  narrower interval, and no fitted weights.
- These are the development quarters on which the corrections were designed,
  and the unseen-race read confirmed the seven at −0.0091 but not the two
  later groups as an addition. A fair expectation on unseen races at native
  capacity is the confirmed figure scaled by what carries over, about −0.008.
- The corrections were fitted from the screen forecast's offset. Refitting
  them from native forecasts is untested and could do slightly better.

### A larger joint correction, judged on the native forecasts

With native forecasts for four quarters, a bigger correction can be told from
capacity repair: it has to gain on the native forecast. `joint_variants.py`,
variants declared before scoring, each a single price-taught model (15
leaves, 600 rounds) added to the same native forecast:

| Joint correction | Inputs | On screen | On native | Native race log loss | Against the eight-group joint model |
|---|---:|---:|---:|---:|---:|
| Eight groups (section 10) | 359 | −0.01138 | −0.01022 [−0.01243, −0.00825] | 1.80914 | |
| **Plus franking, hidden form and past late moves** | 416 | −0.01384 | **−0.01261 [−0.01507, −0.01033]** | **1.80676** | **−0.00238 [−0.00333, −0.00149], 4/4** |
| Plus the generalist's own inputs as context | 608 | −0.01496 | −0.01334 [−0.01575, −0.01106] | 1.80603 | −0.00312 [−0.00424, −0.00202], 4/4 |
| Eight groups, deeper recipe (31 leaves, 1,000 rounds) | 359 | −0.01291 | −0.01138 [−0.01394, −0.00908] | 1.80799 | −0.00115 [−0.00174, −0.00054], 4/4 |
| **More groups and the deeper recipe together** (added after the rows above were read) | 416 | −0.01596 | **−0.01431 [−0.01705, −0.01186]** | **1.80506** | **−0.00409 [−0.00532, −0.00289], 4/4** |

- The three extra groups add −0.0024 inside the joint model, in every
  quarter, and 91% of the screen figure carries to native. As separate
  corrections they looked weaker or redundant (past late moves took a
  negative weight in the weighted sum); a single model uses them better.
- Context adds a further −0.0007 [−0.0014, −0.0001], better in three quarters
  of four, for 192 more inputs. Marginal; not worth carrying unless it holds
  elsewhere.
- A deeper recipe on the original eight groups adds −0.0012 in every
  quarter, 88% carried to native. The two steps together
  (`more-groups-deeper`, a fifth variant added after these were read) add up
  almost exactly: −0.0143 on native, 90% of the screen figure, the best
  development forecast at 1.80506.
- By quarter the 416-input model gains −0.0084, −0.0125, −0.0184, −0.0127 on
  the native forecast.

**Where the best forecast now stands** (native with the eight-group joint
model, same races as section 2):

| | Start of this study | Now |
|---|---:|---:|
| Gap to Betfair SP, 11,902 races | 0.080 | 0.064 |
| Behind the 08:00 exchange price, 6,194 races | 0.0215 | 0.0076 |
| Behind the 11:00 exchange price, 7,292 races | 0.041 | 0.028 |

## 11. Fourth pass: the loss to the 08:00 price is in newcomers (12 November)

`fourth_pass.py`, on the 6,194 scored races with a full-field 08:00, 11:00, one-hour and BSP price
(50,371 runners). The forecast is the native full 063 with the `more-groups-deeper` correction.

| Forecast | Race log loss |
|---|---|
| Native full 063 | 1.74512 |
| With the joint correction | 1.73293 |
| 08:00 exchange price | 1.72901 |
| 11:00 | 1.70904 |
| One hour before the off | 1.69863 |
| BSP | 1.67810 |
| Equal geometric mix of forecast and 08:00 price | 1.71732 |

- **Handicaps are level with the 08:00 price**: chase +0.0002, flat +0.0001, hurdle +0.0010. The
  deficit of 0.0039 sits in flat non-handicaps (+0.0172, 88% of it) and in races whose median
  runner has at most one earlier run (+0.0378, all of it). Hurdle non-handicaps are ahead (-0.0091).
- **Where the two disagree, the result lies between them.** Experienced horses (5+ runs) the 08:00
  price rates much lower than the forecast (log gap below -0.5): 241 winners against 312 forecast
  and 158 priced, and BSP moves 46% back towards the forecast. Much higher: 237 winners against 163
  and 328. The mix beats the 08:00 price by 0.0117 and the forecast by 0.0156, so each holds
  information the other lacks. Against BSP the mix is worse than BSP (+0.011), as before.

### Owner and newcomer inputs: both null inside the joint correction

Two groups declared before scoring, built like the trainer and sire error inputs (shrunk, 365-day
decayed residual and market-gap means on strictly earlier dates):

- `owner` (`owner.py`): owner names from the Racing TV archive linked to 710,224 starters since
  September 2020 (34,170 owners). Keys: owner, owner with trainer, owner with horse state.
- `newcomer` (`newcomer.py`): dam; sire, dam's sire and jockey with horse state (debut, back from a
  break, fit); trainer with state and age band.

| Group | Alone, screen (vs twin) | Added to `more-groups-deeper`, native |
|---|---|---|
| owner | -0.0022 [-0.0035, -0.0010], 3/4 (-0.0023) | +0.00017 [-0.0003, +0.0007], 2/4 |
| newcomer | -0.0049 [-0.0064, -0.0036], 4/4 (-0.0050) | -0.00026 [-0.0007, +0.0001], 2/4 |

Each carries real information alone and none beyond what the horse-state, trainer-by-state and
franking inputs already give the joint model. A dam's earlier runners include the horse's own past
runs, which is most of why `newcomer` looked large alone. Not adopted. The lesson from the second
hold-out read holds again: a group's gain alone says little about its gain on a stack that holds a
correlated correction; judge additions inside the joint model on native forecasts.

### Exact age from the foaling date: a look-ahead leak

`foal_age.py`, declared before scoring: age in days, against the field, and the foaling day of the
year. Alone -0.0015 [-0.0021, -0.0010], 4/4; inside `more-groups-deeper` on native -0.0016
[-0.0021, -0.0010], 3/4 (1.80350). The foaling day of the year ranked among the top inputs, which
prompted a look at the dates:

- 22% of horses have a foaling date of 1 January, a placeholder in the Sporting Life profile.
- Every profile was fetched in 2026 (70,768 in June), long after most of the races.
- The placeholder is commoner the shorter the career: 35% of horses with one run, 28% with three or
  four, 11% with twenty or more.
- Placeholder horses win less often than BSP implies in every age band: two-year-olds 104 winners
  against 133 priced (-22%), ages four and five 419 against 485 (-14%), all ages -7% to -22%. Horses
  with a real date win 1-3% more often than priced.

An input that beats BSP by that much from a data-quality flag is not information available before
the race: whether the profile holds the real date depends on how the career went. Rejected, and the
1.80350 figure is void. Nothing in the model or the store reads `foaled` today; the rule from here
is not to build inputs from it. The same mechanism as the `sex` leak of October.

Also closed in this round: re-scaling the forecast by the field's experience gains at most 0.0008
fitted in-sample (calibration in newcomer races is sound).

### Exact age again, on Racing TV dates and a leak-safe design: null (13 November)

The store rebuilt on 13 November (`fe_signature` a92d85a28e586076) takes foaling dates from Racing
TV. Audit of the raw `foaled` column:

- No 1 January placeholders; the dates equal the old source on all 65,598 horses where that had a
  real one. 8,436 old placeholders now have a date and 7,469 have none.
- 11% of horses have no date: 22% of one-run horses against 8% of those with twenty or more runs,
  and 38% of horses first seen in 2026 (the backfill is unfinished).
- Runners without a date win 12% less often than BSP implies (1,100 against 1,248), so a missing
  value still separates runners.
- The store's own inputs (`profile_exact_age_days`, `profile_foaling_day_of_year`,
  `rtv_profile_foaled_abroad`) are filled for 0.0% of rows in every year, because a profile counts
  only if captured before the race and all were captured in November 2026. No model can be trained
  on them from history.

`foal-age-clean`, declared before scoring: the same six inputs from the raw date, given only in
races where every runner has a date (74% of scored runners) and blank for the whole field
otherwise, so missingness cannot separate runners.

| Group | Alone, screen | vs twin |
|---|---|---|
| foal-age (old dates, placeholder leak) | -0.0015, 4/4 | -0.0016 |
| foal-age-clean | -0.00002 [-0.0004, +0.0004], 1/4 | -0.00003 |

Exact age adds nothing once the leak is removed: the whole earlier gain was the placeholder. Age in
whole years is enough. The capture-time rule in the store costs nothing, and the foaling-date
backfill is not worth finishing for accuracy.

The dated study stores (`modelling_feature_store_20261109`, `_20261111`) were deleted on
12 November; `frame.py` now reads identity and card columns from the current store when the study
store is absent.

## 12. What the selection dropped, and two new presentations (13 November)

The 12 November selection kept 87 of 1,317 store columns. Read by family:

| Family | Candidates | Kept | Reading |
|---|---|---|---|
| Pace and running style | 37 | 0 | Built from At The Races data: 0% of rows before 2023, 17-25% in 2023, about two thirds since |
| Going | 53 | 2 | Seven buckets from 193 raw descriptions; win rates per bucket |
| Distance | 58 | 3 | Win rates per band |
| Stewards reports (`bha_`) | 123 | 0 | Matches the earlier null |
| First-time headgear and equipment | 12 | 1 (`has_headgear`) | Matches the earlier null |
| Rider claims and race eligibility | 4 | 0 | Dropped in the first two steps |
| Race context (field size, class, handicap flag, distance, race type) | - | 0 | Almost every kept input is the runner against its field |

Two presentations of information already held, each declared before scoring and screened as a
correction on the stored forecasts against a permuted twin:

- `pace-comments` (`pace_comments.py`): early position scored from the Racing TV close-up comment,
  which exists for every result since September 2020 (96% of 710,224 runs scored: 13% led, 31%
  prominent, 21% midfield, 23% held up, 12% rear). Horse style, likely leaders, pace pressure, lone
  leader, and each course's leader bias from earlier dates.
- `matched-form` (`matched_form.py`): past performances weighted by how closely each run's distance,
  going, course and class match today's, on a continuous going scale read from the raw descriptions
  (99.5% readable).

| Group | Inputs | Alone, screen | vs twin |
|---|---|---|---|
| pace-comments | 13 | -0.00027 [-0.0007, +0.0002], 3/4 | -0.00024 [-0.0008, +0.0003] |
| matched-form | 39 | -0.00017 [-0.0009, +0.0006], 2/4 | -0.00014 [-0.0009, +0.0007] |

Both null. The pace family was starved of history, but full history does not rescue it: running
style and pace set-up add nothing to the forecast. The coarse going buckets and per-category win
rates lose nothing measurable either: a continuous, similarity-weighted version does no better.
Neither was taken to the full-capacity test.

## 13. Are the kept inputs working? (13 November)

**Race context, tested directly.** `python -m src.modelling admit`: the 87-input model with 21 raw
context columns added back (distance, going, race type, surface, handicap, novice/maiden/bumper,
prize money, course shape), on the 11,632 confirmation races at reduced capacity. Against the same
columns shuffled: -0.00002 [-0.0013, +0.0014]. Against the 87 alone: +0.0011, 1/4. Context adds
nothing beyond what the runner-against-field inputs already carry. (Field size could not be twinned
because the model checks it against the context frame; `admit` now refuses such candidates.)

**Profile of the 87** (`kept_inputs_audit.py`, output `kept_inputs_audit.csv`): fill rate by year,
repeated values, zeros standing for no history, variation inside a race, tails, and win rate across
within-race fifths. Three inputs looked wrong:

- `weight_vs_field` and `weight_change_vs_field` lost data in the store rebuilt on 13 November. They
  are now claim-adjusted and blank wherever the rider's claim is unverified: 31% of 2020-2023
  runners (6% in 2025), against 0% in the store the selection used. `weight_change_vs_field` fell
  from 90% to 48% filled in 2021.
- `speed_fig_best_5` is exactly zero for 36% of runners: every winner scores 0 lengths behind.
- `hidden_form_mean` is filled for 36% of 2021 runners and 6% of 2025 runners: early on it mostly
  describes ordinary runs from before the database starts.

Sound on inspection: Racing TV sectional inputs empty before 2023 (source start); rating inputs at
zero for newcomers (the true starting value); draw inputs flat alone (they work with course).

**Repairs measured inside the model** (`input_repairs.py`): the 87-input recipe refitted on the
confirmation folds with a few columns changed or added, against the unchanged model (1.82894).

| Arm | Change [95% interval] | Quarters better |
|---|---|---|
| Weight inputs as the 13 November store has them | +0.00092 [-0.0004, +0.0023] | 2/4 |
| Claim-adjusted weight where known, plain weight otherwise | +0.00076 [-0.0005, +0.0020] | 1/4 |
| Margin figures with winners credited their winning distance (5 added) | +0.00075 [-0.0011, +0.0025] | 1/4 |
| Hidden form blanked for horses first seen before 15 March 2021 | +0.00035 [-0.0008, +0.0014] | 2/4 |
| The two kept margin inputs replaced by winner-credited versions (added after the row above was read) | +0.00069 [-0.0008, +0.0021] | 1/4 |

None differs from the unchanged model beyond the 0.001-0.002 noise between fits. The lost weight
data costs nothing measurable and the fallback recovers nothing measurable; the two oddities in the
margin and hidden-form inputs are real but harmless.

Afterwards, at the user's request, the weight comparisons were given a fallback in the feature code
(`burden_weight` in `src/feature_semantics.py`: claim-adjusted where the claim is verified, allocated
weight otherwise), restoring full coverage. It sits in the working tree with the other session's
uncommitted rider-claims work and needs a store rebuild.

**Correction to the margin arms.** The two margin rows above read the store's `lengths_behind`,
which is the gap to the horse in front; the distance behind the winner is `beaten_lengths`. They are
void. Rerun on cumulative distances, with a control that uses the same simplified rolling window and
leaves winners at zero:

| Arm | Change vs the unchanged model | Quarters better |
|---|---|---|
| Kept margin inputs rebuilt, winners credited | +0.00099 [-0.0004, +0.0023] | 1/4 |
| Kept margin inputs rebuilt, winners at zero (control) | +0.00065 [-0.0006, +0.0019] | 1/4 |
| Credit against control | +0.00034 [-0.0007, +0.0015] | 2/4 |

Most of the apparent cost is this script's approximation of the store's rolling window; the winner
credit itself is neutral. At the user's request it was made in the feature code
(`add_speed_figure_features`: a winner scores minus the distance to the next finisher, capped at 20
lengths; test in `tests/test_margin_history_repair.py`). Like the weight fallback it is uncommitted
and needs a store rebuild; neither is expected to change accuracy.

## 14. Two suggestions from the user: race-relative views and "laying one out" (13 November)

**Race-relative views.** The store compares a runner with its field by difference from the field
mean, z-score, rank percentile and gap to the best other rival. `rival-structure`
(`input_repairs.py`) adds four more views for ten kept ability measures (40 columns): the average
pairwise win share against each rival, the log softmax share of the field, the gap to the mean of the
three best rivals, and the count of rivals within half a standard deviation. Added to the 87 inputs:
−0.00018 [−0.0025, +0.0021], 3/4 quarters. Null as an addition. Replacing the existing views with
these was not tested.

**Plot races.** The suggestion: some owners and trainers do not ask for a horse's best in qualifying
runs, to earn a lenient mark for a target race. `plot_races.py` keys the shrunk past residual and
closing-price gap of each trainer and each owner by the horse's handicap situation (handicap debut,
below last winning mark, eased 2lb or more since last run, other handicap, not a handicap), by
whether its last three runs were all at 16/1 or longer, and by today's switch (class drop, trip
change of two furlongs or more, both).

| Test | Change | Quarters better |
|---|---|---|
| Alone, as a correction on stored forecasts (screen) | −0.00245 [−0.0036, −0.0012] | 4/4 |
| Same against shuffled twin | −0.00247 | 4/4 |
| Added to the best joint correction, native forecasts (`with-plot` vs `more-groups-deeper`) | +0.00019 [−0.0004, +0.0007] | 1/4 |

Real alone, nothing inside the joint correction: the third group with this pattern after owner and
newcomer. The existing trainer keys (handicap or not, code, track, freshness) and the horse's own
price-gap history already carry it. Connections that outperform the forecast in these situations
are found by the corrections as they stand.

## 15. Four more ideas inside the corrections (13 November)

All judged on the four development quarters (11,905 races), native forecasts.

**Second-round corrections** (`second_round.py`). The best joint correction was walked forward over
2023Q3-2025Q2 too, so every race from 2023Q3 has a corrected out-of-fold forecast. Trainer, jockey,
sire and horse records and the run-by-run horse state were rebuilt against that corrected forecast
(84 inputs) and a second correction fitted on top.

| Comparison on native | Change | Quarters better |
|---|---|---|
| Round two vs round one | -0.00039 [-0.00074, -0.00004] | 4/4 |
| Shuffled twin vs round one | -0.00016 [-0.00036, +0.00005] | 4/4 |
| Round two vs twin | -0.00023 [-0.00065, +0.00019] | 3/4 |

Round one has taken nearly all of it; 40% of the small gain is a level adjustment the twin also finds.

**Three new input groups**, each alone as a correction on the screen forecast, then all three added
to `more-groups-deeper` (`with-three`, 481 inputs):

| Group | Alone (vs twin) | Quarters |
|---|---|---|
| `filter`: per-horse local-level filter of the price gap, uncertainty growing with days off and youth (`horse_filter.py`, 11 inputs) | -0.00472 [-0.0065, -0.0031] (-0.00491) | 4/4 |
| `debut-choice`: prize of the race chosen for a newcomer against the trainer's habit, and trainer records by that choice, track, jockey and season (`debut_choice.py`, 39 inputs) | -0.00338 [-0.0047, -0.0021] (-0.00337) | 4/4 |
| `market-rating`: Luce rating from past closing prices linked through shared rivals, from 2020-09 (`market_rating.py`, 12 inputs) | -0.00066 [-0.0017, +0.0004] (-0.00068) | 2/4 |
| All three inside the joint correction, native, vs `more-groups-deeper` | -0.00010 [-0.00095, +0.00074] | 2/4 |

Null inside. Two side findings: the rivals-beaten residual has no persistence between a horse's
consecutive runs (lag-one covariance -0.002 of variance 0.084), so only the price gap has a level to
filter; and how good the market thought a horse was in past races carries nothing by itself, only
where it disagreed with the model does. With owner, newcomer and plot races that makes six groups
real alone and null inside: the joint correction at 450 inputs is saturated for information of this
kind (records of entities and horses against the model's own forecasts).

## 16. The 11:00 exchange price as an input (13 November)

New information instead of new views of old information: the Betfair last traded price of each runner
at 11:00 on the day (`research/error_anatomy/morning_price.py`, group `price-1100`, eight inputs: log
share of the field, that share against the model's forecast, price against the favourite, rank, age
of the last trade, number of price changes, move since 08:00, share of the field priced). 95.7% of
runners in the long frame have a price; 10,817 of the 11,905 scored races have one for every runner.
A model using it cannot forecast before 11:00.

| On native forecasts, four dev quarters | Race log loss |
|---|---|
| Native model | 1.81937 |
| Best joint correction (`more-groups-deeper`) | 1.80506 |
| The same with the 11:00 price (`with-1100`) | 1.77915 |

`with-1100` against `more-groups-deeper`: −0.02591 [−0.03015, −0.02201], 4/4 quarters. Alone on the
screen forecast the group is −0.0395 against its shuffled twin, 4/4.

On the 10,817 races priced for the whole field:

| Forecast | Race log loss |
|---|---|
| Best joint correction without the price | 1.78213 |
| Raw 11:00 price | 1.76177 |
| Joint correction with the 11:00 price | 1.75431 |
| Closing price | 1.72286 |

The model with the price beats the raw 11:00 price by −0.00746 [−0.01020, −0.00464], 4/4, and is
still +0.0315 behind the close. Without the price the model is 0.020 behind the 11:00 price; so the
price carries most of the gain, and the model adds about a fifth of the 11:00-to-close distance on
top of it. On the 1,088 races not priced for the whole field the change is −0.0069
[−0.0197, +0.0068], 2/4: no harm where the price is missing.

Not on the hold-out, which stays unread. The user intends a Sporting Life price as the live source
later; its first show sits nearest the exchange price an hour before the off, so that would be a
different (later) forecast time and needs its own history before it can be fitted.

## 17. One race read closely: stable change, discipline, Irish raiders, weight (14 November)

The user asked why Dinoland (Cheltenham 12:35, 14 November) was 9.8% when he became favourite. The card
price the model saw was 7.0, fourth in the betting (11.7% of the book), so the forecast was two points
under the morning market; the rest was later money. His inputs were sound (26 runs, all correct). Three
things about him are in no input, and the user added two more (race fitness, weight carried), so each
was counted on the 11,905 scored races and then screened.

Winners against the stored forecast and the closing price (1.00 = as expected):

| Situation | Runners | Winners | Won / model | Won / market |
|---|---|---|---|---|
| First run for a new yard | 4,012 | 353 | 1.06 | 1.01 |
| Runs 2-4 for a new yard | 9,310 | 935 | 1.02 | 1.02 |
| Settled after a move, form up 0.15+ | 4,711 | 643 | 1.00 | 1.02 |
| Today's race type beats career record by 0.15+ | 1,489 | 272 | 1.10 | 1.10 |
| Chase, 2+ chase wins for this yard | 3,604 | 471 | 0.95 | 0.99 |
| Rating up 15+ in six runs | 4,898 | 560 | 0.96 | 0.99 |
| Irish yard in Britain | 1,128 | 109 | 0.92 | 0.84 |
| Jumps handicap, ran within 21 days | 7,119 | 847 | 0.98 | 1.00 |
| Jumps handicap, off 150+ days | 3,182 | 315 | 1.08 | 1.06 |
| Jumps handicap, top third of the weights | 9,708 | 1,173 | 0.95 | 0.97 |
| Jumps handicap, bottom third of the weights | 7,562 | 698 | 1.03 | 1.01 |
| Chase 3m+, bottom third of the weights | 1,174 | 131 | 1.11 | 1.04 |

The model is not low on horses improved since a stable change. It leans slightly high on top weights
and low on bottom weights. Two groups, declared before scoring: `yard` (`yard_change.py`, 53 inputs:
form by spell for the present yard and in today's race type, rating gains, the trainer's base, and
residuals keyed by trainer with stage of the spell and with home/away) and `burden` (`burden.py`, 16
inputs: weight against the field with trip, discipline and ground, and against weights carried before).

| | Alone, vs twin (screen forecast) | Inside the joint correction (native) |
|---|---|---|
| `yard` | -0.00119 [-0.00228, -0.00017] 3/4 | |
| `burden` | -0.00073 [-0.00123, -0.00026] 4/4 | |
| both (`with-yard-burden`) | | +0.00003 [-0.00073, +0.00080] 2/4 |

Both are small alone and nothing inside: 1.80508 against 1.80506. The store's `region` column is one
constant ("UK") for every race; a trainer's base has to be derived from where its runners ran.

## What this means

- The generalist is slow to learn about specific horses and stables. When the
  market rated a horse above the model last time, it was usually right, and
  it will usually do so again. The model's features describe form and class;
  they do not record that the model itself was wrong about this horse.
- These inputs use past closing prices. They fit the plan's rule (prices of
  **earlier** races are public history; the current race's price is never
  used), and `horse_prior_market_*` already follows the same rule inside the
  generalist. They do not belong to the market tier.
- They explain why earlier residual-style specialists found little: those
  looked for new information in new data. The information here was already in
  the results and prices; what was missing was the comparison with the
  forecast.

## Caveats

- Retrospective, on reused quarters. The families were chosen after examining
  errors on these same forecasts (the half-split in section 3 used the scored
  quarters as its second half). The corrections themselves are walk-forward
  and each beats three twins, but the choice of what to build was informed by
  the scored period.
- The stack's weights and recipe were chosen on the same quarters.
- The forecasts that feed the inputs are screen-capacity seed-42 fits. A
  native-capacity generalist makes different errors, so the sizes may change.
- Serving needs a stored history of honest out-of-fold forecasts for every
  past race, refreshed as results arrive. That is new infrastructure.
- Not run: these inputs inside the generalist. The 10 November Tier 2 result
  suggests corrections are the right form for groups of this size, but own
  horse at −0.002 is large enough for a refit to resolve.

## Next steps

0. **Position after the third round (section 8).** Confirmed: the seven
   corrections of section 7. Not confirmed as an addition: horse state and
   recent stable form (right sign, a third of the development size). Dead:
   plain price history without the forecast, past late moves on top of the
   stack. Structural: one wider correction model ties the weighted sum and is
   the simpler form. The remaining disagreement sits on horses with one to
   four runs and no input tried here predicts it.
   The input audit (section 9) found the training rows sound; the one real
   gap is the runs the database lacks. Their form figures give −0.0006 now;
   their detail (class, price, opposition) is on Sporting Life horse profiles
   for about 9,100 horses and has not been fetched.
1. **Done: confirmed on unseen races** (section 7) **and measured at native
   capacity on four quarters** (section 10: −0.0102 with the joint model, nine
   tenths of the screen figure). What remains before this can serve is a
   stored history of out-of-fold forecasts kept up to date as results arrive.
2. **Fill the hidden form.** 15.6% of hurdlers have runs the database lacks;
   point-to-point and foreign results are the obvious source.
3. **Extend the idea.** Residuals against the model by owner (not collected),
   by trainer and headgear change, by trainer and class move; and a residual
   for the performance/spread model.
4. **Replace `trainer_prev_runs`** with a trailing-window count at the next
   feature rebuild.
5. **Price timing (parked by the user, 13 November).** The 11:00 price is a
   different distance from the off for each race. Two arms to run together when
   taken up: 11:00 with hours-to-the-off as an input, and the price a fixed hour
   before the off (held for 99% of races; also the nearest stand-in for Sporting
   Life's first price).
6. **Big-race entries have no free history.** The BHA API serves
   `races/{year}/{raceId}/{division}/entries` and `/nominations` only for the
   current year, and after a race both return the final declared runners, not
   the earlier entry stages (checked on the 2024 and 2026 Dewhurst). Entries
   could only be recorded forward. Breeze-up times: ledger 21, none published.

## Files

- `research/error_anatomy/frame.py`: scored runners with model, market and
  context.
- `research/error_anatomy/entity_residuals.py`: the inputs and the stages
  `screen`, `long`, `stack`, `combine`, `stress`.
- `research/error_anatomy/extend_oof.py`: out-of-fold forecasts for
  2022Q3–2024Q2.
- `research/error_anatomy/test_entity_residuals.py`: definition and leak tests.
- `research/error_anatomy/market_disagreement.py`, `soft_corrections.py`
  (stages `screen`, `stack`), `native_check.py`, `second_pass.py`,
  `sale_inputs.py`, with tests in `test_soft_and_sales.py`. Outputs:
  `market_disagreement.json`, `soft_corrections.json`, `soft_stack_soft-long-rich.json`
  and its forecast, `native_check.json`, `native_stack_check_soft-long-rich.json`,
  `second_pass.json`.
- Joint variants: `joint_variants.py`; outputs `joint_variants.json` and
  `joint_variants.parquet`.
- Native capacity: `native_folds.py` (`fit`, `check`). Outputs:
  `native/native_2025Q4.parquet`, `native_2026Q1`, `native_2026Q2`,
  `native_folds_check.json`.
- Input audit: `card_audit.py`, `card_vs_result.py`, `hidden_form.py`.
  Outputs: `card_vs_result.json`, `draw_cloth_signature.parquet`,
  `trainer_renames.csv`, `soft_stack_soft-long-rich_hidden.json`.
- Third round: `third_pass.py`, `horse_state.py`, `market_history.py`,
  `joint_corrector.py`, `holdout_state.py`, tests in `test_horse_state.py`.
  Outputs: `third_pass.json`, `third_pass_state.json`,
  `soft_stack_soft-long-rich_state.json` and `_moves.json`,
  `joint_corrector.json`, `holdout/registration_state.json`,
  `holdout/receipt_state.json`. `soft_corrections.py` was edited after each
  registration (new groups and an output tag only), so its current hash
  differs from the registered ones; nothing is committed, so the registered
  versions exist only as those hashes.
- `data/research/error_anatomy_20261110/`: `early_oof_seed42.parquet`,
  `long_inputs.parquet`, `stack_corrections.parquet`, `combine.json`,
  `stack_horse_market-gap_residual_wave2.json`, `long_screen.json`,
  `entity_residuals_screen.json`, `stress_screen.json`, one `*_summary.json`
  per screen, and the logs.
