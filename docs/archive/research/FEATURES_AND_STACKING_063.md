# 063 — Feature use, sub-models and a meta learner (4 November 2026)

**Question.** Does the win model make good use of its inputs, or does it need
more racing knowledge, more race context and more interactions? And which
intermediate targets could feed an ensemble with a meta learner?

**Answer.** The inputs can be used better, but not in the way expected.

- **Interactions:** the model already uses them heavily. Deeper trees add
  nothing measurable.
- **Race context:** adding it directly, including a finer race type, gives no
  measurable gain.
- **Domain-shaped inputs:** these do help when they show something the model
  could not otherwise see. Timeform's rating compared with today's field is the
  clearest case: −0.0029 [−0.0048, −0.0009], better in all four quarters.
- **Training target:** a larger market share in the target gains −0.0020
  [−0.0037, −0.0004], better in three of four quarters.
- **Together:** the better inputs and target gain **−0.0063 [−0.0084, −0.0041],
  better in all four quarters**. That is about 6% of the gap to the market.
- **Meta learner:** it is not worth building yet. Sub-models gained up to
  −0.0016 on today's model, but only −0.0002 once the inputs and target were
  improved. Averaging three fits of the same model gained −0.0010.

**Confirmed on 5 November 2026.** Both candidates passed a registered test at
native capacity on 2025Q3–2026Q2. The 64 extra inputs with a 0.75 market share
gain **−0.00628 [−0.00823, −0.00426] against 057, better in all four quarters**.
Nothing is installed, and the golden hold-out has not been read. The other
figures in this document are from the development screen unless stated.

## How it was measured

| Item | Setting |
|---|---|
| Yardstick | Race log loss of the actual winner; lower is better |
| Baseline | 057 recipe (exploded softmax) on the 129 production inputs |
| Quarters | 2024Q1–2025Q2 only. Confirmation quarters and golden hold-out not read |
| Capacity | Screen: 400 trees at 0.04 |
| Data | 2 November cache, the one behind saved arms 057–062 |
| Direct arms | 12,947 races, 2024Q3–2025Q2. Baseline 1.84977, market 1.74606 |
| Stacked arms | 16,540 races, 2024Q2–2025Q2. Baseline with temperature 1.86557, market 1.76494 |
| Harness check | Reproduced the saved 057 fold to 2.2e-16 |

Ranges are 95% date-clustered intervals. Earlier work put the noise from column
order alone at about 0.002 between feature sets, so smaller differences mean
little unless they also hold across quarters.

## Interactions

Limiting tree depth shows how much the model relies on combining inputs.

| Maximum depth | Race LL change | Quarters better |
|---|---|---|
| 1 (no interactions) | +0.02015 [+0.01691, +0.02333] | 0 of 4 |
| 2 | +0.00846 [+0.00624, +0.01072] | 0 of 4 |
| 3 | +0.00419 [+0.00239, +0.00589] | 0 of 4 |
| 4 | +0.00210 [+0.00056, +0.00369] | 0 of 4 |
| 8 | −0.00075 [−0.00239, +0.00103] | 3 of 4 |

Interactions are worth at least 0.02, and they level off by depth 8. The model
is not short of them, so hand-built multiplied columns are unlikely to help.

## Race context

| Arm | Race LL change | Quarters better |
|---|---|---|
| 8 noise columns (control) | +0.00034 [−0.00125, +0.00204] | 2 of 4 |
| 28 race context columns | −0.00075 [−0.00252, +0.00089] | 3 of 4 |
| The same columns on the wrong races (control) | +0.00067 [−0.00091, +0.00237] | 1 of 4 |
| Past form weighted towards runs like today's | −0.00110 [−0.00292, +0.00074] | 3 of 4 |
| Finer race description, 19 columns | −0.00024 [−0.00186, +0.00147] | 3 of 4 |
| The same columns on the wrong races (control) | +0.00047 [−0.00109, +0.00222] | 2 of 4 |
| Finer description + record in this kind of race | +0.00138 [−0.00033, +0.00305] | 1 of 4 |

The finer description comes from race titles and the runners' ages. It has 14
race kinds, apprentice, amateur and fillies/mares flags, and the youngest and
oldest age in the race. The record arm adds each trainer's, jockey's and horse's
wins against a fair share in this kind of race, from earlier days only.

Race kind does matter to how far the model trails the market, so the idea was
worth testing:

| Race kind | Races | Gap to market |
|---|---|---|
| Hunter chase | 113 | 0.0488 |
| Flat Group/Grade | 336 | 0.0625 |
| Flat Listed | 280 | 0.0690 |
| Chase Group/Grade | 165 | 0.0725 |
| Chase handicap | 1,607 | 0.0780 |
| Hurdle novice | 557 | 0.0794 |
| Hurdle Group/Grade | 164 | 0.0881 |
| Flat handicap | 7,398 | 0.0920 |
| Nursery | 284 | 0.1045 |
| Hurdle handicap | 2,148 | 0.1048 |
| Flat bumper | 364 | 0.1218 |
| Hurdle novice handicap | 337 | 0.1259 |
| Flat maiden | 1,460 | 0.1259 |
| Flat novice | 1,054 | 0.1303 |
| Hurdle maiden | 980 | 0.1305 |

The gap by kind is measured on 19,086 races over all six quarters, so its scale
differs slightly from the direct arms.

Likely reasons the labels add so little:

- 71 of the 129 inputs are already scaled against today's field.
- Timeform ratings and comments are written for this particular race.
- Maidens and novices have the largest gaps, but those gaps may come from
  information the market holds, such as a debutant's reputation. A race label
  cannot supply that.
- Earlier specialist models were also worse: hurdle-only +0.0056,
  Wolverhampton-only +0.0100 (049).

Race kind is not ruled out as useful. The interval reaches −0.0019, and the test
was at screen capacity. It gives no reason to prioritise it, though.

## Domain-shaped inputs

| Arm | Race LL change | Quarters better |
|---|---|---|
| 8 rating-gap columns | −0.00285 [−0.00477, −0.00090] | 4 of 4 |
| Timeform parts only | −0.00167 [−0.00356, +0.00021] | 3 of 4 |
| Mark-at-the-weights parts only | +0.00069 [−0.00100, +0.00243] | 2 of 4 |
| Timeform over mark parts only | +0.00026 [−0.00153, +0.00197] | 1 of 4 |
| Field gaps for 10 other strong inputs (20 columns) | −0.00016 [−0.00194, +0.00162] | 3 of 4 |
| All of the above + context + weighted form (64 columns) | −0.00406 [−0.00646, −0.00193] | 4 of 4 |

`rtv_card_tf_rating` was the gap worth filling. It already allows for the
weights, and only 5% of its variance lies within a race, yet the store had no
form of it relative to the field. The rule from this is narrow: shape an input
where a strong signal is hidden by its scale. Adding field gaps to everything
else did nothing.

## The training target

The first stage trains on a blend of market and result. Production uses 0.5,
and that value has no recoverable audit.

| Market share | Race LL change | Quarters better |
|---|---|---|
| 0.25 | +0.00232 [+0.00055, +0.00421] | 0 of 4 |
| 0.5 (today) | baseline | |
| 0.75 | −0.00204 [−0.00367, −0.00039] | 3 of 4 |
| 1.0 | −0.00064 [−0.00260, +0.00125] | 2 of 4 |

A market price says something about every runner, while a result names only the
winner, so it is a far less noisy label. The curve is smooth with its best point
near 0.75. It was picked from these three values on these quarters, so it needs
confirming.

## Intermediate targets

Each sub-model is a LightGBM on the 129 inputs. It is judged by how much it adds
on top of the baseline in a walk-forward stack.

| Target | Skill on target (within-race corr) | Gain over baseline | Quarters better |
|---|---|---|---|
| Closing market | n/a | −0.00161 [−0.00288, −0.00026] | 4 of 5 |
| 11:00 market | n/a | −0.00063 [−0.00143, +0.00017] | 5 of 5 |
| Performance figure (pounds against this field) | 0.488 | −0.00059 [−0.00091, −0.00026] | 5 of 5 |
| Market move | n/a | −0.00020 [−0.00104, +0.00059] | 3 of 5 |
| RTV time figure | 0.450 | −0.00012 [−0.00030, +0.00006] | 3 of 5 |
| Placed | 0.365 | −0.00007 [−0.00038, +0.00022] | 2 of 5 |
| Run-by-run form model | 0.424 | −0.00001 [−0.00033, +0.00030] | 2 of 5 |
| Finishing speed | 0.406 | −0.00002 [−0.00013, +0.00007] | 2 of 5 |
| Completed the race | 0.154 | +0.00013 [−0.00008, +0.00037] | 1 of 5 |
| Late gain | 0.267 | +0.00013 [−0.00001, +0.00028] | 2 of 5 |
| Early position | 0.389 | +0.00017 [−0.00018, +0.00052] | 2 of 5 |
| Handicap mark change | n/a | +0.00039 [+0.00007, +0.00068] | 2 of 5 |
| Noise (control) | n/a | +0.00006 [−0.00002, +0.00014] | 1 of 5 |

Being able to predict a target does not make it useful. Running-style targets
are predictable, but what they carry is already in the model.

**The run-by-run model** has one row per pair of today's race and one past run,
for up to four runs within two years: 2,011,556 rows for 568,544 runners. Each
row holds what happened in that run, how it differed from today, and today's
conditions. The target is today's finish in pounds against this field, and a
runner's score is the mean across its runs. It saw form inputs only and could not
learn which runs matter most, so this rejects this version, not sequence models
in general.

## Ensembles and the meta learner

| Stack over the baseline | Race LL change | Quarters better |
|---|---|---|
| Average of 3 seeds | −0.00097 [−0.00155, −0.00035] | 4 of 5 |
| Closing market head | −0.00161 [−0.00288, −0.00026] | 4 of 5 |
| Closing market head + figure | −0.00148 [−0.00281, −0.00009] | 4 of 5 |
| Head + figure + seeds | −0.00150 [−0.00278, −0.00013] | 4 of 5 |
| Head + figure + seeds + 7 subject models | −0.00100 [−0.00251, +0.00052] | 4 of 5 |
| Every target | −0.00062 [−0.00228, +0.00128] | 3 of 5 |
| Every target + subject models + seeds | 0.00000 [−0.00194, +0.00199] | 3 of 5 |
| 7 subject models on top of the full model | +0.00021 [−0.00045, +0.00080] | 2 of 5 |
| 7 subject models instead of the full model | +0.01830 [+0.01486, +0.02183] | 0 of 5 |
| Head + figure, weights by race type × handicap | −0.00061 [−0.00201, +0.00113] | 3 of 5 |
| Head + figure, weights by race kind | +0.00055 [−0.00114, +0.00251] | 3 of 5 |
| Temperature by race kind | +0.00103 [+0.00038, +0.00169] | 2 of 5 |
| 7 subject models, weights by race kind | +0.00608 [+0.00407, +0.00810] | 0 of 5 |
| Boosted meta learner: head + figure + context | +0.00019 [−0.00059, +0.00100] | 2 of 5 |
| Boosted meta learner: everything + context | +0.00058 [−0.00026, +0.00149] | 3 of 5 |

The seven subject models split the inputs into ability, form, expert, telemetry,
connections, conditions and profile. Alone they score 1.916, 1.980, 1.949, 2.070,
2.035, 2.126 and 2.114. Splitting the inputs loses the interactions that make the
model work.

Adding columns to the stack loses more through fitting noise than it gains. The
weights by race kind were fitted on very little data in the early folds, so their
poor results partly reflect that.

## Do the gains add up?

Same 9,455 races, 2024Q4–2025Q2. Baseline 1.82790, market 1.72089.

| Single model | On its own | + market head and figure |
|---|---|---|
| Baseline | reference | −0.00163 [−0.00305, −0.00010] |
| Target 0.75 | −0.00146 [−0.00352, +0.00058] | −0.00198 [−0.00393, +0.00007] |
| 8 rating columns | −0.00273 [−0.00512, −0.00026] | −0.00392 [−0.00600, −0.00180] |
| 64 columns | −0.00251 [−0.00522, +0.00033] | −0.00409 [−0.00630, −0.00163] |
| 64 columns + target 0.75 | −0.00494 [−0.00747, −0.00233] | −0.00518 [−0.00769, −0.00265] |

Over the full 12,947 races, the 64 columns gain −0.00406, target 0.75 gains
−0.00204, and both together gain −0.00629 [−0.00840, −0.00412], better in all
four quarters. These two add almost exactly.

The market head and the higher market share are the same idea reached two ways.
Once the target carries more market, the head adds only −0.0002.

**What changes against the market:** within-race correlation with the market rises
from 0.8963 to 0.9039. None of the models adds to the market in a blend: all five
sit at about −0.0001 with intervals spanning zero.

## Confirmation

Both candidates and the pass rule were written to `registration_063.json` before
any fit. The arms ran at native capacity with seed 42 on 2025Q3–2026Q2, 11,905
races, against the saved 057 arm. The harness first reproduced 057's 2025Q3 fold
with a largest difference of 0. A shutdown stopped the run after five of the eight
candidate folds, and it resumed from the saved folds on unchanged code and data.

**Pass rule:** lower race log loss than 057, a 95% interval entirely below zero,
and better in at least 3 of 4 quarters. If both pass, keep the 8 columns unless the
64 beat them with their own interval entirely below zero.

| Arm | Race LL | Change against 057 | Quarters better | Gap to market | Result |
|---|---|---|---|---|---|
| 057 | 1.82675 | reference | | 0.08194 | |
| 8 rating columns + share 0.75 | 1.82329 | −0.00346 [−0.00506, −0.00185] | 4 of 4 | 0.07848 | Pass |
| **64 columns + share 0.75** | **1.82048** | **−0.00628 [−0.00823, −0.00426]** | **4 of 4** | **0.07566** | **Pass, chosen** |

| Quarter | 057 | 8 columns + 0.75 | 64 columns + 0.75 |
|---|---|---|---|
| 2025Q3 | 1.86741 | 1.86523 | 1.86151 |
| 2025Q4 | 1.81061 | 1.80604 | 1.80271 |
| 2026Q1 | 1.77181 | 1.76924 | 1.76623 |
| 2026Q2 | 1.84675 | 1.84202 | 1.84131 |

- **Choice:** the 64 columns beat the 8 by −0.00282 [−0.00413, −0.00156], better in
  all four quarters, so the rule keeps them.
- **Against the screen:** the estimate was −0.00629, so the gain did not shrink at
  native capacity.
- **Size:** it closes 7.7% of 057's gap to the market (market 1.74482).
- **Against the market:** within-race correlation rises from 0.899 to 0.9065. A
  walk-forward blend with the market adds −0.00015 [−0.00047, +0.00021], so there
  is still no proof of information beyond the market.

## Limits

- About 50 comparisons were made on the screen quarters. The registered test is
  the safeguard against that, but the confirmation quarters have also now been
  read by experiments 057–063.
- The confirmation used the frozen 2 November cache. The store has moved since, so
  an installed model must be re-measured on the current store.
- The 64 columns exist only in this research script, not in the store.
- Which blocks earn the gain is unknown. The target share and inputs were not
  separated at native capacity, and three of the four blocks are null alone on the
  screen, so some columns may be removable.
- CatBoost (062) and the figure model (058) were not rebuilt on the improved
  model. They may still add something to it.

## Recommendation

1. **Install the 64 columns + 0.75 market share.** The columns must first move
   into the feature build, which changes `fe_signature` and forces a store rebuild.
   Measure the refit on the current store before promoting it.
2. **Then average three seeds of it.** That is the cheapest ensemble available and
   needs no meta learner. Its gain on this model has not been measured.
3. **Keep the meta learner simple if it is built at all.** Use a linear stack with
   two or three weights, fitted out of fold, with no race-context weights. Any
   sub-model should have to beat a seed average and a noise control.
4. **Add inputs only where a strong signal is hidden by its scale.** Do not add
   hand-built interactions, blanket field gaps or finer race labels.
5. **One sequence idea remains open:** a run-by-run model with Timeform and
   connections inputs that learns which past runs matter most.

## Reproduce

```bash
python -m scripts.feature_structure_063 run depth seeds context ratings ratings_parts form relative_gap depth8 race_kinds teacher combined combined_teacher views targets transfer
python -m scripts.feature_structure_063 stack
python -m scripts.feature_structure_063 additive
python -m scripts.feature_structure_063 register
python -m scripts.feature_structure_063 confirm
python -m scripts.feature_structure_063 confirm_report
```

Results are in `data/research/feature_structure_063/`: `summary_063.json`,
`additive_063.json`, `registration_063.json` and `summary_063_confirm.json`.
