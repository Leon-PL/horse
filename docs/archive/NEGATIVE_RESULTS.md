# Negative Results Ledger

## 107. Hold-out: the 11:00 betting rule PASSES its registered test (15 November 2026)

[Pre-registration and result](research/HOLDOUT_1100_107.md), registered in
4079adf before scoring. Production correction plus the 11:00 price inputs,
fitted once to 10 June, applied to 11 June–31 October (5,378 races). Rule:
threshold 0.02, price one tick worse than last traded, odds ≤ 51. 1,944
bets: geo CLV **1.0445**, one-sided lower bound 1.0357 (pass > 1.0); ROI
+8.3% [+0.7, +15.9]; every priced runner 0.880. Not yet settled: real fills,
stake size, price impact.

## 106. The correction's teacher: market weight 0.5 or 0.9 instead of 0.75 — no change (15 November 2026)

The joint correction with the 11:00 price (`with-1100`) refitted with its
target at 0.5 and 0.9 market weight (`with-1100-blend50`, `-blend90` in
`research/error_anatomy/joint_variants.py`, declared before scoring;
`research/blend_weight_106/clv.py`). Native log loss against 0.75
(1.77915): 0.5 +0.00042 [−0.00020, +0.00108] 1/4; 0.9 +0.00007 [−0.00031,
+0.00048] 2/4. The research 105 betting test at 11:00 moves in step with the
weight but within noise: threshold 0.02, one tick worse, ROI +3.7% (0.5),
+5.4% (0.75), +5.9% (0.9); geo CLV 1.018, 1.026, 1.028. Keep 0.75; the
weight is not a lever.

## 105. Betting the corrected forecast at a morning price: beats the close retrospectively; 11:00 beats T-60 (15 November 2026)

[Report](research/T60_CLV_105.md). Scored quarters 2025Q3–2026Q2, value rule
from `src.bet_settlement` against the race share at the cut, odds ≤ 51,
reduction factors applied. T-60, threshold 0: geo CLV 1.0047, ROI +3.4%
[+0.6, +6.3], gone at one tick worse (+0.2%). 11:00, threshold 0.02: geo CLV
1.0424, ROI +7.1% [+2.6, +11.6], and +4.6% [+0.4, +9.2] at two ticks worse.
Same universes: geo CLV 0.946 (T-60), 0.904 (11:00). Retrospective (recipe
chosen on these quarters), taken at LTP: an open lead, not a confirmed edge.
Model + BSP against BSP (walk-forward two-weight blend): corrected −0.00044
[−0.00091, +0.00006] 3/3, native −0.00009; the close holds almost everything.

## 104. Price timing: a fixed hour before the off beats 11:00; hours-to-off adds nothing (15 November 2026)

[Report](research/PRICE_TIMING_104.md). Inside the joint correction, native,
11,905 races: against the 11:00 price arm (1.77915), adding hours from 11:00
to the off +0.00006 [−0.00021, +0.00033] (null); the price one hour before
the off −0.00979 [−0.01263, −0.00699], 4/4; five minutes before −0.01797,
4/4. T-60 closes ~59% of the gap to BSP against 43% at 11:00. Not yet
checked as CLV. The 11:00 group includes 43 races off before 11:00, where
the price can be in-play (tiny; fixed in the new groups).

## 103. Model form: elastic net far behind, linear trees a tie (15 November 2026)

[Report](research/MODEL_FORM_103.md). 87 inputs, admission folds 2025Q3–2026Q2
(11,632 races), against screen LightGBM (1.82894). A race-wise elastic-net
conditional logit: +0.02261 [+0.01836, +0.02680], 0/4. LightGBM with
`linear_tree=True`: −0.00056 [−0.00243, +0.00136], 3/4, within noise. Keep
plain LightGBM; linear trees only worth a native-capacity check.

## 101. Debutants: Timeform's pre-race comment adds nothing beyond the text score (14 November 2026)

[Report](research/DEBUTANT_TEXT_101.md). On 15,477 debutants no comment
phrase (market interest, price, family, positive or negative wording) misses
the stored forecast or BSP by more than z 1.6. A debutant-only TF-IDF ridge
on won − forecast, fitted on earlier dates with its strength fitted on a
held-out later slice, gets weight 0 in two of three periods and +0.0019
[−0.0021, +0.0061] in the third. The 87-input `rtv_text_logit` already reads
the comment. With sibling form, sale price, debut records, connection
profiles and morning money all closed, the debutant gap is private money.
Do not retry public debutant signals.

## 100. Irish races without a class: effective class from prize money — not admitted (14 November 2026)

[Report](research/IRISH_CLASS_100.md). Irish races have no class (92% blank),
so `class_change_vs_field` is missing for 95% of Irish runners. A class read
off prize money against the previous year's British scale cuts that to 9%.
`admit`, 2025Q3–2026Q2: +0.00029 [−0.00150, +0.00200] against baseline,
−0.00097 [−0.00225, +0.00031] against twin (3/4); in Irish races alone
+0.00136 against baseline and +0.00011 against twin (2,630 races). Also
checked: the Irish flat draw gap (9.2% against 3.0%) is bumpers, which have
no stalls, not missing data.

## 099. Seasonal comebacks: the horse's own reappearance record — null inside the correction (14 November 2026)

[Report](research/COMEBACKS_099.md). Percentile and residual on earlier
comebacks (90+ days off), on the run after them, and against the rest of the
career. Alone against its twin −0.00082 [−0.00181, +0.00007], 3/4. Inside the
joint correction +0.00001 [−0.00030, +0.00029] against twin, 2/4 (11,905
races). The seventh group that is real alone and null inside.

## 098. Trainer moves: size of the move — not admitted as inputs, null inside the correction (14 November 2026)

[Report](research/TRAINER_MOVES_098.md). Inputs built only from earlier runs:
the new trainer's Elo minus the old one's (all runs in the yard, and the
first three only), runs for the current yard, and the trainer's earlier new
arrivals against SP. A change between names sharing a surname (C & M
Johnston → C Johnston) counts as the same yard.

The forecasts mis-rate moves badly. On the 87-input model's out-of-sample
forecasts (2022-07 to 2026-06-10, first three runs for a new yard, 42,728
runners), the biggest upgrades won 1,234 against 1,000 forecast (z +8.5) and
the biggest downgrades 461 against 573 (z −5.1). Betfair SP prices most of
it: the same upgrades won 1,234 against 1,175 at BSP (z +2.0), and
downgrades are priced exactly.

As inputs to the model it does not pay (`src.modelling admit`, screen
capacity, 2025Q3–2026Q2, 11,632 races):

| Arm | Against baseline | Against shuffled twin |
|---|---:|---:|
| All five inputs | +0.00128 [−0.00093, +0.00317], 1/4 | −0.00086 [−0.00238, +0.00074], 4/4 |
| First-three-runs Elo change only | +0.00147 [−0.00069, +0.00331], 0/4 | +0.00054 [−0.00097, +0.00195], 1/4 |

The affected runners are 9% of the field. Rescaling their forecasts by their
group's own miss gains only −0.00107 race log loss in-sample (an upper
bound), about what a refit comparison can resolve.

Inside the joint correction it adds nothing (`joint_variants`
`with-trainer-move` against `with-trainer-move-twin`, inputs shuffled within
quarter, native forecasts 2025Q3–2026Q2): 1.80531 against 1.80530,
+0.00001 [−0.00073, +0.00066], 2/4; against `more-groups-deeper` +0.00025
[−0.00051, +0.00095]. The correction already recovers most of the miss: for
the first three runs after the biggest upgrades the corrected forecast
expects 264 winners (279 won, z +1.1), and for the biggest downgrades 118
(106 won, z −1.2). With the `yard` group (93dcdfd), the stable-change line
is closed: do not rebuild it as another input family.

## Cheap screens: relatives' form, going reasons, pairwise and completion models, improvement, store pedigree and first string — null (9 November 2026)

The [cheap screens](research/CHEAP_SCREENS_2026-11-09.md) learn a small
regularised correction from a group's inputs on top of the stored full 063
forecasts and compare it with the same fit on a twin whose inputs are shuffled
within quarter (2025Q3–2026Q2, 11,905 races, retrospective).

| Group | Against its twin | Reading |
|---|---:|---|
| Relatives' form as rates (dam's other offspring: runs, wins, places, position, rating) | −0.00017 [−0.00052, +0.00017], 2/4, three twins | Not robust |
| Relatives' ratings: siblings' peak ability in pounds, raw and relative to their trainers | +0.00002 [−0.00046, +0.00041], 1/4 | Null |
| Going reasons: BHA going withdrawals joined to the withdrawn meeting's going, plus the store's 26 BHA going/withdrawal columns | +0.00020 [−0.00029, +0.00071], 2/4 | Rejected; the new comparisons alone cover 3–5% of runners and are null at Tier 0 |
| Store pedigree rates not in full 063 (26) | −0.00036 [−0.00074, +0.00002], 3/4; net −0.0001 | Absorbed |
| `trainer_first_string_*` without its odds-ranked columns | +0.00003 | Null; its signal came from `odds`, a market input |
| Completion (jumps): out-of-fold chance of failing to complete (AUC 0.69–0.72) from full 063 inputs plus 20 completion and jumping-error columns | +0.00020 [−0.00012, +0.00053], 1/4 | Rejected; the generalist already prices it |
| Improvement: sire, trainer and horse residuals against a margin-and-weight ability filter | +0.00015 [−0.00028, +0.00061], 1/4 | Rejected |
| Bounce: last-run residual, career best, close finish, quick return | −0.00016 [−0.00060, +0.00028], 2/4 | Unresolved; all of it on jumps races |

A head-to-head component (LightGBM with a Bradley–Terry loss over every pair
in a race, full 063 inputs and its 75% market target) scored 1.82796 alone
against 1.82598 for one exploded-softmax fit, and the log-pool gave it zero
weight in every quarter. With 062's CatBoost pairwise result, two pairwise
learners have now failed to add anything to the exploded softmax.

Do not rebuild relatives' form in another encoding expecting a gain. Ability
in pounds from margins and weights is not in this list: it adds information on
jumps races only (see the report).

## Text representation controls — little gain from more tokens (8 November 2026)

The [text development screen](research/TEXT_MODEL_IMPROVEMENTS_2026-11-08.md)
compares 19 recipes on the same 6,178 complete text fields in 2024Q1 and
2025Q2. Differences below are race log loss against the corrected calibrated
hard-target reference. They are development results.

| Control | Change | What to retain |
|---|---:|---|
| Word trigrams | +0.00145 | Word unigrams/bigrams |
| Two-year recency half-life | +0.00209 | Uniform weights for this learner |
| Separate comment/verdict/rank normalisation | +0.00967 | Joint normalisation; wins at all three matching penalties |
| Weak L2 (1e-5) with hard targets | +0.02114 | Tune penalty with the target |
| Character patterns | −0.00055 | Mixed across quarters, with substantially slower fitting |
| Rare terms (min_df=5) | −0.00033 | Keep min_df=20 unless a larger controlled gain pays for the vocabulary |

The largest gains in this screen come from the market-informed target. The
representation controls above use hard targets; they do not test every
interaction with the teacher. Self/rival attribution remains in target tuning
because its cost is small and it makes comparison phrases horse-specific.

The selected text model improves standalone loss by 0.01918 over four later
quarters. Adding it to the saved generalist/performance pool gives only
−0.00010 [−0.00149, +0.00146] on 1,142 races. The generalist already contains
older learned text inputs; this probe does not establish a new ensemble slot.

## 063. Race context, finer race types, subject models and meta learners — no gain (4 November 2026)

`scripts/feature_structure_063.py` screened how the 129 inputs are used, on the
development quarters at screen capacity. Direct arms cover 12,947 races and
stacks 16,540. Full tables are in
[FEATURES_AND_STACKING_063.md](research/FEATURES_AND_STACKING_063.md). Do not rebuild
these without a new reason:

| Idea | Race LL change | Quarters better |
|---|---|---|
| 28 race context columns | −0.00075 [−0.00252, +0.00089]; shuffled control +0.00067 | 3 of 4 |
| 19 finer race-kind columns from race titles | −0.00024 [−0.00186, +0.00147]; shuffled control +0.00047 | 3 of 4 |
| Finer kinds + record in this kind of race | +0.00138 [−0.00033, +0.00305] | 1 of 4 |
| Past form weighted towards runs like today's | −0.00110 [−0.00292, +0.00074] | 3 of 4 |
| Field gaps for 10 other strong inputs | −0.00016 [−0.00194, +0.00162] | 3 of 4 |
| Depth 8 trees | −0.00075 [−0.00239, +0.00103] | 3 of 4 |
| Run-by-run form model in the stack | −0.00001 [−0.00033, +0.00030] | 2 of 5 |
| 7 subject models on top of the full model | +0.00021 [−0.00045, +0.00080] | 2 of 5 |
| 7 subject models instead of the full model | +0.01830 [+0.01486, +0.02183] | 0 of 5 |
| Temperature by race kind | +0.00103 [+0.00038, +0.00169] | 2 of 5 |
| Boosted meta learner with race context | +0.00019 [−0.00059, +0.00100] | 2 of 5 |

Sub-model targets for completion, position, late gain, finishing speed, placing,
time figure and handicap mark also added nothing. The closing market head gained
−0.00161 and the figure −0.00059, but both fall to about −0.0002 once the inputs
and target are improved.

Two ideas from the same screen were not refuted: rating-gap inputs and a 0.75
market share in the target. With all 64 screened columns they confirmed at native
capacity on 2025Q3–2026Q2: −0.00628 [−0.00823, −0.00426], better in all four
quarters. This does not rescue the context arms: they are null alone, and which
blocks earn the combined gain has not been measured. The run-by-run test does not
cover a model that learns which past runs matter most.

## 062. CatBoost pairwise — ties the 057 model at best; not worth a second framework (4 November 2026)

`scripts/catboost_pairwise_062.py` used CatBoost's `PairLogit` loss on the 129
production inputs, with pairs built from 057's three stages. Results on
2025Q3–2026Q2 (11,905 races):

| Arm | Against production | Against 057 | Quarters better than 057 |
|---|---|---|---|
| Results only, rate 0.2, 100–170 trees | +0.0169 [+0.0134, +0.0206] | +0.0215 [+0.0180, +0.0252] | 0 of 4 |
| Soft target, rate 0.05, 1,382–1,600 trees | −0.0037 [−0.0064, −0.0011] | +0.0009 [−0.0014, +0.0031] | 1 of 4 |

The soft-target arm ties 057 and does not beat it. It does help as a second
model: −0.0016 [−0.0029, −0.0003] over 057, on 8,370 races. But the figure model
already does that job (−0.0017). Added to the 057 + figure blend, CatBoost gains
only −0.0008 [−0.0017, +0.0001]. That does not pay for a second framework and
about 40 minutes a fit. The arm was chosen after the fast arm had been scored on
these quarters, so it is not a registered result.

**The lesson:** compare a new learner on the same soft target as production.
The results-only arm alone would have closed this idea for the wrong reason.
Details are in `MODEL_IDEAS_057_060.md`.

## 061. Finishing order below third place — nothing more (2 November 2026)

This follows 057, the exploded softmax over the first three places with stage
weights (1, 0.25, 0.0625).

**Setup.** `scripts/exploded_depth_061.py` screened five variants on the
development quarters 2024Q1–2025Q2 at screen capacity (19,086 races). The
grid and the rule were fixed before any fit. Race log loss against 057:

| Variant | Change [95% CI] | Quarters better |
|---|---|---|
| Every finisher, weight 0.25 per place down | −0.00002 [−0.00109, +0.00102] | 3 of 6 |
| Top three at full weight, flattened (temperatures 1, 0.8, 0.64) | +0.00531 [+0.00313, +0.00764] | 0 of 6 |
| Every finisher at full weight, flattened 0.8 per place | +0.01274 [+0.00963, +0.01587] | 0 of 6 |
| Same, flattened 0.65 | +0.02551 [+0.02097, +0.03004] | 0 of 6 |
| Same, flattened 0.5 | +0.03725 [+0.03110, +0.04350] | 0 of 6 |

**What this shows.**

- **Lower places, down-weighted, add nothing.**
- **Flattening makes the model over-confident.** One score per runner has to
  spread wider to fit a flattened later choice. At 0.5, the top pick averages
  a 34.2% chance but wins 31.1% of the time; the down-weighted models are
  calibrated.

Do not use temperatures on later stages to learn scores. Keep 057's three
places.

## 058–060. Luck weights, connection profiles and the figure model on its own — not confirmed (2 November 2026)

These are three of the four ideas in `MODEL_IDEAS_BRIEF.md`. The fourth, the
exploded softmax (057), is confirmed; see the progress log.

**Setup.**

- Harness: the 048/049 harness on the 056 store, with the 129 production
  inputs and the production win recipe.
- Folds: purged quarterly walk-forward, scored on 2025Q3–2026Q2 at native
  capacity (11,905 races, 109,631 runners, non-finishers kept).
- Choices: made on 2024Q1–2025Q2 and registered before confirmation.

Race log loss against production (1.83135):

| Idea | Change [95% CI] | Quarters better |
|---|---|---|
| 059 luck-adjusted training weights | −0.00029 [−0.00112, +0.00058] | 3 of 4 |
| 060 learned trainer, jockey and sire profiles (13 inputs) | +0.00011 [−0.00092, +0.00113] | 1 of 4 |
| 058 figure and spread model, used instead of production | +0.00480 [+0.00084, +0.00903] | 1 of 4 |

**059, luck weights.**

- Races whose beaten runners had trouble (from Racing TV close-ups and BHA
  stewards' events) were down-weighted by 1 − 2 × the troubled losers' market
  chance.
- The development screen gave −0.0014, 6 of 6 quarters. At native capacity
  that shrank to a fifth and crossed zero.
- As with the trouble inputs in ledgers 2b and 7, unlucky results are not what
  drives the model's errors.

**060, connection profiles.**

- Each trainer, jockey and sire got a rank-4 profile of results against SP by
  context. Profiles were refitted every quarter on earlier races only.
- They are priced: no profile input correlates with won − market beyond
  |0.007|. The strongest are the trainer's new-yard, second-run and debut
  projections, at 0.0067, 0.0065 and 0.0061.
- So the yard-change and debutant parts of the 055 gap are not in the
  connections' record against the market.

**058, figure model on its own.**

- It predicts pounds behind the winner plus a spread, and turns them into win
  chances with a Thurstone race model. On its own it is worse than
  production.
- It is not dead. Blended out of sample, it adds −0.0042 over production and
  −0.0017 over the 057 model, against 0.0002–0.0006 for blending near copies.
- Its spread is wider for debutants (8.22 lb) and new-yard runners (8.09 lb)
  than for others (7.42 lb).
- It closes under 10% of the 055 steamer and drifter bias.

**8 November follow-up:** the new modelling layer's native pilot uses strictly
chronological residuals and an identified, regularised pool. On 1,142 later
2026Q2 races it adds **−0.00199 [−0.00322, −0.00070]** to the generalist, while
remaining weaker alone. This retains the figure head for broader evaluation;
the stronger 063 generalist and repeat-fit averaging controls still need testing.
See [the reproduction report](research/PERFORMANCE_SPREAD_REPRODUCTION_2026-11-08.md).

Do not retry luck weights or market-residual connection profiles in these
forms. Full tables: `MODEL_IDEAS_057_060.md`. Receipts:
`data/research/model_ideas_057/summary_05{8,9}.json` and `summary_060.json`.

## 055. Fixes for the 054 model's worst misses — all within noise (2 November 2026)

The 055 review (`MODEL_ERROR_REVIEW_055.md`) found three model-side patterns
among the worst misses:

- debutants over-rated, worst from big yards;
- the last run missing from our DB (abroad or point-to-point), although its
  finishing position is in the pre-race form;
- ratings carried across disciplines.

`scripts/screen_controllable_055.py` sizes each one as a race-wise logit
adjustment to the 049 `general` arm's out-of-fold predictions (126 inputs,
2025Q3–2026Q2, 11,901 races). Each adjustment is fitted on earlier quarters
only and scored on the next. Every candidate uses races on strictly earlier
dates.

| Adjustment | Race LL change (mean of 3 quarters) | Quarters better |
|---|---|---|
| Debutant shrink (1 parameter) | −0.00021 | 1 of 3 |
| Trainer and sire debut record: win rate, actual/expected vs prior SP, market respect | +0.00029 | 1 of 3 |
| Off-DB last run × its finishing position | +0.00011 | 1 of 3 |
| Jumper on the Flat / Flat horse over jumps | +0.00033 | 0 of 3 |

Each is below the 0.001–0.004 noise floor of a refit, and the signs flip
between quarters.

- **Debutants.** They are over-rated in 3 of 4 quarters (501 wins against
  552.9 expected), but by different amounts each quarter, so a shrink fitted
  on the past does not carry forward. The same 1-parameter shrink fitted on
  development scored −0.00075 on the 054 hold-out. That was a single look at
  golden-period data and is not a confirmation.
- **Off-DB last run.** Runners whose last run is missing from the DB win
  slightly more than expected (205 against 187.0), but the finishing-position
  pattern does not generalise.
- **Discipline carry-over.** Calibration in both segments swings from 0.72 to
  1.23 by quarter. That is noise; Lord Byron was an anecdote.

Do not build these into the store. The controllable gain that is confirmed is
the 052 comment score (−0.0029, 4 of 4). Receipts:
`data/research/controllable_055/summary.json`.

## 053. Unused Racing TV card data — nothing beyond the Timeform comment (2 November 2026)

Same harness as 049 and 052: the 126 inputs, production win recipe at native
capacity, purged walk-forward on 2025Q3–2026Q2, 11,901 races. Race log loss,
against the 126-input model:

| Added inputs | Change [95% CI] | Quarters better |
|---|---|---|
| Timeform comment score (052) | −0.0029 [−0.0049, −0.0010] | 4 of 4 |
| Card data alone | −0.0011 [−0.0024, +0.0002] | 4 of 4 |
| Card data + a comment model that also reads pace notes and the last three close-ups | −0.0028 [−0.0051, −0.0006] | 4 of 4 |

The card data (`scripts/rtv_card_extras.py`) was:

- trainer, jockey, icon and text badges as 0/1 flags;
- the number of Racing TV pundit tips;
- the draw comment ("Favours Low") times the runner's relative stall;
- owner runs and smoothed strike rate from earlier days only.

On their own these are not confirmed. Added to the comments, card data, pace
notes and earlier close-ups give nothing: −0.0028 against −0.0029, well inside
the 0.001–0.004 per-quarter noise between fits that differ only in columns.
The pace notes and close-ups lower the text model's own loss on the
selection quarters by only 0.0044 (1.9538 to 1.9494). Its strongest close-up
words are "impressive", "easily", "weakened" and "behind", which our margin
and position history already encode.

Each flag is close to what SP implies (actual/expected 0.91–1.07): trainer
"in form" 1.03, pundit tips 1.05, owner strike-rate quintiles 0.98–1.02, and
draw fit flat. So the market already prices them.

**Not tested, because the card copy changes after the off (2026-10-30
snapshots):** the race pace forecast changed after the off in 2 of 34 races,
and smart stats (whose trainer profit figures move with results) in 1 of 34.
Trainer badges and pace notes changed only before the off, so the stored
history matches the pre-off card. Pace maps, course pace statistics and
prompts exist only for the snapshot days.

## 049. Race-type and course specialist win models — worse than the general model (1 November 2026)

Same 126 inputs (`feature_sets/win_selected_048.txt`), production win recipe
at native capacity, purged walk-forward on 2025Q3–2026Q2. Each specialist is
trained only on earlier races of its own subset and scored on the same test
races as the general model. Race log loss; the market is BSP-implied, else SP.

| Subset | Test races | Market | General gap | Specialist gap | Specialist − general [95% CI] | Quarters better |
|---|---|---|---|---|---|---|
| Hurdles | 2,820 | 1.6641 | +0.1030 | +0.1086 | +0.0056 [−0.0001, +0.0110] | 1 of 4 |
| Wolverhampton | 629 | 1.6855 | +0.0967 | +0.1067 | +0.0100 [−0.0057, +0.0280] | 1 of 4 |

Neither model adds to the market. A conditional-logit blend
`p ∝ market^a · model^b` was fitted on earlier quarters and scored on
2025Q4–2026Q2. It is no better than the market alone:

- all races: +0.0000 [−0.0002, +0.0004];
- hurdles: +0.0023 general, +0.0020 specialist;
- Wolverhampton: +0.0065 general, +0.0081 specialist.

The fitted model weights are close to zero or negative.

The inputs already carry each horse's whole cross-code history, and the
general model can split on `race_type` and `track` itself. Specialising
therefore only removes training data:

| Model | Training rows | Trees chosen by early stopping |
|---|---|---|
| General | 504k–590k | 2,836–3,776 |
| Hurdles | 121k–143k | 1,244–1,821 |
| Wolverhampton | 24k–29k | 780–1,163 |

Do not retry other single codes or courses as standalone models. The general
arm reproduces 048's native confirmation to within 9e-16.

Script: `scripts/specialist_models.py`. Receipts: data/research/specialist_049.

## 044. Fixed RTV history family — uncertain development gain (30 October 2026)

One preset 400-tree comparison adds 26 causal telemetry/history inputs to the
corrected 154-input baseline. Q4 2025 covers 3,234 fields / 30,094 runners;
exactly two new fits, no subset/seed/quarter/capacity search. Race LL improves
1.84690718→1.84372079 and Brier .08545918→.08534314. Established-handicap
LL improves 1.98538384→1.98148413 across 1,071 fields. Overall date-cluster
95% LL-delta interval [-.00717584,+.00095211] crosses zero: no input selection
or predictive-gain claim. Do not mine these outcomes for a better subset.
The four true pre-race snapshot inputs lack historical evaluation. Historical
telemetry backfill has unknown original revision times. Implementation remains
available in the normal pipeline; mandatory factual repairs are independent of
this verdict. [Implementation and evidence](research/RTV_FEATURES_REPAIRS_044.md).

## 038. Date-local smooth win calibration — not selected (29 October2026)

One fixed90-calendar-day calibration proposal uses native temporary heads
fitted through17September2025, a seven-day purge, and24September–22December
training-only calibration forecasts. Final033 trees are reused unchanged.
There are exactly two new temporary fits shared with the separate place
comparison; no seed, quarter, capacity or calibration-family search.

On the same2,591 already-inspected Q1 fields, win race LL worsens
1.798224325→1.799563197 and Brier .089957954→.090005218. Date95% LL-delta
interval[-.000497202,+.003039582] crosses zero. Established-handicap point
scores worsen too. No win change is selected. This does not prove the older
calibration slab is ideal; it rejects this fixed replacement. Do not repeat
the same smoothing/window search against these outcomes.

The separately evaluated place layer passes its registered overall Brier
gate; see [038 results](research/PREDICTIVE_REVIEW_038.md). Its gain is not a reversal
of the win result or evidence of prospective market parity.

## 32. Earlier-fitted history-context blend — small uncertain gain (2026-10-27)

Experiment013 follows012's inspected external-form lead, so it is exploratory,
not fresh confirmation. Fit two whole-race mixture coefficients onQ3OOS
forecasts, validateQ4 and freeze beforeQ1/April–May scoring. Selected new-model
weights are0.4943for fields without external form and1.0for fields containing
it. No new trees or test-selected fallback. Q1LL1.81799380→1.81767994,
delta−0.00031387,95%[-0.00247654,+0.00203919]. TransferLL1.90437727→1.90248792,
also uncertain. Brier/binary point estimates improve but primary uncertainty
gates fail; market remains substantially ahead. No promotion or reversal of012.
Do not search more context gates on these same outcomes. See
[results](research/MODEL_HISTORY_BLEND_013.md).

## 31. History depth, pooled rates and availability family — not qualified (2026-10-27)

Experiment012 adds16existing inputs to149after one outcome-free duplicate
removal. Three native fits, unchanged recipe/fields, exact saved baselines.
On9,356races, LL1.87204489→1.87133338, delta−0.00071151,
95%[-0.00269877,+0.00126808]; winter worsens. Brier/binary point estimates
improve, but primary uncertainty gates fail. Matched market gap closes only
0.38%in point estimate. No promotion. The776-field external-form diagnostic
is a useful hypothesis, not permission to select that slice retroactively;
its separate exploratory ensemble is013. Preserve all original scores.
See [results](research/MODEL_HISTORY_012.md).

## 30. Narrow relative floors on extreme-tail forecasts — no qualified gain (2026-10-26)

Experiment011 floors only probabilities below1% that are heavily suppressed
relative to the pre-isotonic Platt stage, then normalizes complete fields.
Strength selected on2025Q3;2025Q4 validates or falls back to identity. The149
model learns0.1 but validationLL worsens0.00002891; the176 model fails the
strict fitting Brier guard. Both selected strengths are zero. No change to
production or previous experiment results.

A separately registered post-result diagnostic chooses the earlier fitting
LLminimum without the Brier veto,0.25 for both models. It repairs the known
Naas near-zero probability to0.0972%, but does not establish overall lift:
149Q1LLdelta−0.00005136,95%[-0.00044380,+0.00015391], transferdelta+0.00002678;
176Q1delta+0.00012877,95%strictlypositive, transfergainuncertain. Brier changes
are tiny, so strict rejection is not evidence of material Brier damage, but
removing that veto still fails to yield a reliable later gain. The diagnostic
is explicitly promotion-ineligible and cannot reverse011/009. No near-zero
winning forecast occurs in either later period; no prospective efficacy claim.
Close this floor/blend search on these inspected quarters. See
[results](research/MODEL_TAIL_SMOOTHING_011.md).

## 29. Broad shrinkage of isotonic forecasts toward Platt — not confirmed (2026-10-26)

Experiment010 keeps all009 trees and feature sets fixed. One Brier-constrained
coefficient is fitted on2025Q3 OOS forecasts, validated onQ4 and frozen before
later evaluation. The149-input model selects82.867%Platt; the176-input pace
diagnostic selects zero adjustment under its earlier Brier constraint.

On2,591January–March races, primary LL1.81799380→1.81818510, delta+0.00019130
(95%[-0.00187169,+0.00209070]); Brier improves slightly. April–May with the same
December trees improves LL1.90437727→1.90275709 but its interval also crosses0.
Neither later period contains a winner assigned below1e-7 by the old model:
do not call this direct validation of protection against near-zero winners.
Some severe underestimates improve, but that does not establish overall lift.
The market remains substantially ahead; no production change or rescue of009.
This does not reject every tail smoother. Do not repeat this broad blend or
choose only its favourable period. See [results](research/MODEL_CALIBRATION_010.md).

## 28. Existing pace/setup family in independent forecasts — no reliable gain (2026-10-26)

Experiment009 tested the149-input recipe against the same recipe plus27
existing run-style, field-pressure and course-bias features. These columns
were absent from the earlier206ATR-feature trial. Six native historical fits
completed on the three registered development quarters; no feature/parameter
sweep and no production change.

On9,356races, primary race log loss changes1.87204489→1.87310002, delta+0.00105513
(95%paired interval−0.00190667to+0.00513894). Only one quarter improves. Brier
improves slightly0.08754346→0.08752490, but uncertainty includes no benefit;
binary loss worsens. The candidate fails the registered rule and is not
promoted. The matched market gap also widens slightly. Do not repeat this
unchanged family or choose subsets from these test results.

Exploratory error attribution shows one extreme underestimation dominates
the net LL deterioration. Retain that race and all original scores. It
motivates a separate training-only calibration/tail-robustness hypothesis,
not a post-hoc rescue of this candidate. See [results](research/MODEL_PREDICTIONS_009.md).

**Read this before proposing a new feature, data source or edge.**

These ideas were built and measured but did not meet their declared criteria.
Some results are strongly negative; others remain uncertain. An interval
crossing zero does not prove a hypothesis false. The record prevents repeating
unchanged failed tests or rediscovering an apparent edge caused by stale data.
Several entries cost weeks. For win-probability forecasting, use the current
`MARKET_PERFORMANCE_PLAN.md` and proper-score protocol; the older CLV/ROI
checks below concern betting claims.

Dates are when the result was established. "Dead" means it was measured and
did not survive; it does not mean the code was deleted — most of the data
collection is still in place and worth keeping.

---

## 0. How an edge claim is judged here

Bitter experience produced this bar. Apply it to every candidate:

1. **Assess incremental information against the right target.** For a market
   edge, inspect association with the same-time market residual
   (`won − market_implied_prob`). Near-zero marginal correlation lowers the
   prior, but can miss interactions and does not rule out improving an
   independent forecast. Test incremental proper scores on later races.
2. **CLV at BSP is the metric; ROI is secondary.** Betfair Starting Price is
   sharp. Beating it on closing-line value is the only durable claim. Read it
   from `median_clv` / `geometric_mean_clv` / `pct_beat_close`
   (`src.clv.summarise_clv`), **never the mean ratio alone** — the mean is
   upward-biased at long prices and has reversed a verdict here (§2).
3. **The ROI@BSP-vs-CLV consistency check is the leak tripwire.** A strategy
   with high ROI and CLV ≈ 1.0 is almost always leaking, not winning. This is
   what caught the `trainer_freq` leak (§3).
4. **Confirm the measuring instrument before the measurement.** Check
   `scripts/project_status.py` first. A stale feature store has produced a
   phantom edge at least once (§1).
5. **Never fit a second stage on in-sample first-stage predictions.** Use
   `wf_eval.oof_predict_expanding`. Getting this wrong buried a lead for
   three months (§2).
6. **Cluster confidence intervals by race.** Bets in one race are not
   independent; `wf_eval.bootstrap_roi_ci` resamples races. An i.i.d.
   bootstrap over bets is how a result gets called "significant".
7. **Keep a representable baseline and test it fairly.** A model that nests
   market-only must be compared with that same forecast on identical fields.
   It can still lose on later races; representability does not guarantee
   future non-worsening. Audit fitting, selection and fallback timing before
   interpreting the failure. Treat `wf_eval.check_combination_nesting` as a
   diagnostic, not a guarantee about unseen outcomes.
8. **Do not overturn a result on a partial re-run.** A two-fold re-run of §2
   disagreed with the recorded verdict; eight folds reproduced it. Extend the
   sample before rewriting the conclusion.

---

## 1. Place-market edge — OVERTURNED (2026-06-24)

A walk-forward place-market strategy showed roughly **+15% ROI**. It was a
**stale feature-store artifact**: the store on disk had been built by older FE
code than the evaluation assumed. Rebuilt correctly, the same walk-forward
gives **+1.5% with a confidence interval spanning zero**, and it is negative on
jumps. There is no place-market edge.

**Lasting consequence:** this is why `fe_signature` exists, why
`tests/test_fe_signature_coverage.py` forces every FE dependency to be
registered, and why the serving app refuses to price a card when the promoted
model's feature generation does not match the store.

## 2. Market combination (Benter two-stage) — DEAD, re-confirmed (2026-10-12)

Combining the fundamental model with the market, settled at BSP with 2%
commission, expanding quarterly walk-forward. **Measured twice, with two
different harnesses, and the answer held both times.**

| | original (2026-06-29) | corrected harness (2026-10-12) |
|---|---|---|
| Combination ROI @ BSP | **−2.64%** CI[−4.45, −0.81] | **−3.55%** CI[−5.42, −1.55] |
| Model alone | −3.65% | −4.49% |
| Bet all (in band) | −1.83% | −2.68% |
| Mean CLV (biased; see below) | 1.0047 | 1.0076 |
| Folds positive | 2/13 | 1/8 |

The 2026-10-12 figures come from a rewritten harness: blend fitted
**out-of-fold**, the production per-race conditional logit
(`model.fit_market_anchor`) instead of a binary logistic regression, purged
folds, and race-clustered confidence intervals. The earlier harness had all
three of those wrong (details below) — and **fixing them did not change the
conclusion**. It is still significantly negative and still loses to simply
backing everything in the band.

### What the two runs agree on

- **The anchor helps the model**: the combination beats the model alone in both
  runs (−3.65 → −2.64; −4.49 → −3.55).
- **The selection rule is the weak link**: the combination still loses to
  bet-all, so picking runners where `combo_prob > mkt_imp` selects bets that
  are *worse than average* inside the band. The loss is in the selection, not
  in the blend.
- **α is positive and stable** — +0.037 to +0.057, positive in **8/8** folds
  (mean +0.046). The fundamental model genuinely carries information the price
  lacks, consistent with the production golden hold-out. It is simply far too
  small to overcome commission: β sits at ~0.97, i.e. the market is ~20× more
  informative.
- **CLV does not clear 1 on an unbiased estimator.** The mean ratio reads
  1.0076, but the median is 0.9865 and the geometric mean lower still; the
  picks edge out bet-all on CLV while still drifting against the close. See
  the measurement note below.

**Conclusion: no bankable edge in the market combination.** Do not re-run this
without a *new* selection rule or a materially better fundamental model; the
combination layer itself has now been measured twice and is not the lever.

### Harness defects found and fixed (worth knowing, did not change the verdict)

1. **The blend was fitted on in-sample model predictions** — the win model was
   trained on the fold's training rows, then the blend fitted on that same
   model's predictions over those same rows. In-sample probabilities are
   over-sharp, so the blend over-weights the model.
   `tests/test_benter_harness.py` demonstrates this on synthetic data: the
   in-sample fit yields a larger `alpha` and a *worse* out-of-sample race log
   loss than an out-of-fold fit on identical data. Fixed with
   `wf_eval.oof_predict_expanding`.
2. **A binary logistic regression stood in for the conditional logit.** Fixed
   by calling the production estimator, so research and production now measure
   the same object (hard rule #2).
3. **No purge gap**, while production purges `config.PURGE_DAYS`. Fixed.
4. The model-alone and bet-all baselines were only accumulated on folds where
   the combination happened to bet, and the win model was given the *place*
   model's hyperparameters. Both fixed.

### Where the loss actually comes from (2026-10-12)

The combination beats the model alone yet loses to backing the whole band, so
the damage is in the **selection rule**. `scripts/audit_selection_rule.py`
localised it on 196,664 runners across the eight folds:

| price band | share of stakes | ROI selected | ROI whole band | gap | median CLV |
|---|---|---|---|---|---|
| 1.5-3 | 4.6% | -1.12% | -2.55% | **+1.43** | 1.0000 |
| 3-6 | 18.4% | -2.15% | -3.03% | **+0.88** | 0.9962 |
| 6-12 | 30.4% | -0.79% | +0.42% | **-1.21** | 0.9804 |
| 12-25 | 28.7% | -6.41% | -2.87% | **-3.54** | 0.9843 |
| 25-50 | 17.9% | -3.49% | -4.99% | **+1.49** | 0.9691 |

**The longshot hypothesis was wrong.** The obvious explanation — that a value
filter is a longshot filter in disguise, since that is where a model least
agrees with the price — does not hold. The selection rate barely rises with
price (45% to 55%), and at the *longest* prices the filter is one of its best.
The loss is concentrated in the **middle of the book**, 6-25, which carries
59% of stakes.

Nor is it a calibration failure. Predicted win rate is within 0.0023 of actual
in every band. What separates the losing bands is that the **market** is
closer to the outcome than the model is; the filter only adds value where the
model is the better of the two.

Measured against a **matched** baseline (backing everything in the same
universe the rule bets in, not the whole band), the only sub-rule that beats
no-filter is `odds < 6`: **+0.98 points**, n=23,584 — still -1.95% absolute,
so useful as a diagnosis rather than a strategy. Note the trap: `odds < 12`
scores -1.29% against a whole-band baseline of -2.68% and looks like a
+1.39-point win, but backing *everything* at odds < 12 returns -1.07%, so
inside its own universe the filter is 0.22 points worse. Essentially all of
the apparent gain was avoiding long prices.

One structural finding for any future selection rule: an **absolute edge
threshold is vacuous here.** Because the fitted beta sits at ~0.97, the
anchored probability stays very close to the market, so `combo_prob/mkt_imp >
1.10` matches only **16 bets out of 104,146**. A rule has to work on rank or
relative edge within a race, not on a ratio threshold.

### Measurement note: never read CLV off the mean alone (2026-10-12)

CLV is this project's primary metric, and the research harnesses were
reporting it as the **mean of `entry_odds / close_odds`**. That estimator is
upward-biased, badly so at long prices, because the upside of a price ratio is
unbounded and the downside is not: a 25.0 shot closing at 12.5 scores 2.0,
closing at 50.0 scores 0.5, and averaging those gives 1.25 for two moves that
are symmetric in probability terms.

It changes verdicts. The same selections scored:

| price band | mean CLV | median CLV | geometric mean |
|---|---|---|---|
| 1.5-6 | 1.0006 | 0.9982 | 0.9873 |
| 6-12 | 0.9969 | 0.9804 | 0.9719 |
| 12-25 | **1.0147** | 0.9843 | 0.9739 |
| 25-50 | **1.0229** | 0.9691 | 0.9594 |

The mean says these picks beat the close; the median and geometric mean say
they drift. On the honest measures the combination's picks have **median CLV
0.9865 against 0.9786 for bet-all** — better than backing everything, but
still below 1, so there is no closing-line value to bank.

`src/clv.py` always reported median, geometric mean and %-beat-close
correctly; it was the harnesses that quoted only the mean.
`src/wf_eval.clv_summary` now routes every harness through the same function,
and any CLV figure quoted in this ledger from before this date that came from
a harness mean should be treated as optimistic.

### A caution about how this entry was nearly mis-corrected

A first re-run limited to **two folds** gave −1.59% with a CI spanning zero and
the combination beating bet-all, and that was briefly written up here as
"re-opened". Eight folds reversed it. Two folds of racing data is not enough to
overturn anything — and a partial re-run that happens to agree with the
hypothesis is exactly when to extend the sample rather than publish.

A related error: the combination losing to **bet-all** was read as a nesting
violation (a market-anchored model nests the market at α=0, β=1, so it cannot
be much worse than the market). That reasoning does not apply — "bet every
runner in the band" is a staking rule, not the market probability model, so
losing to it is not evidence of a broken fit. The genuine nesting check is
`wf_eval.check_combination_nesting`, which compares race log loss against
market-only; it **passes** in the corrected runs.

## 2b. ATR sectionals — 20 of 21 signals already priced (2026-10-12)

The At The Races / TPD sectional scrape is the most expensive dataset this
project has acquired: a Fastly TLS/JS challenge to clear, `curl_cffi` or a
system `curl` binary, ~2.5s per request, GB-only, a resumable multi-week
backfill. It produced **216 `atr_*` columns** in the store — velocity curves,
finishing-speed percentages, closing speed, turn of foot, energy efficiency,
stride length and frequency, sectional-adjusted upgrades, and nine
in-running comment tags.

**None of them were ever in the allowlist, and none had ever been triaged.**
That work was in flight when the session that built it ended.

Triaged with `scripts/triage_features.py` on 184,035 rows that carry the data
(coverage 2024-01-04 to 2026-06-10, 31% of priced rows), scoring each column's
partial correlation with the market residual — both demeaned within race, with
the price partialled out on probability, log-odds and quadratic terms:

| signal (strongest column of family) | coverage | partial corr | t | verdict |
|---|---|---|---|---|
| `atr_closing_in_hand` | 100% | **+0.0105** | +4.5 | shortlist |
| `atr_turn_of_foot` | 100% | +0.0073 | +3.1 | priced |
| `atr_efficiency_score` | 100% | +0.0070 | +3.0 | priced |
| `atr_stride_x_gallop` | 80% | −0.0060 | −2.3 | priced |
| `atr_max_speed` | 100% | +0.0057 | +2.4 | priced |
| `atr_closing_speed` | 100% | −0.0056 | −2.4 | priced |
| `atr_energy_eff` | 100% | −0.0054 | −2.3 | priced |
| `atr_finishing_speed_pct` | 100% | +0.0051 | +2.2 | priced |
| `atr_peak_stride_length` | 80% | +0.0035 | +1.3 | priced |
| `atr_speed_score` | 100% | +0.0032 | +1.4 | priced |
| `atr_cmt_trouble` | 100% | **−0.0014** | −0.6 | priced |

(11 of 21 shown; the remaining comment-tag families all score below ±0.005.)

**One signal clears the bar, marginally.** `atr_closing_in_hand` — a horse
finishing fast while apparently not fully extended — at +0.0105 with t=+4.5.
For scale, the dead sale-price feature scored 0.0000 and the best *rejected*
sectional signal here is 0.0073, so this is barely separated from the pack. A
partial correlation of 0.01 explains about 0.01% of residual variance.

**The one genuinely interesting structural finding:** the *derived* "had more
to give" measure is the strongest signal in the family, while the *explicit*
in-running trouble comment (`atr_cmt_trouble`) is the **weakest of all 21** at
−0.0014. The prediction that a data-native trouble signal would beat parsed
text was correct in direction — and both are too small to matter. That also
closes the idea, recorded in §7, that sectionals would succeed where stewards
reports failed: they fail the same way.

**Status:** the shortlisted signal still needs the CLV test at BSP under the
locked walk-forward; correlation is necessary, not sufficient. Expect it to
fail — the market-combination work established that β sits at ~0.97, so a
signal this small cannot clear commission.

**Caveat:** this ran against the store built 2026-07-03, whose `fe_signature`
differs from the current FE code. The result is uniform enough (20 of 21 under
0.0075) that code drift is very unlikely to reverse it, but re-confirm after
the store rebuild before treating it as final.

**Re-confirmed 2026-10-13 on the rebuilt store** (fe `63d88ff721a02632`,
development period only, 2023-08-11..2026-06-03): **20 of 21 signals priced**
(largest |0.0073|, `atr_stride_x_gallop`); `atr_closing_in_hand_avg_3` the only
shortlisted signal at **+0.0115** (t +4.2), against +0.0105 before. The
verdict is final. Its golden-hold-out test is not possible (sectionals end
2026-06-20; see `docs/archive/SESSION_LOG.md`); the walk-forward economic test is in
section 15.

**This fires decision gate 3 in [ACTION_PLAN.md](archive/ACTION_PLAN.md):** the
sectionals were the last unexploited asset already in hand. With them priced,
market microstructure (traded volume, ladder depth) is the only live
hypothesis left that attacks the informational gap rather than re-processing
public form.

## 2c. Why the model loses: its disagreements are uninformative (2026-10-12)

A head-to-head of the market-blind fundamental model against the market on
**25,388 walk-forward races**, each with exactly one recorded winner:

| | top-1 accuracy |
|---|---|
| model's top pick | 0.2921 |
| market's top pick | 0.3398 |
| they pick the same horse in | 55.1% of races |
| when they agree, that horse wins | 0.3878 |

**The entire deficit is in the 44.9% of races where they disagree**, and there
the market wins the head-to-head 0.2809 to 0.1746.

### The decisive test

In those disagreement races, compare the model's pick to what the market
already charged for it:

| | actual win rate | market-implied | edge |
|---|---|---|---|
| the horse the model **promoted** | 0.1746 | 0.1788 | **−0.43 pp** |
| the horse the model **demoted** | 0.2809 | 0.2814 | **−0.05 pp** |

**The model's disagreement carries no information.** Its preferred horse wins
exactly as often as the price says, and the horse it rejects wins exactly as
often as the price says. This holds in every disagreement-magnitude bucket
(−0.80 to +0.54 pp from 1.2× through 3.5×+), every field-size band, every
exposure band and every race type — nothing exceeds ±1.9 pp, and nothing
survives a race-clustered interval.

So the model is **not mis-ranking correctable things**. It is re-ordering
horses whose probabilities the market has already got right.

### The mechanism

Standardised mean difference between the model's pick and the market's pick,
across all 11,389 disagreement races:

| feature | model's pick minus market's pick (SD) |
|---|---|
| `horse_glicko_rank` | **−0.65** (lower = better) |
| `horse_glicko_vs_field` | **+0.56** |
| `horse_elo_momentum_vs_best_in_race` | **+0.54** |
| `horse_elo_momentum_vs_field` | +0.51 |
| `horse_elo_momentum` | +0.43 |
| `combined_elo_vs_field_z_in_race` | +0.37 |
| `horse_avg_pos_3` | −0.36 (better recent finishes) |

**When the model overrules the market it is keying on rating momentum and
relative rating strength** — recent trajectory and "rates better than this
field". These are the most public, most cheaply computed signals in racing,
which thousands of market participants also calculate.

And that basis is worse than useless. Conditioning on the quartile where each
driver most favoured the model's pick:

| driver | model pick's edge vs its price |
|---|---|
| `horse_elo_momentum` strongest | **−1.11 pp** |
| `horse_glicko_vs_field` strongest | **−1.16 pp** |
| `horse_glicko_rank` strongest | −0.38 pp |

### What is NOT wrong

- **Not calibration.** Model probability over actual win rate is 0.98–1.04× in
  *every* prior-starts bucket (0, 1, 2-3, 4-7, 8-19, 20+) and every momentum
  quintile — as tight as the market's own 0.98–1.02×.
- **Not a broken subgroup.** No exposure band, field size or race type shows
  an edge distinguishable from zero.
- **Not a bias toward unexposed horses.** The model's top pick has a median 9
  prior starts and is a debutant 3.0% of the time; the *market's* pick is a
  debutant 5.9% of the time. If anything the model under-backs newcomers.

### A fix was tested, and it failed

The model's most *confident* errors do cluster on lightly-raced horses with
spectacular ratings — a horse with one good run gets a huge
`glicko_vs_field` and momentum with no record to regress it toward the mean.
Bets on 1-start horses returned −11.57% against a −4.83% matched baseline.

But declining those bets does nothing:

| rule | n | ROI @ BSP | vs matched baseline |
|---|---|---|---|
| current (`combo_prob > mkt_imp`) | 104,146 | −3.55% | −0.87 |
| drop thin-form, hot-momentum bets | 96,837 | −3.31% | **−0.63** |
| the dropped bets alone | 7,309 | −6.70% | −4.02 |

0.24 points of improvement, and none of the segment effects survive a
race-clustered interval (1-start bets: CI[−20.10, −3.45] against a −4.83%
baseline that sits inside it). The tail is real-looking and too small and too
noisy to bank.

### Methodological note, recorded because it nearly produced a wrong answer

Three example races were first selected as "the model's most confident errors"
and all three showed the same vivid pattern: the model backing a 1-start horse
on a huge rating while a well-fancied debutant won. It was a **cherry-pick** —
that selection filter surfaces precisely those cases, and the aggregate
contradicts the story (see "What is NOT wrong"). Hand-picked extremes in this
project have repeatedly suggested structure that the population does not have.
Always put the aggregate next to the anecdote.

### Can it be resolved?

Not by calibration, not by excluding a segment, and not by better ranking —
no re-ordering of the same information can beat a price that already contains
it. The one fix this diagnosis points at directly is **§4c of the action
plan**: the model has no way to know *which of its features the market has
already priced*, so it spends its capacity, and its disagreements, on momentum.
Training with the market log-odds as a fixed offset (`init_score`) would force
it to learn only the residual. That is the untried item, and this is the
strongest argument yet for trying it.

Beyond that the ceiling is informational, not architectural (§1c).

## 2d. Debutants: the model can rank them, but only over a 14x range (2026-10-12)

Follow-up to §2c, asking whether the "well-fancied debutant" the market sees
is measurable from public data.

**The model is not blind to debutants.** On 18,130 debutant runners it reaches
**AUC 0.773** against the winner, and in races with two or more debutants it
picks the winning one 48.3% of the time against 19.6% for random.

**But debutants are its worst band relative to the market**, and the mechanism
is compression, not bias:

| debutant probabilities | sd | p10 | p90 | p90/p10 |
|---|---|---|---|---|
| model | 0.082 | 0.0132 | 0.1834 | **13.9x** |
| market | 0.108 | 0.0039 | 0.2102 | **53.6x** |

AUC by prior starts, model minus market: **0 starts −0.052** (the largest gap),
1 start −0.030, 2-3 −0.029, 4-7 −0.033, 8-19 −0.035, 20+ −0.043.

The market spreads debutants across a ~54x range of probability; the model
compresses them toward the 8% base rate. It is perfectly calibrated *on
average* over debutants (1.01x) while being unable to separate the 1.4 shot
from the 200/1 shot.

**Note it under-spreads at both ends.** Its p10 is 0.0132 against the market's
0.0039, so it also cannot identify the no-hopers — and that half of the problem
is a *public* one (weak yard, weak pedigree, weak jockey booking). It also
matters more than it looks: in a per-race softmax, pushing genuine 200/1 shots
down redistributes probability to the real contenders, so improving the bottom
end sharpens the top end for free.

### What public proxies exist, and what is missing

| signal | in store | allowlisted |
|---|---|---|
| `sire_*` | 14 | 5 |
| `damsire_*` | 14 | 4 |
| `dam_*` (the dam's own produce record) | 2 | **0** |
| sibling / progeny / produce form | **0** | **0** |
| first-time-out / debut features | 11 | **0** |
| `trainer_first_string_*` | 7 | **0** |
| `first_time_headgear` | 1 | **0** |

Two gaps worth acting on:

1. **Sibling form does not exist at all.** "Full brother to a Group winner" is
   the most-used public debutant signal in racing and the market leans on it
   heavily. `dam` is already in the results table, so it is computable: for a
   debutant, the career record of other horses sharing its dam — wins, best
   official rating, best class — strictly from races before today. Mind hard
   rule 1: this is a history feature and must exclude the current race, and
   siblings running on the same day must be date-strict.
2. **Everything debut-specific that already exists is invisible to the model** —
   the dam's produce record, all 11 debut features, all 7
   `trainer_first_string_*` columns. Free to test with
   `scripts/triage_features.py`.

**Expectation-setting:** sale price is the canonical public proxy for a fancied
debutant and it tested **dead on debutants specifically** (residual correlation
0.005, §5). That is real evidence against this whole family. Sibling form is
different in kind — racecourse evidence from relatives rather than a price —
but do not expect much.

## 2e. Public data audit: what is collected and never used (2026-10-12)

The store holds **1,178 columns**; the allowlist is **151**. Whole families sit
at zero, and the allowlist is three-quarters trainer/horse/jockey history.

**Collected and never shown to the model:**

| family | in store | allowlisted |
|---|---|---|
| sectionals (`atr_*`) | 216 | 0 |
| weather | 10 | 0 |
| going detail | 11 | 0 |
| headgear / first-time equipment | 4 | 0 |
| debut features | 11 | 0 |
| trainer-first-string | 7 | 0 |
| course / track detail | 22 | 3 |
| draw | 11 | 2 |

The important nuance: **the allowlist was selected for raw win prediction, not
for orthogonality to the market.** A family can be legitimately unimportant for
predicting winners and still carry residual signal, so "not allowlisted" is not
the same as "tested and dead". `scripts/triage_features.py --prefix <family>_`
settles each one in minutes.

**Not collected, though published:**

- **Stalls position — entirely absent.** There are 11 draw columns but no
  stalls side (high / low / centre) and no rail movement. **Draw bias is close
  to meaningless without it:** a low draw is decisive at one configuration and
  irrelevant at another, and rail movements change the effective trip. This is
  in every published going report and is the largest course-data omission.
- **GoingStick numeric reading — absent.** All 11 `going_*` columns are
  one-hots of the going *word*. The penetrometer value is published per GB
  meeting and is a finer-grained measure than the category.
- **Exact distance in yards** — only `distance_furlongs` is stored. Races are
  run over e.g. 1m 2f 7y, which matters for stamina at the margin.
- **Course configuration** — inner/outer, round/straight not captured.

Given §2b (sectionals priced) and the saturation finding, most of the
already-collected families will probably also be priced. The stalls-position
gap is different: it is a genuine collection omission that makes an existing
allowlisted feature (`draw`) weaker than it needs to be.

## 2f. Label audit: labels are clean, bar amended results (2026-10-13)

`plan.txt` Stage 1 asked for this and it was never done. Every accuracy figure
and every refutation in this ledger rests on `won` and `finish_position` being
right, so an unmeasured error rate means every reported gap carries an unknown
offset. Run with `scripts/audit_labels.py` over 749,296 runners / 79,121 races
(2020-09-15 to 2026-10-11), cross-checked against Betfair's own settled result.

**Internal consistency is perfect:**

| check | result |
|---|---|
| races with no winner stored | **0** |
| races with >1 winner | 128 (0.162%) — dead heats, legitimate |
| `won` disagreeing with `finish_position == 1` | **0** |
| races with fewer rows than the declared field | **0** |
| non-finishers (`fp <= 0`) | 6.79% of runners (falls, pulled up, unseated) |

**Against Betfair's settled result:** 586,970 runners matched (78.3%), **252
disagreements = 0.0429%**, near-symmetric (130 where we say won and Betfair
says lost, 122 the other way). Place labels against the exchange's settled
place market: 123 of 677,988 = **0.0181%**.

So the labels are not a source of error worth worrying about, and the
model-versus-market gap in §2c carries no meaningful label offset.

### The residual 0.04% is amended results, not scrape noise

**30October2026 update:** [repair042](research/AMENDED_RESULTS_042.md) now addresses
this previously documented source defect. The ordinary parser honours final
amendments, and primary-source checks qualify164 complete fields, including
131 changed winners and463 changed placings. The corrected history and its
combined current refit are installed as `20261030_015917_823047`. The frequency estimate below is
historical; it does not account for propagation into later horse, jockey and
trainer histories. There is no measured predictive-gain claim from042 yet.

The obvious explanation — dead heats — is wrong: only 9 of the 252 fall in a
race where our DB records multiple winners. The rest arrive in **pairs**, and
always the same shape:

```
2021-02-17 Dundalk    Barretstown      fp=2  ours=0  betfair=1
2021-02-17 Dundalk    Star Harbour     fp=1  ours=1  betfair=0
2021-04-24 Sandown    Potterman        fp=2  ours=0  betfair=1
2021-04-24 Sandown    Enrilo           fp=1  ours=1  betfair=0
```

Our DB stores the **first past the post**; Betfair settles on the **official
amended result** after a disqualification or placing reversal. ~243 runners is
~121 affected races, 0.15% of the history — consistent with the real-world rate
of amended results in GB/IRE, and spread evenly across years (36/52/51/30/64/10
for 2021-2026).

**Why it is worth knowing despite being tiny.** For *training* labels 0.15% of
races is immaterial. For *settlement* it is not neutral: the backtesters settle
on our `won`, so a bet on a horse later disqualified is credited as a win the
exchange never paid. The direction is **favourable to us**, which is the
direction that matters. The magnitude is well under 0.1 pp of ROI on a 104k-bet
run, so nothing in this ledger changes — but the cheap fix is to prefer
Betfair's settled `won` for settlement wherever it is matched, and that would
remove the bias entirely.

### A trap in `attach_bsp` worth knowing

`attach_bsp` assigns `out[c] = merged[c]`, so calling it with `cols=("won",)`
**silently overwrites the caller's own label** with Betfair's. An audit written
that way compares the exchange against itself and reports flawless agreement
forever. This was nearly shipped; `tests/test_audit_labels.py` now constructs a
deliberate one-race disagreement so the test fails if the label is ever
clobbered again.

## 2g. The never-allowlisted families: all priced (2026-10-13)

Acting on 2e. Every family that sits in the store and not in the allowlist,
triaged with `scripts/triage_features.py`. **43 signals across 12 families, and
not one clears the bar** (|partial| >= 0.01 with q <= 0.05):

| family | signals | strongest abs(partial) |
|---|---|---|
| `weather_*` | 7 | **0.0000** |
| `going_*` | 5 | 0.0019 |
| `draw*` | 7 | 0.0023 |
| `course_*` | 7 | 0.0028 |
| `trainer_first_string_*` | 7 | 0.0034 |
| `first_time_*` | 4 | 0.0035 |
| `dam_*` | 2 | 0.0011 |
| headgear / first-time equipment | 4 | 0.0049 |

So "in the store but not allowlisted" turned out, in every case, to mean the
same thing as "tested and dead". The allowlist was not leaving signal behind.

*Scope (2026-11-01):* that holds for signal **beyond the market**. For the
model's own accuracy, the old list did leave signal behind. Re-derived by
out-of-sample race log loss (048), 126 inputs beat it by 0.0194 nats at
production capacity on untouched quarters. All of that gain is priced: the model still trails the
SP by 0.095. See [048](research/FEATURE_SELECTION_048.md).

### Why the weather features are *exactly* 0.0000 - a structural lesson

All seven weather signals score 0.0000, not merely small. So do
`course_favours_leaders` and `course_quirk_score`. The reason is not that
weather is uninformative: **those features are race-level constants.** Every
runner in a race shares the same temperature, wind and course quirk, so they
have no within-race variance, and both this triage and any
per-race-normalised model cancel them exactly.

A race-level feature cannot differentiate runners by construction. It can only
pay through a **runner-specific interaction** - this horse's record in *these*
conditions. The two going features built that way, `going_wr_x_runs` (+0.0019)
and `going_specialist_edge` (+0.0011), are the only non-zero members of their
family, which confirms the mechanism while being far too small to matter.
`weather_wind_x_dist` scores 0.0000 because wind and distance are *both*
race-level, so the interaction is too.

**Consequence:** collecting the GoingStick number would add another race-level
column and cannot help on its own. Only worth it as an input to interactions
with each horse's own going record - and that family scores 0.0019, so expect
nothing.

**Qualification from the 2026-10-25 audit:** the statement above that *any*
race-normalized model cancels race constants is too broad. A common additive
logit cancels in softmax, and within-race centering removes constants from this
triage. The deployed model uses binary sigmoid probabilities before race
normalization: normalizing `sigmoid([0, 1])` gives `[0.4062, 0.5938]`, while
adding the same 2 to both logits gives `[0.4804, 0.5196]`. Context can therefore
change confidence even without changing rank. Trees can also learn contextual
interactions that a marginal screen does not test. The measured screening
results remain as recorded; they do not prove that every such feature is
irrelevant to probability scores. Any pruning or inclusion still needs a
proper chronological ablation.

## 2h. Sibling (dam-produce) form: priced overall; the debutant lead failed on the hold-out (2026-10-13)

Built because 2d found the model spreads debutant probabilities over a 13.9x
range against the market's 53.6x, and sibling form - "full brother to a Group
winner" - is the most-used public read on an unraced horse and **did not exist
in the store at all**.

`src/sibling_features.py` computes it from the `horse_pedigree` dam, strictly
self-excluding (a horse is not its own sibling) and date-strict (siblings
running the same day are invisible to each other, since their results are not
known pre-off). 64.7% of runners carry sibling history, averaging 17.5 prior
sibling runs.

**Overall: priced.** On 396,898 rows with sibling history, every signal is
under abs(0.005):

| signal | partial | t |
|---|---|---|
| `sibling_any_winner` | +0.0049 | +3.1 |
| `sibling_win_rate_shrunk` | +0.0035 | +2.2 |
| `sibling_best_or_vs_field` | +0.0031 | +2.0 |
| `sibling_runs` | +0.0013 | +0.8 |

### The one result that clears the bar - treat it with suspicion

Restricted to debutants and one-start horses (n=3,746):

| signal | partial | t |
|---|---|---|
| `sibling_best_or_vs_field` | **-0.0641** | -3.9 |
| `sibling_any_winner` | -0.0346 | -2.1 |
| `sibling_win_rate_shrunk` | -0.0160 | -1.0 |

Note the **sign**. Negative means that among unraced horses a *better* sibling
record goes with a *worse* result than the price implies - i.e. **the market
over-rates the well-bred newcomer**, and the implied trade is to fade them
rather than back them. That is the opposite of what the feature was built for.

**Why to be sceptical anyway:**

- It is a **subgroup examined after the headline test came back null**, which
  is the exact pattern that generates false discoveries and has already misled
  this project twice in one session (the cherry-picked examples in 2c, and the
  two-fold re-run in 2). With roughly 6 exposure bands x 7 signals available to
  look at, the multiple-testing bar is about abs(t) > 3.2, which -3.9 clears
  only narrowly.
- n = 3,746, so the standard error on the correlation is about 0.016.
- Correlation with the residual is **necessary, not sufficient** (see 0). Every
  refuted entry in this ledger cleared some correlation test first.

**The only thing that would settle it:** CLV at BSP under the locked
walk-forward, restricted to unraced runners, with the ROI-vs-CLV tripwire, on
the rebuilt store - this screen ran against the DB directly. Until then it is
a lead, not a finding.

### Hold-out verdict (2026-10-13): does not confirm

Tested once on the golden hold-out (races from 2026-06-11, never seen by any
residual analysis), with the population, statistic and bar written into
`docs/archive/SESSION_LOG.md` before the script existed
(`scripts/holdout_sibling_test.py`): runners with at most one prior UK/IRE
start and sibling history, in races with a complete BSP market; partial
correlation of `sibling_best_or_vs_field` with won - p_BSP, within-race
demeaned, price basis partialled out, race-clustered bootstrap SE.

| | n runners (races) | partial corr | SE | t | one-sided p |
|---|---|---|---|---|---|
| discovery (vs T-5 price) | 3,746 | -0.0641 | ~0.016 | -3.9 | - |
| **golden hold-out (vs BSP)** | **3,711 (848)** | **-0.0160** | 0.0179 | -0.89 | **0.19** |

**Fails** (bar: p < 0.01). The sign agrees, but the effect is a quarter of
the discovery's and its 95% interval, [-0.051, +0.019], excludes the
discovered -0.064: the winner's curse of a subgroup found after a null
headline, as this entry predicted. **The sibling lead is closed.** One
discrepancy recorded rather than explained: the same population definition
yields 61,558 runners in the development period, against the 3,746 the
discovery reported, so the discovery applied restrictions it did not
document.

### A bug the tests caught, worth recording

The first implementation shrank the sibling win rate toward **the mean of
whatever frame was passed in**. That breaks store-append parity: the same
settled row gets one value in a full build and a different one after an
append, because the population mean moves. The "settled rows do not change
when later races are appended" invariant in
`tests/test_sibling_features.py` caught it. The shrinkage target is now a fixed
constant, and any future `*_shrunk` feature must do the same.

## 3. `trainer_freq` stablemate leak — RESOLVED (2026-06-26)

A **+56.8% ROI@BSP** was traced to `trainer_freq`, built with a `cumcount`
that encodes same-race finish order — the feature knew the result. Fixed in
`150506e`; the store was rebuilt and re-validated: **+56.8% → +0.7%**.

Note the first suspect (`trainer_wins_*`, fixed in `ab70426`) was *not* the
real leaker. Caught by the ROI-vs-CLV tripwire, not by inspection.

## 4. Par-time / going-corrected speed figures — redundant (2026-06-27)

`winning_time` was backfilled to 99.9% coverage and the measured going
validated, then used to derive going-corrected speed figures (`tspf`). This is
the **most discriminative new feature family** built, and it is still
**CLV-flat** — the market already prices it.

**Keep the data** (it is good data, and it makes other speed features
better). Do not add the features to the allowlist.

## 5. Sale price (Tattersalls) — fully priced (2026-06-29)

Built a working Tattersalls sale-price scraper (4DCGI API). The feature is
dead: residual correlation **0.0000** overall, **0.005** on debutants. The
market prices public sale prices completely.

This sharpened the picture in §8: the debutant edge the market holds is
*genuinely private* information (gallops, fitness, intent), not a public price
anyone could scrape. **Do not scrape Goffs for this.**

## 6. Intent signals — all three dead (2026-07-03)

| candidate | verdict |
|---|---|
| Declared wind surgery | fully priced (residual t = −0.33) |
| Trainer yard→track travel | noise |
| Jockey booking | CLV 1.033 is the SP-vs-LTP **move artifact** — the feature is built from odds, so it is not tradeable |

The jockey-booking result is the instructive one: a feature derived from odds
can show apparent CLV purely because it proxies the price move itself. Data
kept; features not allowlisted. Store rebuilt at 1178 columns.

## 7. Stewards reports — redundant with form (2026-06-26)

BHA (token gotcha: `Origin` header + last `Bearer`) and IHRB (`sitemap.xml`)
report scrapers were built and verified. The derived "trouble in running /
unlucky" features proved redundant with ordinary form.

Better replacement, if revisited: derive trouble from **sectionals** instead —
a horse closing fastest at the line but beaten is the complete, full-coverage
version of the same signal, with no text parsing. That is what
`atr_closing_in_hand` measures (see 2b). `src/stewards_features.py` and the two
report scrapers were deleted on 2026-10-13 — recover from git if revisited.

**Caveat (055, 2 November 2026): the Irish half of this test had almost no
data.** In the surviving `data/processed/stewards_events.parquet`:

- 91% of IHRB `horse_key` values are jockey or trainer initials ("A.P.O'",
  "M.J");
- 0.3% of IHRB events are fixture-matched;
- IHRB coverage effectively ends in 2020;
- the `non_trier` tag occurs once in 81,545 rows.

The verdict therefore stands for BHA (Britain) only. A repaired IHRB parse
would mainly add non-trier rulings, which matter for individual horses. Glin
Road Boy, one of the 055 misses, was such a case. The rulings are too rare to
move race log loss much.

## 8. Why the model loses to the market — characterised (2026-06-27)

The market beats the model specifically on **favourites and debutants**, and
the gap vanishes once the private information is exposed. The **price move**
captures about half the accuracy gap, but:

- it is **not tradeable** (move continuation correlation ≈ 0.01), and
- it destroys orthogonality, since the feature is built from odds.

Gotcha worth remembering: `market_id` is a **meeting** key, not a race key —
normalise within `(date, off_time, track)`.

## 9. Feature-store saturation (2026-06-26)

The store is effectively exhaustive for public form. Adding the **173 unused**
stored features produces **no CLV improvement**. The one genuinely untapped
area identified is **market microstructure** (`ltp_*` columns in the BSP file:
traded volume, ladder depth, timing and shape of the move) — still unexplored.

## 10. Quality-control gates (2026-06-22)

The thin-history gate was built, then quantified as **unhelpful**; it defaults
**OFF**. Related finding: a **form-only model has no edge**. Burn-in is now
absolute rather than relative.

## 11. Course-dynamics features (2026-06-22)

A new course-dynamics FE stage produced 18 candidate features; **4 made the
allowlist** with marginal positive lift. Not a breakthrough, but not dead.

---

## 12. Consolidation: what was deleted and why (2026-10-13)

The repository had accumulated one live pipeline and a long tail of side
experiments, with no way to tell them apart by reading the tree. 32 files were
deleted and two live modules pruned — **≈6,100 lines in all**. **Nothing here was removed because it looked
unused — every item was either refuted with a number in this ledger, or
provably unreachable.** The code is in git history; the measurements are here.

### Refuted strategies and feature families

| Deleted | Evidence |
|---|---|
| `place_market_model`, `paper_trading`, 3 scripts, 2 tests | 1 — the +15% was a stale-store artifact; on correct features +1.5% with a CI spanning zero, going negative. Opt-in and unrun since June (one 290-byte paper-trade log). |
| `stewards_features`, `bha_stewards_scraper`'s report half, `ihrb_daily_enquiries_scraper` | 7 and 2b — "unlucky" features are redundant with form, and the sectional version (`atr_closing_in_hand`) beat the text version while both were tiny. |
| `par_time_features` | 4 — the most discriminative new family measured, and CLV-flat. |
| `tattersalls_sales_scraper`, `build_sale_price_join`, `test_sale_price_signal` | 5 — residual correlation 0.0000; the debutant edge is private information, not public sale price. |
| `backfill_wind_surgery`'s feature consumers, `test_intent_signal` | 6 — wind surgery fully priced, travel is noise, booking's CLV was an SP-vs-LTP artifact. (The backfill script is kept: the DB column stays.) |

### Provably unreachable

- **`rtv_scraper` + `add_rtv_features`** (≈890 lines). ATR replaced Racing TV as
  the sectional source and `add_rtv_features` was no longer in the stage list,
  so the function could not run. Removing it is a **provable no-op on feature
  values**: the four `rtv_acceleration_*` names sat in a list filtered by
  `if col in df.columns`, and the `has_rtv_history` block behind
  `if "rtv_top_speed_avg_3" in df.columns` — and those columns cannot exist when
  the stage that creates them never runs. `src/app_helpers.py` also imported
  eight RTV names and used none of them, three being a **duplicate set of merge-key
  normalisers** (`_normalise_horse_key` and friends) pulled from the retired
  scraper instead of `src/utils.py` — a quiet hard-rule-3 violation that only
  showed up when the scraper was removed.
- **`strategy_calibrator`** (717 lines). Zero importers, and it grid-searched
  betting parameters **on the test set** — the overfitting trap that produced
  two of this project's false edges. Not worth repairing.
- **`config.MODEL_FILE` / `SCALER_FILE` / `FEATURE_COLUMNS_FILE`** and seven
  March-2026 `models/` artifacts (49 MB) from the removed LambdaRank path:
  referenced nowhere, and actively misleading about which model serves.
- **17 one-off scripts** whose questions are answered here: `train_600d*` (two
  pointed at a run snapshot that no longer exists), `ab_market_add`,
  `ab_race_model`, `probe_stablemate_leak` (leak found and fixed in 150506e),
  `repair_feature_leakage_snapshot` (superseded by the rebuild), `audit_edge`,
  `gpu_tabular_benchmark` (needs a bundle that was never kept),
  `wf_each_way` (hardcoded snapshot **and** a hand-rolled fold loop, which hard
  rule 6 exists to forbid), `cleanup_non_runners` and `retrain_clean_store`
  (completed repairs).

### Dead code *inside* live modules

Module-level deletion missed the larger share. A reference scan over every
top-level function in `src/`, re-run after each pass so a helper used only by
something already deleted is caught on the next one, found:

- **`src/app_helpers.py` was 63% dead** — 882 of 1,404 lines. 17 orphaned
  chart/frame builders, then 14 more that only those called: the Insights and
  Diagnostics panels of the retired monolithic `app.py`. The current panels
  never referenced them, which is why nothing failed and nothing noticed.
- **`src/live_prediction.py` carried two caches, one superseded.** The
  `live_feature_cache_*` family (5 functions, 48 lines, plus two directories it
  created on import) was replaced by the `lookahead_cache_*` family that
  `app_helpers` actually imports. Two cache systems in the serving module, one
  dead — the single most confusing thing in the live path.

Six single orphaned functions were **kept** on purpose; they are listed in
`docs/archive/PROJECT_OUTLINE_before_cleanup_2026-11-05.md` §4.3. A reference count of zero is a prompt to look, not
a licence to delete: a settlement formula, an append-machinery helper and a
market-anchor teacher all have obvious future callers, and stripping a
module's API surface on a grep is how a codebase loses optionality.

### `_FE_UNWIRED_SOURCES` is gone, and that is the point

`featured_dataset` carried a documented exemption list so that editing
`par_time_features` or `stewards_features` would not force a multi-GB rebuild.
That mechanism existed **only** to work around dead code. Deleting the code
removed the need for it, along with its test. The real guard —
`test_every_fe_dependency_is_hashed_or_explicitly_exempt` — is untouched.

The general lesson: a workaround that protects dead code is a signal to delete
the code, not to keep the workaround.

### Cost of doing this mid-build

The deletions touched `feature_engineer.py` and `featured_dataset.py` while a
full store rebuild was running, so the finished store (749,296 rows, 1,178
features, through 2026-10-11) carries `fe_signature` `724ce1afe9ad7611` against
a current `3e3b9bbb5d79bacc`. The guard will correctly force a rebuild. It was
**not** hand-re-stamped, even though the reasoning above says the values are
unaffected: re-stamping a store on an argument rather than a measurement is
exactly the move that produced the phantom +15% place edge. Spend the rebuild.

## 13. Phase A audit: what the pipeline was silently getting wrong (2026-10-13)

A stage-by-stage audit (NEXT_AGENT_BRIEF §3). Nothing below changed a
refutation in this ledger; several items change what the data and the
instrument can be trusted for.

### The results database was missing races, systematically

The old date-level sync kept **only the first 50 listed races per day**
(`race_links[:50]`) and skipped any race page whose fetch raised, after which
the date counted as present and was never revisited. Neither path logged
anything louder than a warning.

| | days with exactly 50 races | days with more than 50 |
|---|---|---|
| 2020-21 (capped sync) | 50 | **0** |
| 2022-25 (an uncapped collection path) | 6-10 a year | 30-38 a year |
| 2026 (capped sync, incl. the June-October refresh) | 25 | 5 |

Histogram of races per day: 53 days at 49, **105 at 50**, 24 at 51. A dry-run
relist of 30 sample dates found 32 missing races; the listing puts UK meetings
first, so the cap cut **Irish** cards (Navan, Cork, Limerick, Sligo, Kilbeggan,
Tramore). The skip path is separate and hits any day: 2021-08-08 was missing a
whole 7-race Haydock card on a 28-race day, and **2022-12-17 (42 races) was
missing entirely**. Sampled 2022-25 dates came back complete.

Why it matters despite being under 1% of races: every horse that ran in a
missing race has a hole in its history (wrong days-since-run, a skipped rating
update, a missing form line), the market saw that run and the model did not,
and the 2026 holes fall on exactly the horses live serving prices.

Repaired by a full relist through the new resumable ledger
(`python -m src.database --audit-history --sync 2219`): **543 races / 6,410
runner rows added** (2020: 44, 2021: 229, 2022: 27, 2023: 27, 2026: 216, of
which 47 are the newly synced 2026-10-12), all at the truncated meetings
(Navan 53, Limerick 43, Killarney 40, Tramore 35, Punchestown 32, ...). Races
per day around 50 is now smooth (49: 41, 50: 36, 51: 42) and 2021 has 46 days
above 50 instead of none. Correction to the dry run: 2022-12-17 was *not*
missing - all six cards were abandoned in the freeze.

3,472 listed races still yield no result: 3,084 in meetings with nothing
stored, 388 isolated inside stored meetings. A random sample of 80 found
**none that ran and was missed**: ABANDONED 54, DORMANT 19, RACEVOID 3,
Arabian 2, HTTP 404 2 (95% upper bound on genuine misses ~3.7% of the set).
The sync now closes such races with their reason (`RaceNotRun`), so its
"pending" count means "failed, unexplained".

**Checked against an independent census.** Betfair's public files list every
GB/IE win market, so `scripts/betfair_db_audit.py` run against them is a
full-history completeness test of the scraper: 79,579 of 79,693 Betfair races
(99.9%) are in the DB. Of the 114 absent, 45 ran on 2020-09-14, the day before
the DB starts; 66 are Arabian races the DB excludes by design; **3 are genuine
omissions** (Chepstow 2023-04-10 13:35, Bellewstown 2024-07-05 18:35, Curragh
2025-06-27 19:30), races Sporting Life never listed. The same audit exposed a
join bug: `src/betfair.py` lacked the 2020 abbreviations `Aint`, `List`, `Nott`,
so 124 Aintree/Listowel/Nottingham races (Sep-Dec 2020, public files only - the
BASIC parquet never used them) could not join. Fixed.

### The flat draw backfill was finished, not pending

`draw_backfill_done` covers 46,791 flat races through 2026-06-13, and the
3,645 later flat races were scraped by the fixed scraper. Verified on the data
rather than the checkpoint: the per-race Spearman correlation between `draw`
and weight carried in flat handicaps is **~0 (|mean| < 0.06) in every month
from 2020-09 to 2026-10**. Saddle-cloth numbers, which the old bug stored,
are assigned by weight in handicaps and would give ~-1. `draw` is a real stall
draw everywhere.

### The feature build was not a pure function of the database

`add_weather_features` called the Open-Meteo API at build time for rows with
no stored weather, and filled whatever failed with constants (14.0 C, 1.5 mm,
...); `add_trainer_travel_features` read an unsigned CSV. Neither input is in
`data_signature`, so two builds of the same database could differ. Both
families were already measured dead (2g: every weather signal 0.0000; 6:
travel is noise). Both stages were removed after a parity check: engineering
106,467 rows / 10,460 races with and without them drops **exactly their 10
columns and leaves the other 1,058 bit-identical**. `weather.py`,
`trainer_locations.py` and `bha_stewards_scraper.py` left `_FE_SOURCE_FILES`.

### `fe_signature` changed with line endings

It hashed raw bytes; the tree holds a mix of LF and CRLF sources and git
autocrlf rewrites them on checkout, stash or worktree add. Converting one file
(`feature_store.py`) moved the signature `3e3b9bbb...` to `e7f6193d...`. That
fails safe but forced ~56-minute rebuilds for no change, and made signatures
incomparable between checkouts. Now normalised; identical in the main tree and
a fresh worktree.

### Live picks silently dropped every horse's history on a fresh scrape

Found by running the serving path end to end (brief Stage 8) on the clean
production model. A **fresh** racecard scrape returns `horse_id` as text; the
racecard CSV cache, and the training store (whose history also arrives
through a CSV round trip), hold integers - and the store's carried rating and
counts state is keyed by those integers. Featurising the same 351-runner card
both ways: **25 of the model's 149 features differed, on 68-98% of runners**
(days since last run, previous-race counts, distance/weight/class change,
place rate, rating interactions); `travel_from_last_km` was 100% missing.
In the picks app a fresh scrape happens on the first load of each day and on
every "Refresh odds", so those prices were computed for horses with their
history erased - with no error anywhere. No test compared a fresh scrape with
a cached one.

Fixed at two layers outside the FE signature: `get_scraped_racecards` now
returns a fresh scrape through the cache's own CSV parse, and
`features_for_live_cards` coerces `horse_id` to the store's integer form
(`tests/test_serving_safety.py` covers both). Re-verified end to end through a
forced fresh scrape: integer ids, `travel_from_last_km` 93% present, no model
feature mostly missing, per-race win probabilities summing to 1.

### `promote` let unverifiable runs through

It refused a feature-generation **mismatch** but promoted a run with **no
recorded signature** after a log warning. All 8 saved runs predate the
`trainer_freq` leak fix (2026-06-26) and 6 carry no signature, so one click
would have served a leaky model. Unverifiable runs now need the same explicit
override as a mismatch.

### Plan 4b's class-weighting hypothesis describes code that does not run

ACTION_PLAN 1b led with "the win classifier trains with scale_pos_weight ~
(N-1)". It does not: every call site uses `_train_win_classifier`, which
passes no class weight to LightGBM, and the LightGBM place model is unweighted
too (only the XGBoost paths take `scale_pos_weight`; the focal-loss trainer is
never called). "Drop class weighting" is already the configuration. What is
left of 4b is regularisation and learning rate.

### Seven allowlisted features are race-level

`base_win_rate`, `base_place_rate`, `num_runners`, `race_type_Chase`,
`handicap`, `field_std_or` and `field_avg_elo` never vary within a race;
`weight_per_furlong` (0.7% of its variance within-race) and `or_x_handicap`
(1.6%) nearly never do. In the current binary GBM they are not dead by
construction: `num_runners` sets each runner's base rate before per-race
normalisation, and the rest can act through interactions. Under a per-race
softmax or a market offset (plan 4c) their main effects cancel exactly. The
allowlist holds 150 features, not the 151 documented.

### A free closing-price source, and the first golden hold-out

Betfair publishes daily BSP files for anyone
(`promo.betfair.com/betfairsp/prices`, `scripts/fetch_betfair_sp_public.py`):
per runner, BSP, morning and pre-play volume-weighted prices, price ranges,
and **matched volume** (morning, pre-play, in-play) - volume the BASIC tars do
not carry at all. All 4,440 files for 2020-09-14..2026-10-12 (GB and IE)
downloaded; validated:

| check | result |
|---|---|
| BSP vs the BASIC parquet, 589,123 shared runners | 99.99% within 1% (BASIC rounds to 2 dp) |
| winner flag vs BASIC / vs our DB | 99.99% / 99.94% |
| DB runners matched, by quarter | 95-100% in every quarter, incl. the BASIC gaps 2020Q4 (94.9%), 2021Q3 (97.9%), 2021Q4 (96.9%) |
| golden hold-out (races from 2026-06-11) | 4,812 races, 96.2% with a complete closing and morning market |

Three traps, each silent if missed and each tested: the file is dated the day
after its races; off times are UK local while the DB is UTC (every BST race
would fail to join); two header styles.

Two measured facts any morning-price protocol must respect:

- A runner with no morning trade carries `morningwap = 1.0`, a placeholder
  (2.5% of runners). Requiring a valid price drops any race containing one.
- **Pre-off prices sit well short of BSP on the typical runner**: median
  morning VWAP / BSP = 0.8925 (geometric mean 0.8236); pre-play VWAP / BSP =
  0.88. Most runners are outsiders, whose BSP routinely exceeds the traded
  price. This is why bet-all CLV at T-5 is already 0.9786 (2), and it makes
  ACTION_PLAN 7's "median CLV > 1.0" a much higher bar at morning prices.

**The golden hold-out** (`config.GOLDEN_HOLDOUT_START = 2026-06-11`,
committed in `3800c33` before any Phase B experiment) qualifies because every
CLV and market-residual harness drops rows without a Betfair price, and the
hand-downloaded data stopped on 2026-06-10: no such analysis has ever seen
those races. They have closing and morning prices now, but **no T-5 price**
(`ltp_300s` exists only in the BASIC tars). A T-5 result can therefore only be
confirmed on the hold-out once the operator downloads BASIC data for June to
October 2026; a morning-protocol result can be confirmed now.

### Stalls position and rail movements: collected, not yet tested (plan 5a-bis)

The BHA API behind britishhorseracing.com publishes each GB fixture's going
report: stalls position by distance, rail movements, watering, and (per
fixture, on a second endpoint) GoingStick with occasional inside / centre /
outside splits. `scripts/fetch_bha_going.py` collected all **9,311 fixtures
2020-09..2026-10**; stalls text is present on 96.8% of flat fixtures and the
fixtures join to **100% of the 8,682 GB meetings** in the DB. The API
enforces a long-window quota (HTTP 429 for every request for over an hour once
tripped); at 5 s per request it completed.

Not done, deliberately: turning the text into a draw signal. The stalls field
is free text in 1,253 forms ("7f 36y outside; remainder inside", "straight
course - stands side, round - inside"), and which stall numbers sit nearest
the rail depends on each course's handedness and straight-course layout - days
of careful work for a family whose prior is low (every racecard prints the
stalls position; draw features scored at most 0.0023 in 2g). GoingStick was
not fetched (~9,300 requests, ~13 hours at the tolerated rate) because a
single reading is race-level and cancels (2g). The data is in place for
whoever takes this on; see docs/archive/ROADMAP.md.

### Checked and clean

- Arabian purge held: 0 `(A)` runners, 0 Arab race names, 0 "Extras" tracks.
  Sporting Life files Arabian races under a "TRP Extras" pseudo-course that
  passes the UK/IRE filter; they are now dropped at listing time as well.
- Pedigree: 100% sire/dam coverage for debutants in every quarter 2024Q3-2026Q4.
- `data_signature` excludes exactly 31 derived one-hot columns, each
  reproducible from a categorical that is hashed; no other column matches.
- `attach_bsp` now refuses to replace a column the caller owns (2f's trap)
  unless `overwrite=True`; no current caller overlaps.

## 14. Market microstructure, free version: the close absorbs it all (2026-10-13)

ACTION_PLAN 5a's hypothesis is that *where and when money enters* carries the
private information that beats the model. Betfair's public files add matched
volume (morning, pre-play, in-play), so a first version is testable without
the paid ADVANCED/PRO tiers. `scripts/microstructure_screen.py`, development
period only (574k runners / 64k races, 2020-09-15..2026-06-03), six tests
written down before the run, each either **continuation** (does a signal known
at the decision point predict the move after it?) or **orthogonality to the
close**. Within-race demeaned, price basis partialled, race-clustered
bootstrap SEs (calibrated on null data: t mean 0.01, sd 1.10).

"Excess volume" = log(runner's share of the race's morning matched volume) -
log(its morning market share): money beyond what its price would attract.

| test | partial corr | t | reading |
|---|---|---|---|
| T1 excess volume -> move morning VWAP -> BSP | +0.169 (liquid runners +0.103) | +69 | morning money predicts where the price goes |
| **T2 excess volume -> won - p_BSP** | **+0.0004** | +0.4 | **nothing beyond the close** |
| T3 excess volume -> won - p_morning | +0.015 (liquid +0.021) | +13.5 | real information, ahead of the morning price |
| T4 move morning -> T-5 -> move T-5 -> BSP | -0.033 | -9.2 | reversal: bid-ask bounce (the same noisy LTP enters both moves with opposite signs) |
| T5 excess volume -> move T-5 -> BSP | +0.014 | +8.1 | small continuation after T-5 |
| **T6 move morning -> T-5 -> won - p_BSP** | **+0.0007** | +0.5 | **nothing beyond the close** |

**The closing price contains all of it.** Morning money is informative - it
predicts the result better than the morning price (T3) and predicts the later
move (T1, T5) - but BSP has fully absorbed it by the off (T2, T6 ~ 0). For a
bettor settling at BSP there is nothing here. What remains is *timing*: taking
an earlier price on the horses the money is backing, before the close catches
up. T5 says that effect exists after T-5 but is small. Its economic size is
measured below.

### The timing edge, priced (`scripts/microstructure_clv.py`)

One rule per protocol, fixed before the run: in each race, back the runner
with the highest excess morning volume if its decision price is 1.5-50;
matched baseline = every runner in that band. Development period, 2%
commission, race-clustered CIs.

| protocol | rule | n | median CLV [95% CI] | ROI at taken price | quarters + |
|---|---|---|---|---|---|
| T-5 LTP -> BSP | bet all in band | 411,632 | 0.9809 [0.9801, 0.9816] | -2.29% | 2/21 |
| T-5 LTP -> BSP | **top excess volume** | 55,412 | **0.9953 [0.9932, 0.9978]** | -2.15% [-4.24, -0.12] | 6/21 |
| same, races off >= 14:00 UTC | top excess volume | 41,458 | 0.9954 [0.9929, 0.9985] | -2.13% | 6/21 |
| morning VWAP -> BSP | bet all in band | 498,798 | 0.9210 | -6.11% | 0/24 |
| morning VWAP -> BSP | top excess volume | 64,209 | *1.0187 [1.0157, 1.0217]* | *+1.59% [-0.30, +3.44]* | *15/24* |

**At T-5 the timing edge is real and too small to bank**: +1.4 CLV points over
the matched baseline, CIs well separated, robust to dropping every race near
the morning cutoff - and still below 1.0, with ROI no better than backing the
band. Fails ACTION_PLAN 7.

**The morning row is an artifact, and was labelled one in the script before it
ran.** The signal uses the whole morning's volume, which is only known when
the morning ends, while `morningwap` averages the whole morning's prices -
including the longer ones a steamer traded at before the money arrived. It
credits the bettor with prices that had already gone: the jockey-booking trap
(6) in a new form. **Do not read 1.0187 as a result.** The honest version
strikes the bet at the price available when the morning closes; Betfair does
not publish the cutoff, but the morning share of pre-play volume collapses for
races off after ~11:00 UK (median share 1.24 before 11:00, 0.20 at
11:00-11:30, 0.10 at 11:30-12:00), so `ltp_1100` is being extracted from the
BASIC stream for that test.

### The honest early-price tests: break-even at best

> **Corrected 2026-10-14 (ledger 20): not break-even.** These tests compared
> the 11:00 price with BSP without the reduction Betfair applies to matched
> odds when a runner is withdrawn later (17% of these runners). With it, the
> m11 rule's median CLV is **0.9849 [0.9822, 0.9877]** and its ROI **-1.84%
> [-3.75, +0.12]**. The table below is kept as recorded.

`scripts/extract_betfair_bsp.py` now also records the last trade at or before
11:00 UK (`ltp_1100`, present for 96.4% of runners) and at T-60. Same rule,
same matched baseline, development period, races off at least 30 minutes
after the decision:

| decision price | rule | median CLV [95% CI] | geo CLV | beat close | ROI at taken price | quarters + |
|---|---|---|---|---|---|---|
| 11:00 UK | bet all in band | 0.9524 | 0.9143 | 43.4% | -2.04% | 2/24 |
| 11:00 UK | **top excess volume** | **1.0000 [1.0000, 1.0011]** | 0.9711 | 49.7% | **+0.85% [-1.12, +2.85]** | 13/24 |
| T-60 | top excess volume | 0.9927 [0.9902, 0.9949] | 0.9708 | 47.6% | -1.89% | 8/24 |
| T-5 | top excess volume | 0.9958 [0.9939, 0.9980] | 0.9801 | 47.6% | -2.12% | 8/24 |

**The closest thing to a positive result this project has produced, and it is
not an edge.** Struck at 11:00, following the morning money lifts CLV ~5 points
over the matched band and reaches break-even - and the value is gone by T-60,
the shape information absorption predicts. But it fails ACTION_PLAN 7: the
median's interval *touches* 1.0 rather than excluding it, the geometric mean
is below 1, it beats the close less than half the time, and ROI's interval
spans zero. The median sits on 1.0000 partly because of ties: BASIC rounds BSP
to the two-decimal grid LTPs trade on, and ties concentrate in the short-priced,
heavily traded runners this rule selects - read the geometric mean and the
beat-close rate alongside it. Capacity would be thin anyway: the median
runner trades about GBP 345 in the whole morning.

Confirming even break-even on the golden hold-out needs BASIC data for June to
October 2026 (the hold-out has volume but no 11:00 price) and, at ~4,000
selections, could not resolve an effect of +0.85% (ROI CI roughly +-6 points).

### A data gap found on the way: the canonical BSP parquet was incomplete

`data/processed/betfair_bsp.parquet` was written at 17:09 on 2026-06-21; the
last two BASIC tars arrived at 17:38 the same day and were never extracted.
The re-extraction found **99,917 runners** missing - exactly 2020Q3-Q4 and
2021Q3-Q4, the "gaps" every T-5 analysis in this ledger worked around. It cost
those analyses power, not correctness (a missing early quarter thins the
training data; it cannot bias a later test quarter). The full extraction is
now canonical (the old file is in `data/backups/`); BSP and `ltp_300s` agree on
99.998% of the 646,895 runners the two share.

## 15. The `sex` look-ahead leak, and plan 4c without it (2026-10-13)

### The third phantom edge, caught by the tripwire

Plan 4c - a per-race conditional logit with the market's log-probability as a
fixed offset, so trees can only learn the market's residual
(`scripts/wf_market_offset.py`, T-5 protocol, 13 purged quarterly folds
2023Q2-2026Q2, 40,462 races, pre-registered rules) - first came back with:

| rule | n | ROI @ BSP [95% CI] | median CLV |
|---|---|---|---|
| edge > 0.10 | 13,681 | **+11.62% [+3.69, +19.77]** | 0.9930 |
| top-edge | 33,699 | +5.58% [+1.17, +9.97] | 0.9972 |

with a race-log-loss gain of only -0.00040 and median CLV *below* 1.0: the
ROI-vs-CLV tripwire (0.3) fired. The residual model's top features in almost
every fold were `sex_H` and `sex_F`.

### The leak

The `sex` column is the horse's sex **on the day its results page was
scraped** (669k rows in March 2026), not race day:

- 2,842 two-year-olds are coded as mares and 255 as entire horses (both mean
  five or older); 4.76% of rows carry an impossible age/sex pair, falling
  toward the scrape date: 6.95% for 2020 races, 6.44% for 2023, 1.85% for
  2025, 0.06% for 2026;
- only 2.3% of horses with 8+ runs ever show a second code, although every
  filly becomes a mare and every entire colt a horse at five.

Sporting Life freezes a retired horse's code and updates the active ones, so
the code records the horse's *future*: "c" = a colt never gelded (kept entire
for stud value), "g" includes colts gelded later, "m" on a young filly = she
raced on past four, "f" on an old race = she retired young. **Model-free, the
21,292 runners with an impossible race-day code returned +21.5% at BSP
[+15.5, +28.2].** By sex code in 2020-23, entire males ("h") +43.7%, fillies
("f") -11.9%; the gap narrows toward the scrape date.

Every other per-race field was checked the same way and is as-of-race: trainer
changes over 47% of careers, official rating 99.9%, headgear 84%, form 100%
(and the next race's form string ends with this race's position 96.6% of the
time).

### Plan 4c without the leak

Same harness, every `sex_*` feature excluded:

| | with `sex_*` | **without** |
|---|---|---|
| race log loss vs market | -0.00040 | **-0.00000 [-0.00030, +0.00020]** |
| edge > 0.10 | +11.62% (13,681 bets) | **-4.98% [-19.83, +10.59]** (2,958) |
| top-edge | +5.58% | +2.01% [-2.17, +6.31] |
| edge > 0 | +0.69% | -1.50% [-3.20, +0.21] |

**The market-offset model learns nothing the market does not already know.**
Early stopping keeps 10-100 trees per fold - the residual signal is so weak
the model stops almost at once. Plan 4c, the best-motivated untried idea in
the project, fails cleanly: not a configuration problem but an informational
one.

**Re-confirmed on the rebuilt store** (fe `63d88ff721a02632`, birth-sex
`sex_Female`, and the now-complete Betfair coverage, which adds three early
quarters: 16 folds 2022Q3-2026Q2, 49,554 races):

| rule | n | ROI @ BSP [95% CI] | median CLV | folds + |
|---|---|---|---|---|
| race log loss vs market | - | **-0.00010 [-0.00040, +0.00020]** | - | 9/16 |
| bet all (band) | 378,152 | -2.17% [-3.00, -1.36] | 0.9783 | 3/16 |
| edge > 0 | 194,962 | +0.12% [-1.44, +1.68] | 0.9852 | 7/16 |
| edge > 0.05 | 41,590 | +1.82% [-2.49, +6.13] | 0.9926 | 6/16 |
| edge > 0.10 | 9,121 | +6.50% [-3.85, +16.92] | 0.9976 | 9/15 |
| top-edge | 40,151 | +1.53% [-2.38, +5.59] | 0.9933 | 7/16 |

Every interval spans zero and every median CLV is below 1. The residual
model now leans on `or_vs_field_x_hcap`, so the official rating was checked
for the same kind of leak: it is race-day. 86.7% of handicap runners sit
exactly on their race's rating-to-weight line, winners and losers alike
(86.9% vs 86.7%), and the handicapper's reassessment appears in the *next*
race's rating (winners +4.62 lb, losers -0.84).

### The last shortlisted sectional signal, given its economic test

`atr_closing_in_hand_avg_3` (+0.0115 partial, 2b) as the residual model's
*only* feature (`wf_market_offset.py --only`), on races that carry sectionals:
4 purged folds 2025Q3-2026Q2 (sectionals start 2023-08, so the locked
8-quarter burn-in leaves four), 9,462 races. Race log loss vs market
**-0.00000 [-0.00040, +0.00030]**; edge > 0.10 finds 63 bets; every rule's
median CLV is *below* the matched band's (0.9750-0.9773 vs 0.9811). A partial
correlation of 0.0115 is real and worth nothing. **Closed** - the last of the
brief's two unsettled leads (the other, sibling form, failed on the hold-out).

### Consequences

- **Fixed in `e689d98`**: sex is collapsed to the sex at birth
  (`sex_Female` / `sex_Male`) - the only part history supports. Live cards
  carry race-day sex, so this also removes a train/serve skew. The
  never-working `first_time_gelded` is removed rather than "fixed" into a
  leak.
- **Every model trained on the allowlist since `sex_F`/`sex_H` entered it has
  used future information**, including the production run promoted earlier
  today. It is retrained on the rebuilt store.
- **No refutation in this ledger is weakened.** The leak flattered the model,
  so every recorded negative (market combination -3.55%, the disagreement
  analysis, ...) was measured on a model better-informed than any real one
  could be. They stand, and would only get worse.
- **The tripwire is now three for three.** High ROI with CLV at or below 1.0
  has meant a leak every time it has fired here.

## 16. GBM vs logistic regression vs the market (plans 3a, 4b) (2026-10-13)

ACTION_PLAN 1b's diagnosis rested on 646 races where the production GBM ranked
*worse* than logistic regression (top-1 0.268 vs 0.292, ~1.3 sigma).
`scripts/wf_ranking_compare.py`, clean store, the eight most recent
development quarters (2024Q3-2026Q2), 24,405 races with a complete T-5
market; same training rows, recency weights and early stopping for every
model; per-race paired differences with race-bootstrap CIs:

| model | top-1 | top-3 | race log loss |
|---|---|---|---|
| production GBM (`CLASSIFIER_PARAMS`) | 0.2930 | 0.6327 | 1.8768 |
| loosened GBM (lr 0.05, 63 leaves, light regularisation) | 0.2898 | 0.6245 | 1.8894 |
| logistic regression (production baseline pipeline) | 0.2893 | 0.6225 | 1.8979 |
| **market (T-5, overround-free)** | **0.3396** | **0.6830** | **1.7609** |

| pair | top-1 [95% CI] | race log loss [95% CI] |
|---|---|---|
| GBM - logistic | +0.0036 [-0.0007, +0.0077] | **-0.0211 [-0.0241, -0.0182]** |
| loosened - production GBM | -0.0032 [-0.0066, +0.0002] | **+0.0126 [+0.0107, +0.0144]** |
| GBM - market | **-0.0466 [-0.0522, -0.0413]** | **+0.1159 [+0.1103, +0.1217]** |

- **3a: the GBM does not rank worse than logistic regression.** It ties on
  top-1 and is clearly better on race log loss. The 646-race result was
  noise; the prior written before the run expected exactly this.
- **4b: the configuration is not costing discrimination.** Class weighting was
  never on (13); loosening the regularisation makes log loss significantly
  worse.
- **Gate 2 fires.** With weighting and regularisation addressed the model ranks
  no better than a linear model on the same 149 features, and both trail the
  market by about five points of top-1. The feature set, not the model, is the
  binding constraint - and gate 3 (public form priced) and gate 4 (residual
  training and free microstructure fail) say the reachable features are
  exhausted.

## 18. Non-finishers were dropped from training and evaluation - the fourth tripwire catch (2026-10-14)

**Found** in the retrain after the concurrent session's commit: the training
report showed each-way bets at **+79.3% ROI (483 bets) with a closing-line
value of 0.78** - high ROI with CLV below 1, the pattern that has meant a
leak every time it has fired here. It had been in every training report
(the previous promoted run: +82.8%).

**The leak.** Since the first commit, `prepare_multi_target_data` (training,
the training app's evaluation, autotune, `model_lab`) and the backtester's
`walk_forward_validation` kept only rows with `finish_position > 0`.
Non-finishers - fell, pulled up, unseated, refused, stored as position 0 - are
**23.6% of chase runners, 15.2% of hurdle runners and 0.5% of flat**
(2020-09..). So the test set held only horses already known to have
completed, and a bet on a faller could never lose. The clearest case: Hexham,
2025-10-03, a nine-runner chase in which six failed to finish; the test set
held the three finishers, the place model rated each of them ~35%, and the
each-way strategy backed all three - first, second and third. Training was
biased the same way (P(win | completes), served on every declared runner).

**Fixed in `c40f85c`.** Non-finishers stay in, as losers that never place:
place labels through `bet_settlement.placed_mask` (`fp <= places` alone
counts position 0 as placed - the trap a naive fix walks into), the
normalised-position target puts them last, `rps_per_race` ranks them after
every finisher. `tests/test_non_finishers.py`; three of its four tests fail
on the old code.

Training-app evaluation, same settings and store (fe `221fcd4d`):

| strategy | run `20261014_100406` (fallers dropped) | run `20261014_105017` (fallers kept) |
|---|---|---|
| top pick | -8.0% (14,329 bets) | -13.3% (14,348) |
| value | **+2.1%** (5,360) | **-15.2%** (4,088) |
| each-way | **+79.3%** (483) | **-8.8%** (123) |

Both models on one complete test set (14,331 races, 133,241 runners, 8,129
of them non-finishers), and the old model again on finishers only:

| model, test set | race log loss | top-1 | top pick | value | each-way |
|---|---|---|---|---|---|
| old, all runners | 1.8920 | 0.2940 | -13.3% | -17.8% (4,025) | -5.0% (87) |
| new, all runners | 1.8896 | 0.2941 | -13.4% | -15.3% (4,085) | -18.7% (65) |
| **old, finishers only (the leak)** | **1.8333** | **0.3048** | **-8.1%** | **+2.1%** (5,356) | **+96.1%** (416) |

The inflation was almost entirely in the *evaluation*: dropping the fallers
from the scored set alone turns -17.8% into +2.1% and -5% into +96%. Training
on finishers only cost little (new vs old: log loss -0.0024, top-1 equal). The
new run is promoted because it is trained on what it is served.

**What it does and does not change.**
- **Every research conclusion in this ledger stands.** The walk-forward
  research scripts (`wf_market_offset`, `wf_ranking_compare`,
  `wf_benter_combination` behind ledger 2, triage, microstructure, family
  lift) read the feature store directly and always included every runner.
- **Every ROI the training app or its backtest tab has ever shown was
  inflated**, most for jumps and each-way. The "place-market edge" (ledger 1)
  was overturned for a different reason, but measured through the same
  filter.
- **The production model was trained on finishers only.**
- **The tripwire is now four for four.**

## 19. Morning money on debutants and favourites (pre-registered 2026-10-14)

*Written and committed before the test was run.*

**Hypothesis.** Following the money at 11:00 reached break-even overall
(ledger 14: median CLV 1.0000, ROI +0.85% [-1.12, +2.85]). If that value comes
from private information the morning market has not absorbed, it should
concentrate where the market's informational lead over form is largest
(ledgers 8 and 2d): **debutants** and **favourites**.

**Data.** Development period only; the m11 protocol of ledger 14: races off at
or after 11:30 UK, every runner priced at 11:00 (`ltp_1100`, BASIC) and at
BSP, morning volume from the public BSP files. *Debutant* = empty pre-race
form string. The results start on 2020-09-15, so a count of earlier runs in
the database would mislabel older horses as debutants, while the racecard
form string grows run by run and includes runs before the database started.
*Favourite* = the shortest 11:00 price in the race (ties included).

**Rule** (unchanged from ledger 14): in each race, back the runner with the
highest excess morning volume if its 11:00 price is between 1.5 and 50.

**Primary tests (two):**
- **H1:** the rule's selections that are debutants; matched baseline: every
  debutant in the band.
- **H2:** the rule's selections that are the 11:00 favourite; matched
  baseline: every 11:00 favourite in the band.

**Pass, for each:** median CLV (11:00 price / BSP) above 1.0 with a
race-clustered **97.5%** interval excluding 1.0 (Bonferroni over the two),
geometric-mean CLV above 1.0, and ROI at the taken price after 2% commission
above zero. Reported but not tested: the complements (selections that are not
debutants, not favourites) and a within-segment rule (the debutant, or the
favourite, with the highest excess volume in its race).

**A pass is a candidate, not a result**: confirming it needs Betfair BASIC
files from 2026-06-11 (the hold-out has volume but no 11:00 price), scored once.

### Result (`scripts/morning_money_segments.py`): both fail

> **Corrected 2026-10-14 (ledger 20):** with later non-runner reductions
> applied to the 11:00 price, H2's median CLV is **0.9963 [0.9934, 0.9997]**
> (ROI -1.70%), so the "near miss" is below the close. Table kept as recorded.

60,733 races, 537,646 runners (2020-09..2026-06), 2% commission:

| cell | bets | win rate | median CLV [CI] | geo CLV | beat close | ROI at 11:00 [CI] | ROI at BSP | quarters + |
|---|---|---|---|---|---|---|---|---|
| **H1** pick & debutant | 2,535 | 0.238 | 1.0008 [0.9895, 1.0205] (97.5%) | 0.9780 | 50.0% | +4.79% [-7.08, +17.81] | +2.03% | 11/24 |
| baseline: debutants in band | 25,020 | 0.107 | 0.9429 [0.9366, 0.9506] | 0.9034 | 44.1% | +2.71% [-2.75, +8.33] | -0.33% | 13/24 |
| **H2** pick & favourite | 25,491 | 0.363 | **1.0122 [1.0086, 1.0151]** (97.5%) | 0.9931 | 52.1% | +0.99% [-0.97, +3.05] | -1.11% | 16/24 |
| baseline: favourites in band | 60,416 | 0.321 | 0.9958 [0.9932, 0.9992] | 0.9755 | 48.6% | -0.57% [-2.02, +0.84] | -0.87% | 10/24 |
| all picks (ledger 14 m11, reproduced) | 60,733 | 0.236 | 1.0000 [1.0000, 1.0011] | 0.9711 | 49.7% | +0.85% [-1.12, +2.83] | -2.72% | 13/24 |
| pick & not debutant | 58,198 | 0.235 | 1.0000 [1.0000, 1.0008] | 0.9708 | 49.7% | +0.67% [-1.41, +2.66] | -2.92% | 13/24 |
| pick & not favourite | 35,242 | 0.143 | 0.9887 [0.9828, 0.9948] | 0.9555 | 47.9% | +0.75% [-2.25, +3.87] | -3.88% | 12/24 |
| within-debutant top excess | 9,505 | 0.141 | 0.9577 [0.9494, 0.9682] | 0.9217 | 45.8% | +1.05% [-6.51, +8.85] | -0.44% | 10/24 |

- **H1 fails.** The money is no more informative on debutants: the median CLV
  interval spans 1.0, the geometric mean is below 1, and 2,535 bets cannot
  resolve the ROI (interval 25 points wide).
- **H2 fails, as a near miss in the predicted direction.** Where the money
  backs the 11:00 favourite, the median bet beats the close (1.0122, interval
  clear of 1.0) and it carries most of the rule's timing value (non-favourite
  picks: 0.9887). But the geometric mean is 0.9931 - favourites that drift
  lose more than shorteners gain - and the ROI, +0.99%, has an interval
  spanning zero. Same shape as the T-5 timing effect in ledger 14: real, and
  too small to bank after commission.
- **No hold-out test is warranted**; neither passed on development data.
- The debutant *baseline* returns +2.71% at 11:00 against -0.33% at BSP with a
  median CLV of 0.943 - debutants drift after 11:00 on average, yet the
  winners among them do not. Descriptive, untested, and the interval spans
  zero; noted only as a question.

## 20. Execution timing: when to strike, not what to back (EXPLORATION_PLAN B; pre-registered 2026-10-14)

*Written and committed before the test was run.*

**Hypothesis.** Morning money predicts the later price move (ledger 14, T1/T5)
although it predicts nothing beyond BSP. So for a bet already worth making,
striking early when the price is predicted to shorten, and waiting for BSP
otherwise, should cost less than always taking BSP. This needs no selection
edge.

**Two corrections that ledgers 14 and 19 did not make.** Both apply to every
early strike scored here.
- **Non-runner reductions.** Betfair cuts the odds of a back bet by the
  reduction factor of every runner withdrawn after it was matched; BSP, struck
  after all withdrawals, is not cut. An early LTP compared with BSP without
  this is flattered whenever a horse comes out later. Every factor is applied,
  including any below the 2.5% Betfair may waive (unverified): the
  conservative reading. An LTP that predates a withdrawal before the cut is
  scaled by that factor too.
- **The spread.** An LTP is a traded price, not a back price. The primary
  scoring takes one tick down Betfair's ladder.

Strike price at decision time *t*:
`P_t = max(1.01, tick_down(LTP_t, 1) x rf_stale_t x rf_after_t)`
(`bet_settlement.betfair_tick_down`, `apply_reduction_factor`). Target:
`y_t = log(P_t / BSP)`.

**Data.** Development period only. Ledger 14's frame
(`microstructure_screen.load()`: complete markets, public BSP as the close,
morning volume) joined to `data/processed/betfair_timing.parquet`, a
re-extraction of the BASIC tars that adds, per cut, the LTP's age, the count
of LTP changes so far, the reduction factors of runners withdrawn later, and
the withdrawn runners' share of the book at the cut. Races off at or after
12:05 UK; every runner traded at 11:00, T-60 and T-5.

**Decision points.** 11:00 UK, T-60, T-5.

**Features, all known at *t*.** log LTP_t and its square; 1/LTP_t; the
overround at *t* including runners withdrawn later (without them, a low early
overround would announce a withdrawal); favourite at *t* and price rank;
field size at *t*; jumps, handicap and all-weather flags; local off hour;
log minutes to the off; log age and log count of LTP changes at *t*;
log `rf_stale_t`; excess morning volume, log runner and race morning volume,
`log(morningwap / LTP_t)`; the move from 18:00 the evening before and from
08:00 to *t* (zero plus a missing flag where untraded); at T-60 and T-5 the
move since 11:00, at T-5 the move since T-60. Never the result, BSP,
`rf_after` or anything after *t*.

**Model.** Least squares on standardised features, one per decision point,
on purged quarterly expanding folds (`wf_eval.expanding_period_folds`, at
least four training quarters). LightGBM is fitted alongside and replaces the
linear model at a decision point only if its pooled out-of-fold R-squared of
`y_t` is at least 20% higher (relative), judged before any policy is scored.

**Policy, fixed:** at 11:00, strike if the predicted `y` exceeds
**tau = 0.01**; otherwise at T-60 on the same test; otherwise at T-5;
otherwise take BSP.

**Universes, fixed at 11:00:** U1 every runner whose 11:00 LTP is 1.5-50; U2
the 11:00 favourite (shortest in the whole 11:00 book, ties kept) in the band;
U3 ledger 14's pick (highest excess morning volume among the band).

**Primary metric, per universe:** `G` = mean over the universe of `y` at the
strike time if struck, 0 if BSP was taken: the cost saved per bet, in CLV
points. Race-clustered bootstrap; Bonferroni over the three universes, so
98.33% intervals.

**Pass, per universe:** `G >= 0.5` CLV points and the 98.33% interval's lower
bound above 0, with the one-tick haircut and reductions applied. If no
universe passes, Idea B stops (the plan's rule).

**Reported, not tested:** `G` with no haircut, a 1% haircut and two ticks
under the same policy; tau of 0 and 0.02; `G` without the reductions (how
much they flattered ledgers 14 and 19), and ledger 14's m11 rule and ledger
19's H2 re-scored with them; always striking at each *t*; the share struck at
each point; the realised PnL gain, mean of `won x (P_taken - BSP) x 0.98`
with its interval, which must not be significantly negative where `G` is
positive (the tripwire); the struck bets as a standalone strategy against
ACTION_PLAN 7 (median CLV and interval, geometric mean, beat-close, ROI at
the taken price); quarters positive; morning matched volume on struck
runners (capacity).

**A pass is a candidate, not a result.** The hold-out needs Betfair BASIC
files from 2026-06-11, which only the operator can download, and BASIC cannot
show the real spread (no ladder), which only a recorded order book can.

### Result (`scripts/execution_timing.py`): fails in every universe; and the reductions overturn ledgers 14 and 19

60,337 races, 533,587 runners (2020-09-15..2026-06-03); 20 purged quarterly
folds (2021Q3-2026Q2), 463,717 runners scored. LightGBM cleared the 20% R²
rule at every decision point (out-of-fold R² of `y`: 11:00 0.105 vs linear
0.082; T-60 0.039 vs 0.027; T-5 0.046 vs 0.019), so it drives the policy.
`G` in CLV points (1 = 1% better price than BSP), 98.33% intervals for the
primary, 95% otherwise:

| universe | n | struck early | **G, one tick + reductions** | no haircut | two ticks | no reductions | quarters + | PASS |
|---|---|---|---|---|---|---|---|---|
| U1 all in band | 401,782 | 2.9% | **+0.096 [+0.036, +0.162]** | +0.207 | -0.014 | +0.165 | 15/20 | **no** |
| U2 11:00 favourite | 51,538 | 2.1% | **+0.145 [+0.059, +0.242]** | +0.174 | +0.117 | +0.176 | 14/20 | **no** |
| U3 morning-money pick | 52,683 | 5.5% | **+0.158 [+0.046, +0.268]** | +0.286 | +0.031 | +0.321 | 11/20 | **no** |

- **Idea B fails.** The policy saves a real but tiny amount: a tenth to a
  sixth of a CLV point per bet, against a bar of 0.5. It strikes early on
  only 2-6% of runners; always striking early costs 3-14 points
  (always at 11:00: -13.8 in U1). tau of 0 or 0.02 changes nothing. The
  realised PnL gain agrees in sign (U1 +0.15 [-0.47, +0.47] per 100 staked;
  U3 +0.68 [+0.44, +0.94]). By the plan's rule, B stops.
- Waiving reductions under 2.5% barely moves anything (U1 +0.097).

**The correction that matters: ledgers 14 and 19 were flattered by
non-runners.** 17.4% of the m11 runners had a horse withdrawn after 11:00
(12.7% with a factor of 2.5% or more). A bet matched at 11:00 would have had
its odds cut; BSP is not. The 11:00 prices are inflated by almost exactly
that factor (`log(11:00 / BSP)` averages +0.41 where later reductions exceed
30%, -0.13 after applying them, against -0.12 in races with none), so the
correction is the market's own arithmetic, not a modelling choice. Re-scored:

| rule | as recorded | with reductions | reductions >= 2.5% only | reductions + one tick |
|---|---|---|---|---|
| ledger 14 m11 (all picks), median CLV | 1.0000 [1.0000, 1.0011] | **0.9849 [0.9822, 0.9877]** | 0.9856 [0.9830, 0.9886] | 0.9675 |
| ledger 14 m11, geometric CLV | 0.9711 | 0.9505 | 0.9510 | 0.9306 |
| ledger 14 m11, ROI at 11:00 | +0.85% [-1.12, +2.83] | **-1.84% [-3.75, +0.12]** | -1.79% | -3.89% [-5.75, -1.98] |
| ledger 19 H2 (pick & favourite), median CLV | 1.0122 [1.0088, 1.0148] | **0.9963 [0.9934, 0.9997]** | 0.9968 [0.9940, 1.0000] | 0.9848 |
| ledger 19 H2, ROI at 11:00 | +0.99% [-0.74, +2.80] | **-1.70% [-3.38, +0.07]** | -1.65% | -2.89% [-4.54, -1.14] |

"The closest thing to a positive result this project has produced"
(ledger 14) was not break-even: the median 11:00 strike on the morning money
**loses** to the close by 1.5 CLV points once the withdrawals are counted,
and ledger 19's favourite near-miss is below 1.0. **Any early-price test in
this ledger before this entry ignored reductions and is optimistic by about
1-1.5 CLV points at 11:00** (less at T-60 and T-5: 7.6% and 3.6% of runners
have a later withdrawal against 21.7% at 11:00). That includes ledger 14's
T-5 and T-60 rows; their verdicts (fail) only get stronger.

**A descriptive lead, not a result.** Read as bets of their own, the policy's
early strikes beat the close: U1 11,537 bets, median CLV 1.0163 [1.0061,
1.0253], geometric 1.0341, beat-close 51.4%, ROI at the taken price +6.30%
[-3.01, +16.54]; U3 2,885 bets, median 1.0372 [1.0256, 1.0488]. This was
not the pre-registered test. It is development data, the ROI interval spans
zero, two ticks of spread erase the U1 gain, and the strikes sit in thin
markets (median morning volume GBP 487 in U1). The exploratory checks below
ask whether the LTP was ever a price you could back.

### Exploratory checks on the lead (`scripts/execution_timing_diagnostics.py`; not tests)

**It is all 11:00.** U1 strikes by decision point, one tick + reductions:

| strike at | n | median CLV | geometric | ROI at taken price | two ticks: median, ROI |
|---|---|---|---|---|---|
| 11:00 | 5,895 | **1.0488 [1.0385, 1.0575]** | 1.0290 | **+14.7% [+3.9, +25.8]** | 1.0204, +11.1% [+0.7, +21.8] |
| T-60 | 2,429 | 0.9936 [0.9743, 1.0029] | 1.0450 | +25.5% [-5.6, +59.9] | 0.9545 |
| T-5 | 3,213 | 0.9909 [0.9788, 1.0000] | 1.0353 | -23.7% [-40.6, -6.0] | 0.9500 |

T-60 and T-5 strikes have no edge after a tick (their geometric means are
long-price tails). The 11:00 numbers are the kind of result this ledger has
learned to distrust, so three artefacts were checked:

- **Stale prices: no.** The struck 11:00 LTPs are *fresher* than average
  (median age 156 s vs 285 s for all U1 runners; 42 changes so far vs 34),
  and the freshest (under 60 s old, 1,575 bets) still show median CLV 1.033
  [1.020, 1.050].
- **The undocumented morning period, and the final-field normalisation: no.**
  Betfair does not say when its "morning" ends (morning VWAP sits nearer the
  11:00 price than the T-60 price in 81% of big moves, consistent with an
  11:00 cut but not proof), and the race-level morning aggregates are
  normalised over the runners that eventually ran, which could betray a
  later withdrawal. Refitting the 11:00 model **without any morning field**:
  linear has no edge (12,082 strikes, median CLV 0.980, ROI -5.4%); LightGBM
  (R² 0.082 vs 0.105 with them) strikes 2,292 times with median CLV **1.0435
  [1.0200, 1.0679]**, ROI +16.8% [-3.9, +37.3], two ticks 1.0049 [0.9899,
  1.0297]. The edge holds in races with no later withdrawal (median 1.041),
  and the strike rate is barely lower where one comes (0.49% vs 0.59%).
- **Bid-ask bounce: partly, not wholly.** An LTP after a *rise* may be a
  layer matching at the top of the spread, a tick or more above the best
  back price; after a *fall*, a backer has just taken the price, so it was
  there to back. The model does lean on rises (64% of its 11:00 strikes
  follow one, against 48% of all runners), but the strikes that follow a
  fall still beat the close: 2,103 bets, one tick + reductions, median CLV
  **1.0336 [1.0190, 1.0478]**, geometric 1.0227, beat-close 53.4%, ROI +9.1%
  [-6.8, +27.4] (after a rise: 3,781 bets, 1.0563, ROI +18.2%).

**Reading.** After four checks, the 11:00 lead is the most robust timing
signal this project has measured: a model of the 11:00 book's shape picks
runners whose price is long relative to where the market closes, about 3-5%
at the median, with no morning-volume input needed. It looks like thin
morning prices reverting. It is still only descriptive: found after the
pre-registered test failed, on development data, with ROI intervals that
span zero, and BASIC cannot show **how much** could be backed at those
prices (the median struck runner traded GBP 487 all morning).

**If it is pursued, test exactly this frozen rule, once, on data never
seen:** the 11:00 LightGBM of `execution_timing.py` with the four morning
fields removed (`execution_timing_diagnostics.no_morning`: immune to the
undocumented morning cut-off), refit deterministically on all development
U1 rows; strike if the prediction exceeds 0.01, U1 band, races off at or
after 12:05 UK, one tick down plus reductions; pass = median CLV above 1
with the 95% interval excluding 1, geometric mean above 1, ROI at the taken
price above 0. **Power:** about 115 strikes a quarter, so the June-October
2026 hold-out (~150) cannot resolve a 4% median effect (its interval would
be roughly +-0.09); it needs about 600 strikes, roughly five quarters of
BASIC data, or an order-book recording that also measures the depth at the
price (the same Stream API recording Idea E phase 2 needs). for debutants (EXPLORATION_PLAN D): stopped at step 0 (2026-10-14)

The test needs measured breeze times for the two-year-olds sold at European
breeze-up sales, matched to their debuts. **No European breeze-up published
official times in 2020-2026.** Trade coverage is consistent: British sales
have no official timing (Tattersalls hands buyers stopwatches; private timing
gates and lasers run by buyers' agents), vendors oppose official times
because a slow breeze can mean no sale, and the time sheets that "circulate"
after the major sales are private. Goffs said it was open to publishing times
at its new Naas sale (June 2025) and Aktem at its second edition; neither is
confirmed, and either would give one sale a year, a few dozen debut runs
against the ~1,500 needed to see a partial of 0.03. The only systematic
clocker times are commercial (Racing Post), which this project does not
scrape.

**Not pursued.** The prior was already low (sale price is fully priced, even
on debutants, ledger 5). Revisit only if a sales company starts publishing
official times, and then only after several sales' worth have accumulated.

## 22. Morning market-making, phase 1: feasible on paper, unknowable on free data (EXPLORATION_PLAN E) (2026-10-14)

`scripts/morning_mm_feasibility.py`, development period, races off at or
after 11:30 UK (so "morning" volume is all before 11:00): 64,168 races,
573,583 runners, GBP 953m of morning matched volume. Descriptive, no test.

**The arithmetic.** Matched volume is zero-sum at the traded price. Per GBP
of backer stake matched at price `P` on a runner whose fair probability is
`p` (BSP, overround removed), the layer's expected value is `1 - P*p` and the
backer's `P*p - 1`, before commission. So the *markout* of a fill follows
from the price, the close and the side. The public files do not say who
crossed the spread, so two cases bound it: against the **informed side**
(the runner with the most excess morning volume, ledger 14) a market-maker
is assumed to have laid; elsewhere it fills both sides equally. Later
non-runner reductions are applied to the morning price (ledger 20).

Volume-weighted markout per GBP matched, %, 2% commission per winning bet:

| fills at | group | lay | back | both sides | break-even half-spread |
|---|---|---|---|---|---|
| morning VWAP | all runners | -0.15 | -2.81 | -1.48 | 0.7 ticks |
| morning VWAP | informed (top excess volume) | **-2.45** | -0.10 | -1.28 | **1.7 ticks** (as layer) |
| morning VWAP | the rest | +1.24 | -4.43 | -1.60 | 0.6 ticks |
| 11:00 LTP | informed | **-1.69** | -0.85 | -1.27 | **1.2 ticks** (as layer) |
| 11:00 LTP | the rest | -0.66 | -2.57 | -1.62 | 0.65 ticks |

By price band the informed-side lay loss is worst at 5.0-13.0 (-6.4% at
VWAP, about 2-2.7 ticks) and smallest on short favourites (under 2.0: +0.7%).
The "both sides" column charges commission on every winning bet; a two-sided
maker pays it on net market winnings, so the true cost is lower.

**Capacity.** Median morning volume per runner GBP 454 (mean 1,661); per
racing day GBP 303k median across all GB/IRE runners, of which GBP 113k is
on the informed runner. A new maker would win some fraction of that.

**Verdict: phase 1 does not stop E.** The break-even half-spread, about 0.7
ticks against balanced flow and 1.2-1.7 ticks against informed flow, sits
inside the plausible 1-2 ticks, and capacity is not trivial. But phase 1
cannot answer the questions that decide it: how often a passive offer is
filled at all, queue position behind the incumbent makers, and the real
aggressor mix. The observed morning prices are already the outcome of
professional market-making, and the uninformed runners' lay markout at VWAP
(+1.24%) is the margin those incumbents earn. Phase 2 (live app key, Stream
API order-book recording for 4-8 weeks, a paper simulation with queue
position) is an operator decision; no order is ever placed.

## 23. Forecast ground changes, bet early (EXPLORATION_PLAN C; pre-registered 2026-10-14)

*Written and committed before the test was run.*

**Hypothesis.** When rain is forecast between an early decision time and the
off, early prices under-react for horses that like soft ground; the market
catches up when the official going changes. The test is **continuation**:
does the forecast-times-preference signal, known at the decision time,
predict the price move *after* it?

**Data.** Development period from 2024-03-01 (Open-Meteo's day-ahead forecasts
start in early 2024) to the hold-out's purge cutoff; turf races only; ledger
14's frame (complete markets, public BSP as the close).
- *Forecast rain*: the sum of `precipitation_previous_day1` at the course over
  each hour from the decision to the off (each hour's value as forecast 24 h
  before it, so every forecast used was issued before the decision;
  `scripts/fetch_forecast_previous_runs.py`). Never the `weather_*` columns,
  which record what happened.
- *Ground preference*, per runner, from its previous runs only: finish
  percentile minus SP-rank percentile, averaged on soft ground (Good to Soft
  or softer) minus the same on firm (Good to Firm or firmer), each shrunk
  toward zero (k = 2). Not the store's `horse_going_*` columns, which are
  keyed on the race's final going and so are look-ahead here.
  (`scripts/ground_forecast.ground_preference`, `tests/test_ground_forecast.py`.)
- *Decision price*: the last trade at or before 08:00 UK on race day, every
  runner traded; later non-runner reductions applied (ledger 20).

Feature distributions were looked at before writing this (never outcomes):
forecast rain from 08:00 to the off has median 0 mm, 90th percentile 2.0 mm;
preference has SD 0.114, 90th percentile 0.139.

**Primary test (one).** `x = preference x forecast rain`; `y = log(P_08:00 /
BSP)` with reductions, no haircut. Within-race partial correlation of `x` and
`y`, controlling for the 08:00 price basis (as `triage_features`) **and the
preference itself** (so a general drift of soft-ground horses is not read as
a forecast effect); race-clustered bootstrap SE, 2,000 draws. **Pass: partial
at least +0.01 and two-sided p below 0.05.**

**Economic rule (scored whatever the primary shows; counts only if the
primary passes).** In races with at least 2 mm forecast from 08:00 to the
off, back the runner with the highest preference if it is at least 0.15 and
its 08:00 price is 1.5-50; strike one tick down with reductions. Matched
baseline: every runner in the band in those races. Pass (ACTION_PLAN 7):
median CLV above 1 with the 95% interval excluding 1, geometric mean above 1,
ROI at the taken price above 0. Reported in ledger 19's format.

**Reported, not tested:** the 18:00-evening-before decision (fewer complete
markets); the sire's soft-ground edge in place of the horse's preference;
runners with both soft and firm history only.

**Hold-out:** needs Betfair BASIC files from 2026-06-11 (operator).

### Result (`scripts/ground_forecast.py`): fails

14,869 turf races, 124,892 runners (2024-03-01..2026-06-03), every runner
traded at 08:00, forecasts for all 86 courses.

| test | partial | SE | t | p | n (races) | |
|---|---|---|---|---|---|---|
| **primary: preference x rain, 08:00** | **+0.0060** | 0.0029 | +2.04 | 0.041 | 124,892 (14,869) | **fails (below 0.01)** |
| runners with soft and firm history | +0.0048 | 0.0054 | +0.88 | 0.38 | 42,602 (10,070) | |
| sire's soft-ground edge instead | +0.0065 | 0.0034 | +1.92 | 0.054 | 119,499 (14,869) | |
| evening before (18:00) | +0.0014 | 0.0089 | +0.15 | 0.88 | 16,027 (1,820) | |

| economic rule (08:00, >= 2 mm forecast) | n | median CLV [95% CI] | geo | beat close | ROI at taken price | ROI at BSP | quarters + |
|---|---|---|---|---|---|---|---|
| baseline: band in those races | 10,188 | 0.8785 [0.8706, 0.8856] | 0.8293 | 36.8% | -9.19% | +3.12% | 1/10 |
| rule: top preference >= 0.15 | 661 | 0.9028 [0.8568, 0.9444] | 0.8526 | 39.5% | -0.85% [-25.5, +27.1] | +10.20% | 6/10 |

- **C fails.** The signal points the predicted way (soft-ground horses
  shorten after 08:00 when rain is forecast) and is nominally significant,
  but at 0.006 it is below the pre-registered 0.01, and the pedigree version
  is no stronger. The economic rule beats its baseline on CLV only because
  08:00 prices are far shorter than BSP across the board (the 08:00 book is
  thin and over-round: the median runner drifts 12%); it is nowhere near 1.
- The evening-before decision cannot be tested properly: only 1,820 races
  had every runner traded by 18:00 the day before.
- By the plan's stop rule (no continuation signal of 0.01), C stops. The
  forecast cache (`data/weather_forecast/`, 86 courses, 2024-01..2026-10) is
  kept.

## 24. Tote exotics priced from BSP (EXPLORATION_PLAN A): step 0 blocked on the operator (2026-10-14)

Not run. The data exists but the only complete source forbids collecting it.

- **Sporting Life result pages carry every Tote dividend** in their
  `__NEXT_DATA__` JSON, back to at least 2020-09: `tote_win`, `place_win`
  (per place), `exacta_win`, `trifecta`, `swingers` (three pairs),
  `straight_forecast` (the CSF), `tricast`, `place_pot` and `quad_pot`
  (dividend, pool size and winning stakes), plus `deductions` and
  `on_course_book_percentage`. Checked on 2020-09-19, 2022-03-05 and
  2025-10-04. Dividends are quoted per GBP 1. **No exacta or trifecta pool
  sizes**, which the plan's stake-cap trap needs.
- **Sporting Life's terms: "Data capture including, but not limited to,
  screen scraping is expressly prohibited"**, content is "for personal use",
  and no commercial use. By the plan's rule (as for Racing Post), no
  80,000-page backfill was started. The main pipeline already scrapes these
  pages daily; reconciling that with the terms is the operator's call.
- **At The Races** publishes one results page per day with dividends
  (~2,200 pages for the whole period; robots.txt allows `/results/`,
  disallows `/ajax/`). Its terms page and a results page both returned HTTP
  429 to the project's fetcher on two tries 25 minutes apart, so its terms are
  unread. Note for the sectional scraper: it may be rate-limited too.
- **Takeouts are steep.** Placepot measured from one page: pot x winning
  stakes / pool = 72.9% returned (~27% deduction). Published figures (old,
  unofficial): win 13.5-19.25%, exacta ~24%, trifecta and jackpot 29%,
  placepot 27%; Irish pools a further 5%. An exotics rule must beat a
  quarter of every stake.

**Unblocking needs one operator decision:** permission to collect Sporting
Life's dividend fields (the scrape is ~80k pages, ~1 request/s, about a day,
resumable), or a licensed source (the Tote's own data, or a paid racing data
API that carries dividends and pool sizes).

## 25. Full ATR family in the independent probability model (2026-10-26)

Experiment 002, registered before execution: existing 149 inputs versus 355
columns (206 added historical sectional features), identical complete fields,
native LightGBM fitting/calibration and fixed 2025Q3, 2025Q4 and 2026Q1 folds.
Ten future-dependent composites/descendants were removed before fitting;
the remaining additions passed temporal invariance checks. This asks about
independent probability quality, separately from earlier market-residual tests.

All six fits completed. On 9,356 races / 85,863 runners, race log loss worsened
**1.872045 -> 1.876713**, candidate minus baseline **+0.004668**, paired 95%
interval **[+0.000852, +0.009229]**; date-cluster interval
**[+0.000980, +0.009326]**. All three quarters worsened. Runner Brier worsened
0.08754346 -> 0.08755972 and binary log loss 0.30088334 -> 0.30140404.
Small winner-accuracy/ECE gains do not rescue a probability forecast that
worsens the registered proper scores.

**Do not repeat this unchanged family addition or select subsets against its
test quarters.** It fails every advancement condition. This is not a rejection
of all possible sectional methods, nor an independent market-parity claim.
Any reopening needs a materially changed hypothesis and predeclared evidence.
The one bounded race-objective comparison was subsequently completed under
the newer probability-scoring plan; see section 26.
See [experiment report](research/MARKET_COMPARISON_002.md) and the frozen models,
source/input manifests and paired forecasts in
`data/audit/market_performance/002_atr_family/`.

## 26. Race-objective comparison: linear loses; tree primary gain unresolved (2026-10-26)

Experiment 003 recovered the prior June scratch script: it dropped actual
non-finishers and did not establish the current all-starter probability
comparison. One bounded repeat used the retained 149 inputs, exact cached
binary forecasts and identical 2025Q3/Q4/2026Q1 fields. All six challenger
fits finished before scoring; two fixed candidates, no post-test search.

On 9,356 races / 85,863 runners:

| Recipe | Runner Brier | Binary LL | Race LL |
|---|---:|---:|---:|
| Binary baseline | .08754346 | .30088334 | 1.87204489 |
| Conditional linear logit | .08806124 | .30328719 | 1.89083039 |
| Race-softmax LightGBM | .08736141 | .30062028 | 1.87103531 |

The fixed linear recipe loses all three quarters and every proper score.
Race-LL delta +.01878550, 95% interval [+.01088235,+.02613680]; reject it.

**The tree is not a blanket negative.** Its Brier improves in every quarter,
pooled runner delta -.00018205 (one-sided race-cluster upper95 -.00008701).
Race LL improves in two quarters, but pooled delta -.00100958 has interval
[-.00510065,+.00327606]; date sensitivity [-.00503771,+.00326771]. It fails
the registered primary confidence gate. Neither candidate advances. Do not
change the primary criterion to Brier after seeing this secondary gain.

On 8,942 matched T-5 fields the tree's LL 1.86768444 remains well behind
market 1.74928256. Its point gain closes only ~0.93% of the baseline LL gap;
freshness sensitivity gives the same conclusion. No independent parity.

**Stop the bounded architecture expansion.** Preserve the fixed tree as an
unpromoted research challenger; its primary benefit remains unresolved and
would require registered confirmation on other data. Do not tune a favourable
quarter, Hessian, regularisation, calibration, feature subset, seed or blend
against these outcomes. This result does not reject every possible race model.
See [full report](research/MARKET_COMPARISON_003.md) and immutable evidence under
`data/audit/market_performance/003_race_objective/`. Final full suite: 742 passed;
nine actual model artifacts replay saved probabilities to 1.11e-16. Production
and canonical feature/data snapshots are unchanged.

## 27. Monthly refitting: small Brier gain, primary benefit unresolved (2026-10-26)

Experiment 004 tested the accepted plan's fixed monthly refresh against exact
cached quarterly binary forecasts. Same 149 features, recipe, daily feature
history, all-starter fields and seven-day calendar cutoff. Six new fits,
three exactly shared first-month models; no cadence or hyperparameter search.

On 9,356 races / 85,863 runners, race LL improves 1.87204489 -> 1.87097477.
All three quarter point estimates improve, but difference -.00107012 has
95% race interval [-.00436065,+.00347671] and date-cluster interval
[-.00426195,+.00305721]. **The primary advancement gate is unmet.**

This is not a blanket negative: runner Brier improves .08754346 -> .08746770
in every quarter; pooled one-sided upper95 difference is -.00001120. Binary
LL also improves. Mean parameter age falls from 52.46 to 22.21 days. Preserve
these secondary findings without relabelling them primary confirmation.
Overall ECE worsens .00146349 -> .00197548. The six changed-model months alone
also have a primary LL interval crossing zero and do not change the decision.

On 8,942 matched T-5 fields monthly LL 1.86755050 remains behind market
1.74928256. Its point gain closes 1.04% of the quarterly model's market gap;
the 7,719-race freshness subset also fails parity. No current-race prices were
added, no market correction fitted and no canonical outcome changed.

Close this fixed cadence comparison. No weekly/daily/window/calibration sweep
on these dates. Retain the monthly recipe as unpromoted research; primary
confirmation requires other data and a registered test. This does not close
operational all-settled fitting or data-refresh work. Full suite 745 passed;
32 status checks passed after an explicit parameter-cutoff reporting change.
Nine actual artifacts replay 576 shuffled fields to 1.43e-14. Production and
the canonical data/store remain unchanged. See [full report](research/MARKET_COMPARISON_004.md).

## 28. Market-history proposals034–035 (2026-10-29)

The corrected154-input baseline includes the adopted033 repairs and prior-price
history. A new400-tree baseline was fitted once and reused for both screens.
Transfer/recency034 adds5 inputs and passes its small point-score screen; one
native159-input win check follows, with saved154-input native baseline and no
place/current-data fit. Native overall LL1.798224→1.795672 and Brier improve,
but established-handicap Brier worsens .09756486→.09760743. Overall date-cluster
LL interval[-.005913,+.000824] crosses zero. It fails the fixed advancement rule.
Preserve the modest overall gain as uncertain; do not claim all transfer
features are useless or force another installed fit.

Market-relative finishing context035 adds3 inputs. Its single small fit worsens
target LL1.981946→1.982161 and Brier .09162586→.09165907. Stop there. The joint
rule required both standalone families to pass, so no joint proposal follows.
Eight checks per helper and real full/prefix/seed equality pass; these failures
are measured predictive results rather than causal-history implementation errors.
See [034–035 report](research/MARKET_HISTORY_REVIEW_034_035.md).

## 29. Simple market-assisted refresh036 (2026-10-29)

Reuse the same frozen December154-input model across earlier coefficient fit,
purged validation and May–10June development test. No new tree model. Whole
fresh Betfair T-5 last-traded books are required; no executable-quote claim.
Fit a market temperature and bounded two-coefficient model/market combination.
Validation chooses the raw market before the test. On1,351 later fields,
market LL1.778875 versus combination1.780645; on487 established handicaps,
1.789571 versus1.792246. Both date-cluster difference intervals favour market.
The temperature's tiny overall test gain is uncertain and did not earn prior
selection. Do not switch to it after seeing the test. No correction is deployed.
This closes this fixed simple correction, not all market-assisted approaches.
See [036 report](research/MARKET_ASSISTED_REFRESH_036.md).

## Things that are true and worth not re-deriving

- **RTV RaceiQ coverage starts on 2023-06-10 and is near-complete after it**
  (corrected 2026-10-29; this line used to say coverage was low at selected
  tracks). The old cache holds 98.8% of GB races and ~80% of Irish races
  (86–100% from 2024). There is no earlier RaceiQ history to backfill.
  `add_rtv_features` remains retired; the scraper was rebuilt as a collector
  only — see [RTV audit](operations/RTV_SCRAPER_AUDIT_2026-10-29.md) and
  [RTV scraper](operations/RTV_SCRAPER.md).
- **At The Races scraping** needs `curl_cffi` or the system `curl` to clear the
  Fastly TLS/JS challenge. ~2.5s per request, GB-only, resumable.
- **Bodyweight / run-to-run weight change** — the single most valuable
  biometric — is **not published in GB/IRE**. Do not hunt for it here.
- **Pedigree** is noise for exposed horses. If revisited, gate it to
  debutants, first-time-soft and new-trip runners only.

## Still open (not refuted, not done)

- **Market microstructure** — traded volume and ladder depth from Betfair
  historic PRO files. The only untapped orthogonal signal identified (§9).
- **Confirmation of the fixed race-softmax recipe** — confirmed by 057
  (2 November 2026) on registered quarters 2025Q3–2026Q2:
  - a race softmax on production's soft target: −0.0022 [−0.0039, −0.0003],
    4 of 4 quarters;
  - the exploded softmax with stage weights (1, 0.25, 0.0625): −0.0046
    [−0.0064, −0.0026], 4 of 4 quarters.

  The exploded softmax was installed on 4 November as
  `20261104_235300_444862`; a store rebuild and refit remain before it can
  serve live cards.
- **Each-way 1/5-odds re-run** and a locked CLV harness — parked.
- **Retrain on the repaired store** — completed 2026-10-25; production
  `20261025_193016_331639` passes provenance and replay checks. See
  [the project audit](archive/PROJECT_AUDIT.md). Data refresh, further model comparisons
  and genuinely fresh forward confirmation remain separate work.


##039–040. Market-adjusted connection form and own cadence: not selected (29October2026)

Both families use the corrected154-input frozen baseline, unchanged native win
training source and exact numeric/cache/forecast replay. One400-tree candidate
per family; no new baseline/native/place/calibration/current-data fit. Fixed
training24September2023–23September2025; previously inspectedQ42025development.

039 adds trainer/jockey60-day observed-minus-market-expected win rates, shrunk
with20neutral starts, and priced support. OverallLL1.845981→1.845247, but target
LL1.981946→1.983893 and Brier .09162586→.09172128.040 adds a horse's gap relative
to its own last-five median, the median and support (three prior intervals,
730days). OverallLL1.845981→1.845848, target1.981946→1.982299, target Brier
.09162586→.09164178. Both overall date95%LL intervals cross zero.

Stop these fixed recipes after the small screens. Their source/cadence coverage
is high for the established target; do not explain away the result by selecting
a favourable subset or change windows/shrinkage after seeing it. The tests do
not reject every possible connection-form/cadence model.27related checks pass;
all678085source rows have strict full/prefix/append equality. No new source
error was established. [Full definitions and results](research/ESTABLISHED_CONTEXT_039_040.md).

##041. Raw place-head information for win: not selected (29October2026)

Reused the frozen December154-input win/place heads, with zero new tree fits.
Two tiny coefficient fits compare a bounded positive raw-place correction
against identity and a win-only temperature control. January–February fits,
purged March validation and May–10June development test use the same saved
heads. The fitted place coefficient is exactly0; validation selects identity.
The remaining sharpening worsens overall validation LL1.840156→1.842319 and
target1.880653→1.881153; both Briers worsen. Later-test overall LL1.878711→1.881416
and target1.936083→1.937554 also worsen. No native transfer, new tree fit or
production change follows.19helper/optimizer checks pass, native replay is exact,
and current-label/price poisoning leaves both heads unchanged. Preserve the
pre042 source-label version of this evidence. [Definition and results](research/AUXILIARY_PLACE_WIN_041.md).

## 045. Repaired skill ratings: much better skill measures, still no market gain (30 October 2026)

The Elo, Glicko, TrueSkill and jockey/trainer ratings were repaired (Elo K too
small for its scale, non-finishers ignored, jockeys rated on their horses,
separate surface/race-type/distance pools, undecayed momentum, Glicko RD
collapse, Glicko-2 φ double-counted, no TrueSkill layoff uncertainty) and
re-tuned with `scripts/screen_ratings.py`. Held-out race log loss on 29,786
development races: Elo 2.125 → 2.010, Elo + jockey + trainer 2.030 → 1.952,
TrueSkill μ 2.050 → 1.996, Glicko 2.012 → 2.004. On top of SP every rating, old
or new, changes log loss by at most 0.00015 nats. The market prices all of it;
the rating store is a better description of form, not an edge.

**Field-relative newcomer starts ratchet.** Starting debutants below the rated
runners they meet, by a fixed gap (50/100/200 points) or a learned one, always
lost to a fixed start, and lost more in the later test period (Elo gap 200:
fit 2.105, test 2.198, against fixed 2.023 / 2.010). Maiden fields are mostly
recent debutants, so each cohort starts relative to the last one's start and
the scale drifts. Do not retry it without anchoring debutants to something
outside the debutant pool. [Method, settings and the full table](research/RATING_SYSTEMS_045.md).

**Plackett–Luce adds nothing over TrueSkill (1 November 2026).** About 50
settings of an online Plackett–Luce rating were swept: Weng & Lin's Bayesian
update, and a gradient update with Elo's K schedule. The best is the gradient
form with Benter's place discount (d 0.85): fit 2.032, test 2.022, against
TrueSkill's 2.010 / 1.996. Added to TrueSkill it gains 0.0003 nats in the fit
period and nothing on test; over SP +0.00006. Undamped, the back of the field
dominates its gradient. Kept as a screen option in `scripts/screen_ratings.py`
only.

## 046. Projected race profile and RTV-derived inputs over the market (31 October 2026)

**Race-profile projection is dead.** Race shape is partly predictable before
the off, mostly from trip, going and course. On 17,874 development races,
finishing-shape correlation is 0.62, tempo about 0.22 and winner-up-front
about 0.16. That predictability does not help pick runners. Every runner
family scored the same as k pure-noise columns over the T-5 market (walk-forward
conditional logit):

- trip on effective duration;
- stamina reach;
- shape fit;
- run style × projected shape;
- the 009 store pace family (−0.00081 against −0.00084 for 23 noise columns).

The 009 family was also built on the mis-built ATR raw-mph inputs. Do not
rebuild projected race profiles.

**Report a size-matched noise baseline.** The first run of this test had its
sign flipped, so a 23-column family's out-of-sample overfitting cost looked
like a gain in 6 of 6 folds. Always report a noise baseline of the same size,
and a positive control (BSP over T-5 scored +0.018).

**RTV-derived families add nothing over the price.** Each family stayed within
noise over the T-5 market: the Timeform card, sectional time figures, and
within-race stride and jumping measures. Beyond a market-free form proxy, four
candidates were dead too: pace-adjusted form, early and closing time gaps,
stride × going, and pace hint × run style.

The first three families do make the form model more accurate (046 in the
progress record), but that is not an edge. Reading the Timeform card as
private information would be a mistake: the market prices it fully.
[Details](research/RTV_DERIVED_046.md).

## 047. Re-encoding the 154 model inputs: real faults, no accuracy gain (1 November 2026)

A code review of the promoted inputs found encoding faults:

- missing base values coded as "best in race" by `_vs_best_in_race`;
- the leader's margin discarded;
- debutants and maidens given 60 / 365 days;
- raw finishing positions that ignore non-finishes;
- race-median OR substitutes inside OR rank and spread;
- thin-data jockey/trainer rates set to 0;
- a raw £ prize difference;
- `draw / num_runners` above 1;
- five exact duplicate columns.

Ten repair sets were screened (quick_win, four development quarters, 11,901
races). None moved race log loss: pooled changes ran from −0.00056 to
+0.00087, every interval spanned zero, and Brier was flat. The same screen
gave the 046 inputs −0.0187. LightGBM already compensates through other
columns. Dropping the duplicates was slightly worse (+0.00130 [−0.00032,
+0.00307]), because column subsampling gives a duplicated signal extra weight.
Do not re-encode existing inputs expecting accuracy; the gains came from new
information. The faults were fixed in code anyway, for correctness, on the
same day. The store was not rebuilt, so no rebuilt-store measurement exists
yet. [Faults, evidence, the full table and the fixes](research/FEATURE_IMPLEMENTATION_REVIEW_047.md).

## Tier 2. Ability and same-day inputs inside the generalist: no better than as corrections (10 November 2026)

Full 063 refitted with the 14 ability-in-pounds and 5 same-day bias inputs,
three seeds, 2025Q3–2026Q2 (11,905 races), against the cached three-seed
average: −0.00050 [−0.00160, +0.00063], 3/4 quarters. The gate (interval below
zero) fails. The same inputs as two summed residual corrections give −0.00065
[−0.00138, +0.00020], and adding those corrections to the refitted arm gives
nothing more (+0.00026). The hope that a refit would unlock a larger gain, as
it did for the 063 rating gaps, is refuted for these groups. Do not move
screen survivors worth about 0.0005 into the generalist: three seeds per arm
cannot resolve them (±0.0011) and the corrections already capture them.
Averaging all six fits scores −0.00072 [−0.00127, −0.00016], 4/4, against the
three-seed reference; about −0.0003 of that is expected from six fits instead
of three. Exploratory, not a registered comparison.
[Report](research/TIER2_INPUTS_2026-11-10.md).

## Error anatomy, 10-11 November 2026: what held, what did not

Positive result first, because it changes what is worth trying: inputs that
measure each horse, trainer and sire against the model's own out-of-fold
forecast (results and past closing prices) were confirmed on one registered
read of 5,761 unseen races, −0.0091, z −4.8
([report](research/ERROR_ANATOMY_2026-11-10.md)). Do not rebuild these from
scratch; extend them.

Refuted or weakened in the same study:

- **Sale price is dead for accuracy too**, not only for betting. Pre-racing
  auction prices and the sire's sale record: −0.0002 at best, no better than a
  shuffled twin, in three training arms.
- **Jockey residuals against the model:** +0.0002 against their twin. Trainer
  and sire residuals are real; the jockey's is not.
- **Form franking alone** fell from −0.0041 on development races to −0.0013
  (z −1.6) on unseen races. Unconfirmed as a separate group.
- **Same-day bias** was −0.0001 on the unseen races.
- **The rest of the feature store** (1,176 unused numeric columns) predicts
  the market's disagreement with the model no better than the model's own 193
  inputs (R² 0.138 against 0.132). There is little left in it.
- **Screen-capacity references flatter large corrections.** A correction that
  sees only the generalist's own inputs gains −0.0015 on a 400-tree fit and
  half of that on a native fit. Check large-recipe screens against a
  native-capacity forecast.
- **Run-by-run horse state and 30/90-day stable form, as an addition to the
  seven confirmed corrections: not confirmed.** In development the two groups
  added −0.0036 to the stack and −0.0047 in rehearsal (z −4.7), and horse
  state alone carried over to the native 2025Q3 forecast (−0.0047). The
  registered second read on the unseen races gave −0.00175, se 0.00100,
  z −1.7, p 0.04 against a pass mark of 0.005. Alone against the base, horse
  state was −0.0065 (z −5.1) there, so it is a real correction and probably a
  better own-horse group than the one it would replace; its value **on top
  of** the seven is unproven. Development increments on a stack that already
  holds a correlated correction shrank by almost two thirds here; expect the
  same of the next one.
- **A horse's past closing prices without the model's forecast:** −0.0003,
  2/4 quarters, +0.0004 on the native forecast. The value is in the price
  against the forecast, so it needs stored out-of-fold forecasts and cannot be
  an ordinary feature-store column.
- **Past late price moves** (08:00, 11:00 and one hour to BSP at earlier
  runs): −0.0016 alone, but a negative weight and a worse stack once the
  horse's past closing gaps are present. Redundant.
- **One joint correction model over all 359 error inputs:** ties the weighted
  sum of eight (−0.0006 [−0.0017, +0.0006], 2/4). Simpler, not more accurate;
  the per-group recipe is too small for it (+0.0022 worse).
- **Input audit against racecards: no fault in the training rows.** Draw,
  form, days off, ratings, weights and trainer names hold up; the old cached
  racecards (before 21 June 2026) carried cloth numbers as draw and are not a
  reference. Do not re-audit these fields without a new reason. The one gap
  is runs the database lacks: their card form figures add −0.0006
  (`hidden-form`, 4/4, development only).
- **Racing patterns read off the worst misses** (mark well below a recent
  high, handicap debut after runs at huge prices, stable switch, first-time
  headgear, return to a course where placed): each wins as often as the
  stacked forecast expects across 109,593 runners (no z above 1.6). The
  misses are gambled horses; the patterns are selection on the outcome.
- **Owner error inputs** (owner names from the Racing TV archive; the model's
  past residual and market gap by owner, owner with trainer, owner with horse
  state): −0.0022 alone, +0.0002 [−0.0003, +0.0007] added to the best joint
  correction on native forecasts. Null; section 11 of the error report.
- **Newcomer error inputs** (dam; sire, dam's sire and jockey by horse state;
  trainer by state and age band): −0.0049 alone (4/4), −0.0003
  [−0.0007, +0.0001], 2/4 inside the joint correction. Overlap with the
  horse-state and trainer-by-state inputs, not new information. The remaining
  loss to the 08:00 price is all in newcomers (flat non-handicaps, fields of
  unraced horses), and keyed error histories do not reach it.
- **Exact age from the foaling date (`foaled`): LEAK, do not use.** −0.0016
  at native capacity inside the joint correction, but 22% of horses carry a
  1 January placeholder in profiles all fetched in 2026; the placeholder is
  commoner the shorter the career and those horses win 7–23% less often than
  BSP implies. The precision of the date encodes the later career. Section 11
  of the error report. Any future input from the pedigree profile (fetched
  once, long after the races) needs the same check: does its presence or
  precision vary with career length?
- **Exact age on Racing TV foaling dates, leak-safe (inputs only in races
  where every runner has a date):** −0.00002 [−0.0004, +0.0004], 1/4. Null:
  the earlier −0.0016 was entirely the placeholder leak. Runners with no
  Racing TV date still win 12% less often than BSP implies, so never feed the
  raw date with per-runner gaps. The store's `profile_*` foaling inputs are
  0% filled historically (capture-time rule) and that costs nothing.
- **Pace from in-running comments, full history** (early position from the
  Racing TV close-up comment since September 2020; style, likely leaders,
  pace pressure, lone leader, course leader bias): −0.0003 [−0.0007, +0.0002].
  Null. The store's pace inputs lack history before mid-2023, but history is
  not what they were missing.
- **Form matched to today's conditions** (similarity-weighted past
  performances over distance, a continuous going scale from the 193 raw
  descriptions, course and class): −0.0002 [−0.0009, +0.0006], 2/4. Null; the
  seven going buckets and per-category win rates lose nothing measurable.
- **Raw race context added back to the 87-input model** (21 columns:
  distance, going, race type, surface, handicap, race kind, prize, course
  shape): −0.00002 against the same columns shuffled. Null, tested directly
  with `admit`; the selection did not drop context by a ranking artefact.
- **Repairs to kept inputs** (claim-adjusted weight with a plain-weight
  fallback; winners credited their winning distance in the margin figures;
  hidden form blanked for horses from the database's first six months): all
  within ±0.001 of the unchanged model. The claim-adjusted weight inputs are
  blank for 31% of 2020–2023 runners in the 13 November store and that costs
  nothing measurable (+0.0009 [−0.0004, +0.0023]).
- **More race-relative views of kept ability measures** (pairwise win share,
  log softmax share, gap to the three best rivals, count of close rivals; 40
  columns added to the 87): −0.0002 [−0.0025, +0.0021], 3/4. Null as an
  addition; replacing the existing mean/z/rank/best-rival views is untested.
- **Trainer and owner record by handicap situation ("laying one out")**:
  past residual and closing-price gap keyed by handicap debut, below last
  winning mark, eased mark, three unfancied runs, class drop or trip switch.
  −0.0025 4/4 alone as a correction, +0.0002 [−0.0004, +0.0007] 1/4 once the
  existing joint correction is present. Already carried by the trainer and
  horse price-gap inputs. Third group (after owner, newcomer) to be real
  alone and null inside; judge new correction groups inside the joint model.
- **Second-round corrections** (entity and horse records re-measured against
  the corrected forecast, second correction on top): -0.0004 on native, of
  which -0.0002 appears with shuffled inputs; real vs twin -0.0002
  [-0.0007, +0.0002]. Round one has taken it.
- **Filtered horse state, debut race choice, price ratings through rivals**:
  -0.0047, -0.0034 and -0.0007 alone; all three inside the joint correction
  -0.0001 [-0.0010, +0.0007], 2/4 on native. Six groups are now real alone
  and null inside; the joint correction is saturated for records of entities
  and horses against the model's forecast. New information is needed, not
  new keys or weightings of the same residuals.
- **Yard change, discipline for the present yard, Irish yards in Britain, weight
  with trip** (14 November, from one race read closely): -0.0012 (3/4) and
  -0.0007 (4/4) alone; both inside the joint correction +0.00003
  [-0.0007, +0.0008], 2/4 on native. The model is not low on horses improved
  since a stable change (won/model 1.00). Eight groups real alone, null inside.
