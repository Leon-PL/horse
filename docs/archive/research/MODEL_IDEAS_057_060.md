# 057–060 — Four ways to make the win model stronger (2 November 2026)

**Question.** Do four of the ideas in [MODEL_IDEAS_BRIEF.md](MODEL_IDEAS_BRIEF.md)
move the win model's probabilities closer to the actual results? The yardstick
is race log loss of the actual winner. ROI and CLV are not used. Idea 5 (a
language model reading the comments) was not run.

**Answer.** Learning from the whole finishing order (057) is a clear gain:
**−0.0046 [−0.0064, −0.0026], better in all four quarters**. That is larger
than the 052 comment score. Predicting a performance figure and its spread
(058) is worse than production on its own (+0.0048). Blended with the model,
though, it adds information: −0.0042 over production and −0.0017 over the 057
model. Blending two near copies of one model gains only 0.0002–0.0006. Luck
weights (059) and learned connection profiles (060) are null, at −0.0003 and
+0.0001, with both intervals crossing zero.

The best combination, the 057 model blended with the 058 figures, is −0.0065
[−0.0092, −0.0037] against production. That closes about 7% of the gap to
the market. The 055 drifter and steamer bias barely moves.

A follow-up (061) used places below third. It found nothing more: down-weighted
they tie with the top three, and flattening them makes the model
over-confident.

A CatBoost pairwise test (062) ties the 057 model and does not beat it. On
production's soft target it is −0.0037 [−0.0064, −0.0011] against production
and +0.0009 [−0.0014, +0.0031] against 057. Blended with 057 it gains −0.0016,
about as much as the figure model, but added to the 057 + figure blend it gains
only −0.0008, with an interval that crosses zero. A fast version on results
alone was much worse (+0.0169). CatBoost stays out of production.

## Instrument

All four experiments use the same setup:

- **Data:** the 048/049 harness, with the development cache rebuilt on the 056
  store (FE `69babeb9c4262ee2`, data `826c9478e6f726b8`). It holds 615,473
  runners in 65,848 races, 2021-05-15 to 2026-06-03. The golden hold-out is
  never loaded.
- **Folds:** purged expanding quarterly folds (`wf_eval.expanding_period_folds`).
- **Inputs:** the production inputs, `feature_sets/win_selected_056.txt`
  (129 inputs, including the comment score).
- **Recipe:** the production win recipe (`feature_selection.fit_win_model`),
  changed only where an idea changes it.
- **Choices:** made on the selection quarters 2024Q1–2025Q2 at screen
  capacity (400 trees). Each choice was written to a registration file before
  any confirmation fit: 057 at 15:30 and 059 at 15:38. The first confirmation
  fit finished at 15:45.
- **Confirmation:** 2025Q3–2026Q2 at native capacity, on 11,905 races and
  109,631 runners. Non-finishers stay in every field.
- **Intervals:** paired per-race differences, with 95% intervals from a
  date-clustered bootstrap.

| Re-measured baseline, 11,905 races | Race log loss |
|---|---|
| Market (BSP, else SP) | 1.7448 |
| Production recipe, 129 inputs | 1.8314 |
| Gap | 0.0865 |

This matches 052's 0.0866 with the comment score.

Code and data:

- **Shared harness:** `scripts/model_ideas.py`.
- **Per-experiment scripts:** `scripts/exploded_softmax_057.py`,
  `thurstone_058.py`, `luck_weights_059.py` and `connection_profiles_060.py`.
  `scripts/run_model_ideas.sh` runs them all in order.
- **Modules:** `src/race_objectives.py`, `src/thurstone.py`,
  `src/luck_weights.py` and `src/connection_profiles.py`, each with unit tests.
- **Receipts:** `data/research/model_ideas_057/`.

## 057. Learn from the whole finishing order — confirmed

**What changed.** The production learner is a per-runner cross-entropy on soft
labels (half market, half result). Two challengers swap only the loss
(`src/race_objectives.py`):

- **Race softmax, winner only.** The loss is a softmax over the race, using
  the same soft target as production. This isolates the change of objective.
- **Exploded softmax.** It reads the result as three choices: the winner from
  the whole field, then second from the runners left, then third. Stage
  weights are (1, r, r²).
  - Non-finishers stay in every stage's field and are never chosen.
  - A dead heat or a missing place ends that race's sequence.

**Shared settings.** Everything else is the production learner:

- the same hyper-parameters and recency weights;
- the same purged early-stopping split, early-stopped on the winner's race
  log loss;
- a refit on every row at the selected tree count.

**Stage weights, chosen on development quarters** (screen capacity, 19,086 races):

| Arm | Race LL | vs production | Quarters better |
|---|---|---|---|
| Production | 1.85792 | – | – |
| Race softmax, winner only | 1.85350 | −0.0044 [−0.0063, −0.0027] | 6 of 6 |
| Exploded, r = 0.25 | 1.85231 | −0.0056 [−0.0075, −0.0038] | 6 of 6 |
| Exploded, r = 0.5 | 1.85575 | −0.0022 [−0.0043, −0.0001] | 6 of 6 |
| Exploded, r = 1 | 1.86345 | +0.0055 [+0.0029, +0.0082] | 0 of 6 |

The rule picked r = 0.25, so the stages are weighted (1, 0.25, 0.0625).
Giving second and third the same weight as the winner hurts. Lower places
carry real information, but much less than the win: beaten horses are eased,
as Benter found.

**Confirmation** (native capacity, 11,905 races):

| Arm | Race LL | Gap to market | 2025Q3 | 2025Q4 | 2026Q1 | 2026Q2 |
|---|---|---|---|---|---|---|
| Production | 1.83135 | 0.0865 | 1.87154 | 1.81423 | 1.77494 | 1.85473 |
| Race softmax, winner only | 1.82918 | 0.0844 | 1.86981 | 1.81347 | 1.77204 | 1.85093 |
| Exploded (1, 0.25, 0.0625) | **1.82675** | **0.0819** | 1.86741 | 1.81061 | 1.77181 | 1.84675 |

| Comparison | Race LL change | 95% interval | Quarters better |
|---|---|---|---|
| Softmax − production (objective) | −0.00216 | [−0.00385, −0.00034] | 4 of 4 |
| Exploded − softmax (order information) | −0.00243 | [−0.00380, −0.00105] | 4 of 4 |
| **Exploded − production** | **−0.00459** | **[−0.00635, −0.00261]** | **4 of 4** |

The brief's success test is met: the interval is below zero and the gain holds
in every quarter. The two parts are about equal. Treating the race as one
choice among the runners is worth about 0.002, and the second and third
places add another 0.002. Together they close 5% of the gap to the market.
The confirmation gain is smaller than the screen's −0.0056, because early
stopping at native capacity narrows it.

It cannot be a leak:

- Finishing positions are used only as training labels, never as inputs.
- Test rows are scored from the same 129 inputs.
- The model still trails the market by 0.082.

**Cost.** A native fold takes 15–23 minutes, against 7–9 for production,
because the gradient is computed in a numba kernel on every boosting round.

## 059. Train on luck-adjusted results — not confirmed

**What changed.** Training rows get extra weights. Evaluation is still on the
actual results, unweighted. Each runner gets a trouble severity from its own
race only (`src/luck_weights.py`):

| Source | Coverage | Severity |
|---|---|---|
| Racing TV close-up comment | 99.5% of runners | 1 for fell, unseated, brought down, badly hampered, lost all chance, lame, bled; 0.5 for hampered, short of room, no clear run, checked, carried wide |
| BHA stewards' events (`stewards_events.parquet`; IHRB left out as mis-keyed, 055) | 56,416 runners | 1 for fell, unseated, brought down, lame, bled, distressed; 0.5 for interference, lost shoe, breathing |

Several things are not counted as luck:

- Habitual traits: slowly away, jumping errors, hanging, pulling.
- Pulled up.
- A winner's trouble.

10.1% of runners carry trouble (7.5% moderate, 2.6% severe). In 21% of races
the troubled losers carried more than 0.1 of the market's win chance, and in
5.6% more than 0.25.

Sectional times, the brief's third source, were not used:

- The ATR cache covers only 2023-08 to 2026-06.
- Its derived run figures were found mis-built in the 2026-10-29 audit.
- The close-ups already describe the same incidents.

**Scheme, chosen on development quarters** (screen capacity):

| Scheme | Race LL | vs production | Quarters better |
|---|---|---|---|
| Race weight 1 − 2 × (troubled losers' market chance) | 1.85648 | −0.0014 [−0.0027, −0.0002] | 6 of 6 |
| Race weight 1 − 4 × (same) | 1.85710 | −0.0008 [−0.0022, +0.0006] | 5 of 6 |
| Troubled loser's own row × (1 − severity) | 1.85670 | −0.0012 [−0.0024, −0.0002] | 5 of 6 |

**Confirmation** of the registered race-level scheme (β = 2):

| | Race LL | 2025Q3 | 2025Q4 | 2026Q1 | 2026Q2 |
|---|---|---|---|---|---|
| Production | 1.83135 | 1.87154 | 1.81423 | 1.77494 | 1.85473 |
| Luck-weighted | 1.83106 | 1.87245 | 1.81403 | 1.77274 | 1.85460 |
| Change | **−0.00029 [−0.00112, +0.00058]** | +0.00092 | −0.00020 | −0.00220 | −0.00013 |

The interval crosses zero and one quarter is worse, so the idea is not
confirmed. The development gain of −0.0014 shrank to a fifth of that at native
capacity. Like the trouble inputs in ledgers 2b and 7, trouble moves the
model only within noise. The model's errors are not driven by a minority of
unlucky results.

## 060. Learned profiles for trainers, jockeys and sires — not confirmed

**What changed.** Each trainer, jockey and sire gets a learned profile
(`src/connection_profiles.py`, numpy only). The profile models a runner's
result against the market:

    won − SP chance  ~  b[entity] + u[entity] · Σ v[context]

It is fitted by alternating ridge least squares, with rank 4, a two-year
half-life and ridge equal to 30 runners. The context columns are:

| Entity | Context columns |
|---|---|
| Trainer | discipline; debut / second run / new yard / layoff; handicap; class; age; trip; field size; season |
| Jockey | discipline; handicap; class; age; trip; field size |
| Sire | discipline; trip; going; age; debut |

**Avoiding leakage.**

- The profiles are refitted every quarter, from 2020Q4 to 2026Q2. Each fit
  uses only races before that quarter, minus the 7-day purge
  (`expanding_period_folds`), and scores only that quarter's runners. No row's
  profile includes its own result.
- The golden hold-out is removed before any fit.
- Factor coordinates rotate freely between refits. So only quantities that do
  not depend on the rotation become inputs, 13 in all:
  - the profile's prediction for the runner's own context;
  - the entity's bias;
  - the entity's sample size;
  - the trainer's projections for debutants, second runs and new-yard runners;
  - the sire's projection for debutants.
- Every cached row has a value.

**Confirmation:** the production inputs plus the 13 profile inputs.

| | Race LL | 2025Q3 | 2025Q4 | 2026Q1 | 2026Q2 |
|---|---|---|---|---|---|
| Production | 1.83135 | 1.87154 | 1.81423 | 1.77494 | 1.85473 |
| With profiles | 1.83146 | 1.87283 | 1.81294 | 1.77516 | 1.85486 |
| Change | **+0.00011 [−0.00092, +0.00113]** | +0.00129 | −0.00129 | +0.00022 | +0.00014 |

The profiles are better in only one quarter. They are also priced. On the
confirmation runners, no profile input correlates with won − market beyond
|0.007|:

- the strongest are the trainer's new-yard, second-run and debut projections,
  at 0.0067, 0.0065 and 0.0061;
- the contextual fits are 0.0027 for trainers, −0.0013 for jockeys and
  −0.0005 for sires.

That is the level ledger 2b calls priced. The yard-change and debutant parts
of the 055 gap are not in the connections' past record against the market.
Whatever the market knows there, it learns from somewhere else.

## 058. Predict a performance figure and its spread — worse alone, adds in a blend

**What changed.** Instead of one score per runner, two models predict two
numbers (`src/thurstone.py`):

- **A mean figure:** how many pounds behind the winner the horse will finish.
  - Beaten lengths become pounds at 15 ÷ furlongs lb per length: 3 lb at 5f,
    1.9 lb at a mile, 0.94 lb at two miles.
  - Margins are capped at 30 lb, so eased, tailed-off runners do not dominate.
  - The model is fitted on differences within each race, so no race par is
    needed.
- **A spread:** how far the horse's run may land from that mean. It is a
  second model, fitted on the first model's out-of-fold squared errors.

The win chance is the probability that the horse's run is the best in the
race, with normally distributed runs. It is computed by numerical
integration.

**Setup.**

- One scale on all spreads is fitted on the purged early-stopping slab, which
  never overlaps the scored quarter.
- Both models use the 129 production inputs and the production
  hyper-parameters, recency weights and early stopping.
- Non-finishers and runners without a recorded margin (6.2% of runners
  together) have no figure, so they are left out of both fits. At scoring time
  they keep their predicted figure and spread and stay in every evaluated
  field.

| Fold | Training figures | Mean-figure trees | Spread trees | Spread scale | Out-of-fold error SD |
|---|---|---|---|---|---|
| 2025Q3 | 472,490 | 3,137 | 1,628 | 1.615 | 8.09 lb |
| 2025Q4 | 504,051 | 2,590 | 1,420 | 1.602 | 8.09 lb |
| 2026Q1 | 532,092 | 3,360 | 1,677 | 1.569 | 8.08 lb |
| 2026Q2 | 553,055 | 3,966 | 1,859 | 1.562 | 8.07 lb |

**On its own** (native capacity, 11,905 races):

| | Race LL | Gap to market | 2025Q3 | 2025Q4 | 2026Q1 | 2026Q2 |
|---|---|---|---|---|---|---|
| Production | 1.83135 | 0.0865 | 1.87154 | 1.81423 | 1.77494 | 1.85473 |
| Figure and spread | 1.83614 | 0.0913 | 1.87867 | 1.81870 | 1.78536 | 1.85095 |
| Change | **+0.00480 [+0.00084, +0.00903]** | | +0.00714 | +0.00447 | +0.01042 | −0.00378 |

It is worse in three quarters out of four. As a replacement for the production
model, it fails the brief's test.

**As a second opinion.** The two models' probabilities are combined
geometrically, as p ∝ (first)^a × (second)^b, then renormalised within the
race:

- The two weights are fitted on the earlier confirmation quarters and scored on
  the next one, over 8,370 races in 2025Q4–2026Q2.
- The production + figure blend was part of the report written before
  confirmation.
- The other pairs were added afterwards, as controls. Averaging two models
  helps a little even when they know the same things, so each gain needs
  comparing with a blend of near copies.

| First model | Second model | Change vs first model alone | 95% interval | Quarters better | Weight on second |
|---|---|---|---|---|---|
| Production | Luck-weighted production (near copy) | −0.00033 | [−0.00090, +0.00021] | 1 of 3 | 0.19–0.58 |
| Production | Production + profiles (near copy) | −0.00024 | [−0.00091, +0.00037] | 2 of 3 | 0.15–0.53 |
| Exploded | Winner-only softmax (near copy) | −0.00042 | [−0.00105, +0.00024] | 2 of 3 | −0.02–0.09 |
| Exploded | Luck-weighted production | −0.00058 | [−0.00128, +0.00013] | 2 of 3 | 0.07–0.23 |
| Production | **Figure and spread** | **−0.00417** | **[−0.00603, −0.00230]** | **3 of 3** | 0.35–0.38 |
| Exploded | **Figure and spread** | **−0.00172** | **[−0.00288, −0.00051]** | **3 of 3** | 0.20–0.23 |

What this shows:

- **Blending near copies gains 0.0002–0.0006, never significantly.**
- **The figure model's gain is real information.** Over production it is ten
  times the near-copy level. Over the 057 model it is three times the
  near-copy level.
- **The 057 model contains production.** Adding production's recipe back to it
  gains nothing significant.

| 8,370 races, 2025Q4–2026Q2 | Production | Exploded | Exploded + figure blend |
|---|---|---|---|
| 2025Q4 | 1.81423 | 1.81061 | 1.80863 |
| 2026Q1 | 1.77494 | 1.77181 | 1.77094 |
| 2026Q2 | 1.85473 | 1.84675 | 1.84451 |
| Pooled change vs production | – | | **−0.00651 [−0.00923, −0.00374]**, 3 of 3 |

**Does the spread behave as the brief expected?** Partly.

Expected wins below are the summed race-normalised probabilities.

| Group | Runners | Wins | Production expects | Figure model expects | Market expects | Mean spread (lb) |
|---|---|---|---|---|---|---|
| Debutants | 7,852 | 653 | 686.7 | 640.4 | 667.7 | 8.22 |
| First run for a new yard | 4,012 | 353 | 338.5 | 339.1 | 350.7 | 8.09 |
| Everyone else | 97,767 | 10,899 | 10,879.8 | 10,925.5 | 10,886.6 | 7.42 |

| 11:00 → BSP move (055's labels) | Runners | Wins | Production expects | Figure model expects | Market expects | Mean spread (lb) |
|---|---|---|---|---|---|---|
| Steamer (chance up ≥ 1.5×) | 8,827 | 898 | 667.2 | 685.9 | 948.2 | 7.88 |
| Drifter (chance down ≥ 1.5×) | 18,419 | 653 | 1,102.1 | 1,079.0 | 611.9 | 7.57 |
| Stable | 67,409 | 9,173 | 8,954.7 | 8,959.1 | 9,163.9 | 7.34 |
| No 11:00 market (whole races) | 14,976 | 1,181 | 1,181.0 | 1,181.0 | 1,181.0 | 7.90 |

- **The spread is wider where it should be.** Debutants average 8.22 lb and
  new-yard runners 8.09 lb, against 7.42 lb for everyone else. Steamers sit
  above stable runners too, at 7.88 lb against 7.34 lb.
- **Debutants: the figure model is closer, but not provably.** Production
  expects 34 debutant wins too many and the figure model 13 too few. Both
  errors are within about 1.3 standard deviations of chance (√653 ≈ 26).
- **New-yard runners: no change.** Both models expect about 339 wins against
  353.
- **Steamers and drifters: the bias barely moves.** The figure model closes
  19 of production's 231-win shortfall on steamers and 23 of its 449-win
  excess on drifters, under 10% of each. That fits 055: those errors come from
  money arriving after 11:00, not from uncertainty the inputs can see.

**Verdict.** The brief's success test is not met: on its own the figure
model is worse. But it carries information the boosted classifiers lack, and
it adds −0.0017 on top of the 057 model. The exploded + figure pairing was
measured after the confirmation fits, so it is a strong lead, not a registered
result.

**Cost.** A native fold takes 15–23 minutes (the two regressions, the
out-of-fold halves and the quadrature).

## 061. Beyond third place — nothing more to gain (follow-up to 057)

**Question.** 057 learns from the first three places. Does more of the
finishing order help?

**Arms** (`scripts/exploded_depth_061.py`). Each arm is a stage depth plus a
rule for each stage's weight and temperature:

- **Weight:** the stage's share of the loss.
- **Temperature:** below 1 flattens that stage's choice, treating it as
  noisier.

| Arm | Places used | Stage weights | Stage temperatures |
|---|---|---|---|
| 057 as registered | 3 | 1, 0.25, 0.0625 | all 1 |
| Every finisher, down-weighted | all | 0.25 per place down | all 1 |
| Top three, flattened | 3 | all 1 | 1, 0.8, 0.64 |
| Every finisher, flattened | all | all 1 | t per place down, t = 0.8, 0.65 or 0.5 |

Non-finishers stay in every stage's field and are never chosen, so they rank
below every finisher.

**Rule.** The grid and the rule were fixed in the script before any fit: the
lowest pooled race log loss on the development quarters 2024Q1–2025Q2 at
screen capacity (19,086 races).

| Arm | Race LL | vs 057 | Quarters better than 057 | Top pick's average chance | Top pick won |
|---|---|---|---|---|---|
| Production | 1.85792 | +0.00561 | – | 30.7% | 30.7% |
| 057 (top three, down-weighted) | 1.85231 | – | – | 30.7% | 31.2% |
| Every finisher, down-weighted | 1.85229 | −0.00002 [−0.00109, +0.00102] | 3 of 6 | 30.7% | 31.4% |
| Top three, flattened 0.8 | 1.85762 | +0.00531 [+0.00313, +0.00764] | 0 of 6 | 31.4% | 30.8% |
| Every finisher, flattened 0.8 | 1.86505 | +0.01274 [+0.00963, +0.01587] | 0 of 6 | 31.2% | 30.7% |
| Every finisher, flattened 0.65 | 1.87782 | +0.02551 [+0.02097, +0.03004] | 0 of 6 | 33.3% | 30.9% |
| Every finisher, flattened 0.5 | 1.88956 | +0.03725 [+0.03110, +0.04350] | 0 of 6 | 34.2% | 31.1% |

**What this shows:**

- **Places below third add nothing.** Down-weighted, every finisher ties
  with the top three (−0.00002, three quarters each way). With weights of
  0.016 and less, the extra places barely enter the loss.
- **Flattening hurts, and more flattening hurts more.** The model has one
  score per runner for every stage. To fit a flattened later choice it has
  to spread those scores wider, and that makes its win probabilities
  over-confident. At t = 0.5 its top pick averages a 34.2% chance but wins
  31.1% of the time. The down-weighted models are calibrated.
- **Down-weighting is the right way to use lower places.** Discounted
  Harville models flatten later places when predicting place chances from a
  fixed win model. When the places are used to learn the scores themselves,
  the flattening leaks back into the winner probabilities.

The rule mechanically registered "every finisher, down-weighted", by 0.00002.
That is a tie, so it is not worth a native confirmation: the model would be
the same, at twice the fit time. **057's three places stand.** Receipts:
`registration_061.json` and `data/logs/exploded_depth_061.log`.

## 062. CatBoost pairwise — ties the 057 model on the soft target, and adds little to the blend

**What changed.** A different learner and loss: CatBoost's `PairLogit`, which
models who beats whom, pair by pair (`scripts/catboost_pairwise_062.py`).

- **Pairs:** the pairwise form of 057. The winner meets every other runner,
  second meets the runners below it, and third likewise, weighted
  (1, 0.25, 0.0625) times the production recency weight.
- **Win chances:** a race softmax of the scores, with one temperature fitted on
  the early-stopping slab.
- **Everything else:** the same 129 inputs, folds and purged early-stopping
  split, then a refit on every row.

CatBoost stays out of the project. It is installed beside the locked
environment in `data/tooling/catboost`, so `.venv`, `uv.lock` and production
are unchanged.

| Arm | Labels | Learning rate | Trees | Time per fold |
|---|---|---|---|---|
| Fast | Results only | 0.2 | 100–170 | 2–3 minutes |
| Soft | First stage on production's half-market soft target | 0.05 | 1,382–1,600 | 38–46 minutes |

The soft arm's first stage is every ordered pair in the race, weighted by that
soft target. Stages two and three stay on the results. Both arms use one Newton
step per leaf, because CatBoost's default of 10 doubles the time per iteration.
All four soft folds took 2 hours 46 minutes.

**Alone** (11,905 races):

| | Race LL | 2025Q3 | 2025Q4 | 2026Q1 | 2026Q2 |
|---|---|---|---|---|---|
| Production | 1.83135 | 1.87154 | 1.81423 | 1.77494 | 1.85473 |
| Exploded (057) | 1.82675 | 1.86741 | 1.81061 | 1.77181 | 1.84675 |
| **CatBoost pairwise, soft** | **1.82769** | 1.86911 | 1.80930 | 1.77487 | 1.84733 |
| CatBoost pairwise, fast | 1.84827 | 1.88636 | 1.83373 | 1.79701 | 1.86605 |

| Comparison | Change | 95% interval | Quarters better |
|---|---|---|---|
| Soft − production | −0.00365 | [−0.00644, −0.00111] | 4 of 4 |
| Soft − exploded | +0.00094 | [−0.00140, +0.00310] | 1 of 4 |
| Fast − production | +0.01693 | [+0.01344, +0.02058] | 0 of 4 |
| Fast − exploded | +0.02152 | [+0.01799, +0.02521] | 0 of 4 |

**In blends** (8,370 races, 2025Q4–2026Q2, weights fitted on earlier quarters
only):

| Blend | Compared with | Change | 95% interval | Quarters better |
|---|---|---|---|---|
| Exploded + soft | Exploded | −0.00163 | [−0.00292, −0.00031] | 3 of 3 |
| Exploded + figure | Exploded | −0.00172 | [−0.00288, −0.00051] | 3 of 3 |
| Exploded + figure + soft | Exploded + figure | −0.00079 | [−0.00169, +0.00008] | 3 of 3 |
| Production + soft | Production | −0.00478 | [−0.00702, −0.00253] | 3 of 3 |
| Exploded + figure | Production | −0.00651 | [−0.00923, −0.00374] | 3 of 3 |
| Exploded + figure + soft | Production | −0.00730 | [−0.00995, −0.00444] | 3 of 3 |
| Exploded + fast | Exploded | −0.00057 | [−0.00121, +0.00006] | 2 of 3 |
| Production + fast | Production | +0.00014 | [−0.00032, +0.00062] | 1 of 3 |

Beside 057, the soft arm took a weight of 0.35–0.46. In the three-model blend
the weights were 0.45–0.54 for 057, 0.17–0.20 for the figure and 0.29–0.40 for
CatBoost. Its gain in that blend was −0.0014, −0.0001 and −0.0008 by quarter.

**What this shows.**

- **CatBoost does not beat the LightGBM 057 model.** On the same soft target
  and the same three stages the two are level: +0.0009, with an interval that
  spans zero, and better in only one quarter.
- **The fast arm's loss was not a verdict on pairwise CatBoost.** It came from
  the results-only labels, the fast settings, or both. The soft arm changed
  both at once, so this cannot separate them.
- **As a second model it helps as much as the figure model.** −0.0016 over 057
  is nearly three times the near-copy level of 0.0002–0.0006, and about the
  same as the figure's −0.0017. It sees the same inputs and labels as 057, and
  their log chances correlate 0.990 within a race, against 0.868 for the
  figure. So the gain is probably two different tree fits averaging out each
  other's errors, not new information. This was not measured.
- **As a third model it adds little.** The two second models overlap: each adds
  about −0.0017 alone, and together −0.0025. CatBoost's extra −0.0008 has an
  interval that just crosses zero, and it was almost nothing in 2026Q1.
- **This is a lead, not a registered result.** The soft arm was chosen after
  the fast arm had been scored on these same quarters, and after 2025Q3 had
  been seen. Its settings were not tuned further.

**Decision.** CatBoost stays out of production. It would add a second framework
and about 40 minutes a fit, for a gain over the two-model blend that cannot be
told from zero.

## What to do next

1. **The exploded objective (057) is installed.** It was promoted on 4 November
   as `20261104_235300_444862`. On 3,268 hold-out races it beat 056 by −0.0046
   [−0.0097, +0.0003], matching the walk-forward result. It needs one more
   store rebuild and refit before it can serve live cards; see `docs/archive/CONTINUATION_before_cleanup_2026-11-05.md`.
2. **Decide on the figure model as a second model.**
   - Serving would have to run two models and carry two blend weights, fitted
     out of sample as here.
   - To make it a registered result, fix the weights and the pairing now and
     score the next fresh quarter.
3. **Retire luck weights and connection profiles.** Both are recorded in the
   negative-results ledger.
4. **Leave CatBoost out (062).** It ties 057 and adds little to the blend. Its
   gain beside 057 may come from averaging two different fits. If so, averaging
   several LightGBM fits could capture part of it with no second framework.
   That has not been measured.
