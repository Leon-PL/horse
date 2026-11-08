# Meta-learner plan: a stacked model built from specialist sub-models

Initial proposal, 2026-11-07. The [8 November review and revised sequence](MODELLING_REVIEW_AND_PLAN_2026-11-08.md)
correct this proposal's baseline, OOF API and untouched-holdout assumptions.
The foundations and supported selection path now live in `src/modelling/`;
see [modelling operations](../operations/MODELLING.md). The specialist phases
below remain research candidates, not implemented or qualified models. Check
actual readiness with `scripts/project_status.py`.

## Goal

Replace the single win model with a meta-learner that combines sub-models. Each
sub-model is a specialist that sees one kind of information. The meta-learner
weights them by context and keeps the market out until a separate, final tier.

A stack only gains when its sub-models make different mistakes. In order of
strength, that difference comes from:

1. different information (data the other sub-models do not see)
2. different targets (winner, performance in pounds, head-to-head)
3. different model families fitted to the same table (the weakest source).

## Starting point (snapshot, 2026-11-07)

- **Current model.** One LightGBM model trained on a race-level softmax over
  finishing order (`config.WIN_OBJECTIVE = "exploded"`). Its training target is
  half the BSP-implied probability and half the result
  (`WIN_SOFT_LABEL_BLEND = 0.5`). Market prices are never model inputs. The
  final market blend (`apply_market_anchor`) is off (`MARKET_ANCHOR = False`).
- **Feature store.** 757,580 runners from 2020-09-15 to 2026-11-04, with 1,231
  columns. `project_status` reported it stale: the code and the ATR and RTV
  caches have changed beneath it. The promoted model was trained on an older
  feature version.
- **Limits.**
  - About six years of data. Every sub-model needs out-of-fold predictions, and
    the meta-learner needs a later period of its own.
  - Tracking and sectional data begin in mid-2023.
  - LightGBM is the only permitted model framework (CLAUDE.md, `RacePredictor`).
    A neural sub-model needs that rule lifted for sub-models.

## Sub-models

| # | Sub-model | What it uses that already exists | Missing compared with the ideal | Phase |
|---|---|---|---|---|
| S0 | Current full model | The inputs in `feature_sets/win.txt` | Baseline, and the strongest single input | 0 |
| S1 | Ability | Elo, Glicko, TrueSkill and margin Elo; official-rating changes; beaten lengths; weight; form; class | A tracker that measures ability in pounds | 1 |
| S2a | Clock | Speed figures (about 90% coverage since 2021) | None | 1 |
| S2b | Tracking | RTV RaceiQ (top speed, stride, acceleration, speed lost, jumping; from 2023-06, about 90% coverage) and ATR sectionals (splits, energy distribution, efficiency; from 2023-08, 57–68% coverage) | Full GPS for every run | 2 |
| S3 | Trip and luck | ATR in-running comments, RTV speed lost and lengths gained jumping, BHA stewards' reports | Video | 2 |
| S4 | Pedigree and unexposed horses | Sire, dam and dam-sire rates by going, distance and surface; age; sex; foaling date | Genetics, breeze-up times | 1 |
| S5 | Conditions fit | Going, distance, surface, course shape and direction, first time at a course, distance and going specialists | None | 1 |
| S6 | Pace and draw | Running-style index, field pace pressure, draw bias, ATR and RTV early pace | A race simulator, same-day bias | 2 (simulator in 4) |
| S7 | Connections and readiness | Trainer, jockey and trainer-jockey families; days since last run; runs in the last 14 and 30 days; headgear; wind operations; class and handicap moves; travel | Gallops data, vet records | 1 |
| S8 | Text | Timeform comment score (`rtv_text_logit`, 2021 onward); Sporting Life commentary and verdicts; ATR comments; stewards' reports | Language-model embeddings | 2 |
| S9 | Performance spread | Same inputs with a new target: performance in pounds, predicted as an average and a spread, then the race simulated many times | None | 3 |
| S10 | Head-to-head | Pairwise model; form lines through common opponents (Bradley–Terry) | None | 3 |
| S11 | Sequence and graph | None yet | Needs PyTorch | 4 (needs the framework rule lifted) |
| M1 | Market price | BSP; last traded price at 11:00, 1 hour, 5 minutes and 60 seconds before the off; opening odds (from 2020) | Depends on when the bet is placed | 3 |
| M2 | Market flow | Price paths from Betfair BASIC streams; volume and average traded price from the public BSP files | Order-book depth (Betfair PRO data) | 3 (PRO data in 4) |

There is no separate fitness model, because there is no gallops or vet data.
Its available inputs are in S7.

## Phases

### Phase 0: foundations, no new models

1. Rebuild the store and retrain S0, so every comparison uses the same features.
2. Build a harness in `research/meta_learner/` with one registry entry per
   sub-model: its feature list, target and settings. For each sub-model, the
   harness:
   - trains with the exploded-softmax objective
   - produces out-of-fold predictions with `src.wf_eval.oof_predict_expanding`
     on quarterly, purged folds (`expanding_period_folds`)
   - saves each runner's race-centred log-odds and an availability flag to one
     Parquet file keyed by race and runner.
3. Fit meta-learner v0: a linear conditional logit. Each runner's score is a
   weighted sum of sub-model log-odds, and a softmax runs within each race. It
   is fitted by minimising race log-loss (`src.wf_eval.race_logloss`) and has
   about 15 parameters.
4. Add diagnostics:
   - the correlation matrix between sub-models
   - each sub-model's correlation with `won − market_implied_prob`
   - how much the meta-learner's log-loss worsens when each sub-model is
     removed.

**Done when:** with S0 as its only input, the harness exactly reproduces S0's
own out-of-fold log-loss.

### Phase 1: specialists built from existing features (S1, S2a, S4, S5, S7)

- Take most feature lists from the store's column prefixes, which already group
  the features by family.
- Train the specialists on results only, not on S0's market blend. Otherwise
  each specialist partly copies the market, and they become correlated with
  each other through it. Check that choice once, on S1, against the 0.5 blend.
- Meta-learner v1 adds context-dependent weights. Each sub-model's weight
  becomes a linear function of field size, race type, the log of the horse's
  run count and the availability flags.
- **Gate:** the meta-learner beats S0 on race log-loss in each of the last four
  development quarters.
- **Expectation:** specialists trained on subsets of the features S0 already
  uses may add little. If so, Phase 1 mainly proves the harness, and Phase 2 is
  the real test.

### Phase 2: sub-models with new information (S2b, S3, S6, S8)

- These should supply most of the independence from the market: tracking data,
  text, and how the race is likely to be run.
- Before its coverage begins, each sub-model outputs a flag saying it has no
  prediction. The meta-learner learns to give it zero weight there, so races
  before 2023 stay usable.
- Text: extend the existing comment-score pipeline (`src/rtv_text.py`) to
  Sporting Life verdicts and commentary, ATR comments and stewards' reports.

### Phase 3: different targets and the market tier

- Build S9 (performance spread) and S10 (head-to-head).
- Tier 2 combines the tier-1 result with M1 and M2, using context-dependent
  weights. It generalises `apply_market_anchor` in `src/model.py`.
- **Decision needed before this phase:** when bets would be placed (11:00, five
  minutes before the off, or at BSP). That decides which prices the market tier
  may use without leaking.
- Run one pre-registered test on the golden hold-out (races from 2026-06-11).

### Phase 4: beyond current data and framework (optional)

- A sequence or graph neural network (needs the LightGBM-only rule lifted, for
  sub-models only)
- A pace simulator
- Betfair PRO data for order flow
- Video analysis for the trip model

## How the time periods nest

- Sub-model predictions for quarter *q* come only from fits on earlier data,
  with a purge gap before *q*.
- The meta-learner for quarter *q* is fitted only on out-of-fold predictions
  from quarters before *q*.
- Out-of-fold predictions start around 2022-Q1, and meta-learner scoring starts
  around 2023-Q1.
- Golden hold-out races (from 2026-06-11) stay out of all development fitting
  (`src.wf_eval.development_mask`).
- For serving, the sub-models are refitted through the cutoff date, and the
  meta-learner is fitted on all out-of-fold predictions.

## Project rules that apply

- Work starts in `research/`. Nothing in `src/`, `scripts/` or `tests/` may
  import it.
- Use the walk-forward primitives in `src/wf_eval.py`. Do not hand-roll fold
  loops.
- Any module moved into `src/` that can change a feature value must be
  registered in `_FE_SOURCE_FILES` (`src/featured_dataset.py`).
- Read CLV with `src.clv.summarise_clv`, never the mean of `taken_odds / bsp`.
