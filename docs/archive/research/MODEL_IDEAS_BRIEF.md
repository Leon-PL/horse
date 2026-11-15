# Brief: unconventional ways to improve the win model (2 November 2026)

This brief hands five untested ideas to a new session. Read CLAUDE.md, then
this file, then [NEGATIVE_RESULTS.md](../NEGATIVE_RESULTS.md) before starting.

**Goal:** make the win model's probabilities closer to the true outcomes,
measured as race log loss against actual winners. The user wants model
strength, not a betting edge, so do not judge these ideas by ROI or CLV.

## Current production (056)

Experiment 056 is installed: production `20261102_143544_663385` uses 129
inputs. These are the 048 inputs plus the Timeform comment score
(`rtv_text_logit`, `rtv_text_prob`, `rtv_verdict_rank`); see
`feature_sets/win_selected_056.txt`. The store is current
(FE `69babeb9c4262ee2`).

These ideas are research. Do not rebuild the store, change
`config.FEATURE_ALLOWLIST` or promote a model unless the user asks.
`select_features prepare` writes a ~3 GB matrix, so check that no other
heavy job is running first.

Number new experiments from **057**, checking that the number is not already
taken in the progress log.

## What has already been tried (do not repeat)

| Idea | Result | Where |
|---|---|---|
| Market probabilities as soft labels (distillation) | Already in the production win recipe | `WIN_SOFT_LABEL_DISTILL`, `src/model.py` |
| Market log-odds as a fixed offset (Benter design) | Learns nothing: race LL vs market −0.00000 | ACTION_PLAN, gate 4 |
| Race-softmax LightGBM, winner only | −0.0010 [−0.0051, +0.0033], unresolved | Ledger 26 |
| Plackett–Luce / TrueSkill ratings as inputs | Better ratings, no market gain | Ledger 045 |
| Race-type and course specialist models | Worse than the general model | Ledger 049 |
| Pace and race-shape projection | Noise | Ledger 046 |
| Fixes for the worst misses (debutants, sire/trainer debut record, off-DB runs) | All within noise | Ledger 055 |

**The realistic ceiling.** The 055 review found that about 46% of the gap
to SP is money arriving after 11:00. No model on public pre-race data can see
it, so roughly half the gap is the realistic target. The rest is morning
information: debutant reputation, yard changes and runners whose Timeform
comment defers to the market.

## The five ideas, in recommended order

### 1. Learn from the whole finishing order, not just the winner

**What.** The production win learner is a binary cross-entropy on soft labels:
each race teaches it one fact, who won. A rank-ordered ("exploded") softmax
reads a result as a sequence of choices: the winner from all runners, then
second from the runners left, then third. Benter used this, giving lower
places less weight because beaten horses are eased.

**Why it might help.** It gets several training signals per race instead of
one. The Plackett–Luce work (045) used finishing order only to build a rating
input, never as the model's own training target.

**How to test.**
- Write a custom LightGBM objective over race groups. Gradient is the
  stage-wise softmax gradient; a diagonal Hessian `p(1-p)` per stage is
  enough. It stays inside LightGBM, the only allowed framework.
- Stages: winner, second, third. Non-finishers rank below all finishers and
  are never a stage's chosen runner. Weight later stages down (for example
  1, 0.5, 0.25).
- Three arms on identical inputs and folds:
  1. production (binary on soft labels);
  2. race softmax, winner stage only;
  3. exploded softmax, three stages.

  Arm 2 against arm 1 isolates the objective change; arm 3 against arm 2
  isolates the extra order information.
- Choose the stage weights on development quarters only, before the
  confirmation run, and write them down first.

**Success.** Arm 3 beats arm 1 on pooled race LL with a date-clustered CI
below zero, and in most confirmation quarters.

**Cost.** Smallest of the five: one objective and one walk-forward run.

### 2. Predict a performance figure and its spread, then simulate the race

**What.** Predict how well each horse will run (beaten margin converted to
pounds) and how uncertain that is, then compute win chances by simulating
the race (Thurstone model).

**Why it fits the evidence.** 055 found that the model over-rates drifters
and under-rates steamers and debutants. Those are the horses whose ability is
uncertain. A single score per horse cannot say "this one could be anything".
A mean with a spread can, and an uncertain outsider then gets more win chance.
Margins exist for every finisher, so the labels are far denser than won or
lost.

**How to test.**
- Margins: use `src/result_margins.cumulative_beaten_lengths`. The raw
  `lengths_behind` column is the gap to the previous finisher, not to the
  winner. Convert lengths to pounds with a distance-dependent scale.
- Comparability across races is the hard part. Options:
  - a race par from the field's pre-race ratings, so the figure is
    "par minus pounds behind";
  - a pairwise loss on margin differences between runners in the same race,
    which needs no par.
- Two LightGBM regressions: one for the mean figure, one for the log squared
  residual (the spread). Win probability comes from Monte Carlo or numerical
  integration per race.
- Non-finishers have no margin. They can be left out of the regression fit,
  but **must stay in the evaluation** (ledger 18: dropping them faked a +79%
  each-way edge).

**Success.** The same criterion as idea 1. Also check whether the predicted
spread is larger for debutants and yard changers, and whether the drifter and
steamer bias from 055 shrinks.

**Cost.** About 2–3 days. Do it after idea 1, especially if idea 1 shows the
model is short of label information.

### 3. Train on luck-adjusted results

**What.** Some results are mostly bad luck, such as the likely winner falling
at the last or being badly hampered. Give those races less weight in
training, using:
- the close-up comments (`source_pipeline.load_matched_text`);
- the stewards reports;
- the sectional times.

Always evaluate on the actual results.

**Why it is still open.** Trouble added almost nothing as an input (ledgers
2b, 7). This uses it differently: it cleans the training labels instead of
adding a model input.

**Risk.** Easy to bias the model. A trouble flag must come from the race being
trained on, never from a later race, and the reweighting must be fixed before
looking at confirmation results.

### 4. Learned profiles for trainers, jockeys and sires

**What.** LightGBM handles thousands of trainer names poorly, so the model sees
hand-built strike rates. Instead, learn a small vector per trainer, jockey and
sire, for example by factorising a trainer × race-type matrix of
performance against expectation. The vector captures style: "targets big
handicaps", "debutants come on for the run".

**Why it might help.** It aims at two parts of the 055 gap, yard changes and
debutants.

**How.** Use numpy only. Each fold fits the vectors on that fold's training
dates; use `wf_eval.expanding_period_folds`. Any new FE module must be
registered in `_FE_SOURCE_FILES`.

### 5. Read the comments with a language model instead of counting words

**What.** The 052/056 comment model counts word pairs, so "not one to trust"
and "one to trust" look alike. A language model could extract specific facts:
- trouble last time;
- ground or trip preference;
- hedges like "market will tell" (055: runners whose comment defers to the
  market hold 13.5% of the gap).

**How.** Test on a sample first, for example one quarter of comments, against
the current comment score on the same races. Only scale up if the sample
shows a gain. Expect API costs for about 700,000 comments at full scale, and
ask the user before running at that scale.

## Harness and rules

- **Evaluation harness.** Use the 048/049 harness:
  - `scripts/select_features.py` `Cache`;
  - `src/feature_selection.fit_win_model` (the production learner,
    `capacity="native"`);
  - `scripts/specialist_models._predict` and `_interval` (date-clustered
    bootstrap CI).

  Confirmation quarters are 2025Q3–2026Q2, on purged expanding folds.
- **Rebuild the cache after a store rebuild.** The cache refuses a store that
  has been rebuilt since it was prepared, and the store was rebuilt for 054 and
  056. Run `python -m scripts.select_features prepare` once the 056 session is
  done.
- **Finish positions.** These are not in the cache targets. Read them from the
  store by `store_row`, as `specialist_models._context` does for race type.
- **Baselines on this harness.** The 052 report gives, on 11,901 races:
  - model 1.834;
  - market 1.745;
  - gap 0.0894, or 0.0866 with the comment score.

  Re-measure the baseline rather than reusing these after the cache is rebuilt.
- **Hard rules:**
  - The golden hold-out (races from 2026-06-11) stays out of research; keep
    the harness's development mask.
  - Use the `src/wf_eval.py` primitives.
  - No global `shift` across horses.
  - Use the merge keys from `src/utils.py`.
  - Run `scripts/project_status.py` before trusting a production number.
- **Recording results.** Record each result in NEGATIVE_RESULTS.md if it
  fails, and in MARKET_PERFORMANCE_PROGRESS.md either way, with the numbers.
  The user's style is plain English with explicit tables; see CLAUDE.md.
