# Experiment 003 — race-level probability training

**Complete: neither challenger advances under the registered rule.** The
linear model is worse; the race-softmax tree has a secondary Brier gain but
an inconclusive primary race-log-loss difference. Retain the current production
recipe. Do not turn the secondary result into a post-hoc promotion rule.

## Results and decision

All six fits completed, followed by scoring on **9,356 races / 85,863 runners**.
The exact cached binary baseline is reproduced from experiment 002. Total
candidate fitting and scoring time was 2,827.93 seconds.

| Forecast | Runner Brier ↓ | Binary log loss ↓ | Race log loss ↓ | Winner accuracy |
|---|---:|---:|---:|---:|
| Retained binary baseline | 0.08754346 | 0.30088334 | 1.87204489 | 29.03% |
| Conditional linear logit | 0.08806124 | 0.30328719 | 1.89083039 | 29.01% |
| Race-softmax tree | 0.08736141 | 0.30062028 | 1.87103531 | 29.52% |

The linear recipe worsens every proper score and every quarter's race log
loss. Its pooled race-LL delta is **+0.01878550**, 95% race interval
**[+0.01088235, +0.02613680]**; date interval
**[+0.01112053, +0.02691215]**. Reject this fixed linear recipe.

The tree's mean race-LL delta is **-0.00100958**, but the race interval
**[-0.00510065, +0.00327606]** and date interval
**[-0.00503771, +0.00326771]** cross zero. It improves two of three quarters,
and pooled Brier/binary LL improve, but it fails the fixed primary confidence
gate. This is **inconclusive primary evidence**, not proof that race-softmax
training is worse or that it provides no useful information.

The tree's runner Brier improves in all three quarters, pooled difference
**-0.00018205**; its one-sided 95% race-cluster upper bound is **-0.00008701**.
Race-mean Brier difference is -0.00032778, central-95% interval
[-0.00053791, -0.00015264]. These are distinct weightings. Lower pooled ECE
(0.000595 versus 0.001463) and higher accuracy are useful secondary findings,
but cannot override the registered primary gate.

| Period | Binary race LL | Linear race LL | Tree race LL | Tree minus binary |
|---|---:|---:|---:|---:|
| 2025Q3 | 1.92063259 | 1.92906599 | 1.91976737 | -0.00086522 |
| 2025Q4 | 1.86228139 | 1.88758831 | 1.85658455 | -0.00569684 |
| 2026Q1 | 1.81799380 | 1.84275379 | 1.82263612 | +0.00464232 |

Stop the bounded architecture expansion here, as planned. Preserve the tree
as an unpromoted research challenger; any reconsideration needs separately
registered confirmation on other data, with no tuning against these quarters.
Do not select the favourable quarter, change the primary measure, optimise a
blend or sweep hyperparameters after reading this report. Independent market
parity remains unachieved; keep M0 as the separately labelled market-informed
reference. Data refresh/refit-cadence validation and genuinely future forecasts
remain separate parts of the accepted plan.

## Same-time market and freshness context

The strict comparison retains **8,942 races / 81,696 runners**. It excludes
394 unmatched/changed fields and 20 winner-label disagreements as whole races,
without overwriting outcomes. Exact inputs are the same as experiment 002.

| Forecast on matched races | Runner Brier ↓ | Race log loss ↓ | Winner accuracy |
|---|---:|---:|---:|
| Binary baseline | 0.08787478 | 1.86879634 | 29.17% |
| Conditional linear logit | 0.08836610 | 1.88682582 | 29.21% |
| Race-softmax tree | 0.08770307 | 1.86768444 | 29.63% |
| T-5 BASIC market | 0.08414852 | 1.74928256 | 34.50% |

Tree minus market race LL is **+0.11840189**, 95% race interval
**[+0.10769787, +0.12957907]**; date interval
**[+0.10698154, +0.12999058]**. The point improvement closes about 0.93% of
the baseline's race-LL gap, well short of the plan's preferred 10% milestone
for a substantial training-design change. None of the independent forecasts
passes practical parity on either race or date resampling. The tree's
one-sided upper bounds are +0.12745909 race LL / +0.00387186 runner Brier;
date sensitivity +0.12850479 / +0.00386969, above +0.002 / +0.0001 margins.

On the fixed **7,719-race** freshness subset, race LL is binary 1.86806567,
linear 1.88237104, tree 1.86596487 and market 1.75239937. The tree's small
point gain and the substantial remaining market gap both persist. This
secondary report neither fits corrections nor changes the advancement rule.

## Prior evidence and reason for this bounded repeat

The deleted `scripts/_ab_pl_objective.py` was recovered from Git at
`823a46c^`, with its exact bytes retained in the experiment directory. It
explicitly removed every runner with `finish_position <= 0`. Its comparison
therefore excluded true losing starters, used the older June 12 snapshot,
400 fixed trees, two percentage folds, no calibration, and no reported mean
race log loss. Its negative result remains relevant prior evidence, but it
does not qualify the current repaired all-starter forecasting protocol.

The accepted plan permits one bounded comparison on the repaired protocol,
then stops this modelling expansion if it fails. No broad architecture,
objective, feature, Hessian, seed or regularisation search is authorised by
this experiment.

## Frozen choices

- Compare the exact cached 149-feature baseline forecasts from experiment 002
  against a conditional linear logit and a conditional LightGBM softmax model.
- Use the same 2025Q3, 2025Q4 and 2026Q1 fields, cutoffs, seven-day purge and
  native training data preparation. Each fit must reproduce the baseline's
  training counts and first/last dates. Retain actual non-finishers.
- Train one cross-entropy contribution per race, with 180-day recency weights.
  All runners within a race share its weight. Do not divide the loss by its
  runner count.
- Linear model: training-only median imputation, missing indicators and
  standardisation; fixed ridge coefficient 0.01; numerical optimisation to a
  declared convergence threshold. No common intercept, which cancels in a
  conditional race softmax.
- Tree model: current fixed tree settings, at most 2,000 rounds; native
  purged inner capacity selection, followed by refitting all eligible history.
  Use complete fields in tree fits, without runner subsampling. This compares
  fixed training recipes, not the isolated causal effect of changing the loss.
- Use the native Platt/isotonic calibration chain. Calibration probabilities
  come from a separate model fitted only to earlier purged history. The final
  forecast model may fit all settled historical labels through its cutoff;
  it does not generate its own in-sample calibration predictions.
- Finish all three linear fits before the tree fits; inspect no full-data
  performance until both candidates complete every period.

The registration is `data/audit/market_performance/003_registration.json`.
A prose clarification distinguishing the calibration-generating model from
the final refit was recorded before any historical candidate fit; no numerical
choices changed. Both registration versions and the clarification are retained.
The final registration SHA-256 is
`52694521a9fe1d138294ade89ea8280c21765d1634de91b8144bd5d4fd2476c3`.
The run manifest additionally binds the code, exact cached inputs, binary model
artifacts and same-time market snapshot. Source is frozen while fitting.

The custom objective consumes raw scores and returns gradients and Hessians,
as specified by the installed [LightGBM 4.6 training API](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.train.html).
Finite-difference checks verify the weighted loss gradient and diagonal
Hessian, and native fits verify cutoff exclusion, calibration separation,
full refit, shuffled fields and save/load replay.

## Fixed advancement and reporting rules

For either candidate, require pooled mean race-log-loss improvement with
both race-resampled and date-cluster central-95% upper bounds below zero,
non-worsening runner Brier and binary log loss, and improvement in at least
two of three quarters. The directional upper bounds correspond to 97.5%
one-sided bounds for each of the two fixed candidates; neither a partial run
nor a favourable subgroup can select a winner. Report both candidates.

Retain all paired forecasts and proper scores on identical complete fields.
The secondary market comparison uses the previously reconstructed T-5 BASIC
last-traded-price snapshot and fixed whole-field freshness criteria: 300
seconds for runner price changes and 90 seconds for market observation.
It fits no market correction and does not select a feature subset.

These dates have been inspected by earlier project research. Results are
retrospective development evidence, not a fresh holdout, executable-price
performance or as-issued live racecard replay. Passing advances a research
recipe; production integration, qualification and fresh forward confirmation
remain separate requirements.

## Evidence

Registration, original prior source, evidence review, input/code manifests,
runtime versions, fitted models and predictions are retained under
`data/audit/market_performance/003_race_objective/`. Logs are under
`data/audit/20261026/experiment-003-*.log`.

Before execution, 13 focused checks passed, including native fits for both
recipes and the two-candidate advancement gate. Full regression subsequently
passed **742 tests**, 24 warnings, in 477.00 seconds. A real-artifact replay of
the three saved linear models and three cached binary baselines matched
recorded forecasts on shuffled fields to 1.11e-16. After fitting, **all nine
artifacts** (six challengers and three binary baselines) passed the same
64-field-per-model replay, maximum error 1.11e-16. Final deep status passes:
database, store, feature code and production match, with the existing 14-day
data-age warning. Production remains `20261025_193016_331639`; feature signature
`221fcd4db0fc5631`, data `f2a120a477c117a1`. Feature values and stored results
are unchanged. Native source, full implementation, runtime versions, input
hashes, whole-field exclusions and final evidence checksums are retained.
