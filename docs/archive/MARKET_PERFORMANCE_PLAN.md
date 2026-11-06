# Plan to approach market forecasting performance

**Dated research proposal, 25 October 2026.** Model IDs and baseline measurements
below describe its starting point. Use the [README](../../README.md) for current
operation and [implementation progress](../MARKET_PERFORMANCE_PROGRESS.md) for
completed work and later installed corrections.

Written 2026-10-25, after the project audit. The user wants both an independent
forecast and a forecast that may use market odds considered. The objective is
accurate, calibrated win probabilities. Betting profitability is a separate,
later question.

This updates the older action plan's priority order **for probability
forecasting**, while retaining the research ledger's measured results and
the project's integrity rules.

## 1. Starting position and the size of the gap

Production is `20261025_193016_331639`. The verified retrospective comparison
contains 3,471 complete races / 32,357 runners, including 947 non-finishers,
from 2026-07-15 through 2026-10-12.

| Lower is better | Current independent model | Normalized result SP reference | Model worse than reference |
|---|---:|---:|---:|
| Runner Brier | 0.0860888 | 0.0820915 | 4.87% |
| Binary log loss | 0.2967461 | 0.2804954 | 5.79% |
| Race log loss | 1.8773931 | 1.7517307 | 7.17% |
| ECE, ten bins | 0.0011182 | 0.0047485 | Model is better on this diagnostic |

The independent model needs approximately a **6.69% reduction in its own race
log loss** and a **4.64% reduction in its own Brier score** to match this
reference on this population. The latest repair delivered about 1% on race
log loss, so another small calibration adjustment is unlikely to close the gap.
The key problem is distinguishing runners, not just matching average win rates.

These reference prices are the result feed's historical starting prices.
They are not automatically Betfair BSP, an executable quote, or the market
available at an earlier prediction time. Rebuild the benchmark for the chosen
operational horizon before treating these numbers as a live target.

Evidence: [audit](PROJECT_AUDIT.md), and the comparison/price-context JSON and
paired forecasts under `data/audit/20261025/`.

## 2. Two tracks with different definitions of success

**Track A — independent of current-race odds.** Historical form and legitimately
available past-race prices may be used; current-race prices are withheld from
the predictor. Aim to close the remaining market gap, with intermediate
milestones of 10%, 25% and 50% of that gap on locked development folds.

**Track B — market-informed forecast.** Begin with probabilities from the market
actually observable at prediction time. Learn a correction only when it improves
later forecasts. Returning those market probabilities unchanged is an honest
baseline and gives exact parity with that particular reference. It is not
evidence that the fundamental model has learned market-level information.

Both tracks must be compared with the same market, timestamp, runner universe
and scoring population. Matching a five-minute forecast does not establish
parity with the eventual closing price.

## 3. Work package P0: define and enforce the experiment

**Priority: first. Estimate: 2–3 engineering days.**

1. Select one primary operational horizon: provisionally five minutes before
   the scheduled start known at that time. Specify how schedule revisions,
   delayed starts and suspended/in-play markets are handled. Keep 11:00 as a
   separate secondary experiment rather than searching many horizons.
2. Name the reference precisely: source, last-traded/back/lay/BSP/SP price,
   observation time and probability conversion. Preserve normalized result SP
   and BSP as separately named retrospective references where available.
3. Add one benchmark entry point, provisionally
   `scripts/benchmark_forecasts.py`, using `src/wf_eval.py` and the strict
   complete-field validation already exercised by the audit comparison.
   Consolidate the scoring helpers currently imported from a research script
   into shared evaluation code; do not create another conflicting harness.
4. Freeze chronological outer folds before experimenting. A proposed starting
   design is six non-overlapping quarterly development blocks, covering flat
   and jumps seasons, with the existing seven-day purge. Inner earlier dates
   choose features, stopping, calibration and blend weights. The June 2026
   onward period has already been inspected; it may be labelled retrospective
   development, never renamed an untouched holdout. Preserve the existing
   default research boundary and require an explicit recorded override.
5. Use mean race log loss as the primary measure, with runner Brier as a
   guardrail and binary log loss, reliability, ranking and coverage reported.
   Retain both runner-weighted and race-weighted Brier, with distinct labels.
   Profit, CLV and ECE alone do not decide a forecasting winner. Proper scores
   evaluate the reported probability distribution, not just its top pick.
   See [Gneiting and Raftery](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
6. Save paired predictions for every model on exactly the same eligible races.
   Keep actual non-finishers as losers. Apply a declared whole-race rule to
   unresolved fields and deadheats, and report all exclusions. Investigate
   the remaining 22 field conflicts; a lack of evidence is not a licence to
   invent runners or remove genuine starters.
7. Record data, feature, training-recipe and protocol hashes; selected columns;
   fitting/calibration cutoffs; price provenance; seed; and package versions.
   Cache out-of-fold predictions against that complete manifest.

**Pass condition:** replaying a frozen baseline reproduces its probabilities
and metrics; later prices/results cannot alter an earlier forecast's inputs;
row permutations and missing-price cases cannot change the scored universe
silently. Every experiment emits a comparable report and paired forecasts.

## 4. Work package P1: make deployment fitting current

**Priority: alongside P0. Estimate: 2–3 engineering days.**

The new model's fitted parameters stop at **2026-07-08** because the next 90
days were retained for evaluation. Its feature history reaches October 12.
Those are different cutoffs. Reusing the evaluation artifact for production
leaves avoidable model-fitting delay.

- Separate historical evaluation from a production `fit_as_of` path, reusing
  the same model-fitting components. Historical folds fit only on their past.
  A deployment artifact can fit all validated, settled results available at
  its declared cutoff after the recipe has been selected.
- Keep calibration based on purged, time-separated predictions. Never fit it
  to a final model's predictions on the same labels used to train that model.
- Save the production refit's own training cutoff and link it to its recipe's
  validation report. Do not present the earlier artifact's holdout scores as
  a fresh test of the refitted artifact.
- Refresh results/features through a declared cutoff in a new store generation;
  preserve the October 25 audit snapshot. Start with a fixed monthly model
  refresh during historical replay, then assess a weekly cadence if needed.
  Daily feature updates and parameter refitting are separate operations.
- Compare fixed versus refreshed models using a forward replay that implements
  that same cadence. Do not infer an improvement merely because a refit is newer.

**Pass condition:** each replayed forecast uses only information available at
that forecast time; the production path fits through its stated cutoff and
passes the existing native-framework, serialization and serving checks.

## 5. Work package P2: diagnose what information is missing

**Priority: before adding complexity. Estimate: 1–2 engineering days.**

Use the shared forecasts to attribute the probability-score gap by broad,
predeclared groups: flat/hurdle/chase, handicap status, field size, career
exposure, favourite probability and sectional-history coverage. Inspect how
much loss comes from underestimating winners versus overstating others.
Market-price bands are diagnostic only in the independent track; they must
not silently become inputs to that track's deployed predictor.

For each group show race count, score difference, uncertainty and contribution
to the overall gap. Do not select a specialist because one small subgroup has
a favourable point estimate. Check whether the older favourite/debutant finding
still holds with corrected fields and the current artifact.

Classify previous negative results by the question they actually answered:

| Earlier evidence | Consequence for this plan |
|---|---|
| No CLV/profit improvement | Strong warning against a betting-edge claim; not automatically a test of independent forecast quality. |
| Near-zero residual correlation with the market | Low prior for a market correction; can miss interactions and need not imply redundancy with the independent model. |
| No incremental proper-score improvement in a valid chronological ablation | Do not repeat unchanged without a material data or method change. |
| Invalid provenance or incompatible populations | Repair and repeat only the load-bearing comparison, not every historical search. |

The ledger's multi-quarter market-offset result is particularly strong
negative evidence: approximately -0.00010 race log loss versus the market,
with an interval spanning zero across 49,554 races. Treat it as a low prior
for a new complex residual model, not an invitation to repeat the same sweep.

**Deliverable:** a short loss-attribution report and at most three independent
feature-family hypotheses, each with a stated mechanism and stop condition.

## 6. Track A experiments: close the independent model's information gap

**Estimate: 5–8 engineering days, with compute time separate.**

### A1. Test existing feature families against the right target

The current allowlist has **149 features and zero `atr_*` columns**; the store
has **216 ATR columns**. Earlier sectional tests mostly asked whether those
features add information beyond prices. That does not establish that the
current independent predictor already captures their information.

Run controlled family-level additions/removals, preserving identical rows,
training procedure and calibration. First candidates:

1. Historical sectional speed/energy/finishing information already cached.
2. Runner-specific pace fit interacting with draw, field composition and
   conditions, using existing data before building new collectors.
3. Career-depth, missing-history and estimate-uncertainty representations,
   with appropriate pooling for sparse trainer/horse/pedigree histories.

Inspect existing experiment artifacts before repeating any identical proper-
score ablation. Use coverage/availability indicators and retain uncovered
horses in the overall evaluation. Never turn unavailable sectionals into
measured zero performance, and never use a race's own post-race sectionals.

Only investigate rail/stalls text or additional collection after an existing-
data interaction shows promise. The October 17 ATR transport benchmark was
blocked on its first request, so a large unattended backfill is not an
established capability. See [that benchmark](../operations/ATR_BENCHMARK_2026-10-17.md).

**Advance:** a family reduces probability loss consistently across earlier
validation blocks and has acceptable overall coverage. Verify the shortlisted
combination once; do not test all possible family combinations.

### A2. Run a small, bounded training-design comparison

Compare the current binary classifier plus normalization with one conditional
race-softmax model trained directly on winner log loss. Use equal race weight
for the race objective. Include a simple conditional-logit baseline before
building a larger model. These are variable-size race groups, not globally
fixed runner classes.

The previous Plackett–Luce attempt failed; allow this one comparison on the
verified protocol, then stop if it still loses. A ranking objective optimized
for NDCG is not a substitute for a probability objective; LightGBM documents
these as different objectives in its [version 4.6 parameter reference](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html).

The project already uses a 180-day recency half-life. Do not propose adding
recency as though absent. If existing proper-score results do not settle it,
allow one predeclared alternative weight/window comparison, not a large
window-by-decay-by-parameter search.

Market-probability distillation is also already implemented and described as
previously tried. Recover its outcome-based score comparison first. If no
reliable comparison survives the protocol audit, permit one hard-label versus
50/50 outcome/market-teacher test, with identical features, dates and calibration.
The teacher is used only for historical training targets; inference remains
independent of current-race odds. Judge it against actual winners, not by how
closely it imitates prices. An existing valid negative closes this item.

Prefer experiments capable of closing at least 10% of the remaining gap for
a substantial architecture change. Retain smaller stable gains when their
implementation cost is low. No single experiment is assumed to reach parity.

### A3. Add complexity only after measurable lift

Try one partially pooled flat/jumps or exposure-aware variant only if P2
identifies a stable deficit and the pooled model cannot handle it. More
separate models can discard useful sample size. Ensembles require genuinely
different out-of-fold errors and a blend fitted on earlier predictions.

Do not start with a transformer, a large ensemble, another Elo variant or a
large Optuna search. Their expense is not evidence of new information.

## 7. Track B experiments: market first, corrections second

**Estimate: 2–4 engineering days after P0.**

Keep a separate, explicitly price-aware forecast and report these candidates:

| Candidate | What to test | Default when it fails validation |
|---|---|---|
| M0: same-time market probabilities | Reliable price-to-probability baseline, complete observed field, timestamp and feed-state handling | Report missing/stale/unavailable inputs honestly; do not substitute future SP. |
| M1: one-parameter market calibration | Whether a restrained probability-sharpness correction improves proper scores | M0 unchanged. |
| M2: market plus independent model | The existing out-of-fold combination, constrained/shrunk toward market-only | Zero fundamental weight. |
| M3: small market-residual model | Only after materially better independent features or a new information hypothesis survives | Zero residual. |

A useful residual form is

`p(i | race) = q_i * exp(g_i) / sum_j(q_j * exp(g_j))`.

Here `q` is the same-time market probability. Setting every `g` to zero exactly
recovers the market. This guarantees representability of market-only, **not**
non-worsening on unseen races. Choose corrections on earlier validation data;
later results can still be worse. Do not choose fallback weights after seeing
the eventual winner.

Reuse `fit_market_anchor`, `market_probs_from_odds`, and the existing offset
research rather than inventing another blend. Keep current-race price inputs
separate from outcome/settlement columns and the independent allowlist. Saved
artifacts must explicitly apply the observation-time market term at inference.
If binary datasets are cached, reapply `init_score`: LightGBM's
[documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#save_binary)
states that it is not saved in those dataset files.

For historical replay, reconstruct the market definition and runner status as
known at the decision time. Do not silently delete a later withdrawal or apply
its reduction before it was announced. Declare the scoring treatment of later
withdrawals in advance; preserve as-issued forecasts, and label any subsequently
conditioned forecast separately. Retain actual DNF starters as losing runners.

The BASIC feed sends LTP only when it changes, rather than repeating it in
every one-minute message. Reconstruct state by carrying legitimate observations
forward and record both feed activity and last price-change time; an unchanged
price is not automatically a missing update. LTP is not proof of an executable
back/lay quote. [Betfair's explanation](https://support.developer.betfair.com/hc/en-us/articles/25918481395740-How-is-the-BASIC-Historical-Data-last-price-traded-ltp-updated).

**Advance:** M1/M2 improve the same-time reference on development blocks without
material Brier deterioration. If they do not, shipping M0 as a separately named
market forecast is still useful; it is market parity, not independent model skill.

## 8. New data: only after the cheap experiments identify a need

**No large collection or purchase is on the critical path.**

- First use the corrected history, existing sectionals and existing public
  market files. Check the actual age/coverage of each source.
- If a concrete gap remains, run a small, licensed sample whose publication
  timing and expected contribution are specified in advance. Compare its
  incremental proper-score gain with its cost and maintenance burden.
- Depth/spread/volume data is a possible later market-informed hypothesis,
  not a guaranteed solution for a form model. BASIC lacks the richer ladder
  data supplied by higher tiers; establish a budgeted pilot rather than
  committing to a large archive. See the [official data specification](https://support.developer.betfair.com/hc/en-us/articles/360002423271-Where-can-I-view-the-data-specification-for-each-historical-data-package).
- Do not restart sale-price, intent, sibling, weather-only or other failed
  betting hypotheses merely because they sound plausible. Any reopening must
  explain the changed prediction target, information or method and include
  the older result in its report.

## 9. Confirmation and promotion rules

The existing inspected dates remain development evidence. A genuine final test
starts on the first full race day **after the candidate recipe and update rule
are frozen**, with predictions recorded before outcomes. Do not declare dates
fresh merely because this chat has not looked at them.

Proposed practical parity margins, to be fixed before that test:

- Mean race log-loss difference from the same-time market: at most **+0.002**.
- Runner Brier difference: at most **+0.0001**.

These permit roughly 0.1% deterioration at today's reference scale; exact
equality is a different, stricter claim. Declare practical parity only when
the one-sided 95% upper paired confidence bounds are below both margins.
An interval spanning zero is not proof of equivalence. For a stronger
improvement claim, require the race log-loss bound below zero with the Brier
guardrail satisfied. Report all other scores and any tradeoff transparently.

Use paired whole-race resampling and a predeclared meeting/date-block
sensitivity check. Estimate the needed number of races from development
variance, including clustering, before starting confirmation. Plan to cover
at least 12 weeks and relevant race types, but extend if the required sample
is larger. A very small advantage may take much longer to establish.

Changes to the selected method restart its untouched forward test unless
they are updates dictated by the previously frozen training rule. Every
promotion also requires provenance compatibility, save/load replay, all
appropriate regression tests and explicit handling of unavailable prices.

## 10. Order, cost controls and stopping points

| Order | Work | Approximate engineering effort | Decision |
|---|---|---:|---|
| 1 | P0 benchmark/time/field contract and P1 current deployment fitting | 4–6 days | No modelling expansion until replay is sound. |
| 2 | P2 loss attribution and M0/M1/M2 market ladder | 3–5 days | Establish honest same-time market parity and whether the independent forecast adds anything. |
| 3 | A1 existing-data family ablations | 3–5 days | Keep only stable improvements; prioritize the identified deficit. |
| 4 | A2 bounded race-aware comparison; A3 only if justified | 2–4 days | Stop architectural expansion if it fails to improve probability scores. |
| 5 | Freeze challenger(s), record forward forecasts, assess confirmation | 1–2 days setup, then outcome collection | Promote only on the declared evidence. |
| 6 | Small new-data pilot, if the loss analysis justifies it | Separate bounded project | Stop if incremental forecasting value is absent. |

These are engineering estimates, not elapsed-time promises. Plan roughly
three to five working weeks for the first development round; independent
forward confirmation takes additional calendar time.

Research should have a win-only execution path that reuses production fit
logic, avoiding place fits and betting diagnostics in every win-probability
experiment. Finish with the full production pipeline and native checks for a
shortlisted candidate. Use the existing projected feature loading, a frozen
store and cached chronological predictions. Rebuild features only when values
change, rather than for every allowlist or blend-weight experiment. Respect
the 16 GB machine's memory budget when scheduling fits.

Keep a numbered experiment manifest with the hypothesis, comparison, allowed
choices and stop rule written before execution. Do not inspect a final period
after every tweak. If two bounded independent-model rounds show no stable
progress, retain it as a research forecast and use the explicitly labelled
market-informed output for parity. That is a useful product outcome, while
independent market-level performance remains an unproven research goal.

## Immediate first deliverables

1. A frozen forecasting protocol and same-time price/field replay test.
2. A shared win-only benchmark and paired prediction report.
3. A separate deployment `fit_as_of` specification with correct provenance.
4. A market-gap attribution report and at most three family hypotheses.
5. M0/M1/M2 and independent-family experiment manifests, with advancement rules.

No new production changes, experiments, purchases or scheduled collection
were executed as part of writing this plan.
