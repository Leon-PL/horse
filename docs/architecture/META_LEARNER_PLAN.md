# Meta-learner plan

The plan for moving from one win model to a small stack of components. Revised
9 November 2026. This version merges the 7 November proposal, its 8 November
appendix of further sub-model ideas (now in git history only) and the
corrections from the [8 November modelling review](MODELLING_REVIEW_AND_PLAN_2026-11-08.md).
Measured results live in the linked reports and the
[negative-results ledger](../NEGATIVE_RESULTS.md). Check readiness with
`scripts/project_status.py`, not with dates in this document.

## Goal

- **Objective.** Better, reproducible win probabilities, judged by race log
  loss on complete fields. Whether they beat a same-time market, and whether
  any price can be bet profitably, are separate questions with separate
  reports. Market-residual correlation, CLV and returns do not veto a
  forecasting gain.
- **Structure.** A strong generalist plus components that are wrong in
  different places, combined by a small combiner. Market prices enter only in a
  separate, optional final tier.
- **Where diversity comes from,** strongest first: different information,
  different targets, then different model families fitted to the same table.
- **The bar.** A component earns its place only by improving the combined
  forecast over the strongest single-model reference. That reference is now a
  three-seed average of the generalist. A good standalone score, a name or low
  correlation with other components does not count.

## Where it stands (9 November 2026)

**Infrastructure.** `src/modelling/` holds immutable model recipes, the win
model, a performance/spread model, an explicit text model, the fixed-weight
pool, supported feature selection and a candidate artifact
(`win_stack_candidate`, not yet promotable). Commands and guarantees are in
[modelling operations](../operations/MODELLING.md). Experiments live in
`research/meta_learner/`.

**Measured results.** All are retrospective, because these periods have
informed earlier research. Details are in the
[first ensemble benchmark](../research/FIRST_ENSEMBLE_BENCHMARK_2026-11-09.md).

| Comparison | Race log-loss change [95% interval] | Quarters better | Outcome |
|---|---:|---:|---|
| Full 063 generalist against the original 129 inputs, screen, 2024Q3–2025Q2 | −0.0067 [−0.0091, −0.0046] | 4/4 | Advanced |
| The same at native capacity, 2025Q3 pilot | −0.0069 [−0.0105, −0.0031] | Pilot | Locked as the prospective challenger |
| Three-seed average against one full 063 fit, 2025Q3–2026Q2 | −0.0013 [−0.0021, −0.0004] | 4/4 | The reference for components |
| Performance/spread pooled with one fit, against the three-seed average | +0.0001 [−0.0007, +0.0010] | 2/4 | Superseded by the next row |
| **Performance/spread pooled with the three-seed average** | **−0.0008 [−0.0012, −0.0004]** | 4/4 | **Lead: the first component to beat the repeat-fit reference** |
| Explicit Timeform text model pooled with the three-seed average | +0.0001 [−0.0001, +0.0002] | 1/4 | Research only |
| Handicap leeway pooled with the three-seed average (64% of races) | +0.0000 [−0.0001, +0.0002] | 2/4 | Research only |
| Tracking residual specialist (184 unused RTV/ATR inputs), on the average | −0.0005 [−0.0009, −0.0001] | 4/4 | Robust against three shuffled twins (−0.0008, 4/4) |
| Same-day bias residual specialist (new), on the average | −0.0004 [−0.0009, +0.0001] | 4/4 | Robust against three twins (−0.0005, 4/4) |
| Relatives' form (new), pedigree in the store, first string without odds | −0.0001 to +0.0000 | — | Null or absorbed |
| Standalone tracking model (2023+ rows) added to the average and performance/spread pool | −0.0008 [−0.0013, −0.0003], the same as the pool without it | 3/4 | Null: 2–4.5% weight |
| **Stack: average pooled with performance/spread, plus separate tracking and same-day corrections, summed** | **−0.0017 [−0.0024, −0.0010]** | 4/4 | **Retrospective lead; −0.0030 against a single fit** |
| Ability in pounds (new, 14 inputs), all races, added to the stack | −0.0003 [−0.0008, +0.0003] | 3/4 | Against three twins −0.0004, 4/4; the gain is all in jumps races |
| Ability in pounds on jumps races only, added to the stack (segment chosen after the all-races screen) | −0.0007 [−0.0010, −0.0003] | 4/4 | Exploratory; stack plus it −0.0036 against a single fit |
| Relatives' ratings with trainer adjustment (new); going reasons from BHA withdrawals (new) with the store's BHA withdrawal columns | +0.0000 and +0.0002 against twins | 1/4, 2/4 | Null |
| Head-to-head (Bradley–Terry on full 063 inputs) in the pool | 0.0000 (zero pool weight in every quarter) | — | Null; alone 0.0020 worse than one full 063 fit |
| Completion (jumps), improvement and bounce as corrections | +0.0002, +0.0002, −0.0002 against twins | 1/4, 1/4, 2/4 | Null; bounce helps only on jumps, like ability |
| **Error corrections, 10 November: the model's own past errors by horse, trainer and sire (results and past closing prices against its out-of-fold forecasts), four corrections with weights fitted on earlier quarters, added to the stack** ([report](../research/ERROR_ANATOMY_2026-11-10.md)) | **−0.0036 [−0.0055, −0.0019] against the stack; −0.0053 against the average** | 4/4 | **Retrospective lead, the largest group screened; each group beats three twins and a 7-day timing lag changes nothing. Needs a registered read on unseen races** |
| **Registered read on unseen races, 11 November: seven price-taught corrections (own horse, market gap, residual, second wave, franking, same-day, tracking) on a seed-42 fit, 5,761 races from 11 June to 6 November 2026** ([report, section 7](../research/ERROR_ANATOMY_2026-11-10.md)) | **−0.0091, z −4.8, one-sided p = 0.0000007**; own horse and market gap alone −0.0060, z −4.5 | 3/3 periods | **Confirmed against a screen-capacity base. Franking (−0.0013, z −1.6) and same-day bias (−0.0001) did not hold alone. About four fifths carries over to native capacity on 2025Q3** |
| **Corrections on native-capacity full 063, four quarters, 12 November** ([report, section 10](../research/ERROR_ANATOMY_2026-11-10.md)) | **−0.0102 [−0.0124, −0.0083] with one joint correction model; −0.0093 with the weighted stack. Native 1.81937 to 1.80915** | 4/4 | **Nine tenths of the screen-capacity gain carries over for every group: information, not capacity repair. Development quarters; expect about −0.008 unseen. Needs stored out-of-fold forecast history to serve** |
| **One joint correction with franking, hidden form and past late moves added (416 inputs), on native full 063, 12 November** ([report, section 10](../research/ERROR_ANATOMY_2026-11-10.md)) | **−0.0126 [−0.0151, −0.0103]; native 1.81937 to 1.80676**; −0.0024 [−0.0033, −0.0015] over the eight-group joint model | 4/4 | **With the deeper recipe as well (31 leaves, 1,000 rounds): −0.0143 [−0.0171, −0.0119], native 1.80506, the best development forecast. Own inputs as context add a marginal −0.0007. Development quarters only; the deeper combination was added after the first variants were read** |
| Second registered read, 11 November: run-by-run horse state and 30/90-day stable form added to those seven, same races ([report, section 8](../research/ERROR_ANATOMY_2026-11-10.md)) | −0.0018, z −1.7, p = 0.04 against a 0.005 mark (development: −0.0036 on the stack, −0.0047 in rehearsal) | 3/3 periods | **Not confirmed as an addition.** Horse state alone is −0.0065 (z −5.1) against the base, better than the own-horse group it would replace (−0.0042), and it carried over to native capacity on 2025Q3 (−0.0047) |
| Ability in pounds and same-day bias as generalist inputs (212 inputs, three seeds) against the three-seed average, 10 November | −0.0005 [−0.0016, +0.0006] | 3/4 | Fails the gate; no better than the same inputs as corrections (−0.0007). They stay corrections |

Full 063 is the 129-input recipe plus 64 inputs (rating gaps, signal gaps,
context-weighted form and race context), trained with a 75% market share in its
target. Against a single full 063 fit, performance/spread improved log loss by
−0.0011, about as much as averaging three seeds. The benchmark pooled it only
with one fit. Pooled with the three-seed average itself, at about 10% weight,
it improves the average by a further −0.0008 in every quarter, so the two gains
overlap only partly. The [cheap screens](../research/CHEAP_SCREENS_2026-11-09.md)
found this; the pool uses the benchmark's combiner and cached forecasts.

Earlier evidence points the same way. Experiment 063 (ledger entry 063) found
that seven subject models stacked on the full model changed log loss by +0.0002
[−0.0005, +0.0008], and by +0.0183 when they replaced it. A boosted
meta-learner with race context gave +0.0002, and a run-by-run form model in the
stack gave −0.00001. Sub-model targets for completion, position, late gain,
finishing speed, placing, time figure and handicap mark added nothing.

**What this means for the plan:**

- Most measured gains come from the generalist's inputs and target, and from
  averaging repeat fits. Performance/spread is the one stacked component that
  also improves the repeat-fit average.
- A different target on the same inputs can add something beyond seed
  averaging (performance/spread), but not reliably (handicap leeway).
- Specialists that split families the generalist already uses are measured as
  null. Do not rebuild them unchanged.
- The remaining room is information the generalist does not have, model forms
  it cannot express (a learned run-by-run sequence model is untested) and the
  market tier.

**Current step.** Prospectively confirm one full 063 fit (seed 42, legacy text
inputs) against the original 129-input control. The protocol is four calendar
quarters, a common purged refit before each, and forecasts five minutes before
the first published scheduled start. It begins once decision-time capture and
canonical input and prediction parity are ready. The locked recipe and protocol
are in
`data/research/meta_learner/first_ensemble_20261109/native_quick_2025Q3/prospective_candidate.json`.

The stack (three-seed average, pooled with performance/spread, plus separate
tracking and same-day corrections, summed) is a second candidate for the same
protocol.
Before it is locked it needs native-capacity seeds, its own recipe file and a
check that the tracking and same-day inputs are available at the forecast time
in serving, not only in the store. A jumps-only ability-in-pounds correction is
a possible third correction, found after the fact on the same quarters.

## Rules every component follows

1. **Time boundaries.** Every learned stage excludes the scored period: model
   parameters, feature selection, text fitting, calibration, spread residuals,
   pool weights and any market correction. Folds are whole calendar quarters
   from `src.wf_eval.expanding_period_folds`, passed as explicit `Fold` objects
   to `oof_predict_expanding`, with a seven-day purge.
2. **Representation.** Each component supplies a whole-field probability
   distribution. The combiner works on race-centred log probabilities, not
   binary log-odds. A pool containing only the generalist is exactly the
   generalist.
3. **Coverage.** A component that cannot score every runner in a race is masked
   for that race, and the combiner falls back to the remaining components.
   Missing measurements may stay NaN inside a model trained to handle them.
4. **Combiner.** Non-negative log-pooling weights summing to one, shrunk
   towards the generalist and fitted only on earlier out-of-fold quarters.
   Context-dependent weights come only after fixed weights help, with a small
   predeclared set of interactions.
5. **Gates.** Fixed before scoring: paired race-loss differences with
   date-clustered intervals, stability across quarters, calibration, coverage,
   seed variation and fitting cost. The comparison is against the repeat-fit
   reference, not a single fit. Candidates pass through the tiers in
   [How candidates are evaluated](#how-candidates-are-evaluated) before any
   refit.
6. **Prior evidence.** Read the negative-results ledger before building. A
   materially changed input or objective can justify a retest. A new ensemble
   does not reset earlier results.
7. **Hold-out.** Races from 11 June 2026 stay out of development fitting, but
   that period has been examined, so it is not an untouched test. Do not move
   the date to recycle examined data. Fresh confirmation is prospective, with a
   locked recipe and decision-time inputs.
8. **Timing.** The research contract forecasts five minutes before the first
   published scheduled start. The betting horizon is still a product decision.
   BSP cannot enter a forecast made before the off, and a last traded price is
   not an executable quote.
9. **Serving.** A stack is saved as one artifact with its ordered recipes,
   weights, calibrators, cutoffs, coverage and fallback rules, and data
   identities. It becomes promotable only after the live and place adapters are
   qualified. The place head uses win probabilities, so its output must be
   re-verified.
10. **Frameworks.** LightGBM for boosted components; scipy and scikit-learn for
    linear text models and combiners. A neural component needs an explicit
    decision to add a framework.

## How candidates are evaluated

Refitting the generalist for every idea is slow, and it is also noisy. Two
full 063 fits that differ only in seed differ by up to ±0.0015 on the four
later quarters (single quarters swing by 0.0023). A single-seed refit cannot
resolve the 0.0003–0.001 effects most candidates are likely to have. Candidates
therefore pass through tiers, and only survivors earn a refit. Method,
validation and costs are in the
[cheap screens report](../research/CHEAP_SCREENS_2026-11-09.md); the code is
`research/meta_learner/screening.py` and `screen_candidates.py`.

| Tier | What it does | Cost | Use |
|---|---|---|---|
| 0 | Conditional logit on the stored generalist forecasts, a few step coefficients per input | Seconds | Forecast-level components and small input groups |
| 1 | Three heavily regularised LightGBM corrections from the stored forecasts, against a twin with the inputs moved to random rows of the same quarter | About 1.5 minutes | Every candidate; the twin removes what the fit gains from capacity alone |
| Pool | The benchmark's log-pool with the three-seed average | Seconds | Components that produce their own forecasts |
| 2 | Generalist refit with the candidate's inputs, repeat seeds in both arms | About an hour at screen capacity | Only for groups whose screen gain is large, or whose value lies in restructuring existing inputs |
| 3 | Native capacity, then prospective confirmation | Hours to months | The final recipe |

Lessons fixed by the validation:

- **A correction must not be able to recalibrate.** The best temperature for
  full 063 moves between 0.956 and 1.057 from quarter to quarter. Screens that
  also saw the reference forecast or race context learned the training
  quarters' calibration and made later quarters worse. Screens see only the
  candidate's inputs.
- **Judge against a twin, not against nothing.** Any fit has a capacity cost;
  the twin measures it with the same inputs and coverage.
- **Screens measure new information, not restructuring.** They recover about a
  tenth of the 063 refit gain, because those 64 inputs re-express what the
  original 129 already held. Candidates of that kind go straight to Tier 2.
- **Verdict rule, fixed before scoring:** advance on a Tier 0 interval wholly
  below zero on the three-seed average, or a Tier 1 gain over the twin of at
  least 0.0003 in three or more quarters; reject when both lower bounds are
  above −0.0003. Survivors are re-checked against three twins.

## Which features each sub-model gets

Decided 9 November 2026. Diversity from a different target and diversity from
different information call for opposite feature rules.

| Sub-model type | Features | Target | Reason |
|---|---|---|---|
| Generalist (S0) | The generalist list | Win (race softmax) | Strongest single input. It keeps interactions between domains, which the combiner cannot rebuild |
| Different-target models (S9, S10, S12–S15) | The generalist list or a broad list | Each its own | Their diversity comes from the target. Restricting their features as well would weaken them twice |
| Information specialists (S2b, S3, S4, S6, S7, S8) | Their domain plus the shared context set | Win | They exist to make an underused domain heard |
| Splits of families the generalist already uses (S1 ability, S2a clock, S5 conditions) | None | None | Not built. Subject models of this kind were null in 063 |

**Why not one feature list for every sub-model, with only the target
different.** Gradient-boosted trees choose splits greedily, so every model leans
on the same strong rating and form features, and their errors stay correlated.
A different target mostly changes the weights on those features, not which
features are used. Weak but independent signals rarely win a split while the
strong features are available. Sources that start in 2023 look weaker than they
are in a model fitted on every year. The 9 November results are mixed:
performance/spread on the generalist's inputs added about what a second seed
adds to one fit, and a further −0.0008 on top of the three-seed average.

**Why restrict the information specialists:**

- A restricted model has to express what its domain contains.
- It can be fitted only on rows where its source exists.
- The combiner can mask it when the source is missing or broken, as ATR was for
  July–October 2026, without disturbing the other components.
- Its incremental value to the combiner can be measured on its own.

**Shared context set.** Race type, distance, going, surface, field size, class,
handicap flag and age. Without these, a specialist cannot tell, for example,
that a finishing-speed figure means something different at 5 furlongs than at
3 miles.

**Domain membership.** Take it from the feature catalog written by
`python -m src.modelling prepare` and its recorded dependencies, not from column
prefixes alone. For example, `horse_prior_market_*` is market-derived, and RTV
columns include both measured tracking and analyst opinion.

**Residual variant.** A specialist can start from the generalist's out-of-fold
race scores (LightGBM `init_score`), so it learns only the correction its domain
adds. Compare the standalone and residual versions by their incremental value
to the combiner. The starting scores must be out-of-fold, never in-sample.
Measured 9 November for tracking: the residual form improved the stack and
the standalone form added nothing. Separate residual corrections, summed,
also beat one joint correction.

**Training labels.** Whether a specialist learns from results only or from the
generalist's blend of results and market prices is a paired comparison on
earlier development data, not a fixed rule. Results-only labels may add
diversity, but weaker predictions may cancel it.

**Caveat.** Tracking and text already appear among the generalist's inputs, so
an information specialist is a candidate, not automatically new information. It
must earn its place through its incremental value to the combiner.

## Candidate components

IDs follow the original proposal. S12–S15, C1–C2 and M3 come from the
8 November appendix.

### Generalist and its inputs

| ID | Component | Status |
|---|---|---|
| S0 | Full 063 generalist and its repeat-fit average | The reference. Prospective confirmation locked |

Gains so far have come from the generalist, so new information can also enter
as generalist inputs, through supported feature selection rather than as
components:

- **Ability in pounds.** A running estimate of each horse's ability measured in
  pounds, allowed to change faster for improvers than for exposed horses.
  Screened 9 November (`research/meta_learner/ability.py`): information beyond
  three twins in all four quarters, but the gain is all in jumps races. A
  jumps-only correction on the stack gave −0.0007, 4/4, with the segment chosen
  after the fact.
- **Relatives' form for unexposed horses.** The career record of horses sharing
  the dam, keyed by dam plus dam-sire, measured in rating points rather than
  win rates (win rates depend on the class the offspring ran in), weighted by
  how many offspring it rests on and adjusted for trainer. It is not in the
  store. Expect a small effect: published estimates put the inherited share of
  racing ability at roughly 0.1–0.35, and knowing both parents explains only
  about half of that. Sale price, the obvious public proxy, tested as fully
  priced by the market, and debutant fixes in 055 were null. Screened
  9 November in rating points with trainer adjustment: null against its twin
  (+0.00002), as were the simple rates. Dropped.
- **Ground preference from non-runner reasons.** BHA reports record why horses
  were withdrawn, for example "Self Cert (Going)", which shows what ground the
  connections think suits a horse. Screened 9 November: rejected with the
  store's BHA withdrawal columns (+0.0002 against its twin); the new going
  comparisons cover 3–5% of runners and are null at Tier 0. Dropped.

### Different targets (generalist inputs)

| ID | Component | Evidence and status |
|---|---|---|
| S9 | Performance spread: performance in pounds as a mean and a spread, integrated into win chances | Pooled with the three-seed average: −0.0008, 4/4 (9 November, retrospective). In the stack |
| S12 | Handicap leeway: performance left after the official mark at the weights | Tested 9 November: null. A handicap-mark target was also null in 063. Stop unless the inputs change |
| S10 | Head-to-head: pairwise model with a Bradley–Terry ranking | Tested 9 November: zero pool weight in every quarter; alone 0.0020 worse than one full 063 fit. A pairwise learner also only tied 057 in 062. Stop |
| S13 | Completion risk (jumps): chance of completing × chance of winning if it completes | A completion target added nothing in 063. 9 November: a dedicated completion model (AUC 0.69–0.72) adds nothing as a correction (+0.0002 against its twin). Stop |
| S14 | Improvement trajectory: how far a horse will improve, from age, runs, gaps and late-maturing sire lines | 9 November: sire, trainer and horse residuals against the ability filter, null (+0.0002 against its twin). Stop |
| S15 | Bounce and recovery: how a horse runs after a career-best or a hard race, from sectional effort and days since | 9 November: last-run residual, career best, close finish and quick return, −0.0002 against its twin (2/4), all of it on jumps. Kept only inside the jumps hypothesis with ability in pounds. Sectional effort untested |

### Information specialists (domain plus context)

| ID | Component | Data | Evidence and status |
|---|---|---|---|
| S2b | Tracking and sectionals | RTV RaceiQ from 2023-06 (about 90% coverage); ATR sectionals from 2023-08 (partial) | Partly among the generalist's inputs. The 194 unused RTV/ATR inputs as a residual specialist: −0.0005 on the average, −0.0007 against three twins, 4/4 (cheap screens). In the stack as its own correction. A standalone tracking model fitted on 2023+ rows scored 1.9705 alone (average 1.8247) and added nothing in the pool (2–4.5% weight) |
| S3 | Trip and luck, looking back (trouble in past runs) and ahead (chance of trouble today from style, draw, field size and track) | ATR comments, RTV speed lost and jumping, BHA stewards' reports | Luck weights (059) and trouble inputs were null or weak. Low priority |
| S4 | Pedigree for unexposed horses | Sire, dam and dam-sire records; relatives' form once built | Screened 9 November: unused store pedigree rates carry a little information but no net gain (−0.0001); relatives' form is null both as simple rates and in rating points with trainer adjustment. Stop |
| S6 | Pace, draw and same-day bias | Running style, field pace pressure, draw; draw, pace and going bias from earlier races on the same card | The existing pace family (ledger 28) and race-profile projection were null. Same-day bias (draw and running-style bias from races off at least 30 minutes earlier on the card): −0.0004 on the average, −0.0005 against three twins, 4/4. In the stack while the five-minute horizon stays |
| S7 | Readiness and stable intent | BHA post-race vet findings; retained jockey and first colours when a stable has several runners (`trainer_first_string_*`, unused) | Stewards' reports measured as priced, and jockey booking was dead. `trainer_first_string_*` screened null once its odds-ranked columns (market-derived) are removed. Low priority |
| S8 | Text beyond Timeform | Sporting Life commentary and verdicts, ATR comments, stewards' reports | The explicit Timeform model was null while its legacy inputs stay in the generalist. Each new source needs its own availability rule |

### Combiner inputs (only after fixed weights help)

| ID | Input | Evidence and status |
|---|---|---|
| C1 | Disagreement between components, to shrink weights where they disagree | Needs at least two components with real weight |
| C2 | Race predictability: how open the race is, to sharpen or flatten the forecast | Temperature by race kind made forecasts worse in 063 (+0.0010) |

### Market tier (after serving qualification, at the declared horizon)

| ID | Component | Data | Evidence and status |
|---|---|---|---|
| M1 | Prices at the decision time | Last traded prices from 11:00 to 60 seconds before the off, from 2020 | Compare against a calibrated market-only control at the same time. The two-stage market combination failed as a betting rule (ledger 2); this tier is judged as a forecast |
| M2 | Market flow | Betfair BASIC price paths; public BSP volume and average traded price; no order book | Price moves up to the decision time carry information the generalist lacks. Later moves were not predictable from them |
| M3 | Market bias: where the market is systematically wrong, such as overrating long shots, by field size, race type, meeting, time and liquidity | Prices and race context only | Untested |

### Later, needing a framework decision or new data

- **S11 sequence and graph networks.** See the
  [design notes](#sequence-and-graph-networks-design-notes-s11). Experiment
  063's run-by-run test does not cover a model that learns which past runs
  matter.
- A pace simulator, Betfair PRO order-book data and video analysis of trips.

## Proposed order after the confirmation starts

Updated 9 November after the cheap screens. Step 2 is the only one that
refits the generalist.

Done: **separate specialists.** Two corrections summed beat one joint
correction by −0.00027 (4/4 quarters); fitted weights add nothing over unit
weights; a standalone tracking model adds nothing in the pool. New specialists
are screened as their own correction and summed onto the stack.

Done: **new information.** Ability in pounds holds against three twins
(−0.0004, 4/4) but adds only on jumps races; relatives' ratings and going
reasons are null. A jumps-only ability correction (−0.0007 on the stack, 4/4,
segment chosen after the fact) is a candidate third correction. It needs a
locked recipe and prospective confirmation before it joins the stack.

Done: **forecast-level components.** Head-to-head got zero pool weight;
completion, improvement and bounce corrections were null against their twins.
Recipes and gates were registered before scoring.

1. **The jumps margin specialist test, registered 9 November**
   ([registration](../research/JUMPS_MARGIN_SPECIALIST_REGISTRATION_2026-11-09.md)):
   a jumps-only ability-in-pounds correction on the frozen three-seed average,
   scored once on the first 3,300 jumps races from 11 June 2026 (about 1,367 so
   far; the rest expected by early March 2027), pass at one-sided p < 0.01.
   Bounce inputs were left out by a rule fixed before comparing them. Before
   scoring it needs a commit, base inputs for those races, the frozen base and
   correction fits, and a one-shot scoring script.
2. **Tier 2 for same-day and ability inputs inside the generalist: done
   10 November, fails its gate**
   ([report](../research/TIER2_INPUTS_2026-11-10.md)). Three seeds of full 063
   plus the 19 inputs against the cached three seeds: −0.0005
   [−0.0016, +0.0006], 3/4. The same inputs as corrections on top give −0.0007,
   so they stay corrections. The 184 tracking inputs were not refitted (memory);
   nothing suggests a different answer.
   **Error corrections, found 10 November**
   ([report](../research/ERROR_ANATOMY_2026-11-10.md)): the generalist's
   errors repeat by horse, trainer and sire (trainer errors correlate 0.19
   between two halves, against 0.09 for BSP). Inputs that record each
   entity's past results and closing prices against the model's own
   out-of-fold forecast give −0.0036 on the stack. Next: base forecasts for
   the races from 11 June 2026, frozen corrections and weights, one registered
   read. Serving needs a stored history of out-of-fold forecasts.
3. **The sequence network,** if the framework decision is made. It is the one
   model form not yet tested.
4. **The market tier,** after serving qualification.

Each step stops if it does not beat the current stack under its fixed gate.

## How the time periods nest

- Development data ends on 3 June 2026; races from 11 June are excluded.
- Component forecasts for quarter *q* come only from fits on earlier data, with
  a seven-day purge.
- The pool for quarter *q* is fitted only on out-of-fold forecasts from earlier
  quarters. In the first benchmark, 2025Q1 and 2025Q2 initialised it, and it
  expanded from there.
- Supervised feature selection must sit inside the outer training boundary, or
  use a list locked before the scored period.
- For serving, components are refitted through a settled cutoff, and the pool
  is fitted on all out-of-fold forecasts.

## Sequence and graph networks: design notes (S11)

**Two networks, not one.** The combiner can weight them separately: the graph
network should help most in handicaps between exposed horses with dense form
lines, and the sequence network for horses with varied runs to compare. Each
one's value can be measured on its own. They also train differently: the
sequence network can take races in any order, while the graph network must
process every race in date order. Leakage is easier to rule out one network at
a time. They share code for turning a run into a vector and for the race
scorer. A joint version, where each past-run token also carries its rivals'
current summaries, is a later candidate that must beat the pair.

**Sequence network.** One token per past run, up to about the last 20:

- timing: days before today, days since the previous run, age, month
- race conditions: course, race type, surface, distance, going, class,
  handicap flag, prize, field size, region
- the horse's situation: weight, official rating, draw, headgear with a
  first-time flag, wind operation, jockey, trainer
- result: relative finishing position, margin in pounds, how a non-finisher
  stopped, speed figure
- run shape from 2023: RTV and ATR tracking and sectional measures, with a
  missing bit per source
- a frozen sentence embedding of the in-running comment.

A query token for today's race (conditions, weight, mark, draw, connections,
pedigree) attends over the past runs, so the network learns which runs matter
for today. Debutants get a learned "no history" token. Past prices and pre-race
Timeform opinions are left out.

**Graph network.** Horses, races, jockeys, trainers and sires are nodes;
"ran in" edges carry the result. Working through races in date order, each
horse's summary is updated after each race from the result and its rivals'
summaries, which is a learned version of TrueSkill. At prediction time it also
gathers the current summaries of rivals from its past races, which captures
whether those races have worked out.

**Both** end in self-attention across today's runners and a race softmax,
trained on finishing order with an auxiliary performance-in-pounds target.

**Leakage traps:** today's run and later results must be excluded; past-run
fields must be as they were then (the `sex` look-ahead leak); text is used by
`available_date` and `phase`; career counters must be checked as-of race time;
lookup tables and scaling are rebuilt in each fold; horses join on `horse_id`;
non-finishers stay in.

**Cheap proxy already tried.** Past form weighted towards runs like today's, a
hand-built version of the attention query, was null alone in 063 and is part of
full 063.

## Project rules that apply

- Work starts in `research/`. Nothing in `src/`, `scripts/` or `tests/` may
  import it.
- Use the walk-forward primitives in `src/wf_eval.py`. Do not hand-roll fold
  loops.
- Any module moved into `src/` that can change a feature value must be
  registered in `_FE_SOURCE_FILES` (`src/featured_dataset.py`).
- For betting claims, read CLV with `src.clv.summarise_clv`, never the mean of
  `taken_odds / bsp`.
