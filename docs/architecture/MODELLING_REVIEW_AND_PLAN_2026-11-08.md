# Modelling review and proposed migration

8 November 2026. Review and proposal only. This review does not rebuild
features, train candidates on project history, or change the selected model.

## Recommendation

Start a new, small modelling layer alongside the existing predictor. Retain the
consolidated source publications, feature engine/store, identity rules,
race objectives, probability mathematics, settlement and useful evaluation
primitives. Keep the existing predictor as a benchmark and compatibility adapter.
Do not rewrite the project or implement all eleven proposed specialists at once.

The immediate objective should be **better, reproducible win probabilities**.
Whether those probabilities improve on a contemporaneous market, and whether a
price can actually be traded profitably, are separate questions. Feature and
ensemble selection should optimise race log loss; market-residual correlation,
CLV and returns should not veto an otherwise useful forecasting improvement.

Build the smallest ensemble that demonstrates complementary information. Start
with the generalist, the existing text model and a reproduction of experiment
058's performance/spread model. Screen additional information specialists only
after this comparison works. Promote a component for its incremental value to
the ensemble, rather than its name, standalone score, or low correlation alone.

## What exists today

### The actual pipeline

```text
Accepted source dataset
    -> process_data
    -> shared feature store
    -> configured feature projection
    -> training.train_run / RacePredictor
    -> evaluation + saved candidate in data/runs/<id>
    -> explicit models/production.json selection
    -> live features + selected predictor
```

`train.py` delegates to `src/training.py`. The default evaluated fit reserves the
latest 90 days, purges the training boundary, fits the heads and calibrators, and
saves a candidate. `fit_as_of` is a separate deployment refit with no claimed
test score; it can reference a validated recipe. Saving and promoting are
separate operations. These distinctions should survive the migration.

There is one win estimator **and a separate place estimator**, rather than a
stack of win models. The win estimator is LightGBM with the exploded race-softmax
objective: winner, second and third stages weighted 1, 0.25 and 0.0625. The
first-stage target blends results with market-implied probabilities 50:50 where
a complete teacher field exists (BSP preferred, then SP). Race-temperature
calibration converts scores to a probability distribution over the field.
The optional same-race market anchor is disabled.

Thus the forecast is free of current-race prices **as inference inputs**, but
is not market-independent in its training. Prior-race market features and public
Timeform opinions are also inputs. The plan should name these distinctions.

The selected run is `20261104_235300_444862`, fitted through 3 November with
129 features. Its deployment metadata links evaluation run
`20261104_212157_439212`. The name `triple_ensemble_models.joblib` is historical;
`RacePredictor` explicitly documents that it does not implement stacking.

### Readiness and evidence limits

Read-only metadata inspection confirms:

- Store built on 6 November, with 757,580 runner rows, 1,231 columns and a
  2,870,667,563-byte feature Parquet, through 4 November.
- Stored FE signature `457c83789e6b29ab`; no dataset manifest is recorded in
  that store metadata. The current refresh code explicitly rebuilds stores
  that predate source publication. This is a required migration step, not a
  reason to redesign ingestion.
- The full accessible status check at 12:47 on 8 November returned
  `servable: false`. It found one recent pending race, partial source readiness,
  changed results and ATR/RTV observations beneath the store, and current FE
  signature `d6c80a5994f8af79`, which differs from the stored signature.
- The selected model records FE generation `bc357c18a5e5a261`, different from
  both the store and current code. Rebuild/refit is necessary before comparing
  candidates or serving compatible forecasts.
- Normal-access inspection initially could not read store metadata. A
  read-only elevated inspection and full status check succeeded; their results
  above establish the mismatch independently of that permission limitation.

The working tree already contains extensive feature and workbench consolidation
changes. Review and migration must preserve them. This assessment inspects the
current files, rather than assuming committed HEAD represents the installation.

### Research separation: improved, but modelling is still entangled

The top-level separation is real: `research/` may import pipeline code, while
pipeline code cannot import research or command wrappers. `tests/test_layout.py`
enforces this. The earlier experiments are recoverable from git at `26d1b6f`;
reports and local evidence were retained.

The remaining modelling problems are narrower:

1. `src/model.py` combines feature discovery, preparation, targets, training,
   tuning, calibration, evaluation, betting analysis, explanation and persistence
   in roughly 4,000 lines. Several paths reach private training methods.
2. `src/feature_selection.py` still contains the 048 experiment machinery, but
   its documented `scripts/select_features.py` command was retired. Its fitting
   helper explicitly forces the older cross-entropy objective; it does not
   reproduce today's exploded-softmax learner.
3. Specialist settings would currently require overriding shared `config`
   values, especially the feature list and soft-label switches. That is an
   unsafe basis for several different recipes in one process.
4. A supervised text learner is fitted by the source materialisation path.
   Its local model files have a different lifecycle from saved racing models.
5. `feature_sets/registry.json` supplies workbench descriptions and model roles;
   training still reads `config.FEATURE_ALLOWLIST`. It is not an executable
   model registry or training specification.

These justify extracting the modelling orchestration, not replacing the tested
feature calculations or indiscriminately relocating old files.

## Where the 129 features came from

The active default is **`feature_sets/win_selected_056.txt`**, not `win.txt`.

| Selection | Origin | Implication |
|---|---|---|
| `win.txt` | Original top-150 pilot gain ranking, followed by manual edits | Historical baseline; not the current generalist recipe |
| `win_selected_048.txt` | 126 features from purged walk-forward recursive elimination | A measured reduction with retained local evidence |
| `win_selected_056.txt` | Those 126 plus three Timeform text inputs | Today's 129-feature default |

048 screened the eligible numeric universe, removed near-empty/constant/exact
duplicate columns, ranked features by held-out absolute SHAP across six
selection quarters, refitted progressively smaller subsets, and used a paired
one-standard-error rule. Four later quarters confirmed the choice.

The historical report found that 126 features performed about as well as all
1,127 eligible features, at much lower fitting cost. It reported a 0.0194
improvement over the old hand-edited list at native capacity. This is evidence
for the process on that generation, not a fresh measurement on today's repaired
data and objective. The selection and confirmation periods have since informed
many experiments.

There is no automatic selection stage in ordinary training today. It consumes
the saved list. `FEATURE_PRUNE_FRACTION = 0` also disables the optional pilot
pruning path; the configured correlation threshold does not independently
reselect features in that default path.

The large store does not mean every fit loads every column:
`featured_dataset._model_load_columns` already projects the active features plus
required labels/context, and `FeatureStore.load_featured` accepts columns.
Keep that capability and make it explicit for multiple models. A trainer should
read the union of its requested inputs, or load each model's projection, without
changing a global allowlist.

### Proposed supported selection process

- Keep the broad store as a shared source of features. Publish small, ordered,
  versioned model input lists; do not delete useful store columns to save fit
  memory.
- Give features explicit model-facing metadata: family, dependencies,
  source/availability rule, eligible roles, and whether they are raw,
  history-derived, or learned predictions. Prefixes are a navigation aid, not
  an information barrier: for example `horse_prior_market_*` is market-derived,
  and RTV includes both measured tracking and analyst opinions.
- Provide a supported selection operation using the same learner adapter,
  target, weighting, early stopping and calibration contract as the intended
  model. Separate reusable selection code from experiment registrations.
- Run hygiene and coarse family ablations first; apply recursive selection
  where fitting cost warrants it. Use sparse-period coverage and regime slices
  so newer tracking features are not rejected solely for missing early years.
- Select within earlier training/development data, confirm later, and save the
  ordered inputs, rejected reasons, data/recipe identity and per-period scores.
  Record seeds because column subsampling can move close comparisons.
- Reuse the resulting list for routine refits. Feature selection is a supported
  deliberate operation, not an expensive hidden side effect of every fit.
- For a claim of historically honest out-of-fold performance, supervised
  selection must occur inside the relevant outer training boundary, or use a
  list locked before the scored period. A list selected using 2024–2026 labels
  does not make predictions from 2022 independently selected merely because
  the estimator itself was trained before 2022.

## What the text model does

`src/rtv_text.py` fits a penalised linear conditional-logit model using TF-IDF
word and word-pair features from each runner's Timeform comment and the verdict
sentences naming that runner, plus verdict mention order. It is not an LLM or
an embedding model.

Each calendar quarter has its own model, trained on earlier results with the
purge gap. Inspection found 24 `.npz` artifacts in
`data/rtv_cache/text_models`, from 2021Q1 through 2026Q4. Their saved training
cutoffs respect the quarterly schedule; 2026Q3 ends on 24 June and 2026Q4 on
24 September.

The present flow is:

```text
RTV comments/verdicts + canonical result labels
    -> ingestion.rtv_observations.ensure_text_models
    -> quarterly vocabulary, IDF and fitted weights
    -> history scores / pre-race snapshot scores
    -> rtv_text.add_text_features in feature engineering
    -> rtv_text_logit, rtv_text_prob, rtv_verdict_rank
    -> inputs to the general win and place models
```

`rtv_text_prob` normalises across scored runners, so incomplete coverage is not
a complete-field standalone forecast. History scores are now read through the
selected source dataset; pre-race snapshot values take precedence. The fitting
and snapshot scorer still access the local quarterly model directory.

The original 052 development experiment found a small gain. Installation 056
found essentially no measurable gain on its later 90-day comparison. Both
results should remain visible; neither supports promising a large text edge.
Historical pre-race availability rests partly on retrospective backfill and
limited snapshot comparisons, not a timestamped historical capture for every
race. Pre-race previews and post-race reports require different timing rules.

**Proposed change:** collect and publish text as observations, but own fitting,
versioning and scoring of supervised text models in the modelling layer. A
missing quarterly artifact should not silently trigger training during forecast
generation. Retain immutable historical score tables for reproducibility while
migrating; do not invalidate all old feature definitions merely to move files.

Treat text as a first-class ensemble component with its vocabulary, IDF,
weights, training cutoff, input text version and score provenance. Initially
compare direct text stacking against keeping text as generalist features. The
same information may appear in both; that is redundant rather than inherently
invalid, and must earn its place through ablation. Broader SL/ATR/BHA text and
embeddings are later experiments, each with its own availability contract.

## Changes needed to META_LEARNER_PLAN.md

### Keep

Keep the strong generalist baseline, information-based specialists, a small
conditional-logit combiner, chronological out-of-fold predictions, a separate
optional market tier, and incremental ablation. These are sound foundations.

### Correct before implementation

1. **The golden holdout is already examined.** Config reserves races from
   11 June 2026, but the 056 and 057 reports compare August–October/November
   held-out scores, 055 reviews worst holdout misses, and the benchmark protocol
   explicitly calls 2026Q3 retrospective. The 057 evaluation metadata trains
   through 30 July and scores from 6 August. Therefore the proposed final
   one-time untouched test cannot be claimed on that whole period. Retain the
   existing restriction while auditing exposure; do not simply move the date
   in config and call an examined period fresh. Lock a new prospective
   confirmation protocol before collecting its scored outcomes.
2. **The OOF API does not implement the promised folds.**
   `oof_predict_expanding` creates equal-sized row blocks; it does not accept
   the calendar folds returned by `expanding_period_folds`. It can split a race
   or day across prediction blocks. Extend the shared primitive to consume
   explicit complete-race/date folds and validate them; keep one fold owner.
3. **All learned stages need the time boundary.** Model parameters, feature
   selection, text fitting, calibration, spread residual targets, blend weights
   and optional market corrections must each exclude the outer scored period.
   For a fixed historical test, later text quarters cannot learn earlier test
   outcomes. For rolling updates, that can be legitimate only under a locked
   prequential schedule. Declare which experiment is being run.
4. **Use centred log probabilities or native race scores.** If a component
   supplies probabilities, centre `log(p)` within the race. Binary log-odds
   `log(p/(1-p))` changes the representation and does not reconstruct S0 under
   a within-race softmax. With only S0, force identity: weight one and no extra
   learned temperature. A freely fitted single coefficient is recalibration,
   so exact equality is not a valid expectation for that fitted model.
5. **Separate coverage from failure.** A valid component distribution must
   cover the whole active field. Record runner support and race coverage.
   Missing source measurements can remain NaN in a model trained to handle
   them; an unavailable component must be masked explicitly. Define complete
   generalist fallback and test partial-field cases. Do not rely on the meta
   learner eventually learning a zero weight from an availability flag.
6. **Make the initial combiner small and identified.** Begin with nonnegative,
   regularised log-pooling weights shrunk towards the generalist, normalised
   over available components. Include identity and simple arithmetic averaging
   controls. A shared race intercept cancels under softmax. Pure race context
   must interact with component scores to change rankings. Add a small,
   predeclared set of context interactions only after fixed weights help.
   Temperature and unrestricted weight scale should not both be free.
7. **Do not force every specialist to use a hard-label exploded objective.**
   Preserve the current generalist recipe. Compare hard vs distilled targets
   in a small paired experiment, chosen on earlier development data. Different
   labels can improve diversity, but weaker component predictions may erase
   that benefit. Text and performance/spread models necessarily have different
   targets and fitting methods.
8. **Promote evidence-backed diversity earlier.** Experiment 058 reported
   about -0.00172 race log loss when blended with the exploded model, on three
   later quarters. It is a stronger starting hypothesis than many overlapping
   subsets, although it needs reproduction on the repaired data and was partly
   explored after the original registration. Its mean/spread construction also
   has missing-margin and non-finisher assumptions to revisit explicitly.
9. **Treat specialists as candidates.** Tracking and text already appear in
   S0's 129 features, so Phase 2 is not automatically new information. The
   negative-results ledger already covers several luck, connections, pace,
   card and market ideas. A materially changed input or objective can justify a
   retest; the existence of a new ensemble does not reset prior evidence.
10. **Revise the gates.** Better in every one of four reused quarters is a
    useful diagnostic but not sufficient independent confirmation. Compare
    paired race-loss differences, date-clustered intervals, period stability,
    calibration, coverage, seeds and operational cost. Fix the acceptance
    criteria before scoring. A specialist can be weaker alone and useful in
    combination; low prediction correlation alone proves little.
11. **Define the serving artifact.** The ensemble must save its ordered
    component recipes/inputs, fitted weights, calibrators, text dependencies,
    cutoffs, coverage/fallback rules and data identities. Extend existing run
    storage and promotion checks rather than making a parallel registry. The
    workbench can display the executable registry instead of becoming another
    recipe owner.
12. **Clarify scope and framework policy.** Existing text regression and the
    proposed linear meta model already use scipy/sklearn. Keep LightGBM for
    boosted components initially and explicitly permit these linear combiners
    and text models. Neural models, video, GPS and new frameworks are not
    prerequisites. Experiments should omit costly place fitting; eventual
    serving must check the place pipeline's coupling to new win probabilities.

The central methodology agrees with the primary scikit-learn guidance:
[stacking trains its final estimator on cross-validated base predictions](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html),
and [selection/preprocessing must respect training boundaries](https://scikit-learn.org/stable/common_pitfalls.html).
Its generic stacking class is not a substitute for this project's race-grouped
chronological harness.

## Proposed implementation sequence

### 0. Establish the modelling contract and benchmark

- Record the existing dirty-tree state and audit holdout exposure. Preserve old
  runs and evidence. Reconcile documentation with the actual configuration.
- Declare forecast timing and field semantics before a new experiment. Use the
  existing five-minutes-before-scheduled-start benchmark as the provisional
  research contract; an actual betting horizon remains a product decision.
- Finish readiness checks, select accepted inputs, and rebuild the store once
  through its existing owner. Do not reset running source backfills.
- Reproduce the current 129-input generalist on that generation, with a small
  fixed benchmark of complete races. Keep the existing promoted artifact.
- Freeze the retrospective comparisons and the future confirmation rule.

**Exit:** a reproducible current-generation baseline with exact data, recipe,
feature order, field and calibration contracts; every score labelled with its
actual evaluation status.

### 1. Build shared modelling infrastructure

Prototype the workflow under `research/meta_learner/`, then promote qualified
reusable code into `src/modelling/`. It should own:

- immutable per-model specifications: ordered inputs, target, estimator,
  weighting, fit/update schedule and missing-data policy;
- a common fit/predict interface over the existing numerical implementations;
- one shared complete-race chronological fold/OOF mechanism;
- prediction tables keyed by race and runner, with component/fold IDs,
  training and calibration cutoffs, availability/support and recipe identity;
- a small combiner and nested evaluation using earlier OOF rows only;
- run manifests and resumable caches keyed by actual modelling inputs.

Reuse source manifests and existing run receipts. Do not recreate the retired
source-pinned study system or make every unrelated file edit a new study.
Keep compatibility entry points in `src/model.py`; extract behaviour as it is
needed rather than carrying tuning and betting code into every component.

**Exit:** S0-only identity, no race splitting, strict chronological lineage,
whole-field normalisation, missing-component fallback, batch/single agreement
and save/load agreement all pass. Runtime code imports no research.

### 2. Make generalist reduction a supported operation

Adapt the retained selection primitives to the actual learner contract. Add
explicit column projection and model-facing family/dependency metadata. Compare
the refreshed legacy 129-input baseline with a newly selected compact generalist
and a broad-feature control. Use earlier folds for all choices, later folds for
evaluation, and separate any historical reuse from fresh confirmation claims.

**Exit:** one operation regenerates a candidate feature set and its evidence;
routine fits consume a fixed saved list. The broad store remains shared.

### 3. Test a minimal ensemble

Predeclare a limited comparison:

1. the generalist alone;
2. generalist plus the existing quarterly text distribution;
3. generalist plus the reproduced performance/spread distribution;
4. all three, if the earlier comparisons justify it.

Compare text as a direct component with text inside the generalist, so a
change in information is distinguishable from a change in combination method.
Use complete-field predictions, fixed-weight pooling first, and nested earlier
OOF data for every fitted combiner and calibration. The spread learner needs its
own honest earlier residual predictions. Numerical integration can reproduce
058 initially; a large Monte Carlo simulator is unnecessary.

**Exit:** a modest ensemble improves on the strongest current-generation
single model under the predeclared paired evaluation, with stable calibration
and an operationally worthwhile fitting cost. If it does not, retain the
single model and the reusable infrastructure.

### 4. Add specialists selectively

Screen a few coherent groups rather than launching S1–S8 together. Ability and
clock, conditions/pace/tracking, and pedigree/readiness are reasonable candidate
boundaries, with small shared race context and explicit feature dependencies.
All runners remain scoreable even when their history is thin. Newer tracking
gets both a common recent-era comparison and a full-history fallback test.

Add predeclared context gating only if it improves the fixed-weight ensemble.
Use out-of-time ablations to decide which experts survive. Review past null
results before expanding text, luck or connection inputs. Defer head-to-head,
graphs and neural sequence models until the simpler system earns complexity.

**Exit:** every additional component or interaction has a measured incremental
benefit and an explicit support/cost budget.

### 5. Qualify serving and the optional market tier

Package the ensemble through existing training/run/promotion interfaces. Keep
the previous run available. Check all component dependencies, current-card
missingness, non-runners, field changes and live/historical feature parity.
The existing place head can initially retain its fitted weights, but its
projection/calibration uses win probabilities; verify place constraints and
scores instead of promising place output will be unchanged.

Only then evaluate an optional market-aware output at the declared forecast
horizon. Compare it against the same-time market and a calibrated market-only
control. BSP cannot enter a T-5 prediction. LTP is not an executable quote.

Lock the recipe/update schedule and record prospective forecasts with the
actual decision-time inputs before looking at their outcomes. Extend ordinary
forecast evidence rather than rebuilding the removed recorder/study machinery.
Promote only after those checks and the agreed confirmation gate.

**Exit:** reproducible, recoverable serving with explicit fallback, independent
confirmation appropriate to the claim, and separate reporting for forecast
quality and betting performance.

## Scope for the first implementation

The first implementation should cover Phases 0–1 and the supported feature
selection seam in Phase 2. It should deliver a trustworthy benchmark and an
ensemble-ready interface, not promise an accuracy gain before experiments.
The minimal ensemble is the next measured work package. No model or source
change is made as part of this review.

## Evidence inspected

- Current `train.py`, `config.py`, `src/training.py`, `src/model.py`,
  `src/race_objectives.py`, `src/wf_eval.py`, `src/feature_selection.py`,
  `src/deployment_validation.py`, `src/run_store.py`, feature-store loading,
  RTV text/observation/publication paths and workbench feature registry.
- `META_LEARNER_PLAN.md`, `FEATURE_CONTRACTS.md`, project/research layout rules,
  `FEATURE_SELECTION_048.md`, `MODEL_IDEAS_057_060.md`,
  `MARKET_PERFORMANCE_PROGRESS.md`, `NEGATIVE_RESULTS.md` and
  `FORECAST_BENCHMARK_PROTOCOL.md`.
- Selected/evaluated run metadata, actual store metadata and Parquet footer,
  quarterly text-model metadata and retained experiment evidence directories.
- Focused verification: **119 passed**, 171 warnings, in 199.28 seconds.
  Modules checked: layout, walk-forward evaluation, feature selection, RTV
  text, exploded win fitting, deployment training, training orchestration,
  probability pipeline and market-free inputs. The warnings concern sklearn
  feature-name handling and pandas future downcasting behaviour.
- The first test invocation stopped at fixture setup because its scratch
  parent directory did not exist; after creating that directory, the complete
  targeted run above passed. No pipeline code was changed to make it pass.
- This is a targeted architecture review, not a full feature-value or source
  audit, full-suite regression run, or new predictive-performance experiment.
