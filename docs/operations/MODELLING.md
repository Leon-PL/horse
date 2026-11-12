# Modelling foundations

The supported entry point is `.venv\Scripts\python.exe -m src.modelling`.
It uses accepted source datasets and compatible feature stores, never collects
sources or silently rebuilds history. `train.py` and the selected production
model keep their existing behaviour.

## Recipes and ownership

`src/modelling/spec.py` defines an immutable `ModelSpec`: ordered numeric
inputs, LightGBM parameters, hard/distilled target, finishing-order weights,
seed, recency weights, burn-in, purge, screen capacity and calibration fraction.
Defaults are captured once when the recipe is constructed. JSON recipes saved
by a benchmark can be reused with `--spec`; repeat that option for multiple win
experts. No operation changes `config.FEATURE_ALLOWLIST` or another model's
settings. Screen and native fits have distinct recorded capacities.

`WinModel` reuses the established race objectives, market teacher and temperature
calibration. It fits win only, keeping genuine non-finishers. Training uses
complete, unambiguous single-winner fields. Prediction validates whole fields,
keeps missing numeric measurements as NaN and restores the caller's row order.
Native capacity uses the existing purged stopping rule and a full refit. The
calibrator learns from a separate earlier training slab, never the scored fold.

`PerformanceSpec` and `PerformanceModel` define a separate performance/spread
component. A centred margin regression predicts mean performance; another
regression learns uncertainty from strictly earlier-only residual predictions.
A later purged training slab calibrates spread scale. Missing margins and
non-finishers are excluded from performance targets and remain in every scored
field. Prediction requires only the saved models/scale and declared features.

A performance recipe has `"kind": "performance"` and its own ordered inputs and
numerical controls. Existing win recipes remain valid without a `kind` field.
`benchmark --spec ... --spec ... --pool` and candidate fitting dispatch both
types through the same OOF and artifact paths. Performance fits additionally
project `beaten_lengths` and `distance_furlongs` as training context; those
columns are not silently appended to the input list. Supported feature
selection currently applies to the numeric win learner.

`TextExpert` is an explicit calibrated TF-IDF/conditional-logit expert. It takes
supplied raw comments, verdicts and earlier result labels. Its recipe can vary
regularisation, word/character representation, self/rival attribution, recency,
finishing-order targets and the historical market-teacher share. The last two
are training choices; prediction needs no results or prices. It records the
fitted vocabulary, IDF, weights, scale, optimisation, cutoff and input digest;
prediction never initiates fitting. An incomplete text field masks the expert
for the whole race. The [text development report](../research/TEXT_MODEL_IMPROVEMENTS_2026-11-08.md)
records the evaluated recipes and the separate current candidate. This explicit
raw-text component is not dispatched by the numeric `--spec` CLI yet.
The [validated text recipe](../../reference/modelling/timeform_text_market_v1.json)
can be passed directly to `TextExpert(**settings)`. Its historical prices
inform training; no price columns are required for `predict`.

The legacy source materialiser still maintains the quarterly scores
consumed by the existing 129-input model. Moving that existing path requires
qualification of the replacement and is not done by changing feature semantics
under old models.

The reusable mechanics are in `src/modelling/`. Experiments and registrations
belong in `research/meta_learner/` and `data/research/meta_learner/`. Production
code imports no research. LightGBM remains the boosted estimator; the small
linear text/combination models use the installed scipy/sklearn stack.

## Compatible data

Inspect the installation before fitting:

```powershell
.venv\Scripts\python.exe scripts/project_status.py
```

Rebuild through the established owner, optionally into a separate candidate
store:

```powershell
.venv\Scripts\python.exe -u -m scripts.build_feature_store_batched --store-root data/processed/modelling_feature_store_20261108 --chunk-rows 60000
```

On a busy 16 GB machine use `--no-helpers --chunk-rows 20000` to keep feature
calculation in one process. This changes execution resources, not feature rules.

The modelling loader verifies the feature signature and the store's exact
accepted source dataset, historical result signature and ATR/RTV/BHA signatures.
It reads projected columns and joins teacher prices through the selected
canonical source links. It never follows a moving collector pointer. Source
readiness limitations and excluded invalid fields are recorded separately from
statistical claims. One pending recent race need not be silently included as a
loser in a historical benchmark.

## Baseline and ensemble infrastructure

```powershell
.venv\Scripts\python.exe -m src.modelling benchmark --store-root data/processed/modelling_feature_store_20261108 --fold-start 2026-04-01 --capacity native --threads 2 --output data/research/meta_learner/baseline
```

The default recipe uses the configured feature list in the store's column order,
matching the established baseline. `--start`/`--end` restrict the dataset; they
also reduce available training history. `--fold-start` restricts scored quarters
while retaining earlier training history. The reserved holdout and purge are
always excluded from this development command.

Outputs include resolved recipes and dataset/feature identities in
`registration.json`, keyed `oof.parquet`, `report.json`, and per-fold caches with
fit/calibration cutoffs. Resuming the same registration reuses verified folds.
Different data, feature order, settings, modelling code or corrupt cached bytes
cannot silently reuse a fit. An unrelated source-file edit does not create a
new study or feature generation.

Benchmark receipts remain tied to the source version that produced them.
After changing modelling code, use a new output directory; the candidate fitting
guard will reject a benchmark from an older modelling version. Preserved
research snapshots describe historical measurements and do not bypass that
guard.

`oof_predict_expanding` now accepts explicit quarterly `Fold` objects and checks
all masks before fitting: complete days/races, strict purge, unique fold labels
and no duplicate predicted rows. Its default block mode also splits whole days.

Use several `--spec` files and `--pool` to evaluate a fitted combiner. The pool
is fitted on earlier OOF quarters only and scored on later quarters; it needs
additional OOF history. It uses nonnegative log-pooling weights summing to one,
with fixed shrinkage towards the generalist. An S0-only pool is exactly identity,
including exact zero probabilities. Incomplete expert coverage masks that
expert for the whole race; complete failure falls back to the generalist.
Malformed finite distributions are rejected rather than renormalised silently.
No context gating or same-race market tier is enabled.

These are **retrospective comparisons**. The formerly reserved June 2026 onward
period has been examined elsewhere, and the existing feature lists informed
earlier research. Historical OOF fitting alone does not make feature selection
independent of those periods. Do not move the holdout date to recycle examined
data. Fresh confirmation needs a locked recipe/update schedule and prospective
decision-time inputs and forecasts.

## Feature reduction

Prepare the eligible universe without loading a full wide DataFrame:

```powershell
.venv\Scripts\python.exe -m src.modelling prepare --store-root data/processed/modelling_feature_store_20261108 --output data/research/meta_learner/selection_matrix
```

Preparation reads columns in batches into a float64 memory-mapped matrix,
preserving input precision. It saves canonical context, ordered column names,
content digests and a family/kind catalog. Catalog entries explicitly identify
learned text and prior-market inputs, while marking unresolved column-level
dependencies as such. Prefix classification is not evidence of independence.
The broad feature store is retained.

During selection, fits use a narrow context table and an explicitly ordered
numeric matrix. Only the active input subset is materialised; the learner does
not repeatedly copy a full wide DataFrame. Matrix inputs require exact ordered
feature names and matching row counts. The active matrix and LightGBM's own
working data still require memory, so the preparation memmap is not a promise
that a full-history broad fit will have the same footprint as a compact model.

Select on earlier periods, then confirm on later periods:

```powershell
.venv\Scripts\python.exe -m src.modelling select --prepared data/research/meta_learner/selection_matrix --selection-end 2025-06-30 --confirmation-start 2025-07-08 --capacity screen --threads 2 --output data/research/meta_learner/selection
```

The dates above reproduce the broad historical research separation, not a fresh
independent test. The gap also applies between selection labels and the first
confirmation race. Confirmation may therefore omit the first days of a quarter.

Hygiene uses earlier rows only and reads one column at a time. After it, one
representative is kept of each cluster of near-identical columns (absolute rank
correlation of 0.95 or more on earlier rows; `--near-copy 0` keeps them all).
A current model input, then the better-covered column, represents its cluster.
The store holds many versions of one measurement, and elimination otherwise
splits their credit and carries hundreds of redundant inputs through its first
fits.

Recursive elimination uses the paired date-clustered one-SE rule. Inputs are
ranked on each fold's unseen races, by `--importance`:

- `shap`: each input's share of the model's mean absolute contribution. Cheap;
  it measures how much the model leans on an input.
- `permutation`: the rise in race log loss when the input is shuffled across
  the fold's latest 1,500 races. It measures what the forecast loses without
  the input, and costs one prediction pass per input per fold.
- `auto` (default): SHAP share while more than `--permutation-below` (150)
  inputs remain, the loss prevented after that, where the choice is made.

`--confirm-capacity native` repeats the baseline and selected arms at native
capacity after a screen-capacity selection. A 400-tree fit can favour inputs
that only make up for the capacity it lacks
(`docs/research/ERROR_ANATOMY_2026-11-10.md`, sections 5 and 10), so a list
chosen at screen capacity should not be adopted on screen figures alone.
Every subset uses the same explicit `WinModel` learner/calibration recipe.
Confirmation compares the frozen selected list, the baseline and the broad
hygiene-kept universe on identical races; it does not change the choice.
Each finished step/arm is checkpointed and verified before reuse. The output
contains `selected_features.txt`, a resolved `selected_spec.json`, hygiene
rejections, the selection path and paired confirmation differences. Repeat
confirmation at native capacity before adopting a list selected at screen
capacity. Feature selection is deliberate, not a hidden step on routine refits.

To test new columns without repeating the elimination, ask whether adding them
to a recipe helps:

```powershell
.venv\Scripts\python.exe -m src.modelling admit --prepared data/research/meta_learner/selection_matrix --candidates hidden_form_wins hidden_form_best --from 2025-07-08 --capacity screen --threads 2 --output data/research/meta_learner/admit_hidden_form
```

It fits three arms on the folds from that date: the recipe, the recipe with the
candidates, and a twin with the candidates shuffled within each quarter. The
real arm against the twin is what the candidates' information is worth; the
twin against the recipe is what extra inputs do by themselves. `--candidates`
also takes one text file of names. Nothing is added to any list.

## Candidate fitting and saved runs

```powershell
.venv\Scripts\python.exe -m src.modelling fit --store-root data/processed/modelling_feature_store_20261108 --benchmark data/research/meta_learner/baseline --as-of 2026-06-03 --threads 2
```

This reuses the benchmark's resolved recipes and verified OOF predictions, fits
the final pool on OOF forecasts and refits its win components through the stated
settled cutoff. It records no held-out performance for the refitted weights.
`src.run_store` lists and loads the lean `win_stack_candidate` artifact, including
its pool, ordered recipes and component lineage. Such a run is deliberately
ineligible for promotion, including through the old feature-mismatch override:
the live/place adapter and its validation are a later qualification step.
The existing selected model is never replaced by a modelling command.

## Remaining research

The first scope supplies the benchmark, explicit recipes, shared OOF/pool
mechanism, candidate persistence and supported selection. A richer information
specialist, wider performance/spread confirmation, wider text sources, contextual
gating and the timed market tier require separate registered comparisons.
The selection list or ensemble must earn adoption; infrastructure alone is not
an accuracy gain.

The [initial performance/spread reproduction](../research/PERFORMANCE_SPREAD_REPRODUCTION_2026-11-08.md)
records the first fixed native-capacity comparison and its evidence limits.

## Current default list (selection of 2026-11-12)

`config.FEATURE_ALLOWLIST` points at `feature_sets/win_selected_20261112.txt`, 87 inputs chosen by
`select` from the 1,317 numeric store columns (1,065 after the hygiene and near-copy steps; store
`fe_signature` 06c07aff1f3f1c6c). Selection used races to 2025-06-30 (six folds, 2024Q1 to 2025Q2);
confirmation used 11,632 races from 2025-07-08.

| List | Inputs | Screen capacity | Native capacity |
|---|---|---|---|
| `win_with_rider_context.txt` (previous default) | 133 | 1.82997 | 1.82491 |
| Selected | 87 | 1.82894 | 1.82446 |
| All 1,065 | 1,065 | 1.82910 | not fitted |

The selected list is level with the previous default (native difference -0.00046, 95% interval
-0.0024 to +0.0014, better in three of four quarters): it was adopted for being smaller, not more
accurate. It keeps 67 of the previous inputs and adds 20, including `hidden_form_mean` and
`hidden_form_best_vs_field`; none of the rider-context inputs was selected. Run outputs are in
`data/research/selection_20261111/select/`. This list is not the full 063 recipe, whose 64 extra
inputs are built at fit time and are not store columns.
