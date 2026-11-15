# Modelling foundations implemented, 8 November 2026

This implements the first work package from the
[review and migration plan](MODELLING_REVIEW_AND_PLAN_2026-11-08.md).
The existing predictor, feature calculations, accepted sources and production
selection are retained. The [operating guide](../operations/MODELLING.md)
contains the supported commands.

## Delivered

- `src/modelling/spec.py`: immutable, resolved and ordered win recipes. Separate
  components no longer need to mutate the global feature list, target or
  learner settings. Parameter ordering survives a JSON round-trip.
- `data.py` and `win.py`: projected, checked reads from an accepted source/store
  generation, complete-field validation and a win-only learner sharing the
  existing objectives, teacher, weights and temperature mathematics. Native
  fitting matches the legacy win learner's raw scores and calibration in the
  regression fixture.
- `src/wf_eval.py`: explicit calendar folds, validation before any fitting,
  complete-date/race boundaries and duplicate/overlap/purge checks. Default
  block mode also keeps dates intact.
- `evaluation.py` and `pool.py`: canonical keyed OOF predictions, verified
  resumable caches, nested chronological pooling, explicit whole-field expert
  availability and generalist fallback. A one-component pool is exactly
  identity; it does not secretly recalibrate the baseline.
- `selection.py`: bounded column-batch preparation into a float64 memmap,
  earlier-only hygiene/recursive elimination, date-clustered one-SE choice,
  separate later confirmation, resumable checked steps and new candidate
  feature/spec outputs. Wide fits pass a narrow context table and an explicit
  ordered numeric matrix to `WinModel`, avoiding repeated full-width DataFrame
  copies. Routine fits still consume a fixed list.
- `catalog.py`: shared navigation metadata for model inputs, including learned
  text and prior-market inputs. Unknown individual dependencies remain marked
  unknown. The workbench now uses the same family classification.
- `text.py`: an explicit TF-IDF text expert with saved vocabulary/IDF/weights,
  cutoff and input identity. Prediction never initiates fitting. The legacy
  quarterly materialiser remains in place for the existing model until a
  replacement is qualified.
- `candidate.py` and the existing run registry: lean, reloadable win candidate
  artifacts with ordered components and fitted pool weights. These cannot be
  promoted through the existing serving interface, including its old feature
  mismatch override; serving/place qualification remains a later step.
- `python -m src.modelling`: supported `benchmark`, `prepare`, `select` and
  `fit` operations. Experiments remain separate under `research/meta_learner`.
- `research/meta_learner/text_probe.py`: a small retrospective combination test
  which reuses baseline OOF forecasts and stored text scores. Its pool uses an
  earlier fitting period and a purged later scoring period. It does not claim
  text is new information absent from the generalist.

## Data build and resource control

The separate store is
`data/processed/modelling_feature_store_20261108_serial`.
It contains all **758,304** accepted runner rows through **6 November 2026**,
using dataset **cdcc01a5d698ad70fe4f7b6f4c0cd9398f6eb15bcb944eb10100a5c3a9c0b636**.
The existing shared store and selected model were not replaced.

The first parallel candidate build caused excessive resource pressure and was
stopped. The successful retry used the existing inline feature path with
`--no-helpers --chunk-rows 20000`. The build command now exposes this resource
option; the default pipeline execution is retained. The successful build
processed 38 chunks and held roughly 2.7–3.9 GB after preparation. Its resource
control change updates the normal feature-code signature; no metadata was
restamped to bypass compatibility checks.

The initial readiness report, interrupted build log, successful serial build
log and verification logs are under
`data/audit/modelling_foundations_20261108/`.

The final quick installation check at 16:39 still reports the original shared
store and selected model as stale, plus partial source refreshes and one pending
race. The selected run remains `20261104_235300_444862`. This separate checked
research store does not repair or replace the installation's serving generation.

## Verification

- Focused foundations checks: **60 passed**, including native learner parity,
  recipe immutability, complete-race chronological folds, missing-expert
  fallback, identity, cache integrity, selection/confirmation separation,
  selection resumption, artifact reload and promotion refusal.
- New modelling/research code passes Ruff. The working diff passes whitespace
  checks. The command help route works.
- Broad regression run: **1,761 passed, three skipped, two failures**. Both
  failures were corrected and the affected group reran successfully:
  **14 passed**. The broad run took 4,260 seconds while the candidate store was
  being built. A second full run was not needed after the targeted fixes.
- The broad run exposed an existing collector test whose two-second wall-clock
  tolerance failed during slow fixture I/O. Its clock is now frozen so it tests
  the declared backoff policy. The corrected test passes in isolation; no
  collector/worker code was changed.
- The other failure exposed a compatibility regression introduced by the new
  model-type guard: legacy artifact-only checkpoints may have no metadata.
  Their existing explicit override and load behaviour is restored. Registered
  win-only candidates remain ineligible for promotion.
- After adding the explicit matrix route, the affected regression group passed:
  **138 passed**. Seven matrix-specific checks also passed, including exact
  named-frame/matrix prediction and calibration agreement at screen and native
  capacity for both supported objectives, ordered-input validation and shared
  training/calibration matrix storage. These are targeted reruns, not a second
  full-suite run.

## Measured baseline and text probe

The native benchmark is saved under
`data/research/meta_learner/baseline_native_2026q2/`. It scores **2,544 races**
(23,725 runners) from **1 April through 3 June 2026**, retaining the existing
holdout exclusion and seven-day purge. Race log loss is **1.8458393**, race-mean
Brier score **0.0964159**, and top-choice accuracy **29.717%**. These establish
a reproducible retrospective baseline, not an independent accuracy claim or a
comparison with the old fitted artifact.

The registered text probe is saved under
`data/research/meta_learner/text_probe_2026q2/`. Pool weights were fitted through
30 April and scored from 8 May, with a seven-day purge. On those later races:

| Model | Race log loss |
|---|---:|
| Generalist | 1.8508472 |
| Generalist plus stored text distribution | 1.8505258 |

The paired change is **−0.0003213**, with a 95% date-cluster interval of
**[−0.0011228, +0.0005092]**. The fitted text weight is **3.745%**. This small,
uncertain difference does not establish a gain; no blend was adopted. The
generalist already consumes text-derived inputs, so this is a test of adding
the existing text distribution as a separate component, not new information.

The measured source version is preserved with the baseline in
`source_snapshot/` and its SHA manifest. Its modelling-code fingerprint begins
`18d538231e90`; the later optional matrix interface changes that fingerprint
to one beginning `416eacdf5ccf`. The ordinary named-frame mathematics is retained
and native matrix equivalence is tested, but historical receipts were not
rewritten. The strict fitting guard therefore requires a fresh benchmark
directory before fitting a candidate with the current source version. The
snapshot is a review/replay archive for this measurement, not a new runtime
source-pinning system. All new comparisons use canonical date/race/runner
ordering; whole-history bit-for-bit equivalence with legacy event-time ordering
is not claimed.

## Wide feature preparation and memory check

The actual-data preparation under
`data/research/meta_learner/preparation_smoke_2026q2/` contains **23,725 rows ×
1,305 eligible numeric inputs**, with a float64 matrix, canonical context,
family catalog and content hashes. This checks the supported preparation path;
it does not select or adopt a new feature list.

`research/meta_learner/matrix_smoke.py` also fitted all 1,305 inputs using
10,446 earlier training runners and scored 10,610 later runners. The sampled
peak resident memory was **0.834 GB** on this small slice. This 20-tree screen
checks routing and resources, not model quality, and does not establish a
full-history memory bound. Full-universe recursive elimination was not run.

## Limits and remaining work

The implementation does not claim a forecasting gain. The old June 2026 onward
holdout has already been examined elsewhere, and the previous input lists were
selected using historical development labels. New benchmark reports therefore
remain explicitly retrospective and preserve the existing holdout exclusion.

The new selection workflow is supported and tested; it does not automatically
replace the 129-feature list or run an unbounded full-universe search as a side
effect of fitting. A selected list needs its own recorded confirmation before
adoption. The legacy text-score training lifecycle is preserved for existing
features, while the explicit expert provides the replacement seam.

The performance/spread reproduction, further information specialists,
context-dependent gates, timed market tier, live/place adapter and prospective
confirmation remain subsequent measured work packages. No candidate is silently
promoted, and the existing installation's stale selected-model generation is not
made compatible merely by this separate candidate build.

Follow-up: an [initial native performance/spread pilot](../research/PERFORMANCE_SPREAD_REPRODUCTION_2026-11-08.md)
has since completed with a promising paired improvement. Wider-period and
stronger-generalist confirmation remain outstanding. Its independent artifact
verification also corrected the baseline race count above to 2,544; the earlier
2,650 count was a reporting error. Forecasts and numerical scores were unchanged.
