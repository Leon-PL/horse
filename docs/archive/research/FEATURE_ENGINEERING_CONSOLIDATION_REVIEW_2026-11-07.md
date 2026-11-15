# Feature engineering consolidation review — 7 November 2026

## Assessment

The next consolidation should focus on **feature contracts and ownership**.
The project already has a shared operational route for training builds and
state-backed live features. Replacing that machinery would discard useful
correctness and memory protections. Inside the route, however, feature meaning,
dependencies, timing, missing values and state ownership remain distributed
across large functions, source-specific implementations and consumer heuristics.

The practical consequence is that adding or changing a feature still requires
knowing incidental behaviour elsewhere in the pipeline. Moving functions into
smaller files would help navigation, but would not resolve those dependencies.

This is an architecture and targeted correctness review, not an evaluation of
predictive value or an exhaustive leakage certification. It includes the new BHA
family. No feature calculations, ingestion code, stores or models were changed.

## Baseline and existing strengths

Reviewed checkout: `ac68908b1ff3bb6af06728f13f657df8ea6abc96`.

- `src/feature_engineer.py` has 5,039 lines and schedules 40 stage calls with
  the current default rating configuration. Existing family modules already
  include ATR, RTV, BHA, established form, surface and prior market features.
- The active allowlist is `feature_sets/win_selected_056.txt`: **129 unique
  names**. There are 14 feature-set text files. The broad historical `win.txt`
  is not the active recipe.
- BHA contributes **161 numeric candidates** to normal feature calculation;
  they are not automatically selected into the active 129-feature recipe.
- Training enters through
  [`store_refresh.get_training_dataset`](../../src/store_refresh.py#L81).
  It checks whether the existing store can append and otherwise uses the
  bounded builder in [`store_build`](../../src/store_build.py#L264).
- The normal live route is
  [`features_for_live_cards`](../../src/live_prediction.py#L377) through
  [`FeatureStore.featurize_live`](../../src/feature_store.py#L436).
  It reads persisted rating/count state and discards in-memory advances.
  `feature_engineer_with_history_core` remains as a reference/test path;
  its existence does not mean normal live requests rebuild full history.
- Atomic publication of the five store outputs, carried state, input/code
  signatures, accepted source manifests and existing parity tests are valuable.
  The split builder also already rejects overlapping state entities at the
  process handover in
  [`_add_second_half_state`](../../src/store_build.py#L179).

The read-only runtime status check at **21:32 on 7 November** found an installed
store with 757,580 rows through 4 November, an older feature signature, changed
ATR/RTV inputs and a model/store generation mismatch. Consequently the installed
artifacts are not a current validation baseline for the new code or BHA family.
This is observed rebuild/retrain debt following changes, not evidence that the
new architecture itself produced incorrect predictions. No rebuild or retrain
was performed for this review.

## Findings

### 1. Feature schema validation is too permissive — fix first

[`FeatureStore.featurize_live`](../../src/feature_store.py#L474) completes missing
training one-hots by column-name prefixes such as `going_` and `surface_`.
Those prefixes also describe continuous derived features. An isolated probe
confirmed that a missing **`going_numeric` becomes zero**, just like a missing
`going_Good` dummy. This verifies an unsafe fallback; it does not establish that
ordinary current cards actually omit `going_numeric`.

Separately, [`model.get_feature_columns`](../../src/model.py#L679) intersects
the allowlist with eligible numeric columns. It raises when none match, but
accepts a partial match with an INFO log. An isolated request for
`known_feature` and `missing_feature` returned only `known_feature`.
Thus missing calculation outputs can reduce a training feature vector silently.

**Recommendation:** make a selected recipe an explicit, ordered schema. Validate
required names, dtypes and permitted missing values before training or serving.
Declare categorical domains and their dummy columns explicitly. Zero completion
should apply to declared absent categories; an absent required computed column
should fail with an explanation. Keep an explicitly permissive mode for candidate
exploration if needed.

### 2. Execution order depends on incidental side effects

[`engineer_features`](../../src/feature_engineer.py#L4837) owns the stage order,
temporary columns, process split, state scope, final defaults and optional file
writing. Requirements and outputs are mostly implied by code and comments.

A particularly revealing example is its seeded-rating branch around line 4907:
rating sweeps delete `_event_dt`, causing later rolling operations to use
day-level ordering. When ratings are supplied from state, the engine deliberately
repeats that deletion to preserve batch parity. Changing a rating implementation
can therefore change unrelated downstream temporal behaviour.

[`FeatureStore._with_dist_category`](../../src/feature_store.py#L268) also mirrors
distance bins normally created in a feature stage so ratings can run first.
The same definition consequently has two owners.

**Recommendation:** declare stage inputs, outputs, ordering/time rules and state
ownership. Resolve distance classification once before its consumers. Make the
current ordering rule explicit before considering any change to it. Removing the
`_event_dt` dependency is a separately versioned semantic change, not a harmless
file move.

The shared execution contract should also state that runners in the same race
read the same prior state before settled results advance it, and that live-card
calculation never commits advances. A list of function dependencies alone does
not establish these temporal guarantees.

### 3. State is carried efficiently, but exposed through process globals

[`counts_state_scope`](../../src/feature_engineer.py#L700) temporarily rebinds
`_COUNTS_STATE`; rolling helpers also use a shared `_TIMELINES` cache. An isolated
two-thread probe made scope A observe scope B's state.

Normal large builds use separate processes and store operations have locks, so
this is not a finding of current worker corruption. It is a real restriction on
future concurrent in-process feature previews, candidate builds or separate
stores. A lock attached to one store does not define all feature-engine access.

**Recommendation:** pass an explicit execution context with private working state
and caches, using namespaced, versioned family state. Preserve the builder's
existing isolation and handover checks. Keep any compatibility scope short-lived
while families move to the explicit interface.

### 4. Preparation and feature calculation have overlapping ownership

[`data_processor.encode_categorical`](../../src/data_processor.py#L438) creates
both one-hot outputs and expanding entity-frequency features. Those frequencies
depend on prior rows, although this module otherwise looks like preparation.
Live serving then repairs the card-local counts through
[`_continue_freq_counts`](../../src/feature_store.py#L411), which reads the
complete processed-history columns and counts appearances on each request.

The within-race collapse already protects frequency counts against finish-order
leakage; retain that behaviour. The issue is divided ownership and a full-history
read inside an otherwise incremental live route.

**Recommendation:** retain stable preparation of types, units, canonical keys and
normalized source values. Give stateful encodings a feature-family owner and
persist their accumulator state. Keep a compatibility projection until batch,
append and live parity is demonstrated. Profile the live path before claiming a
speed improvement.

### 5. Source feature families use different history contracts

The families have useful source-specific protections, but their definitions are
not expressed through a common catalog:

| Family | Current history behaviour | Contract that needs to be visible |
|---|---|---|
| ATR | [`published_atr_rolling`](../../src/atr_features.py#L563) retains canonical prior starts, including unobserved slots, and applies availability guards | Window denominator, metric support, eligibility time and source versions |
| RTV | [`_history_observations`](../../src/rtv_features.py#L332) drops runs with no measured telemetry, then rolls over three surviving observations; [`add_history_features`](../../src/rtv_features.py#L370) joins at target race-date granularity | Observation window rather than start window, per-metric support, age/tolerance and date precision |
| BHA | [`compute_features`](../../src/bha_features.py#L118) retains prior-start slots and separates coverage from incident values | Assumed availability, report support, attribution and amendment limitations |

A synthetic example makes the difference concrete. Suppose four prior starts
have values `100, 110, missing, 120`. RTV's last three usable observations average
**110** with support 3; the last three canonical starts average **115** with
support 2. Neither definition is intrinsically wrong. Similar `mean3`/`avg3`
names do not tell a reviewer which definition applies.

BHA's user-selected **`race_plus_48h`** policy must remain intact during a
structural refactor. It permits the selected snapshot's report text after an
assumed 48-hour delay; actual capture timestamps remain recorded. Original
publication and amendment times are not established. This assumption belongs in
feature metadata and the UI, alongside a distinction from recorded-time
eligibility. See the [current BHA definitions](BHA_FEATURE_CANDIDATES_2026-11-07.md).

**Recommendation:** share terminology and contract validation, not necessarily
the same rolling algorithm. Each history output should specify entity/key,
window unit, eligible-history rule, support, missingness and decision-time
policy. Preserve family behaviour initially; compare intentional harmonizations
as separate candidates.

### 6. Derived and fitted feature artifacts need a shared identity

ATR run measurements are transformed inside feature code. RTV materializes run
figures and learned comment scores through `ingestion.rtv_observations`, while
`rtv_features` and `rtv_text` contain both producer transformations and consumer
logic. BHA derives cached observations from accepted source versions inside its
feature module.

RTV text scores deserve particular visibility: they come from a purged,
quarterly fitted model, with vocabulary, IDF, weights and `trained_through`
metadata in [`rtv_text`](../../src/rtv_text.py#L220). They are not merely raw
source measurements. Existing source signatures cover the published values;
this finding is about clear ownership and inspectable lineage, not proof that
current text fitting leaks.

**Recommendation:** introduce a common manifest for derived/fitted feature
artifacts: accepted source versions, transformation version, fit cutoff, output
schema and availability policy. Keep source collection/publication stable.
Existing materialization can remain in place initially, referenced through this
interface. Feature builds and UI reads should not trigger fitting or collection.

### 7. Column roles and feature recipes are inferred by consumers

[`candidate_feature_columns`](../../src/feature_selection.py#L60) discovers
candidates through numeric dtypes and the model's exclusion list, bypassing the
active allowlist. This correctly makes BHA candidates available without mutating
the selected model inputs, but candidate eligibility still depends on downstream
heuristics rather than explicit feature ownership.

The wide build mixes source fields, labels, diagnostics, selected features and
candidates. Final imputation is also scattered: odds receive neutral defaults,
history often remains missing, and raw weather columns receive UK-average values
in the engine's final block, although weather has no active feature stage.
That is inconsistent semantics, not a demonstrated defect in the active
129-feature model.

**Recommendation:** a feature catalog should distinguish selected production
inputs, evaluated candidates, diagnostics, identifiers, labels and source fields.
For each feature record owner/family, dependencies, dtype/units, null policy,
history/availability rule, state version and source/artifact dependencies.
Keep raw missing values available for inspection; record imputation as a named
transformation. Candidate promotion requires measured evidence using the
project's existing evaluation primitives and negative-results ledger.

Do not immediately compute only the 129 selected columns: intermediate and
race-relative dependencies must be represented before selective execution is
safe. Retain a broad candidate-build recipe alongside the selected recipe.

### 8. Existing parity tests and signatures need refinement, not replacement

The feature-store tests deliberately disable the local allowlist and compare
numeric values and NaN masks, which is a useful safeguard. However,
[`_assert_append_matches_batch`](../../tests/test_feature_store.py#L59) first
intersects batch features with columns present in the append. An entirely absent
append column can escape that comparison. Its row alignment also uses
`race_id` plus `horse_name` rather than canonical runner IDs.

**Recommendation:** assert declared schema, column order/dtypes, key uniqueness
and missingness before value comparison. Use canonical runner keys. Retain
batch/append/live tests and add populated source cohorts plus chunk-size,
row-order, same-race, cutoff and outcome-perturbation invariants where missing.
The BHA family already has several of these tests and can inform the shared
contract suite.

The current AST/source/input signatures and import-graph coverage tests are
strong protections. Their breadth also means changes to an unrelated candidate
or execution implementation can invalidate a whole store. Keep this conservative
behaviour during extraction. Only after dependencies are complete should recipe
semantics, execution/state versions and source inputs be distinguished to avoid
unnecessary rebuilds. Orchestration currently changes values, so simply removing
it from the signature would be unsafe.

The engine also defaults to `save=True` and can write a separate
`featured_races.parquet`. Move persistence to explicit store/export services
eventually, preserving compatibility for known callers until retired.

## Proposed target and migration order

Keep `store_refresh`, `FeatureStore` and the bounded/atomic builder. Behind their
existing interfaces, introduce a small feature package:

```text
src/features/
    contracts.py     # feature/stage specifications and execution context
    registry.py      # feature roles and explicit recipes
    engine.py        # validated plan shared by batch, append and live
    history.py       # reusable, tested state/history primitives
    artifacts.py     # derived/fitted artifact manifests
    families/        # horse/form, connections, pedigree, ratings, race, sources
```

This is a direction, not a requirement to implement a generic framework or
rewrite the optimized NumPy/Numba kernels. Existing modules can provide adapters
until their contents move. Useful current normalization and semantic helpers
should be reused, along with the existing `FE_PROFILE` stage timings, bounded
chunking and process helpers.

| Order | Deliverable | Completion evidence |
|---|---|---|
| 1 | Freeze a current code/input baseline and strengthen schema boundaries | Exact selected schema; missing computed fields fail; dummy domains explicit; comparison cannot skip absent columns |
| 2 | Declare stage contracts and private state/context | Batch, append and live use the same declared plan; state isolation verified; current time/order behaviour explicitly recorded |
| 3 | Extract families mechanically and consolidate duplicate encoders | Output/missingness parity across modes and chunk sizes; frequency history carried in state; no core-to-research imports |
| 4 | Add catalog, recipes and artifact lineage | Every selected output has an owner and dependency closure; BHA timing assumption and RTV fit cutoffs inspectable; candidate builds remain available |
| 5 | Profile and then optimize | Stage time/RSS and live latency measured; production dependency pruning and narrower invalidation only with parity evidence |
| 6 | Extend Racing Workbench with feature inspection | Catalog, runner-level values/history, missingness/support, recipe/build health and recipe comparisons read the same contracts |

Steps can overlap, but **behaviour-preserving extraction and changes to feature
meaning must be reviewed separately**. Otherwise a large refactor can conceal
changed backtest inputs, or parity tests can enshrine a new accidental default.
Historical `FE_REFACTOR_PLAN.md` prose still describes an older fallback and
56-minute rebuild despite its historical banner; replace the current operational
view with generated registry/stage information and retain that document as a
dated migration record.

## Workbench implications

Once the contracts exist, a Features area can answer useful questions without
embedding another feature implementation in the UI:

- For this runner, what is the feature value, how many eligible historical
  observations support it, and why is it missing or zero?
- Which starts and source reports were eligible at the decision cutoff?
  Which source snapshot, timing assumption or fitted artifact was used?
- Is the feature selected, a candidate or diagnostic? Which other features
  depend on it? What changed between two recipes or builds?
- Which families have missingness, support, distribution or parity regressions?
  Which store inputs/signatures are stale, and what does the next build require?

Use the existing job service for explicit builds and comparisons, with progress
and logs. Page rendering should read artifacts. Source ingestion can remain as
the accepted foundation the user is already satisfied with.

## Validation performed

- Read-only code/caller, stage-order, state, source timing, signature and test
  review; active feature-list and BHA-output counts checked from current code.
- `scripts/project_status.py` run to establish actual installed compatibility.
- **37 focused tests passed** in `test_fe_signature_coverage.py` and
  `test_bha_features.py` (21.20 seconds).
- **7 further tests passed** in `test_feature_store.py` and
  `test_live_parity.py` (241.73 seconds): batch/append values, persisted rows,
  rating end-state, DataFrame fragmentation and the reference live path.
  These synthetic tests are useful regression evidence, not a complete
  populated-source or normal store-backed live audit. They emitted existing
  pandas downcasting and third-party escape-sequence warnings.
- Four isolated probes confirmed the prefix-completion fallback, partial
  allowlist acceptance, different history-window denominators and same-process
  global-state interference described above. They use synthetic inputs and do
  not modify production artifacts.

Local audit outputs are under
`data/audit/fe_consolidation_20261107/` (gitignored). This review does not claim
that every feature is leakage-free, that every current live card triggers a
schema fallback, or that a refactor will improve model returns.

## Follow-up review — 8 November 2026

Reviewed the changes from `ac68908` through `55b9e6e`. They strengthen the
accepted-source boundary: source helpers now require a selected dataset, ATR
has one published-history implementation, RTV snapshots use saved canonical
links, prices no longer fall back to the collector matcher, and BHA honours
the selected dataset's root. Old stores without recorded source lineage are
rejected for live use and rebuilt by training refresh. Tests now select isolated
source datasets instead of depending on the installed collector caches.

These are useful improvements and should be retained. They do not yet address
the exact output schema, global state, ordering side effects, frequency-encoding
ownership or column-level catalog findings above. In particular,
`FeatureStore.featurize_live` still completes columns by prefix, model feature
selection still permits partial allowlists, and append parity still intersects
the output column sets.

Two boundary gaps were reproduced against the updated code:

1. **A missing declared input is still treated like an optional absent source.**
   `read_table` returns an empty frame when a table listed in the selected
   receipt is missing on disk. If its declared crosswalk is missing, a linked
   read returns the source rows without the crosswalk, and `model_table` returns
   them without requiring `match_status`. Isolated probes returned an empty
   declared pedigree table and accepted one RTV card row without link status.
   This permissive reader behaviour predates the latest changes. Normal
   `load_manifest` verifies files up front, so these probes do not establish
   that current intact datasets bypass validation; the gap matters for an
   already-selected manifest, missing files after selection and contexts
   installed directly. A declared missing table/crosswalk should raise an
   integrity error. A source absent from the receipt can remain optional.
2. **External registry roots do not survive frame re-entry.** The new path
   helper honours `manifest['root']` inside `use_manifest`. However,
   `_frame_result` retains only the manifest ID in a returned DataFrame, while
   `frame_manifest` reloads that ID from the default registry. Returning a frame
   from an external-root context and passing it to another decorated entry
   point after leaving the context produced `ValueError: Dataset manifest is
   missing or invalid`. The external manifest was valid and remained present.
   Preserve the root alongside the ID, or retain a complete validated manifest
   reference. Add an end-to-end external-root re-entry test alongside the BHA
   in-context test.

The next engineering priority remains exact schemas and declared categorical
domains, followed by explicit execution/state contracts and family extraction.
The source-only cleanup makes those steps easier to validate; it should not be
represented as completing them.

Validation: **106 targeted tests passed** across `test_source_dataset.py`,
`test_bha_features.py`, `test_fe_signature_coverage.py`, `test_rtv_features.py`
and `test_featured_dataset.py` (96.08 seconds). This is a targeted result, not a
new full-suite qualification. The other agent's documented 13,293-runner frozen
comparison reports exact processed/store/state/price parity; that historical
comparison was read, not rerun here. Probe outputs and test results are under
`data/audit/fe_update_review_20261108/`. No pipeline code was changed by this
follow-up review.

The read-only quick runtime status at 10:03 on 8 November still reports the
installed store at feature signature `457c83789e6b29ab`, versus current code
`5eaf59189052d233`, changed ATR/RTV inputs and a model/store generation mismatch.
The previously owed rebuild and refit remain outstanding. No rebuild or fit was
started by this review.

## Implemented contracts — 8 November 2026

The two source-boundary gaps above are now closed, and the first feature-contract
improvements have been implemented. See the current
[feature contracts](../architecture/FEATURE_CONTRACTS.md):

- Declared source files and runner crosswalks cannot silently disappear;
  optional sources absent from a manifest retain their missing-data behaviour.
- Frames retain an external registry location and re-enter through a verified
  manifest loader using that location.
- Explicit categorical declarations replace prefix-based zero completion;
  store metadata records the dummy domain. Training requires the complete
  selected feature schema, and prediction checks saved names and numeric types.
- Private execution contexts replace process-wide counts and timeline caches.
  The engine explicitly establishes its existing post-rating day order, and
  ratings/distance stages share one bin definition.
- Append parity checks require the complete eligible schema, column order,
  dtypes and canonical runner keys before comparing values. They exposed a
  pre-existing rolling-column order difference: the reference sweep now follows
  the established seeded/persisted order, with bounded operations before EWMA.
- A session-level fixture dataset isolates module-scoped test preparation as
  well as per-test calculations.

The historical-engine comparison on accepted dataset
`1df0d286e8288f4cf5873238a8d9c8ad43f548f4814433b1adf1fd24e1edd1e8` covered
31 October–6 November 2026: **2,557 runners, 1,388 columns and 129 selected
features**. Values, column order, dtypes and serialized carried state were
exactly equal between the old and new seeded engines. This isolates the kernel
changes using the current accepted-source readers; it is not a full historical
store rebuild or a new model qualification. The intentionally changed reference
column order is covered separately by the stricter parity suite.

The repeatable comparison command is
`python -m research.feature_contracts_20261108.qualify --dataset <manifest-id>`.
It writes audit outputs only. Evidence is under
`data/audit/feature_contracts_20261108/`. The installed store and model were left
untouched and still require the normal rebuild/refit for the new feature-code
generation. The complete dependency catalog, family extraction, persisted
frequency encoders and finer artifact lineage remain later consolidation work.

Validation of this implementation:

- The full `tests/` run completed with **1,717 passed, four failed and three
  skipped**. All four failures were older test assumptions, subsequently
  corrected and rechecked: two incomplete rebuild mocks, the name-based
  duplicate-runner gate, and a recovery-report fixture invoking real publication
  with deliberately missing canonical IDs. The initial full-run log is retained
  unchanged; a second full run was not performed after these test-only fixes.
- The corrected rebuild/integrity modules passed **25 checks**, including three
  new identity-gate cases. The final source, schema, context, recovery and rebuild
  recheck passed **82 checks**, with no remaining failures in those selections.
  Feature/store/live/model tests passed in the full run.
- New modules, the qualification command and the new/corrected test files pass
  Ruff. Existing legacy files retain pre-existing lint findings; no broad lint
  cleanup was included. `git diff --check` passed.
- Quick runtime status records current code signature `d6c80a5994f8af79`, versus
  the installed store's `457c83789e6b29ab`. The installed model/store generation
  mismatch remains; no production rebuild, refit or promotion was performed.
