# Feature execution and schema contracts

Current implementation, 8 November 2026. The shared entry points remain
`store_refresh.get_training_dataset` for training and
`live_prediction.features_for_live_cards` for live cards. `FeatureStore` and
`store_build` retain carried state, bounded chunking and atomic five-file
publication.

## Accepted source inputs

Source readers require a selected dataset. An optional table absent from that
dataset supplies no observations. A table declared in its receipt must exist;
`source_path` and `read_table` raise `SourceIntegrityError` if its file disappears.
A linked read also requires a declared, present crosswalk with complete row
coverage, link-status fields and canonical IDs for matched rows. `model_table`
therefore cannot accept rows merely because a crosswalk was lost.

Manifest loading verifies content hashes. The presence and crosswalk checks also
apply inside an already-selected context, including after initial selection.
They do not fall back to collector caches or republish sources.

Processed and featured frames retain their dataset ID. When a manifest declares
an external registry root, frames retain that root too; a later entry point
reloads and verifies the manifest from that location. Normal registry references
remain relative to the configured data directory. `processing_inputs` and
`dataset_build` share this tagging behaviour.

## Feature and categorical schemas

`src/feature_schema.py` owns the shared categorical fields, fixed domains and
required numeric-column checks. Preparation uses the same Flat/Hurdle/Chase
and Male/Female domains as schema completion.

Store metadata records `categorical_columns`, derived from prepared categorical
values and the explicit season domain. Live calculation completes only missing
columns that are both declared dummies and present in the training schema.
A name starting with `going_`, for example, has no special zero interpretation.
An older reference store without this metadata can derive the declarations from
its prepared categorical columns. Normal live readiness still enforces code and
store compatibility.

With `FEATURE_ALLOWLIST` configured, training requires every named input to be
present, unique, numeric and eligible as a feature. It preserves the established
frame column order. Candidate discovery explicitly uses
`get_feature_columns(..., apply_allowlist=False)` and retains its exclusions.
Prediction validates the saved model's exact required names and numeric types;
existing NaN values are preserved for the model's learned missing branches.

## Working state and temporal order

`src/feature_context.py` owns a `FeatureContext` with a working count-state
dictionary and timeline cache. Each `counts_state_scope` creates a private
context; nested scopes restore the outer context even after exceptions.
Concurrent threads have separate contexts. Callers must still provide distinct
mutable working dictionaries for independent executions. Store-backed live
requests load private state, discard advances and never publish it.

The existing downstream day-level history order is now a named engine step,
`_use_day_order_after_ratings`, in both inline and seeded-rating modes. Its
behaviour no longer depends on ratings deleting a temporary column. It preserves
the established outputs; changing the time policy is a separate semantic change.

Ratings and distance features use `feature_semantics.distance_category` for the
same historical bins and unknown-distance fallback. BHA retains the agreed
`race_plus_48h` policy. Source-specific history-window definitions are preserved.

## Validation and compatibility

Append parity checks compare the whole eligible feature schema and dtypes
before numeric values and NaN masks, aligning by canonical race and horse IDs.
Missing append columns cannot escape through an intersection.
The reference rolling sweep now emits bounded operations before EWMA, matching
the established persisted/seeded column order. This corrects a column-order
difference exposed by the stronger check without changing computed values.

The new feature-schema and context modules are registered in the feature-code
signature. Existing stores need the normal rebuild and matching model refit
before use with the new generation. No model is automatically promoted.

The [review and follow-up](../archive/research/FEATURE_ENGINEERING_CONSOLIDATION_REVIEW_2026-11-07.md)
record the original findings. A complete column-level dependency catalog,
family extraction, persisted frequency encoders and finer artifact lineage
remain further consolidation work. The changes here establish checked source,
schema and execution boundaries for that work.
