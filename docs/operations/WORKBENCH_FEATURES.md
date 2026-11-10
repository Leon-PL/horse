# Workbench Features

Open `http://127.0.0.1:8765/#features/overview`. Features sits alongside ingestion
in Racing Workbench and is linked from each race's **Features** tab.

## Model roles and feature sets

Computed values live in the shared feature store. Named feature sets select
columns from those values. Model roles reference one or more named sets, and a
set can be shared by several roles. This supports future general and specialist
submodels without treating the page's selection as one global model input list.

`feature_sets/registry.json` declares these relationships. Initially it contains
the existing general model selection. To register another selection, add a text
file under `feature_sets/`, an entry in `feature_sets`, and its model-role
references. IDs are unique simple names; paths must remain inside that folder.
Duplicate names, unknown set references and excluded labels/identifiers are
rejected. Example additions:

```json
{
  "feature_sets": [
    {"id": "chase-specialist", "label": "Chase specialist", "file": "chase_selected.txt"}
  ],
  "model_roles": [
    {"id": "chase", "label": "Chase model", "scope": "Chase races", "feature_sets": ["chase-specialist"]}
  ]
}
```

These entries are configuration references, not evidence of fitted models.
`scope` is a descriptive label in this release. Selecting a set in the page
filters inspection and quality checks; it never changes `FEATURE_ALLOWLIST`,
fits a specialist model or changes the deployment pointer. The saved model's
definition-signature check is shown separately and does not establish matching
source revisions, fit dates or complete model-input membership.

## Views

- **Overview:** store age/coverage, definition and source-dataset compatibility,
  named selections, model-role relationships and feature-family navigation.
- **Race inspector:** stored values by canonical race/horse IDs, 24 feature
  columns per page and at most 300 runners. Cells open the exact stored value,
  its definition where documented, family support and prior-start context.
  Missing values, measured zeros, absent columns and infinite values are kept
  distinct. CSV exports contain the shown columns plus build, definition,
  source-dataset and selection references.
- **Feature catalog:** searches the selected set or all stored candidates and
  available BHA definitions; shows dtype, availability in the store, named-set
  membership and configured model roles. Family classification is navigation
  metadata. Individual formulas/units not yet documented are explicitly marked
  as pending; the page does not invent calculation explanations.
- **Quality checks:** explicit bounded scans of up to 5,000 stored runners,
  256 columns and 31 days. Reports missing/constant/infinite values, observed
  ranges, missing selected columns and duplicate canonical runner keys.
  Constants and nulls require source/applicability review. A completed scan is
  not a leakage certificate or a model-readiness decision.
- **Builds & validation:** queues shared-store updates or full rebuilds against
  a fixed accepted dataset, and saves bounded checks as Jobs. The existing
  append/batched-build owners make the compatibility decision and retain atomic
  store publication. Stage progress and logs are available through Jobs.

The stored-race picker returns up to 50 races from the first 10,000 runners in
the selected range. Narrow the dates when necessary. Prior-start context is
bounded to 1,000 queried starts and ten displayed starts, excludes the inspected
race and later starts, and is **not an exact contributor trace**. Evidence links
use the stored dataset when recorded; legacy stores without that dataset cannot
claim source lineage. Current BHA definitions expose the agreed 48-hour policy
and its publication/amendment limitations separately from stored values.

## Read and action boundaries

GET requests do not build features, collect or publish sources, load model
objects, create a store, or recover pending store commits. Inspection uses the
existing store lock and reports interrupted commits as unavailable. A lightweight
build reference distinguishes artifact changes even when the feature-definition
signature is unchanged; it is not a full content-integrity hash.

Builds are explicit actions in the existing job manager. They use the selected
immutable dataset, the common processing path, carried-state append or the
bounded atomic builder. They do not fit or promote a model. Checks record the
selected feature-set signature and refuse to run a silently changed selection.
Collection workers remain independent. A conflicting store writer is rejected
by its existing lock.

Historical recalculation using checkpoints, exact contributor traces for every
feature, and automatic comparisons between complete retained feature generations
are later extensions. This release reads stored values; it never applies today's
accumulated state to an old race. Saved exports provide bounded comparisons now.

Tests cover many-to-many role/set membership, selection isolation, legacy and
missing stores, no writes on browsing, pending-commit handling, source/history
cutoffs, missing/zero/infinite values, scan bounds, schema gaps, metadata
compatibility, API/static routes and existing atomic build dispatch.
