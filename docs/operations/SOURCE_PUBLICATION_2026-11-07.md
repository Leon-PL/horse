# Shared source publication — 7 November 2026

Ordinary processing, feature builds and training now select immutable accepted
source versions. The collector-specific raw stores and active backfills remain
in place. [Racing Data Flow](../archive/Racing%20Data%20Flow.md) describes the boundary.

## Implementation

- `source_contracts.py`: grain/key/type contracts, namespaced identity receipts,
  event/observation/availability metadata and separate coverage measures.
- `source_dataset.py`: atomic content-addressed source versions, verified replay,
  dataset manifests, scoped readers and process-pool manifest propagation.
- `ingestion/publication.py`: adapters for canonical results, pedigree, weather,
  ATR, RTV, FX and Betfair; per-provider adoption failures retain accepted data.
- `source_health.py`: explicit consumer requirements and publication delays,
  independent of the last collection attempt's state.
- `source_prices.py`: manifest-selected prices and persisted-ID attachment shared
  by model fitting and CLV; no late fallback to a newly appearing native cache.

Results and pedigree are captured in one database read transaction. The native
`data/raw/race_results.parquet` export remains available for interchange and
explicit research reads. Ordinary builds use the selected accepted results.

ATR promotion reads the worker's SQLite checkpoint in a read transaction. It
requires completed data jobs, current collection versions, terminal receipts for
all five tabs, valid native IDs, matching canonical identities and elapsed
publication delay. Invalid units receive reasons. Previous promotions survive
reopened worker jobs, and unrepresented legacy runners are retained.

No pinned ATR or Betfair collector file was edited by this implementation.
Publication does not overwrite the worker database, original ATR cache or raw
evidence. Provider failures are isolated from independent source adoption.

## Feature behaviour

ATR history is now selected by availability at each target decision. Missing
starts occupy their window slots; absent comments remain unknown. New support
and age columns expose stale but populated history. The unselected legacy speed
composite uses field-relative scaling; stride/course interaction uses course
circumference in miles, avoiding batch-wide scaling.

RTV readers honour saved link rejections. Pre-race snapshots retain their
recorded capture cutoff. Upcoming cards supply canonical IDs for the small live
snapshot stream. Serving reuses the feature store's immutable historical sources
and publishes only that small stream; it does not rerun historical adoption.

Source fingerprints include identity and timing sidecars as well as raw values.
Existing conservative full-rebuild behaviour remains. Training refresh records
its rebuild reason and manifest IDs in `data/sources/builds.json`.

The implementation changes feature definitions. The existing store/model are
not made compatible by publishing observations. Model fitting, evaluation and
selection remain explicit operations.

## Operations

```powershell
.venv\Scripts\python.exe -m src.ingestion --publish-sources
.venv\Scripts\python.exe -m src.ingestion --source-status
.venv\Scripts\python.exe scripts/project_status.py --json --quick
```

The first command performs local adoption and ATR promotion without HTTP. Normal
`--sync` also adopts accepted stores after collection. Builds and training select
the last complete dataset commit; they do not repeat adoption. A first installation
bootstraps once. Matching unchanged input and identity fingerprints reuse the
accepted source version. Publication is substantial work on the full dataset;
it is separate from both ordinary dataset selection and the small serving path.

The dataset pointer advances atomically after all source versions are selected.
An adoption lock prevents competing full publications, while readers continue
using the previous committed dataset. Failed new candidates retain observations
but requalify their links against current canonical fields. Frames carry only a
manifest ID to avoid repeatedly deep-copying large receipts in pandas.

Exact replay into an isolated candidate store:

```powershell
.venv\Scripts\python.exe -m scripts.build_feature_store_batched --dataset-manifest DATASET_ID --store-root data/processed/candidate_store
```

Keep `data/sources`: lean training runs refer to its manifests and versions.
Removing old versions requires checking retained run and store references first.

## Qualification

The first real local publication retained 757,580 canonical runners, 73,575
pedigrees and 80,536 race-level weather rows. It promoted 5,794 ATR races and
published 232,703 combined ATR observations. All seven logical source groups
were present in manifest
`2ae89f356bf1f945fa096ad37cad95bfc0e755bdd1628c1a4085f8c4cfa6fb98`.
That first capture took 460.87 seconds; it predates the final policy/sidecar
qualification, so use the final qualification receipt for the installed version.

Regression coverage includes interrupted and invalid publication, idempotence,
hash-verified replay, immutable absence, spawned-process version selection,
namesakes and conflicting IDs, repeated snapshots, publication delays, ATR
promotion/retention, support/age, chunk parity, source health, price attachment,
training orchestration and transactional feature-store updates.

The full source/training/store regression run passed **237 tests**. Four further
cases cover unversioned direct appends, invalid CLV prices, quarantined name-only
price targets and canonical recovery of legacy evaluation rows. The final
source/pricing/training/deployment subset passed **57 tests**. Targeted lint and
feature-signature coverage checks also pass. These counts overlap; there are
**241 distinct targeted cases** in the combined checks.

Real-data qualification used committed dataset
`5fb184f9f1fc61ca470bd8aeac50af17c156c008d874a38aad5a7068fefaa786`.
Selecting it took **3.16 seconds**, with publication explicitly disabled in the
check. One-chunk and two-chunk builds matched all **129 production inputs** for
**1,464 runners** from 1–4 November, at relative/absolute tolerance 1e-6. The
qualified feature signature is `5d6eb8e0082a4d3a`. Frames carry manifest IDs;
the warmed two-chunk build took about ten seconds in the preceding check.

The cold full-source publication took 520.67 seconds in the timed revalidation;
a repeat with the ATR worker still adding observations took 239.56 seconds.
This is why normal dataset selection and serving do not repeat adoption.

Command results, separate candidate stores and real-data qualification are in
`data/audit/source_publication_20261107/`. The installed full store was not rebuilt
and no real-data model was fitted or promoted by this change. The readiness
instrument reports that refresh/rebuild/refitting are still required.

Concurrent app-removal and configuration edits were preserved. The feature-code
signature was checked again against the current workspace and still matches the
qualified signature above.

## Fallback paths removed

Later on 7 November the no-dataset branches were removed, so processing, feature
building, store refresh, model-fitting prices and CLV read sources only through
the selected dataset.

- `source_dataset.source_path`, `read_table` and `model_table` take no fallback
  path. With no dataset selected they raise, and `require_manifest()` says how
  to select one.
- An entry point given a frame without a dataset ID uses the last committed
  dataset (`committed_dataset()`) and never publishes. With no committed dataset
  it stops. Called without a frame, selection is unchanged.
- ATR: the name-joined `merge_atr_sectionals`, its batch-wide scores and
  `atr_merge_scope` are gone. `published_atr_rolling` is the only definition.
- RTV: pre-race snapshots join by saved race and horse IDs only, and unlinked
  snapshot rows raise. The collector's own files are located by
  `ingestion.rtv_storage.native_path`, for publication only.
- FX: `ingestion.fx` owns the reference file paths and no longer imports
  feature code.
- Prices: `source_prices` no longer falls back to `src.betfair`, and
  `compute_clv` has no `bsp_path`. Bets with only names are resolved to
  canonical IDs first.
- Stores: every stamp records the dataset. A store without one is rebuilt
  (`store_predates_source_publication`), and live prediction refuses it. The
  installed store (`457c83789e6b29ab`) predates publication, so it needs the
  rebuild already owed before it can serve again.
- `project_status.py` and `select_features.py` compute fingerprints under the
  committed dataset. With none, the status is BAD and names the publish command.

Two dtype faults in the published path surfaced once it became the only path,
and both are fixed: RTV snapshot links read back as pandas strings, and ATR
history for a dataset whose ATR rows all failed to link. Neither changes the
qualified output below.

Qualification used the same frozen dataset
`5fb184f9f1fc61ca470bd8aeac50af17c156c008d874a38aad5a7068fefaa786` and window
(1 October to 4 November 2026, **13,293 runners**). The code before and after
was run from separate copies against a private copy of the dataset. These were
exactly equal, including column order and dtypes: the processed frame (97
columns), the built store (13,293 × 1,233) and its processed table, the carried
counts and ratings state, BSP attached by ID and by name, and the CLV summary.
The data, ATR, RTV and FX signatures and the recorded manifest were also equal.
Only `fe_signature()` moved, from `5d6eb8e0082a4d3a` to `5f1b9689bce1820e`. The
comparison's own copies report other values, because configured paths are part
of that hash.

Tests select an explicit dataset (`tests/conftest.py`): the committed FX
reference by default, or fixture sources published by the `publish_dataset` and
`publish_prices` fixtures. No test reads a collector cache or the real data
folder through the source layer. The full suite gave 2,011 passed, 10 failed
and 3 skipped, against 2,013 / 7 / 3 before. Seven failures are unchanged. Two
ATR efficiency tests that reuse the ATR fixture were then ported and pass alone.
The last one, `test_independent_feature_benchmark`, drives
`scripts/audit_atr_invariance.py`, which patches the removed merge. That
script, its two siblings and the test were deleted in the November 2026 cleanup.

## Scope

This provides exact dataset-version lineage and stage input boundaries. It does
not claim an exact dependency graph for every derived feature column. Research
columns and weather remain available. Dependency-based partial history rebuilds
remain disabled pending parity evidence. No model is automatically promoted.
