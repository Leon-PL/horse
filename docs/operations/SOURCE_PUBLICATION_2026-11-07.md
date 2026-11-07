# Shared source publication — 7 November 2026

Ordinary processing, feature builds and training now select immutable accepted
source versions. The collector-specific raw stores and active backfills remain
in place. [Racing Data Flow](../../Racing%20Data%20Flow.md) describes the boundary.

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

## Scope

This provides exact dataset-version lineage and stage input boundaries. It does
not claim an exact dependency graph for every derived feature column. Research
columns and weather remain available. Dependency-based partial history rebuilds
remain disabled pending parity evidence. No model is automatically promoted.
