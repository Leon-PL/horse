# Racing Workbench

The local project app covers ingestion, preparation and stored feature inspection.
The [Features guide](WORKBENCH_FEATURES.md) describes model roles, named feature
sets, quality checks and shared-store build jobs. It uses the existing Python environment and introduces
no new runtime dependency. [Implementation plan](../architecture/INGESTION_WORKBENCH_PLAN.md).

## Start

From the repository root:

```powershell
.venv\Scripts\python.exe scripts/workbench.py
```

Open **http://127.0.0.1:8765**. Use `--port 8766` to choose another local port.
Stop the HTTP server with Ctrl+C when launched from a terminal. Closing the app
does not stop independently running collectors or workbench background jobs.

## Screens

| Screen | What it provides |
|---|---|
| Overview | Canonical counts, accepted source groups, source readiness, native worker activity and stale-heartbeat detection |
| Races | Date/search filters, runner fields, native/accepted source tables, established-identity comparisons, evidence downloads, ATR charts, prepared input preview and publication changes |
| Sources | Shared health layout for SL, RTV, ATR, Betfair, BHA, pedigree, FX and weather; contracts, missingness, identity reasons and provenance |
| Coverage | Accepted observations by source/date at the source's own grain; reference and publication-window states; click through to races |
| Features | Shared-store status, named model input sets, race values, catalog, bounded quality checks and explicit feature-build jobs |
| Jobs | Existing worker state, bounded refresh, pedigree/weather options, retained SL replay, local source publication, dataset integrity checks, race checks and ATR graceful stop/resume |
| Review queue | Bounded race/ATR findings, persistent acknowledgement/investigation states, bookmarks, notes and saved correction proposals |

The top selector binds a view to a complete accepted dataset. Native observations
are a separate layer and may be newer or unqualified. Viewing pages never runs
collection, advances publication or bootstraps an absent source. Automatic checks
refresh Overview and Jobs every 30 seconds; they do not collect data.

Sources have different grains. BHA counts are fixture reports, pedigree counts
are horses, and ATR/RTV observations are runner records. Coverage cells are counts
of accepted observations, including any retained unlinked rows; they are not a
claim that a whole day or field is complete. Source health exposes link reasons
and usable-row counts separately.

## BHA stewards

The Sources → BHA view can inspect fixture reports independently of canonical
race availability. It exposes enquiries, runner notes, fixture comments,
non-race incidents, extracted documents, native race notes and report revisions.
The race inspector uses a unique date/course/UK-local-time association for race
notes. Ambiguity leaves the notes unassigned. Fixture context never establishes
runner attribution. PDF printed-publication labels are not treated as first
availability timestamps.

Retained HTML is shown as text. Evidence downloads are provider-scoped and check
their recorded content hash. ATR charts retain each metric's native interval
labels and units; missing values do not become zero-valued chart points.

## Operations and review

Only an explicit action queues work. Background commands call the existing source
owners, including their process locks, publication delays, retry policies and
code pins. Sources already held by active collectors are excluded from another
refresh. ATR resume preserves its last recorded request delay.

Regular collection and replay ranges are limited to 31 settled days. Race search
is paginated and accepts up to 93 days; coverage accepts up to 366 days. Review
checks inspect at most 1,000 canonical races and 100 ATR queue findings per range.
These bounds are shown in the interface. They are not whole-history certification.

Notes, bookmarks, decisions, proposals and job receipts live under
`data/workbench/`. Source databases and immutable versions remain with their
existing owners. A correction proposal records the original race fingerprint,
before/after values, evidence, reason and validation findings. It does **not**
edit canonical data. Apply reviewed repairs through the existing targeted repair
owner, then publish a new accepted dataset; generic in-grid editing is not part
of this release.

Prepared-input inspection runs the existing processing pipeline on the selected
accepted race. It does not generate features or write a feature store. Categorical
codes in this small preview are local to that frame. CSV exports contain canonical
rows; JSON evidence packs include the selected dataset and available native and
accepted source records.

The server binds to loopback and validates Host, action tokens and same-origin
requests. It is a local workspace, not a remotely hosted multi-user service.

## Code and verification

`src/workbench` owns queries, the local API, review storage and background-command
dispatch. `scripts/workbench.py` is the launcher. `ui/dist` contains the buildless
browser client. Source parsing, normalisation, validation, processing and health
policies remain with their existing pipeline owners.

`tests/test_workbench.py` checks read-only browsing, immutable dataset selection,
BHA ambiguity and local-time matching, coverage grains, evidence hashes, durable
review state, correction validation, active-collector exclusion, HTTP actions and
download/path boundaries. Browser checks and real-data smoke checks complement
these fixtures. No feature build or model fitting is required to run the app.
