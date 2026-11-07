# Racing Workbench: ingestion implementation plan

The first module is ingestion and preparation, ending at the input to feature
building. It runs locally against the project's existing collectors and accepted
datasets. Later project stages can add modules to the same application shell.

**Delivered, 7 November 2026:** all five delivery steps below are implemented.
The local app runs at http://127.0.0.1:8765. The
[operating guide](../operations/RACING_WORKBENCH.md) describes the screens and
first-release boundaries, including reviewed correction proposals rather than
generic canonical-data editing.

Qualification passed 19 workbench/layout checks and 24 BHA ingestion checks.
Live browser verification covered source comparisons, BHA reports and publication
changes, prepared inputs, coverage and job execution. Isolated browser fixtures
verified notes, bookmarks, proposals and escaped source text. The narrow layout
had no page overflow. A background read-only check inspected 47 real races for
4 November and returned no findings within that scope. Evidence is saved under
`data/workbench/qualification/`. Existing source collectors were preserved.

## Delivery sequence

1. Add a bounded, read-only query service over canonical results, native source
   stores, immutable publications, worker receipts and retained evidence.
2. Add a local HTTP API and responsive browser application: overview, searchable
   races, race inspection, source health, coverage, jobs and review.
3. Add background operations through the existing ingestion APIs: bounded refresh,
   retained-response replay, publication, integrity checks and ATR stop/resume.
   Opening a page must never start collection or publication.
4. Add persistent bookmarks/notes, issue triage, correction proposals, evidence
   exports, publication comparisons and prepared-data inspection.
5. Verify with isolated source fixtures, HTTP tests, real-data smoke checks and
   browser checks. Document launch commands and operational limits.

## Data and interface rules

- One source layout and health vocabulary. Separate worker activity, collection
  outcome, freshness, coverage, identity quality and accepted publication.
- Choose a complete dataset manifest explicitly. Browsing cannot bootstrap or
  advance it. Show native observations separately from accepted observations.
- Keep source grains and availability assumptions visible. Unknown, inapplicable,
  awaiting publication, unmatched and rejected are different states.
- BHA is a first-class source. Show fixture reports, PDF extraction, runner notes,
  enquiries, non-race incidents, revisions and evidence. Native race notes can be
  associated only by a unique date/course/local-time match. Fixture context is
  labelled as such; there is no inferred runner-level BHA attribution.
- Use existing normalisers, validation, processing and collector locks. Never
  duplicate scraping, feature definitions or source-health policies in the UI.
- Queue long operations independently of page requests. Record durable job
  receipts and logs, validate operation parameters and prevent duplicate jobs.
- Corrections require evidence, a before/after proposal and validation. Keep an
  audit trail and publish a new version; immutable publications are not edited.
- Local server binds to loopback. Actions require a session token and same-origin
  requests. Retained HTML and report text are displayed as text, not executed.
- No feature build, model fitting or promotion is part of the ingestion UI.

## Acceptance checks

- A race is searchable by date, course, name, horse and canonical ID.
- Native and accepted values, provider IDs, link/timing receipts and source
  evidence can be inspected without loading the full history into each page.
- ATR arrays have unit-labelled charts; BHA fixture and race associations are
  explicit; an ambiguous BHA association remains unassigned.
- Source health and coverage include all eight groups: SL, RTV, ATR, Betfair, FX,
  pedigree, weather and BHA. Optional gaps do not masquerade as mandatory failures.
- Missing stores and stale worker receipts render as useful states, not crashes.
- Background operations use current collector code and preserve active-worker
  locks and provider backoff. UI refreshes do not launch duplicate workers.
- Notes and review decisions survive app restarts. Exports include selected
  dataset and source provenance. Jobs and repairs remain inspectable.
- Browser and API checks exercise real interactions, empty states, errors,
  evidence escaping and responsive layout.

## Implementation choice

Use the installed Python runtime, SQLite/PyArrow readers and a buildless browser
client. Keep the application/query layer in `src/workbench`, the operating command
in `scripts`, and static UI assets in `ui/dist`. This avoids restoring the removed
Streamlit dependency and introduces no new runtime package requirement.
