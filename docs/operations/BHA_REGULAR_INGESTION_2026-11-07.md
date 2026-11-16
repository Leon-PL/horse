# BHA reports in regular ingestion — 7 November 2026

> **Current state (16 November 2026):** no feature reads BHA data. Research 109
> removed the BHA feature stage on 15 November
> ([FEATURE_PRUNING_109](../research/FEATURE_PRUNING_109.md)), and from 16 November
> BHA is no longer a default sync source. The collector and its published data
> are kept; run it explicitly with `--sources bha`. The rest of this page
> describes the collector as built on 7 November.

BHA is now a default source in `python -m src.ingestion --sync 14`. It collects
the requested past days, checks recent amendments and publishes full reports,
documents and race notes through the same immutable source boundary as the other
providers. Its source key is `bha`; it is no longer a standalone CSV-only tool.

**Feature derivation (removed 15 November by research 109):** 161 prior-history numeric candidates are now wired
into ordinary feature building for normal feature selection. No separate models
were fitted. Read [the definitions and timing policy](../archive/research/BHA_FEATURE_CANDIDATES_2026-11-07.md);
the latest user decision uses an explicit 48-hour feature-availability assumption.
The collector still records actual capture times; it does not rewrite them as
historical publication times.

## Information collected

The public results-page feed and its full PDFs have different coverage. The live
Aintree comparison found going/GoingStick readings, rail information, non-runner
reasons and suspension details in the PDF in addition to the web enquiry text.
Keeping only the old report summary and a PDF link missed that context.

The collector now retains:

- Complete API response bytes and each original report object, including unknown
  fields added by the provider. Named fields expose enquiry text/HTML, runner
  notes, fixture comments, non-race incidents and abandonment reasons.
- Complete PDF bytes, extracted full text, page counts and text-quality markers.
  Going, rails, non-runners, suspensions and steward names have separate text
  columns when the document contains those sections.
- Race-by-race notes, with BHA fixture/year, race number and advertised local
  time. "Nothing to report" entries remain explicit source observations.
- Native identities, source URLs, content hashes, observation times, revisions
  and per-day completion/failure receipts. A report disappearing from a later
  complete response is retained and marked absent rather than silently deleted.

The code checks PDF date/course against the fixture before accepting its text.
Unexpected layouts, unreadable/scanned documents and unavailable PDFs remain
visible as pending/retry work; their received bytes are retained. Extracted
table text is not represented as already validated horse-level structured data.

The source adapter publishes `bha/reports`, `bha/documents` and `bha/race_notes`.
Keys remain BHA-native: year/fixture, with race number for notes. It does not
guess canonical horse IDs from capitalised names. These are collected inputs;
new model features and their evaluation are a separate stage.

## Reliability and speed

One dated API lookup normally retrieves all that day's fixtures; pagination is
validated against page numbers, total rows, unique fixture keys and the requested
date. No date is marked complete until the whole response is validated. Empty
days also receive durable receipts.

SQLite commits each complete day and each accepted document. A changed report
becomes a revision of the same fixture rather than an extra current row. It also
queues a document recheck. Recent days/documents are checked again after 24 hours
within a seven-day amendment window. Older dates can be explicitly rechecked.
The original `data/raw/bha_stewards_reports.csv` is preserved unchanged and imported
once; it is not treated as proof that every old date was completely collected.

Requests share a connection pool and a one-second minimum start gap, including
document requests. Authentication is bootstrapped once per batch. API tokens are
never sent to document hosts or stored in source responses. Genuine limits and
Retry-After create a persisted provider cooldown; the drain worker waits rather
than repeatedly calling the site. Isolated item failures do not halt other dates.

The old archive path repeatedly read, keyed, concatenated and rewrote a 60.6 MB
CSV. Normal refreshes now update only changed records. Legacy normalisation also
avoids parsing cached text twice and replaces repeated scalar date inference.
A 300-report retained-source comparison measured 0.838 seconds before and
0.068 seconds after: **12.3 times faster for that normalisation step**. It verified
that every original source-record field was retained. This is not a whole-scraper
or network throughput claim.

A repeat native refresh of the completed 6 November day took **0.123 seconds and
made zero HTTP requests**. This excludes shared publication of other providers.
The first real coordinated run imported 8,463 old reports, collected four new
reports and four full PDFs, and exposed 29 race-note blocks. BHA collection/import
took 138.5 seconds before the normalisation optimisation. The entire first
coordinated run took 733.9 seconds because it also adopted the other source stores;
that total must not be attributed to the BHA scraper alone.

## History and timing

The saved archive originally ended on 20 June 2026. The historical catch-up uses
the same collector and queue as normal ingestion. It fills missing report days
and downloads the fuller documents for the older retained reports as well.
Progress is in `data/bha/status.json`; a first accepted recent day is not proof
that the entire historical queue is complete.

Reports are post-race observations. Accepted tables record when the exact content
was observed; they do not assign historical availability from today's version of
an old document. Printed PDF "Published" footers can be generated on download and
are retained as labels, not used as the first publication time. The raw history
now supports linked history features using the explicitly configured two-day
assumption described above. Actual historical publication/revision times remain
unverified. The old BHA/IHRB event-tag file is preserved, not extended with
unvalidated text matches by this collector.

This deployment covers BHA stewards' reports and their attached context. It is
not a claim to mirror every BHA horse profile, rating, racecard or website page.

## Commands

```powershell
# Included in the usual source plan:
.venv\Scripts\python.exe -m src.ingestion --sync 14
# Check one source through that same coordinator:
.venv\Scripts\python.exe -m src.ingestion --sync 1 --sources bha
# Historical reports plus fuller PDFs; bounded batches share a durable queue:
.venv\Scripts\python.exe -u -m src.ingestion.bha_stewards --start 2020-09-15 --end 2026-11-06 --drain --publish
# Read progress without fetching:
.venv\Scripts\python.exe -m src.ingestion.bha_stewards --status
```

Defaults are 31 API dates and 64 documents per batch. Normal source refreshes are
bounded; `--drain` continues batches. `--publish` adopts accepted sources after the
drain finishes. A native observation can therefore precede its next accepted
dataset publication. Regular coordinated ingestion also adopts current progress.

Do not start a second collector. Competing invocations report `busy`. A `STOP`
file in `data/bha/` stops between requests, including during cooldown. Remove it
only when intentionally resuming. The worker stops if either collection module
changes; review and restart it rather than collecting under mixed code versions.
Do not delete SQLite sidecars or lock files.

## Evidence

`data/audit/bha_regular_ingestion_20261107/` retains the initial working-source
snapshots, public section definitions, live JSON/PDF evidence, historical PDF
checks (2020 and 2023), normalisation timings, zero-request cached refresh,
coordinated-publication receipt and test XML. The wider 109-test source suite
passed, followed by focused checks of the final BHA runtime. The launch and
deployment receipts record exact code hashes and actual historical progress.

The final focused runtime suite passed 24 tests. The historical worker launched
at 18:22 UTC with one-second pacing. A subsequent check found 8,807 retained
reports and zero item failures. The first updated accepted snapshot contained
8,649 reports, 68 full documents and 491 race-note blocks; later collection is
tracked separately until publication. The deployment preserved every other
selected source version and the original report CSV hash. Use live status for
subsequent totals rather than treating these checkpoints as completion.

The [initial audit](../archive/operations/BHA_STEWARDS_SOURCE_AUDIT_2026-11-07.md) describes the state
before this integration, including the scope limits of the earlier negative
feature experiment.
