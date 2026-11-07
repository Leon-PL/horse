# BHA stewards: retained source, missing active ingestion — 7 November 2026

**Subsequent deployment:** BHA is now in regular ingestion, with retained full
PDFs, durable revisions and accepted-source tables. See the
[current operating guide](BHA_REGULAR_INGESTION_2026-11-07.md). The findings below
describe the earlier state that prompted the repair.

BHA stewards' reports exist in this project, but the normal source refresh does
not collect them. They have no accepted source adapter, dataset-manifest entry or
current feature consumer. Calling this only an "unwired collector" obscures both
the stale archive and the breadth of the source.

## Verified state

At 17:24 UTC, a bounded live check of the public website's report feed returned
HTTP 200 and all four reports for 6 November 2026, on one page. The existing
collector can obtain the public frontend token dynamically; no account setup or
new access workaround was needed. The audit retains the JSON response separately
from the existing archive, without retaining or printing the authorization token.

| Item | Observed state |
|---|---|
| Collector | `src/ingestion/bha_stewards.py`; standalone fetch/save/backfill commands |
| Normal refresh | Absent from `pipeline.DEFAULT_SOURCES` and source dispatch |
| Accepted source publication | No BHA/stewards contract or publication adapter |
| Raw report archive | `data/raw/bha_stewards_reports.csv`; 8,463 fixture reports, 15 September 2020–20 June 2026 |
| Derived event archive | `data/processed/stewards_events.parquet`; 74,870 BHA events, ending 20 June 2026; another 6,675 rows are IHRB |
| Existing BHA event matching | 74,448 marked fixture-matched; 422 unmatched. This is not a fresh audit of runner-level linkage |
| Current feature consumption | None found in production source/feature code |

The raw dates use mixed string formats. Explicit mixed-format parsing recovers
all dates; a default single-format parse incorrectly makes most rows appear
undated. There are no duplicate fixture-ID/year rows in this archive. Neither
archive was changed by this audit.

## Information worth assessing

The existing BHA-derived event archive includes 2,277 lameness tags, 1,667 bleeding
tags, 883 breathing tags and 4,848 lost-shoe tags. It also records hanging,
behaviour, jumping incidents, equipment permissions and interference. These are
historical parser outputs, not newly revalidated medical findings or proof of
predictive value. The current live sample also contains veterinary references
and incident explanations.

The [official report page](https://www.britishhorseracing.com/racing/stewards-reports/)
distinguishes enquiries/reports, runner notes (previously "Why they Ran Badly"),
fixture notes and non-race-related incidents. A source audit should inspect all
of these sections, rather than equating the source with trouble-in-running tags.

The project ledger's [section 7](../NEGATIVE_RESULTS.md) records an earlier
negative experiment with derived trouble/unlucky features. That experiment was
not rerun here. It does not demonstrate that every report section, veterinary
explanation, parsing method or future feature is redundant. The ledger also
records serious IHRB identity/coverage defects, so that evidence must not be
generalised to a complete, correctly linked Irish source.

## Collector gaps

- It normalises fixture-level reports but does not produce validated canonical
  race/runner links. The old derived event file does not provide an active path
  from new reports to current canonical identities.
- Normal collection keeps selected fields rather than a retained full JSON
  response with observation time and revision history. PDF links are stored;
  the PDFs are not downloaded or checked for additional content.
- The separate runner-notes fields are empty in all 8,463 archived rows. The
  splitter recognises only one legacy delimiter. The full enquiry HTML/text is
  retained, so empty separated fields alone do not prove missing source text.
  Completeness against the rendered report and PDF needs its own check.
- A date is considered done if any existing report row has that date. This can
  skip partial fixtures or later amendments. Successful empty dates have no
  durable completion receipt and can be repeatedly downloaded.
- New text is part of the deduplication key, so an amended report can become a
  second row rather than an identified revision of the same fixture.
- The standalone writer replaces CSV/Parquet directly, without the normal
  atomic publication path. Long cooldowns can also outlast its short retry loop.

## Correct place in the pipeline

Treat BHA as a first-class entry in source audits, with its collection status
shown explicitly even when no production feature uses it. Restoring collection
should retain full source responses and amendments first, then publish reports
and independently validated runner events under the shared source contracts.
It needs a durable fixture/date ledger, paced requests, visible failures,
revision rechecks and atomic outputs before a broad backfill.

Only reports available before the prediction being evaluated can become inputs.
Post-race explanations may describe a previous performance; they cannot explain
the same race to a pre-race model. Historical report content without an original
publication timestamp does not establish its exact availability time.

This audit does **not** enable a new default source, start a backfill, rebuild
features or alter a model. Evidence, coverage summaries and the successful live
probe are in `data/audit/bha_source_review_20261107/`. There were no runtime code
changes requiring regression tests.
