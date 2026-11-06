# Stage 1 follow-up: recovery, performance and identity review

**5 November follow-up:** the remaining identity case is repaired. The
[migration report](STAGE1_IDENTITY_2026-11-05.md) records preserved source rows,
the next-day refresh and compatible-history verification. Counts below retain
the earlier recovery snapshot.

This follows the [pipeline audit](STAGE1_DATA_PIPELINE_AUDIT_2026-11-04.md)
and [first implementation](STAGE1_IMPLEMENTATION_2026-11-04.md). It exercises
the collectors against the installed data and records the failures that only
became apparent during a real catch-up.

**Subsequent historical recovery:** [the backlog follow-up](STAGE1_BACKLOG_2026-11-04.md)
resolved 3,456 of the 3,457 older jobs with source evidence. One identity-review
case remains; the table below records the state before that follow-up.

## Installed source outcome

| Source | Verified outcome | Remaining limitation |
|---|---|---|
| Sporting Life | 757,083 runners / 80,489 races through 3 November; 1,774 runner records added; no pending jobs or unverified listings in the recent window | 3,457 older pending jobs, including the identity issue below |
| Racing TV | History generation published through 3 November; 755,636 run observations, 755,489 card observations and 714,750 text-score rows | These counts describe different observation tables; they do not certify a complete match for every canonical runner |
| Public BSP | 2,532 rows added, bringing the cache to 818,601 runners through 3 November; all 510 canonical fields in 21 October–3 November have complete price matches | Historical pre-off trades still depend on the archive source |
| Currency references | Latest observation 4 November | Dated references retain their existing conversion contract |
| ATR | Existing 232,138-row cache preserved; recovery queue retained from 21 June | Provider refused requests; 3,934 eligible races remain pending through 3 November, with 1,266 races at excluded venues outside that queue |

The overall source status is **partial** because ATR remains incomplete.
An up-to-date maximum result date does not establish complete historical data.

The initial catch-up used the earlier serial coordinator and took 1,159.16
seconds. Its BSP publication failed on mixed string/integer runner IDs, and
ATR exhausted five escalating retries without fetching a race. Both failures
were retained in the report. The repaired BSP merge subsequently succeeded.

## Changes made after the live catch-up

- Canonical Sporting Life collection runs first. FX, BSP, RTV and ATR then run
  independently with a bounded worker pool. Existing provider request limits
  remain in place. Status is saved after each source event, and app progress
  callbacks run on the calling thread.
- Sporting Life checks the source page's own race ID and date against the
  requested URL before accepting either results or a terminal race state.
  The requested date can no longer conceal a different source date.
- The BSP merge converts archive/public runner IDs to nullable integers before
  publication and uses `bf_horse_key` consistently with the downstream join.
  Known prices and all timed-price columns survive the merge.
- BSP refresh receipts fingerprint the raw files, accepted price cache,
  canonical fields, requested range and relevant parsing/matching code. An
  unchanged refresh reuses its validated coverage without merging the full
  archive again. Content changes invalidate the receipt even when a file's
  size and modification time are preserved.
- Routine ATR refreshes make one retry after a blocked request, then persist
  a one-hour retry time. Subsequent app refreshes retain the queue and return
  a partial result during that cooldown. The standalone historical backfill's
  longer retry policy remains available. Blocked responses never become
  confirmed no-data entries.
- ATR recovery batches date parsing and constructs its resolved-race lookup
  once. The same shared track/time normalisers still define the join.
- Readiness requires a fresh report covering all five normal sources through
  yesterday. A successful single-source or historical-range refresh cannot
  masquerade as current all-source readiness.

## Measured unchanged-refresh cost

| Installed-data operation | Before / first validation | After / reuse |
|---|---:|---:|
| BSP validation and coverage | 44.83 s to validate and establish a receipt | 0.295 s to reuse it |
| ATR cached/cooldown check, with profiling enabled in both measurements | 83.80 s | 4.15 s |

The ATR profile attributed 80.57 seconds to 32,089 scalar race-key calls,
including 76.93 seconds in repeated date parsing. The revised profile has
two batch key-building calls. Both measurements returned the same 3,934
pending races. The source cache stayed byte-for-byte identical to the backup.

These local measurements explicitly forbade HTTP and preserved the BSP file's
hash and modification time. They are diagnostic samples with other work on
the machine, rather than repeated isolated benchmarks. They measure cached
work, not provider throughput or a full historical rebuild.

A subsequent recent all-source check took **11.38 seconds**, including seven
Sporting Life listings. It fetched no result pages, reused RTV and BSP, checked
FX and reported ATR's cooldown. Its outcome remained partial. This is not a
comparison with the initial catch-up, which downloaded and materialised new data.

## Archive preservation

Verification compares every original row and value, plus canonical identities:

- All 816,069 original price-cache rows remain present.
- All 815,973 previously valid BSP values are unchanged.
- All six `ltp_*` columns are unchanged, including missing values.
- 109,031 stored abbreviated course keys were expanded through existing
  Betfair aliases, for example `plump` to `plumpton`. Raw course names and
  original prices remain unchanged.

The verifier initially compared stored alias keys directly and flagged those
normalisations. It was corrected to validate canonical identities and compare
all original non-key values as well; both checks pass.

## Historical backlog and the identity defect

The source sample confirmed an Arabian exclusion and old dormant races, while
a 404 remained unresolved. A bounded shared refresh of 30 September–8 October
closed **15** old jobs with source-confirmed terminal reasons. It changed no
canonical facts and did not rewrite the CSV.

The audit's 14-of-15 field, race **688054**, is a different problem. Its current
Sporting Life payload contains these two starters:

| Source horse ID | Name | Source placing |
|---|---|---:|
| 1040489 | Sierra Nevada | 1 |
| 1038101 | Sierra Nevada | 8 |

They also have different source birth dates. The database currently enforces
`UNIQUE(race_id, horse_name)`, so it cannot represent both source identities
under that display name. The validator correctly rejects the proposed field
as duplicate runner names. Fetching the same page again does not resolve it.

The accepted 14 runners were preserved and this field's ledger entry was
marked pending with the observed validation reason. The factual database
revision remains **1816**. Closing 15 terminal jobs and exposing this previously
closed field leaves **3,457** historical pending jobs.

A read-only identity census also found **105 exact names associated with
multiple source horse IDs**, across 1,904 stored runner rows; three names have
three IDs. These are review candidates. Each needs provenance checks before
declaring a merge erroneous.

**Priority follow-up:** make source horse identity, with its source namespace,
the canonical identity; keep display names and aliases separately. Review
name-based historical feature grouping and cross-source joins at the same
time. A database constraint change alone would leave later stages combining
ambiguous histories. This needs a coordinated identity migration and feature
qualification, rather than an invented name suffix or blanket reinsertion.

## Which work still requires recovery or a later stage

| Information / issue | Appropriate route |
|---|---|
| ATR's June–November gap | Resume the shared bounded queue after provider access recovers; existing cache and pending jobs are retained |
| Older SL pending jobs | Source-backed classification through explicit historical ranges; preserve unexplained 404s and failures |
| Ambiguous horse identities | Reviewed identity migration spanning canonical storage and historical feature grouping |
| Newly retained SL winning time, actual off time and explicit currency | Reparse retained raw payloads; historical pages predating retention require an explicit fetch if those fields are needed |
| Betfair timed pre-off trades | Historical archive input; public daily BSP and VWAP cannot recreate those observations |
| Missing genuine pre-race snapshots | Capture prospectively; a historical post-race card cannot establish what was available before the off |
| Pedigree failures and optional weather | Existing bounded enrichment queue / explicit research collection |
| Consolidated RTV derived files and text-fit invalidation | Later storage and feature-build cleanup; generation publication is coherent, but whole derived files are still materialised |

RTV still computes learned/derived views during its source refresh for
compatibility. A cleaner stage boundary would publish parsed observations in
stage 1 and build the learned views together in stage 2. The current generation
pointer and caches make that boundary safer without altering fitted definitions.

## Readiness and evidence

The deep readiness check observed a feature store built at 19:15 on
4 November, containing 757,083 rows through 3 November. Its code and source-data
checks pass with FE signature `bc357c18a5e5a261`. The promoted model
`20261102_143544_663385` records the older signature `69babeb9c4262ee2`.
Readiness remains blocked by ATR coverage and the model/feature-generation
mismatch. These collection changes do not establish predictive performance.

Evidence is in `data/audit/stage1_continuation_20261104/`:

- `backup.json` and `backup/`: SQLite snapshots and source/metadata backups
  captured before collection.
- `initial_refresh.json`, `repaired_refresh.json`, `final_refresh.json`:
  installed source runs and per-source outcomes.
- `bsp_integrity.json`: preservation of original price rows and timed trades.
- `cached_source_measurements_before.json`, `cached_source_measurements.json`,
  `atr_profile_before.txt`, `atr_profile.txt`: no-network measurements and profiles.
- `pending_sample.json`, `historical_classification.json`,
  `field_688054_review.json`, `identity_issue_recorded.json`,
  `name_identity_audit.json`: historical classification and identity evidence.
- `readiness_deep.json`: full read-only source/store/model comparison;
  `readiness_final.json`: final quick status after historical ledger updates.
- `integrated_tests.txt`: **257 passed**, with one pandas concatenation
  deprecation warning, in 238.04 seconds. Other `*_tests.txt` files retain
  the intermediate isolated regression checks.
  `lint.txt` records clean source/test lint; the app's three existing
  semicolon-style warnings are unchanged.

The added checks exercise source scheduling, progress publication, strict page
identity, failed-source isolation, archive/public ID compatibility, shared BSP
matching, receipt invalidation, interrupted publication, ATR cooldown/resumption
and readiness after subset or historical refreshes.
