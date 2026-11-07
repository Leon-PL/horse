# Betfair runner identity recovery — 6 November 2026

The historical backfill stopped at 22:50 UTC while publishing a batch of 4,988
retained markets. Its parsed work remained in the queue. The immediate trigger
was Gowran Park, 12 June 2022 at 15:50: two distinct Sierra Nevada selections
share a cleaned display name.

## Cause and correction

The native WIN market `1.200083997` identifies them as selections `45802885`
(BSP 4.61) and `620329` (BSP 19.52). PLACE market `1.200083998` uses the same
selection IDs, with prices 2.16 and 4.48. These are exchange IDs, distinct from
the Sporting Life horse IDs. The publisher previously required date, course,
off time and cleaned name to be unique, so it rejected the legitimate pair.

- Publication now matches Betfair selection IDs within the race. Missing IDs
  can fall back to a name only when it is unique on both sides. Conflicting
  known IDs never merge through a name.
- The PLACE parser now retains `bf_runner_id`. Before publication, the normal
  publisher upgrades older queued **and already published** PLACE parses from
  their exact retained source SHA. Every previously parsed field must replay
  unchanged before IDs are added. Updated published records are restaged and
  included in ordinary publication; no separate manual backfill is required.
- Existing repeated observations of the same proven selection are consolidated
  in their accepted order, filling gaps. Every superseded observation is saved
  for review before the accepted cache is replaced. Duplicate incoming source
  identities remain an error.
- An unidentified old row whose name now refers to two selections cannot safely
  be assigned to either horse. Its complete observation is saved for review,
  and both identified native observations enter the active cache.
- Price attachment uses the same identity matcher. Sporting Life IDs are never
  treated as Betfair IDs. An ambiguous name-only target stays unmatched, even
  when the two source prices happen to agree. Row count and order are preserved.

The existing WIN BSP cache also contained `Sierra Nevada USA`, carrying the same
selection ID as the first Sierra Nevada, at price 4.6137024746. That alias is one
selection's additional observation, not the second horse. Its complete old row
is preserved for review; accepted 4.61 is kept, and the separate 620329 runner
is added. The old unidentified PLACE entry is likewise retained for review.
The full WIN cache contained 27 repeated selection observations in total; their
original values are retained with full numeric precision in review records.

Review files live under `data/betfair/history/identity_review/`, with content-based
names and cache provenance. They are written durably before cache replacement.
If that preservation fails, the accepted cache and queue publication state remain
unchanged. Raw market files and consolidated payloads are retained.

## Validation and recovery

Evidence is under `data/audit/betfair_identity_20261106/`. The stopped-state backup
includes the consistent queue, all four accepted price caches, source snapshots
and hashes. The regression suite passed 93 tests, with one restricted-profile
encryption skip. After the legacy-alias and published-record upgrade changes,
17 focused identity/recovery cases passed; these overlap the regression suite.
The one-time raw replay uses four workers and bounded 128-market checkpoints;
a group containing an unverified record cannot partially commit.
Eight additional storage/precision checks cover transient reader locks, permanent
replacement failure, exact legacy price preservation and matcher pins.
Targeted lint and whitespace checks passed.

Real retained-market replay verifies all four outputs contain **15 runners**,
including both Sierra Nevadas with their distinct exchange IDs and correct prices.
It checks that name-only attachment does not guess between them. The stopped
batch is published through the ordinary publisher, then the same checkpointed
WIN/PLACE catch-up resumes. Read the recovery publication, preservation verification
and restart receipts in the audit directory for the observed production outcome.

Production publication recovered all **4,988 stopped markets**, bringing the queue
to **17,157 published** and 148,129 pending. All older PLACE observations now carry
selection IDs where settled runners exist, including 6,011 previously published
markets. The four caches contain all 15 Gowran runners. Complete preservation
checks compare every accepted non-null value against the stopped-state backup,
and preserve all 27 repeated WIN observations plus the unidentified PLACE row.

Archive consolidation separately hit Windows `PermissionError` while replacing
its progress JSON. Shared atomic JSON/Parquet publication now retries a temporary
Windows reader lock for at most 2.25 seconds. Replacement stays atomic; a persistent
failure preserves accepted bytes and removes the temporary file. Consolidation
resumes its existing checkpoint with the original hash-verification rules.

The worker's seven code pins include `src/betfair.py`, `src/runner_identity.py`
and `src/ingestion/storage.py`, alongside the four historical ingestion modules.
Archive consolidation and the
independent ATR worker keep their original source files and processes. No feature
store rebuild, training run or model promotion is part of this repair.
