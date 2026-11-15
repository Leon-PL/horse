# ATR tab coverage and catch-up — 6 November 2026

**7 November update:** the five-tab contract is unchanged. Daily ID persistence,
legacy hint repair, export recovery and queue performance were improved in
[the ID and speed deployment](ATR_SPEED_AND_IDS_2026-11-07.md). That deployment
adds `atr_ids.py` to the fingerprint, bringing the current pin set to five modules;
the four-file list below describes the earlier checkpoint.

The main collector and the running historical worker now use a **five-tab
collection contract**. The earlier jumps-only pilot was not connected to the
worker; this deployment fixes that gap and a second missed endpoint, Distance Ran.

## What was checked

Live browser navigation was inspected on Chepstow 4 November 12:20, Wolverhampton
2 November 20:30, Newcastle 29 November 2025 14:00, Wexford 4 November 15:10, and
the pre-race card at Uttoxeter 7 November 12:15. The result menus differ by race.
The union of their result surfaces is classified as follows:

| Result tab | Treatment |
|---|---|
| Full Result | Canonical outcomes, prices and runner/race details already collected from the results sources; ATR also supplies identity/comments on its performance fragments |
| Sectional Times | Collected: splits, totals, markers, pace/front flags and notes |
| Sectional Tools | Collected: speeds, efficiency, finishing percentage, energy profiles and labelled extras |
| Stride Data | Collected: length/frequency curves, markers, references and peaks |
| Jumps Data | **Now integrated**: early/middle/late/average loss, A–E fluency, consistency and dated course benchmark |
| Distance Ran | **Newly found and integrated**: runner distance, difference from the race median, race median, revision date and beta-status flag |
| Future Form | Explicitly excluded from historical prediction inputs: it incorporates later results |

All five known performance endpoints are requested, even if a particular menu
does not advertise one. This mattered in the live pilot: the Chepstow jumps
result also returned distance-travelled measurements, despite not listing the
Distance Ran tab. The collector does not rely on a single example's menu.

The separate pre-race strip contains **Racecard, Quick Form, Form Snapshot,
Pace, FormScan, Tour Quotes and Printouts**. These are inventoried as pre-race
snapshots, not historical result measurements. Quick Form includes scored
assessments, Pace includes pace ratings, and Tour Quotes uses the latest stable
tour comments. These extra snapshots are **not collected by this historical
result backfill**. Their original pre-race contents cannot be recovered merely
by visiting today's version of an old result. A future capture process would
need to retain them with their actual observation times. This report does not
claim that all ATR editorial content or every pre-race snapshot is collected.

## Main scraper and ordinary refreshes

`src/ingestion/attheraces.py` requests `times`, `tools`, `stride`, `jumpsdata` and
`distanceran`. New raw fields belong to `ALL_ATR_COLS`, so the existing feature
boundary removes them until a historical derivation is deliberately added.
Optional tabs do not overwrite the original three tabs' placing/comment
metadata with empty placeholders. Native IDs and source date/local-time checks
still protect the canonical joins.

`atr_parser_version=2` continues to identify the unchanged original parsing
functions. **`atr_collection_version=3` identifies the wider five-tab contract.**
These are different guarantees. A runner cached by collection version 2 no
longer causes a version-3 refresh to skip its race. Legacy three-tab absence
decisions are separate from `atr_no_data_keys_v3.json` and cannot suppress the
new endpoint checks. Incremental cache updates replace exact matching runner
records rather than appending another copy of them.

## Durable completion checks

The authoritative SQLite queue now includes `tab_coverage`, keyed by canonical
race ID and tab kind. It records:

- Pending, received data, recognised empty response, HTTP absence, block,
  unresolved layout or identity review, for each endpoint.
- URL, response hash, HTTP status, provider and update time.
- Runners listed, runners with actual measurements, native IDs/names, expected
  runner count and missing canonical runners where available.

Older three-tab receipts without measured-runner counts are labelled
`legacy_received`, not verified telemetry. They are reclassified from retained
source responses during catch-up. Race acceptance and per-tab reports use the
same signal definition, including genuine speed-only observations when split
and curve data is absent.

A horse list is not telemetry. A real Wexford response contained runner
identities, zero placeholders and no measurements. A new real-response test
checks all five empty templates. Missing distance cells are accepted only for
the recognised identity-only empty shape; changed metric layouts remain
unresolved. Missing telemetry is not filled with zero loss or zero distance.

The enclosing race cannot be accepted as data until its manifest accounts for
**all five endpoints**. Blocked later tabs leave the race in the retry queue and
preserve existing saved runners. Missing source runners remain explicit; a
finisher-only source is not labelled complete tracking of the whole field.
Counts in `status.json` distinguish completed tab collection from source data
availability. A valid empty response means that ATR supplied no measurements at
that observation, not proof that the source can never publish a later revision.

Raw response bodies remain content-addressed in `data/atr_cache/raw.sqlite`,
including errors and distinct revisions. URL/hash provenance links parsed
observations to that archive, whose observation records retain first/last seen
times. Later parser improvements can reuse these bodies.

## Catch-up and restart

The worker stopped cleanly with **33,918 runner records**. The upgrade preserved
their complete payload digest and all **62,432 queued UK races**. It reopened
**4,224 previously completed races** under the wider contract; no saved runner
was reset or deleted. The apparently lower completed-race count reflects the
new completion requirement, not loss of the old work.

The same worker replays retained successful fragments and requests missing
endpoints through the same throttled client. It does not start a competing
jumps worker or download the original three tabs again when suitable retained
responses are available. Blocks, Retry-After handling, publication grace and
shared provider cooldowns remain in force. Source pages/indices can still be
challenged; those gaps remain visible and pending.

The initial deployment resumed at **12:08 UTC**; final reporting qualification
and resumption completed at **12:30 UTC**, with the same two-second base gap
plus jitter. Launcher PID **6748**, worker PID **16432**. Use the live status file
rather than treating these PIDs as permanent. Collector pins now cover
`ingestion/attheraces.py`, `ingestion/atr_rebuild.py`, `ingestion/atr_tabs.py` and
`ingestion/atr_jumps.py` (paths after the 6 November move). Feature derivation remains outside this raw-worker
guard. Review and stop the worker before editing its four collection files.

The old three-tab throughput figures are historical; they should not be reused
as a five-tab ETA. Replaying three retained tabs makes catch-up cheaper than a
fresh five-request collection.

## Detecting future omissions

`src/ingestion/atr_tabs.py` classifies links in the actual race tab strip and
ignores unrelated sidebar/navigation links. An hourly bounded page check runs
within the worker, sharing its client and cooldowns. Newly advertised,
unclassified tabs put the worker into **`needs_tab_review`**, rather than being
silently ignored. A layout that no longer parses remains unresolved. Browser
inspection provides the retained initial baseline; an inaccessible HTTP page is
reported as blocked/deferred/unverified, never as a successful inventory check.

This is a monitored contract for the five known performance endpoints, not a
claim that an unseen future ATR product is already supported. The current queue
still has its documented UK scope and known Irish venue exclusions; the live
Wexford probe returned no performance measurements on any of the five endpoints.
Canonical Irish results and separate RTV coverage are not deleted or excluded
from the project by that ATR-specific rule.

## Validation and evidence

**272 distinct checks passed** across collection, recovery, field identity,
feature boundaries, ingestion and signature coverage. The six original parsing
and identity functions were unchanged by AST comparison. A live five-endpoint
pilot matched **13 runners** to canonical results: ten flat runners with distance
measurements, and three jumps runners with both jumping and distance data.
The final qualification re-parsed retained bodies and verified that adding the
new tabs preserved the original three parsers' existing values.

At **12:31 UTC** the live main queue contained **87 runner records with jumps
measurements and 825 with distance measurements**, including the three jumps
records from the user's Chepstow example. The worker was running and its source
pins matched the reviewed code. This verifies collection into the real backfill,
beyond the earlier separate pilot. Counts continue changing.

Evidence is in `data/audit/atr_tab_coverage_20261106/`: the consistent queue
backup, before-files, passing test reports, live/qualified observations,
browser inventory baseline, adoption receipt and launch record. The accepted
historical cache is still separate from this rebuilding generation. No normal
feature-store build, model fit or promotion was performed.

For current progress inspect:

- `data/atr_cache/rebuild_v2/status.json`: per-tab states and navigation audit.
- `data/atr_cache/rebuild_v2/rebuild.sqlite`: jobs, runners and tab receipts.
- `data/atr_cache/rebuild_v2/daily/`: derived exports, refreshed in batches.

After shutdown, resume the same worker with `python -m src.ingestion.atr_rebuild
--request-delay 2`. Do not run `--prepare` on this existing queue. The explicit
`--upgrade-tabs` command is for a reviewed collection-contract migration, not a
routine restart or a way to bypass an unexplained fingerprint mismatch.
