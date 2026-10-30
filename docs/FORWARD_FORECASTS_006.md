# Forward forecast recording — package 006

The Picks app now has a **Forecast record** page. It records the retained
independent model and a clearly labelled same-time market reference before a
race, preserving the inputs needed to replay each forecast. The CLI uses the
same implementation. This establishes forward baseline monitoring; it has
not demonstrated better predictions or independent market parity.

## Recording contract

A new study copies the current production model and metadata, freezes the
recipe, feature-generation signature, source files and runtime, and records
the comparison rules before eligible recording begins. The first eligible
date is the next full UK calendar day. The model artifact stays fixed while
settled feature history may advance under unchanged feature code. Each
eligible race requires history from one or two calendar days earlier.

The comparison uses UK and Irish Matchbook WIN markets, normalized reciprocal
**best back decimal prices in GBP**, with at least £2 advertised at the chosen
price for every active runner. This reference differs from the historical
Betfair BASIC last-traded-price benchmark. It does not establish execution,
liquidity at a larger stake, closing-line value or profitable betting.

The first observed scheduled start is retained. It must be known before the
recording window opens, and a changed schedule excludes the attempt. The
complete capture must be published between **5 minutes 30 seconds and 5
minutes before that start**. Publication is checked after the input files and
capture manifest have been written. Receipt and quote times must satisfy the
registered age and cutoff rules. The source clock tolerance is two seconds;
prices timestamped after the decision cutoff remain ineligible.

The independent forecast uses the actual saved native model and the existing
live feature calculation. It fetches one fresh Sporting Life racecard rather
than reusing discovery prices or a cached day's cards. Native history loading
can take about two minutes on this machine. A second fresh card is therefore
required after inference: every input column, dtype and value must match the
initial card exactly. Changed inputs exclude the independent forecast; they
are not patched onto old features. The new card receipt must still be within
90 seconds of publication. The market is refreshed after these steps.

Runner names, exact exchange identifiers and complete fields must match.
Missing back prices, unknown runner states, stale quotes, in-play markets,
ambiguous identities and incomplete cards cannot be filled from another
source. Separate independent-only or market-only evidence remains available
when a complete pair cannot be formed. Exchange IDs are stored as strings to
avoid losing digits in table and browser number conversions.

## Evidence and assessment

Every completed attempt has an exclusive folder retaining the available raw
decoded feeds, request/receipt times, cards, model inputs, history vintage,
probability vectors, manifest and publication receipt. Eligible pairs require
all of this evidence, including both initial and refreshed cards. Checksums
detect accidental changes. The first eligible
complete pair per event is selected through an exclusive index; later clicks
cannot replace it. A study whose forecast source or model changes must be
replaced by a new study. Earlier studies and attempts remain readable.

Assessment reads saved forecasts and a read-only results database snapshot.
It scores only eligible complete pairs on identical validated result fields.
True non-finishers remain losers. Pending results, inconsistent fields and
later withdrawals are reported separately, with the original probabilities
and runner lists preserved. No partial field is renormalized after a result.
Coverage includes discovered races without an eligible capture.

Reports include proper probability scores and paired race/date uncertainty
where enough races exist. They always set `confirmation_assessed: false`.
The proposed parity margins are recorded for a future confirmation study,
but its sample size and final decision plan have not been registered. These
baseline observations cannot automatically declare parity or promote a
challenger. No scheduled collection or orders are introduced.

## Use

Open the Picks app and select **Forecast record**. Start a study, refresh the
schedule well before the recording window, select a race and record a fresh
forecast. Allow time for native calculation before the publication window.
The page shows saved probabilities and why an attempt was selected or kept
only for monitoring. Compare eligible records with saved results after the
normal results update.

Equivalent manual commands, using a new study directory:

```powershell
.venv\Scripts\python.exe scripts/record_forecasts.py --study data/forecasts/forward/my_study --create
.venv\Scripts\python.exe scripts/record_forecasts.py --study data/forecasts/forward/my_study --list
.venv\Scripts\python.exe scripts/record_forecasts.py --study data/forecasts/forward/my_study --event EXACT_EVENT_ID
.venv\Scripts\python.exe scripts/assess_forward_forecasts.py --study data/forecasts/forward/my_study
```

The list step preserves first observations; it does not record forecasts.
Commands run once and return. Successful paired monitoring captures can still
be outside the eligible window; inspect the saved eligibility and reasons.

## Qualification

Qualification receipts, tests and live input replay are in
`data/audit/forward_forecasts/006/`. The first operational study and the revised
study are retained separately. Today's captures precede the October 27
registered start and serve only as operational checks. Final counts and
checksums are recorded in this package's completion certificate.

- **793 tests passed**, 108 warnings, 583.03 seconds. The 24 new recorder
  checks cover native model replay, publication deadlines, first selection,
  missing/stale/thin/lay-only quotes, exact IDs, discovery pagination, card
  changes, whole-field exclusions and tamper detection.
- **Two live captures, 18 runners:** Southwell at 16:47:18 UTC and Kempton at
  16:51:16 UTC on October26. Both saved native model inputs and raw market
  prices reproduce the recorded probabilities with **maximum error zero**.
  The first capture correctly failed card freshness; the revised capture's
  exact card recheck passed. Neither is an eligible forward observation.
- UI smoke checks cover an empty page, the registered study, saved runner
  probabilities, the results comparison and a retired study whose capture
  controls are disabled. The CLI and app assessments both correctly report
  zero eligible settled pairs and retain their results snapshots.
- All 41 sealed package005 evidence files and 13 archived inputs verify.
  Current model artifacts, registry pointer, feature files and raw CSV match
  their preserved hashes. The current SQLite database has identical schema
  and every row in all six tables compared with the preserved backup; its
  physical file hash differs, so logical equality was checked explicitly.

One interim native check was run concurrently with live capture and failed
because the capture held the existing store lock. Its log is preserved. The
final full suite ran without a competing capture and passed all 793 checks.
This did not require weakening the store lock or changing feature code.

The ready-to-use study is **`data/forecasts/forward/20261026_baseline_v2`**,
with protocol SHA-256
`9b161b61a956055aa164ec433455dd6df3e5b97cd31e56c4987311083bbacfa4`.
The earlier `20261026_baseline` study is retained for inspection. Archive
snapshots preserve both studies as they existed at package closure; later
recordings are separate additions.

The existing production run **20261026_140950_205296**, recipe and historical
feature generation remain in place. The final deep consistency scan passes:
database, feature values, recorded generation and production model agree.
Packages 001–005 remain closed. Further
independent research still needs a distinct, preregistered hypothesis and
fresh confirmation data; recording the current market reference is not
evidence that the independent model has caught up.
