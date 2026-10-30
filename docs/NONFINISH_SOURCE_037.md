# Preserve structured nonfinish reasons:037

The ordinary result collector now preserves verified Sporting Life casualty
reasons when the finishing position is numeric zero. This is an unconditional
source repair in `src/data_scraper.py`; future normal result ingestion uses it
without selecting an experimental feature set or retraining a model.

## Confirmed information loss

In678,085 pre-April2026 historical runner rows,41,941 have finishing position0:
41,866 contain only generic DNF and75 have no label. Those generic labels do
not distinguish failure to complete from its cause. The source already provides
`ride.casualty.reason`, while the collector previously inspected only
`finish_position`. Numeric0 therefore discarded known reasons.

A fixed cached sample contains48 full results /417 entries, of which26 are
nonrunners. Its391 starters include37 nonfinishers, all with a structured reason:

| Source reason | Preserved result label | Recovered records |
|---|---|---:|
|PulledUp|PU|27|
|Fell|F|5|
|UnseatedRider|UR|4|
|Refused|R|1|

The corrected collector recovers all37. Every other parsed result column is
identical to the original parser on all391 starters. Parsing the same sources
as racecards produces identical391 card rows, with result signals cleared.
Generic nonfinish counts are unchanged. The26 nonrunners remain excluded.
This sample was collected for an earlier error review and is not representative
coverage evidence for the whole database.

## Result contract and limits

An explicit raw finish label has precedence. A positive finishing position also
has precedence over an inconsistent casualty object. Numeric/string zero or an
empty position can recover the structured reason; missing, malformed or
unmapped reasons remain unknown. The four mappings above are source-verified.
No commentary or current profile statistics are used to guess an outcome.

The existing `finish_pos_label` database column already persists this value;
no schema change is needed. Dated factual ride descriptions and raw reason
annotations from the48 cached sources are preserved in a separate audit file.
They are not added as predictors. Historical generic DNFs are not relabelled
without their original source; there is no mass scrape or database mutation.

The installed154 predictors currently treat nonfinishers generically. This
repair preserves information for future reason-specific reliability/ability
work; it is **not a measured prediction improvement**. A fall should not be
silently equated with poor completed-race ability, but the value of a new
distinction still needs a bounded comparison once dated coverage is adequate.

## Related external-history recognition repair

The separate, unselected `src/public_history_features.py` prototype recognized
PU/UR abbreviations but missed the structured PulledUp/UnseatedRider spellings.
Those aliases are now recognized too. A comparison on51 cached profiles,
restricted to1,012 dated results through31 March2026, recovers17 known starts
across13 horses. Unknown positions without a reason and nonrunners remain
excluded; same-day and later results remain unavailable to an earlier card.
The existing prototype's outcome-quality formula is unchanged.16 related
checks pass. This research repair does not add its features to normal forecasts
or retroactively revise old experiment results.

## Operational validation

The feature signature remains `11bf1232d575eea6` and production model
`20261029_085433_011356` keeps the same weights.205 collector/related tests pass,
including all four observed reasons, raw-label precedence, zero representations, invalid reasons,
positive finishers, pending cards and withdrawn entrants. Source replay confirms
only the expected37 label changes.

Deep readiness passes at11:04GMT: database contents, store, feature code and
installed model match. Native live verification passes at11:01GMT on115 recent
fields /1,108 runners and Bath's12-runner12:42 card. Single-race and batch
probabilities agree exactly. These are operational checks, not new accuracy
observations. The initial parallel readiness attempt encountered the store's
exclusive lock; the completed check above ran after live verification finished.

Because the forward protocol pins the collector source as well as model code,
a replacement study was created with the **same** model:
`20261029T105914Z_nonfinish_source_037_20261029_085433_011356`.
It is eligible from30 October2026, with no observations or scheduled recorder.
The earlier033 study is retained unchanged as historical evidence.

Evidence: `data/audit/nonfinish_source_037/` contains the before-parser snapshot,
fixed source hashes, exact label differences, source annotations, history-label
counts, tests, readiness/live receipts and final status. The audit entry point
is `scripts/audit_nonfinish_source.py`. No model fitting was required.
