# BHA numeric candidates — 7 November 2026

The ordinary feature pipeline now includes `add_bha_features`. It derives **161
numeric candidates** together; no individual feature model, ablation model or
predictive-value trial was trained for this deployment. Existing feature
selection can remove sparse, duplicate or unhelpful candidates.

`src/bha_events.py` attributes the source text to canonical horses;
`src/bha_features.py` turns the observations into prior-history features.
The collection worker and its pinned extraction modules are unchanged.

## Families

| Family | Source observations and summaries |
|---|---|
| Reported physical issues | Lameness, bleeding, breathing problems, distress, injury; combined physical-issue history |
| Veterinary findings | Examination mentioned and explicit negative/clear findings are distinct |
| Behaviour and starts | Hanging/jumping left or right, racing freely, slow starts, refusals |
| Jumping and interference | Jump errors; interference suffered and interference caused remain distinct |
| Equipment | Lost shoes, tack problems, red hood to post, early to post, earplugs removed, partially shod |
| Performance explanations | Disappointing performance, no explanation, unsuitable going, preference for softer/faster going |
| Conditions | Previous unambiguous GoingStick reading and recent mean; prior rail-change report; direction/going interactions |
| Withdrawals | Observed withdrawal counts in 365 days, health/going classifications, most recent classification and days since |
| Coverage | Available report counts over three/five starts, named-in-report support and age of the most recent covered start |

For each of 31 event/group measures, the feature family exposes `_last`,
`_rate_3`, `_rate_5` and `_count_5`. Nine selected measures also have
`_days_since_5` and `_starts_since_5`, measured inside the last-five-start window.
Nineteen coverage/context/withdrawal columns complete the 161 candidates.

Rates use available covered starts as their denominator. Missing starts occupy
their chronological window positions: five old covered starts are not silently
substituted for the actual latest five starts. A missing last-start report gives
a missing `_last`, even if an older start has a report.

A zero incident flag means **not reported in a covered report**, not a clinical
finding that the horse was healthy. Unavailable reports give missing incident
values and zero observed coverage. An explicit negative examination has its own
flag. No automatic positive or negative performance adjustment is imposed.

## Attribution and timing

Race links use the dated venue and BST-adjusted local time. A small advertised
time change can be linked through unique named-horse evidence within 15 minutes;
an adjacent-hour guess is forbidden. Ambiguous horse names are not assigned.
Clauses naming several horses are attributed locally; an unrecognised capitalised
horse also ends the preceding horse's claim. Simple negation, pronoun references
and shared preparation permissions are handled separately.

Withdrawal names must identify one canonical horse globally; ambiguous names
remain unlinked. They have a separate event stream because withdrawn horses do
not appear as starters in the results. Full supporting text, source family,
report identity and availability time remain in the derived observation table.

Only strictly prior starts whose feature availability is at or before the
prediction cutoff can contribute. The current race's report cannot enter its own
features. Input order, chunk boundaries, outcomes and current prices cannot change
BHA feature values.

**The latest user decision on 7 November 2026 is a two-day assumption**, replacing
the earlier capture-only choice. `config.BHA_FEATURE_AVAILABILITY_POLICY` is now
`"race_plus_48h"`: a report becomes eligible exactly 48 hours after its linked
race time in UTC; a withdrawal becomes eligible 48 hours after its source event.
This permits historical feature derivation even though the first recorded source
capture is 7 November 2026 at 17:55 UTC.

The source and observation `available_at` fields retain their actual capture
times. The 48-hour rule is a feature assumption, not evidence of original
publication or amendment times. The latest captured report/PDF variant in the
selected immutable snapshot can therefore be used for an earlier historical
prediction once the 48 hours have elapsed; later amendments cannot be dated
reliably. Setting the policy back to `"recorded"` restores strict capture-time
eligibility, including exclusion of later PDF captures from earlier predictions.

## Feature selection and store refresh

`src.feature_selection.candidate_feature_columns(frame)` discovers all safe
numeric columns **before** applying the current production allowlist. It retains
the model's exclusions for outcomes, identifiers and other unsafe inputs. The
existing production list and default `model.get_feature_columns` behaviour are
unchanged; new candidates are not silently added to a fitted model.

The feature code signature covers both new modules and the configured timing
policy. Store metadata includes a
`bha_signature` of source content available through its history cutoff. A later
BHA backfill or amendment that changes those inputs triggers the normal store
refresh rather than leaving old columns stale. The new code also requires the
normal feature-store refresh before its candidates appear in that store.
This deployment did not rebuild the full production store or fit a model.

Derivation reads the build's immutable BHA/SL source versions. Explicit unversioned
frames get missing BHA values rather than opening a live collector database.
The observation cache is keyed by selected source paths and derivation code;
subsequent chunks reuse it. Publication of more completed BHA history therefore
enters later feature builds without a separate feature-specific scraper.

## Materialised data

```powershell
.venv\Scripts\python.exe -B -m src.bha_features --out data/processed/bha_candidates --historical
```

The command does not fit a model or generate fabricated race records. It writes:

- `observations.parquet`: source-backed runner observations with IDs, text and tags;
- `withdrawals.parquet`: identified historical non-runner observations;
- `current_horse_state.parquet`: 161 history candidates per horse at one recorded instant;
- `historical_features.parquet`: dated canonical runner keys and their 161 candidates, with timing-policy metadata (when `--historical` is supplied);
- `coverage.json`: definitions/names, populated counts, source paths, code hashes and timing policy.

The first completed capture-only pass used 758,304 canonical starts and produced 462,785 runner
observation rows (some starts have separate report/PDF captures), 275 withdrawals
and 73,695 horse states. Of those horses, 50,988 have report history in their latest
five starts; 2,982 have a reported physical issue in that window. There were 584
unmatched source sections out of 51,785; unmatched text was not forced onto a
horse. These are extraction/coverage measurements, not predictive results.

The subsequent two-day regeneration wrote all 758,304 historical runner rows.
493,672 (65.1%) have eligible BHA report history within their latest five starts;
12,566 have a reported physical issue in that window. This artifact contains
real canonical runner keys, and its Parquet metadata records `race_plus_48h`.
Original observation records and capture times were compared with the previous
cache and are identical. The full production feature store has not been rebuilt.

The test evidence in `data/audit/bha_feature_derivation_20261107/` covers
attribution, negation, time conversion, availability, future/current-race
exclusion, chunk/order parity, candidate discovery and input invalidation.
The integrated feature/store checks passed 114 tests. Model fits performed: zero.
The timing-policy change is separately verified in
`data/audit/bha_two_day_20261107/`: 40 checks passed, including the exact 48-hour
boundary, preserved capture times, same-race/future exclusion, chunk parity and
store-signature invalidation. Regenerated counts are in the current
`data/processed/bha_candidates/coverage.json`; the previous capture-only receipt
is retained under the timing-change audit's `before/` directory.
