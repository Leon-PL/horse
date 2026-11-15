# Race-type corrections and distance-history review032

## Confirmed corrections enter ordinary training

The remaining historical025 classification leads were checked against dated
Sporting Life performance records. A runner was selected by ID for each event,
without using its finish or price. All53 event checks succeeded.52 events,
430 historical runner rows, require a different coarse race type. With the16
earlier verified events, the normal reference catalog now covers68 fields.

Each replacement is guarded by event ID, date and course. Public profile
identity/name, event identity/date/course and structured historical run type
were checked. Raw profiles, hashes, the fixed roster and collection receipts
are saved under `data/audit/trip_suitability_032/`. Current profile ratings,
lifetime totals, RPR/TS and later race results were not added as predictors.

Examples of misleading sponsor names include Chasemore, Chipchase, Chase
Distillery, Cruises Chase and Hullabaloos Chase. Historical records are fixed
from structured source evidence, rather than automatically rewritten because
of a name or distance threshold.

Ingestion also had seven remaining title-parser disagreements on the expanded
source-labelled corpus. A later actual Stakes title now takes precedence over
an earlier sponsor's Chase word. Explicit verified sponsor occurrences are
ignored while preserving any later actual Chase. A restriction to "Flat Race
Jockeys" no longer turns a Hurdle into a Flat race. Galway Hurdle Handicap
remains Hurdle: the generic word Handicap is not treated as proof of Flat.

The repaired parser agrees with all19180 source-labelled fields in this
metadata check, with zero new disagreements.121 importer checks and12 ordinary
preprocessing/integration checks pass. This verifies the audited corpus,
not every possible future race title.

The corrections are installed in `reference/verified_race_types.json` and apply
through ordinary preprocessing. They are not an optional experimental feature
list. The canonical feature rebuild and matching model update are complete,
installed with the selected033 family as `20261029_085433_011356`. Deep
readiness and native live checks pass. The previous029 model remains
recoverable. No predictive-gain
claim follows merely from confirming a wrong race type.

## Distance-history proposal

The legacy Flat-oriented distance buckets put248999 of249229 rows carrying a
jumps discipline into the same above11.5f bucket. This loses the distinction
between two-mile and three-mile jumps form, and can mix long Flat records into
the selected distance-category win rate. This is an information limitation;
the old numeric bucket calculation itself is reproducible.

A research proposal keeps separate Turf/AW/Hurdle/Chase/Bumper histories and
smoothly weights prior trips using a fixed15% relative-distance bandwidth.
Only the last20 actual starts within730 days are retained. Finishing quality
is field-size-normalised; a settled DNF scores0. Five pseudo-starts at0.5
shrink the weighted quality. A separate specialist edge centres outcomes on
that horse's own recent mean; support reports the effective number of nearby
trips. There are three additional inputs, with no coefficient search.

Six tests verify prior-day/future/same-day exclusion, separate scales,
distance discrimination, stale/nonstarter handling and seeded live equality.
The full real-data prefix and append calculation agree exactly. Preparation
took144.10s. The research inputs use the verified metadata catalog.

One400-tree candidate win fit reused the exact030 baseline. It took76.74s
including preparation/prediction. No place fit, calibration, quarter sweep,
capacity search or production refit was performed for this proposal.

| Q4development population | Baseline log loss | Proposal log loss | Baseline Brier | Proposal Brier |
| --- | ---: | ---: | ---: | ---: |
| All3233 fields/30081 runners |1.847941|1.845222|0.085482|0.085411|
| Established handicaps1071 fields/9777 runners |1.987342|1.988217|0.091748|0.091862|

The overall log-loss change is-0.002720, date-cluster95% interval
[-0.006360,+0.000999]. The point estimate is encouraging, but it does not
establish an improvement; the target established handicaps worsen slightly.
The family remains unselected, without further tuning or full-capacity fits
now. Do not label it an established general failure or a qualified gain.
This proposal did not change the151-input selection; the later033 update adds
three different, qualified inputs. These dates were already inspected
development data, not an untouched holdout.

Evidence starts with `legacy_audit.json`, `parser_validation.json`,
`type_source_receipts.json`, `trip_audit.json` and `quick_screen/report.json`.

## Canonical rebuild and fitting

`scripts/rebuild_race_type_repairs.py` (removed 2026-11-06, in git history) preserved the previous store with
hardlinks and rebuilds the normal feature stages. Every Elo component is
recomputed. Glicko/TrueSkill reuse requires identical code/config/runtime,
all processed values identical except the reviewed race types and their
indicators, the exact52-field/430-row correction population, and equality
with a fresh native prefix. This avoids repeating expensive unaffected work.

The existing canonical-repair controller now accepts a case directory and
explicit installed parent, so this batch can reuse the same qualification,
calibration, checkpoint and deployment code. It reads this batch's current
teacher fingerprint, rather than silently inheriting the older022 teacher.
Its interruption/restart regression test passes on a tiny fixture.

The completed base rebuild took3812s and preserved754497 runners through
27 October2026. It changed six of the151 selected inputs, mostly through
race-type-specific Elo propagation; global Elo, Glicko and TrueSkill are
identical. Source receipts and exact input comparisons are saved in
`selected_input_changes.json` and `rating_delta_details.json`.

The subsequent033 family passed its one native win check and was integrated
by appending only three independent columns to this corrected base. Its
historical qualification reuses the completed win fits, adds matching place
fits, then performs one current-data win/place update through27 October2026.
All verified032 corrections apply unconditionally. See
[the combined results and installation](PRIOR_MARKET_HISTORY_033.md).
