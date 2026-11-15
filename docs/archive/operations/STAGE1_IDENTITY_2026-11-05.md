# Stage 1: runner identity repair and compatible history

**Subsequent source follow-up:** [recovery and source health](STAGE1_SOURCE_HEALTH_2026-11-05.md)
adds ATR observations and safer publication. That later change requires another
feature refresh; the successful build/readiness results below describe the
earlier identity-repair snapshot.

The historical Sporting Life identity case is fixed. Race **688054**, Gowran
Park on 12 June 2022, now contains all **15 runners**, including two different
horses named Sierra Nevada. Their source IDs are `1040489` (winner) and
`1038101` (eighth). The source field, stored field and downstream rating join
retain 15 rows; the previous name-based paths stored 14 and could produce 17
after joining ratings.

This follows the [audit](STAGE1_DATA_PIPELINE_AUDIT_2026-11-04.md),
[collection improvements](STAGE1_IMPLEMENTATION_2026-11-04.md),
[installed recovery](STAGE1_RECOVERY_2026-11-04.md) and
[historical queue recovery](STAGE1_BACKLOG_2026-11-04.md).

## What changed

Canonical runners use `(race_id, Sporting Life horse_id)`. Display names remain
available for people. Missing-ID legacy input can match a name only when it
identifies one runner on both sides of the race. Different known IDs never
match through a shared name. A legacy writer preserves an existing known ID
and rejects an ambiguous name instead of selecting one horse.

The change covers field replacement, runner text, result-context selection,
reviewed-result and amendment receipts, rating output and reuse, batched builds,
append/live joins, prediction metadata, app diagnostics and result settlement.
Older name-only settlement records remain unsettled when their identity cannot
be determined. Draw, wind-surgery and field-reconciliation repair tools also
reject ambiguous name-only targets.

ATR and RTV IDs remain in their own namespaces. A native ID cannot substitute
for a Sporting Life ID. Name-based ATR crosswalks and RTV pre-race snapshots
leave ambiguous names unmatched, preserving runner counts rather than copying
one observation onto both namesakes.

The identity helper participates in feature-code fingerprints. The schema
helper is exempt because it changes storage and receipts; accepted value changes
are covered by source/data fingerprints. Existing horse-form and rating histories
already preferred IDs. Display-name frequency features and unrelated ranking
ties retain their existing definitions.

## Migration and preservation

An explicit `src.database.migrate_runner_identity()` migrates legacy constraints
under the source writer lock and one SQLite transaction. Fresh databases create
the new indexes directly. Existing installations need the explicit migration;
ordinary initialization does not silently rebuild a large results table.

The migration preserves columns, **row IDs**, accepted values, indexes, revision
triggers and original repair-receipt JSON. ID-based and missing-ID fallback
uniqueness use separate partial indexes. Repeating the migration is safe;
an interrupted transaction rolls back.

A complete database copy was migrated and checked first. All **757,083 existing
rows and row IDs** matched exactly. Installation repeated that comparison against
a fresh backup. Only the omitted horse was restored from retained response:

`42a9cd841ca66888854d1d38f6c660734b56bdb202237dc28aebaabbf431910e`.

The other 14 rows in that field remained unchanged, including timestamps and
separate enrichment. Adding the missing runner advanced factual revision from
1816 to 1817. No original result was rewritten to fit the declared field size.

An independent comparison also confirms unchanged original values and row IDs
for 2,789 runner-text observations, 3,508 context observations, 993 reviewed-result
receipts and 1,534 amendment receipts. Added identity columns do not rewrite
their original JSON or evidence hashes (`evidence_integrity.json`).

The final query-plan review caught a migration performance regression: the
removed name constraint had also indexed `race_id`. The two partial identity
indexes cannot serve a lookup covering known and missing IDs together. A
dedicated nonpartial race index now preserves that access for results and
runner text. The query plan changes from `SCAN results` to an indexed search.
Ten repeated local field lookups during build load averaged 0.7406 seconds
before and 0.000069 seconds after; all 15 rows and the factual revision match.
These are lookup measurements. End-to-end refresh timing has not been remeasured.
`race_lookup_index.json` records the plans and timings.

## Source refresh after midnight

The next normal refresh collected the newly completed 4 November date:

| Item | Installed result |
|---|---|
| Canonical results | **757,580 runners / 80,536 races**, through 4 November |
| New rows after the identity repair | 496, all dated 4 November |
| Recent factual revision found | One runner's reported `horse_runs`: 2 → 3 |
| Factual revision after refresh | 2314 |
| RTV materialisations | Through 4 November; 756,132 runs / 755,985 cards |
| ATR | 232,138 cached rows; **3,967 eligible races pending**, provider blocked |
| Public BSP | Newest files unavailable; existing prices and timed trades retained |
| Currency reference | Through 4 November |
| Optional pedigree cache gap | 655 horses / 708 runner rows; not requested by this results-only refresh |
| Historical Sporting Life review cases | **Zero** |
| Recent Sporting Life pending field | Race 946937, 4 November Nottingham, `DORMANT` |

The recent dormant page stays open during the normal seven-day revision window.
It is not declared terminal prematurely. The combined refresh correctly reports
`partial`. Its 437.013-second elapsed time includes result rechecks and new-day
RTV materialisation; it is not a cached-refresh benchmark.

An independent post-refresh comparison retained all 757,083 original row IDs.
**757,082 original rows still match exactly**; the remaining row has the reported
`horse_runs` revision and new scrape timestamp. Added rows are the restored 2022
horse and 496 new-day runners. SQLite integrity, race/horse-ID uniqueness and
every declared field size pass.

The revised lifetime run count is retained source evidence. Feature engineering
already overwrites `horse_runs`, `horse_wins` and `horse_places` using strictly
prior races. The processed-input fingerprint still includes the raw counter,
so changing it can conservatively force a rebuild even when its feature value
would be unchanged. A later optimization could separate source-only counters
from feature-relevant revision checks, with explicit dependency tests. This
repair retains the conservative fingerprint contract.

## Feature history and readiness

A complete compatible history is **rebuilt, qualified and installed**: 757,580
rows through 4 November, over 13 batches of about 60,000 rows. The previous
five-file generation is retained separately. Source-content and code
fingerprints stayed fixed across the build; RTV reads used one generation.

Before installation, qualification passed processed/featured row counts, unique
identities, the complete canonical runner set, every result outcome, both Sierra
Nevada outcomes and all external-input fingerprints. Publication used the
existing recoverable five-file commit. Build time was 5,309 seconds; total
build, qualification and publication time was **5,539.374 seconds**.

Installed FE signature: `1355eccb0838ad9b`; processed-input signature:
`d94b262598f1663c`. `qualification.json` and `publication.json` retain the
complete checks and metadata.

The completed fourth batch has already been inspected independently: race
688054 has exactly 15 featured runners. The two namesakes have distinct pre-race
Elo/Glicko values, correct source IDs and finish positions 1 and 8.
`field_feature_preview.json` retains this checkpoint.

Separate project work promoted model `20261104_235300_444862` overnight. It was
fitted against the previous feature generation. This repair does not fit or
promote a model and does not establish new predictive-performance results.
The independent comparison aligned all **757,083 common runners** and measured
the **497 additions**. Of 129 selected inputs, 83 change somewhere; 532,879
common runners differ beyond `rtol=1e-10, atol=1e-12`. The first such differences
are in the restored 12 June 2022 field. Ratings carry changes forward through
later races, so counts alone overstate their typical magnitude. For example:

| Selected input | Mean absolute change over finite pairs | Largest absolute change |
|---|---:|---:|
| `combined_elo_vs_field` | 0.002762 | 75.778714 |
| `trainer_elo_vs_field` | 0.000720 | 0.714084 |
| `jockey_elo_vs_best_in_race` | 0.001297 | 0.571686 |

Glicko ranks change for 55 common runners and margin-Elo ranks for 212.
`feature_impact.json` records per-input counts, numerical magnitudes and
availability changes. This is a comparison of the installed generations,
including source updates and safer joins, rather than a performance evaluation
or a single-change model experiment. The saved model requires an **evaluated
refit** against this generation.

A fictional future-card fixture derived from the full 15-runner field passes:
both source IDs remain distinct, all **129 selected inputs** are invariant to
reversing row order, and metadata plus both persisted state files remain
byte-for-byte unchanged. No forecast was issued. `live_identity.json` records
the check and separate ratings for both namesakes.

The final idle-store `project_status.py` deep check confirms **feature store OK**:
code and canonical input data match. Overall serving readiness remains blocked
by the recent dormant SL field, incomplete ATR collection and the model's old
generation (`bc357c18a5e5a261` versus installed `1355eccb0838ad9b`).
`readiness_final.json` is the final report; `readiness.json` contains the same
result for convenience.

The first status pass overlapped the exclusive live-feature fixture lock and
reported an unknown store signature. The idle-store rerun resolves that to the
actual model-generation mismatch. The transient result is retained separately
in `readiness_concurrent.json`; it does not indicate a missing rebuilt signature.

## Verification and evidence

The broad identity/source/application run passed **338 tests**. A subsequent
focused run passed **30 tests**, including the new end-to-end namesake build,
append and live-feature test. Counts overlap. Checks cover legacy migration,
idempotence, rollback, ID normalization, reviewed repairs, original receipts,
ambiguous crosswalks, settlement and rating reuse.

After the race-lookup index correction, **65 database/identity/fingerprint
regressions passed**. The migration test now requires the indexed field-lookup
plan as well as unchanged rows, row IDs, revisions and repeatability.

Changed identity/schema/database/repair sources and identity tests pass lint;
working-tree whitespace checks pass. Existing unrelated feature-store lambda
and rating-reuse semicolon lint findings remain unchanged.

This is not a claim that the entire existing suite is green. The earlier
full-suite audit reproduced two failures on saved baseline code: four tied-rank
differences in the unselected `horse_avg_position_rank_pct_in_race` append
comparison, and a stale exact-registration assertion in the independent ATR
benchmark. The benchmark's invariance check passed. Baseline evidence is in
`data/audit/stage1_implementation_20261104/checks/baseline.txt` and
`baseline_atr.txt`. This repair does not change those feature definitions or
benchmark registrations.

Evidence is under `data/audit/runner_identity_20261104/`:

- `backup/races.db`, `backup/pre_install.db`: original and fresh pre-install
  snapshots; `before/` retains source snapshots from this implementation.
- `staged_verification.json`, `installation.json`: preservation, retained-source
  replay, restored ID and installed snapshot receipts.
- `source_refresh.log`, `source_integrity.json`: new-day collection and an
  independent post-refresh comparison.
- `final_tests.txt`, `identity_end_to_end_tests.txt`: regression results.
- `build_inputs.json`, `feature_build.log`: guarded history inputs and progress.
- `backup/feature_store/`: previous generation retained for rollback.
- `qualification.json`, `publication.json`: complete candidate checks and the
  installed generation's signatures and timing.
- `feature_impact.json`, `live_identity.json`: selected-input differences and
  the read-only future-card identity/reversal check.
- `race_lookup_index.json`, `index_tests.txt`: indexed lookup measurements and
  the 65-test regression follow-up.
- `readiness_final.json`: idle-store deep validation and actual remaining
  source/model blockers.

Remaining collection work is provider-dependent ATR recovery, normal rechecks
of recent incomplete pages, public BSP publication lag and the optional pedigree
queue. Pedigree uses the shared `--enrich` budget rather than requiring a separate
legacy backfill route. Missing genuine
pre-race observations need prospective capture. Moving RTV learned views into
a separate feature-build stage remains an architectural follow-up.
