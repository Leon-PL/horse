# Deployment fitting and current-data qualification — 005

**Historical deployment checkpoint.** The fitting dates and production model
below describe package 005. Use the [README](../../README.md) for current operation
and [027](BSP_TEACHER_REFRESH_027.md) for the later installed teaching-data update.

**Status: implemented, qualified and activated.** Production is
`20261026_140950_205296`, the retained recipe refitted through October 19.
This completes the operational fitting/data-refresh work in P1 of the original
market-performance plan. Closed feature, architecture and cadence experiments
remain closed. The overall forecasting plan is still in progress.

| Cutoff | Previous production | Current production |
|---|---|---|
| Actual parameter fitting | 8 July 2026 | 19 October 2026 |
| Feature history | 12 October 2026 | 25 October 2026 |

The earlier run `20261025_193016_331639` remains available for rollback.
There is no new held-out or market-parity claim for this deployment artifact.

The old promoted artifact was created on October 25 but its parameters fit
labels only through July 8. Its evaluation held back 90 days. Deployment
fitting now has its own explicit cutoff and uses the existing native
`RacePredictor.fit_as_of` for both win and place. Capacity selection and
calibration remain purged and chronological. Complete single-winner fields
are required; actual non-finishers remain losing runners. Ambiguous fields
are excluded whole and counted in the saved fitting metadata.

## Recipe and evidence

A deployment configuration inherits its evaluated settings from a validation
certificate. The certificate fingerprints native fitting/inference sources,
orchestration, effective configuration, selected features, model framework,
seed, hyperparameters, calibration, and reference metadata/model/evidence.
Changed settings or evidence fail before collection or fitting. Feature
generation must also match before any refresh. The final run records its
cutoff, training/calibration dates and validation reference, with empty
held-out performance fields. It cannot pass itself off as another evaluation.

New evaluation runs record their recipe and can export a certificate through
`scripts/create_validation_report.py`. The existing October 25 run is migrated
explicitly using its preserved audit, unchanged configuration, the retained
149-column recipe, native-source hashes and actual saved-artifact replays.
Its historical metadata and scores are preserved. This is qualification of
the retained recipe, with no new forecasting-performance or cadence claim.

The shared API, CLI and training app use the same deployment path. The app
shows actual fitting dates and identifies the source evaluation separately.
The public API supports a worker limit for the 16 GB machine. Review and
promotion remain separate operations.

## Data generation and memory repair

Before refreshing, eight audited inputs were copied and verified under
`data/snapshots/20261026_pre_refresh/`: all five store files, raw CSV, production
pointer and a consistent SQLite backup. Original data contained 749,261 rows /
79,664 races through October 12. Historical audit paths can be resolved to
this manifest after the live generation advances.

The fitted generation is also preserved under
`data/snapshots/20261026_deployment_005/`. The closing package archives both
model artifacts, their metadata and the validation certificate, so later
dataset refreshes or registry cleanup do not erase this refit's inputs or
validation lineage.

The bounded 14-day refresh skips optional pedigree/weather backfills. It
added 4,473 validated rows through October 25, closed 29 races as not run, and
left 16 inconsistent fields pending. Missing or contradictory starters are
never guessed. Earlier audit data remain preserved.

The first append exposed a memory problem: native storage concatenated two
full historical feature matrices. It was stopped before a staged commit.
All five original file hashes are verified before retrying. The replacement
calls the same native feature calculations and carried states, then streams
historical/new parquet rows in 16,384-row batches. Processed-history columns
and missing values are preserved. All five outputs and signatures commit
together under the existing transaction lock. The existing batched builder
shares the consolidation helper, replacing duplicated storage code.

Feature-value functions and their signature sources are unchanged. The FE
signature remains `221fcd4db0fc5631`. A native synthetic comparison checks all
feature values against the existing append; transaction tests inject failure
and retry to verify that carried state cannot advance twice.

## Qualification results

The store now contains **753,734 runners / 80,135 races** through October 25,
data signature `f3259ac47f8249b0`. Every historical feature value was compared
in bounded batches with the preserved snapshot: **749,261 rows across all
1,163 columns match exactly**. The new historical/new parquet consolidation
remained at approximately 6 GB process memory while streaming.

The refit uses **665,252 eligible runners / 71,207 races**, after the existing
eight-month burn-in. Complete-field checks exclude **162 fields / 1,549 rows**
from the pre-burn-in input population through the cutoff. These include
invalid labels/identities/field sizes and degenerate fields; this is separate
from the 16 recent fetches still pending. True non-finishers remain included.
The final win and place models use 1,856 and 2,987 selected trees respectively.

Calibration predictions begin September 18, 2025, after a September 11 base
training cutoff and seven-day purge. Calibration labels end October 19, 2026.
The deployment artifact has zero held-out races and empty performance fields;
the separately evaluated recipe remains its validation reference.

Native integration/storage checks passed **35 tests**. The full regression
passed **769 tests**, 108 warnings, in 779.68 seconds. An earlier full run
reported 768 passes and one outdated fault-injection target after writer
extraction; the test was corrected to the shared writer, passed independently,
and the entire suite then passed. No production error was hidden by weakening
the rollback assertions.

Original in-memory win/place probabilities were recorded before saving.
Reloading through the serving registry on **64 shuffled/regrouped fields /
574 runners** reproduces both outputs **exactly**. Win sums, place ranges,
feature lists, fitted method, source/data hashes, actual cutoff and recorded
calibration dates pass. Production activation repeats the same checks through
the production loader; both probability outputs still match. Promotion is an
operational refit of the selected base recipe, not advancement of the uncertain
monthly or race-softmax challengers.

Execution manifest raw SHA-256:
`6b992dfeb206daedfefb71333abb49160d4b95a29e58803eded300b76fc4565b`.
It freezes **27 source files / nine input artifacts** before fitting. The
validation-certificate SHA-256 is
`b895615f0eabf6efe19d070233bc07700b28e873cf742a3c77ff93b8de6d9a33`.

Deep production status passes: all links are ready, database and features
match, history is one day old and fitted parameters are seven days old.
Closing checksum verification accompanies the saved evidence. Fresh forward
comparison, the separate market-informed forecast and
bounded missing-information work remain in the original plan. Those determine
future predictive claims; the historical recipe scores are not new scores of
this refitted artifact.

Evidence: `data/audit/deployment/005_current_fit/`, snapshot manifest above,
and `data/audit/20261026/deployment-005-*.log`.
