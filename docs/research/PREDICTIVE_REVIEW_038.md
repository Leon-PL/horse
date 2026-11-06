# Six predictive weaknesses: implementation and bounded evaluation

Work started29 October2026 at the user's request to improve all six findings.
The existing033 deployment and all earlier evidence are the reference.

1. Preserve dated performance circumstances and build a causal, missing-aware
   historical context representation. Audit cached real sources before fitting.
2. Compare one fixed recent90-calendar-day calibration recipe against frozen033
   Q1 forecasts. Use two native temporary heads fitted before that window, a
   seven-day purge, and smooth maps that cannot create isotonic zero plateaus.
3. Capture reported rider claims, penalties and dated factual ride descriptions
   in normal ingestion, separately from declared weights and career summaries.
   Unknown observations stay unknown. Do not rewrite old results by guessing.
4. Run a restartable forecast recorder with immutable attempts, coverage and
   heartbeat evidence. Pin a new study after implementation settles.
5. Assess every issued field separately from the stricter identical-field
   benchmark; preserve withdrawals, timing, missing results and card changes.
6. Compare the existing place projection with training-fitted shrinkage and
   training-estimated missing slots. Keep win probabilities fixed for that
   comparison to identify the place change's effect.

Predictive changes must lower the predeclared proper scores with date-cluster
uncertainty. Earlier inspected Q1 dates remain development. A negative result
does not earn a production change. The calibration experiment registers its
exact sources and inputs before two fixed-cap temporary tree fits; it neither
refits the baseline nor searches quarters, seeds, feature subsets or capacities.

## Completed development comparison

The two fixed native temporary heads finished. A reporting-only missing
experience column initially stopped the readout; `read_probability_refinement.py`
joins the original saved cohort metadata with exact identities and labels.
Its own registered repair receipt preserves the original procedure and both
fits. No coefficient, capacity, test dates or advancement condition changed.
The saved154-input baseline replays with maximum error zero.

| Same2,591 Q1 fields /22,824 runners | Original | Candidate |
|---|---:|---:|
|Win race log loss|1.798224325|1.799563197|
|Win runner Brier|0.089957954|0.090005218|
|Place runner Brier|0.174255392|0.173851156|
|Established-handicap place Brier,993 fields|0.192047590|0.191674051|

The recent smooth win map fails its advancement rule. Its log-loss delta is
+0.001338871, date95% interval[-0.000497202,+0.003039582]. Keep the original
win method; do not tune another map on these inspected outcomes.

The place layer passes its registered rule. Overall Brier delta is
-0.000404236, date95% interval[-0.000700465,-0.000103740]. The target point
delta is-0.000373539; its interval[-0.000795320,+0.000027134] crosses zero.
This supports the overall development improvement, not a separately confirmed
target-cohort gain or prospective improvement. Place comparisons retain the
original win probabilities, isolating the place change.

The training-fitted shrinkage is0.00035895, close to no additional shrinkage,
versus the old fixed strength2. Missing-slot expectations are learned with a
100-field zero-deficit prior and remain small. They do not use the current
field's future completion count. Smooth maps, slot expectations and win lower
bounds are saved with the model; older artifacts retain their old probabilities.

## Dated performance context and normal collection

The fixed pilot reused48 cached result sources and retrieved24 prior fields.
All72 sources match the existing runner identities and declared weights:
625 starters,182 explicit positive claims,443 unknown claims, five penalties,
and625 dated ride descriptions. Unknown claims are not observed zeros; a
penalty is not added again to an already allocated weight. No canonical
finishing label or feature value was rewritten.

The append-only `result_context_observations` table now receives supplied
context in the same transaction as normal result ingestion. `load_result_context`
keeps this evidence separate from ordinary historical feature matrices, so
adding metadata does not fabricate earlier observations or trigger a feature
rebuild. Source evidence is retained, not guessed from present-day profiles.

The unselected `performance_context` family distinguishes source mentions of
impediments, slow starts, jumping errors, easing and finishing strongly, plus
known accidental/pulled-up nonfinishes. These are wording descriptors, not
automatic excuses, ability scores or probability bonuses. Actual starts with
missing commentary advance the bounded history. Full and prefix/seed histories
agree on9,043 records for356 available horses from the fixed373-horse pilot;452 rows have prior description
support. Coverage is too sparse and the source sample partly error-selected
to support a native accuracy claim. Source publication times are not recovered
by a retrospective scrape; future capture supplies actual observation times.

## Forward evidence and issued-field assessment

The recorder discovers events before the decision, runs bounded attempts,
retains every failure, limits retries across restarts, and writes a heartbeat.
It refreshes settled histories through the ordinary pipeline without changing
model weights. A single-instance Windows task starts at09:00 and logon,
stops at03:00, and catches missed starts when the signed-in computer is
available. Source/model mismatches stop recording rather than silently changing
the study. The final study and installation receipt are recorded on completion.

The original identical-field benchmark stays available. The additional issued-
field score retains confirmed later withdrawals at their original probabilities
and scores their wins as zero. It never renormalises away their forecast mass.
An absent runner is not enough evidence: exact event/market/runner IDs,
normalized name, explicit withdrawal flag, observation time and a checksum-
verified source receipt must agree. New starters, ambiguous fields, incomplete
results and unsupported absences remain visible exclusions. Card/result changes
and attempt-rejection reasons are reported separately. Assessment snapshots
preserve both results and withdrawal evidence.

## Evidence and entry points

- `data/audit/predictive_review_038/probability/`: frozen plan, source capsule,
  two native checkpoints, fitted maps, readout repair, paired forecasts and report.
- `data/audit/predictive_review_038/context/`: source selection,24 fetched fields,
  parsed observations, causal pilot histories and coverage report.
- `scripts/adopt_recent_place.py`: separate historical qualification and current
  calibration using the unchanged final estimators; explicit promotion flag.
- `scripts/start_forecast_recorder.ps1` and `install_forecast_recorder.ps1`:
  hidden local recording, process-local Python configuration and Windows task.
- `scripts/project_status.py`: model-serving readiness plus recorder health.

The final full regression suite passes1,286tests. Subsequent targeted checks
pass50 forward/place tests,51 source/database tests, and19 recorder/issued/store-refresh
tests. These groups overlap and are not additional unique test counts. They
include preservation of older verified withdrawal proofs, explicit tail-risk
reporting and result snapshots scoped to issued races. The full-suite run also repaired two missing
taxonomy annotations from the earlier established-form/prior-market stages.
These annotations do not alter feature values or the feature-store signature.

The historical qualification is run `20261029_133847_609414`. Its canonical
place predictions replay the prototype exactly; its win predictions replay033
exactly. Current calibration uses `installation_v4/current_inputs.parquet`,
copied under the store lock. The raw training-prefix signature matches033's
`267f5d1582796e06`;667,873 eligible runners in71,477 fields through27October
are required before fitting. This prevents accidentally using the smaller
historical evaluation snapshot as current calibration data. Earlier reporting
attempts and the stopped mismatched-data attempt remain preserved; none was
promoted and no old artifact was overwritten.

## Installed artifact and operational checks

Production run **20261029_141132_815571** was installed at14:18:24UTC on29October.
Its SHA256 is `73747ec2166fb6598687c9dedbffbaa37f20694256933cda4d03e864b4e156b0`.
The154 inputs,3,992win trees and2,991place trees remain unchanged. Current place
calibration uses3,427 fields /32,498 runners from30July through27October,
with two temporary heads fitted through23July and a seven-day purge. This
full-data installation has no separate held-out performance estimate.

The installed win probabilities agree exactly with033 on100 recent fields;
single/batch place forecasts also agree exactly. The native live check passes
at14:23UTC on115 complete recent fields /1,108 runners and the real Ffos Las
15:00 card with seven runners. It checks probability constraints, learned
expected place mass, identity and both serving paths. It establishes operation,
not accuracy on future races. Receipts are in038/installation_v4 and038/live.

The new forward protocol records the refined and retained legacy place
forecasts together before the race, alongside win and market forecasts.
Later place comparisons use the immutable issued top-k definition. This
provides prospective evidence for the deployed place change without silently
recomputing its control. Tail diagnostics report extreme-probability errors;
they do not alter the issued probabilities.

The first Windows registration attempt was denied because its logon trigger
covered all users. Restricting it to the current user allowed registration.
The first actual startup then exposed a SQLite-versus-CSV input fingerprint
mismatch, causing an unnecessary staged rebuild. It was stopped before any
canonical feature files were replaced. Recorder refresh now processes the
atomic CSV export, matching ordinary ingestion, and permits append/no-op only.
An incompatible history stops for maintenance instead of launching a rebuild.
Two regression cases cover matching input fingerprints and refusal of a
required rebuild. The corrected source receives a new pinned study; earlier
empty studies, logs and setup receipts remain preserved.

The deep readiness check at 14:45 UTC passes: database contents, feature-store
values/signatures, current feature source and installed model agree. The
interrupted staging directory never replaced canonical files. Its recorder
warning describes the deliberately stopped earlier session, not a model-serving
failure. Receipt:038/readiness.json.

Final activation passes at **14:51 UTC**. Windows task
`HorseForecastRecorder_038` is running under the current user, with next daily
start30October at09:00. The corrected startup completes as a safe no-op,
leaving canonical feature files unchanged. Discovery and assessment succeed;
fresh15-second recording heartbeats contain no errors. Model, protocol and
source hashes verify. Final receipts are038/final_status.json,
038/readiness_operational.json and038/scheduler_running.json.

Current study:
`20261029_predictive_review_038_final_20261029_141132_815571`.
Protocol SHA256:
`5f7760ab96f57c1e7276fc84673e7ab81a2212ddb89c0a3f6c71d81e048dddc9`.
First eligible date30October2026; zero eligible observations at activation.
The descriptive readout requires84calendar days,1,000 paired fields and60race
dates, with a seven-day settlement grace. Earliest possible sample close is
21January2027, conditional on the count requirements. No automatic accuracy,
improvement or market-parity declaration is registered. Recording requires
the computer to be awake and the current user signed in.

| Review issue | Delivered change | Remaining evidence |
|---|---|---|
| Established-horse performance context | Causal, missing-aware context histories; exact temporal replay and source pilot | Wider dated coverage before fitting or selecting new predictors |
| Calibration transfer and tails | Fixed recent calibration trial; recent place layer adopted; win change rejected; tail monitoring | Fresh forward accuracy under the deployed calibration |
| Lost source detail | Normal ingestion preserves reported claims, penalties, descriptions and structured reasons | Unknown historical details remain unknown |
| No continuous evidence | Restartable current-user Windows recorder, immutable attempts, logs and heartbeat | Eligible capture begins 30 October; computer must be awake and signed in |
| Final-starter evaluation bias | Separate issued-field scoring, original probabilities and verified later withdrawals | Ambiguous or incomplete fields remain explicit exclusions |
| Fixed place shrinkage/full slot mass | Training-fitted smooth place map, shrinkage and expected missing slots | Overall development improvement needs prospective replication |
