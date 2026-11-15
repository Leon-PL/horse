# RTV result repairs and feature implementation — 30 October 2026

The 043 comparison identified stale official positions in 75 Sporting Life
fields and additional RTV fields. The 044 audit verifies the complete repair
queue against newly fetched Sporting Life responses and independent RTV order.
Source facts are repaired separately from the decision to select new predictors.
Installation progress and immutable receipts are under
`data/audit/rtv_repair_features_044/installation/`.

## Reviewed source repairs

All **75 disputed fields** have complete independently verified orders:
**336 positions change, including 19 winners**. Four declarations at Limerick
919045 are confirmed non-starters and removed. There are 857 actual starters
in those corrected fields. Another **14 missing Sporting Life fields / 132
starters** are recovered. The prepared history has **755,357 runners / 80,302
races**, through 29 October 2026.

The remaining RTV-only events include Arabian and charity racing and an
unlisted Bellewstown division. No Sporting Life identity is invented for them.
This is not a promise that every raw source event belongs in the prediction
universe. The review roster records the exclusion or unresolved identity.
The listed Newbury 933576 Arabian race is replayed through the normal parser
and its failed-sync entry is closed with the explicit `not run: ARABIAN`
reason. The 89 corrected/recovered fields are marked complete without errors.

For the renamed My Mars / Unbreakable Duke case, identity is corroborated by
foaling date, sire/dam/damsire and five European participations on independently
saved profiles. Finishing position and odds do not establish that identity.
Other name changes retain stable Sporting Life horse IDs and canonical race-day
names. Ordinary predictor fields in existing repaired rows stay unchanged;
all historical form and ratings are rebuilt because the targets changed.

Fontwell 944574 exposes an ingestion edge case: Sporting Life gives terminal
amended position eight among seven starters, with a withdrawn declaration.
The parser accepts only the consistent declaration-bounded terminal demotion.
Malformed or incomplete amendment patterns still fail the existing tests.

`src/verified_result_repair.py` requires complete before/after fields,
independent evidence hashes and explicit confirmed non-starter identities.
It records original rows, verifies unchanged identities and non-target fields,
and supports one caller-managed SQLite transaction for installation rollback.
Reviewed observation timestamps keep the database export identical to the
history actually rebuilt and fitted. No original model, source snapshot or
forward study is rewritten.

Evidence: `verification.json`, `review_roster.json`, `sources/`,
`repair_before.parquet`, `repair_after.parquet`, `new_fields.parquet`,
`prepared_source.json`, and `crosscheck_repaired/` in the 044 audit directory.

The final comparison in `crosscheck_final/` covers **44,237 shared races**.
All **389,999 comparable official positions**, **412,892 comparable starting
prices** and **412,903 nonfinisher classifications** agree, with zero remaining
official-position conflict races. Age agrees on 412,882 / 412,903 runners
(99.9949%); carried/allotted weight on 408,777 / 412,903 (99.0007%). These
remaining source/semantic differences are preserved, not overwritten from a
second source without independent evidence. Physical finish differs from
official order by design after disqualifications and amendments. The audit's
broader renamed-horse diagnostics are not the strict production identity join.

A further RTV parser repair resolves all 12 nonfinisher differences: `casualty`
records can carry numeric tail ordinals, which do not constitute a finish.
The explicit state now overrides the ordinal; physical finishers later
disqualified remain distinguishable. Two casualties had plausible numeric FSP
values and one also passed the sectional shape checks. Those completed-race
measurements and affected field-relative summaries are removed/recalculated.
Parsed quarter tables and numeric caches are refreshed with original files
preserved. `casualty_quality.json` records an exact full-history check that
every retained fitted input is unchanged; no additional model fit is needed
for this correction to the unselected RTV inputs.

## Implemented inputs

There are **26 historical inputs** in
[rtv_history.txt](../../feature_sets/rtv_history.txt). Seventeen means use up to
three recent runs with usable telemetry, omitting missing values per metric.
They cover finishing-speed percentage, top speed,
acceleration, stride length/frequency/range, jumping efficiency, lengths gained
jumping, speed lost, recovery, positive-jump percentage, field-relative FSP and
top speed, late speed ratio, early position, late gain and pace variability.
Five last observations retain relative FSP/speed, jump index, late-speed ratio
and late gain. Three coverage counts and age of the latest observation make
missingness explicit.

The four direct inputs in [rtv_pre_race.txt](../../feature_sets/rtv_pre_race.txt)
are Timeform rating, Timeform stars, pace position and draw advantage. They
require genuine saved snapshots available before both the decision time and
scheduled off minus five minutes. Retrospective card ratings are excluded.
Asynchronous pace/card responses become available at their own saved times;
London race times are converted to UTC using the race date, including BST.

Historical observations have a conservative two-calendar-day lag. An aggregate
expires 730 days after its latest observation became available; its last-three
run summary can include older runs, with observation age exposed explicitly.
Features exclude current and future races; physical nonfinishers
cannot supply finishing-speed percentage. Plausibility limits, sectional
distance/count checks, bad-runner/race flags and measured-peer normalisation
filter telemetry. Unknown values remain missing. Provider ranks and percentage
summaries are not copied as measurements. The collector maps RTV's actual
`sector_position` field to the sectional representation used by the features.

The historical backfill lacks original publication/revision timestamps. Its
lagged values support retrospective development; they are not proof that an
original live decision saw the same values. True pre-race evidence begins with
the saved snapshots. The four direct inputs therefore have no historical model
trial attached to this installation.

Implementation: [rtv_features.py](../../src/rtv_features.py) and
[rtv_feature_collection.py](../../src/rtv_feature_collection.py). Numeric caches
are `data/rtv_cache/feature_runs.parquet` and `feature_pre.parquet`. Materialised
coverage at qualification is 413,224 run observations for 47,408 horses and
2,720 pre-race observations across 45 races. All 26 historical inputs have
usable values in the real-data qualification cohort.

In the rebuilt 2026 history, 92,962 / 100,144 runners (92.8%) have prior RTV
observations. FSP is available on 92,552, top speed on 92,934, late-sectional
ratio on 78,744, and jump index on 30,674 runners. Jump metrics naturally apply
to a smaller racing universe. Coverage is recorded without inspecting outcome
slices in `feature_coverage.json`; missing observations are retained as missing.

## Main pipeline

Normal CLI/app/recorder history refreshes use `src/source_pipeline.py` to
collect Sporting Life and RTV, maintain strict identity links and update the
numeric caches. RTV retains a ten-day revision window even when the SL catch-up
request is shorter. A failed feature refresh stays retryable. Race-day capture
also updates genuine pre-race observations, independently of forecast workers.

The feature stage is part of the ordinary full, incremental and live feature
paths. The source module participates in `fe_signature()`. Historical cache
content is recorded in store metadata and checked by refresh/status; live
signatures include newly captured observations. Future immature observations
do not invalidate an earlier history cutoff; changes to available historical
facts do. A changed historical generation requires rebuilding the store.

[win_with_rtv.txt](../../feature_sets/win_with_rtv.txt) provides the existing 154
inputs plus the fixed 26-input historical family. For explicit research:

```powershell
.venv\Scripts\python.exe train.py --features feature_sets/win_with_rtv.txt
```

The default input list remains the validated 154. A deployment refit cannot
override the validated selection with this research option.

## Measured selection decision

One preset 400-tree development comparison uses the corrected generation,
training 24 September 2023–23 September 2025 and testing Q4 2025. Both arms use
the same 3,234 whole races / 30,094 runners. Exactly two new fits are made;
there is no subset, quarter, seed or capacity search.

| Measure | Existing 154 | With fixed RTV 26 |
|---|---:|---:|
| Race log loss | 1.84690718 | 1.84372079 |
| Runner Brier | 0.08545918 | 0.08534314 |
| Established-handicap race log loss, 1,071 races | 1.98538384 | 1.98148413 |

Date-cluster 95% interval for overall log-loss change is
**[-0.00717584, +0.00095211]**. The point estimates pass the preset point rule,
but uncertainty includes no gain. **No RTV input is selected for the deployed
model.** This is useful implementation and a modest exploratory lead, without
a predictive improvement, market-parity or betting-return claim. Do not repeat
a subset search on these inspected outcomes.

Evidence: `feature_plan.json`, `screen_reference_v2/`,
`quick_screen_v3/report.json`, `predictions.parquet`, and `screen_decision.json`.
This development screen used the preserved cache before the later casualty
classification repair. Its original scores are not relabelled as an evaluation
of the corrected cache; the corrected RTV family remains unselected.

## Qualification and installation

The rebuilt feature signature is `b33c884d9edd8a6f`. All 154 retained inputs
are exactly equal to the former generation on 343,034 runners before the first
repair, 11 June 2023. Later histories intentionally reflect corrected targets.
Real-data checks on 5,499 runners establish permutation, subset/append and
current/future-label/price independence. Focused tests cover cache signatures,
quality filtering, snapshot causality, complete-field repair rollback and
normal pipeline integration: the final six-file focused suite passes **59
tests**. A separate full-source transaction replay reproduces the exact rebuilt
CSV SHA256 `ee6eb3041bc228226669fa539f22d27cf3bcb91174ac46c31edcfa53ca35de77`.

The required current-data refit retains the existing statistical recipe and
154 selected inputs, fitting through 28 October 2026. Original historical
validation models and scores remain unchanged and explicitly labelled pre044;
the corrected refit carries no newly invented held-out score. Native component
checkpoints make interruption resumable without repeating completed fits.

The completed refit is `20261030_135650_092286`, with 3,705 win trees and
2,990 place trees; all eight native components completed. Model SHA256 is
`3d25ece407c92f85f19f6bf526ebed1cf9797c7cee2ea34a21fd5843533c3b74`.
It fitted 668,436 runners in 71,537 complete fields through 28 October. Its
normalized loaded-source signature `9c1ee3e430f37b5f` includes the complete
29 October source snapshot; the separately checked fitting prefix through
28 October is `bf700d4ffadb63a6`. Loading a later causal history and fitting
through an earlier declared date are distinct scopes.

The final deep check caught an export-contract issue: the repair preview had
retained SQLite's internal `scraped_at`, which ordinary database refreshes
deliberately omit. The original preview and refit metadata are preserved.
`source_contract/` qualifies its removal: every non-timestamp processed value
and all 154 persisted fitted inputs match exactly over all 755,357 rows, and
native model bytes remain identical. Store and model provenance now use the
ordinary timestamp-free contract. The export helpers are corrected and a
regression test verifies that changing a timestamp cannot alter the export,
while a result correction does. This correction requires no further tree fits.

The now-retired `install_rtv_generation_044.py`
prepares an exact actual-source preview, preserves rollback files, locks the
store and opens one SQLite transaction. It publishes source code, numeric
caches, canonical CSV, rebuilt histories and the fitted model before committing
the source batch. `verify_rtv_generation_044.py` (removed 2026-11-06, in git history)
checks win/place bounds and mass, single/batch equality, current-label/price
independence, and exact staged/installed saved-card inputs and probabilities.
The saved-card test is operational replay, not prospective evidence.
Staged and installed QA pass on 106 complete fields / 1,010 runners and a saved
five-runner card, with single/batch maximum error zero and exact selected live
inputs and probabilities. Full database-content readiness passes at 17:23 UTC
under the normalized contract; database, store, code and promoted model agree.
The normalized contract has an additional `qa_contract_after/report.json`
check against the original staged selected inputs and probabilities. Earlier
interim publication/readiness receipts remain preserved, including the failed
deep check that exposed this issue.

Only after installed serving QA does a fresh source/model-pinned recorder study
replace the active pointer. Existing studies stay immutable. The Windows helper
retains the existing daily/logon triggers and verifies the new action.
`published.json`, `qa_before/report.json`, `qa_after/report.json`,
`recorder_registration.json`, `scheduler.json` and final readiness receipts
describe the completed installed state; their existence and success, rather
than this prose, determine completion.

Completed **30 October 2026 at 17:28 UTC**. The final receipt is
`installation/final_status.json`; normalized source/store/model readiness and
exact serving parity pass. `HorseForecastRecorder_038` is running with its
original daily/logon triggers and study
`20261030T172001Z_rtv_features_044_20261030_135650_092286`, eligible 31 October.
Protocol SHA256 is `41767cd739d3faaf1025722524919146558180c01ebc3317cf0b7ce7f99c8187`.
The heartbeat and successful market discovery are verified with no recent
errors. Fresh RTV capture covers 45 races / 43 pace maps and materialises
7,444 genuine pre-race observations. Earlier studies and every interim audit
remain preserved. There is no further fit or rebuild pending for this rollout.
