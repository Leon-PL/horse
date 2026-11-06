# 5 November 2026 (18:57): faster ATR pacing and service-error retries

The live backfill now runs with `--request-delay 2` (plus jitter). A 20-race
trial measured 414/hour versus 332/hour at the previous three-second setting,
about 25% faster while actively collecting. All three tabs and identity checks
are unchanged. Website access still prevents a firm full-history completion
date; the uninterrupted-processing projection is now roughly six days.

One transient 503 during the trial exposed an excessive global one-hour pause.
The exact endpoint recovered to HTTP 200. Temporary service/network faults now
use short escalating global waits and longer per-race retry delays, allowing
healthy races to proceed. Actual throttles/challenges retain long backoff;
server Retry-After is respected. The failing field subsequently succeeded.

108 relevant tests and a 41-test follow-up pass (overlap); parsing, validation
and storage ASTs match the previous qualified version. Records were preserved
and the runtime hash update is audited in `retry_policy_deployment.json`.
Current FE **f91cd05955e05228**; no feature build, fit or promotion was run.
Worker PID at restart: **1340**, started 18:55 UTC. Live progress and pacing:
`data/atr_cache/rebuild_v2/status.json` and `launch_options.json`.
On resume use `.venv\Scripts\python.exe -m src.atr_rebuild --request-delay 2`;
preserve the queue and its current fingerprints. Details and all evidence are
in [the rebuild report](../operations/ATR_REBUILD_2026-11-05.md) and the ATR rebuild audit
directory, especially `speed_result.json` and `speed_launch_final.json`.

# 5 November 2026 (17:58): corrected ATR full rebuild resumed after shutdown

[Collector changes, backfill and recovery](../operations/ATR_REBUILD_2026-11-05.md).
The omissions and mixed-unit bug below are fixed. Extraction now preserves
frequency curves, Front markers, race-level pace, separate instantaneous speeds
and acceleration seconds, source chart attributes and payload provenance.
The label-aware derivation also fixes reads of older mixed speed arrays.

`src.atr_rebuild` reconstructs a fresh dataset for 62,432 UK races, using the
old cache only for 28,186 native-ID hints. Old parsed observations and no-data
decisions are not imported. Retain the existing accepted cache and backup
until its replacement is qualified. Source identity and time checks reject
wrong fields. Verified tracking omissions, including unreported nonfinishers,
retain available source runners with explicit coverage metadata.

152 relevant tests and lint pass. The pilot qualified 31 fields / 253 records;
the background run reached 37 fields / 295 before shutdown. Both SQLite
databases passed integrity checks; pinned code matches. Consistent backups and
the derived daily exports were recovered. Restarted at 17:56 UTC: worker PID
17748; observed 46 resolved fields / 381 records at 17:57, still advancing.

Live progress: `data/atr_cache/rebuild_v2/status.json`; current log:
`worker.stderr.log` alongside it. Normal refreshes report this active worker
instead of starting a second collector. Discovery challenges wait an hour
while known-ID collection continues; endpoint blocks back off globally. The
local process cannot run while the computer is off, but committed races survive
restart. Use `python -m src.atr_rebuild` in the pinned project environment to
resume; do not run `--prepare` again or clear its queue.

Evidence: `data/audit/atr_rebuild_20261105/`, especially `tests_qualified.txt`,
`pilot_qualification.json`, `latest_recovery.json`, `resume_launch.json` and
`resume_source_health.json`. Current FE **7ee6994aa3026400**; no feature rebuild,
model fit or promotion was performed. The earlier seven-hour estimate covered
recent gaps only; the fresh full reconstruction is a multi-day job.

# 5 November 2026 (09:18): BSP provenance and ATR endpoint repair

**09:33 content follow-up:** no ATR job is running; the 10:14 cooldown does not
restart it automatically. A read-only audit found omitted per-section stride
frequency, race-level pace/finish percentage and Front markers. It also found
135 rows mixing 0–30 mph acceleration time into the speed array, which can
become an invalid closing speed downstream. Fix this before broad backfill or
feature rebuilding. All 30 new races have retained HTML for offline correction;
older raw-response availability is unverified. Evidence: `content_audit.json`
in the ATR diagnosis audit directory; details appended to the report below.

[Repair and remaining access limitation](../operations/BSP_AND_ATR_2026-11-05.md).
The actual BSP cache combines Betfair Historical BASIC imports with public daily
GB/Irish WIN CSV additions; November 4 coverage remains complete.

ATR now discovers exact native race IDs from daily result articles, retains
successful mappings across sessions, and calls `/ajax/sectionals-razor/` for
times, tools and stride. Current country-segment horse URLs preserve their IDs;
summer UTC conversion no longer falls back to a different race. Raw responses
and actual HTTP statuses are retained in `data/atr_cache/raw.sqlite`. Blocks on
a later tab cannot accept an incomplete race. Both backfill paths defer races
younger than 48 hours, retaining their pending status.

Two normal coordinator batches recovered **30 races / 248 runner records** on
2 November. All 248 match canonical records and carry all three arrays. All
232,149 previous rows and dtypes are exact; accepted cache now **232,397**.
First batch: 25 races in 153.012 seconds; second: five more in 77.275 seconds,
including backoff when the 1 November index was challenged. **3,936 pending**,
including 60 within the publication allowance; next retry **10:14:09 UTC**.

**Unattended recovery remains partial:** saved IDs avoid repeat page challenges,
but uncached-day index access is still intermittent. No new false absences were
recorded. Another 379 older absence decisions lie before this recovery window
and need historical revalidation; their keys are retained separately.

130 regressions pass in the pinned project environment; lint and changed-file
whitespace checks pass. Evidence/backups: `data/audit/atr_diagnosis_20261105/`,
especially `final_verification.json`, `recovery.json` and `recovery_next.json`.
Current FE is **aed36c32a39b83d9**, installed store **1355eccb0838ad9b**.
New ATR inputs and the collector fingerprint require a later feature refresh;
no feature build, model fit or promotion was performed here.

# 5 November 2026 (07:47): source health, recovery and atomic ATR publication

[Follow-up report](../operations/STAGE1_SOURCE_HEALTH_2026-11-05.md). Targeted refreshes
now preserve untouched providers and timestamps in `source_refresh/status.json`;
`last_run.json` records the exact selected operation. Older backfills retain
newer coverage. Busy/unreadable metadata is distinguished from unsigned stores.
ATR Parquet and no-data JSON now use atomic publication, with Parquet row-count
validation before replacement.

The live retry checked 48 SL pages with **no factual changes** (revision 2314,
same CSV). BSP added **496 prices**, fully covering 4 November's 47 finished
fields; all 818,601 original price rows and timed trades match the backup.
ATR added **11 rows** for Musselburgh 12:05 on 4 November, with all starters
matched and all 232,138 old rows preserved. A subsequent 50-race budget stopped
on the next blocked field, Nottingham 14:20: **3,966 pending**, next retry after
**08:25:46 UTC**. It added no false confirmed-absence entries.

The earlier identity-compatible store remains installed, but it now needs
refreshing: its signature is `1355eccb0838ad9b`, current code is
`13fb7f7dd1e187e0`, and the ATR input cache changed. Canonical data still matches.
The atomic-publication edit shares a fingerprinted module with ATR joins;
separating those responsibilities is an identified follow-up. Batch source
recovery before the next full-history build. No build, fit or promotion was
performed in this source pass. The saved model still needs an evaluated refit.

139 source/publication/diagnostic regressions and 52 serving/registry regressions
pass (overlap). The private 232,149-row atomic ATR round trip preserves all
values/types without touching the accepted cache. Evidence and backups:
`data/audit/stage1_followup_20261105/`; final readiness: `readiness.json`.

# 5 November 2026 (earlier): identity repair and compatible history installed

[Migration and preservation report](../operations/STAGE1_IDENTITY_2026-11-05.md).
The canonical migration and downstream identity joins are installed. Race
688054 has all 15 runners, retaining both Sierra Nevada IDs. All 757,083
original rows and row IDs were preserved by the migration; only the missing
horse was added. The normal post-midnight refresh then added 496 runners and
updated one reported `horse_runs` value, yielding 757,580 runners / 80,536 races
through 4 November. Old historical SL review cases are zero; one recent
Nottingham page remains `DORMANT`. ATR remains blocked with 3,967 pending races.

The compatible full history is rebuilt and installed: FE `1355eccb0838ad9b`,
data `d94b262598f1663c`, through 4 November. All 757,580 canonical runner IDs and
outcomes match. Source/code guards and external-input fingerprints passed;
the previous five-file store is backed up. Build/qualification/publication took
5,539.374 seconds. Receipts: `data/audit/runner_identity_20261104/qualification.json`
and `publication.json`; progress and timing are retained in `feature_build.log`.

The 338-test regression run, subsequent 30-test focused run and 65-test index
follow-up pass, including namesake build/append/live paths. The final query-plan
review restored a dedicated race lookup index lost with the old uniqueness
constraint; indexed lookups retain identical rows and factual revisions.
No model fit or promotion is part of this
source repair. The 057 notes below describe the separate overnight installation;
its proposed rebuild is now complete here. The real-field future-card fixture
passes all 15 identities, reversal invariance of 129 selected inputs and unchanged
persisted states. The generation comparison finds differences in 83 selected
inputs, often tiny propagated rating changes, with some larger individual
changes. The old model therefore needs an evaluated refit. Evidence:
`feature_impact.json` and `live_identity.json` in the same audit directory.
The final idle-store deep readiness check confirms feature store **OK**, with
matching code and canonical data. Overall readiness remains blocked by the
recent dormant field, incomplete ATR collection and model generation mismatch
(`bc357c18a5e5a261` vs `1355eccb0838ad9b`). Receipt: `readiness_final.json`.
The earlier concurrent status probe hit the live-fixture store lock and is
retained separately as `readiness_concurrent.json`; the signature is present.

# 4 November 2026 (night): 057 installed, not yet serving

Production is **`20261104_235300_444862`**: the exploded win objective with the
same 129 inputs as 056, fitted through 3 November. `WIN_OBJECTIVE = "exploded"`
in `config.py` drives the win fit; `src/feature_selection.py` and
`scripts/benchmark_recent_probability.py` stay pinned to `cross_entropy`.

| Check | Result |
|---|---|
| Hold-out vs 056, 3,268 races both runs scored | 1.8512 vs 1.8557, −0.0046 [−0.0097, +0.0003] |
| By month | August −0.0032, September −0.0045, October −0.0059 |
| SP market on those races | 1.7722 |
| Rebuilt store vs the 056 store | 755,309 rows and 133 columns identical; 1,774 new rows |
| Settled-field serving check | 102 races, 968 runners pass |
| Live-card serving check | Not run; it needs the rebuild below |

The candidate trained on five more days than 056, because a source refresh
moved its hold-out from 1 August–29 October to 6 August–3 November. The
comparison uses only races both runs scored.

**Not servable yet.** At the initial installation the promoted run and store
both carried FE `bc357c18a5e5a261`, while identity work had changed the feature
code. The 5 November rebuild and input comparison are now complete above.
The installed store carries `1355eccb0838ad9b`; 83 selected inputs change
somewhere, so the next model step is an **evaluated refit** against that
generation. The source repair has not fitted or promoted a model.

After model work, run `scripts/project_status.py` and
`scripts/verify_live_upgrade.py`, retaining the serving receipt under
`data/audit/install_057/serving/`. The source gaps in the 5 November report
remain visible and require their own resolution.

Timing, with other jobs running: rebuild about 78 minutes, evaluated fit about
2 hours, deployment fit about 2 h 25 min at 6 threads. The refit needs
feature and training code left alone throughout. Receipts:
`data/audit/install_057/`.

# 4 November 2026: stage 1 audit, consolidation and installed recovery

**Historical recovery follow-up:** [report](../operations/STAGE1_BACKLOG_2026-11-04.md).
The shared `--sl-backlog-max-races` route closed 3,456 of 3,457 historical
jobs in 387.188s using 485 listing loads and five race-page requests. All
757,083 accepted rows match the pre-run backup exactly; revision stays 1816.
Only race 688054 remains pending, explicitly for identity review. The Data
page exposes the same bounded recovery path. Proof and backups are in
`data/audit/stage1_backlog_20261104/`.

Final checks: 234 source/parsing/readiness/provenance regressions pass; all
3,451 listing closures are pinned to 471 retained response hashes. The five
remaining closures are independently verified Arabian result pages. All 27
old 404 jobs were resolved using actual listing statuses. ATR was retried at
23:18 UTC and remained blocked (0 fetched, queue 3,934); next retry after
00:18 UTC on 5 November. Normal source/check timestamps are preserved.

Identity impact is now reproduced with the actual processing/rating code:
the source's 15 runners become 14 under the name uniqueness constraint;
ID-based storage retains 15, but the current race/name rating join produces
17 rows. Most horse form and rating state already prefers IDs. Coordinate
storage, rating-output keys, append/live joins and repair writers before
installing an identity migration; preserve source namespaces and then rebuild
and qualify features/models. The canonical schema remains unchanged.

See [the audit](../operations/STAGE1_DATA_PIPELINE_AUDIT_2026-11-04.md),
[implementation](../operations/STAGE1_IMPLEMENTATION_2026-11-04.md) and
[live recovery follow-up](../operations/STAGE1_RECOVERY_2026-11-04.md).

- SL: 757,083 runners / 80,489 races through 3 November, 1,774 records added.
  RTV and BSP also extend through 3 November; BSP added 2,532 prices while
  preserving all archive prices and timed trades. FX reaches 4 November.
- ATR's 3,934-race queue from 21 June persists. The provider refused requests;
  routine refreshes now make one retry and persist a one-hour cooldown.
- Short source operations run independently after canonical SL collection,
  with persisted progress. Unchanged BSP validation reuses a content-fingerprinted
  receipt (~0.3s); batched ATR key parsing reduces its cached check from ~84s
  to ~4s in the measured profiled runs. All-source recent checks stay partial.
- Fifteen older terminal jobs were closed through the shared collector.
  Race 688054 is now explicitly pending identity review: the source has two
  IDs named Sierra Nevada, but the schema is unique by race/name. Accepted
  facts remain untouched. There are 3,457 historical pending jobs. The identity
  census found 105 names with multiple source IDs, requiring provenance review.
- The deep status check observed the store built at 19:15, through 3 November,
  FE `bc357c18a5e5a261`, matching code/data. The promoted 056 model still records
  FE `69babeb9c4262ee2`; model qualification is outstanding. No new fit or
  promotion was performed by this source-recovery continuation.
- Evidence and database/source backups:
  `data/audit/stage1_continuation_20261104/`.
- Final integrated regression checks: 257 passed, one pandas warning; source
  lint and whitespace checks pass. Prior full-suite baseline failures remain
  documented in the first implementation report.

# 2 November 2026 (evening): model ideas 057–060 tested, nothing installed

[docs/research/MODEL_IDEAS_057_060.md](../research/MODEL_IDEAS_057_060.md) holds the four
brief ideas (the LLM idea was not run). They were confirmed on 2025Q3–2026Q2,
11,905 races, against production at 1.83135.

| Idea | Race LL change vs production | Verdict |
|---|---|---|
| **057 exploded softmax** (`src/race_objectives.py`), stage weights (1, 0.25, 0.0625) | −0.0046 [−0.0064, −0.0026], 4 of 4 quarters | Confirmed |
| 058 Thurstone figure and spread (`src/thurstone.py`), alone | +0.0048 | Worse |
| 058 in a blend | −0.0017 over 057 (near-copy control 0.0002–0.0006) | Adds information |
| 059 luck weights | −0.0003, interval crosses zero | Null |
| 060 learned connection profiles | +0.0001, interval crosses zero | Null, priced |

**062, 3–4 November:** CatBoost pairwise ties 057 and is not adopted. On the
soft target at rate 0.05 it is −0.0037 [−0.0064, −0.0011] against production and
+0.0009 [−0.0014, +0.0031] against 057. Blended with 057 it gains −0.0016, but
added to 057 + figure only −0.0008, with an interval that crosses zero. Results
only at fast settings gave +0.0169. CatBoost lives in `data/tooling/catboost`,
outside `uv.lock`.

**Done 4 November:** the exploded objective is now in the production win
learner (see the top entry). The other research modules are not wired into FE
or training. Every run resumes
through `bash scripts/run_model_ideas.sh`.

# 2 November 2026 (afternoon): Timeform comment score installed (056)

Production is **`20261102_143544_663385`**, 129 inputs
(`feature_sets/win_selected_056.txt`): the 048 inputs plus `rtv_text_logit`,
`rtv_text_prob` and `rtv_verdict_rank`. It is a deployment refit through
2026-10-29 from the evaluated run `20261102_130936_036383`. The store is
rebuilt (FE `69babeb9c4262ee2`, data `826c9478e6f726b8`). `project_status.py`
is ready and `verify_live_upgrade.py` passes; receipts are in
`data/audit/install_056/`.

- **The stage:** [src/rtv_text.py](../../src/rtv_text.py) fits one text model per
  quarter (2021Q1–2026Q4) on races up to 7 days before the quarter starts.
  Models are in `data/rtv_cache/text_models/` and history scores in
  `feature_text_scores.parquet`.
- **Keeping it current:** `rtv_feature_collection` collects text, fits missing
  quarters and rescores with each RTV history refresh. Live cards are scored
  from the pre-race snapshot. `python -m scripts.backfill_rtv_text [--refit]`
  rebuilds everything.
- **Hold-out result:** −0.0001 [−0.0045, +0.0045] race LL against the 126-input
  run on the same 3,431 races. This is inconclusive; the interval contains
  052's −0.0029. It was installed as the user asked, and they were offered a
  revert.
- **Reverting:** point `config.FEATURE_ALLOWLIST` back to
  `win_selected_048.txt` and retrain. The 054 run was fitted on the previous
  store signature.
- **Live gotcha:** comment scores on a card need that day's Racing TV
  snapshot. The picks app captures it; scripts that call
  `data_scraper.get_scraped_racecards` directly must call
  `source_pipeline.capture_rtv_snapshot()` first.

The five untested model ideas raised on the same day are written up for
another session in [docs/research/MODEL_IDEAS_BRIEF.md](../research/MODEL_IDEAS_BRIEF.md).

# 2 November 2026: 048 inputs and 051 fixes installed (054)

Production is **`20261102_063010_291745`**. It is a deployment refit through
2026-10-29 on the 126 inputs in `feature_sets/win_selected_048.txt`, built
from the evaluated run `20261102_050631_314757`. The store is rebuilt (FE
`32e71a2cdff8a8b5`, data `826c9478e6f726b8`, 755,309 rows). `project_status.py`
is ready, and `verify_live_upgrade.py` passes; receipts are in
`data/audit/install_054/serving/`.

On the held-out window 2026-08-01..10-29, race LL is 1.8492 against 1.7656 for
the SP market.

Gotcha: a script that edits `results` directly must re-export
`data/raw/race_results.csv`, because store builds read the CSV and not the DB.
The 051 repair did not, and the CSV was re-exported before the rebuild.
`repair_overfull_fields.py` now does it itself.

At that point the 052 comment score was not in production; 056 installed it
later the same day (above).

# 1 November 2026 (late): 049–051, config on the 048 inputs, store stale

`config.FEATURE_ALLOWLIST` now points at `feature_sets/win_selected_048.txt`.
At the user's request there has been no store rebuild and no retrain at that
point. The code's FE signature was `32e71a2cdff8a8b5`, against the store's
`bde9317da2c3c963` (since rebuilt, see above).

- **049:** race-type and course specialist models are refuted (ledger 049).
- **050:** the model is calibrated and uses its inputs fully. The gap to the
  market is information it does not have. The data matched raw SL pages
  apart from two defects.
- **051:** both defects are fixed:
  - 48 non-runner rows were deleted from the DB;
  - `reference/connection_aliases.json` is applied in `clean_data`.

  SL commentary and verdicts are now recorded in `runner_text_observations`.
  The same Timeform text exists in the RTV cache from 2020-09, read with
  `source_pipeline.load_matched_text`. See
  docs/MARKET_PERFORMANCE_PROGRESS.md (050, 051).

- **052 (done):** a walk-forward text model on the Timeform comment and
  verdict (`scripts/text_model.py`). Its out-of-fold score improves the
  126-input model by 0.0029 [0.0010, 0.0049], in 4 of 4 quarters. It is not
  in the feature store: adopting it needs a text-scoring stage that fits on
  history and scores live cards, then a store rebuild.
- **053:** the rest of the RTV card adds nothing on top of the comments
  (ledger 053).

# 1 November 2026: walk-forward input selection (048), store rebuilt

The store was rebuilt on the 047 fixes (FE `bde9317da2c3c963`, data
`2ea0d385c37e8d1c`, 755,357 rows to 2026-10-29, done 13:31). A parallel RTV
session then made two changes: it backfilled the Timeform card to 2020-09 and
pinned the time-figure par (`src/rtv_features.py`). The store is stale again
(code `bedfb3b5c0208b6c`) and needs one more rebuild. The promoted model
`20261030_135650_092286` is stale as well and needs a retrain before
serving.

`scripts/select_features.py` (primitives in `src/feature_selection.py`)
re-derived the win inputs by out-of-sample race log loss and produced
`feature_sets/win_selected_048.txt` (126 inputs). On untouched quarters
2025Q3–2026Q2 it beats `win.txt` by 0.0194 [0.0155, 0.0236] at production
capacity, in 4 of 4 quarters. `config.FEATURE_ALLOWLIST` still points at
`win.txt`. Adopting the new list means changing that line and retraining; it
has not been done. Those confirmation quarters have now been used, so a later
re-selection needs new ones. Receipts are in
`data/research/feature_selection_048/`. See docs/research/FEATURE_SELECTION_048.md.

# 30 October 2026 (evening): recorder and source pins removed

The project is in an experimentation phase with no live model. Removed: the
`HorseForecastRecorder_038` scheduled task (definition saved in
`data/audit/recorder_removal_20261030/`), the `src/forward_*` recorder and study
modules, `issued_evaluation`, `confirmation_planning`, the Forecast record page,
the recorder/study-transition scripts (install_/finish_/adopt_/restart_) and
their tests. Study data under `data/forecasts/forward/` was left in place.

Iteration costs reduced: `fe_signature` now hashes each FE source's AST, so
comment/docstring/formatting edits no longer force the ~1 hour store rebuild,
and `train.py --fit-as-of` no longer requires a validation certificate (one is
still checked when passed). The store and runs `20261030_122253_088153` /
`20261030_135650_092286` were re-stamped `b33c884d9edd8a6f` -> `b20404f117a63897`
after proving the old hash reproduced the stamp from the committed sources;
backups and receipt are in `data/audit/recorder_removal_20261030/fe_signature_migration/`.
The FE signature values quoted below are the pre-migration ones.

`.venv` was then rebuilt from `uv.lock` (`scripts/tooling.ps1 venv`) on the
project-local Python 3.14.0 in `data/tooling/python`, replacing the broken
`C:\Python314` base. Live-card and stored-history predictions were
bit-identical before and after; the old env is at
`data/tooling/venv-old_20261030` for rollback. See docs/operations/DEVELOPMENT_TOOLS.md.

# Installed044: RTV repairs, feature pipeline and current refit

Completed30October2026 at17:28UTC. Authoritative receipt:
`data/audit/rtv_repair_features_044/installation/final_status.json`.
See `docs/research/RTV_FEATURES_REPAIRS_044.md`. Full database-content readiness passes
17:23UTC; installed native inputs/probabilities exactly match staged QA on
106complete fields /1,010runners and a saved five-runner card.

Production `20261030_135650_092286`, FE `b33c884d9edd8a6f`,154selected inputs,
3,705win/2,990place trees fitted through28October. All8native components done.
Artifact SHA256 `3d25ece407c92f85f19f6bf526ebed1cf9797c7cee2ea34a21fd5843533c3b74`.
No further fit/rebuild is pending. Current source/store through29October has
755,357runners /80,302fields. Normal source signature `9c1ee3e430f37b5f`;
fitprefix through28October `bf700d4ffadb63a6`. Canonical CSV SHA256
`39936f113cd7dc66dcf63050fd331b34f31053e47219fad7f4f159f98d1d68da`.

75fields repaired:336positions including19winners;4confirmed nonstarters
removed.14missing fields /132starters recovered.89reviewed sync entries are
complete with no errors; excluded Arabian933576 closed with retained reason.
12RTVcasualty ordinals are correctly treated as nonfinishes. Final comparison
covers44,237shared fields:389,999official positions,412,892SPs and412,903
nonfinisher classifications allagree.21age differences and historic carried/
allotted-weight semantics remain documented; no unverified overwrite.

26historical telemetry inputs +4true pre-race snapshot inputs are implemented
in the ordinary full/incremental/live pipeline, cache fingerprints and status.
`feature_sets/win_with_rtv.txt` exposes the fixed26family via train.py --features.
Default154selection remains unchanged. One400tree development pair gives small
uncertain gain (date95%LLdelta[-.00717584,+.00095211]); no RTV selection or new
accuracy/market-edge claim. Original trial used pre-casualty-fix cache; preserved.

Two final source-quality corrections do NOT require more fits: casualty masking
changes only unselected RTV inputs; the ordinary CSV contract excludes internal
scraped_at. Every retained input is EXACT over755,357rows after each correction;
modelbytes unchanged. Old stores/meta/helpers/previews and failed/interim
readiness receipts are preserved under casualty_quality/ and source_contract/.
Sourcecontract migration is explicit in current model metadata. Normal refresh
and the recorder now pass the corrected contract. Focused59integration checks,
80parser checks and timestamp-export regression pass.

Task HorseForecastRecorder_038 RUNNING, daily/logon triggers preserved. Active
study `20261030T172001Z_rtv_features_044_20261030_135650_092286`, first eligible
31October. Protocol `41767cd739d3faaf1025722524919146558180c01ebc3317cf0b7ce7f99c8187`.
Verified heartbeat PID10104 and successful discovery, no recent errors. RTV
history and snapshot healthy; latest capture45races /43pace maps,7,444genuine
pre-race observations. Earlier043 and interim044studies remain immutable.
Actual production store files require the user token for diagnostics; use
sandbox escalation if restricted-token reads fail. Do not loosen file ACLs.

# Installed043: Racing TV source integration (historical)

RTV/SL cross-check and normal source-pipeline integration completed at 09:40 UTC.
See `docs/research/RTV_PIPELINE_043.md` and
`data/audit/rtv_pipeline_043/installation/final_status.json`.

CLI/app training, Data catch-up, picks and the recorder now collect RTV through
`src/source_pipeline.py`. Separate dated tables, exact SL/RTV identity links and
comparison reports are published atomically. The recorder captures expiring
pace/prompts every 15 minutes on a separate worker. Status monitors RTV health.
No RTV predictor has been selected; no model fit or canonical repair occurred.

Full comparison: 44,223 matching races / 412,771 comparable runners. SP agreement
100%; age 99.9949%. There are 19 additional RTV fields and 336 official-position
conflicts in 75 races. Fresh SL pages corroborate 16 stale database positions in
three sampled fields (Wetherby/Newmarket/Galway). Fontwell 944574's SL amendment
is rejected by the existing parser (position eight among seven starters).
Full queue and source receipts are preserved. Canonical repairs and replacement
model qualification for this newly identified queue remain separate work.

Production remains 20261030_015917_823047, FE 3697230d16bbd0d8, 154 selected
inputs. Canonical CSV/model bytes are unchanged. Deep readiness passed 09:38:44
UTC; native pinned/registry forecasts match exactly. Collector, source, recorder,
training and signature checks pass. Zero new tree fits.

Task HorseForecastRecorder_038 is running with preserved daily/logon triggers,
study `20261030T093356Z_rtv_sources_043_20261030_015917_823047`, eligible 31 October.
Protocol `9f3f5c8fb2e09b6439c1a0b2cb657d5c59de6ca4ffaf5068e3435f05d9ab9999`.
Final receipt verifies live heartbeat/discovery, source health and unchanged
model/history hashes. Earlier studies remain immutable. Initial live capture:
45 races / 43 pace maps; yesterday's 38 fields / 387 runners fully linked.

# Installed042: official-result corrections and current refit

Published30October2026 at07:17UTC in the actual project. Production is
**20261030_015917_823047**,154 selected inputs,3,992win/2,987place trees fitted
through28October. Artifact SHA256
`f8497fadeff3370cceaffb5cddca6b2a3bab50acd0506d3700322a1fb26076f4`.
FE stays `3697230d16bbd0d8`; corrected training-prefix signature
`3c06942ccdc4e35b`. All8 native components and the one current refit completed.

Canonical DB/CSV and rebuilt Elo/Glicko/form histories now include164 complete
field repairs /1,534 source rows,131 changed winners and463 changed placings.
Normal scraping honours final amendments and rejects explicitly provisional
results. Every future ordinary training run consumes these corrections.
Three uncorroborated exchange disputes remain quarantined. The initial70-field
stages and old pre042 predictive scores remain preserved, not silently relabelled.

Installed serving QA passes115 complete fields /1,108 runners and the saved
two-runner public card. Single/batch error0, current-label/price independence,
exact early causal prefix and staged/installed live inputs/probabilities pass.
The recorder added387 new runners /38 races for29October without another fit:
current history755,229 runners /80,288 races; signature `8cbba4e89e351d82`.

Task `HorseForecastRecorder_038` is running with study
`20261030T072041Z_amended_results_042_20261030_015917_823047`;
protocol `e9c859ecc6c3758cf3a235e11e7359e9ba6a2bcec618b5f54b8e246845eb4f82`,
eligible31October. Initial deep readiness passed08:19UTC. A separate daily FX
gap was fixed08:43UTC: verified ECB observations28/29October were added with
every earlier observation identical. Only73 Irish runners' prize comparison
input changes, after the fitted cutoff; every other selected input and every
fitted input is unchanged. No additional model fit. Project-local `.env`
`SSL_CERT_FILE` points to the installed verified CA bundle; normal verified
HTTPS now passes, without disabling verification or editing mathematical code.
Latest live QA passes106 recent fields /1,010 runners plus Newmarket's six-runner
11:35 card. Same study/task restarted08:49UTC and discovery succeeded08:54UTC.
Final deep readiness passed08:57:55UTC, no blockers; closure completed08:59UTC.
Authoritative final receipt: `consolidated/installation/final_status.json`.
Initial readiness is preserved in `fx_refresh/readiness_before.json`.

Authoritative evidence: `data/audit/result_amendments_042/consolidated/` and
`docs/research/AMENDED_RESULTS_042.md`. Publication, both initial serving QA receipts and
the completed `fx_refresh/` qualification/installation/transport/live-QA receipts exist.
65 forward checks,44 status/health checks,6 publication integration checks,
30 transaction checks and26 runtime/serving/snapshot checks pass (overlap applies).
All042 work is COMPLETE. No fitting or installation remains queued.
Do not repeat collection, history rebuilding, fitting, publication or registration.
The migrated historical reference `20261029_224707_518692` is not production.

No predictive gain or market parity is claimed for042. The next cheap modelling
lead is a shorter training history: >3-year-old rows are46.12% of rows but1.44%
of current weight. A tiny exact tree-prefix check passed; reconstruct the native
arrays/key before reusing the042 capacity checkpoint as a capped baseline.
No window change, new predictor family or quarter sweep is queued.

041 is COMPLETE_NOT_SELECTED: raw place-head information receives weight0 in
the fixed win combination. Existing December heads reused; zero new tree fits.
See `docs/research/AUXILIARY_PLACE_WIN_041.md`, `data/audit/auxiliary_win_041/final_status.json`.

# Historical feature-speed rollout (29 October 2026; superseded by042 weights)

The user requested the tested optimization in the actual project. It is installed
in `C:\Users\Leonp\Documents\horse`, not only in the temporary worktree.
Current run **20261029_222758_104473**, FE **3697230d16bbd0d8**. Artifact bytes
are identical to parent **20261029_141132_815571**; zero fits and unchanged probabilities.
Full 754,842-row/1,187-column rebuild,
processed inputs and carried state match exactly. 88 main-project tests pass.
Installed saved-card replay and recent-field constraints pass. Full readiness
passes at 2026-10-29T22:45:22; recorder task/source pins were updated and restarted.
New study: **20261029_feature_speed_20261029_222758_104473**. Preserve all earlier studies and receipts.
See `docs/research/LIVE_FEATURE_SPEED_20261029.md` and the speed audit's installation
`final_status.json`. The previous 038 operational identifiers below are historical.

## Preserved 038 handoff

> Research039–040 completed29 October2026: both new win-feature proposals stopped after their small checks; two new fits total, exact cached baseline reused. No canonical/recorder changes. See [results](../research/ESTABLISHED_CONTEXT_039_040.md) and the039–040 section below. The separate live-speed installation was still rebuilding; its owner controls that rollout.

# 038: all six predictive-review findings

The user explicitly requested work on all six issues. No agents were delegated.
The new model is installed and the final recorder is verified running as of
29 October 2026, 14:51 UTC. No training or test process remains active.

- Production **20261029_141132_815571**, installed 14:18:24 UTC on 29 October,
  has SHA256 `73747ec2166fb6598687c9dedbffbaa37f20694256933cda4d03e864b4e156b0`.
  Same 154 inputs, 3,992 win / 2,991 place final trees, fitted through 27 October.
  Histories: 754,842 runners / 80,250 fields through 28 October. FE signature
  `11bf1232d575eea6`; current-history signature `554fd1046c28a932`.
- Recent place calibration is installed and integrated with ordinary training,
  save/load and both serving interfaces. Current calibration: 30 July–27 October,
  3,427 fields / 32,498 runners; temporary heads through 23 July, seven-day purge.
  Previous model retained. Win functions and final estimators are unchanged.
- Frozen development comparison: place Brier .174255392→.173851156; date 95%
  delta [-.000700465,-.000103740]. Target point improves but its interval crosses
  zero. Recent win map worsens and is rejected. All dates remain development;
  no prospective accuracy or market-parity claim. Report and sealed evidence:
  `docs/research/PREDICTIVE_REVIEW_038.md`, `data/audit/predictive_review_038/`.
- Historical qualification **20261029_133847_609414** exactly replays prototype
  place forecasts and original win forecasts. Only **installation_v4** has
  verified current inputs: parent prefix `267f5d1582796e06`, 667,873 eligible
  runners / 71,477 fields. Earlier failed reporting attempts and stopped v3
  mismatched-data attempt remain preserved. Never resume installation_v3.
- Native live check passed 14:23 UTC on 115 complete fields / 1,108 runners and
  Ffos Las's seven-runner 15:00 card; single/batch max error zero. Evidence:
  `data/audit/predictive_review_038/live/report.json`.
- Normal scraping/transactional ingestion preserves reported claims, penalties,
  factual ride descriptions and reasons in append-only context observations.
  Pilot: 72 fields / 625 starters, 182 positive claims, five penalties, 625 dated
  descriptions. Research-only causal context histories pass full/prefix/seed,
  permutation and future-independence checks on 9,043 rows. Coverage/source
  timing is insufficient for predictive adoption; no new tree inputs selected.
- Continuous recorder saves immutable attempts, failures, market/win forecasts,
  refined/legacy place pairs and explicit withdrawal receipts. Assessment keeps
  strict identical-field and issued-field scores separate, preserves original
  probability mass, reports tails and card changes, and scopes result snapshots
  to issued races. Pilot readout remains descriptive, not formal confirmation.
- Full suite: **1,286 passed**. Subsequent overlapping suites: 50 forward/place,
  51 source/database, and 19 recorder/issued/store-refresh tests passed. Two
  earlier missing taxonomy annotations were corrected without feature changes.

Windows task `HorseForecastRecorder_038` was successfully registered with a
current-user logon trigger and daily 09:00 trigger. Its first refresh exposed a
SQLite-versus-CSV fingerprint mismatch and unnecessarily began a staged rebuild.
It was stopped before replacing canonical files. The recorder now loads the
atomic CSV export and permits safe append/no-op only, never automatic rebuild.
Two new regression cases cover that fix. Earlier study/log evidence is preserved.
The task now names the corrected final study:
`20261029_predictive_review_038_final_20261029_141132_815571`.
Protocol SHA256 `5f7760ab96f57c1e7276fc84673e7ab81a2212ddb89c0a3f6c71d81e048dddc9`.
Eligible from 30 October; zero observations so far. Source/model pins verified,
fresh recording heartbeat, no recorder errors. Daily 09:00 and current-user
logon triggers, hidden bounded sessions ending around 03:00. Computer must be
awake and user signed in. The recorder is intentionally left running.

Deep readiness passed 14:45 UTC with exact database/store agreement; operational
readiness and source pins passed 14:51 UTC with healthy recording. Receipts:
`data/audit/predictive_review_038/readiness.json`, `readiness_operational.json`,
`scheduler.json`, `scheduler_running.json`, and authoritative `final_status.json`.
The descriptive pilot requires at least 84 calendar days, 1,000 paired fields
and 60 race dates plus settlement grace; earliest sample close 21 January 2027.
There is no automatic formal confirmation or model-selection decision.

No more fitting or implementation is queued. Let fresh evidence accumulate and
audit coverage before selecting context predictors. Do not rerun completed
research controllers or claim a win improvement, trained context inputs,
prospective place improvement or market parity.

Python requires process-local PYTHONHOME
`C:/Users/Leonp/AppData/Local/Programs/Python/Python314` and PYTHONPATH pointing
to `.venv/Lib/site-packages`; the hidden launcher handles this. Do not change
global Python settings. Protected feature metadata/network need approved
outside-sandbox commands. Preserve all earlier artifacts and worktree changes.

## Earlier completed034–037 handoff (historical; superseded by038 above)

Updated29 October2026,11:06GMT. Latest user: "Please move onto the next best ideas."
No learner, recorder or scheduler is running. No subagents were requested.

## Current installation

Production remains **20261029_085433_011356**,154 inputs,3,992win/2,991place
trees, parameters through27 October2026. Histories extend through28 October:
754,842 runners /80,250 fields. FE **11bf1232d575eea6**; training-prefix data
**267f5d1582796e06**; current-history signature **554fd1046c28a932**.
Model SHA256 **a97c602aceddbc79da06a72d9f861f91a5e41bf6d686def224868d75f627fc15**.

Deep readiness passes **11:04GMT**, including database-content comparison.
Native live check passes **11:01GMT** on115 recent complete fields /1,108
runners and Bath's12-runner12:42 card; single/batch maximum error0.
Evidence: `data/audit/nonfinish_source_037/readiness.json` and
`data/audit/live_upgrade_037/report.json`. An initial parallel status attempt
hit the exclusive store lock; its failed log is retained, then status was
rerun sequentially. There is no persistent mismatch or rebuild requirement.

Current forward study:
**20261029T105914Z_nonfinish_source_037_20261029_085433_011356**,
eligible30 October,0observations/no scheduler. It pins the same model plus
corrected collector source. The previous091505Z033 study is preserved but its
source pins no longer match after037. Do not rewrite old studies or evidence.

## Completed predictive ideas

**034 transferable/recent price history:**5 new inputs,159 total. Audited678085
rows,8tests, exact full/prefix/seed equality. Recovers usable prior prices for
300/377 target runners lacking same-scale support. CheapQ4 point scores pass.
One native win-only check then improves overallQ1 LL1.798224→1.795672 and
Brier .08995795→.08987524, but target Brier worsens .09756486→.09760743.
Overall date-cluster LL interval[-.005913,+.000824] crosses0. **NOT SELECTED.**
No current-data/place/joint fit. Native baseline is saved033 CANDIDATE column
(154inputs), not its old151-input baseline. Three native win components were
needed, completed at10:30UTC. No fitting remains.

**035 market-relative finishing context:**3 inputs,157 total;8tests and exact
prefix/seed equality. One small fit worsens targetLL1.981946→1.982161 and
Brier .09162586→.09165907. **NOT SELECTED**; no native/joint follow-up.

Across034–035 one new corrected154-input400-tree baseline was necessary;
its exact cache c995bd0e70649f003834fbbfd31d169f00ffb9647d9f2cdab5628558df5d4656
is reusable. There was one small candidate per family and one native034
candidate, not a quarter/capacity sweep. Research helpers remain unwired.
Report: `docs/research/MARKET_HISTORY_REVIEW_034_035.md`; final_status.json in both
`data/audit/market_transfer_034/` and `data/audit/market_finish_surprise_035/`.
Preserve sealed plans/manifests/reports; do not rerun completed controllers.

**036 simple market-assisted correction:**0new tree fits. Same frozen033
historical model (trained2025-12-22) across earlier coefficient fit, purged
validation and May1–June10test. October weights never applied retrospectively.
Fresh complete BetfairBASIC T-5 LTP books only;2,080calibration and1,351test
fields. Two tiny bounded coefficient fits. Validation selects exact market
BEFORE test. Test marketLL1.778875, combined1.780645; target487fields market
1.789571, combined1.792246. Combination Brier worsens too. **NO MODEL-ADDED
VALUE SELECTED.** Tiny temperature test gain was not validation-selected.
No correction is wired into normal forecasting.9tests pass; entire small
fit/readout~6.2s. Source capsule and final_status in
`data/audit/market_assisted_036/`; report `docs/research/MARKET_ASSISTED_REFRESH_036.md`.
LTP is not executable; final-starter features are retrospective, the frozen
model ages, and these dates were previously examined development. No parity,
profitability or fresh-confirmation claim.

##037 objective repairs (normal collector plus separate research parser)

Normal `src/data_scraper.py` now recovers casualty.reason PulledUp→PU,
Fell→F, UnseatedRider→UR, Refused→R when finish_position is numeric/string0
or missing. Explicit raw finish labels/positive positions take precedence;
unknown/malformed/unmapped reasons remain unknown. Pending cards clear all
result signals and nonrunners stay excluded. Existing DB label column persists
these values automatically in future normal ingestion. No feature toggle.

Source proof:48cached full results,417entries,391starters,26nonrunners.
37reasons recovered (27PU/5F/4UR/1R); ALL other parsed result columns and ALL
391card rows unchanged. Generic nonfinish counts unchanged.205related tests
pass. Before-parser snapshot, hashes, label diffs and dated factual source
annotations saved in `data/audit/nonfinish_source_037/`.

Pre-April history contains41,866genericDNFs and75unknown zero positions.
Do not invent their reasons or claim a predictive gain. No historical DB
labels were changed, no giant backfill/rebuild/model fit occurred. Installed
features still use generic nonfinish history. The repair preserves information
for future reason-specific reliability/ability tests once coverage is adequate.

Separate unselected `public_history_features.py` now recognizes source enums
PulledUp/UnseatedRider as known external starts. Real51-profile prefix audit
through2026-03-31:1,012results,17previously omitted known starts across13horses.
16related tests pass. Its prototype quality formula stays the same; the family
is not part of default forecasting. Do not revise old frozen trial results.
Report: `docs/research/NONFINISH_SOURCE_037.md`; authoritative final_status.json in037.

## Next useful investigation

Prioritize additional dated information about established horses rather than
another rescaling/transfer variant of the same market histories. Audit coverage,
source timing and interpretation of factual ride descriptions and reported
rider claims before fitting. The48cached result pages already preserve factual
ride_description and structured reasons; do not use profile commentary (which
can describe future races), current ratings, lifetime statistics or headline
form as dated performance evidence. Avoid a mass scrape until a bounded pilot
establishes usable coverage and collection cost. Use normal saved154-input
out-of-sample forecasts for an error review, not an old151-input screen.

User priorities: calibrated win probabilities, established horses, both
independent and market-assisted approaches, source/feature audit before fits,
few fits and no routine quarter sweeps. All objective repairs belong in normal
paths; speculative features earn adoption through a bounded comparison.

##033 adopted recipe evidence

`docs/research/PRIOR_MARKET_HISTORY_033.md` and033/final_status.json retain installation
qualification. Q1development overallLL1.802287→1.798224; target993fields
1.912089→1.905518, Brier improves both; placeBrier .17461220→.17425539.
Matched2,469market fields: model1.784530 versus market1.686613; about4.48% of
previousLLgap closed as a point estimate. Market remains ahead. Current
full-data weights have no additional held-out score. Existing earlier repairs
are unconditional in normal training; no optional repair track remains.


##039–040 research follow-up,29 October2026 at18:29GMT

The user asked to proceed with further ideas while checking another agent's
work.038 source-context/place/recorder additions were inspected and retained.
An independent live-speed installation was rebuilding during this work; its
owner controls that rollout. No canonical source, model, store or recorder
files were changed by039–040.

Two new fixed families were audited on678,085 frozen rows and each received
just one400-tree win fit, reusing the exact154-input cached baseline.039
market-adjusted trainer/jockey form: overall LL1.845981→1.845247, target
1.981946→1.983893 and target Brier worsens.040 own observed racing cadence:
overall1.845981→1.845848, target1.981946→1.982299 and target Brier worsens.
Both overall date intervals cross zero. Neither advances; no full/place/
calibration/current-data or combined follow-up. Both baseline replays have
maximum error0;27 related checks pass. Source-prefix/append equality is strict.
No fitting from this follow-up remains running. Other agent's rebuild is
separate and must not be interrupted or reported as a039/040 job.

Report: docs/research/ESTABLISHED_CONTEXT_039_040.md. Sealed evidence/final_status in
data/audit/connection_form_039 and recovery_pattern_040. The new helpers are
unselected. Frozen learner reuse is scripts/screen_frozen_win_candidate.py;
its small-model caches need its original-source module loader. Do not rerun
completed controllers or tune a favourable subset/window after these results.
