# Installed044: RTV repairs, feature pipeline and current refit

Completed30October2026 at17:28UTC. Authoritative receipt:
`data/audit/rtv_repair_features_044/installation/final_status.json`.
See `docs/RTV_FEATURES_REPAIRS_044.md`. Full database-content readiness passes
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
See `docs/RTV_PIPELINE_043.md` and
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
`docs/AMENDED_RESULTS_042.md`. Publication, both initial serving QA receipts and
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
See `docs/AUXILIARY_PLACE_WIN_041.md`, `data/audit/auxiliary_win_041/final_status.json`.

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
See `docs/LIVE_FEATURE_SPEED_20261029.md` and the speed audit's installation
`final_status.json`. The previous 038 operational identifiers below are historical.

## Preserved 038 handoff

> Research039–040 completed29 October2026: both new win-feature proposals stopped after their small checks; two new fits total, exact cached baseline reused. No canonical/recorder changes. See [results](docs/ESTABLISHED_CONTEXT_039_040.md) and the039–040 section below. The separate live-speed installation was still rebuilding; its owner controls that rollout.

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
  `docs/PREDICTIVE_REVIEW_038.md`, `data/audit/predictive_review_038/`.
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
Report: `docs/MARKET_HISTORY_REVIEW_034_035.md`; final_status.json in both
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
`data/audit/market_assisted_036/`; report `docs/MARKET_ASSISTED_REFRESH_036.md`.
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
Report: `docs/NONFINISH_SOURCE_037.md`; authoritative final_status.json in037.

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

`docs/PRIOR_MARKET_HISTORY_033.md` and033/final_status.json retain installation
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

Report: docs/ESTABLISHED_CONTEXT_039_040.md. Sealed evidence/final_status in
data/audit/connection_form_039 and recovery_pattern_040. The new helpers are
unselected. Frozen learner reuse is scripts/screen_frozen_win_candidate.py;
its small-model caches need its original-source module loader. Do not rerun
completed controllers or tune a favourable subset/window after these results.
