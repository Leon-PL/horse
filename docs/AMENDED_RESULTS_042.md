# Official result amendments042

Installed30October2026 at07:17UTC: production `20261030_015917_823047`
uses the corrected canonical source/history and freshly fitted native win/place
heads through28October. Installed serving checks pass, and the recorder has
appended29October's387 runners without another fit. Its73 Irish purse inputs
also use the restored daily ECB reference. Final deep readiness passed at
08:57:55 UTC, and the verified installation closed at08:59UTC. The recorder is
healthy, with its first eligible comparison date31October. No predictive
improvement has yet been measured from this repair.

The audit found that Sporting Life publishes an original `finish_position`
alongside an `amended_position`. The project ignored the amendment. For
example, [The Gatekeeper was promoted above Thunder Ball at Goodwood](https://www.sportinglife.com/racing/results/2023-08-27/goodwood/754950/city-of-chichester-handicap).
Another [Doncaster result promotes Ecco and places Eaton Hill last](https://www.sportinglife.com/racing/results/2020-12-11/doncaster/601907/sky-sports-racing-sky-415-handicap-hurdle).
The stored winners in these examples were wrong. This affects supervision and
the subsequent horse, connection and rating histories.
The earlier label audit in `NEGATIVE_RESULTS.md` had already identified the
first-past-post versus official-result discrepancy. This work implements and
verifies the repair, including later form/rating histories; it does not claim
that the existence of amended results is a new discovery.

## Normal source fixes

The scraper now uses valid final amended placings and winner labels. It refuses
malformed nonempty amendments and explicitly provisional race stages. Legacy
payloads without a stage retain their established contract. Cards continue
clearing all result evidence. A provisional first-past-post result cannot close
the ordinary sync job before weighing in.

Physical arrival gaps cannot be accumulated in the revised official order.
Reordered fields therefore have unknown margins except the official winner's
zero origin; their winning-time measurement is withheld. Original placings,
gaps and time remain in separate factual amendment metadata. The ordinary
transactional context table preserves that evidence without adding empty or
post-result columns to canonical model inputs. Older context tables migrate
without duplicating unchanged observations.

Seventy-two cached ordinary source pages /625 starters replay all pre-existing
result and card values exactly. Two small factual source fixtures cover a Flat
winner swap and a hurdle disqualification with genuine non-finishers. The
wider parser/database suite passes200 cases; later focused source/repair checks
pass51 cases, overlapping the earlier source cases.

## Historical evidence and one consolidated repair

The original processed history contains754,842 runners /80,250 fields. The
existing ATR cache exposes547 position discrepancies across164 fields. Its
first conservative qualification finds70 complete unambiguous official orders,
including37 changed winners, corroborated by exchange records. Partial or
nonunique cached orders are not automatically trusted.

A wider independent-source census compares77,250 complete fields /725,901
runners with the exchange. It finds134 winner disagreements. After the initial
37 corrections,97 remain. A fixed official-source review of those97 fields
uses the existing results provider and normal request limits, retaining only
the factual result data needed for verification. It qualifies91 more full
amendments. Three additional source-confirmed cases put the disqualified
original winner last among **all starters**, including non-finishers. Their
explicit final position is N, with a legitimate gap in the finishing order.
These are qualified separately from the cached complete-order rule.

The final consolidated batch covers **164 fields /1,534 runners**, with
**131 changed winners and463 changed finishing positions**. Thirty-three
fields have placing changes without a winner change. All original complete
rows and source receipts are preserved. Corrections use actual source horse
IDs, normalized name/date/course/time agreement, complete starter populations,
settled final results and independent winner corroboration. No official order
is inferred from exchange settlement alone.

Three remaining exchange disputes have no amendment in the official-source
response and stay unresolved. The earlier ATR partial/nonunique order leads
also remain outside the automatic repair. Published final placings are
retrospective; their precise original release times are not reconstructed.

The full staged database and CSV preserve every runner and all unrelated
columns exactly. A transaction aborts the entire batch if reviewed current
rows changed. Reapplication verifies the existing correction and performs no
updates. Original rows and corrected values are recorded transactionally in
an append-only repair ledger. Rollback, partial-field, source-conflict and
float-precision checks pass.

Normal processing of the complete consolidated CSV changes only:

- 463 finishing positions and262 individual winner indicators (131 swaps).
- 1,501 raw arrival-gap values and1,446 processed beaten-margin values.
- 54 nonfinish labels, using reported reasons where available and generic DNF
  when the source does not specify a reason.

All other processed values and all unaffected fields match the preserved
history exactly. The corrected processed signature is `3c06942ccdc4e35b`.

## Rebuild and current-data refit

This is a mandatory source correction. It is not an optional feature family
whose adoption depends on its score against known-wrong labels. The selected
154 inputs and statistical recipe remain unchanged. Elo/Glicko and subsequent
form must be rebuilt because their previous inputs contained wrong winners.
The full native builder completed once on the consolidated corrected history:
754,842 runners through28 October, with Elo/Glicko recomputed. Before the first
corrected event, all154 selected inputs match exactly for3,049 runners.
Its source/data pins prevent silently accepting mid-build edits.

In the latest three historical days,26–28October,98 selected features change
somewhere across115 races /1,108 runners; every runner has at least one changed
input. Shared connection/rating histories propagate the earlier corrections.
This descriptive check uses absolute tolerance1e-10 and relative tolerance1e-12
to exclude negligible rebuild rounding. It is not an accuracy comparison.
See `consolidated/recent_input_changes.json` for the individual inputs.

One ordinary current win/place refit through28 October2026 is complete. It routes
normal processing to the staged source and verifies its fingerprint against
the rebuilt feature store before loading it, leaving production
paths available until coherent installation. There is no quarterly sweep,
new feature search or in-sample calibration shortcut.

The first launcher startup changed private directory config and was rejected
by the feature-generation guard before any tree fitting. Its complete plan,
source and log are preserved under `refit/startup_path_failure/`. The corrected
launcher leaves config unchanged and routes only the ordinary input seams.
The recipe, selected inputs, feature generation and rebuilt history are unchanged.

Publication has six real SQLite/filesystem integration checks. These exercise
successful installation, changed source schema, newly arriving results,
rejected promotion, interruption before commit, and a missing receipt after a
successful commit. Before commit, source, raw CSV, all five store files and the
production pointer roll back together. After commit, the coherent new version
is retained for receipt recovery. All six pass without fitting a model.

Recorder monitoring also had an objective false-health case: a fresh heartbeat
was labelled OK while repeated discovery requests failed. Normal status now
requires successful discovery within20minutes (the15-minute polling interval
plus five minutes' grace), consistent study/model identity, and a live process.
Old errors alone do not prevent recovery once discovery succeeds. All44
status/recorder-health checks pass. This does not change feature definitions,
probability math, or which forecasts are eligible for assessment.

The other chat's completed speed migration, production
`20261029_222758_104473`, has exact stored-feature/state equivalence and unchanged
model bytes. That evidence transfers the previous historical validation
reference to the current generation without fitting it again. The migrated
historical reference `20261029_224707_518692` retains the original model bytes,
old labels and historical scores; **it is not the new production model**.
Only `consolidated/refit/validation_final.json` and `final_plan.json` qualify the
new fit. The initial certificate captured an operational receipt before the
other rollout's final update; it is preserved and superseded by immutable local
copies of the completed evidence.

Older predictive experiments, including the rejected041 auxiliary-place check,
retain their original source-label versions. Do not silently relabel their
saved forecasts or treat their scores as measurements of the new fitted model.
Any future paired comparison requires the same corrected truth in both arms
and appropriate rebuilt histories. Fresh forward evidence must assess the
eventual installed model; market parity has not been established.

## Next modelling lead, not implemented

An outcome-free compute audit reproduces the668,218-runner training population.
With the existing180-day half-life and15% same-month uplift, rows older than
two years constitute64.29% of rows but6.10% of weight. Rows older than three
years constitute46.12% of rows but1.44% of weight. Weight effective sample
size is167,446. See `consolidated/training_weight_mass.json`.
This motivates one small shorter-history comparison after042 installation;
it does not establish that those rows can be removed without changing accuracy.
Keep complete feature histories and the absolute burn-in boundary. Compare on
identical future fields and account for weight normalisation, leaf counts and
binning. Tree-prefix reuse can avoid another baseline fit, provided the exact
native training arrays/targets/weights/order are reconstructed and verified
against the saved component key. No training-window change is included in042.

## Daily currency-reference gap found during final readiness

Initial deep readiness passed on30October, but its log exposed a failed
verified HTTPS update of the ECB reference. The cache stopped27October.
That is valid for every fitted input through28October because conversion
uses a strictly earlier observation. Irish results for29October instead used
the last known27October rate.

A verified request with the installed CA bundle retrieved the official
[ECB history](https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.xml).
Every earlier observation matches exactly; only28/29October are added.
The follow-up reconstructs just the387-runner daily tail from the preserved,
corrected28October state. Normal processing changes only euro prize amounts
and their source dates after the fitted cutoff. The fitted prefix signature
is unchanged, so no further model training is needed. The completed update
changes73 euro prize amounts/source dates, and only `prize_money_change_vs_avg`
among selected inputs for those73 runners. Every other selected input matches.
Project-local verified CA configuration restores ordinary future updates;
default HTTPS passed with certificate and hostname verification enabled.
The corrected reference/history was installed08:43UTC. The model's bytes and
all fitted inputs are unchanged. Live QA passes106 recent complete fields /
1,010 runners and the upcoming six-runner11:35 Newmarket card. The recorder
restarted08:49UTC with the same protocol, and successful discovery resumed08:54UTC.
Final deep readiness passed08:57:55UTC without an ECB-update warning. Evidence
is in `consolidated/fx_refresh/`; `consolidated/installation/final_status.json`
verifies the complete installed state, including all1,534 canonical source rows
and their repair ledger. No further fitting or installation remains.

## Authoritative paths

Use `data/audit/result_amendments_042/consolidated/` for the complete correction,
staged source snapshot, history build and current refit. The earlier70-field
stages are preserved intermediate evidence and are superseded for deployment.
The initial staging attempt hashed SQLite before closing its write-ahead log;
its incorrect artifact receipt is preserved and rejected. The later staged
snapshots close/checkpoint their connections before fingerprinting.

The three source batches are the042 root audit, `official_review/` and
`official_last_places/`. `winner_census/` contains the independent complete-field
comparison. The full source review is finished; do not fetch or fit those
closed steps again. Completed controllers refuse overwriting their evidence.
