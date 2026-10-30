# Experiment 004 — monthly versus quarterly refitting

**Status: complete; primary benefit remains unresolved.** Monthly refitting
improves the point estimates in every quarter, including runner Brier, but
does not pass the registered race-log-loss confidence gate. Keep the current
149-feature binary recipe and retain monthly refitting as unpromoted research.
No architecture expansion or production change.

The accepted plan's deployment package separates feature-history freshness
from parameter refitting. A newer model is not automatically a better model.
This fixed replay tests whether monthly parameter updates improve the same
forecasts compared with holding parameters fixed for a quarter.

## Results and decision

All six new fits finished before any candidate performance was inspected.
The primary population contains **9,356 races / 85,863 runners** across all
nine registered months. The three shared first months contain 3,337 races and
have exactly identical probabilities. Elapsed fitting/primary-scoring time
was 3,362.51 seconds.

| Lower is better | Quarterly baseline | Monthly refitting |
|---|---:|---:|
| Runner Brier | .08754346 | .08746770 |
| Race-mean Brier | .09874967 | .09866833 |
| Binary log loss | .30088334 | .30071558 |
| Mean race log loss, primary | 1.87204489 | 1.87097477 |
| Ten-bin ECE, diagnostic | .00146349 | .00197548 |
| Winner accuracy, higher is better | 29.0348% | 29.2727% |

Race-log-loss difference is **-.00107012**, with central-95% race interval
**[-.00436065,+.00347671]** and date-cluster interval
**[-.00426195,+.00305721]**. Both include no improvement. Therefore
**`candidate_advances=false`** despite all three quarter point estimates
improving and both runner-score guardrails passing.

| Quarter | Races | Quarterly race LL | Monthly race LL | Difference |
|---|---:|---:|---:|---:|
| 2025Q3 | 3,532 | 1.92063259 | 1.91995433 | -.00067825 |
| 2025Q4 | 3,233 | 1.86228139 | 1.86068414 | -.00159725 |
| 2026Q1 | 2,591 | 1.81799380 | 1.81704725 | -.00094655 |

Preserve the secondary Brier evidence: runner Brier improves in every quarter,
with pooled difference -.00007576 and one-sided race-cluster upper95
-.00001120. This does not replace the primary confidence criterion. ECE worsens
overall, so there is no blanket calibration-improvement claim.

In the six changed-model months alone (6,019 races / 55,137 runners), race LL
changes 1.87393949 -> 1.87227609. Its difference -.00166340 still has an interval
crossing zero: [-.00639905,+.00483053]; date sensitivity agrees. This diagnostic
does not change the decision. Runner-weighted mean parameter age falls from
52.46 to 22.21 days, median 52 to 23 days; maximum 99 to 40 days. Parameter
freshness improved substantially, while the primary forecasting benefit is
not established.

## Market context

Strict identical-field matching retains **8,942 races / 81,696 runners**.
Whole-field exclusions remain 394 missing/changed fields and 20 winner-label
disagreements; canonical outcomes are unchanged.

| Matched population | Quarterly | Monthly | T-5 market |
|---|---:|---:|---:|
| Runner Brier | .08787478 | .08778709 | .08414852 |
| Mean race LL | 1.86879634 | 1.86755050 | 1.74928256 |

The monthly point estimate closes **1.04% of the quarterly model's race-LL
gap**. Its remaining LL difference versus market is +.11826794, 95% race
interval [+.10710970,+.12962438]. Both independent methods fail the existing
practical-parity criteria.

The fixed freshness population has **7,719 races / 67,710 runners**. Quarterly,
monthly and market race LL are 1.86806567, 1.86765087 and 1.75239937. Monthly
runner Brier is .09175043 versus market .08802249. The primary uncertainty and
large market gap remain; no correction was fitted or cohort selected after
inspection.

Close this cadence comparison without a weekly/daily/window/calibration sweep.
Retain the fixed monthly recipe and its Brier finding as research evidence.
Any claim of primary improvement needs registered confirmation on other data.
Operational all-settled fitting, refreshed data and further information-family
work remain separate items in the accepted plan. No automatic promotion.

## Fixed comparison

- Same 2025Q3, 2025Q4 and 2026Q1 dates, complete fields and frozen daily feature
  values as experiments 002/003. Both methods therefore receive identical
  history features at each forecast date.
- Quarterly baseline: exact cached 002 forecasts and model artifacts, already
  replayed against the current unchanged serving implementation.
- Monthly method: first month of each quarter reuses the identical quarterly
  model. Six subsequent months receive new fits. No redundant first-month fits.
- At each calendar month start, fit all eligible settled labels through that
  calendar start minus seven days, using native early stopping/full refit and
  training-only calibration. Missing race days cannot move the cutoff forward.
- All 149 inputs, hyperparameters, recency half-life of 180 days, burn-in,
  chronological calibration and outcome treatment remain fixed. Current-race
  odds are not model inputs. Actual non-finishers remain losing runners.

| Forecast month | Parameter source | Declared label cutoff |
|---|---|---|
| July 2025 | Cached Q3 model | 24 June 2025 |
| August 2025 | New monthly fit | 25 July 2025 |
| September 2025 | New monthly fit | 25 August 2025 |
| October 2025 | Cached Q4 model | 24 September 2025 |
| November 2025 | New monthly fit | 25 October 2025 |
| December 2025 | New monthly fit | 24 November 2025 |
| January 2026 | Cached Q1 model | 25 December 2025; last available labels 22 December |
| February 2026 | New monthly fit | 25 January 2026 |
| March 2026 | New monthly fit | 22 February 2026 |

The reused January model has an actual fitting cutoff of 22 December because
there are no eligible races through the calendar cutoff of 25 December.
The replay verifies that the available history is identical. Parameter-age
diagnostics use each model's actual last training date, not a fabricated date.

## Registration and assessment

Registration: `data/audit/market_performance/004_registration.json`, SHA-256
`9f491c747c7bc5b23d5f27ac2ef97a6187d6a34c22da8af7e4c2b6861c8a76f8`.
Native manifest raw SHA-256:
`a3b3dcd4f5e9fe4414326e09524f90f2c6d7453f82c68f3803b56e599df1d086`.
Source, registration, data, binary baseline artifacts and market inputs are
fingerprinted before fitting. No partial performance inspection or favourable
early stopping of the experiment.

Primary assessment covers all nine months, including the three shared-model
months. Require lower mean race log loss with race and date-cluster central-95%
upper paired bounds below zero, no worse runner Brier or binary log loss, and
race-log-loss improvement in at least two of three quarters. All six new fits
must finish. A registered diagnostic also reports the six changed-model months
and parameter ages; it does not replace the primary population or gate.

No weekly/daily cadence, rolling-window, feature, calibration, architecture or
hyperparameter sweep is selected from these results. Passing advances a refit
recipe for qualification; there is no automatic model promotion.

The secondary comparison reuses the strict T-5 BASIC LTP matching, proper
scores/parity assessment and fixed 300s runner / 90s market freshness criteria
without changes to those helpers. It fits no market correction. Whole-field
exclusions and paired forecasts are retained.

These are previously inspected development dates. The replay is retrospective,
not fresh confirmation, an as-issued racecard test or executable-price returns.
Canonical data/features are unchanged and no new collection is performed.

## Verification and evidence

Before fitting, **31 focused checks passed**, including native-framework
cutoff/full-refit checks, calendar scheduling and complete-field alignment.
The whole regression suite then passed **745 tests**, 24 warnings, in
418.62 seconds. A reporting-only follow-up passed **32 status checks** in
4.93 seconds: project status now shows the promoted parameters' saved fitting
cutoff separately from feature-history freshness. Missing fitting metadata is
reported as unknown; no parameter-age readiness threshold was introduced.
This change is outside the frozen fitting sources and does not affect forecasts.
All **nine actual model artifacts** reproduce saved forecasts on 64 shuffled
complete fields each: **576 fields / 5,307 runners**, maximum absolute error
1.43e-14 (tolerance 1e-12). Their feature lists, calibration methods, recorded
calibration purge/cutoffs, absence of inherited holdout metrics, actual fitting
cutoffs and parameter ages pass. Shared first-month probabilities are exactly
identical. Deep status passes database/store/code/production consistency, with
the existing 14-day data-age warning; it explicitly reports production's July 8
fitting cutoff (110 days old).

Models, per-month fitting metadata,
forecasts, source archive and runtime versions are retained under
`data/audit/market_performance/004_refit_cadence/`; logs are under
`data/audit/20261026/experiment-004-*.log`. Production remains
`20261025_193016_331639`.
