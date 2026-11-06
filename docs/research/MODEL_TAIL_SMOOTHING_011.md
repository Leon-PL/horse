# Narrow tail smoothing011 — 26 October 2026

The narrow correction **does not qualify for adoption**. Earlier validation
selects no adjustment for either model. A separately recorded trade-off
diagnostic shows that raising a near-zero forecast can repair an individual
catastrophic error, but later overall log-loss changes remain mixed. Production
and all earlier experiments remain unchanged. This is not market parity.

## Correction and chronological selection

Reuse 010's verified saved native and pre-isotonic Platt forecasts; keep all
six 009 trees, features and calibration maps fixed. For each runner with
current probability below 1%, raise the unnormalized value only if it is
below `strength * Platt_probability`. Normalize the complete field afterward.
The other runners retain their relative odds. Races with no activation are
exactly unchanged, including floating-point values.

The frozen strength grid is 0, .001, .003, .01, .03, .1, .25. The cap repairs only
forecasts suppressed by at least 75% relative to their Platt stage. All input
fields must already be complete normalized distributions; the method never
fills a missing runner or uses an outcome at inference. A positive correction
gives an affected runner probability at least
`strength * Platt_probability / (1 + strength)`.

Fit on July 1–September 24, 2025 OOS forecasts: lowest winner log loss among
strengths with no increase in runner Brier or binary loss. October1–December22
validates that exact strength, requiring lower winner loss and no worse Brier
or binary loss. Otherwise select identity; do not choose another coefficient
on validation. Freeze selection before later scoring. The 149-input model is
primary; the 176-input pace model is descriptive and cannot reverse 009.

- The 149-input model learns 0.1. Fitting log loss improves 0.00366187, but
  validation worsens 0.00002891. Validation also increases Brier by 4.67e-9
  and binary loss by 3.06e-6. The selected strength is **zero**.
- The 176-input model selects **zero** during fitting. All positive strengths
  lower fitting log loss, but increase Brier by 7.33e-10 to 2.21e-7, exceeding
  this experiment's strict no-increase rule.

Consequently the registered January–March and April–May forecasts reproduce
their original values exactly. Their losses and market gaps are unchanged.
The primary trial fails its improvement rule. The minuscule Brier changes
must not be described as established material harm; the registered rule
is stricter than a practical noninferiority margin.

## Separate exploratory trade-off diagnostic

After 011 finished, a separate specification records a diagnostic that removes
the fitting Brier veto and chooses the lowest **earlier fitting** log loss
from the same seven strengths. Both models choose 0.25. No validation or test
score selects these coefficients. This is a post-result diagnostic, explicitly
ineligible for promotion, not a replacement protocol or a reversal of 011.
All periods and both models are reported. No tolerance is changed in 011.

| Model and period | Original race log loss | Diagnostic log loss | Delta | Runner Brier delta |
|---|---:|---:|---:|---:|
|149,Jan–Mar,2,591races |1.81799380|1.81794245|−0.00005136|+6.89e-7|
|149,Apr–May,2,439races |1.90437727|1.90440405|+0.00002678|+2.34e-7|
|176,Jan–Mar,2,591races |1.81950698|1.81963575|+0.00012877|+6.45e-7|
|176,Apr–May,2,439races |1.90489967|1.90477636|−0.00012331|+3.60e-7|

The 149-input primary January–March log-loss delta has paired 95% interval
[-0.00044380,+0.00015391]; date-cluster sensitivity also crosses zero.
Its April–May interval is[-0.00019350,+0.00014440]. There is no reliable
overall improvement. The176-input January–March deterioration has a positive
paired interval[+0.00011210,+0.00014432]; its transfer improvement is uncertain.
Both transfer comparisons use the same December22 trees and are model-aging
stress tests, not deployment-cadence estimates.

The correction adds probability to low-chance runners and reduces other
probabilities to keep the race total at 100%. In January–March, the 149 model
improves the winner probability in one race and slightly worsens 379 others;
the remaining races are unchanged. Green Hope rises 0.00085568→0.00140462.
The 176 model has no winning runner directly lifted in that period, and
376 races incur small normalization costs. A large benefit on a rare event
can pay those costs, but it did not occur in this later sample.

At strength 0.25, the 149 model affects 380 January–March fields/680 runners.
Mean redistributed mass is 0.00014134 per race, or 0.0141 percentage points;
the largest redistribution is 0.00452949. This is a narrow correction, but
it still has a measurable cost. These diagnostics retain every valid race;
none is discarded for a poor forecast or an extreme winner.

## The known near-zero error

In the previously inspected December 15 Naas validation race, It’S Bobsled
Time's pace-model probability rises from 1.0068e-9 to 0.000971833 under the
diagnostic floor: approximately **0.0972%**. Its original pre-isotonic Platt
probability was 0.00389187. This demonstrates the mechanism on the actual
complete 14-runner field; it does not validate future performance. The main
model rises 0.00068210→0.00109707 in the same diagnostic.

The registered selected strengths are zero, so neither change is applied
to the registered later forecasts or to production. Neither later period
contains a winning forecast below 1e-7 in the original models. We therefore
cannot estimate the future frequency of catastrophic winners from these
later periods or claim prospective rare-tail protection.

## Evidence and next modeling priority

Method: `src/tail_smoothing.py`. Runner:
`scripts/benchmark_tail_smoothing.py`. The original manifest, source snapshot,
frozen selection, model-linked overlays, paired probabilities, market context
and report are under `data/audit/model_improvement/011_tail_smoothing`:
11 evidence files plus completion receipt. Manifest SHA256:
`4b7bc8c9384a9ea43877c6131e1fbd805dd1f6b6bf7484a70b4ad495b9c5e804`.
The primary calculation takes 8.88 seconds using cached probabilities; no tree
training or feature-store rebuild is required.

The post-result diagnostic has its own specification, source archive,
predictions, report and completion receipt in
`data/audit/model_improvement/011_tail_tradeoff_diagnostic`, run by
`scripts/diagnose_tail_tradeoff.py`. It cannot make a promotion declaration.

All 32 focused smoothing/calibration/scoring tests pass, including 14 new checks
for positivity, normalization, exact identity, field boundaries, row order,
serialization, earlier selection and rejection. Read-only `verify_011.py`
validates 11 primary and 4 diagnostic evidence files, source archive CRC/contents,
all source/input hashes, unchanged production/study/model artifacts and exact
saved-probability/metric replay (maximum difference 0). Logs:
`data/audit/20261026/model-011-*.log`.

The strict market overlap remains inherited from 010: 2,469 Q1 fields and
1,188 May fields, with April unavailable, complete-field exclusions retained
and no starting-price substitution. With identity selected, the market gap
is unchanged. All dates are inspected retrospective development; the method
was proposed after010's results, so even the later chronological diagnostic
is exploratory. No fresh confirmation or betting claim.

Close this particular floor and broad-blend line here. Further threshold or
strength searches on the same quarters would be weak evidence. The next
productive independent-model work is the original plan's remaining history
depth/missing-history/estimate-uncertainty family, after checking earlier
artifacts to avoid repeating an existing ablation. The goal is better
discrimination between runners; calibration alone has not closed the gap.
