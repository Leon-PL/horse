# Fixed-tree calibration shrinkage010 — 26 October 2026

The tested adjustment is **not promoted**. It improves some severe
underestimates and slightly lowers Brier score, but does not improve the
primary January–March winner log loss. April–May improves in point estimate
with uncertainty crossing zero. The independent forecast remains materially
behind the same-time market.

## What was tested

All six trees, feature sets and fitted Platt/isotonic maps from009 remain
fixed. The research adjustment mixes the existing race probabilities with
their normalized, pre-isotonic Platt probabilities:

`adjusted = (1 - w) * existing + w * Platt`

One coefficient is fitted on July1–September24,2025 out-of-sample forecasts,
minimizing winner log loss subject to no increase in runner Brier score.
October1–December22 validates that exact coefficient: lower winner loss and
no worse runner Brier or binary loss are required. Otherwise the adjustment
falls back to zero. Selection is frozen before generating the later test
alternatives; no coefficient is refitted on validation or test labels.

The primary149-input model learns **w=0.8286734247242075** and passes
validation. This is a large movement toward Platt, despite the initial
expectation that the Brier constraint might favour a small correction. The
pace176 diagnostic learns **w=0**: its earlier Brier constraint allows no
positive blend. It remains unchanged and cannot reverse the009 rejection.

Native forecasts are reconstructed exactly, with zero maximum difference
from public inference and retained009 probabilities. This isolates the
calibration adjustment from feature or tree changes. No native serving code,
feature generation or production model was modified.

## Later probabilities

All valid complete fields, including DNFs and extreme winners, are retained.
Lower scores are better. Main comparisons use the149-input model.

| Period | Races | Original race log loss | Adjusted race log loss | Original runner Brier | Adjusted runner Brier |
|---|---:|---:|---:|---:|---:|
| Jan–Mar2026, primary | 2,591 | 1.81799380 | 1.81818510 | 0.09050170 | 0.09046597 |
| Apr–May2026, transfer | 2,439 | 1.90437727 | 1.90275709 | 0.08662470 | 0.08660078 |

Primary log-loss delta is **+0.00019130**, paired95% interval
**[-0.00187169,+0.00209070]**; date-cluster interval
[-0.00215779,+0.00208051]. Binary loss improves slightly
0.30701468→0.30699826, but that does not satisfy the primary rule. Runner
Brier's one-sided upper95% delta is positive, so its small point improvement
is not strong evidence either. Race-mean Brier has different race weighting
and a negative interval; it must not replace the registered runner metric.

Transfer log-loss delta is **-0.00162019**, paired95% interval
**[-0.00416095,+0.00082580]**; date-cluster interval
[-0.00412212,+0.00089557]. Both periods use the same trees fitted through
December22. Transfer is therefore an artifact-aging stress test, not an
estimate of performance under monthly or daily retraining. Eight invalid
transfer fields/76runners are rejected as whole fields under the existing
label/identity rules, independently of their predictions.

## What this says about extreme errors

The original009 example remains intact: It’S Bobsled Time at Naas on
December15,2025 received approximately1e-9 from the pace model after the
isotonic map returned zero, despite a pre-isotonic probability near0.00389.
The trace is preserved separately in
`data/audit/model_improvement/009_followup_calibration/zero_probability_trace.json`.
That diagnosed failure motivated this experiment; it is not new confirmation.

Neither the January–March nor April–May sample contains a winner with
legacy probability below1e-7 for either model. Consequently this comparison
cannot directly validate protection against another near-zero winning
forecast. The pre-isotonic branch is positive and a positive mixture would
mathematically lift a zero-map output, but the fitted pace adjustment was
rejected before test. Its extreme-tail defect is **not fixed in production**.

There are less extreme examples: Green Hope on January20 rises from
0.00085568 to0.00482232, reducing its log-loss penalty by1.72911. Across all
2,591races, however, the adjustment adds0.49565 total loss: gains on selected
outliers do not establish a better overall forecast. The original maximum
winner penalty falls7.06361→5.33450. In transfer, Crokes Cross rises
0.00085020→0.00377381; maximum penalty falls7.07004→5.57967, with a total
loss reduction3.95164 across2,439races. No race is removed for these figures.

## Market comparison

Identical complete fields are paired with the preserved T-5 exchange
snapshots. Late withdrawals, changed/incomplete fields and winner-label
disagreements are excluded as whole fields, with counts saved. These are
conditional final-starter comparisons, not live publication or betting tests.

| Period and overlap | Original log loss | Adjusted log loss | Market log loss |
|---|---:|---:|---:|
| Jan–Mar,2,469races | 1.80427736 | 1.80439342 | 1.68661347 |
| May only,1,188races | 1.89176776 | 1.88956641 | 1.76215524 |

January–March excludes119 unmatched/changed fields and3 outcome
disagreements. April market snapshots are unavailable: the transfer market
comparison covers May only, not the full April–May score above. Its1,251
unmatched fields include the unavailable April coverage. No starting-price
substitute is used. All snapshot files and their300-second replay manifests
verify against preserved fingerprints.

## Evidence and interpretation

The manifest, source archive, frozen selection, model-linked overlayJSON,
eight probability-stage extracts, four market comparisons, predictions,
report and completion receipt are in
`data/audit/model_improvement/010_tail_calibration` (19sealed evidence files
plus their completion receipt).
Manifest SHA256:
`7bed2ba70130fc41c98b42a88b68bc109d0525d3002f66bd0b0e3628f4b60003`.
Fitting/inference/scoring completed in225.33seconds, without training trees.

All18 focused calibration/scoring tests pass. Saved overlay replay and all
reported probability metrics agree within1e-12 (actual native/overlay
probability differences0). Independent read-only verification checks all
19evidence files, source archive contents/CRC, source/data/model fingerprints,
market manifests, frozen-selection ordering and unchanged production/study
artifacts. Logs are `data/audit/20261026/model-010-*.log`.

All periods are previously inspected retrospective development data. No
untouched or prospective confirmation is claimed. The original009 result
and production20261026_140950_205296 remain unchanged. The useful next
calibration hypothesis is a narrowly targeted positive tail smoother learned
on earlier forecasts; repeating this broad blend or selecting its better
period would not provide new evidence. Calibration alone has not closed the
much larger market gap.
