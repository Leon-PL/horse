# History-context ensemble013 — 27 October 2026

The blend gives small point improvements over the retained model in both
later windows, but **does not establish a reliable gain or qualify for a
live switch**. The market remains substantially ahead. All earlier results
and the production model remain unchanged.

This follows the [history-family comparison](MODEL_HISTORY_012.md) and its
inspected external-form diagnostic. It is explicitly exploratory retrospective
research: chronological weight fitting does not turn an already inspected
context hypothesis into fresh confirmation.

## Method and earlier selection

Blend the 149-input baseline and 165-input history model, with one weight for
every runner in a race. Context is known before the race: whether any horse
has external prior form absent from this database (`has_hidden_form=1`).
This does not filter out runners or select races by their eventual winner.

Fit one coefficient per context on July 1–September 24, 2025 OOS predictions,
using the existing convex winner-loss optimizer with a non-increasing runner
Brier constraint. Validate that exact coefficient on October 1–December 22:
lower winner loss and no worse Brier/binary loss are required, otherwise
use the baseline. Each context needs at least 50 fit and 50 validation races.
There is no validation refit or test-selected fallback.

| Context | Earlier fit races | Validation races | Selected history-model weight |
|---|---:|---:|---:|
| No external form in field |3,157|2,659|0.4943050313|
| Contains external prior form |131|324|1.0|

Both coefficients pass earlier validation and are frozen before later
predictions are scored. All models/trees/calibration maps are fixed. No
new trees are fitted; only these two mixture coefficients are learned.
The baseline coefficient is one minus the table's weight.

## Later probability scores

January–March is the primary chronological comparison. April–May is a
transfer stress test using the same December 22 trees and coefficients;
it does not simulate a daily or monthly retraining cadence. Fitting and
validation periods are not pooled into the claimed test evidence.

| Period | Races | Retained149 log loss | Global165 log loss | Context blend log loss |
|---|---:|---:|---:|---:|
| Jan–Mar2026 |2,591|1.81799380|1.81928807|1.81767994|
| Apr–May2026 |2,439|1.90437727|1.90368543|1.90248792|

Primary blend-minus-baseline delta is **-0.00031387**, paired 95% interval
**[-0.00247654,+0.00203919]**, date sensitivity
[-0.00256112,+0.00199095]. Runner Brier improves 0.09050170→0.09046092 and
binary loss 0.30701468→0.30694807, but the Brier upper bound also crosses zero.
Winner accuracy rises 29.4867%→29.5767%; ten-bin ECE worsens. The uncertainty
gates fail. The blend's point improvement over global165 is also uncertain.

Transfer delta is **-0.00188935**, paired 95% interval
**[-0.00433240,+0.00052741]**, date sensitivity
[-0.00425866,+0.00068829]. Brier improves 0.08662470→0.08657558 and binary
loss 0.29986421→0.29962894, with uncertain Brier improvement. Winner accuracy
slips 28.8130%→28.7584%. Again, do not substitute a secondary metric or this
better point estimate for a successful primary result.

Every valid complete field is retained, including DNFs and extreme winners.
Eight invalid transfer fields/76 runners fail the existing whole-field rules,
matching the earlier transfer sample. The primary test has 258 external-form
fields and 2,333 other fields; transfer has 190 and 2,249 respectively.

## Matched market context

The same frozen complete-field overlap and actual T-5 market probabilities
from 010 are retained; no starting-price substitution. April lacks market
snapshots, so the transfer market comparison covers May only.

| Population | Races | Retained model | Context blend | Market log loss |
|---|---:|---:|---:|---:|
| Jan–Mar matched fields |2,469|1.80427736|1.80391693|1.68661347|
| May matched fields |1,188|1.89176776|1.88810511|1.76215524|

The point reduction closes about 0.31% of the primary market gap and 2.83% of
the May gap. Neither is market parity. These are conditional final-starter
comparisons, not prospective as-issued racecards or executable betting tests.

## Verification and interpretation

All 66 focused history/blend/calibration/native-scoring checks pass, including
8 new whole-field/context/identity/fallback/serialization checks. Source,
data, model and production-study fingerprints remain intact. The 9 sealed
evidence files plus completion receipt verify; source archives pass CRC and
content-hash checks. Old 010 probability/market cache fingerprints also verify.

Reloaded overlay predictions reproduce both periods exactly. A fresh native
replay on 64 shuffled complete transfer fields differs by at most 1.11e-16.
All baseline transfer probabilities reproduce 010 exactly. The earlier 012
qualification checks 17 sealed files and 64 shuffled fields per new model.
No fresh full-suite run is claimed for these isolated research additions.

Artifacts: `data/audit/model_improvement/013_history_context_blend`.
Manifest SHA256:
`aa35f235655dede4c1656ccbad4e9c9b7537c1dc270ac9e00466f0653983e106`.
Source: `src/history_blend.py` and `scripts/benchmark_history_blend.py`.
Read-only qualification: `data/audit/model_improvement/verify_013.py`.
Logs: `data/audit/20261026/model-013-*.log`. Computation takes 39.50 seconds,
using the already completed models to generate the transfer forecasts.

The context lead was identified from 012 after its later outcomes were
inspected. These results are therefore exploratory even though coefficients
use earlier data only; they do not qualify as independent confirmation.
The experiment evaluates the complete blend and does not isolate the causal
contribution of context selection from averaging two different forecasts.
It cannot reverse 012's failed global-family advancement result.

The useful finding is a localized history-coverage lead and a low-cost way
to test it without retraining. The observed overall gains remain tiny
relative to the market gap. Do not search further gates or weights on these
same quarters to manufacture a qualifying result. Production
20261026_140950_205296 and the active baseline_v3 study stay unchanged.
