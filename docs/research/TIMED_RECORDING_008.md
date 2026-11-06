# Manual recording inside the eligible window — package008

**Timing contract with historical deployment examples.** The named v3 study and
its registered pilot describe the 26 October installation. Use the [README](../../README.md)
for the current study; a new study does not automatically inherit the old pilot plan.

The Forecast record page now offers **Record in eligible window**. This one
manually started action calculates the independent forecast ahead of time,
waits if needed, then fetches the racecard and market prices again before
publication. It addresses the observed mismatch between approximately two
minutes of native calculation and a 30-second publication window.

The saved model, probability calculation, reference and eligibility thresholds
are unchanged. This is an operational improvement, not a demonstrated gain in
predictive performance or a market-parity result.

## One attempt, bounded waiting

Discover the race before its window opens, select it and start the action
within ten minutes of that window. The existing window remains **5 minutes
30 seconds to 5 minutes before the first observed scheduled start**. The action
targets the final card recheck two seconds after the window opens, leaving
time for network requests and publication.

1. Verify the pinned study and its first observed schedule. Reject a changed
   schedule, late discovery, a missed cutoff or a start more than ten minutes
   before the wait target.
2. Fetch the initial card and calculate the probabilities through the native
   feature/model path. Preserve the exact inputs.
3. Wait only within this attempt, in steps of at most one second. Compare
   elapsed UTC and monotonic time. Significant clock changes, a machine delay
   past the cutoff and a wait beyond its bound fail closed.
4. Fetch the card again. All columns, dtypes and values must match exactly,
   including any odds supplied in the card. Changed inputs invalidate the
   computed independent probability vector. Check that history did not change.
5. Refresh event state and WIN market prices, validate the entire paired field
   and publish. The existing final publication-time check still applies.

A slow calculation or network request can still miss the deadline. A card
change or missing price can still prevent a complete pair. These attempts
retain their available evidence and exclusion reasons; waiting does not turn
them into eligible forecasts. Discovery prices and old cards are not used as
current final observations.

Timed attempts add immutable `timing_intent.json` and `timing_receipt.json`
files. Eligible timed pairs require both, alongside the existing full input
and publication evidence. The receipt records an actual wait or its failure.
Interrupting an action before completion cannot select a partial forecast.

The action ends after one attempt. It does not create a background collector,
recurring automation, order or wakeup. Keep the manually started action open
until it finishes. **Record fresh forecast** retains immediate capture for
operational checks.

## New study, same pilot conditions

The implementation change requires a new study because the recorder freezes
its source code. **`data/forecasts/forward/20261026_baseline_v3`** pins the same
production artifact **20261026_140950_205296** and retains the original market,
field, quote-age, history and publication rules. The implementation adds the
optional bounded timing mode; earlier protocols are not rewritten.

The v3 companion pilot is registered before its first eligible day,
**October 27, 2026**, with the same minimum 84 calendar days, 1,000 eligible published
pairs, 60 race dates and seven-day results grace period. Formal confirmation is
still unregistered. v2 and its pilot plan remain preserved, with their
capture controls disabled under the new code. Saved attempts remain readable.
No previous forecast or eligible sample is transferred or relabelled.

Use the app, or run one manual timed attempt:

```powershell
.venv\Scripts\python.exe scripts/record_forecasts.py `
  --study data/forecasts/forward/20261026_baseline_v3 --list

.venv\Scripts\python.exe scripts/record_forecasts.py `
  --study data/forecasts/forward/20261026_baseline_v3 `
  --event EXACT_EVENT_ID --at-window
```

`--at-window` requires `--event`. It cannot create a study or discovery job.
Inspect `selected_for_benchmark` and the recorded reasons; an operational pair
before the registered start date remains ineligible.

## Evidence

Qualification receipts and live timing evidence are stored under
`data/audit/forward_forecasts/008_timing/` and
`data/audit/20261026/recording-008-*.log`. Final test counts, source/runtime
snapshots, original-package integrity checks and operational results are
recorded in the completion certificate. No forecasting-lift claim is made.

The live operational check used the 20:30 Southwell race on October 26. Native
calculation finished at 20:22:13 UTC, then the action waited 138.54 seconds. It
resumed at 20:24:32.000342 and published at **20:24:33.448130 UTC**, inside the
registered 20:24:30–20:25:00 window. The six-runner market vector replays exactly
from the preserved prices.

The complete pair was **correctly rejected**: racecard odds changed for
Benvoy (2.38→2.50), Solar Pass (5.50→5.00) and Vaguely Royal (4.50→4.33). The other
card fields matched. The model vector was not issued against those changed
inputs. This attempt also precedes the October 27 eligible start, so it does
not contribute to pilot counts. Timing success does not establish paired
forecast coverage or model performance.

This exposes the next coverage question: how quickly the native live path
can rebuild current inputs when a card changes, and whether the changed odds
affect the selected model inputs. That needs a bounded dependency/performance
check. The exact-card policy is retained; this record is not repaired or
relabeled as an eligible pair.

Full regression: **833 tests passed**, 108 warnings, 490.45 seconds. This
includes the 27 planning/readout tests introduced in 007 and 13 timing/recorder
checks added in 008, alongside the previous 793-test baseline. The timing checks
cover bounded waits, schedule/deadline checks, clock jumps, fresh reads after
waiting, changed cards, history updates and interruption. App checks confirm
the new action, old-study readability and the live market-only display with
zero eligible pairs. Earlier 006/007 sealed evidence and original study files
verify; the model and forecast/pilot policy fields match the previous study.

The separate native diagnostic recomputed all 149 selected inputs from the
refreshed six-runner card against the same unchanged history. No selected
feature changed and the reconstructed win probabilities differed by **zero**.
This is one captured field, not a general proof that every possible odds
change is irrelevant. It is diagnostic evidence only: the original recorded
independent probability column remains unavailable and the rejection stands.

The next bounded check should establish current-odds independence across the
native input and prediction path, then specify any input-policy change in a
new study. Until then, unchanged-card eligibility can exclude price-moving
races. Any paired scores describe that conditional sample and cannot be
assumed to represent all available races. No rule is loosened here.
