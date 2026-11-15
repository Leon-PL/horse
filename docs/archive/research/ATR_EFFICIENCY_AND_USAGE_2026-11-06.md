# ATR efficiency, consistency, fluency and field usage

## What was already considered

The configured recipe is `feature_sets/win_selected_056.txt`: 129 inputs,
including these five direct ATR features:

- Peak stride length relative to the field.
- Best prior finishing-speed percentage relative to the best in the field.
- Best prior closing speed relative to the field.
- Prior **energy-profile efficiency** relative to the field.
- Prior front-running comment rate relative to the field.

So efficiency information was partly represented. The literal efficiency grade
was stored but not converted into a historical predictor; native efficiency-score
history existed in the wider feature set but was constructed incorrectly.
Generic finishing-position consistency also exists elsewhere, while ATR-specific
efficiency and finishing-pattern repeatability were missing.

The complete [49-column usage map](../../data/audit/atr_efficiency_20261106/field_usage.json)
distinguishes ordinary feature consumers, new research consumers, identity,
tracking quality and retained source context. It describes collection/engineering
and the configured input list, not installed store/model compatibility.

## Corrected efficiency handling

ATR's [official guide](https://www.attheraces.com/sectionalsinfo) defines its
letter grades as energy-distribution efficiency. The source numeric score behaves
as an error: lower scores correspond to better grades. The audited fresh sample
showed median scores of 0.455 for A++, 1.73 for A+, 5.31 for A, 11.49 for B,
21.6 for C, 42.235 for D, 72.46 for E and 145.08 for F.

The ordinary feature code incorrectly marked higher scores as better and applied
an upper cutoff of 40. Among **30,562 audited records**, **12,929 score values**
exceeded that cutoff. Their raw records remained intact; only the derived score
values became missing, leaving recent averages biased toward efficient runs.
The issue had been noted in the older scraper audit but remained in current code.

The fix now:

- Uses the **minimum** as the best prior efficiency-error score.
- Preserves all finite nonnegative error scores, including D/E/F observations.
- Rejects negative/infinite errors.
- Treats an unrated zero with no energy/optimum profile as missing. The sample
  contained 15 such placeholders; a genuine graded zero remains valid.
- Leaves raw stored measurements and other feature calculations intact.

The helpers share this interpretation through `efficiency_error_scores` in
`src/atr_features.py`. Normal feature signatures invalidate earlier derived
generations. No feature-store rebuild, model fit or promotion was run here.

## New efficiency and consistency candidates

The separate research pack now adds these five per-run measurements to the
original seven:

| Measurement | Definition |
|---|---|
| Efficiency log error | `log(1 + valid native error)`, retaining the poor-run tail |
| Efficiency grade rank | Ordered A++, A+, A, B, C, D, E, F; unknown grades remain missing |
| Worst departure from optimum | Largest absolute actual-minus-optimum gap at matching chart points |
| Pacing-profile irregularity | Median absolute deviation of those gaps around their median |
| Cadence rhythm irregularity | Residual variation at 3f/2f/1f/Finish after removing a smooth linear trend, divided by mean cadence |

The grade and numeric score mostly encode the same information. Their separate
representations should be compared, not assumed to be independent signals.
Lower variability also does not imply a better horse: it can describe consistently
poor performance. Level, variability and observation count belong together.

Each per-run metric has a prior-three-appearance median. Two additional inputs
measure **between-run repeatability** over the last five eligible appearances:

- Median absolute deviation of efficiency log error.
- Median absolute deviation of finishing strength relative to the race.

Both require at least **three observed runs**; a single observation cannot earn
a false zero-variability score. Appearances without ATR data still occupy window
slots. The pack now has 12 run measurements, 14 prior predictors and two quality
inputs. All prediction inputs retain canonical IDs and the conservative 48-hour
publication assumption described in the [first report](ATR_RICH_FEATURES_2026-11-06.md).
Historical publication/revision times remain unverified.

Profile pairs keep their original positions when either side is missing. Cadence
and stride-length curves are handled independently, so a valid frequency curve
does not require an available length curve. Rhythm irregularity is a coarse
cadence-pattern proxy, **not an official fluency grade** or a clinical diagnosis.

## What “fluency” refers to

**Correction after inspecting the user's Chepstow example:** the earlier audit
covered only the three endpoints already collected. ATR has a separate **Jumps
Data** tab with an **A–E fluency grade**, early/middle/late/average jumping speed
loss, and within-race consistency. It was omitted from this inventory and the
collector. Its course benchmark and revision date are also useful source data.
See [the jumps-tab audit](ATR_JUMPS_AND_SOURCE_GAPS_2026-11-06.md) for verified
values, collection status, missing finishers and historical-use restrictions.

The retained-source audit inspected 150 times, 150 tools and 150 stride fragments.
Its grey grade-pill title was the site's `Efficency Score` label. That **A–F
sectional efficiency** measure is separate from **A–E jumping fluency**. The
49-column field-usage map describes the existing sectional collector, not all
data ATR publishes.

For another jumping measure, [RaceiQ's Jump Index](https://raceiq.com/jump-index/) is a
separate score out of ten that considers jumping efficiency, speed and fluency.
The project already collects and derives prior RTV features for jump index,
lengths gained jumping, speed lost, recovery time and positive-jump percentage.
Those direct jumping measures are not in the currently configured 129-input list.
The existing ATR comment tagger also has a prior jumping-error rate, a much
coarser description of mistakes and non-fluent jumping.

## What remains stored without a direct model consumer

| Retained information | Current status |
|---|---|
| Split/cumulative times and opening-section distance | Available for distance-aware derivations; raw split variation would confuse unequal section lengths |
| Per-section pace/Front flags and section notes | Retained; most current run-style use comes from comment tags instead |
| Race early-pace flags and full pace-bar colour stops | Context retained; published race finishing percentage is used by the new pack; colours are not treated as numeric measurements |
| Stride reference lines and full chart attributes | Retained for interpretation/replay; a peak-average line must not be labelled an optimal/par curve without verification |
| Native identifiers, source dates, URLs and hashes | Used for joining, validation and replay; these are not intended as ability predictors |
| Tracking counts and missing runners | Quality/coverage evidence; missing telemetry is not a zero performance |
| Adjusted placing | A consumer exists, but the older audit's saddle-cloth fallback concern is a separate unresolved semantic issue |

This is broader than a list of speed features. Some information is unmodeled,
some is intentionally metadata, and some repeats an existing measure. The map
allows these cases to be reviewed separately without deleting useful raw evidence.

## Coverage and validation

The expanded pilot matched **31,141 source records** to canonical identities.
It retained every race appearance for those horses: 195,704 historical rows from
the 757,580-row database. Removing unrelated horses avoids unnecessary computation
without changing horse-specific history.

| New family | Usable source measurements | Prior predictor coverage in this source cohort |
|---|---:|---:|
| Efficiency log error / grade rank | 98.32% | 60.18% |
| Optimum-profile worst gap / irregularity | 99.95% | 60.72% |
| Cadence rhythm irregularity | 59.34% | 39.90% |
| Efficiency repeatability | At least three eligible observations required | 17.38% |
| Relative-finish repeatability | At least three eligible observations required | 17.66% |

These are coverage results in a **partially rebuilt source cohort**, not whole-
project coverage or predictive improvement. The older history still being
collected is especially important for the consistency measures.

**178 tests passed**, including native error direction/tail retention, placeholder
zeros, grade ordering, aligned profile gaps, smooth-versus-irregular cadence,
minimum consistency history, current/future-data isolation, scraper/recovery
contracts and feature-signature coverage. Targeted lint passed.

Earlier efficiency-specific research tested an imperfect construction. Corrected
features need chronological evaluation on identical complete fields, with
nonfinishers and runners missing current-race telemetry retained. The new pack
remains unselected; no predictive fit or accuracy claim was made.

Outputs: `data/research/atr_efficiency_20261106/`. Audit evidence, field inventory,
source-label checks, test report and recovery receipts are under
`data/audit/atr_efficiency_20261106/`.

## Collector continuation

Collection paused cleanly at **31,041 saved records** before editing the shared
feature file. A consistent queue backup preceded the reviewed metadata update;
no raw rows were reinterpreted, removed or reset. The parser file is byte-identical.

The worker now guards the collector and queue code. `atr_features.py` is not
executed during raw collection, so its unrelated guard dependency was removed;
the ordinary feature signature continues to guard its calculations. A regression
test confirms feature-file edits no longer interrupt collection while collector
or worker edits still change the guard fingerprint. The worker resumed at the
same two-second pacing and advanced beyond the saved checkpoint.
