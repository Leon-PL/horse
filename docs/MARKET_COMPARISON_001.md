# First market comparison — 2026-10-25

The independent forecast still trails the five-minute market reference. The
market-informed baseline matches that reference by retaining its probabilities;
the learned correction adds no demonstrated value. This result supports using
the market as the anchor while independent-model development continues.

## Fixed comparison

Retrospective July 1–September 30, 2026: **3,380 complete identical races /
31,235 runners**. Independent model: 149 current features, hard labels,
LightGBM, purged capacity selection, full refit through June 24, training-only
Platt/isotonic calibration. Correction forecasts came from the earlier fit
through April 25; fitting May 2–30 and validation June 6–24. All choices were
frozen before this paired test was scored.

| Forecast | Runner Brier ↓ | Race log loss ↓ | Winner accuracy |
|---|---:|---:|---:|
| Independent | 0.086853 | 1.873924 | 29.45% |
| T-5 market | 0.083297 | 1.759440 | 34.49% |
| Market temperature (identity selected) | 0.083297 | 1.759440 | 34.49% |
| Bounded correction (market only) | 0.083310 | 1.759737 | 34.49% |

The independent model's race log loss is **6.51% worse** than the market
reference. Paired race log-loss difference: +0.114483, 95% race-bootstrap
interval **[+0.098933, +0.129117]**; date-cluster interval [+0.100480,
+0.128989]. Runner Brier difference is +0.003557. This is a clear gap,
not evidence of practical parity.

The independent model has slightly lower ECE (0.001747 versus 0.002006),
while both proper probability scores and winner accuracy are worse. Further
calibration alone is unlikely to close the discrimination gap. This is a
diagnostic inference, not a tested rejection of every calibration method.

Pre-test validation chose market temperature 1.0. The combination assigned
the independent model **zero weight** and sharpened market probabilities
slightly (alpha=0, beta=1.04067285). On test data that correction is slightly
worse: race log-loss delta +0.000297, 95% interval [-0.000714, +0.001268].
It demonstrates neither independent information nor an improvement over M0.

## Practical parity and sensitivity

The predeclared practical margins are +0.002 race log loss and +0.0001 runner
Brier, using one-sided 95% upper paired bounds and date-cluster sensitivity.
The independent forecast fails both. The market-only correction fits within
these margins retrospectively (race-resampled upper bounds +0.001097 and
+0.00004684; date-resampled +0.001059 and +0.00004896). Identity temperature
matches the market by construction. **None of this is forward confirmation**
or proof that the independent model has reached market performance.

The frozen freshness diagnostic retains **2,944 races**: all runner LTP
updates within 300 seconds and market observation within 90 seconds. The
independent race log loss remains worse (1.855516 versus 1.746214), and the
correction remains slightly worse than the raw market (1.746379). The main
conclusion is not explained by the oldest LTP tail.

## Where the gap lies

These predeclared slices are diagnostics, not selection rules. Experience
uses prior_form_runs, correcting the raw experiment exposure slice that used
a scrape-time lifetime total. Overall predictions and scores are unchanged.

| Slice | Races | Independent minus market race log loss |
|---|---:|---:|
| Flat | 2730 | +0.116038 |
| Chase | 204 | +0.117390 |
| Hurdle | 446 | +0.103640 |
| contains lightly raced | 992 | +0.129979 |
| all at least 3 starts | 2388 | +0.108046 |

The gap spans race types and remains in experienced fields. It is larger
where at least one runner has fewer than three recorded prior starts; this
supports the planned sparse-history investigation, without establishing that
an uncertainty adjustment will work. The largest slice is Flat racing, so
improving only the smaller jumping categories cannot close the overall gap.

## Scope and decision

The reference is normalised Betfair BASIC **last traded price**, not an
executable back/lay quote. Its schedule and active field were reconstructed
from messages available by the fixed decision time. Later withdrawals remain
in the full market snapshot. The paired feature-store comparison excludes
whole races with changed/missing fields or conflicting labels, and is
conditional on identical decision and final starter fields. Other late
racecard inputs lack historical publication timestamps. No as-issued live
performance, profitability or untouched-holdout claim is made.

Keep the current production model unchanged. Keep M0 as the market-informed
forecast reference; there is no evidence here to prefer the correction. Do
not repeatedly select features against this quarter. The next registered
independent test uses earlier development quarters and 355 eligible columns,
with the ten future-dependent sectional composites/descendants excluded.
Review the remaining family for future invariance before running it.

## Evidence and verification

The experiment directory retains the original manifest, source ZIP, frozen
160-column full-history projection, model artifacts, calibration and test
predictions, correction recipe, original report and parity assessment. The
original report is preserved; its exposure slice is superseded by the
corrected segments in parity_assessment.json. Later diagnostic wording and
experience-column fixes are recorded in the final implementation source ZIP.

Full regression: **726 passed**; after the final diagnostic corrections,
**61 focused checks passed**, including the corrected experience split,
interleaved-field rejection, walk-forward checks and feature signature
coverage. Current production artifact replay matches saved forecasts to
5.6e-17. Deep status verifies database, features and production consistency;
the existing data-age warning remains.

The plan is still in progress. This completes its foundations and first
fixed market-ladder comparison; independent feature experiments, later model
comparisons and fresh forward confirmation have not been completed.
