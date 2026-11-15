# Features from the expanded ATR scrape

**Follow-up:** the [efficiency/consistency audit](ATR_EFFICIENCY_AND_USAGE_2026-11-06.md)
corrects legacy efficiency handling, extends this research pack and maps all
49 collected fields. The seven-measurement pilot below remains its first snapshot.

Seven candidate measurements and nine corresponding prediction inputs were
implemented as a separate research helper on 6 November 2026. They are **not
selected model inputs**. No collector file, ordinary feature stage, fitted model
or accepted ATR cache was changed. The historical backfill continued running.

## What the richer collection retains

The corrected times/tools/stride collectors retain:

- Split and cumulative times with missing positions preserved; sectional pace
  and Front markers, plus section notes.
- Labelled sectional-average speeds, **separate** instantaneous top and finish
  speeds, and 0–30 mph acceleration **seconds**. Mixed units are no longer treated
  as equivalent speed measurements.
- Full stride-length and frequency curves, independent marker labels and source
  reference lines, instead of relying on peak summary values alone.
- Published race-level early pace, race finishing-speed percentage and opening
  section distance, alongside each horse's finishing percentage.
- Labelled extra metrics, chart attributes, raw responses and source hashes,
  plus tracking-coverage flags and the runners omitted by the source.

The existing feature stage already derives closing/maximum sectional speed,
turn of foot, energy efficiency, peak stride measures, upgrades and comment
history. The new helper focuses on measurements that stage does not consume.

## New measurements

| Candidate | Calculation and interpretation |
|---|---|
| Acceleration time | Reported seconds from 0 to 30 mph; historical median gives a starting-acceleration profile. Smaller is quicker. |
| Finish relative to the race | Horse finishing-speed percentage minus ATR's published race figure, in percentage points. Conditions a strong-looking finish on that race's context. |
| Speed change at the line | `100 × (instantaneous finish speed / final sectional-average speed − 1)`. Describes the change through the finish. |
| Burst reserve | `100 × (instantaneous top speed / fastest sectional-average speed − 1)`. Separates a brief peak from sustained segment speed. |
| Late stride-length change | Compare the mean at 1f/Finish with the mean at 3f/2f; report percentage change. |
| Late cadence change | The equivalent change in stride frequency, using its own marker labels. |
| Combined stride-speed change | Multiply the two late/middle ratios, then convert the result to percentage change. Units cancel; this is a pattern-change index rather than an independently measured speed. |

Neither slowing nor stride shortening automatically means fatigue: easing,
course shape, jumping and race circumstances can cause them. "Burst reserve"
is a candidate name, not proof that a horse had unused ability.

Markers are matched explicitly. Missing, duplicated or unknown markers stay
missing; a missing final speed is never replaced by the previous section.
Burst reserve requires all named sectional speeds. Instantaneous mph and
acceleration seconds are excluded from that sectional sequence.

## Prediction inputs and timing

Each measurement supplies a **median over the horse's last three eligible race
appearances**. Runs without ATR measurements still occupy a slot. Two quality
inputs accompany them: the number of those appearances with observations, and
days since the last eligible observation. This gives seven candidate predictors
plus two quality inputs.

Only canonical horse IDs link history; display names cannot conflate different
horses. The as-of join releases a race's measurements at least 48 hours after
its scheduled UTC off, strictly before the target off. Current-race telemetry,
outcomes and future appends cannot affect that race's prediction inputs.

The 48-hour lag follows the current collector's conservative policy. Historical
publication/revision times have not been reconstructed, so it is **not a proof
of actual historical availability**. Test longer delays before making an
operational or economic claim. A longer delay can be set in the helper or CLI.

## Measured coverage

The first snapshot matched **29,184 fresh source records** to canonical runner
IDs, with no unmatched records. The history calculation included all **757,580
canonical race appearances**. Of the observed source rows, 23,311 came from fully
tracked fields and 5,873 from source tables that omitted nonfinishers.

| Measurement | Usable value on the observed run | Prior-run predictor available |
|---|---:|---:|
| Acceleration | 63.49% | 42.78% |
| Finish relative to race | 99.97% | 59.14% |
| Speed change at line | 93.89% | 57.36% |
| Burst reserve | 95.68% | 57.76% |
| Late stride length | 59.95% | 38.98% |
| Late cadence | 59.99% | 39.00% |
| Combined stride pattern | 59.95% | 38.98% |

**17,263 runners** in this source cohort had at least one usable prior candidate.
These percentages describe the partially rebuilt source cohort, **not coverage
across the whole training dataset**. Older history is still being backfilled.
Requiring explicit stride markers also reduces usable coverage relative to the
mere presence of a chart.

Outputs: `data/research/atr_rich_20261106/run_metrics.parquet`,
`prior_candidates.parquet` and `summary.json`. The raw run metrics are post-race
measurements and must not be supplied directly as prediction columns.

## Evaluation still required

Earlier ATR families mostly failed to add useful information beyond market
prices, including the shortlisted closing-in-hand signal's economic test.
See [ledger section 2b](../NEGATIVE_RESULTS.md#2b-atr-sectionals--20-of-21-signals-already-priced-2026-10-12).
Those findings are a reason to test a small pack, not create hundreds of
interactions or assume the repaired source guarantees a model gain.

After sufficient backfill, compare the current baseline with each family and
the small combined pack using the existing purged chronological evaluation.
Assess race log loss and Brier on identical complete fields, then market residuals
and BSP CLV where relevant. Previously inspected dates remain development data.
Keep nonfinishers and runners without current-race ATR observations in prediction
targets: selecting targets on current tracking availability would bias a trial.

No predictive fit or performance estimate was run for this prototype. Integration
would additionally require normal feature-signature registration, incremental
state and training/live parity checks.

## Code and checks

- [Research helper](../../src/research/atr_rich_features.py).
- [Reproducible derivation command](../../scripts/derive_atr_rich_features.py).
- [Measurement and timing tests](../../tests/test_atr_rich_research_features.py).

Seventeen tests cover formulas, independent marker alignment, missing sections,
mixed units, canonical identities, full-race window slots, publication delay,
duplicate appearances, and current/future-data isolation. Targeted lint passes.
The derivation confirmed the three running-worker file hashes were unchanged;
the worker remained running and increased its saved-row count during the pilot.

```powershell
.venv\Scripts\python.exe scripts/derive_atr_rich_features.py
```

Use a separate `--output` directory for later snapshots or `--lag-hours 72` for
a longer-delay sensitivity check. These commands write research artifacts only.
