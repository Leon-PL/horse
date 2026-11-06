# Confirmation feasibility and baseline pilot — package007

**Historical planning analysis and registered pilot.** These assumptions and
sample targets belong to the studies named below. See the [README](../../README.md)
for the current study and its registration status; a new study requires its own plan.

The proposed market-parity margins are much tighter than a short independent
forecast comparison can establish. Historical variance implies about
**1,215,235 paired races** to demonstrate practical parity if a future
independent forecast truly has the same mean scores as the market and retains
this variance. This is a planning approximation, not a requirement to improve
the model or a prediction that the current model will reach parity.

The archived independent forecast still has a positive loss gap. Under that
observed mean, there is **no sample size that gives the targeted high power
to declare parity**. More observations improve the estimate of that gap;
they do not remove it.

The immediate forward stage is therefore registered as a baseline pilot. It
measures recording coverage, input timing, current forecast scores and
variance. It cannot automatically declare parity, advance a correction or
promote a model.

## Frozen planning analysis

The planning manifest was saved before calculating losses. It uses two
already inspected archives, with no fitting, feature search or weight changes:

| Variance source | Complete races | Dates | Historical race LL gap | Runner Brier gap |
|---|---:|---:|---:|---:|
| Retained independent recipe, 2025Q3/2025Q4/2026Q1 | 8,942 | 270 | +0.119514 | +0.00372627 |
| Unpromoted market correction, 2026Q3 | 3,380 | 92 | +0.00029654 | +0.00001357 |

These refer to the archived fits, not the current October deployment refit.
Both reference normalized Betfair BASIC last-traded prices. Live recording
uses Matchbook best-back prices, a different source and population. Transfer
of historical variance is an assumption to examine in the fresh pilot.

All original fields are validated before calculation. The one fixed price
freshness subset requires all runner price ages at most 300 seconds and market
observations at most 90 seconds. It retains 7,719 independent-comparison races
and 2,944 correction-comparison races. Whole fields remain intact. No price,
runner or outcome is substituted.

Sample-size estimates use the **largest** requirement across pooled and
calendar-quarter variances, both full and fresh fields. They account for
whole-race and whole-date dependence. Brier keeps its runner weighting:
race influence is `(race Brier-difference total - overall runner difference *
field size) / mean field size`. Equal averaging of race Brier differences
would answer a different question.

## Sample-size scenarios

Differences below are candidate minus market; lower is better. The margins
remain **+0.002 race log loss and +0.0001 runner Brier**.

| Assumed future mean differences, LL / Brier | Independent variance | Archived correction variance |
|---|---:|---:|
| Exactly equal scores: 0 / 0 | 1,215,235 races | 2,244 races |
| Half of each allowed margin: +.001 / +.00005 | 4,860,938 | 8,975 |
| Hypothetical advantage: −.005 / −.0002 | 99,203 | 216 |
| Each archive's observed historical mean | No high-power parity sample | 3,093 |

The hypothetical advantage is a sensitivity assumption, not a fitted or
achieved forecasting result. The correction remains unpromoted: its earlier
comparison found no improvement over the raw market. Its lower variance
illustrates why a forecast close to market probabilities can be easier to
confirm. Returning M0 itself equals that reference by construction and does
not establish independent forecasting skill.

At an illustrative 25 eligible races per calendar day, the exact-mean scenario
would take 48,610 days for the independent variance and 90 days for the correction
variance, before any extra constraints. Those rates are **not measured live
coverage**. Missing prices, field changes and manual recording will affect
the actual rate. A 12-week minimum alone is not an adequate sample-size rule.

The formula is the one-sided normal approximation
`N = (z(1-alpha) + z(target marginal power))² * variance / headroom²`, where
headroom is margin minus assumed mean difference. Here alpha = .05. Four
conjunctive gates are planned: both scores must pass using both race and date
uncertainty. Each gate is powered at 95%, giving a conservative 80% joint-power
lower bound by the union bound under the approximation. This does not assume
independent endpoints. Requiring all gates controls the global false-pass
rate; it is not a search for whichever test passes.

The [NIST sample-size reference](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm)
supports the normal approximation and its dependence on assumed variance and
effect size. The statistical discussion of jointly required endpoints in the
[FDA multiple-endpoint guidance](https://www.fda.gov/media/162416/download?attachment=)
also explains the loss of joint power and why individual endpoint power must
be higher. These are statistical principles applied here; this project is
not a clinical trial. Variance estimation error, changes of season, date
dependence beyond the estimated clusters and source changes can alter the
realized power. These estimates are feasibility evidence, not a guaranteed
formal design.

## Registered baseline pilot

An exclusive, checksum-anchored companion plan was added to
`data/forecasts/forward/20261026_baseline_v2/readout_plan.json`. The original
study protocol, model and earlier recordings remain unchanged.

- Start: **October 27, 2026**, the next full UK calendar day.
- Retain the pinned independent model and the identity market reference.
  The unpromoted correction is not added to live forecasts.
- Close the sample at the end of the first full UK day satisfying all three
  requirements: **84 calendar days, 1,000 eligible published pairs and 60 distinct
  race dates**. Include every eligible pair through that day.
- Determine closure using forecast publication metadata. Wins, losses,
  probability scores and settlement availability do not control the stop.
- Wait seven full calendar days after closure for results. Retain and report
  pending, inconsistent and changed result fields under the existing rules.
- Earliest possible sample close is **January 18, 2027** and earliest possible
  readout is **January 26**, if both count requirements are already satisfied.
  Lower recording coverage extends the pilot.
- Interim descriptive reports remain available. They cannot shorten the
  sample, establish parity, tune a candidate on pilot outcomes or promote one.

The 1,000-pair threshold is an operational minimum for a variance and coverage
pilot, **not the powered sample size for a parity claim**. The pilot's dates
become development evidence. A later confirmation requires its own frozen
candidate, update rule, reference, power assumptions, sample/stopping rule
and fresh start. Margins should not be loosened after looking at that
candidate's confirmation results.

The Picks app displays pilot pair/date progress. The shared readiness checker
reads publication metadata without reading result labels:

```powershell
.venv\Scripts\python.exe scripts/check_forward_readout.py `
  --study data/forecasts/forward/20261026_baseline_v2 `
  --plan data/forecasts/forward/20261026_baseline_v2/readout_plan.json
```

`--register` is only for creating a new exclusive plan before its first full
day. Re-running it on an existing plan fails. `--output NEW_PATH` optionally
saves an exclusive readiness receipt. The command runs once; no collection
schedule or wakeup is created.

The frozen variance analysis is reproducible using
`scripts/plan_forecast_confirmation.py --study STUDY --output-dir NEW_DIRECTORY`.
It refuses to overwrite an earlier analysis and records input/source hashes.
Evidence is in `data/audit/confirmation/007_feasibility/`, with qualification
logs under `data/audit/20261026/confirmation-007-*.log`.

Qualification: **62 focused checks passed** in 25.12 seconds, including 27 new
tests, plus an app smoke check for pilot counters and the saved forecast
table. The earlier full 793-test regression receipt is preserved; these
isolated additions were checked with the relevant scoring/recorder tests.
The model, frozen recording implementation and feature generation are
unchanged. Earlier package006 evidence and original study files are verified
again during sealing.
