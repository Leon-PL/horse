# Experiment 002 — independent sectional feature family

**Complete: reject this family under the registered recipe.** All three
periods finished; production is unchanged. The objective is calibrated win
probability performance; betting returns do not select this experiment's winner.

## Results and decision

Across **9,356 races / 85,863 runners**, the added sectional family worsened
all three pooled proper probability scores. Race log loss was higher in every
quarter. The candidate fails the registered advancement rule; retain the
149-feature independent recipe. No feature subset is selected from these
test results.

| Forecast | Runner Brier ↓ | Binary log loss ↓ | Race log loss ↓ | Winner accuracy |
|---|---:|---:|---:|---:|
| Baseline, 149 columns | 0.08754346 | 0.30088334 | 1.87204489 | 29.03% |
| Candidate, 355 columns | 0.08755972 | 0.30140404 | 1.87671309 | 29.25% |

Candidate minus baseline race log loss: **+0.00466820**, paired race-bootstrap
95% interval **[+0.00085219, +0.00922854]**; date-cluster interval
**[+0.00097963, +0.00932609]**. Runner Brier difference is +0.00001626;
binary log loss difference is +0.00052070. Slightly higher winner accuracy
and lower pooled ECE do not override worse proper probability scores.

| Period | Baseline race log loss | Candidate race log loss | Candidate minus baseline |
|---|---:|---:|---:|
| 2025Q3 | 1.92063259 | 1.92763771 | +0.00700512 |
| 2025Q4 | 1.86228139 | 1.86496364 | +0.00268225 |
| 2026Q1 | 1.81799380 | 1.82195439 | +0.00396059 |

This closes the proposed unchanged full-family addition. It does not establish
that every possible use of sectionals is unhelpful. Any reopening needs a
materially different, justified hypothesis and new predeclared evidence.
The next bounded modelling comparison in the accepted plan is conditional
race logit / race-softmax, beginning with the retained 149-feature recipe.
The previous failed Plackett–Luce attempt remains a low prior for that trial.

## Question and fixed comparison

Can historical ATR sectionals improve the independent forecast, even though
earlier research found little residual information beyond market prices?
The evidence review found no equivalent earlier independent full-family
proper-score comparison. It retained the older negative market-residual and
betting evidence; this is a different, narrower question.

- Baseline: the existing 149 input columns.
- Candidate: the same 149 plus 206 ATR columns, 355 in total.
- Periods: 2025Q3, 2025Q4 and 2026Q1, with a seven-day purge.
- Both variants use the same complete fields, float32 feature projection,
  LightGBM settings, seed 42, 180-day recency half-life and training-only
  Platt/isotonic calibration. Each historical fit selects capacity using
  earlier training data, then refits all eligible history through its cutoff.
- No current-race odds enter either independent predictor. No test scores
  select features, parameters or a favourable stopping period.

The advancement rule was registered before execution: pooled race log-loss
improvement with an upper paired 95% bound below zero, no worsening runner
Brier or binary log loss, and lower race log loss in at least two of the
three periods. Passing advances a research candidate; it does not automatically
promote it or establish market parity.

## Temporal review and coverage

Two unused composites, `atr_speed_score` and `atr_stride_x_gallop`, use
full-dataset standardisation. They and their eight derived columns are
excluded. The actual feature functions' synthetic-history audit passed all
206 eligible additions under date prefixes, future appends/changes,
current-day outcome changes and row permutations. Negative controls detected
the unsafe composites; positive controls confirmed that legitimate earlier
information affects the features. This is not proof of historical publication
timestamps for retrospectively collected data.

| Period | Complete races | Runners | Runners with ATR history |
|---|---:|---:|---:|
| 2025Q3 | 3,532 | 32,958 | 67.0% |
| 2025Q4 | 3,233 | 30,081 | 67.1% |
| 2026Q1 | 2,591 | 22,824 | 73.4% |

Runners without sectional history remain in the comparison. Missing
measurements are not recoded as measured zero performance.

## Secondary market context

A separate registration, made before experiment 002 scores were inspected,
fixes the local Betfair BASIC last-traded-price reference at five minutes
before the first published scheduled start. It reports both frozen
independent forecasts against identical complete market fields, with whole
race exclusions for missing/changed fields or disagreeing outcomes. A fixed
freshness diagnostic requires every runner's last price change within 300
seconds and market observation within 90 seconds.

Before trial scores were inspected, the coverage-only check found 8,942
identical complete races / 81,696 runners: 3,348 in 2025Q3, 3,125 in 2025Q4
and 2,469 in 2026Q1. It excluded 394 missing/changed fields and 20 outcome
disagreements. No winner labels were overwritten. The fixed freshness subset
contains 7,719 races. The full market snapshot retains 9,299 unique markets,
including 316 with a later withdrawal; source overlaps agree.

This secondary assessment fits no corrections and does not change the family
advancement rule. It is a retrospective LTP comparison, conditional on the
same decision and final field; it is not an executable-price test, a live
racecard replay or fresh forward confirmation. These dates have previously
been available to project research.

| Forecast on 8,942 matched races | Runner Brier ↓ | Race log loss ↓ | Winner accuracy |
|---|---:|---:|---:|
| Baseline independent | 0.08787478 | 1.86879634 | 29.17% |
| Sectional candidate | 0.08787933 | 1.87297627 | 29.38% |
| T-5 market | 0.08414852 | 1.74928256 | 34.50% |

The baseline's race-log-loss gap is **+0.11951378**, 95% race interval
**[+0.10937481, +0.13054458]**. The candidate widens it to +0.12369371,
interval **[+0.11270102, +0.13543171]**. Date-cluster intervals agree.
Neither forecast approaches the predeclared practical parity margins
(+0.002 race log loss / +0.0001 runner Brier). Baseline one-sided 95% upper
bounds are +0.12833034 / +0.00406072 with race resampling, and
+0.12939536 / +0.00404527 with date clustering; the candidate is worse.

On the fixed 7,719-race freshness subset, race log loss is baseline 1.86806567,
candidate 1.87335820 and market 1.75239937. Both independent forecasts again
fail practical parity in the race and date analyses. The conclusion does not
depend on the oldest price-change tail. Rejected fields and all paired
forecasts are retained separately; no outcome labels were changed.

## Feature-selection guard

After the frozen trial and assessment completed, the ten quarantined columns
were also blocked centrally in `get_feature_columns`, covering unrestricted
selection and explicit allowlists. Two regression cases reproduced the hole
before the fix. Both trial variants and production already excluded these
columns, so the guard changes none of their recorded forecasts. Feature values
and the store remain unchanged; no feature rebuild is needed.

The original fit source ZIP and assessment source ZIP are preserved. The
later guard is a separate downstream change: resuming experiment 002 with
the final working-tree source must fail its source-manifest check rather
than silently mixing implementations. The assessment also verifies the
fitter's manifest text hash with Windows newline handling, while retaining
the file's exact byte fingerprint independently.

## Evidence

Registration, invariance checks, frozen input projection, source archive,
runtime versions, fit metadata and paired forecasts are retained under
`data/audit/market_performance/002_atr_family/` and its adjacent registration
and invariance JSON files. The manifest text SHA-256 is
`25cf986aafd8ba9c072653cb892e4d8cba1a5beb18d33b51d6778d780580dce9`.

Final full regression after the guard: **737 passed**, 24 warnings, in
363.19 seconds. This includes actual LightGBM/XGBoost/CatBoost native fits,
the two guard regression cases and the end-to-end three-period assessment
with native Windows manifest newlines. The earlier pre-fit suite passed
731 tests; its count is not added to the final suite.

Post-guard production replay passed on **64 shuffled races / 635 runners**:
saved win probability error **1.67e-16**, full versus win-only error **zero**.
The earlier 64-race timing sample is illustrative only. Final deep status
verifies database/store/code/production consistency, with the existing
data-age warning now 14 days after midnight. Feature signature remains
`221fcd4db0fc5631`, data `f2a120a477c117a1`; production remains
`20261025_193016_331639`. Source archives, runtime versions and final evidence
checksums accompany the reports. No canonical result or production artifact
was changed in this experiment.
