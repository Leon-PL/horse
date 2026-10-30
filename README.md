# Horse racing prediction system

**Current model update, 30 October 2026:** the RTV review repairs 75 disputed
fields, recovers 14 missing fields and corrects 12 RTV casualty classifications.
The database, rebuilt histories and new win/place weights are installed together.
[Repairs, RTV features and checks](docs/RTV_FEATURES_REPAIRS_044.md).

The 29 October history-state speed optimization remains active.
[Timing, rollout and checks](docs/LIVE_FEATURE_SPEED_20261029.md).

This is the operating guide. The project collects UK and Irish racing results,
builds historical form and ratings, and predicts each runner's chance of winning
and placing. The objective is reliable probabilities on later races. Market
parity and a profitable betting strategy have not been established.

**Racing TV integration, 30 October 2026:** normal training/app/recorder refreshes
collect, compare and compute RTV features alongside Sporting Life. The feature
paths include 26 prior-run telemetry inputs and four genuine pre-race snapshot
inputs. The fixed trial's gain is uncertain, so the deployed selection remains
154 inputs. [Feature design and evidence](docs/RTV_FEATURES_REPAIRS_044.md),
[original source integration](docs/RTV_PIPELINE_043.md).

**Place calibration added29October2026:** the154-input recipe includes
a recent, training-fitted place-probability layer. Its earlier development
place Brier improved from0.174255 to0.173851. Those are the038 historical
scores, not a new score for the freshly corrected042 model. The recorder
captures future evidence automatically.
See [the six-issue review and evidence](docs/PREDICTIVE_REVIEW_038.md) and
[the current handoff](CONTINUATION.md).

Checks034–036 did not earn a model change. Normal collection now preserves
source-confirmed nonfinish reasons (037), reported claims, penalties and dated
ride descriptions (038). The context feature prototype needs wider evidence
before selection. [History proposals](docs/MARKET_HISTORY_REVIEW_034_035.md),
[market-assisted results](docs/MARKET_ASSISTED_REFRESH_036.md),
[collector repair](docs/NONFINISH_SOURCE_037.md).

Use the project in three ways: **Forecast**, **Maintain**, or **Research**.
The [project inventory](docs/PROJECT_INVENTORY.md) identifies the files and tools
behind each workflow, including historical experiments.

## How it works

```text
Results database -> clean history -> historical feature store
                                        |
                                        +-> train and evaluate -> saved candidate
                                        |                             |
                                        |                       explicit promotion
                                        |                             |
Today's racecards + historical features + promoted model -> forecasts
                                                              |
                                                saved pre-race records
                                                              |
                                                  later assessment
```

A feature is a piece of information about a runner, such as its earlier form.
Updating these histories and fitting new model parameters are separate jobs.
New results can update forecast inputs while the promoted model stays fixed.

The CLI and training app share `src.training.train_run`. Normal training uses
`src.store_refresh.get_training_dataset` for historical features. The picks app
uses the promoted model and `src.live_prediction.features_for_live_cards`.

## Check the current state

Run commands below from the project folder in PowerShell:

```powershell
.venv\Scripts\python.exe scripts/project_status.py
```

This is the authority for current readiness. It checks the results database,
feature history, saved runs and promoted model, including whether their
fingerprints agree. Its full check can take a few minutes; `--quick` skips the
database-content comparison.

**Full database-content readiness, 30 October 2026 at 17:23 UTC: passed.**
Source facts, store, current code and promoted model agree. Installed serving
QA passes on 106 complete fields / 1,010 runners and a saved five-runner card,
with exact single-race/batch and staged/installed input/probability agreement.
The internal scrape timestamp is excluded consistently from the normal CSV
contract. Receipts are in `data/audit/rtv_repair_features_044/installation/` and
`source_contract/`. These are operational checks, not a new predictive-accuracy
estimate. Rerun status for the latest state.

**Recorder verification, 30 October at 17:28 UTC: passed.** The existing
daily/logon schedule is running a fresh source/model-pinned study, eligible
31 October. Successful market discovery and RTV history/snapshot collection
are verified; the latest capture materialised 7,444 genuine pre-race observations.

**Current installation:** promoted run `20261030_135650_092286` uses 154 inputs,
separate LightGBM win/place heads with 3,705 / 2,990 trees, and Platt plus
isotonic win calibration. Its place layer learns a smooth map, field shrinkage
and expected place mass from a purged recent 90-day window. The final heads
were fitted through **28 October**. Results
and feature histories now extend through **29 October**: 755,357 runners in
80,302 races. The 044 review corrects 336 positions, including 19 winners,
in addition to the earlier 042 corrections. Daily currency updates
also work through verified HTTPS; the latest history uses the corrected rates.
Normal training and live forecasts consume these fixes automatically.

The three 033 inputs describe market assessments and opposition in completed
earlier races. Historical market probabilities also inform the training target.
All confirmed Glicko, comparable-mark, source-rating and race-type repairs are
unconditional in ordinary CLI/app training; there is no optional repair track.

For the preceding 033 feature generation, on the same 2,469 historical market fields, race log loss improves
1.789123 to 1.784530, versus the market's 1.686613. Lower is better: this closes
about 4.48% of the prior gap. Win Brier improves overall and for established
handicaps; place Brier improves from 0.174612 to 0.174255. These are previously
inspected development results, not fresh confirmation. The current full-data
weights carry no additional held-out score.
[Results, source checks and installation evidence](docs/PRIOR_MARKET_HISTORY_033.md).

038 keeps those win probabilities fixed. On 2,591 development fields, place
Brier improves 0.174255392→0.173851156, with date 95% delta interval
[-0.000700465,-0.000103740]. The recent win-map trial worsens and was rejected.
These inspected dates do not establish future improvement or market parity.
[Current calibration, source and recorder evidence](docs/PREDICTIVE_REVIEW_038.md).

The earlier performance and continuous-trip proposals remain unselected after
their small screens. Their reports preserve the findings without repeating
large fits: [performance](docs/HANDICAP_PERFORMANCE_031.md),
[race types and trip](docs/RACE_TYPE_AND_TRIP_REVIEW_032.md).

## Forecast: use the installed model

```powershell
.venv\Scripts\python.exe -m streamlit run picks_app.py
```

- **Today's Picks:** inspect win/place probabilities and selection diagnostics.
- **Forecast record:** manually save genuine pre-race forecasts alongside the
  current market reference, then assess saved records when results arrive.
- **Matchbook API:** inspect the exchange connection and prices.

Matchbook credentials belong in the untracked `.env` file. They are needed for
the exchange-dependent pages; historical data collection and offline training
do not require them. Selection and paper-trading displays are separate from
forecast accuracy evidence.

The forecast study matching this installation is
`20261029_feature_speed_20261029_222758_104473`, under
`data/forecasts/forward/`, eligible from **30 October 2026**. Windows task
`HorseForecastRecorder_038` starts at 09:00 and your logon, runs hidden until 03:00,
and prevents overlapping sessions. The computer must be awake and you must be
signed in. Each session refreshes settled histories without refitting weights.
Use `project_status.py` to check its current heartbeat; durable logs are in the
study's `recorder_logs/` folder. **Recorder restart verified at 2026-10-29T22:45:22.** Check its current heartbeat for feed errors. The task writes the original issued forecasts,
market references, refined/legacy place pairs, failures and withdrawal evidence.
Assessment reports both strict matching fields and supported issued fields.
No observations are eligible before 30 October. Earlier studies and unsuccessful
setup receipts remain preserved under their original source pins.

For manual capture, select this study and start **Record in eligible window**
a little before the window; keep that action open until it finishes.
[Recording rules](docs/FORWARD_FORECASTS_006.md) and
[timing rules](docs/TIMED_RECORDING_008.md) explain eligibility; their older study
IDs and pilot plans describe earlier installations.

## Maintain: update history or deliberately replace the model

Open the maintenance app in a separate terminal:

```powershell
.venv\Scripts\python.exe -m streamlit run training_app.py --server.port 8502
```

Use **Data** to inspect coverage and collect recent results, **Train** to fit
candidates, and **Registry** to review and promote a saved run. Tune, Backtest,
Insights and Diagnostics are research/analysis controls.

### Update history without retraining

The following uses the existing shared refresh path. It appends newer dates
when valid and rebuilds if earlier results or feature definitions changed:

```powershell
.venv\Scripts\python.exe -m src.source_pipeline --sync 14 --results-only
.venv\Scripts\python.exe -c "from src.store_refresh import get_training_dataset; get_training_dataset()"
.venv\Scripts\python.exe scripts/project_status.py
```

The first command updates Sporting Life and a bounded recent RTV window,
and skips optional pedigree/weather enrichment; the Data page
offers those separately. A necessary full rebuild can take substantial time.
If feature definitions changed, a compatible model is also required before
serving can resume. Updating histories alone does not make fitted parameters
newer.

### Train, review, then promote

```powershell
.venv\Scripts\python.exe train.py --no-refresh --threads 2
```

This uses current local data, refreshes features as needed, evaluates on the
latest 90 calendar days kept out of fitting, and saves a candidate. Review its
scores, fitting dates and feature compatibility in **Registry**, then explicitly
promote it and rerun the readiness check. A new saved run does not replace
production automatically.

To keep a previously evaluated recipe and fit it with newer settled results,
use **Train -> Refit a validated model with newer results**. That deployment
refit references the earlier evaluation and has no new held-out score.
The [inventory](docs/PROJECT_INVENTORY.md#maintenance-commands) includes the
equivalent commands and explicit full rebuild.

## Research: test one defined change

Start with the [negative-results ledger](docs/NEGATIVE_RESULTS.md) and
[implementation record](docs/MARKET_PERFORMANCE_PROGRESS.md). Use the
[forecast benchmark protocol](docs/FORECAST_BENCHMARK_PROTOCOL.md) for experiment
rules. Older plans are background rather than a standing instruction to run
their work queues.

Define the hypothesis, baseline, data, dates and decision rule before fitting.
Compare identical complete races using race log loss and Brier together.
Use the shared purged evaluation helpers in `src/wf_eval.py`; keep feature
selection, tuning and calibration separate from final evaluation.
Dates already inspected in earlier experiments are not fresh confirmation.
Profitability and execution require separate evidence.

Use the [inventory](docs/PROJECT_INVENTORY.md) to find benchmark, audit and
historical experiment tools. Some are pinned to old inputs or write canonical
data; inspect their settings and effects before running them.

## Setup and development

The live installation uses Python 3.14 and its existing `.venv`. Development
tools use a separate locked environment so setup does not synchronize the
running recorder's packages. From PowerShell:

```powershell
.\scripts\tooling.ps1 setup
.\scripts\tooling.ps1 test
.\scripts\tooling.ps1 validate
.\scripts\tooling.ps1 lint
```

`pyproject.toml` owns declarations; `uv.lock` locks the graph and
`requirements.lock` is its core-runtime pip export. The lock targets
Windows/Python 3.14. See [development tools](docs/DEVELOPMENT_TOOLS.md) for
compatibility limits, the validation pilot, generated tests and measured DuckDB
comparison. The initial Ruff run is a report of existing findings.

For a separate pip installation, `requirements.txt` points to the runtime
export. Local `data/` and `models/` artifacts are excluded from Git; installing
dependencies does not recreate the database, feature store or production model.

Development checks:

```powershell
.\scripts\tooling.ps1 test -TestPath tests
```

Read [CLAUDE.md](CLAUDE.md) for correctness rules before changing calculations.
The [inventory](docs/PROJECT_INVENTORY.md#consolidation-boundaries) records
duplicate refresh paths, source-fingerprint constraints and the next code
consolidation steps. [CONTINUATION.md](CONTINUATION.md) holds session handoff
detail; [PROJECT_OUTLINE.md](PROJECT_OUTLINE.md) retains the older architecture
account. This README is the starting point for operating the project.
