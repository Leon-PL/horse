# Horse racing prediction system

**ATR full rebuild, 5 November (17:58 UTC):** the rich-field omissions and
speed/acceleration parsing error are fixed; 152 relevant checks pass. A fresh
62,432-race reconstruction is running separately from the preserved old cache.
It resumed saved progress after shutdown and continues collecting known IDs
while challenged index lookups wait. [Changes, progress and operating notes](../operations/ATR_REBUILD_2026-11-05.md).

**BSP/ATR follow-up, 5 November (09:18 UTC):** BSP comes from Betfair Historical
archives plus the public daily CSV refresh. ATR now uses the current sectional
endpoints and saved daily race IDs. Normal collection recovered **30 races /
248 runner records**, with all previous observations preserved. Discovery of
uncached days still hits intermittent challenges; **3,936 races remain pending**,
including 60 within the publication allowance. Unattended recovery is partial.
[Provenance, repair and verified limits](../operations/BSP_AND_ATR_2026-11-05.md).

**Source-health follow-up, 5 November 2026:** targeted refreshes preserve other
providers' coverage and check times; busy stores request a retry; ATR cache
writes are atomic. BSP now covers all finished 4 November fields. ATR recovered
one field before blocking again, with 3,966 still queued. New inputs and the
collector update require a feature refresh before the evaluated model refit.
[Recovery, preservation and verification](../operations/STAGE1_SOURCE_HEALTH_2026-11-05.md).

**Runner identity update, 5 November 2026:** source horse IDs now keep namesakes
separate in storage, repairs, rating joins and settlement. The last historical
SL review case is repaired; canonical results and RTV extend through
4 November. The compatible history was rebuilt and installed before the later
source follow-up above. The promoted model needs an evaluated refit against the corrected
history. [Migration, preservation and current checks](../operations/STAGE1_IDENTITY_2026-11-05.md).

**Source refresh update, 4 November 2026:** the shared refresh now reports
Sporting Life, RTV, ATR, BSP and currency-rate collection together. It resumes
longer gaps, checks recent result revisions, preserves reviewed corrections,
and avoids rewriting an unchanged results CSV. ATR's historical catch-up is
budgeted and resumable. Partial results are visible and stop automatic training
refreshes. [Implementation and operating notes](../operations/STAGE1_IMPLEMENTATION_2026-11-04.md).

The earlier 4 November catch-up brought SL, RTV and BSP through **3 November**;
the newer installed position is recorded above. ATR has a resumable historical
queue. Cached BSP refreshes take about 0.3 seconds and the cached ATR check
about 4 seconds in the measured runs. The saved model still belongs to an older
feature generation. [Recovery results and identity review](../operations/STAGE1_RECOVERY_2026-11-04.md).

The historical SL queue fell from 3,457 jobs to one identity-review case,
subsequently resolved by the migration above. The Data page now provides
bounded **Older result recovery**. [Results, source evidence and identity impact](../operations/STAGE1_BACKLOG_2026-11-04.md).

**Current model update, 30 October 2026:** the RTV review repairs 75 disputed
fields, recovers 14 missing fields and corrects 12 RTV casualty classifications.
The database, rebuilt histories and new win/place weights are installed together.
[Repairs, RTV features and checks](../research/RTV_FEATURES_REPAIRS_044.md).

The 29 October history-state speed optimization remains active.
[Timing, rollout and checks](../research/LIVE_FEATURE_SPEED_20261029.md).

This is the operating guide. The project collects UK and Irish racing results,
builds historical form and ratings, and predicts each runner's chance of winning
and placing. The objective is reliable probabilities on later races. Market
parity and a profitable betting strategy have not been established.

**Experimentation phase, 30 October 2026:** the live forecast recorder, its
scheduled task and the source-pinned forward studies were removed; there is no
live model to protect while the model is still being developed. The feature
signature now ignores comments, docstrings and formatting, and
`train.py --fit-as-of` no longer requires a validation certificate.

**Racing TV integration, 30 October 2026:** normal training/app refreshes
collect, compare and compute RTV features alongside Sporting Life. The feature
paths include 26 prior-run telemetry inputs and four genuine pre-race snapshot
inputs. The fixed trial's gain was uncertain, so the deployed selection then
stayed at 154 inputs (replaced on 2 November by the 126 inputs of 048, which
include RTV history). [Feature design and evidence](../research/RTV_FEATURES_REPAIRS_044.md),
[original source integration](../research/RTV_PIPELINE_043.md).

**Place calibration added29October2026:** the154-input recipe includes
a recent, training-fitted place-probability layer. Its earlier development
place Brier improved from0.174255 to0.173851. Those are the038 historical
scores, not a new score for the freshly corrected042 model.
See [the six-issue review and evidence](../research/PREDICTIVE_REVIEW_038.md) and
[the current handoff](CONTINUATION_before_cleanup_2026-11-05.md).

Checks034–036 did not earn a model change. Normal collection now preserves
source-confirmed nonfinish reasons (037), reported claims, penalties and dated
ride descriptions (038). The context feature prototype needs wider evidence
before selection. [History proposals](../research/MARKET_HISTORY_REVIEW_034_035.md),
[market-assisted results](../research/MARKET_ASSISTED_REFRESH_036.md),
[collector repair](../research/NONFINISH_SOURCE_037.md).

Use the project in three ways: **Forecast**, **Maintain**, or **Research**.
The [project inventory](../PROJECT_INVENTORY.md) identifies the files and tools
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

**Current installation (4 November 2026):** promoted run
`20261104_235300_444862` uses the exploded win objective of
[057](../research/MODEL_IDEAS_057_060.md) and the same 129 inputs as 056
(`feature_sets/win_selected_056.txt`):
- the 126 walk-forward-selected inputs of
  [048](../research/FEATURE_SELECTION_048.md);
- three Timeform comment-score inputs from
  [src/rtv_text.py](../../src/rtv_text.py) (056).

**It cannot serve live picks yet.** The run matches the feature store
(`bc357c18a5e5a261`), but the runner-identity work changed the feature code
after the fit began, so live scoring refuses that store. When that work is
finished: rebuild the store, check the inputs against the current store, then
refit and promote. Run `scripts/project_status.py` for the latest state.

On the same held-out races, 057 beat 056: race LL 1.8512 against 1.8557, a
change of −0.0046 [−0.0097, +0.0003] on 3,268 races, 6 August–29 October. That
matches the walk-forward result of −0.0046 [−0.0064, −0.0026]. The SP market
scores 1.7722 on these races.

The run has separate LightGBM win and place heads with 3,325 / 2,998 trees.
The win head learns from the first three places, with stage weights
(1, 0.25, 0.0625), and uses race temperature calibration (0.9353). Its place
layer learns a smooth map, field shrinkage and expected place mass from a
purged recent 90-day window.

The final heads were fitted through **3 November**, on a store rebuilt with
the 047/048 code fixes and the 051 data fixes:
- non-runner rows removed;
- jockey and trainer aliases merged.

Results and feature histories extend through 3 November: 757,083 runners in
80,489 races. Its evaluated twin is `20261104_212157_439212`; receipts are in
`data/audit/install_057/` (057 in
[implementation progress](../MARKET_PERFORMANCE_PROGRESS.md)). Live comment
scores come from that day's Racing TV snapshot, which the picks app captures
when it loads cards. The 044
review corrects 336 positions, including 19 winners, in addition to the
earlier 042 corrections. Daily currency updates also work through verified
HTTPS, and the latest history uses the corrected rates. Normal training and
live forecasts consume these fixes automatically.

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
[Results, source checks and installation evidence](../research/PRIOR_MARKET_HISTORY_033.md).

038 keeps those win probabilities fixed. On 2,591 development fields, place
Brier improves 0.174255392→0.173851156, with date 95% delta interval
[-0.000700465,-0.000103740]. The recent win-map trial worsens and was rejected.
These inspected dates do not establish future improvement or market parity.
[Current calibration, source and recorder evidence](../research/PREDICTIVE_REVIEW_038.md).

The earlier performance and continuous-trip proposals remain unselected after
their small screens. Their reports preserve the findings without repeating
large fits: [performance](../research/HANDICAP_PERFORMANCE_031.md),
[race types and trip](../research/RACE_TYPE_AND_TRIP_REVIEW_032.md).

## Forecast: use the installed model

```powershell
.venv\Scripts\python.exe -m streamlit run picks_app.py
```

- **Today's Picks:** inspect win/place probabilities and selection diagnostics.
- **Matchbook API:** inspect the exchange connection and prices.

Matchbook credentials belong in the untracked `.env` file. They are needed for
the exchange-dependent pages; historical data collection and offline training
do not require them. Selection and paper-trading displays are separate from
forecast accuracy evidence.

There is no forward recorder. It was removed on 30 October 2026 with its
`src/forward_*` modules, scheduled task and study-transition scripts; the code
is in git history before that date, and the earlier studies' data remain under
`data/forecasts/forward/`. The task definition is saved in
`data/audit/recorder_removal_20261030/`.

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

The first command updates the required sources and returns their coverage report.
RTV recovers the requested gap as well as rechecking recent observations. ATR
drains up to 50 outstanding races per invocation; rerun the command to continue
its existing backlog, or set `--atr-max-races` deliberately. A partial report
returns a nonzero exit code. Pedigree (`--enrich`) and research weather
(`--weather`) are explicit optional jobs. A necessary full rebuild can take substantial time.
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

To fit through a settled cutoff with no held-out period, run
`train.py --fit-as-of YYYY-MM-DD`; the run has no held-out score. Passing
`--validation-report` (or using **Train -> Refit a validated model with newer
results**) additionally reuses and checks an earlier evaluated recipe.
The [inventory](../PROJECT_INVENTORY.md#maintenance-commands) includes the
equivalent commands and explicit full rebuild.

## Research: test one defined change

Start with the [negative-results ledger](../NEGATIVE_RESULTS.md) and
[implementation record](../MARKET_PERFORMANCE_PROGRESS.md). Use the
[forecast benchmark protocol](../operations/FORECAST_BENCHMARK_PROTOCOL.md) for experiment
rules. Older plans are background rather than a standing instruction to run
their work queues.

Define the hypothesis, baseline, data, dates and decision rule before fitting.
Compare identical complete races using race log loss and Brier together.
Use the shared purged evaluation helpers in `src/wf_eval.py`; keep feature
selection, tuning and calibration separate from final evaluation.
Dates already inspected in earlier experiments are not fresh confirmation.
Profitability and execution require separate evidence.

Use the [inventory](../PROJECT_INVENTORY.md) to find benchmark, audit and
historical experiment tools. Some are pinned to old inputs or write canonical
data; inspect their settings and effects before running them.

## Setup and development

The installation uses Python 3.14 in `.venv`, built from `uv.lock` on the
project-local runtime in `data/tooling/python`. Development tools use a
separate locked environment under `data/tooling/`. From PowerShell:

```powershell
.\scripts\tooling.ps1 setup
.\scripts\tooling.ps1 venv
.\scripts\tooling.ps1 test
.\scripts\tooling.ps1 validate
.\scripts\tooling.ps1 lint
```

`pyproject.toml` owns declarations; `uv.lock` locks the graph and
`requirements.lock` is its core-runtime pip export. The lock targets
Windows/Python 3.14. See [development tools](../operations/DEVELOPMENT_TOOLS.md) for
compatibility limits, the validation pilot, generated tests and measured DuckDB
comparison. The initial Ruff run is a report of existing findings.

For a separate pip installation, `requirements.txt` points to the runtime
export. Local `data/` and `models/` artifacts are excluded from Git; installing
dependencies does not recreate the database, feature store or production model.

Development checks:

```powershell
.\scripts\tooling.ps1 test -TestPath tests
```

Read [CLAUDE.md](../../CLAUDE.md) for correctness rules before changing calculations.
The [inventory](../PROJECT_INVENTORY.md#consolidation-boundaries) records
duplicate refresh paths, source-fingerprint constraints and the next code
consolidation steps. [docs/archive/CONTINUATION_before_cleanup_2026-11-05.md](CONTINUATION_before_cleanup_2026-11-05.md) holds session handoff
detail; [docs/archive/PROJECT_OUTLINE_before_cleanup_2026-11-05.md](PROJECT_OUTLINE_before_cleanup_2026-11-05.md) retains the older architecture
account. This README is the starting point for operating the project.
