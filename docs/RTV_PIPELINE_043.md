# Racing TV comparison and source integration — 30 October 2026

This is the original source-only audit. The subsequent
[044 repairs and feature implementation](RTV_FEATURES_REPAIRS_044.md) resolve
the official-order queue and casualty parsing errors, recover missing fields,
and install the normal feature pipeline and corrected-data refit. The original
measurements below remain preserved as evidence of the starting state.

The current Sporting Life database and RTV's rebuilt 2023-06-10–2026-10-28
backfill were compared offline, plus the preserved 2021 sample day. Evidence:
`data/audit/rtv_pipeline_043/crosscheck/summary.json`, per-quarter reports,
`official_position_conflicts.csv` and `*_runner_links.parquet`.
Reproduce with `scripts/audit_rtv_crosscheck.py`; it never changes results.

## Does shared information agree?

44,223 SL races are all present in RTV. RTV contains 19 additional races in this
window, including the 14 previously recorded incomplete SL syncs. 412,771 of
412,835 RTV starters in matching races can be compared (99.9845%). The audit
allows a unique position/SP pair to identify renamed horses; production context
joins require exact normalised identities and deliberately exclude those cases.

| Comparable runner field | Agree / compared | Agreement |
|---|---:|---:|
| Starting price, within 1% | 412,760 / 412,760 | 100% |
| Age | 412,750 / 412,771 | 99.9949% |
| Nonfinisher status | 412,759 / 412,771 | 99.9971% |
| Official finishing position | 389,534 / 389,870 | 99.9138% |
| Weight, within half a pound | 408,645 / 412,771 | 99.0004% |

Class, distance, going, margins, pedigree and equipment agree closely; the
per-quarter reports preserve the individual measures. The important exceptions:

- **336 official-position conflicts across 75 races remain.** RTV distinguishes physical finish
  and amended order. The corrected SL database still disagrees in some fields;
  examples include Wetherby 14 October and Newmarket/Fontwell 21 October 2026.
  Do not treat a source's stated amended order as permission to overwrite settled
  labels. These are a review queue, with both values retained.
- Before December 2024 RTV can show allotted/long-handicap weight while SL shows
  carried weight. Modern weights agree essentially completely.
- Irish first prize before 2024 can be the total purse labelled as the winner's
  prize. Preserve SL's prize/currency handling.
- Bumpers, trainer/jockey naming, renamed horses and code-specific days since run
  differ in meaning or presentation. Exact raw-string comparison overstates errors.

The original scraper guide's historical amendment agreement figures predate
the installed 042 correction; this report uses the current database.
Fresh SL spot checks confirm RTV's amended order for all 16 disputed positions
in the Wetherby, Newmarket and Galway sampled fields: those database values are
stale. Fontwell's current SL response is rejected by the existing parser because
it assigns amended position eight in a seven-starter field. Saved source pages
and the read-only comparison are in `spot_checks/`; the remaining queue has not
been independently adjudicated. Correcting canonical histories and qualifying
replacement model weights is separate work from connecting this source.

## Additional information we can use

RTV has **4,554,635 metric observations for 389,384 runners / 42,058 races** and
**4,656,240 furlong splits for 380,938 runners / 41,102 races**. It adds finishing
speed, acceleration, top speed, stride length/frequency and jumping efficiency,
entry/exit speed, recovery and lengths gained jumping. Timeform assessments,
in-running descriptions, pace/draw maps, RaceiQ prompts and bookmaker terms are
also retained.

The clearest new research candidates are causal summaries of prior jumping and
stride/sectional performance, plus genuinely captured pre-race pace/draw context.
These are candidates, not demonstrated improvements. Do not use a current race's
telemetry, retrospective card ratings or scrape-time profile figures as pre-race
inputs. Existing ATR history overlaps some speed/sectional information.

Quality filtering matters: the backfill contains 597 out-of-range metric flags,
162 inconsistent metric ranks, 4,515 winner-time discrepancy flags and seven
sector-count/distance flags. Use values with the appropriate quality exclusions,
not blindly copied ranks or official race times. See `RTV_SCRAPER.md` for semantics.

## What is wired in

`src/source_pipeline.py` is the source orchestration seam. Ordinary CLI/app
training refreshes, the Data app and recorder history refresh call its unified
`sync_database`. It runs SL's transactional ingest/export, then collects up to
ten recent RTV dates, parses all source tables, writes identity links and saves a
comparison report per date. `enrich=False` skips optional weather/pedigree work
while preserving RTV collection. Long historical backfills remain explicit.

The picks app captures expiring RTV race-day data when cards are requested.
The continuous recorder captures it every 15 minutes on a separate worker;
forecast workers and their issue windows are independent of RTV latency.
Historical-card requests never fabricate past snapshots. Completed daily source
refreshes are cached for six hours; race-day captures for 15 minutes.

`data/rtv_cache/pipeline/` contains immutable parsed generations and atomically
published `history.json` / `snapshots.json` manifests. Raw responses and every
dated snapshot remain in RTV's SQLite store. A writer lock avoids competing
app/recorder publishers. A failed update keeps the last complete published
generation and records the error; `project_status.py` exposes RTV health.

For a manual source refresh:

```powershell
.venv\Scripts\python.exe -m src.source_pipeline --sync 14 --results-only
.venv\Scripts\python.exe -m src.source_pipeline --snapshot
```

For offline analysis:

```python
from src.source_pipeline import load_matched_history, load_history_table
runners = load_matched_history("2026-10-01", "2026-10-29")
metrics = load_history_table("metrics", "2026-10-01", "2026-10-29")
sectionals = load_history_table("sectionals", "2026-10-01", "2026-10-29")
```

Matched runners carry separate `sl_race_id`, `sl_horse_id`, `rtv_race_id` and
`rtv_runner_id`. Never equate numeric IDs across providers. Shared project
normalisers join date, track, UTC scheduled time and horse name, rejecting
ambiguous keys. Daily tables replace corresponding backfill dates without
duplicating runners. No canonical labels, prices, selected inputs or fitted
parameters are changed by this integration.

## Installation checks

The live initial refresh collected all 38 races / 387 linked runners for
29 October and updated ten recent dates. Today's snapshot found 45 races and
43 pace maps, with 223 successful endpoint responses. Missing/404 endpoint
responses remain visible rather than being fabricated.

Source, recorder, training and identity/overlay checks are recorded in
`data/audit/rtv_pipeline_043/`. The installed 154-input model and feature-generation
signature remain unchanged. The recorder requires a new pinned implementation
study because its source collector changed; earlier studies remain immutable.

Installed recorder study:
`20261030T093356Z_rtv_sources_043_20261030_015917_823047` (eligible 31 October).
The existing task's daily/logon triggers are preserved. Deep readiness passed
at 09:38:44 UTC, including canonical data agreement. Closure at 09:40 UTC
verified live recorder heartbeat, successful market discovery, source health,
unchanged canonical CSV/model hashes and zero new fits. Evidence is in
`data/audit/rtv_pipeline_043/installation/final_status.json` and `tests.json`.
