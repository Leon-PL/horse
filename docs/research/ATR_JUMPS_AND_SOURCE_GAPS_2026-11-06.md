# ATR jumps data and remaining source gaps — 6 November 2026

**Superseded collection status:** at 12:08 UTC the main scraper and historical
worker were upgraded to collect both Jumps Data and another missed tab,
**Distance Ran**, alongside the original three tabs. The 4,224 earlier fields
are queued for catch-up, preserving all 33,918 saved rows. See [the current
five-tab coverage guide](../operations/ATR_TAB_COVERAGE_2026-11-06.md). The initial
prototype/inspection results below remain useful evidence, not the current
worker's collection scope.

The previous audit missed an entire source tab. The 49-field inventory covered
the three **existing sectional endpoints**, not everything on ATR. ATR's A–E
**jumping fluency grade** is different from its A–F **energy efficiency grade**
and Racing TV's RaceiQ Jump Index. The cadence proxy added earlier does not
substitute for the official jumps measurements.

## Verified source inventory

The user's [12:20 Chepstow result](https://www.attheraces.com/racecard/Chepstow/04-November-2026/1220/jumps-data)
exposes a fourth timing endpoint:
`/ajax/sectionals-razor/1629661/jumpsdata`. The browser tab, its help text and the
raw response were inspected. These fields were **not** in the ongoing three-tab
sectional collector:

| Source measurement | Meaning | New retained field |
|---|---|---|
| Early speed lost | Percentage lost jumping in the first third of the race | `atr_jump_speed_lost_early_pct` |
| Middle speed lost | Same, middle third | `atr_jump_speed_lost_middle_pct` |
| Late speed lost | Same, final third | `atr_jump_speed_lost_late_pct` |
| Average speed lost | Published whole-race jumping loss | `atr_jump_speed_lost_avg_pct` |
| Fluency grade | A–E comparison with previous winning performances at the course | `atr_jump_fluency_grade` |
| Consistency | Variation in jumping speed loss **within this race** | `atr_jump_consistency_pct` |
| Course-winner benchmark | Published average jumping loss for course winners | `atr_jump_course_winner_speed_lost_pct` |
| Par revision date | Date the source says its par data was updated | `atr_jumps_par_updated_date` |

Percentages remain percentage numbers: 13.6 means 13.6%, not 0.136. The endpoint
does not explain the exact consistency formula or expose individual obstacle
measurements in these inspected responses. Do not invent jump counts, obstacle
entry/exit speeds, heights, recovery times or raw GPS from these aggregates.
RTV has some separate jumping measurements; their definitions are not interchangeable.

The collector also retains native horse/race IDs, cloth number, official and
source-adjusted placing, running comments, help definitions, all labelled pills
(including unknown future metrics), source reference, URL, response hash,
retrieval time and field coverage. Raw bodies remain in `data/atr_cache/raw.sqlite`.

## The example shows why all six measurements matter

| Horse | Early loss | Middle loss | Late loss | Average loss | Grade | Consistency |
|---|---:|---:|---:|---:|---|---:|
| Gingerbred, winner | 15.4% | 15.1% | 10.3% | 13.6% | C | 83% |
| Roe And Co, second | 10.0% | 6.6% | 9.6% | 8.7% | A | 83% |
| Emailandy, third | 14.8% | 13.4% | 15.6% | 14.6% | D | 94% |

The runner-up was the more fluent jumper; jumping is only one component of
performance. Emailandy's high consistency coexists with a poor fluency grade:
consistency alone is not a measure of good jumping. Within-race consistency also
differs from the between-race repeatability derived in the earlier research pack.

The course benchmark is **11.42%**, revised **5 November**, after this race on
4 November. An old result page is not an immutable historical snapshot. Preserve
this revision date and retrieval time. For a historical prediction, allow a
published grade/benchmark only when its revision and availability precede the
prediction; an undated revision is unknown. Alternatively build course pars
strictly from earlier races. The raw percentage loss is the better starting point
for research, still subject to the publication/revision availability assumption.
The sectional guide's approximate 48-hour publication lag has not independently
been verified for this separate jumps endpoint.

Only the three finishers have jumps rows. Annie Express (fell) and Ballynaheer
(pulled up) are missing. Missing telemetry is not a perfect zero-loss performance,
and a finisher-only sample must not be called the whole field. Combining a
horse's jumping measurements with its **prior** completion/fall record is more
honest than discarding nonfinishers from its history.

## Useful information beyond the names supplied by the user

These are candidate uses, not demonstrated prediction gains:

| Information | What it could describe | Collection / use status |
|---|---|---|
| Late minus early/middle jump loss | Whether jumping becomes less fluent later in a race; not a direct fatigue diagnosis | Newly collected raw thirds; candidate derivation pending |
| Average loss and section spread together | Consistently good, consistently poor, or uneven jumping | Newly collected; exact native consistency formula unverified |
| Gap to course benchmark | Jumping in course context instead of comparing raw losses across different courses | New benchmark retained; revision must pass the historical cutoff |
| Jump loss versus other observed runners | Relative jumping performance under the same race conditions | Candidate only; label the finisher/tracked subset and its size |
| Repeated jumping level and variability | Typical loss, best recent performance, improvement and reliability across races | Requires dated prior appearances, including appearances without telemetry |
| Cumulative split times at matching markers | Relative position, seconds behind the leader, moves and ground made/lost through the race | Already retained; much of the existing run-style use relies on comments instead |
| Front flags and sectional notes | Changes of lead and time spent leading, providing direct evidence for running style | Already retained; lightly used |
| Race pace, actual/optimum profiles, opening-section distance | Whether the run was helped or compromised by its pace; avoids comparing unequal splits as equal distances | Already retained; partly used and partly research only |
| Full stride length and frequency curves | How a horse changes its way of moving near the finish, beyond peak values | Already retained; new closing-change and rhythm candidates exist separately |
| Tracking coverage and observation counts | Whether a measurement or historical aggregate is supported by enough data | Must accompany comparisons and repeatability |

The average loss and fluency grade largely express the same dimension. Adding
both is not automatically adding independent information. Likewise, adding many
versions of pace/stride measurements does not guarantee improvement. The
[negative-results ledger](../NEGATIVE_RESULTS.md), especially 025 and 046,
records earlier ATR and RTV family tests, including jumping measures. These new
ATR observations need a focused walk-forward comparison and a size-matched noise
baseline before any model inclusion or claim of value beyond the market.
Prediction inputs must also join through the canonical race/runner identity and
use only earlier, available appearances; ATR horse IDs are source identifiers,
not an automatic cross-provider identity match.

## Other tabs on the supplied result

| Surface | What to do with it |
|---|---|
| Full Result | Core outcome, SP, OR, weight, age, jockey/trainer and headgear already overlap canonical result collection; use source comparison for omissions/corrections, not duplicate measures of the same fact |
| Sectional Times / Tools / Stride | Existing collector retains all 49 mapped columns and raw responses; most improvements here need a reparse/derivation, not another download |
| Jumps Data | The genuinely missing source endpoint; needs a targeted historical collection pass |
| Future Form | Subsequent results are future information. Reconstruct only from races known before a prediction; never copy today's future-form summary into old training rows |

This is an audit of the supplied result's data surfaces and the existing collector.
It is not a claim that every ATR racecard, form page, subscription product or
possible underlying provider feed has been exhaustively inventoried.

## Implementation and validation

`src/ingestion/atr_jumps.py` now fetches and parses one explicitly identified race.
It validates native race/horse identity, keeps units explicit, retains source
definitions and unknown measurements, and distinguishes HTTP absence from a
challenge or changed HTML. Unexpected layouts are unresolved, not permanently
marked as no data. Unknown grades remain visible with a quality flag.

The live pilot retrieved **11 horses in two races** on the supplied Chepstow card:
three at 12:20 and eight at 13:25. The 16:10 response had no labelled runner rows
and remains **unresolved**, rather than being accepted as evidence of no data.
The two positive responses have different course benchmarks, so do not key a
benchmark on the course name alone. No data was purchased or login required.

Twenty-seven new tests cover the retained real fixture, values, identity,
column reordering, unit drift, unknown fields/grades, missing runners and request
failures. The combined collection, ingestion and signature regression run passed
**84 tests**, with no failures or skips. Verification and source observations are under
`data/audit/atr_jumps_20261106/`. Raw collector/worker hashes are unchanged and
the sectional backfill continued during this work.

The project readiness check still reports the pre-existing stale feature/model
generation and incomplete recent sources. This raw-data audit did not rebuild
features, fit a model or change that readiness result. Its snapshot is retained
as `project_status.json` beside the audit evidence.

Example bounded collection, from the project root:

```powershell
.venv\Scripts\python.exe -m src.ingestion.atr_jumps --race-id 1629661 --racecard-url https://www.attheraces.com/racecard/Chepstow/04-November-2026/1220 --expected-horse-ids 3399904 3585720 3508309 3409723 3586853 --output data/atr_cache/jumps/1629661.json
```

**Initial prototype scope, before the integration linked above:** measurements
were validated source observations, outside the running sectional worker and
ordinary features. The subsequent five-tab integration now collects jumps and
distance in the main worker and reuses retained responses for catch-up. The
accepted historical cache, normal feature build and model remain separate.
Run the bounded prototype command only with the main collector stopped; it is
not a second backfill worker.
