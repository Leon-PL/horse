# Racing TV (RaceiQ) scraper

`src/ingestion/racing_tv.py` collects everything Racing TV publishes about British and
Irish races, from the JSON API its website calls. It replaces the scraper
deleted on 2026-10-13 (see the [audit](../archive/operations/RTV_SCRAPER_AUDIT_2026-10-29.md)). It
is connected to normal source refreshes through `src/ingestion/pipeline.py`.
Training, the Data app, picks and the recorder collect RTV sidecars automatically.
The installed model has no selected RTV inputs; source collection and predictive
adoption remain separate. See [the current comparison/integration](../archive/research/RTV_PIPELINE_043.md).

## What it collects

Per race (`scrape`), from `https://api.racingtv.com/racing`:

| Kind | Path | From | Content |
|---|---|---|---|
| `day_list` | `/results/list/{date}` | 2009 | meetings, going, races with scheduled start and UTC offset |
| `results` | `/results/{date}/{track}/{hhmm}` | 2009 | positions (physical and amended), margins, SP and favourite code, jockey/trainer/owner, in-running comment, weight, age, draw, equipment, going, winning time, actual off time, tote, TPD sectional-PDF names, RaceiQ flags |
| `racecard` | `/racecards/{date}/{track}/{hhmm}` | 2009 | distance (yards), class, prize, eligibility, rating band, surface, Timeform rating/stars/comment/verdict, pace hints, sex, sire/dam/damsire, days since run, form, smart stats, last year's winner |
| `tips` | `/racecards/.../tips` | 2023 or earlier | Racing TV's selection for the race |
| `odds` | `/runner/odds?runner_ids[]=…` | Dec 2023 or earlier | each bookmaker's last price per runner, each-way places and fraction, shortening/drifting |
| `iq_metrics` | `/iq-results/.../metrics` | 2023-06-10 | finishing speed %, top speed, 0–20 mph, stride length and frequency; over jumps jump index, lengths gained jumping, entry/exit speed, speed lost, time in envelope, recovery time, positive jump % |
| `iq_sectional` | `/iq-results/.../sectional` | 2023-06-10 (most races from Aug 2023) | every furlong: time, cumulative time, race position, gap to leader, pace category; race time index, meeting average, time vs par |

Per day: `tipsters` (`/tips/{date}/tipsters`: named tipsters' selections
with advised price, each-way and outcome). Per course, once: `track`
(`/tracks/{slug}`: surface, track type, course profile text).

**Race day only** (`snapshot`): these payloads exist only on race day and
return 404 the next day, so they must be captured as racing happens:

| Kind | Path | Content |
|---|---|---|
| `pre_pace_maps` | `/iq-races/.../pace-maps` | each runner's average early position and run-style category (over N races), draw advantage in lengths, course statistics by run style (win %, actual vs expected wins) |
| `pre_prompts` | `/iq-races/.../prompts` | RaceiQ text prompts per runner (draw, start, speed, stride, finish, jockey, trainer, position, jumping) |
| `pre_racecard`, `pre_tips`, `pre_odds` | as above | the card, tips and bookmaker prices at the time of capture |

Every capture is kept with its `fetched_at`, so morning and near-off
snapshots can be compared.

**Optional** (`profiles`): horse profiles (`/horses/{slug}`): foaling date,
pedigree with countries, statistics and the full form list (official rating,
going, comments, runs abroad). These describe the horse **as of the scrape**;
they are prefixed `scrape_` and must never be read as race-day values.

### Not collected, and why

- `iq_comparison`: a strict subset of `iq_metrics` (`--with-comparison` adds it).
- TPD sectional PDFs: the same splits, finish times and finishing speed %
  as `iq_sectional` (verified on a sample; times stated as ±0.2 s).
- Race previews: videos and images only. Replays and news: media.
- Jockey and trainer pages, and the track pages' jockey/trainer statistics:
  current figures only, nothing per race.
- `/meetings/{date}`, `/racecards/non-runners`: current day only (empty
  for past dates).

## Speed

Requests run on a thread pool under one shared rate cap (`--rps`,
default 10/s, `--workers`, default 12). Several days are processed together
so that each phase keeps every worker busy. Benchmarked on 2026-10-29: a week
of racing (291 races, 1,784 requests) took 169 s at a 12/s cap, 10.5 requests
a second, with no rate limiting. That is about 0.6 s a race against about
4 s for a one-request-at-a-time version. A rate limit or server error
pauses every worker (exponential backoff, honouring `Retry-After`), and a run
that cannot get through stops; the next run resumes.

The full RaceiQ era (2023-06-10 to now, about 44,000 GB and IE races, about
6 requests each) takes roughly 8 hours at the default.

Racing TV's website terms permit downloading text content to a single
personal computer for personal, non-commercial use. The RaceiQ endpoints need
the site's public API key, which the scraper reads from the site's
JavaScript config at run time (as the BHA scraper does; `RTV_API_KEY`
overrides it).

## Finding every race

Race URLs use UK/IE local time. Each race's URL time and its UTC
`off_time_utc` both come from the day list's scheduled start with its
offset; there is no time-zone guessing.

Older day lists omit some races that the results endpoint still serves (all
of 2021-07-10's first races at Newmarket, York and Chester, also missing from
every payload's `meeting.ordered_races`). After reading the day list, the
scraper checks `data/races.db` for that date; each race at a track with no
listed race within 5 minutes is fetched directly by converting its UTC time to
local. A wrong guess costs one 404. `--no-discover` turns this off.

## Storage

- `data/rtv_cache/rtv_raw.sqlite`: every response body, zlib-compressed:
  - `responses`: one per API path; only 200 and 404 are final.
  - `snapshots`: race-day captures, one row per capture.
  - `days`: per-day status, with a scraper version. Days completed by an
    older version are revisited, but only missing payloads are fetched.
- `data/rtv_cache/tables/{table}/{year}-Q{n}.parquet`: parsed tables, one part per calendar quarter (building a whole year at once peaked at about 9 GB of memory), rebuilt from
  the raw store without network access; read with `racing_tv.load_table(name)`:
  - `races`, one row per race: every scalar field from results (`res_`),
    racecard (`card_`) and sectional (`sect_`), plus cleaned columns
    (`off_time_utc`, `distance_furlongs`, `win_time_s`, `going`,
    `race_class`, `surface`, `first_prize`, `is_abandoned`, …).
  - `runners`, one row per runner, prefixed the same way, plus cleaned
    columns (`finish_position`, `final_position`, `beaten_lengths`,
    `sp_decimal`, `weight_lbs`, `draw`, `jockey`, `trainer`,
    `equipment_symbols`, `timeform_rating`, …).
  - `metrics` (runner × metric) and `sectionals` (runner × furlong, with
    `sector_length_yards` and `sector_speed_mph`).
  - `odds`, `race_tips`, `tipster_tips`, `totes`, `smart_stats`,
    `last_winners`, `meetings`, `tracks`.
  - `quality_flags`: one row per suspicious value; nothing is altered.
  - From `build`, also `pace_maps`, `pace_course_stats`, `prompts`,
    `pre_odds`, `pre_runners` (snapshots) and `horse_profiles`,
    `horse_performances`, `horse_statistics`.
- `data/rtv_cache/validation/*.json`: validation reports against `data/races.db`.
- `data/rtv_cache/equipment_glyphs_learned.json`: glyphs learned by
  `learn-glyphs`.
- `data/rtv_cache/rtv_metrics.parquet`: the old scraper's cache, left untouched.

Nested JSON keys join with `__` and top-level keys with `_`, so
`res_trainer_name` and `res_trainer__name` (different values) both survive.

## Commands

```bash
.venv/Scripts/python.exe -m src.ingestion.racing_tv scrape --start 2023-06-10 --end 2026-10-28
```

```bash
.venv/Scripts/python.exe -m src.ingestion.racing_tv snapshot --date 2026-10-30
```

```bash
.venv/Scripts/python.exe -m src.ingestion.racing_tv profiles --start 2026-01-01 --end 2026-10-28
```

```bash
.venv/Scripts/python.exe -m src.ingestion.racing_tv learn-glyphs
```

```bash
.venv/Scripts/python.exe -m src.ingestion.racing_tv build
```

```bash
.venv/Scripts/python.exe -m src.ingestion.racing_tv validate --start 2025-08-01 --end 2025-08-31
```

```bash
.venv/Scripts/python.exe -m src.ingestion.racing_tv status
```

`scrape` defaults to the RaceiQ era, newest first, GB and IE (`--countries
all` adds the foreign meetings Racing TV lists). Days within 10 days of today
are re-read, so RaceiQ data Racing TV still marks incomplete is fetched
again. `snapshot` defaults to today; run it on race day, ideally morning and
again before racing.

## Equipment glyphs

Racing TV prints equipment as one glyph per item group, each optionally
followed by a superscript (1 = first time, 2 = second time), groups
separated by spaces (`w¹ z`). The table below is the site's own lookup
(`formatEquipmentText` in its app bundle) plus glyphs it renders blank,
decoded from the same runners' Sporting Life headgear (all observations
agreed):

| Glyph | Items | Source |
|---|---|---|
| `0` `+` `B` `b` `D` `H` `h` `V` `v` `z` | p, pt, b, bt, b+e/s, h, ht, v, vt, t | Racing TV table |
| `w` | wind operation (flag, not an item) | Racing TV table |
| `-` `E` `e` `A` `C` `O` | hp, e/s, e/c+t, hv, h+e/s, hb | Sporting Life (10, 4, 2, 2, 2, 2 runners in the sample) |
| `}` `j` `J` `a` `=` | pv, e/c+t, e/c, hvt, p+e/s | Sporting Life, full backfill (7, 4, 2, 2, 2 runners, all agreeing) |
| `o` `G` | bht, b+e/s+h | Sporting Life majority (9 of 11, 5 of 8) and the lower case = upper case + tongue strap convention |
| `*` `g` `d` `F` `c` `L` | hpt, b+e/s+h+t, b+e/s, e/s+v, e/c+h+t, b+e/c | Sporting Life (1 runner each) |

Across all 467,155 runners in the 2023-06 to 2026-10 backfill, one glyph is
left undecoded: `^`, seen once on a non-runner (Tribal Moon, Dundalk
2024-11-27), so there is no evidence for it. Decoded items agree with
Sporting Life for 99.9% of runners each year, and first-time marks for
99.8%. On a combined glyph the superscript does not say which item is new,
so those go to `equipment_first_time_ambiguous` (1–1.6% of runners) instead
of being guessed. A glyph not in the table sets `equipment_decoded = False`
and raises `equipment_glyph_unknown`. `learn-glyphs` then decodes it from
Sporting Life when at least two observations all agree.

## Validation against races.db

The full backfill (2023-06-10 to 2026-10-28, plus a 2021 sample day):
44,242 races excluding 1,976 abandoned and 50 Arab races, and 412,835
runners. Per-year detail is in
`data/rtv_cache/validation/full_2023-06-10_2026-10-28/`.

| Field | Agreement by year (2023 / 2024 / 2025 / 2026) |
|---|---|
| races in races.db found in Racing TV | 100% every year (29 only through discovery) |
| runners matched | 99.98% or better; 85 renamed horses paired by position and SP |
| class | 99.99–100% |
| distance (±0.05 f) | 99.97 / 99.75 / 100 / 100% |
| going (exact text) | 99.9% |
| SP (±1%), margins, non-finishers, age | 99.99–100% |
| draw | 99.9% |
| first prize | 78.4 / 99.99 / 99.99 / 100% (Irish purse issue before 2024, below) |
| weight | 97.6 / 98.0 / 100 / 100% (allotted vs carried, below) |
| finishing position, unamended runners | 100% |
| jockey / trainer, first initial and surname | 98.9–99.9% / 97.9–99.97% |
| sire, dam, damsire | 99.1–99.98% |
| equipment items / first-time marks | 99.9% / 99.8% |
| race type | 97% raw; 99.8–99.9% once bumpers count as Flat (1,320 bumpers filed as Flat by Sporting Life) |

### Where Racing TV fills gaps in races.db

- **Failed Sporting Life syncs.** `result_sync_races` holds 14 races in this
  window that failed the field-size check ("Incomplete or inconsistent
  field: 14 of 13 runners") and were never stored, 13 of them between 13 and
  24 October 2026. Racing TV has a complete result for all 14. The 1,966
  "Empty or incomplete result response" failures are, as far as a
  title match can tell, abandoned races (at least 1,777 match a race Racing TV
  lists as abandoned; it lists 1,976).
- **Races Sporting Life never listed**, e.g. division 1 of the Dore & Co
  Solicitors Handicap at Bellewstown on 2024-07-05 (only division 2 is in
  races.db), and an Irish charity race at the Curragh.
- **Stewards' amendments.** 630 runners had their placing changed. races.db
  holds the official order for 34 / 56 / 70 / 15% of them by year and the
  physical order for the rest, so its winner is sometimes the horse that
  lost the race in the stewards' room. Racing TV has both orders.

## Data quality: what the payloads get wrong, and how it is handled

| Issue | Handling |
|---|---|
| The `iq_sectional` endpoint's `scheduled_start_time` is an hour late in BST (every BST race). | URL time and `off_time_utc` come from the day list; flagged `sectional_schedule_shift` (info). |
| Day lists omit some older races; "completed" lists include abandoned races (runners left `entered`). | Discovery from races.db (above); `is_abandoned`, flagged `race_abandoned`. |
| `trainer_name`/`trainer.name` (and `jockey_*`) differ, usually initials vs full name, but which field holds which varies; some carry `, Ireland` or `J. A.` initials. Older payloads give full names only. | Both kept; derived `jockey`/`trainer` pick the initials form by shape and drop suffixes. Remaining differences are registered initials vs everyday names (A W Carroll / Tony Carroll) and joint licences: Sporting Life shows later joint names on older races. |
| Racing TV shows a horse's **current** name on old results ("Spanish Phoenix", raced as "Turf Phoenix"). | Validation pairs such runners by position and SP and lists them (`renamed_horses`); join on `horse_id` where possible. |
| Stewards' amendments: `finish_position` is the order past the post, `amended_position` the official result. | `final_position` official, `finish_position` physical. races.db holds the physical order for 30–85% of amended runners depending on the year (above), so its winner can be a disqualified horse. |
| Irish `first_prize` before 2024 is the whole purse labelled "Winner": exactly 1/0.6 of Sporting Life's winner prize (60% is the Irish winner's share). | Kept as published; prefer races.db for Irish prize money before 2024. |
| Weights before Dec 2024 are the allotted weight. Sporting Life adds overweight (1–2 lb), and out-of-the-handicap horses show their long-handicap weight (9-5 where 10-2 was carried). | Kept as published: `weight_lbs` is allotted weight, and the long-handicap figure is extra information. |
| `between_distance` is the gap to the preceding finisher (blank for the winner). | `margin_to_prev`; `beaten_lengths` chains it in physical order using `src.result_margins`. |
| Early (2023) sectional payloads omit some runners while positions and gaps count the whole field. | Position and gap checks allow for the missing runners. |
| `starting_price_favorite` is a code: 1 favourite, 2 joint favourite. | `sp_favourite_code`. |
| Barrier 0 means no stalls. | `draw` is NaN. |
| `card_days_since_last_run` counts days since the last run in the same code (always ≥ Sporting Life's figure). | Kept as published. |
| Racecard `horse.age`, `age_days`, `gender` and profile pages are as of the scrape. | Prefixed `scrape_` (cf. the `sex` look-ahead leak). |
| RaceiQ values are revised after publication at some jumps courses (lengths gained jumping mostly ±0.05; top speed up to 0.41 mph). `metrics_complete` can stay false for months. | Latest fetch wins; races are re-fetched for 10 days after they run. |
| A metric's `average` is not the field mean; `percentage` is min-max scaled within the race; ranks use unrounded values. Non-finishers can carry junk finishing speed % (6–27). | Kept as published; rank check flags only strict reversals; `metric_out_of_range` flags junk. |
| Over jumps about 28% of races have a winner's sectional total 0.1–3 s from the official time (median 0.7 s; flat 0.5% of races), most at Worcester, Uttoxeter, Fontwell, Newton Abbot, Southwell and Chepstow, although each runner's sectors add up. About 4% of mismatches exceed 5 s, where one time is wrong; some official times are impossible (10f in 5m 22s at Fairyhouse 2026-05-28; a 3m 3½f chase in 4m 35s at Market Rasen 2026-10-25). | `winner_time_mismatch` graded: info (0.1–1 s), warn (1–5 s), error (> 5 s). `win_speed_mph` and `win_time_implausible` (error) flag official times implying under 18 or over 45 mph. |
| A whole meeting's sectionals can be unusable: Chelmsford 2024-04-13 has the right runners but splits that do not fit the race (10 sectors for 8f, 5 for 14f). | `sector_count_vs_distance` (error). |
| Racing TV lists purebred Arab races (14–22 a year); Sporting Life does not. Plain "Arabian" is not enough: sponsors use it. | `is_arab_race`; excluded from validation's unmatched counts. |
| Weights published as `0-0` (139 runners). Flat weights below 8-0 and jumps below 10-0 are long-handicap figures, as above. | `0-0` parsed as missing (`weight_placeholder`, info); `weight_long_handicap` (info). |
| Ranks are not refreshed when values are revised (e.g. two runners ranked 9th, 102.05 and 96.20). | `metric_rank_inconsistent` (warn); use the values, not the ranks. |
| Abandoned races: 1,976 in the backfill (runners left `entered`, no result), 120 of them with no runners listed at all, mostly in Oct–Nov 2023. | `is_abandoned`; they count as complete without odds (there are no runners to price). |
| The first sector of an odd-yards race is a part-furlong (2m 5f 122y starts with 122 yards). | `sector_length_yards`, `sector_speed_mph`. |
| Sporting Life files NH flat races (here `Bumper`) as `Flat`. | Reported by validation (`n_bumpers_filed_as_flat`). |
