# Market performance implementation

**Market-distilled Timeform text candidate, 9 November 2026.** The explicit
text expert now calibrates with earlier-only fits and masks an incomplete
text field. The legacy calibration fit also starts cold so previous-quarter
weights cannot carry labels from its calibration period. Of 50 screened
recipes, the original word layout with historical market targets and 3e-6
regularisation won on five earlier quarters. It reduced later race log loss
from 1.92807 to 1.90889 on 11,893 races, improving all four quarters
(paired change −0.01918 [−0.02295, −0.01503]). Forecasting requires no prices.

The ensemble probe found no established incremental gain over the saved
generalist/performance pool (−0.00010 [−0.00149, +0.00146], 1,142 races).
A separate verified text candidate was fitted through 6 November; the
installed store and selected win model were left as they were. See the
[text development report](research/TEXT_MODEL_IMPROVEMENTS_2026-11-08.md)
for the saved recipe, model, calibration and adoption decision.

**Text inputs repaired, 8 November 2026 (code only; scores, store and model not
rebuilt).** An audit of the text inputs found no leak in the Timeform comment
score. Its quarter models stop 7–10 days before their quarter. All 68 races
captured before the off (1,190 runners) match the stored history text. It did
find four faults, fixed in code:

- **Comment-score scale drifted.** Early quarter models were over-confident:
  the best rescaling of 2021Q1's scores was 0.54, against about 1.0 from 2023.
  Each model now fits its own scale on its latest training quarter (format 2,
  `text_models/format_2`). On six quarters refitted with the new code, the race
  log loss of the text score alone:

  | Quarter | Before | After |
  |---|---|---|
  | 2021Q1 | 2.1303 | 2.0344 |
  | 2021Q2 | 2.1140 | 2.0472 |
  | 2021Q3 | 2.0353 | 2.0048 |
  | 2021Q4 | 2.0148 | 2.0169 (the only quarter that got worse) |
  | 2022Q2 | 1.9591 | 1.9507 |
  | 2023Q1 | 1.9092 | 1.9088 |

- **Missing text scored as a constant.** 1,926 runners with no comment and no
  verdict scored a constant instead of missing. A runner with no comment that
  the verdict does not name now scores NaN, and the verdict order is missing
  when a race has no verdict.
- **ATR "front" tag mostly flagged winners.** It matched "led" anywhere in the
  comment, so it fired on 82% of winners against 28% of horses finishing 4th or
  worse. It now reads only the first position clause: 57% against 36%, and its
  correlation with winning fell from 0.33 to 0.12.
- **ATR "eased" tag counted closers.** "Never nearer" made up 38% of eased
  flags; it is no longer counted as eased.

The effect on the win model is not measured. To apply the fixes, run an RTV
refresh, which refits all 24 quarter models and rescores the history, then
publish, rebuild the store and retrain.

**Store build speed, 6 November 2026 (no feature or model change).** A full build
went from 5,521 to 845 seconds on the same frozen 757,580 rows, 6.5 times faster,
with no feature removed. `featured.parquet` and `processed.parquet` are
byte-identical to the old build and the carried state has no differences. Nothing
was trained or installed. The changed `fe_signature` makes the next training run
rebuild once. See the [report](operations/STORE_BUILD_SPEED_2026-11-06.md).

**057 installed, 4 November 2026.** The exploded softmax now drives the
production win learner, with the same 129 inputs as 056.

| Check | Result |
|---|---|
| Evaluated run | `20261104_212157_439212`, 3,758 win trees |
| Promoted run | `20261104_235300_444862`, fitted through 3 November, 3,325 win trees |
| 057 vs 056, 3,268 hold-out races | 1.8512 vs 1.8557 |
| Change [95% CI, clustered by date] | −0.0046 [−0.0097, +0.0003] |
| Walk-forward result | −0.0046 [−0.0064, −0.0026], 4 of 4 quarters |
| SP market on the hold-out races | 1.7722 |

- **Reading:** the hold-out agrees with the walk-forward. Its interval touches
  zero, so it is weak evidence alone; all three months improve.
- **Comparison window:** a source refresh moved the 057 hold-out to
  6 August–3 November, so only races both runs scored are compared, 6 August–
  29 October. 350 races scored by only one run are left out. 057 also trained
  on five more days, which should matter very little.
- **Store:** 755,309 rows and 133 columns match the 056 store exactly, with
  1,774 new rows through 3 November.
- **Serving:** 102 settled races pass the mechanical checks. Live cards are
  blocked until the store is rebuilt for the runner-identity code, and the
  model must then be refitted. See `docs/archive/CONTINUATION_before_cleanup_2026-11-05.md`.

Receipts are in `data/audit/install_057/`.

**057–060 model ideas, 2 November 2026 (research; nothing installed).**
Four ideas from [MODEL_IDEAS_BRIEF.md](research/MODEL_IDEAS_BRIEF.md) were tested; idea
5, a language model reading the comments, was not run.

- **Folds and inputs:** purged quarterly walk-forward on the 056 store, with
  the 129 production inputs and the production win recipe.
- **Choices:** made on 2024Q1–2025Q2 and registered before confirmation.
- **Confirmation:** 2025Q3–2026Q2 at native capacity, 11,905 races.
- **Market:** 1.7448; production scores 1.83135.

The full write-up is [MODEL_IDEAS_057_060.md](research/MODEL_IDEAS_057_060.md).

| Idea | Race LL change vs production [95% CI] | Quarters better | Verdict |
|---|---|---|---|
| 057 race softmax, winner only | −0.00216 [−0.00385, −0.00034] | 4 of 4 | Confirmed |
| **057 exploded softmax, stage weights (1, 0.25, 0.0625)** | **−0.00459 [−0.00635, −0.00261]** | **4 of 4** | **Confirmed** |
| 058 figure and spread model, alone | +0.00480 [+0.00084, +0.00903] | 1 of 4 | Worse |
| 058 blended with production (out of sample, 2025Q4–2026Q2) | −0.00417 [−0.00603, −0.00230] | 3 of 3 | Adds information |
| 058 blended with the 057 model, vs production (measured afterwards) | −0.00651 [−0.00923, −0.00374] | 3 of 3 | Strong lead |
| 059 luck-adjusted training weights | −0.00029 [−0.00112, +0.00058] | 3 of 4 | Null |
| 060 learned trainer, jockey and sire profiles | +0.00011 [−0.00092, +0.00113] | 1 of 4 | Null, priced |

**057, exploded softmax.** It learns from the first three places, not just
the winner, and it is the clear gain.

- Half of the gain comes from scoring the race as one choice among the
  runners; the other half from second and third place.
- It closes 5% of the gap to the market.
- Training takes 2–3 times as long, because the gradient is computed in a
  numba kernel (`src/race_objectives.py`).

**058, figure and spread.** It predicts pounds behind the winner plus a
spread (`src/thurstone.py`).

- It carries information the boosted classifiers lack. Blending two near
  copies of one model gains only 0.0002–0.0006.
- It does not fix the 055 steamer and drifter bias.

**061, a follow-up to 057.** It tested finishing order below third place, on
the development quarters. Every finisher at weight 0.25 per place ties with
the top three (−0.00002). Flattening later stages instead of down-weighting
them is worse at every strength, from +0.005 to +0.037, because it makes the
model over-confident. 057's three places stand.

**062, CatBoost pairwise, 4 November 2026 (ties 057; not adopted).** CatBoost's
pairwise loss, on pairs built from 057's three stages:

| Arm | Compared with | Race LL change | Quarters better |
|---|---|---|---|
| Results only, fast settings | Production | +0.01693 [+0.01344, +0.02058] | 0 of 4 |
| Soft target, learning rate 0.05 | Production | −0.00365 [−0.00644, −0.00111] | 4 of 4 |
| Soft target, learning rate 0.05 | 057 | +0.00094 [−0.00140, +0.00310] | 1 of 4 |
| Soft target blended with 057 | 057 | −0.00163 [−0.00292, −0.00031] | 3 of 3 |
| Soft target added to 057 + figure | 057 + figure | −0.00079 [−0.00169, +0.00008] | 3 of 3 |
| Soft target added to 057 + figure | Production | −0.00730 [−0.00995, −0.00444] | 3 of 3 |

The first three rows use 11,905 races and the blends 8,370. The soft-target arm
ties 057. Blended with 057 it gains about as much as the figure model, but the
two overlap, so as a third model it adds little. It was chosen after the fast
arm had been scored on these quarters, so it is a lead, not a registered result.
At about 40 minutes a fit and a second framework, CatBoost stays out of
production and outside the project's locked environment.

**063, feature use and meta learners, 5 November 2026 (confirmed; not
installed).** Full tables are in
[FEATURES_AND_STACKING_063.md](research/FEATURES_AND_STACKING_063.md). Both candidates and
the pass rule were registered first, then measured at native capacity on
2025Q3–2026Q2, 11,905 races. Market 1.74482; 057 scores 1.82675.

| Arm | Race LL change against 057 | Quarters better | Result |
|---|---|---|---|
| 8 rating-gap inputs + market share 0.75 | −0.00346 [−0.00506, −0.00185] | 4 of 4 | Pass |
| **64 extra inputs + market share 0.75** | **−0.00628 [−0.00823, −0.00426]** | **4 of 4** | **Pass, chosen** |

The 64 inputs beat the 8 by −0.00282 [−0.00413, −0.00156] in all four quarters,
so the registered rule keeps them. The screen had estimated −0.00629. Neither
model adds measurably to the market in a blend. Race context, finer race types
and meta learners gave no measurable gain on the screen, and sub-models add only
−0.0002 to the improved model. The 64 inputs exist only in the research script,
and the golden hold-out has not been read.

**Next step:** install the exploded objective in the production win learner.
Receipts: `data/research/model_ideas_057/`.

**056 Timeform comment score installed, 2 November 2026 (installed; the
hold-out does not show the gain):** the 052 comment model is now a feature
stage, [src/rtv_text.py](../src/rtv_text.py), registered in
`_FE_SOURCE_FILES`. It works as follows:
- Each quarter from 2021Q1 has its own text model. It is fitted only on races
  up to 7 days before that quarter starts, and it scores that quarter's
  runners, so history and live cards are scored the same way.
- History scores cover 712,976 runners (`data/rtv_cache/feature_text_scores.parquet`).
- Live cards are scored from the Racing TV pre-race snapshot taken up to 5
  minutes before the off. Withdrawn runners are left out of the verdict order
  and the race softmax.

It adds three inputs: `rtv_text_logit`, `rtv_text_prob` and
`rtv_verdict_rank`. With them, `feature_sets/win_selected_056.txt` has 129
inputs. The store was rebuilt (FE `69babeb9c4262ee2`, data `826c9478e6f726b8`,
755,309 rows to 2026-10-29).

The evaluated run `20261102_130936_036383` holds out the same 90 days as 054
(2026-08-01..10-29). On the same races, the comment score makes no measurable
difference:

| Race log loss, 3,431 races | Value |
|---|---|
| SP market | 1.7656 |
| 126 inputs (054, `20261102_050631_314757`) | 1.8492 |
| 129 inputs, with the comment score | 1.8491 |
| Difference, with − without | −0.0001 [−0.0045, +0.0045] |

By month the difference is −0.0015 (August), +0.0047 (September) and −0.0035
(October).

The interval contains 052's walk-forward gain of −0.0029, so three months
cannot confirm or rule out a gain of that size. The model does use the score:
`rtv_text_prob` is its largest input by gain (34% of the total). The text
inputs are present on every hold-out row, and the store values match the
text cache exactly. Receipt: `data/audit/install_056/holdout_comparison.json`.

The deployment refit `20261102_143544_663385` is fitted through 2026-10-29
with 3,577 win / 2,998 place trees, and it is promoted. `project_status.py`
is ready. `verify_live_upgrade.py` passes:
- 106 recent fields and 1,010 runners;
- single and batch forecasts agree exactly;
- on today's Southwell 15:15 card, all 8 runners carry the comment score.

Receipts are in `data/audit/install_056/serving/`.

Operational note: live comment scores need that day's Racing TV snapshot.
The picks app captures it through `source_pipeline.get_scraped_racecards`.
`verify_live_upgrade.py` calls the Sporting Life scraper directly, so run
`source_pipeline.capture_rtv_snapshot()` first. Without a snapshot, the three
inputs are missing for that card.

**055 review of the 054 model's worst misses, 2 November 2026
(diagnosis):** we re-read 73 of the worst hold-out races (818 runners) from
raw Sporting Life pages and found no value errors. Off-DB form (abroad and
point-to-point) carries 2.7% of the gap. About 46% of the gap to BSP arrives
after 11:00 as market money: drifters are over-rated 1.75× and steamers
under-rated 0.75×. The rest is morning-priced information:
- debutant reputation;
- yard changes;
- runners where the Timeform comment defers to the market, which carry 13.5%
  of the gap.

Side defect: the IHRB stewards events are mis-keyed (91% of keys are jockey
initials). See [MODEL_ERROR_REVIEW_055.md](research/MODEL_ERROR_REVIEW_055.md).

Follow-up the same day: the model-side fixes are all within noise when fitted
on earlier quarters and scored on the next. Race LL changes:
- debutant shrink: −0.00021;
- trainer/sire debut record: +0.00029;
- off-DB last-run position: +0.00011;
- discipline carry-over: +0.00033.

See [ledger 055](NEGATIVE_RESULTS.md). The controllable gain that remains is
the 052 comment score, which 056 is installing.

**054 installation of the 048 inputs and 051 fixes, 2 November 2026
(installed):** the store was rebuilt (FE `32e71a2cdff8a8b5`, data
`826c9478e6f726b8`, 755,309 rows to 2026-10-29). Before that rebuild,
`data/raw/race_results.csv` was re-exported from the database. The 051 repair
had changed only the DB, so the snapshot still held the 48 deleted rows and
the old results of the 21 repaired races. `repair_overfull_fields.py` now
re-exports the snapshot itself.

The evaluated run `20261102_050631_314757` uses the 126 inputs and holds out
the latest 90 days (2026-08-01..10-29, 3,431 races). It scores race log loss
1.8492, against 1.7656 for the SP market (gap 0.0836). On the same window it
also scores:

| | Win Brier | Win ECE |
|---|---|---|
| Model | 0.08357 | 0.0035 |
| Market | 0.08115 | 0.0053 |

Place Brier is 0.170 and place ECE 0.012: about 1.4 pp over in the lowest bin
and 3–4 pp under in 0.5–0.7. No 154-input run was evaluated on this window, so
048's walk-forward (−0.0194, 4/4 quarters) remains the comparison.

The deployment refit `20261102_063010_291745` is fitted through 2026-10-29
with 3,980 win / 2,999 place trees. It is promoted. `project_status.py` is
ready, and `verify_live_upgrade.py` passes: 106 recent fields, single = batch,
and today's card. Receipts are in `data/audit/install_054/serving/`. The 052
comment score is not part of it, because it has no FE stage.

**053 the rest of the Racing TV card, 2 November 2026 (no further gain):**
badges, pundit tips, the draw comment, owner history, pace notes and the last
three close-ups add nothing beyond the 052 comment score (−0.0028 with all of
them against −0.0029 with the comments alone). Card data on its own gives
−0.0011 [−0.0024, +0.0002]. Two card fields change after the off and were
left out: the race pace forecast and smart stats.
[Ledger entry 053](NEGATIVE_RESULTS.md#053-unused-racing-tv-card-data--nothing-beyond-the-timeform-comment-2-november-2026).

**052 Timeform comments as an input, 1 November 2026 (confirmed, small, not
yet in the store):** `scripts/text_model.py` fits a race-wise softmax
regression on word and word-pair TF-IDF from:

- each runner's Timeform comment;
- the verdict sentences that name the runner;
- the order in which the verdict names it.

It is refitted every quarter on earlier quarters only (purged), so every row
gets an out-of-fold score. The penalty, 3e-5, was chosen on the 048 selection
quarters and is interior to the grid. The score, the comment-only win chance
and the verdict order then become three extra inputs to the 126-input model.

Results on 2025Q3–2026Q2 (11,901 races):

- **Race log loss improves by 0.0029 [0.0010, 0.0049]**, in 4 of 4
  quarters (−0.0034, −0.0017, −0.0029, −0.0036).
- The gap to the market narrows from 0.0894 to 0.0866.
- The market+model blend still gains nothing (+0.0000).

Comments alone score 1.9300, against 1.8342 for the model and 1.7448 for
the market. The weights read as form judgements:

| Direction | Words and weights |
|---|---|
| Up | named first in the verdict +5.4, "11/8" last time +1.5, "unbeaten" +1.3 |
| Down | "100/1" last time −2.7, "last of" −2.4, "well held" −1.9 |

Leak checks:

- Comments alone are well short of the market.
- Their gain over a uniform guess is steady from 2022 to 2026 (−0.19 to
  −0.21).
- On 2023 runners a comment shares more words with the horse's *previous*
  close-up than with this race's (Jaccard 0.0122 against 0.0084). "Made all"
  appears in 0.04% of comments and in 1.4% of close-ups.
- On 2026-10-30 the comment and verdict were identical the night before and
  after the off.

**051 data fixes and text sources, 1 November 2026 (fixed in code and data,
store not rebuilt):** this follows up the two defects and the missing text
found in 050.

- **Overfull fields repaired.** `scripts/repair_overfull_fields.py` removed
  48 non-runner or withdrawn rows from 21 races. 20 fields were replaced from
  a validated re-scrape. One Gowran Park field has two horses with the same
  name, so only its NONRUNNER rows were deleted. No overfull field remains,
  and the affected races are backed up in
  `data/backups/overfull_fields_20261101_212414.csv`.
- **One name per jockey and trainer.**
  `scripts/build_connection_aliases.py` links Racing TV person IDs to
  753,392 stored rides. It writes `reference/connection_aliases.json`, which
  maps 94 jockey names (13,171 rides) and 15 trainer names (2,580 rides) to
  each person's latest Sporting Life name. `data_processor.clean_data`
  applies it to history and to live cards alike, and the file is registered
  in `_FE_SOURCE_FILES`.
  - RTV IDs are not stable: they were re-issued in January 2024, some riders
    had temporary IDs in late 2025, and some IDs cover two people.
  - So names whose IDs are used at the same time (35 jockeys, 9 trainers) are
    never remapped.
  - A group is skipped unless all its names share a surname and compatible
    initials. This keeps "B J Murphy" and "B T Murphy" apart (two trainers
    under one RTV ID), along with five RTV mislinks.
  - Joint licences are merged with the solo licence of the same yard.
- **Commentary is captured going forward.** The scraper keeps each
  runner's `commentary` and the race `verdict` (as plain text). Both go to a
  new append-only `runner_text_observations` table: the morning racecard copy
  (`phase='racecard'`) and the result-page copy (`phase='result'`), each
  with the time it was seen. Racecard text is kept out of the cached card,
  and nothing reaches `results`.
- **Racing TV already holds the same text since September 2020.** On the
  619 rides checked in 050, Sporting Life's commentary *is* the Timeform
  analyst comment on the RTV card: identical apart from spacing, with a 10th
  percentile similarity of 1.00. The verdicts match too; SL only appends the
  selections. Coverage on linked SL runners (99.6% link) is:

  | Text | Coverage |
  |---|---|
  | Timeform comment | 99.9% |
  | Timeform verdict | ~100% |
  | Close-up comment (post-race) | 99.6% |
  | Pace hint | 98% from 2024 |
  | Draw comment | 22–60% |
  | Trainer badges ("in form", "break return", "headgear first", …) | 17–22% |
  | RTV race tips | 82,541 races |
  | Smart stats | 66,968 races |

  None of these is an input; the store uses only the numeric Timeform card.
  `source_pipeline.load_matched_text` returns the text per SL runner, so an
  80,000-page Sporting Life backfill is not needed.
- **Integrity so far:** the RTV comment and verdict were identical in the
  pre-race snapshot and after the race on 2026-10-30. The new table will show
  whether result pages ever rewrite the SL copy.

**050 why the model trails the market, 1 November 2026 (diagnosis):** this
uses the 049 out-of-sample predictions on the 126 inputs, 2025Q3–2026Q2,
11,901 races, where the gap to the market is 0.0894.

- **Calibration is fine.** A power recalibration fitted on earlier quarters
  moves race LL by under ±0.0006, and the reliability bins match. The model
  is less *sharp* than the market: its mean top probability is 0.31 against
  0.35.
- **It is not under-using its inputs.** A LightGBM fitted to
  log(market/model) on the same 126 inputs has out-of-fold R² 0.11.
  Applying that fit closes at most 1.0% of the gap.
- **Where the gap is:**
  - races won by the market favourite, which the model prices 0.33 against
    0.41, carry 107% of it; the model beats the market on outsiders;
  - runs outside our database carry 4.7% (point-to-point and French form,
    detected because the source's `days_since_last_run` is shorter than our
    gap); filling them in recovers at most 0.0023;
  - true debutants carry 3.6%.
- **Data checked against raw Sporting Life pages:** 59 races and 619 rides,
  read independently of the scraper. These were the 25 worst misses, 25
  random races and 9 underrated favourites. Finish, SP, weight, draw, OR,
  going, distance, class, form string and days since last run all agree.
  Two real defects were found, both small:
  - in 21 of 80,302 races, 47 non-runner or withdrawn rows are stored as
    runners (Yarmouth 2026-05-13 has 6);
  - jockeys are keyed by display name, and Sporting Life renames riders
    ("M P Sheehy" and "Mikey Sheehy" are both used through 2020–26), so
    about 1% of rides sit under a stale name.
- **What the market knows:** the raw racecard commentary on the misses names
  point-to-point and French form, purchase price, gelding, a new yard, trip
  or handicap switches and gambles. Our inputs see none of this.

**049 race-type and course specialists, 1 November 2026 (refuted):** on the
048 inputs, a hurdle-only model is worse than the general model on the same
2,820 hurdle races by +0.0056 [−0.0001, +0.0110]. A Wolverhampton-only model
is worse on the same 629 races by +0.0100 [−0.0057, +0.0280]. Each is better
in only 1 of 4 quarters.

The general model trails the market by 0.089 overall, 0.103 on hurdles and
0.097 at Wolverhampton. An out-of-sample market+model blend gains nothing
over the market in any of these.
[Ledger entry 049](NEGATIVE_RESULTS.md#049-race-type-and-course-specialist-win-models--worse-than-the-general-model-1-november-2026).

**048 walk-forward selection of the win model's inputs, 1 November 2026
(confirmed, not promoted):** `win.txt` was the top 150 by gain from one
pilot fit on a store that still had three leaks, then edited by hand for four
months. `scripts/select_features.py` re-derives the list by the model's own
out-of-sample race log loss. It runs recursive SHAP elimination over six
purged quarters and stops by the one-SE rule. The result is 126 inputs, 59 of
them new: the Timeform card, margin-Elo, RTV and ATR history, and relative
speed figures. On four later quarters that took no part in selection
(11,901 races), race log loss improves over `win.txt` by **0.0208
[0.0164, 0.0251]** at screen capacity and by **0.0194 [0.0155, 0.0236]** at
production capacity. It improves in 4 of 4 quarters each time, and Brier
improves too. Using all 1,127
inputs is as good (selected vs all −0.0008 [−0.0031, +0.0014]), so the gain
is in the families let in, not in the pruning. The store is rebuilt
(`bde9317da2c3c963`, 047 fixes included). The new list passes the
synthetic, ATR, RTV and current-price leak checks. Not an edge over the
market: the model still trails it by 0.095 nats.
[Protocol, results and checks](research/FEATURE_SELECTION_048.md).

**047 implementation review of the 154 inputs, 1 November 2026 (fixed in
code, store not rebuilt):** nine encoding faults were confirmed on the store,
and ten repair sets were screened on four development quarters. None changes
race log loss beyond ±0.0013 nats. At the user's request all nine are now
fixed in `src/feature_engineer.py`. `feature_sets/win.txt` drops the five
duplicates and is down to 149 inputs. The FE signature has changed, so the
store and the promoted model stay stale until the next rebuild and retrain.
[Faults, results and fixes](research/FEATURE_IMPLEMENTATION_REVIEW_047.md).

**046 RTV-derived inputs and Timeform card, 31 October 2026 (built, not
promoted):** The RTV feature stage now builds three families:

- the Timeform card for the current race;
- sectional time figures from prior runs;
- within-race stride and jumping measures from prior runs.

The store is rebuilt (`1c83cfaeac51bb2e`, which includes the 045 ratings).
Native win fits compare `feature_sets/win.txt` with
`feature_sets/win_rtv_derived.txt`, which adds 19 inputs. On 5,134
development races (2026Q1–Q2), race log loss improves by 0.0177
[−0.0229, −0.0129]. Both quarters improve, as do Brier, ECE and top-1, and
the features close 16% of the gap to the SP. This is not evidence of an edge
over the market. Production is stale until a retrain.
[Build, coverage and results](research/RTV_DERIVED_046.md).

**045 rating systems repaired, 30 October 2026 (not yet in the store):** Elo,
Glicko, TrueSkill and value-added jockey/trainer ratings were rebuilt and
re-tuned. Standalone held-out race log loss improves sharply (Elo
2.125 → 2.010, Elo + jockey + trainer 2.030 → 1.952), but no rating, old or
new, moves SP log loss by more than 0.00015 nats. The store needs one rebuild
and the promoted model a retrain before serving.
[Changes, settings and results](research/RATING_SYSTEMS_045.md).

**042 installed,30 October2026:** production `20261030_015917_823047` uses
corrected official results and freshly rebuilt horse/connection/rating history.
The canonical repair covers164 complete fields,131 changed winners and463
placings. One native current win/place refit through28October is complete;
154 inputs,3,992 win /2,987 place trees. Live history extends through29October:
755,229 runners /80,288 races. Daily ECB updates are restored, with73 post-fit
Irish purse comparisons corrected and no additional model fit. All fitted
inputs remain unchanged by the FX extension.

Initial staged/installed serving inputs and probabilities match exactly; the
latest post-FX check passes106 complete fields /1,010 runners plus the upcoming
six-runner Newmarket card. Final post-restart deep readiness passed08:57UTC
and installation closure completed08:59UTC. The fresh
study is eligible31October. These are correctness/operational findings, not a
new held-out performance estimate. [042 evidence](research/AMENDED_RESULTS_042.md).

**039–041 closed:** connection residual form, own running cadence and raw
place-head information did not earn win-model changes. Do not repeat their
fixed screens or silently relabel their preserved pre042 results.
[039–040](research/ESTABLISHED_CONTEXT_039_040.md), [041](research/AUXILIARY_PLACE_WIN_041.md).

**034–037 completed,29 October2026:** normal model033 remains installed.
Transferable price history034 improved overall native Q1 log loss
1.798224→1.795672, but failed target Brier and the predeclared precision gate.
Market-relative finishing context035 failed its small established-handicap
screen. Simple market-assisted036 selected the unchanged market before its
later test; the fitted combination worsened both probability scores. No extra
tree fits were needed for036. These are bounded negative/unresolved findings,
not a model update. [034–035](research/MARKET_HISTORY_REVIEW_034_035.md),
[036](research/MARKET_ASSISTED_REFRESH_036.md).

Source repair037 is unconditional in the ordinary collector: structured
nonfinish reasons are now preserved. The48-source replay recovers37 reasons
with every other parsed result and all card values unchanged;205 relevant
tests pass. Existing generic historical DNFs are not guessed. No prediction
gain is claimed from preserving information. Model weights and FE fingerprint
stay033; replacement source-pinned study105914Z is eligible30October, with no
observations or scheduled recorder. [037 evidence](research/NONFINISH_SOURCE_037.md).

Entries below retain the dated history of earlier stages.

**033 installed and checked,29 October2026.** Production
`20261029_085433_011356` uses154 normal inputs,3,992win/2,991place trees and
parameters through27October. Daily history was then advanced to28October:
754,842 runners, without another fit. Deep readiness and native live checks
pass on115 recent fields/1,108runners and today's12-runner Bath card.

Three dated prior-market/opposition inputs passed one native win check
alongside52 mandatory source corrections. Q1development log loss improves
1.802287->1.798224 and Brier.09003383->.08995795 overall; for993 established
handicaps,1.912089->1.905518 and.09773430->.09756486. On2,469 identical market
fields,1.789123->1.784530 versus market1.686613 closes about4.48% of the old
gap. Place Brier.17461220->.17425539. Historical win fits were reused with
exact replay. These are inspected development results, not fresh confirmation.
The market still leads and current full-data weights have no extra held-out score.

At033 completion, pinned study091505Z was eligible30October with no observations/scheduler;
037 now supersedes its current-source eligibility using identical model weights.
The normal-pipeline source/state, checks, current run and limitations are in
[033 results](research/PRIOR_MARKET_HISTORY_033.md) and its final_status.json receipt.
Cross-surface/discipline price transfer was subsequently tested in034. Earlier installation/readiness
statements below describe their dated historical stages.

**032 source repair completed with033 (historical preparation):** source-verified52 additional historical
race-type corrections (430 runner rows) are now in ordinary preprocessing.
The parser also passes121 checks and agrees with19180 source-labelled fields.
The combined store/model rebuild is complete in033; readiness passes for
the aligned generation. See [the audit and training state](research/RACE_TYPE_AND_TRIP_REVIEW_032.md)
and docs/archive/CONTINUATION_before_cleanup_2026-11-05.md. The three extra trip inputs remain unselected.

**031 complete:** one new capped fit, with baseline reused. Approximate
weight/opposition performance features worsen established-handicap scores;
no full refit or installation for that proposal. [Results](research/HANDICAP_PERFORMANCE_031.md).

**Completed 030: faster handicap investigation.** Reused saved029 forecasts,
validated 48 complete source fields, and audited weight/mark interpretation.
The only new training was one matched pair of 400-tree win models, taking
107.6 seconds. Removing two near-duplicate momentum inputs worsened overall
and established-handicap probability scores, so no full refit followed.
Model029 remains installed and verified. Rider claims are commonly omitted
source metadata, but the sample does not establish them as the main cause of
the market gap. See [the findings and lighter workflow](research/HANDICAP_WEIGHT_INVESTIGATION_030.md).

**Completed029,28 October2026:** corrected win/place model
`20261028_212411_292221` is installed through27 October with151 selected inputs.
All confirmed Glicko, comparable-mark and nine source-rating repairs are in the
normal pipeline.84 relevant checks, deep readiness and native live checks pass.
The extra169-input family failed: full Q1 log loss1.802287 baseline versus
1.805041 candidate, with worse Brier too. It is not selected. On2469 identical
market fields, corrected baseline log loss is1.789123 versus market1.686613.
No predictive gain or parity is claimed. Familiar handicaps account for95.55%
of the established-field net gap. See
[the complete results and installed update](research/ESTABLISHED_FORM_IMPROVEMENTS_029.md).

**Previous established-horse audit028 (superseded by029).** The saved out-of-sample forecasts
still trail the market in familiar fields. The audit found and repaired
double-counting of inactivity in the normal Glicko calculations. The existing
store and027 weights have not yet been rebuilt; live feature generation blocks
the changed-code mismatch until the combined rebuild/refit.45 distinct relevant
checks pass. This is a verified input repair, without a new predictive-gain
claim. See [the findings and case review](research/ESTABLISHED_HORSE_ERRORS_028.md).
The readiness statements below describe the completed027 generation before028.

**Latest, 28 October 2026:** current model `20261028_142429_576384` is installed
and ready. Verified corrections are now unconditional in
the normal CLI/app training pipeline
([026](research/CANONICAL_FEATURE_REPAIRS_026.md)). All 754,497 historical runners and
80,212 races were rebuilt without losing identities; 40 of 151 selected
inputs changed. Both win and place components were freshly trained on the
corrected inputs through 27 October and installed as
`20261028_134020_942529`. Deep readiness and native artifact checks passed.
Earlier statements about retaining the old feature definitions are superseded.

On the same 2,591 Q1 development races, the correction batch changes race log
loss 1.803081 → 1.802182, Brier 0.090093 → 0.090014, and winner accuracy
30.30% → 30.75%. The small gain is uncertain; the paired log-loss interval
crosses zero. Place Brier rounds to 0.1746 for both. The market still leads.
Verified errors stay repaired regardless of that diagnostic score.

[027](research/BSP_TEACHER_REFRESH_027.md) fills an additional data gap: the standard
exchange-price teaching cache stopped in June despite newer local/public
observations. The audited normal cache now extends through 27 October, with
complete training-price fields rising from 89.5% to 99.8%. The completed
win-only update is installed as `20261028_142429_576384`, selecting 3,712
trees and reusing the freshly corrected026 place component. Exact native
tests and runtime guards verify unchanged feature data and hard place labels.
There is no additional held-out performance claim for that source extension.
Deep readiness has no blockers; native saved-artifact and actual public-card
checks pass. The 14-runner Dundalk card is a real operational receipt.
Probability checks cover 100 recent fields/955 runners; single/batch error
is zero on five checked fields. The fresh pinned
forward setup is eligible from 29 October; no pairs or scheduled collector
are claimed. Both026 and027 evidence are sealed and recoverable.

## Earlier completed stages

022: **increased win-model capacity passes both full validation periods**
([results](research/MODEL_CAPACITY_021_022.md)). Across5,824 races/52,905 runners,
log loss1.828701 ->1.823450, Brier0.08748263 ->0.08731258, top-pick accuracy
30.31% ->30.67%. Each quarter improves, and paired race/date log-loss
intervals exclude0. On5,594 fields with the historical market reference,
the change closes4.82% of the remaining gap; the market still leads.
This follows one positive cheap capacity screen021, with no parameter grid.
023 qualification confirms the stronger win model preserves place scores.
Its final fit through2026-10-25 selected3,720 trees on667,110 runners.
Run20261028_084608_624611 was installed at08:46UTC on2026-10-28, retaining
the verified place component; both saved probability outputs replay exactly.
The earlier016 installation remains recoverable. No further fits are queued.

The live-input parity repairs now pass50 checks: one-category cards retain
training indicator meanings, genuinely missing measurements remain missing,
and preparation changes invalidate cached vectors. A separate source audit
found16 incorrectly labelled races in19,118 matched public-history fields.
The parser corrects all16 with no new mismatch; all158 final integration
checks pass ([audit](research/RACE_TYPE_AUDIT_025.md)). It is active in result and card
ingestion, with an unchanged training feature generation.
Historical label changes require regenerated histories, not indicator edits.
Withdrawal handling also now preserves the actual field and retains started
non-finishers. This recovered three rejected races/30 runners. The completed
history refresh adds763 runners through2026-10-27 without another fit. Deep
readiness passes with no blockers; the public live path passed on an actual
eight-runner upcoming Nottingham card, with exact native single/batch replay
on the checked recent fields. The updated source-pinned forward setup starts
2026-10-29; no prospective pairs or scheduled collection are claimed.

024 [remaining-gap review](research/REMAINING_MARKET_GAP_024.md) reuses frozen forecasts
without another model fit. Sparse-history fields have a larger gap per race,
but fields without a locally unseen horse account for72.5% of the remaining
aggregate gap. Overseas-history improvements are necessary but insufficient.

020: **prize-currency audit and one matched screen complete**
([review](research/PRIZE_CURRENCY_020.md)). Source verification covers258 races/all86
venues; historical ECB rates correct the three selected monetary predictors.
Their original full-history replay is exact, and12 targeted checks pass.
On3,233 test races, log loss1.849008 ->1.850940 and Brier0.08548189 ->0.08551467:
no improvement or production change. No wider currency validation is queued.

018: **feature audit and four matched screens complete**
([review](research/FEATURE_REVIEW_018.md)). The repaired history passed151-input checks
across753,734 rows; exact averages match the independent full-history oracle.
On3,233 later races, original/repairs/coverage/public log loss is
1.849008/1.851337/1.850816/1.849820. None improves both primary scores against
the matched original baseline. No model switch or further quarterly fits.
Live-card pedigree now reaches new arrivals before their first local result,
with identity checks and104 passing ingestion checks. The installed016 model
and matching feature generation remain unchanged. A separate019 check completed
[training-only calibration choice](research/MODEL_CALIBRATION_CHOICE_019.md): temperature
reduces repaired log loss to1.849218 and Brier to0.08546720, still without
demonstrating an improvement over the original baseline. These are
retrospective development results.

016: **completed and installed** ([results](research/MODEL_HISTORY_TEACHER_015_016.md)).
Three chronological test periods/9,356races: LL1.87204489 ->1.85749152,
Brier.08754346 ->.08714276, accuracy29.03% ->29.87%. Race/date confidence
intervals exclude0; matched market gap closes11.73% in point estimate. Market
remains ahead. The151-input 50/50market-taught model forecasts without live odds.
Full win/place refit through2026-10-25 saved as20261027_165656_616818 and installed
with matching corrected histories. Installation failed after the saved fit due
to a ZIP-handle/recovery-path variable collision; source capture is now separately
scoped, recovery tests pass, and installation resumed WITHOUTretraining. Previous
model/store/settings remain recoverable. New forward study starts2026-10-28.
Final deep readiness check passed at17:28:53UTC: database, feature store, training
records and production model all OK; model is servable with no blockers.

017: **feature audit and one-window surface screen complete**
([audit](research/FEATURE_REVIEW_017.md)). AWpredicate matched no real synthetic labels,
missing classes were used as ordinal0, and source includes impossible1–5lb
weights. Causal surface/class prototypes implemented; future unknown weight/
distance parsing fixed. AW153vsbest016151 Q3LL1.90496461 ->1.90470585 but interval
crosses0 andBrier worsens slightly; no upgrade. Userrequestedfasterexploration,
so laterquarterswere stopped afterfirstcheckpoint. Futurefeatureideas default
one-window cheaper matched screens, not unconditional3period nativefits.

014: **first qualified prediction improvement from repaired result margins**
([results](research/MODEL_MARGIN_REPAIR_014.md)). Completed all3 registered quarters on
9,356races: LL1.87204489 ->1.86780412, Brier .08754346 ->.08741422. All quarters
improve; pooled race/date95%intervals exclude0. On8,942matched fields this closes
3.47% of the original market gap. Severe low-probability errors remain. All30
sealed evidence digests verified. No production switch yet: the follow-up015
corrects a separate lost-non-finisher-history defect before adoption.

015/016: **rebuilt histories complete; native comparisons training**. Recovered
45,005 DNFs in declared complete settled fields (92 unresolved); corrected
141,607 later prior-start counts. The151-input hard-label model and one fixed
50/50 market-teacher model share the same corrected store. All3quarters and
same-field comparison with the best completed014 required; no subgroup selection.
Final quarter also validates place forecasts. Recipes/source/data frozen before
new fits. Public-profile2193horse cache and7-feature prototype remain research
only; neither candidate uses them. 930full checks and18new focused recipe checks
passed; current full-suite rerun in progress after training-order validation fix.

012–013: **history-family models and an earlier-fitted blend complete; no
qualified live switch** ([family](research/MODEL_HISTORY_012.md),
[blend](research/MODEL_HISTORY_BLEND_013.md)). Trained three165-input candidates,
reusing exact verified149-input baselines. One duplicate input removed before
any new forecast was scored; two baseline alias pairs recorded for cleanup.
On9,356races, family LL1.87204489→1.87133338 but the interval crosses0 and
winter worsens. Only0.38%of the matched market gap closes in point estimate.
External-form diagnostics motivate a separately frozen exploratory blend.
Coefficients learned on earlierQ3 and validatedQ4 give Q1LL
1.81799380→1.81767994 and April–May1.90437727→1.90248792, both uncertain.
The context lead uses inspected outcomes, so this is not fresh confirmation.
All66focused checks pass;17family+9blend evidence files, source/model/data
hashes and shuffled-field replays verify. All fits completed before the PC
shutdown; saved results recovered without retraining. Production/study stay
unchanged. No further subgroup/weight searches on these quarters.

011: **narrow tail correction complete; earlier validation selects no change**
([results](research/MODEL_TAIL_SMOOTHING_011.md)). A relative floor applies only below1%
and only to forecasts heavily suppressed against their Platt stage.149inputs
learn strength0.1 on earlier forecasts but lose on later validation;176inputs
fail the strict fitting Brier guard. Both select identity. A separately frozen
post-result diagnostic shows that lifting the known near-zero Naas forecast
to0.0972% is possible, but later results remain mixed even without the Brier
veto: tiny uncertain149Q1LLgain, transferloss and worse176Q1LL. No promotion or
rescue of009. All32focused checks pass;11primary+4diagnostic evidence files and
exact replay verify. Production/study stay unchanged. Close this floor/blend
line; return to stronger discrimination, including the original plan's
remaining history/uncertainty family after checking prior ablations.

010: **fixed-tree calibration shrinkage complete; no promotion**
([results](research/MODEL_CALIBRATION_010.md)). A coefficient fitted on earlier2025Q3
forecasts and validated onQ4 mixes the current probability curve toward its
Platt stage. On2,591January–March races, Brier improves slightly but primary
raceLL worsens1.81799380→1.81818510; confidence intervals cross0. April–May
with the same December trees improves1.90437727→1.90275709, also uncertain.
No near-zero-probability winner occurs in either later period, so these data
cannot directly confirm protection against the diagnosed rare failure.
The pace variant's earlier Brier rule chooses zero adjustment. Market scores
remain materially better. All18focused tests pass; exact native/overlay
replays and19sealed artifacts verify. Production and the forward study are
unchanged. Keep the next work on model predictions; broad calibration changes
have not closed the market gap.

009: **actual pace/race-shape model comparison complete; candidate rejected**
([results](research/MODEL_PREDICTIONS_009.md)). Following the user's direction to focus
on predictions, trained all six baseline/candidate models:149existing inputs
versus176with pace/setup features, same native recipe and three historical
test quarters. On9,356races, primary race LL1.87204489→1.87310002, uncertain
delta+0.00105513; only one quarter improves. Brier improves slightly but
uncertainty includes no gain; binary loss worsens. No promotion or narrowing
to a favourable subset. On8,942matched fields, the market gap widens slightly.
Six saved models replay within1e−12;15focused checks pass. Production stays
unchanged. Error attribution points to extreme underestimation as a concrete
next modelling question. Development continues now; the forward pilot does
not block model development. Recording/UI work is lower priority.

Started 2026-10-25 after approval of `MARKET_PERFORMANCE_PLAN.md`.

## Implemented foundations

- `src/forecast_evaluation.py`: shared strict whole-field validation, proper
  probability scores, tie-aware winner accuracy, paired race bootstrap and
  date-cluster sensitivity. Existing calibration and saved-run comparison
  scripts now import the same scoring implementation.
- `src/market_replay.py` and `scripts/replay_market_snapshots.py`: per-market
  chronological replay, including interleaved event files, fixed first-known
  schedule, complete decision fields, timestamped prices and separate later
  withdrawal labels. Whole-field rejection and immutable source manifests.
- `RacePredictor.fit_as_of`: all eligible settled history through an explicit
  date, chronological capacity selection and training-only calibration,
  optional win-only research, no inherited holdout scores. Native fitting and
  serving calibration are shared with the existing production implementation.
  Saved artifacts retain fitting metadata.
- `src/forecast_benchmark.py` and `scripts/benchmark_forecasts.py`: identical
  complete field alignment; separate historical forecasts for fitting and
  validating market corrections; final purged independent refit; M0/M1/M2
  ladder; predeclared diagnostics and manifest-bound cached artifacts.

The detailed contract is in `FORECAST_BENCHMARK_PROTOCOL.md`.

## Data and verification

The new research snapshot contains 6,373 unique settled markets / 59,039
decision-field runners from 2026-05-01 through 2026-10-10. It preserves 254
markets with a later withdrawal. Archive overlaps agree and are deduplicated
by market/selection ID. All three local source archives and the output have
SHA-256 fingerprints.

Within the results database's requested May 1–October 12 range, 6,546 races
were considered: 13 have invalid single-winner fields; 429 are unmatched or
have a different decision field; seven have conflicting winner labels;
6,097 complete identical races / 56,255 runners remain. This is conditional
coverage, not a live racecard backtest. Race IDs and conflicting labels are
saved for investigation without modifying results.

`data/audit/market_performance/market_coverage.json` also records the full
as-issued market score separately, actual source dates and price ages. The
maximum LTP age is much larger than the median; a frozen freshness diagnostic
must accompany the comparison. No executable-price or profitability claim.

The experiment 001 full regression run passed **726 tests** (24 warnings) in 400.07
seconds. A further **61 focused checks passed** after the final experience
diagnostic and walk-forward wording corrections. Native full-refit, cutoff exclusion, future-data invariance, win-only
behaviour and saved-artifact replay are covered. A production replay on eight
races / 42 runners matches audit forecasts to 5.6e-17. Deep project status
still verifies database/store/code/production consistency, with the existing
13-day data-age warning. Production remains `20261025_193016_331639`.

Evidence lives under `data/audit/20261025/market-*.log` and
`data/audit/market_performance/`.

## Experiments

001: **complete**, with results in [MARKET_COMPARISON_001.md](research/MARKET_COMPARISON_001.md).
On 3,380 identical races /31,235 runners, independent Brier is 0.086853 and
race log loss 1.873924; T-5 market Brier 0.083297 and race log loss 1.759440.
The independent model remains clearly behind. Pre-test correction selection
gave it zero weight. The market correction is slightly worse than M0, with
no demonstrated gain, so retain M0 as the market-informed reference. Its
original manifest, source archive, frozen input projection, models and paired
forecasts are retained. No production change or independent parity claim.

002: **complete; candidate rejected** ([results](research/MARKET_COMPARISON_002.md)):
149 existing features versus **355 eligible columns** (206 added `atr_*`
features), on 2025Q3, 2025Q4 and 2026Q1. The registered rule requires pooled
race-log-loss improvement with an upper
95% paired bound below zero, no worsening runner Brier/binary log loss, and
improvement in at least two quarters. A partial run is diagnostic only.
This registration predates inspection of experiment 001's test scores.
All six fits finished in 4,679 seconds. On 9,356 races / 85,863 runners,
baseline race log loss 1.872045 versus candidate 1.876713, paired difference
+0.004668 (95% race interval +0.000852 to +0.009229; date-cluster interval
+0.000980 to +0.009326). Every quarter worsened. Pooled runner Brier and
binary log loss also worsened. Retain the 149-feature recipe; do not search
subsets against these results. The next bounded comparison is the planned
conditional race-logit / race-softmax test, respecting the older failed attempt.
No equivalent earlier full-family independent proper-score comparison was
found. All 206 proposed additions passed the actual feature functions' prefix,
future append/change, current-day label-change and row permutation checks;
positive and negative controls are recorded in `002_feature_invariance.json`.
This is a temporal feature-construction check, not proof of historical
publication-time availability. Fixed sequential fits use the frozen feature
store and four workers, with identical float32 inputs for both variants.
Execution evidence and source snapshots are in `002_atr_family/`; continuation
details are in `data/audit/market_performance/002_docs/archive/CONTINUATION_before_cleanup_2026-11-05.md`.
Secondary T-5 context on 8,942 identical races / 81,696 runners confirms a
large gap: baseline race log loss 1.868796, candidate 1.872976, market 1.749283.
Both fail practical parity, including the fixed 7,719-race freshness subset.
The full local replay contains 9,299 markets; overlaps agree. Whole-field
exclusions and winner-label disagreements are preserved without changing
canonical results. No market corrections were fitted for this assessment.

After scoring, the ten known unsafe composites/descendants were blocked in
every feature-selection mode, preventing unrestricted or explicit allowlists
from reintroducing them. Original trial/assessment sources remain archived;
this guard does not change their features, forecasts or the current store.
Final regression after the guard: **737 passed**, 24 warnings, 363.19 seconds.
Production replay on 64 shuffled fields / 635 runners matches saved win
forecasts to 1.67e-16 and full versus win-only probabilities exactly. Final
deep status passes with the existing 14-day age warning. The original fit,
assessment and final implementation sources are preserved separately.

The initial 365-column family proposal was reduced before fitting: two
sectional composites use full-dataset standardisation. A synthetic future
append changes three historical rows of `atr_speed_score` and
`atr_stride_x_gallop`, while the underlying prior-run finishing-speed average
is unchanged. Both composites and their eight derived columns are quarantined
from the proposed frozen-store trial. Evidence:
`data/audit/market_performance/atr_future_dependence.json`. Repairing their
values requires date-safe construction and a genuine feature-store rebuild;
the current 149-feature forecast does not use these ten columns. The remaining
family passed the temporal invariance review before the trial began.

003: **complete; neither candidate advances**, with full results in
[MARKET_COMPARISON_003.md](research/MARKET_COMPARISON_003.md).
The older Plackett–Luce scratch comparison was recovered from Git: it dropped
true non-finishers, used an older snapshot and lacked the current probability
protocol. One registered comparison tested conditional linear logit and
race-softmax LightGBM against the exact cached 149-feature baseline forecasts,
on the same three development quarters. All six fits finished before score
inspection; no feature, objective or regularisation sweep. Both
recipes share native data preparation and chronological serving calibration.
Source and input hashes are frozen; the data/store/production are unchanged.
On 9,356 races, linear race LL 1.890830 clearly loses to binary 1.872045.
The tree improves Brier .08754346 -> .08736141 and race LL to 1.871035,
but its LL delta interval [-.005101,+.003276] crosses zero; date sensitivity
agrees. It improves two quarters and loses one. Keep the secondary Brier gain
as research evidence; do not move the primary gate or promote it. The matched
market comparison remains far behind: tree 1.867684 versus market 1.749283.
Stop this bounded architecture expansion as planned. Reconsideration needs
registered confirmation on other data, not tuning these quarters. Full suite
742 passed; nine actual artifacts replay to 1.11e-16; final deep status passes.

004: **complete; primary benefit unresolved**
([report](research/MARKET_COMPARISON_004.md)). Six native monthly refits used the retained
149 inputs and identical daily features; three first months share their exact
quarterly models. On all 9,356 races, monthly race LL improves 1.87204489 ->
1.87097477, but its interval [-.00436065,+.00347671] crosses zero; date-cluster
sensitivity agrees. Brier improves .08754346 -> .08746770 in every quarter,
with a negative one-sided pooled upper95 bound. Keep that secondary evidence,
without replacing the primary gate. Mean parameter age falls 52.46 -> 22.21
days; no established primary gain and no cadence sweep or promotion.
Matched monthly/market race LL 1.86755050 / 1.74928256 closes only 1.04% of the
quarterly model's market gap at the point estimate; freshness sensitivity also
fails parity. All nine artifacts replay 576 shuffled fields to 1.43e-14.
Full suite: 745 passed, 24 warnings; reporting follow-up: 32 passed. Deep status
passes and now displays the saved July 8 fitting cutoff separately from the
October 12 feature history. No data refresh, feature rebuild, production change
or architecture expansion.

005: **deployment fitting implemented, qualified and activated**
([report](research/DEPLOYMENT_FITTING_005.md)). The shared API, CLI and training app now
support a declared settled cutoff and a fingerprinted link to an evaluated
recipe. Actual native win/place fitting and training-only calibration are
reused; the refit inherits no holdout scores. Results-only collection and
streamed parquet consolidation keep the refresh bounded. Native feature
calculations and FE signature remain unchanged.

The audited generation is preserved. New data add 4,473 rows through Oct25;
the store now holds 753,734 runners / 80,135 races. Every earlier feature value
matches across 749,261 rows / 1,163 columns. Sixteen inconsistent recent fields
remain pending. Production **20261026_140950_205296** uses the retained 149
inputs, 665,252 eligible training rows / 71,207 races through **Oct19**, replacing
the July8 parameter cutoff. The previous run is retained. Full suite **769
passed**; original versus saved win/place probabilities match exactly on
64 fields / 574 runners, including the production loader. This is operational
qualification, with no new performance or independent-parity claim. The
monthly/softmax research challengers remain unpromoted and the overall plan
remains in progress.

Historical experience diagnostics must use `prior_form_runs`. The raw
`horse_runs` is a scrape-time lifetime total, already excluded from model
inputs, and cannot define a reliable historical experience subgroup. Any
initial experiment 001 exposure slice using that raw total is superseded by
the corrected diagnostic report; primary forecast scores are unaffected.

006: **forward baseline recorder complete and qualified**
([report](research/FORWARD_FORECASTS_006.md)). The Picks app and manual CLI share one
recorder: pin the actual independent artifact and source/rules, discover the
first schedule, fetch a fresh card, run native live inference, recheck the
entire card unchanged, then refresh Matchbook WIN best-back GBP quotes.
Complete fields, quote ages and actual publication in the S-330 to S-300
second window gate the first eligible pair; all other attempts remain for
monitoring. No fabricated probabilities or substitutions for missing prices.

Both live operational captures contain nine runners; saved native inputs and
raw prices reproduce both probability vectors with maximum error zero.
The first capture exposed card ageing during native loading and remains in
its original study. The revised study adds the exact fresh-card recheck and
starts eligible recording on October27. Today's captures are monitoring only.
Saved-attempt display, read-only assessment and retired-study handling pass
UI smoke checks. Formal confirmation sample size and the final decision plan
remain outstanding; reports cannot declare parity or promote a model.
Full regression **793 passed**, 108 warnings; all24 new checks pass. All41
sealed005 evidence files verify, model/feature hashes match and the database
schema and every row in all six tables match the preserved backup. Production
and the historical feature generation remain unchanged. Ready-to-use study:
`data/forecasts/forward/20261026_baseline_v2`. No scheduled collection.

007: **confirmation feasibility and fixed baseline pilot complete**
([report](research/CONFIRMATION_FEASIBILITY_007.md)). A new frozen variance analysis
uses the retained independent archive and the closed, unpromoted market
correction. It preserves runner-weighted Brier and accounts for race/date
dependence. The conservative normal approximation uses 95% marginal power
for four required score/sampling gates, giving at least 80% joint power under
its assumptions. No new model fitting or correction selection.

At exact parity mean with archived variance, required paired races are about
1,215,235 for the independent recipe and 2,244 for the small market correction;
at the correction's observed historical mean, about 3,093. The independent
archive's observed mean is outside the margins, so no larger sample delivers
high-power parity under that mean. The comparison uses historical Betfair LTP;
variance transfer to the current refit and live Matchbook population is
unproven. These are planning scenarios, not candidate advancement.

The immutable companion readout plan starts October 27 and requires 84 calendar
days, 1,000 eligible published pairs and 60 race dates, followed by seven days
for results. Closure uses publication metadata, not outcomes or settlement
availability. It is a variance/coverage/gap pilot, not powered confirmation.
The app shows progress; the readiness CLI reads no result labels. Earliest
possible readout January 26, 2027 if count conditions are met; otherwise later.
All 62 focused checks and the app progress smoke check pass. New modules are
separate from the frozen recorder/model; the original study remains usable.

008: **bounded manual timing complete and qualified**
([report](research/TIMED_RECORDING_008.md)). New Record in eligible window action and
CLI --at-window calculate early, wait within a manually started attempt, then
recheck the complete card and refresh prices. A monotonic/UTC check detects
clock changes and missed deadlines. Waits are bounded; actual publication and
all field/price/history gates remain unchanged. New v3 study pins the same
production artifact and original pilot thresholds; v1/v2 remain preserved.

The live six-runner20:30Southwell attempt waited138.54seconds after native
inference and published20:24:33.448130UTC, within20:24:30–20:25:00. It correctly
rejected the independent pair when three racecard odds changed. Market-only
probabilities replay exactly; no selected pair, no old probabilities attached
to changed inputs and no performance claim. The app displays this accurately.
The unchanged-card coverage limitation is explicit and its bounded native
feature-dependency diagnostic is retained separately from issued forecasts.
Full regression833passed,108warnings,490.45seconds. All13new timing/recorder
checks and the27planning/readout tests from007 pass in the full run. No change
to production parameters or the149selected inputs.
Single-field native replay of the refreshed card changes none of the149inputs
and yields zero reconstructed probability difference. This is diagnostic,
not a general current-odds independence proof. Original market-only record
and rejection remain unchanged. A broader bounded dependency check is the
next coverage task before any new input policy. Active studybaseline_v3,
sameOctober27start and original1000pair/84day/60date/7daygrace pilot conditions.

Do not select features on 2026Q3. Do not infer independent market parity from
a market-input forecast matching the market baseline. Further independent
work and fresh forward confirmation remain part of the accepted plan.
