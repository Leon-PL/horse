# 055 — Where the 054 model was most wrong, and why (2 November 2026)

**Question.** On the new model's out-of-sample predictions, are the biggest
errors caused by wrong data, or by information the market had and the model
did not?

**Answer.** The data is correct. We re-read 73 of the worst races from the raw
Sporting Life pages and found no value errors. The errors come from
information the model does not have:

- **About half arrives after 11:00.** It shows up only as market money
  (gambles and drifts).
- **The rest was already in the morning price.** That covers runs we do not
  store (abroad, point-to-point), debutant reputation and yard intentions, and
  a few modelling blind spots.

Data *coverage* (runs we don't store) accounts for under 3% of the gap.

## Instrument

- **Run:** `20261102_050631_314757`, the evaluated twin of production
  `20261102_063010_291745`. It uses the 126 inputs from 048 and store FE
  `32e71a2cdff8a8b5`.
- **Data:** its own `predictions.csv`, covering the 90-day hold-out
  2026-08-01..10-29: 3,431 races and 32,535 runners. Every race has exactly one
  winner.
- **Joins:**
  - results from `races.db`, joined on the `normalise_horse_key` name. A raw
    name join drops 1,400 apostrophe names ("Rene'S Gold").
  - `betfair_bsp.parquet` (BSP, and `ltp_1100` as the 11:00 morning price),
    joined on the `src/utils` keys, with 97% coverage.
  - the RTV Timeform card text, with 99.9% coverage.
- **Probabilities:** all are normalised within the race over actual runners,
  so non-runner reductions do not distort early prices.
- **Status:** `project_status.py` now reports the store STALE because FE code
  has moved on (`69babeb9…`). That does not affect this analysis, since the
  predictions were made on the matching store. It does block serving until the
  store is rebuilt.

| Race log loss (winner) | All 3,431 races | 2,672 races with an 11:00 price |
|---|---|---|
| Model | 1.8492 | 1.8335 |
| Betfair 11:00 | – | 1.7868 |
| SP | 1.7656 | 1.7547 |
| BSP | 1.7530 (3,322 races) | 1.7467 |

On the morning-priced races, the model→morning gap is 0.0468 and the
morning→BSP gap is 0.0401. **46% of the model's deficit to BSP is information
that reaches the market after 11:00.** For the 100 worst races the split is
0.70 / 0.35: two-thirds was already in the morning price.

## The gap is broad, not a few disasters

The winner's log(p_SP / p_model) sums to 286.7 nats (0.0836 per race):

| Worst races | Share of gap |
|---|---|
| 10 | 6% |
| 50 | 22% |
| 200 | 64% |
| 500 | 121% |

The model *beats* SP in 40% of races. When the SP favourite wins, the gap is
240 nats, or **84%** of the total. That is consistent with 050.

| Winner's history | Winners | Gap per race | Share of gap |
|---|---|---|---|
| Exposed (4+ DB runs) | 2,575 | 0.078 | 70.0% |
| Lightly raced (1–3) | 675 | 0.082 | 19.4% |
| Debutant | 133 | 0.171 | 7.9% |
| Off-DB form (abroad / point-to-point) | 48 | 0.163 | 2.7% |

The model is calibrated *within* each of these segments: sum of p is within
±2% of wins, except debutants. Debutants are +16% for the model and +12% at
SP, while BSP is +1%. The market wins by discriminating inside the segments,
not by correcting a segment bias.

## Data correctness: audited against raw pages

`raw_check.py` (scratchpad) re-fetched and independently parsed
`__NEXT_DATA__` for **73 races**:

- the top 50 races by gap;
- the 25 races whose model top pick lost by most relative to SP (2 overlap).

That covers 818 runners. **SP, weight, age, draw, OR, trainer, jockey,
finishing position and the runner set all match `races.db`.** No non-runner
is stored as a runner. The checker raised 18 flags, all on days since last
run. They are the checker's own artefact: SL's `previous_results` omits
point-to-point and foreign runs, which the stored SL value includes.

What *is* missing is coverage, not correctness:

- **Overseas runs are not stored** (the scraper keeps UK/IRE meetings only).
  - Quddwah won the Group 2 Zabeel Mile at Meydan and was second in the Group
    1 Dubai Turf, but the DB's last run was 15th of 15 in the QEII. The model
    gave him 14%, against 44% at SP.
  - Golden Splendour won at Chantilly on debut; the DB shows him as a debutant.
  - Marco Du Seuil had French Flat form.
- **Point-to-point runs are not stored.** Nine of the audited horses had them.
  `hidden_form_runs` counts these runs but cannot rate them.
- Across the hold-out, off-DB horses carry 2.7% of the gap. 050's upper bound
  for recovering it was 0.0023.

Side defects that do not reach the current model:

- **The IHRB stewards events are mis-keyed.** In `stewards_events.parquet`,
  91% of IHRB `horse_key` values are jockey or trainer initials ("A.P.O'",
  "M.J"). 0.3% are fixture-matched, Irish coverage effectively ends in 2020,
  and `non_trier` appears once in 81,545 rows. Ledger §7 rejected the
  stewards family, but its Irish half ran on effectively empty data. The case
  for revisiting it is Glin Road Boy (below): his only visible recent run was
  a stewards' non-trier ruling.
- **`results.horse_runs`, `horse_wins` and `horse_places` are scrape-time
  values.** For example, Rogue Crusader's debut row says 2 runs. FE overwrites
  them from the DB history (`feature_engineer.py:1170`), so the model is
  unaffected, but ad-hoc analysis must not read them.

## Information the market had

### 1. Money after 11:00: about 46% of the gap

| Runners (11:00 → BSP) | Runners | Wins | Model expected | BSP expected |
|---|---|---|---|---|
| Steamers (prob ×1.5 or more) | 2,321 | 236 | 176.4 | 250.7 |
| Drifters (prob ×0.67 or less) | 5,451 | 175 | 306.8 | 171.4 |
| Stable | 17,417 | 2,261 | 2,188.4 | 2,250.0 |

The model over-rates drifters by 75% and under-rates steamers by 25%. For
these runners the late market move is the information. Of the 50 worst races:

- 22 were late gambles, either after 11:00 or in the on-course ring;
- 21 were priced from the morning;
- 7 were outsiders the market also had below 5%, which is ordinary luck.

### 2. Public information already in the morning price

The Timeform comment on the card defers explicitly to the market on 2,993
runners ("market check advised", "betting can prove a useful guide", "only of
interest if market speaks positively"). Those runners carry 13.5% of the gap,
at 0.113 per race against 0.080. Timeform is saying the form book cannot
settle these races. That fits 052, where the comment text bought only
−0.0029.

The rest of the morning-priced gap is information the model cannot see or
read:

- debutant pedigree, sale price and yard reputation;
- yard changes and sales: the "changed hands for 1,800gns" cases;
- handicap debuts after a break;
- headgear or tongue-tie firsts.

### 3. Model blind spots: public information, mis-weighted

| Case | Model | SP | What happened |
|---|---|---|---|
| Debutants where the model gave more than 20% | 52.0 expected | 43.1 expected | 34 won. Debutants from big yards are over-rated 1.33× (the market knows which ones need the run). Pink Larkspur 41% / 10%, Thi Qar 34% / 9%, Vienna Waltz 27% / 8%. |
| Lord Byron, Carlisle Flat maiden | 32% | 6% | A Grade-class juvenile hurdler having a spin on the Flat. His jumps ratings carried across, and he won a Grade 2 hurdle at Chepstow a month later. |
| Possessive, Musselburgh seller | 54% | 16% | Sold cheaply out of Karl Burke's yard. A negative signal the model has no input for. |

These are small in aggregate: debutants with p > 0.2 are 2% of the gap.

## Case notes (web-verified)

| Race | Winner | Model | 11:00 | SP | Cause |
|---|---|---|---|---|---|
| Ayr 8 Oct | Alfa Blueburd | 5.4% | 2.8% | 29% | **Late gamble.** 33/1 overnight, returned the 2/1 favourite. The trainer said he was surprised by it. Market-only information. |
| Leopardstown 12 Sep, Group 2 | Quddwah | 14% | 48% | 44% | **Data coverage.** All of his 2026 form was at Meydan (Group 2 win, Group 1 second), and none of it is in the DB. |
| Listowel 23 Sep | Glin Road Boy | 7% | 19% | 26% | **Unused public information.** His Sligo second at 22/1 drew a non-trier ruling (rider suspended 14 days, horse banned 60 days), so the form figure hid his ability. Charles Byrnes trained three of the 35 worst misses. |
| Kempton 4 Sep | Rogue Crusader | 3.9% | 29% | 36% | **Debutant reputation.** Opened 6/4 favourite. The Timeform comment said "Market check advised". |
| Sligo 5 Aug | Anightaway | 1.3% | 2.8% | 6.4% | **On-course gamble.** Ring 33/1 → 9/1, official SP 11/1 (confirmed on irishracing.com) while BSP was 22. The trainer credited a break and a first-time tongue strap. |
| Southwell 15 Oct | Battle Standard | 3.9% | – | 18% | **Handicap-debut plot.** First handicap after 6 months off, backed 9/2 → 3/1, won by 3¼ lengths and was raised 12 lb. |
| Southwell 26 Oct | Honey Dale | 5.8% | – | 23% | **New yard.** Changed hands for 1,800gns. Timeform: "only of interest if market speaks positively". |
| Listowel 26 Sep | Local Resident | 4.1% | 11% | 29% | **Late gamble** (op 15/8). His only point-to-point run is not in the DB. |
| Kilbeggan 4 Sep | Begger | 0.2% | 0.6% | 1.3% | **Luck.** A 67/1 winner in a 4-runner chase who "finished alone". Not a model failure. |

## Implications

- **Data cleaning will not close the gap.** The stored values are right.
- **Recoverable public information is small.** Overseas and point-to-point
  form account for at most 2.7% of the gap. Stewards rulings, if the IHRB
  parse were repaired, cover rare events.
- **About 46% of the gap is market money after 11:00.** A model that runs
  before the market cannot see it. 050 and ledger §8 found that the price
  move closes accuracy but is not tradeable.
- **Two modelling items are cheap to test:**
  - shrinking big-yard debutants towards the field;
  - stopping jumps ratings from transferring to Flat runs (and the reverse).

  Each is worth at most about 2% of the gap.

## Follow-up: fixing what we control (same day)

| Item | Status |
|---|---|
| Timeform comment score as an input (052: −0.0029, 4 of 4 quarters) | **The largest confirmed controllable gain.** It is being installed as 056: `feature_sets/win_selected_056.txt`, `src/rtv_text.py` and a store rebuild that started 09:52. |
| Debutant shrink; trainer and sire debut record | **Screened null.** −0.00021 and +0.00029 out of fold. [Ledger 055](../NEGATIVE_RESULTS.md). |
| Off-DB last run and its position from the form string | **Screened null** (+0.00011). Scraping overseas form is not worth building on this evidence. |
| Jumps→Flat / Flat→jumps rating carry-over | **Screened null** (+0.00033, 0 of 3 quarters). |
| IHRB stewards events mis-keyed | **Recorded as a caveat on ledger §7.** The parquet is orphaned (its scraper and features were deleted on 2026-10-13), so nothing in the model reads it. |
| `results.horse_runs/horse_wins` hold scrape-time values | **Harmless to the model**, because FE overwrites them. Analysis must use DB history instead. |

Script: `scripts/screen_controllable_055.py`. Receipts:
`data/research/controllable_055/summary.json`.

## What arrives after 11:00 (same day)

**Data:** development races only, 2023-01-01..2026-06-10. That gives 38,465
races in which every runner traded at 11:00, T-60, T-5 and T-1 and has a BSP.
The prices come from `betfair_timing.parquet`; volume comes from
`betfair_sp_public.parquet`. Snapshot times are relative to Betfair's
*scheduled* off, so "T-1 → BSP" includes any late start. This is a
description of the market's footprint; it does not observe the information
itself.

**Market accuracy through the day (race log loss):**

| Time | Race LL |
|---|---|
| 11:00 | 1.7736 |
| T-60 | 1.7553 |
| T-5 | 1.7484 |
| T-1 | 1.7342 |
| BSP | 1.7286 |

The 0.045 gain from 11:00 to BSP splits as follows:

| Window | Share of the gain |
|---|---|
| 11:00 → T-60 | 41% |
| T-60 → T-5 | 15% |
| T-5 → T-1 | 31% |
| T-1 → BSP | 13% |

So 44% of it arrives in the last five minutes. On the 10,685 races that also
have out-of-fold predictions, the model is 1.8099 and the 11:00 price
1.7599, so 44% of the model's gap to BSP opens after 11:00 (46% on the 054
hold-out).

**It is the money, not a bias.**
- The median share of pre-play volume matched before 11:00 is 3.4%.
- The 11:00 price is nonetheless calibrated: a power recalibration fitted on
  2023–24 improves 2025–26 by 0.0001.
- What the day adds is discrimination, not a correction of the morning
  price's bias.
- Races whose morning trade was the thinnest share of the day gain most
  (0.058, against 0.039–0.045 in the other quintiles). Part of the gain is
  therefore price discovery.

**Not identifiable news.** These candidates leave the late gain about the
same:

| Candidate | Late gain with | Late gain without |
|---|---|---|
| Late non-runners (prices renormalised over actual runners) | 0.0455 | 0.0449 |
| Rain on the day (wet over 5 mm vs dry) | 0.0462 | 0.0448 |

- **Timeform tips** are already priced at 11:00 (wins / 11:00-expected
  0.98–1.01), and tipped horses drift slightly rather than steam afterwards.
- **First-time headgear** is over-bet in the morning (0.953 of expected
  wins). The day's money corrects it to 0.991.

**Where it concentrates:**
- Irish races, 0.060 against 0.042 in Britain, mostly before T-5 (ring and
  stable money).
- Hurdles (0.053) and bumpers (0.055), against chases (0.037).
- Big fields: 0.073 for 17 or more runners, 0.029 for 6 or fewer.

**Who it is about:** mostly ordinary runners.

| Runners | Share of runners | Share of late information | Share of last-5-minute information |
|---|---|---|---|
| Exposed horses with nothing new | 64% | 70% | 75% |
| Back from 180+ days | 6.1% | – | 9.3% |
| Debutants | 5.4% | 6.2% | 2.6% |

- Horses back from 180+ days are over-represented in the last five minutes,
  which is consistent with fitness being judged on course.
- Debutants' information comes earlier in the afternoon, not from the
  paddock.
- Moderate moves (1.25–2×) on winners carry about 72% of the positive late
  information.
- Gambles of more than 2× (2.7% of runners, in 20% of races) carry about 17%.
  Those runners win 8.6%, against 3.7% implied at 11:00 and 9.2% at BSP.

**Reading.** After 11:00 the market collects the judgement and money of the
~97% of bettors who have not yet bet:
- professional form reading;
- stable and on-course money, strongest in Ireland;
- in the final minutes, what can be seen at the course.

None of it is a discrete public fact the model could ingest. That fits ledger
§8, where moves are not predictable from data public before 11:00.
