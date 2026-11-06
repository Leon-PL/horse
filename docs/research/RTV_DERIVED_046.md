# 046 RTV-derived inputs and the Timeform card

31 October 2026. Development period only; the golden hold-out (from
2026-06-11) has not been touched.

## What was built

`src/rtv_features.py` now builds three new families from data already
collected from Racing TV:

| Family | Inputs | Source |
|---|---|---|
| Timeform card | `rtv_card_tf_rating`, `_stars`, `_tipped`, `_improver`, `_doubtful`, `_focus` | The racecard for the current race |
| Sectional time figures | `rtv_time_fig_mean3`, `_best3`, `_last` | Each prior run's final-sector time against a par plus a leave-one-race-out meeting variant |
| Within-race measures | `rtv_{acceleration, avg_length, avg_frequency, max_frequency, stride_length, exit_speed, time_in_envelope, speed_lost, recovery_time, exit_entry}_rel_mean3` | Each prior run's stride and jumping metrics as a z-score within its own race |

The existing `rtv_*_mean3` family averages raw values, which carry trip and
going. Normalising within the race removes both.

**History rule.** Time figures and within-race measures use the last three
runs that matured two days before the race. This is the rule the existing RTV
history family already follows.

**Time-figure par.** The par is a LightGBM regression of log winning time on
distance, going, surface, race type and course. It is fitted only on
RaceIQ-era races, 2023-06-10 to 2024-06-30, so every refresh reproduces the
same par. The fit originally had no start date. The 2020–2023 racecard
backfill would then have added 34,000 older results to it and moved stored
figures by up to 5.2 points, so the start date is now fixed. The figures the
native check used included one stray probe day (2021-07-10). Without it they
move by at most 1.2 points.

**Timeform card.** These are the only RTV inputs that describe the current
race:

- **Live rows** read the card from the pre-decision snapshot.
- **Stored history** falls back to the backfilled card, kept in
  `data/rtv_cache/feature_cards.parquet`. That file is fingerprinted by race
  date, so a late card for a stored race forces a rebuild.
- **Leak checks.**
  - The backfilled values match the evening-before snapshot: 1 change in 490
    runners on 2026-10-30.
  - Over the T-5 market the card adds nothing (see the ledger entry). Post-race
    information would add a lot.

**Snapshot parsing fix.** The snapshot path parsed ratings with symbols
("111p", "94?") as missing, which affected 14% of live cards. The card and
`rtv_pre_timeform_rating` now share one parser.

**Fidelity to the research version.** On the 351,209 rows used by the quick
ablation:

- The card fields match exactly.
- The history features correlate 0.99 or more. The differences come from the
  quality bounds and from linking on SL identities.
- The 17 existing run metrics are unchanged.

## Coverage in the rebuilt store

| Year | Timeform rating | Stars | Time figure | Stride (within race) | Jumping (within race) |
|---|---|---|---|---|---|
| 2023 | 48% | 58% | 25% | 29% | 6% |
| 2024 | 83% | 100% | 87% | 88% | 32% |
| 2025 | 83% | 100% | 92% | 92% | 34% |
| 2026 | 84% | 100% | 92% | 93% | 31% |

Before June 2023 the store has nothing, apart from one probe day (2021-07-10).
RaceIQ sectionals and stride metrics do not exist that early, but the racecard
does.

### Racecard backfill to September 2020

Racing TV results, racecards, odds and tips were scraped from 2020-09-15 (the
first day in the store) to 2023-06-09: 995 days and 37,558 races, all complete.
The card file was then rebuilt. **The feature store has not been rebuilt yet**,
so the table above and the native result still describe the June 2023 start.

Card coverage of `races.db` runners in `feature_cards.parquet`:

| Year | Card | Timeform rating | Stars |
|---|---|---|---|
| 2020 (from Sep 15) | 100% | 78% | 100% |
| 2021 | 99% | 81% | 99% |
| 2022 | 100% | 82% | 100% |
| 2023 | 100% | 82% | 100% |
| 2024 | 100% | 83% | 100% |
| 2025 | 100% | 83% | 100% |
| 2026 | 100% | 84% | 100% |

**Leak check on the old cards.** A card revised after its race would know more
than the T-5 market, so its within-race correlation with `won − market` would
jump in the backfilled years. It does not. The rating's correlation stays
between −0.003 and +0.003 in every year, and its correlation with winning is
0.17 throughout. Stars and tipped sit slightly negative in all years (−0.002
to −0.016), with no break at 2023. Ratings change between a horse's
consecutive races 93% of the time, so they are stored per race, not as the
horse's current rating.

## Native validation

`scripts/validate_rtv_derived_native.py` fits the production win recipe twice
per quarter, with no seed or quarter search:

- **Recipe:** `RacePredictor.fit_as_of` with distillation, Platt + isotonic
  calibration and capacity selection.
- **Arms:** `feature_sets/win.txt` (154 inputs) against
  `feature_sets/win_rtv_derived.txt` (173 inputs).
- **Training:** each fit runs through eight days before its quarter.
- **Store:** `1c83cfaeac51bb2e`, which includes the 045 rating repairs.

The recipe kept all 19 new inputs.

| Quarter | Races | Baseline race LL | Candidate race LL | Change | Date-cluster 95% |
|---|---|---|---|---|---|
| 2026Q1 | 2,592 | 1.79614 | 1.77829 | −0.01785 | [−0.02514, −0.01068] |
| 2026Q2 | 2,542 | 1.88380 | 1.86635 | −0.01744 | [−0.02521, −0.00964] |
| Pooled | 5,134 | 1.83954 | 1.82189 | −0.01765 | [−0.02288, −0.01294] |

The other metrics all improve on the pooled set:

| Metric | Baseline | Candidate |
|---|---|---|
| Brier | .087947 | .087395 |
| Binary log loss | .30065 | .29833 |
| Calibration error (ECE) | .0034 | .0023 |
| Top-1 accuracy | 30.2% | 30.8% |

The change in race-mean Brier is −.00063 [−.00097, −.00029].

Against the market:

| Market | Fully priced races | Baseline | Candidate | Market | Share of gap closed |
|---|---|---|---|---|---|
| Betfair SP | 5,134 | 1.83954 | 1.82189 | 1.72870 | 15.9% |
| T-5 Betfair price | 4,931 | 1.82324 | 1.80532 | 1.72306 | 17.9% |

The earlier quick 400-tree ablation measured −0.0187 over four quarters, so
the native recipe confirms it. Training windows contain Timeform history only
from June 2023, so the full fit learns from about two and a half years of
cards. After the racecard backfill and a store rebuild they would have cards
from September 2020. Only the candidate fits need rerunning, because
`win.txt` has no RTV inputs.

## What this is not

This shows the model got more accurate. It does not show an edge over the
market:

- The model still trails the SP by 0.093 nats per race.
- On the T-5 market harness, none of the families carries information beyond
  the price.

No selection rule or CLV test has been run, and production has not changed.
After the rebuild the promoted model is stale and needs a retrain. Receipts
are in `data/audit/rtv_derived_native/` and
`data/audit/rtv_derived_ablation/`.
