# Price timing: 11:00 against a fixed time before the off (research 104, 15 November 2026)

Question (ERROR_ANATOMY next step 5): the 11:00 exchange price is a
different distance from the off in every race. Does a price at a fixed time
before the off do better, and how much does the last hour add?

Each arm is the best joint correction recipe (`more-groups-deeper`) plus one
price group, declared before scoring in `research/error_anatomy/joint_variants.py`,
groups in `research/error_anatomy/morning_price.py`. Native forecasts,
2025Q3–2026Q2, 11,905 races.

- `with-1100` (existing): the runner's 11:00 share of the field, gap to the
  forecast, price against the favourite, rank, age of the last trade, number
  of changes, the move from 08:00 and the share of the field priced.
- `with-1100-hours`: the same plus hours from 11:00 to the off (race median).
  Races off at or before 11:00 (43 since 2020) have the 11:00 price blanked,
  because it can be in-play. The existing `with-1100` group still includes
  them (about 0.05% of races).
- `with-3600`: the same inputs one hour before the off; the move is from the
  runner's first recorded price (18:00 the evening before, else 08:00, else
  11:00 when it is over an hour before the cut), with the hours the move spans.
- `with-300`: the same five minutes before the off, an upper reference.

| Arm | Native log loss | Against the correction without prices | Against `with-1100` |
|---|---:|---:|---:|
| `more-groups-deeper` | 1.80506 | | |
| `with-1100` | 1.77915 | −0.0259 | |
| `with-1100-hours` | 1.77920 | −0.0259 | +0.00006 [−0.00021, +0.00033], 1/4 |
| `with-3600` (T-60) | 1.76936 | −0.0357 | −0.00979 [−0.01263, −0.00699], 4/4 |
| `with-300` (T-5) | 1.76118 | −0.0439 | −0.01797 [−0.02207, −0.01431], 4/4 |

T-5 against T-60: −0.00818 [−0.01125, −0.00514], 4/4. Betfair SP on the same
races is about 1.745.

## Reading

- Telling the model how far 11:00 is from the off adds nothing. The value is
  in the price being later, not in knowing its age.
- A fixed hour before the off is worth about 0.010 more than 11:00. The
  correction then closes about 59% of the gap to BSP (11:00: 43%).
- The last 55 minutes add another 0.008 (73% of the gap at T-5). A forecast
  that late is mostly following the market; whether a bet placed at T-x beats
  the close is a CLV question these log losses do not answer.
- For serving, T-60 needs a live exchange price for each race an hour before
  its off, and a forecast run per race rather than one morning batch.

Outputs: `data/research/price_timing/joint_variants.log`;
corrections in `data/research/error_anatomy_20261110/joint_variants.parquet`.
