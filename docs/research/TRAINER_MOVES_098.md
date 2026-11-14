# Trainer moves (research 098, 14 November 2026)

Question: does the model account for a horse changing trainer: how much
better or worse the new trainer is, how long the horse has been there, and
whether some trainers improve their new arrivals?

Before this study the store had one input, `trainer_switch` (a different
trainer from the last run), not among the 87 selected inputs. The error study
of 10 November found switchers as a whole forecast correctly (353 won against
345 expected), because moves up and moves down cancel.

## Data check

Each run's trainer comes from that race's Sporting Life result page and is the
trainer on the day, including pages scraped years after the race. Change rates
between consecutive runs match for pages scraped live and pages scraped over a
year later (0.8% against 0.6% within 30 days; 3.9% against 4.0% at 30–90
days). BHA training histories agree for The Rain King, Fortunas Dream, Komodo
Rose and the 2021 Elliott-to-Foster transfer. About 7% of recorded changes
are licence changes at the same yard; this study treats names sharing a
surname as one yard.

## Inputs

`research/trainer_moves_098/inputs.py`, tested in `test_inputs.py`:

| Input | Meaning |
|---|---|
| `tm_elo_change` | New trainer's pre-race Elo minus the old trainer's at the horse's last run for the old yard; missing if the horse never moved |
| `tm_elo_change_fresh` | The same, on the first three runs for the yard only |
| `tm_runs_in_yard` | Earlier runs for the current trainer |
| `tm_arrival_edge` | Trainer's earlier arrivals (first three runs) against SP, earlier dates only, shrunk with 30 pseudo-runs |
| `tm_arrivals_n` | Number of arrival runs behind that edge |

## Screen against stored forecasts

87-input out-of-sample forecasts (`data/processed/forecast_history`),
2022-07-01 to 2026-06-10, first three runs for a new yard:

| Elo change | Runners | Won | Model expected | z | BSP expected | z |
|---|---:|---:|---:|---:|---:|---:|
| −245 to −59 | 8,544 | 461 | 573 | −5.1 | 451 | +0.5 |
| −59 to −30 | 8,547 | 563 | 628 | −2.8 | 545 | +0.9 |
| −30 to −6 | 8,545 | 651 | 688 | −1.6 | 654 | −0.1 |
| −6 to +21 | 8,548 | 829 | 762 | +2.7 | 806 | +0.9 |
| +21 to +162 | 8,544 | 1,234 | 1,000 | +8.5 | 1,175 | +2.0 |

New arrivals at the top fifth of trainers by arrival edge won 794 against 713
forecast (z +3.5) but 768 at BSP (z +1.1). Full table in
`data/research/trainer_moves_098/screen.csv`.

## Admission test

`python -m src.modelling admit`, screen capacity, folds 2025Q3–2026Q2 (11,632
races), the 87 inputs copied from the 2026-11-11 selection matrix:

| Arm | Race log loss | Against baseline | Against shuffled twin |
|---|---:|---:|---:|
| Baseline | 1.82894 | | |
| All five inputs | 1.83022 | +0.00128 [−0.00093, +0.00317], 1/4 | −0.00086 [−0.00238, +0.00074], 4/4 |
| `tm_elo_change_fresh` only | 1.83041 | +0.00147 [−0.00069, +0.00331], 0/4 | +0.00054 [−0.00097, +0.00195], 1/4 |

Not admitted. The five inputs carry some information against their twin, but
less than adding inputs costs at this capacity. The single input does not beat
its twin.

## Inside the joint correction

`research/trainer_moves_098/correction.py` registers the five inputs as the
group `trainer-move` and a twin shuffled within quarter as
`trainer-move-twin`. `python -m research.error_anatomy.joint_variants
--variants with-trainer-move with-trainer-move-twin`, both declared before
scoring, each the best recipe (`more-groups-deeper`) plus one group, 424
inputs, native forecasts 2025Q3–2026Q2 (11,905 races, about 72 minutes each):

| Variant | Native log loss | Against twin | Against `more-groups-deeper` |
|---|---:|---:|---:|
| `more-groups-deeper` | 1.80506 | | |
| `with-trainer-move` | 1.80531 | +0.00001 [−0.00073, +0.00066], 2/4 | +0.00025 [−0.00051, +0.00095], 2/4 |
| `with-trainer-move-twin` | 1.80530 | | |

Null. The correction already recovers most of the miss. First three runs for
a new yard, scored quarters, by Elo change:

| Elo change | Runners | Won | Native forecast expected | z | Corrected expected | z |
|---|---:|---:|---:|---:|---:|---:|
| −245 to −63 | 1,997 | 106 | 126.2 | −2.0 | 118.1 | −1.2 |
| −63 to −31 | 1,996 | 157 | 153.4 | +0.3 | 151.4 | +0.5 |
| −31 to −6 | 1,996 | 180 | 164.4 | +1.3 | 164.1 | +1.4 |
| −6 to +24 | 1,996 | 226 | 187.9 | +3.1 | 192.4 | +2.7 |
| +24 to +159 | 1,996 | 279 | 250.5 | +2.1 | 263.8 | +1.1 |

## Reading

The forecast error is large but confined to 9% of runners. Rescaling those
forecasts by each group's own miss improves race log loss by only −0.00107
in-sample (an upper bound). As model inputs the move costs more than it
gives. Inside the correction its information is already present, through the
horse-state and trainer-residual groups and the `yard` group of 93dcdfd,
which found the same null from the spell-form side. Betfair SP prices it
too, apart from a 5% excess on the largest upgrades. Closed.

Outputs: `data/research/trainer_moves_098/` (inputs, screen, matrix,
`admit_all`, `admit_fresh`).
