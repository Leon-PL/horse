# Irish races without a class (research 100, 14 November 2026)

Question: Irish races carry no class in the results (92% blank since July
2025; only Group and Listed races have one). Does filling the gap help the
model?

## The gap

Among the 87 selected inputs, `class_change_vs_field` is missing for 95% of
runners in Irish races, against 8% in British races. A British runner whose
last run was in Ireland loses it too. Other Irish gaps checked at the same
time:

- ATR closing speed is missing for 92% of Irish runners. ATR covers Britain
  only.
- Draw is missing for 9.2% of Irish flat runners against 3.0% British. Every
  Irish flat race without draws is a bumper ("Flat Race",
  `race_is_bumper` = 1), which has no stalls. Not a data fault.
- RTV stride data is missing for 19.4% of Irish flat runners against 9.9%
  British, from Racing TV's coverage.

## Inputs

`research/irish_class_100/inputs.py`, tested in `test_inputs.py`:

- `class_eff`: the published class where there is one. Otherwise the class
  is read off the race's prize money, against British races of the same code
  in the previous year: the class with the nearest median log prize,
  interpolated between neighbouring classes (2020 uses its own year).
- `class_change_eff`: today's effective class minus the previous run's.
- `class_change_eff_vs_field`: that change minus the race mean.

This cuts missing class change for Irish runners from 95% to 9%. On prize,
Irish races read about one class higher than their British equivalents:
maidens have a median effective class of 3.4, Premier Handicaps 1.0.

## Admission test

`python -m src.modelling admit`, screen capacity, folds 2025Q3–2026Q2 (11,632
races), the 87 inputs plus `class_eff` and `class_change_eff_vs_field`:

| Arm | Race log loss | Against baseline | Against shuffled twin |
|---|---:|---:|---:|
| Baseline | 1.82894 | | |
| With the two inputs | 1.82923 | +0.00029 [−0.00150, +0.00200], 2/4 | −0.00097 [−0.00225, +0.00031], 3/4 |
| Twin | 1.83021 | +0.00127 | |

Split by where the race was run:

| Races | n | Against baseline | Against twin |
|---|---:|---:|---:|
| British | 9,002 | −0.00002 [−0.00187, +0.00207] | −0.00129 [−0.00263, +0.00014] |
| Irish | 2,630 | +0.00136 [−0.00294, +0.00530] | +0.00011 [−0.00260, +0.00298] |

Not admitted. The repair does nothing in the Irish races it was built for.
Prize money is a poor stand-in for class there, and the Irish fields are
already placed by ratings and the market. Closed.

Outputs: `data/research/irish_class_100/` (inputs, matrix, `admit`).
