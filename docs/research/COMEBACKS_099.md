# Seasonal comebacks (research 099, 14 November 2026)

Question: some horses need their first run back, others go best fresh (the
starting point was an analyst's read of Jonbon's Shloer Chase, 13 November).
The store knows days since the last run and the trainer's record with
returners, but not the horse's own record on reappearances.

## Inputs

`research/comebacks_099/inputs.py`, tested in `test_inputs.py`. A comeback
is a run 90 or more days after the horse's previous recorded run. Only runs
on strictly earlier dates are used.

| Input | Meaning |
|---|---|
| `cb_today`, `cb_days` | Today is a comeback; days since the last run |
| `cb_n` | Earlier comebacks |
| `cb_pct` | Finishing percentile on earlier comebacks, shrunk to 0.5 by two runs |
| `cb_pct_next` | The same on the run after each comeback |
| `cb_needs_run` | `cb_pct − cb_pct_next` (negative = needs the run) |
| `cb_pct_vs_career` | Comeback percentile against the horse's other runs |
| `cb_resid`, `cb_resid_other` | Won − stored forecast on earlier comebacks and on other runs |
| `cb_mkt_gap` | Log closing-price share against the forecast on earlier comebacks |
| `*_today` | The four above × `cb_today` |

## Results

Alone, as a correction screened against its shuffled twin
(`soft_corrections screen --groups comeback`): −0.00082 [−0.00181,
+0.00007], 3/4.

Inside the joint correction (`joint_variants --variants with-comeback
with-comeback-twin`, both declared before scoring, the `more-groups-deeper`
recipe plus one group, native forecasts 2025Q3–2026Q2, 11,905 races):

| Variant | Native log loss | Against twin | Against `more-groups-deeper` |
|---|---:|---:|---:|
| `more-groups-deeper` | 1.80506 | | |
| `with-comeback` | 1.80522 | +0.00001 [−0.00030, +0.00029], 2/4 | +0.00016 [−0.00018, +0.00051], 0/4 |
| `with-comeback-twin` | 1.80521 | | |

Null. The horse-state and residual groups already carry the horse's form
after a break. Seventh group to be real alone and null inside. Closed.

Outputs: `data/research/comebacks_099/` (logs);
`data/research/error_anatomy_20261110/joint_variants.parquet` (variant
corrections).
