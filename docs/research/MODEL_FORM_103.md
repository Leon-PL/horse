# Elastic net and linear trees against LightGBM (research 103, 15 November 2026)

Question: on the 87 selected inputs, how far is a linear model behind
LightGBM, and do LightGBM's linear trees (`linear_tree=True`) help?

`research/model_form_103/run.py`. Same matrix, inputs and expanding purged
folds as `src.modelling admit` (2025Q3–2026Q2, 11,632 races). The reference
is the screen-capacity LightGBM baseline of research 100 (400 trees at
0.04). Differences are paired by race, with a bootstrap over race days.

- **Elastic net:** a race-wise conditional logit (softmax over the field).
  Inputs are winsorised at 0.5/99.5%, median-filled with missing flags (128
  terms) and standardised on each fold's training rows. The penalty is chosen
  on the first fold's training rows (last 15% of dates held out, purged):
  λ 0.001, α 0.1, about 105 of 128 terms non-zero.
- **Linear trees:** the screen recipe with `linear_tree=True`,
  `linear_lambda=1.0`.

| Arm | Race log loss | Against LightGBM | Folds better |
|---|---:|---:|---:|
| LightGBM (screen) | 1.82894 | | |
| Elastic net | 1.85155 | +0.02261 [+0.01836, +0.02680] | 0/4 |
| Linear trees | 1.82838 | −0.00056 [−0.00243, +0.00136] | 3/4 |

Per fold, linear trees: +0.00183, −0.00276, −0.00126, −0.00010.

## Reading

- The trees capture about 0.023 of non-linear structure and interactions
  that a linear model cannot. The elastic net is no substitute; as a blend
  member it is far behind.
- Linear trees are a tie at screen capacity: right sign in 3 of 4 folds,
  within noise. Native capacity (early-stopped, more trees) is the
  only setting that could resolve it. Not adopted.

Outputs: `data/research/model_form_103/` (`elastic.parquet`,
`linear_tree.parquet`, `compare.json`, logs).
