# 048 Walk-forward selection of the win model's inputs

**Status, 1 November 2026:** the selection ran and the result is confirmed out
of sample. On four untouched development quarters, the 126 selected inputs
(`feature_sets/win_selected_048.txt`) improve race log loss over `win.txt`
by **0.0208 nats [0.0164, 0.0251]** at screen capacity and by **0.0194
[0.0155, 0.0236]** at production capacity, in 4 of 4 quarters each time.
Brier improves too. Development period only: the golden hold-out has not been
loaded.

**Adopted in config, not yet trained:** `config.FEATURE_ALLOWLIST` points at
`feature_sets/win_selected_048.txt` (it is not an FE key, so `fe_signature`
is unchanged). The store has not been rebuilt for the later RTV card backfill
and par pin, and the promoted model still uses its own 154 inputs. The next
`train.py` run will rebuild the store, then train on the 126.

**The store has moved on since this run.** Everything below was measured on
store `bde9317da2c3c963`, which carries Timeform cards only from June 2023.
Afterwards a parallel session made two changes:

- it backfilled the card for 2020-09 to 2023-06 (`feature_cards.parquet`,
  15:26);
- it pinned the time-figure par (`src/rtv_features.py`, 15:21).

So the store is stale again. The input names are unchanged, but the card's
training history and the `rtv_time_fig_*` values will change. After the next
rebuild, rerun `prepare` and then `confirm` to re-check the gain. That reuses
the confirmation quarters for verification, not for selection.

## Why the old list had to be re-derived

`feature_sets/win.txt` began on 2026-06-22 (`cd8e8f6`) as the top 150 of
about 1,018 columns by LightGBM **gain** from one pilot fit
(`scripts/model_lab.py --write-allowlist`). The cut-off was chosen from one
chronological split on a subsample: top-150 came within −0.0001 log loss of
the full model. After that the list was only edited by hand:

- a re-rank inside the already-cut list ("top-150 of 155");
- four course-dynamics features;
- the `sex` codes replaced (`e689d98`);
- five prior-market and non-finish inputs (033);
- five duplicates removed and some inputs renamed (047).

The problems with that process:

1. **It was ranked on a leaked store and never re-ranked.** On 06-22 the store
   still had the `trainer_freq` stablemate leak (fixed 06-26) and the
   scrape-time `sex` leak (fixed 10-13), and training still dropped
   non-finishers (fixed 10-14). Gain rewards a leaked column above all, which
   is how `sex_F`/`sex_H` got in. It also takes credit from every honest
   column correlated with the leaked one, so the rest of the ranking was
   distorted too.
2. **Gain from one in-sample fit** favours continuous, high-cardinality
   columns. It splits credit between correlated copies of the same thing,
   which is why exact duplicates survived until 047. It says nothing about
   generalisation.
3. **The cut-off rested on one split** with no interval, and it was read off
   the same test split.
4. **It cannot be reproduced.** No command regenerates the current file.
5. Better inputs built since then never got in. The 046 RTV-derived family
   improved native race log loss by 0.0177 and was left out because it is not
   an edge over the market.

## Objective

The objective is the model's own **out-of-sample race log loss**: the mean
negative log-probability given to each race's winner after the race's
probabilities are normalised. Race log loss is the likelihood of a one-winner
race and a strictly proper score. Unlike Brier it penalises a 2% horse priced
at 0.5%, which is where staking decisions are most sensitive. Race Brier is
reported alongside it as a guard.

The market is not an objective. Its race log loss on the same races is
reported only as a **leak tripwire**: if a public-form model beat the closing
price, the likelier explanation would be an input that reads the result.

## Protocol

`src/feature_selection.py` holds the primitives (with tests in
`tests/test_feature_selection.py`). `scripts/select_features.py` runs them.

1. **Universe.** Every eligible numeric store column
   (`get_feature_columns`, no allowlist), so exclusions such as current-race
   odds and quarantined sectional composites still apply. Development rows
   only (`development_mask`): the golden hold-out is never loaded.
2. **Hygiene.** This step uses no labels. It drops columns that are present
   in under 0.5% of rows, constant with no missing values, or bit-for-bit
   duplicates of another column. Where duplicates exist, the name already in
   `win.txt` is kept.
3. **Recursive elimination** on six selection quarters (2024Q1–2025Q2):
   - Each quarter is predicted by a model fitted on everything before it, with
     a 7-day purge (`expanding_period_folds`).
   - The learner is the production win recipe (soft-label distillation,
     recency weights, burn-in, `CLASSIFIER_PARAMS`) at the project's screen
     capacity (400 trees at 0.04).
   - A feature's importance is its mean |SHAP| **on the unseen quarter**. Each
     fold's importances are normalised to shares, then averaged across folds.
   - The weakest 20% are dropped while more than 250 features remain, then
     10% per step, down to 30. The smaller set is refitted each time, so a
     correlated feature can inherit credit its dropped twin was sharing.
4. **Stop rule.** The chosen set is the smallest step whose pooled race log
   loss is within one date-clustered standard error of the best step (the
   one-SE rule). The comparison is paired, because every step is scored on the
   same races. The best step itself is biased low by having been chosen.
5. **Confirmation** on four later quarters (2025Q3–2026Q2) that played no
   part in selection. It compares the chosen set, `win.txt`, the best step
   and every hygiene-kept feature. The deltas use a paired, date-clustered
   bootstrap. The comparison is then repeated at production capacity (early
   stopping and a refit, as `fit_as_of` does) for the chosen set and
   `win.txt`.

## Results

All runs use store `bde9317da2c3c963` (rebuilt for this experiment, with the
047 input fixes) and data `2ea0d385c37e8d1c`. The receipts are in
`data/research/feature_selection_048/`.

### Data and hygiene

The data is 615,394 development runners in 65,840 races, from 2021-05-15 (the
end of burn-in) to 2026-06-03. Of 1,139 eligible columns, hygiene dropped 12:

- 7 exact duplicates, such as `has_horse_glicko` = `has_horse_elo` and
  `horse_going_wr` = `horse_going_win_rate`;
- `region_UK`, which is constant;
- the 4 `rtv_pre_*` card columns, which exist only on live cards.

Every `win.txt` input survived.

### Elimination (selection quarters 2024Q1–2025Q2, 19,086 races)

`win.txt` scores **1.87837**. Selected rows of the 28-step path:

| Inputs | Race LL | vs best (SE) | vs `win.txt` | Race Brier |
|---|---|---|---|---|
| 1,127 (all) | 1.86005 | +0.00324 (0.00097) | −0.01832 | .09553 |
| 370 | 1.85825 | +0.00144 (0.00096) | −0.02012 | .09546 |
| 237 | 1.85768 | +0.00088 (0.00093) | −0.02069 | .09543 |
| 173 | 1.85804 | +0.00123 (0.00086) | −0.02033 | .09549 |
| **126** | **1.85680** | best | **−0.02156** (0.00171) | .09543 |
| 113 | 1.85851 | +0.00170 (0.00077) | −0.01986 | .09551 |
| 75 | 1.85942 | +0.00261 (0.00085) | −0.01895 | .09553 |
| 50 | 1.86283 | +0.00603 (0.00101) | −0.01554 | .09569 |
| 30 | 1.86833 | +0.01152 (0.00128) | −0.01004 | .09588 |

The curve is flat to within about 0.002 from 75 to 1,127 inputs, then climbs
steadily below 75. The best step is 126. No smaller step is within one SE of
it, so the one-SE rule also picks 126. The step SEs reflect sampling noise in
the races, not fit-to-fit noise, so 126 is partly a lucky dip in a flat
region (see the noise control below). That is why the confirmation matters.

### Confirmation (2025Q3–2026Q2, 11,901 races, screen capacity)

These quarters took no part in selection. Deltas are against `win.txt`, with
paired, date-clustered 95% intervals:

| Arm | Inputs | Race LL | Δ race LL | Quarters better | Δ race Brier |
|---|---|---|---|---|---|
| `win.txt` | 149 | 1.86096 | — | — | — |
| **selected** | **126** | **1.84017** | **−0.0208 [−0.0251, −0.0164]** | **4/4** | −0.00078 [−0.00102, −0.00053] |
| selected, elimination column order | 126 | 1.83898 | −0.0220 [−0.0263, −0.0177] | 4/4 | −0.00085 |
| all | 1,127 | 1.84100 | −0.0200 [−0.0241, −0.0159] | 4/4 | −0.00076 |

Per quarter, the selected set beats `win.txt` by 0.0166 (2025Q3), 0.0227
(2025Q4), 0.0217 (2026Q1) and 0.0232 (2026Q2). The gain measured during
selection, 0.0216, holds out of sample.

Two head-to-head comparisons on the same races:

- **Selected vs all: −0.0008 [−0.0031, +0.0014].** Pruning 1,127 inputs to
  126 costs nothing and makes fits about 9× cheaper (35 s vs 235 s per
  quarter), but it adds no measurable strength by itself.
- **Noise control: −0.0012 [−0.0029, +0.0006].** The same 126 inputs in a
  different column order change LightGBM's column sampling. That alone moves a
  quarter by up to 0.004 and the pooled result by 0.001. Differences under
  about 0.002 between candidate sets are fit noise.

The model still trails the closing market by 0.095 nats (down from 0.116 for
`win.txt`), so the leak tripwire never fired.

### Confirmation at production capacity

The same quarters and races, refitted with `CLASSIFIER_PARAMS` (lr 0.008, up
to 4,000 trees), the production purged early-stopping split and a refit at
the selected tree count, as `fit_as_of` does. Calibrators are not refitted.
Both arms use the identical recipe.

| Quarter | Races | `win.txt` | Selected | Δ |
|---|---|---|---|---|
| 2025Q3 | 3,533 | 1.88923 | 1.87508 | −0.0142 |
| 2025Q4 | 3,234 | 1.83831 | 1.81628 | −0.0220 |
| 2026Q1 | 2,592 | 1.79543 | 1.77692 | −0.0185 |
| 2026Q2 | 2,542 | 1.88307 | 1.85878 | −0.0243 |
| **Pooled** | **11,901** | **1.85365** | **1.83424** | **−0.0194 [−0.0236, −0.0155]** |

Race Brier improves by 0.00076 [0.00053, 0.00097]. The gap to the market
narrows from 0.109 to 0.089 nats. Fits are also faster: 6.6–8.6 minutes per
quarter against 9.2–10.3 minutes for `win.txt`. The screen-capacity gain
(−0.0208) carries over to the production learner almost unchanged.

### Where the gain comes from

About 42% of the selected set's out-of-sample |SHAP| sits on 59 inputs that
`win.txt` never had:

| Family | Inputs | Share of importance | New inputs | Share from new |
|---|---|---|---|---|
| Ratings (Elo/Glicko/TrueSkill) | 23 | 29.1% | 6 | 8.3% |
| Horse form and race context | 43 | 27.3% | 18 | 9.4% |
| Trainer/jockey | 24 | 12.3% | 9 | 3.4% |
| Timeform card (current race) | 2 | 10.1% | 2 | 10.1% |
| Official rating/handicap | 8 | 5.8% | 4 | 2.6% |
| Prior-race market | 2 | 4.7% | 0 | 0 |
| RTV sectional/stride history | 11 | 4.4% | 11 | 4.4% |
| Speed figures | 4 | 2.5% | 4 | 2.5% |
| Pedigree | 4 | 1.9% | 0 | 0 |
| ATR sectional history | 5 | 1.8% | 5 | 1.8% |

The largest new contributions are:

- `rtv_card_tf_stars` and `rtv_card_tf_tipped`, the Timeform card from 046;
- `horse_margin_elo_rank` and `margin_elo_edge`;
- `base_win_rate`;
- relative speed figures;
- RTV time figures.

Of `win.txt`'s 149 inputs, 82 were dropped. They include:

- both `trainer_freq_*` inputs, chosen while that feature was leaking;
- most of the `*_vs_field` / `*_z_in_race` / `*_vs_best_in_race` triplets for
  trainer and jockey rates;
- several Glicko variants;
- `handicap`, `race_type_Chase` and `field_std_or`.

All of these families are priced by the market (ledger §2b, §2g and 046), so
none of this is evidence of an edge. These inputs make the model more
accurate, which is the objective here.

### Leak checks on the selected inputs

- **Synthetic fixture** (`scripts/audit_prediction_inputs.py --features
  feature_sets/win_selected_048.txt`). History truncated before a cutoff, full
  history, future results changed, future rows appended, the current day
  pending or with its results swapped, and row order permuted: **no input
  changes**. The only flag is `horse_prior_market_logrel_3_vs_field` under a
  change to every price. It is built from earlier races' prices, so this is
  expected, and `win.txt` shows the same flag. 30 inputs are constant in the
  fixture: all ATR and RTV inputs, which come from caches, plus some race-level
  ones. They are covered below.
- **ATR** (`scripts/audit_atr_invariance.py`, five selected ATR inputs):
  passes. Both controls fire.
- **RTV** (`tests/test_rtv_features.py` and
  `tests/test_rtv_feature_collection.py`): 23 tests pass. They cover current
  race and publication-lag exclusion, future and current labels, prices and row
  order, prefix and append parity, and pre-decision card snapshots. 046 also
  showed that the backfilled card matches the evening-before snapshot and adds
  nothing over the T-5 market, which post-race information would.
- **Current-race price:** `tests/test_market_free_inputs.py` passes for every
  eligible column.

### Serving note

The Timeform card carries 10% of the importance. Live rows read it from the
pre-decision RTV snapshot (046). If the snapshot is missing for a race, the
card inputs are NaN at serving time, as they are for every pre-June-2023 row
the model learns from. The model then degrades towards the
no-card fit rather than failing. Keep the RTV card collection running before
the off.

## Re-running

```
python -m scripts.select_features prepare      # after any store rebuild
python -m scripts.select_features eliminate    # about 4 h on 8 threads; resumable
python -m scripts.select_features confirm
python -m scripts.select_features confirm --capacity native --arms selected
```

`eliminate` rewrites `feature_sets/win_selected_048.txt`. Any later
selection needs fresh confirmation quarters: 2025Q3–2026Q2 have now been
used once.
