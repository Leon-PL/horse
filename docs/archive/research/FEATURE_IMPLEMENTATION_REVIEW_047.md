# 047 Implementation review of the 154 model inputs

1 November 2026. Development period only; the golden hold-out (from
2026-06-11) has not been touched. Store `1c83cfaeac51bb2e` (includes the 045
ratings and 046 RTV build).

The promoted recipe (`feature_sets/win.txt`, run `20261030_135650_092286`)
uses 154 of the store's 1,236 columns. This review read the code behind each
of them, checked suspected faults against the stored values, and measured
fixes with the project's standard screen. The skill ratings themselves were
repaired in 045 and are not reviewed again; the transforms built on top of
them are.

## What the model leans on

Win-head gain is concentrated: `horse_glicko_rank` 16.9%,
`horse_margin_elo_momentum_vs_best_in_race` 7.1%,
`connections_relative_score` 4.4%, `trainer_elo_vs_field_vs_best_in_race`
3.2%, `horse_prior_market_logrel_3_vs_field` 3.0%, `num_runners` 2.8%,
`base_win_rate` 2.4%. About 60 of the 154 inputs come from one function,
`add_race_relative_signal_features`, which turns a base column into
`_vs_field`, `_z_in_race`, `_rank_pct_in_race` and `_vs_best_in_race`.

## Verdict

The code has real encoding faults, listed below, but **fixing them does not
make the model more accurate**. Ten repair sets were each run on four
development quarters. Every pooled change in race log loss is within ±0.0013
nats, and every 95% interval spans zero. The same screen measured the 046
RTV/Timeform inputs at −0.0187, so it can detect a real gain. LightGBM already
works around these faults through other columns. For example,
`horse_prev_races` and `prior_form_runs` separate debutants from horses whose
days or position were filled with a placeholder. More accuracy has come from
new information (046), not from re-encoding what the store already holds.

## Faults found

| # | Fault | Where | Evidence on the store (all 701,848 development rows) |
|---|---|---|---|
| 1 | Missing base values become 0 in `_vs_field` / `_vs_best_in_race`. For `_vs_best_in_race`, 0 means equal to the best runner. | `add_race_relative_signal_features` | `field_avg_elo_10/ewma` are missing for the 9.9% of rows that are debutants. Those debutants supply 48% of the zeros ("best") in `_vs_best_in_race`, and they win 8.4% against 14.5% for the true best. `speed_fig_best_5` is missing for 10.4% of rows and `speed_fig_adj_best_5` for 31.1%, so those rows are coded as field average. |
| 2 | `_vs_best_in_race` is always 0 for the leader, so the model cannot see the leader's margin over the next runner. | same | Applies to 28 inputs, including the #2 and #4 by gain. |
| 3 | Cold starts get placeholder values: debutants `days_since_last_race = 60`, never-won horses `days_since_last_win = 365`. | `add_horse_features` | 45% of raced rows are maidens coded 365. Maidens win 8.8%; horses whose last win was 300–430 days ago win 10.9%; over 730 days, 8.4%. |
| 4 | Form positions are raw and ignore non-finishes. | `add_horse_features` | Last run a non-finish and `avg_pos_3` ≤ 2: 11.9% win. Finished with the same `avg_pos_3`: 21.1%. Correlation of `avg_pos_3` with field size: 0.20. |
| 5 | Unrated runners get the race-median OR (data_processor), and the OR rank, second-top and spread are computed over those stand-ins. | `data_processor.py`, `add_field_quality_features` | 97,095 rows (12.9%), in 51% of non-handicap races. |
| 6 | Jockey/trainer going, trip, class and track rates are raw, and 0 when the entity has ≤2 runs. "No data" then looks like "never wins". | `add_cross_entity_features`, `add_trainer_specialisation_features`, track features | 23–37% of the zeros are thin data. |
| 7 | Prize change is a raw £ difference; the purse distribution is heavily skewed (median £5.2k, 99th percentile £94k). | `add_prize_money_features` | — |
| 8 | `draw / num_runners` uses runners after non-runners. | `add_race_context_features` | Exceeds 1 in 6.8% of drawn rows. |
| 9 | Exact duplicates for a tree: `base_win_rate` ≡ `num_runners`, `days_log` ≡ `days_since_last_race`, `x_vs_field_vs_field` ≡ `x_vs_field` (twice), `x_vs_field_vs_best` ≡ `x_vs_best`. | relative transform, race context | Rank correlation 1.0000. |

Smaller observations, not tested:

- **Speed figures.** The `speed_fig_*` family is lengths behind per furlong^0.6, so every winner scores 0 and the winning margin is lost. The class penalty is a fixed fudge, and Irish races have no class number.
- **Place rates.** These count the top three whatever the field size.
- **`jockey_avg_pos_20`.** It has the same raw-position fault as item 4.
- **`horse_dist_cat_wr_shrunk`.** It shrinks to a fixed 0.10, not to the horse's own overall rate.
- **`track_draw_win_rate`.** It pools every trip at a track, and its correlation with 1/n is 0.25.
- **`weight_per_furlong`.** It is mostly a stand-in for distance, which is not otherwise among the 154 inputs.
- **`field_avg_elo`.** `add_opposition_strength_features` builds it excluding the horse itself; `add_field_quality_features` later overwrites it including the horse. The rolling columns use the first version, and the docstring describes only that version.

## Measured repairs

Script: `scripts/impl_review_variants.py` / `_b.py` build the repaired columns;
`scripts/ablate_impl_review.py` / `_b.py` run the screen. Every original that a
repair replaces was first recomputed by the same code and matched the store
on 100% of rows. Receipts are in `data/audit/feature_impl_review_047{,b}/`.

The screen is `quick_win_feature_screen`: 400 trees, a two-year training
window, one seed and simple race normalisation. It ran over 2025Q3–2026Q2
(11,901 races; baseline race log loss 1.86027). Each candidate replaces
baseline inputs in place.

| Candidate | Replaces | Pooled race LL change [95% date-cluster] | Quarters better | Brier change |
|---|---|---|---|---|
| relative_nan (fault 1) | 5 | +0.00036 [−0.00074, +0.00149] | 2/4 | −0.000004 |
| relative_loo (faults 1+2: "vs best other runner") | 31 | +0.00087 [−0.00065, +0.00248] | 1/4 | +0.000076 |
| cold_start (fault 3: missing, not 60/365) | 7 | −0.00056 [−0.00176, +0.00067] | 3/4 | +0.000010 |
| form_npos (fault 4: field-share position, DNF = last) | 5 | +0.00054 [−0.00075, +0.00182] | 2/4 | +0.000053 |
| prize_draw (faults 7+8) | 2 | +0.00031 [−0.00096, +0.00146] | 2/4 | +0.000040 |
| all of round 1 | 43 | +0.00081 [−0.00092, +0.00246] | 1/4 | +0.000086 |
| or_observed (fault 5) | 3 | −0.00014 [−0.00136, +0.00103] | 2/4 | +0.000004 |
| shrunk_dims (fault 6: shrink to the entity's own rate) | 14 | −0.00011 [−0.00155, +0.00132] | 2/4 | −0.000023 |
| rank_share (ratings rank as share of field) | 2 | +0.00003 [−0.00160, +0.00165] | 1/4 | +0.000014 |
| dedupe (fault 9: drop the 5 duplicates) | −5 | +0.00130 [−0.00032, +0.00307] | 1/4 | +0.000016 |

Negative is better. Removing the duplicates is slightly worse, not neutral: under
column subsampling a duplicated signal is drawn more often, so the copies act
as extra weight on it.

Limits: one seed and a small capacity, scoring the win head only, accuracy rather than
market edge. A native-recipe check was not run because nothing passed the
screen.

## What to do with it

- Do not expect these repairs to improve accuracy.
- The pending retrain with the 046 inputs (`feature_sets/win_rtv_derived.txt`,
  natively −0.0177) is where the measured gain is.

## Fixes applied (1 November 2026, code only)

The user asked for every fault to be fixed. The fixes are in
`src/feature_engineer.py` and the three live feature lists. **The store has
not been rebuilt.** `fe_signature` has changed, so the current store and the
promoted model are stale until the next rebuild and retrain. The screen
results above measured the repairs on reconstructed columns, not on a rebuilt
store.

| # | Fix |
|---|---|
| 1 | `add_race_relative_signal_features` keeps NaN in `_vs_field`, `_z_in_race`, `_rank_pct_in_race`, `_vs_best_in_race` and the context transforms. |
| 2 | `_vs_best_in_race` is the gap to the best *other* runner, so the sole leader gets a positive margin. Tied leaders get 0. |
| 3 | `days_since_last_race`, `days_log` (debutants) and `days_since_last_win` (maidens) are NaN. The same applies to the other history placeholders: `distance_change`, `weight_change_vs_field`, debutant prize history, `horse_avg_lb_behind` (was 3.0), and rolling-form fills (std 3.0, rates 0). Every raw rate over zero runs is NaN (`_rate`), and `_bayesian_shrink` treats that as "prior only". |
| 4 | Horse, jockey and target-encoded positions use a new `_norm_pos_for_history`: the share of the field beaten, with a non-finish counted as last. `horse_avg_position` now uses it, as do `horse_avg_pos_*`, best/worst/consistency, `horse_pos_stddev/range_*`, `jockey_avg_pos_*` and the `*_te_norm_pos` / `_te_ewma` encodings. `elo_x_recent_form` is `elo × (1 − avg_pos_3)`. |
| 5 | Every OR-derived field feature uses `observed_rating`, so race-median stand-ins are NaN. This covers `field_avg_or`, `field_std_or`, `or_rank_in_race`, the second-top gap and range, `or_vs_top`, `or_x_handicap`, `or_vs_field_x_hcap`, and the stablemate OR rank and gap. The base `official_rating` column is unchanged. |
| 6 | The jockey/trainer track, going, trip and class rates are shrunk toward the entity's own `*_win_rate_shrunk`, with 10 pseudo-runs and no ≤2-run cut-off (`_shrink_to_entity`). Column names are unchanged. `track_draw_win_rate` is NaN below 6 runs (was 0). |
| 7 | `prize_money_change_vs_avg = log1p(prize) − log1p(average)`, NaN for an unknown (0) purse. |
| 8 | `draw_pct = draw / highest stall drawn in the race`. |
| 9 | `base_win_rate`, `days_log`, `trainer_elo_vs_field_vs_field`, `horse_margin_elo_momentum_vs_field_vs_field` and `horse_elo_momentum_vs_field_vs_best_in_race` are out of `win.txt`, `win_rtv_derived.txt` and `win_with_rtv.txt`. Relative transforms now take raw bases (`trainer_elo`, `combined_elo`), so three inputs were renamed: `*_vs_field_z_in_race` → `*_z_in_race` and `*_vs_field_vs_best_in_race` → `*_vs_best_in_race`. `win.txt` has 149 inputs. |
| — | `field_avg_elo` is no longer overwritten by the including-self mean in `add_field_quality_features`. It is now the excluding-self mean that its rolling columns use. |

Points to check at the retrain:

- Removing the duplicates measured +0.0013 at fixed `feature_fraction`, so
  re-tune it.
- The historical numbered feature sets (`*_030` … `*_035`, `research_*`)
  still name the old columns, so they would warn if reused.
- `scripts/impl_review_variants*.py` reproduce the old store. Their
  reproduction checks will fail after the rebuild.

Not changed: speed figures (winners 0), top-3 place rates, the fixed 0.10
prior of `horse_dist_cat_wr_shrunk`, `track_draw_win_rate` pooling every trip
at a track, `weight_per_furlong`. These are design choices rather than faults.
