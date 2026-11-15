# Features from ATR Jumps Data and Distance Ran — 6 November 2026

Implemented **11 completed-run descriptors, 17 historical predictors and four
quality measures** in `src/research/atr_new_tabs_features.py`. They are a separate
candidate pack, alongside the earlier rich-sectional candidates. No selected
recipe, ordinary feature store or model was changed. The running five-tab
collector was left untouched.

## The historical predictors

Unless stated otherwise, a median uses the last three eligible canonical race
appearances. Races without usable telemetry still occupy a slot.

| Predictor | Definition / interpretation |
|---|---|
| `atr_jump_loss_pct_median_3` | Typical whole-race percentage of speed lost jumping; lower loss is better fluency |
| `atr_jump_early_loss_pct_median_3` | Typical loss in the first third |
| `atr_jump_late_loss_pct_median_3` | Typical loss in the final third |
| `atr_jump_late_change_pp_median_3` | Late minus early loss, in percentage points; positive means more loss later, not a clinical fatigue diagnosis |
| `atr_jump_phase_range_pp_median_3` | Highest minus lowest loss across all three phases; requires all three |
| `atr_jump_consistency_pct_median_3` | ATR's published within-race consistency percentage; stability does not imply good jumping |
| `atr_jump_grade_rank_median_3` | Native fluency grade, ordinal A=0 through E=4; distinct from sectional efficiency |
| `atr_jump_vs_course_par_pp_median_3` | Average jumping loss minus the published course-winner benchmark; negative means less loss than that benchmark |
| `atr_path_extra_lengths_median_3` | Published signed extra distance relative to the source race median, in lengths |
| `atr_path_extra_pct_median_3` | `100 × (runner yards / race-median yards − 1)` |
| `atr_path_above_median_rate_3` | Fraction of measured appearances with positive published extra lengths |
| `atr_jump_loss_best_3` | Minimum observed average jumping loss in the window |
| `atr_jump_grade_ab_rate_3` | Fraction of available grades that were A or B |
| `atr_jump_loss_mad_5` | Between-race median absolute deviation of loss over the last five eligible appearances; needs three observations |
| `atr_path_extra_mad_5` | Same repeatability measure for extra lengths |
| `atr_jump_loss_recent_change` | Latest eligible race's average loss minus the median of the preceding four slots; needs two measured earlier races |
| `atr_path_extra_recent_change` | Same recent-change construction for extra lengths |

Two quality pairs accompany these predictors:

- `atr_jump_observed_runs_3` / `atr_path_observed_runs_3`: actual contributing
  appearances, from zero to three.
- `atr_jump_observation_age_days` / `atr_path_observation_age_days`: age of the
  newest contributing observation within that window; unknown when none exists.

The source grade, average loss and course-relative loss overlap. The two distance
representations overlap too. These are alternative/related encodings to compare,
not seventeen independent discoveries or evidence of a prediction improvement.

## Missingness and point-in-time rules

1. Canonical horse/race IDs identify history. Shared display names do not link
   horses; duplicate/contradictory appearances fail validation.
2. A source race must be at least 48 hours earlier than the target off, strictly
   beyond that boundary. This is a conservative publication **assumption**, not
   proof of original publication or the absence of later telemetry revisions.
3. Native grades and course benchmarks additionally require their dated revision
   day to have ended before the target off. A missing revision date leaves those
   historical inputs unknown. Each cell is masked at the target's time, so a
   late revision of an old race does not replace more recent appearances.
4. Measurements require a verified finisher. Falls/pull-ups remain history slots
   with missing metrics; they cannot look efficient merely because they covered
   less distance. Their completion/fall history remains available elsewhere in
   the project. These measurements describe completed-run quality, not the
   probability of completing a race.
5. Loss/consistency must be finite percentages within 0–100. Unknown grades,
   unsupported ungraded-zero placeholders and nonpositive course benchmarks
   are not made into good performances.
6. Distance comparisons require positive runner and median distances. ATR can
   supply a default zero difference without either distance; that is not usable
   path evidence. Published lengths retain their own precision: a tiny positive
   length difference can coexist with equal rounded yard totals. No fixed
   yards-per-length conversion is invented.

Between-race repeatability differs from ATR's within-race consistency. Median
absolute deviation is not reported as zero from a single observation. Recent
change also remains missing when its latest eligible race is untracked; an older
good run is not substituted for it.

## Generated dataset and current coverage

The frozen snapshot contains **4,298 source runners**, all matched to canonical
identities, and **67,233 canonical appearances** for those horses. The exported
historical inputs are populated on **8,294 appearances**. Targets include races
without current telemetry; a prediction does not require the target's result.
This is coverage in the partially collected cohort, not the whole project.

| Family | Target appearances with a usable input |
|---|---:|
| Jumping loss / phase / native consistency | 1,467 |
| Dated fluency grade / course comparison | 1,000 |
| Extra-distance level / frequency | 7,818 |
| Jumping repeatability / recent change | 11 / 5 |
| Distance repeatability / recent change | 47 / 18 |

The repeatability/change columns are especially sparse while the historical
backfill is incomplete. They need more source history before a meaningful
predictive comparison. The initial wider export also included source rows with
only default distance differences; the final cohort excludes those metadata-only
responses. Missing canonical appearances within each retained horse's history
are still included.

Files under `data/research/atr_new_tabs_20261106/`:

- **`prior_candidates.parquet`**: canonical IDs/time and the 21 prediction inputs.
- `feature_names.json`: the predictor and quality-column lists.
- `summary.json`: source/feature coverage, rules, hashes and verification.
- `history_inputs.parquet`: frozen canonical history and raw inputs for replay.
- `run_metrics.parquet`: eleven post-race descriptors for diagnosis.

Only the **prior candidates** are prediction inputs. Frozen raw history and run
metrics contain post-race information and must not be fed directly into a model.
Re-running the command produces an updated snapshot as collection progresses:

```powershell
.venv\Scripts\python.exe scripts/derive_atr_rich_features.py --family new-tabs --output data/research/atr_new_tabs_20261106
```

The default `--family rich` retains the previous candidate generator. Source
access is read-only; the command neither restarts the collector nor replaces its
accepted historical cache. Feature-code hashes and the exact inputs are retained
so a later experiment can reproduce the matrix despite ongoing source collection.

## Validation and evaluation boundary

All **57 relevant checks passed**, including the previous rich-feature pack,
the new formulas, grades/revisions, British summer time, publication boundaries,
missing appearances, nonfinishers, identity isolation, current/future-label
invariance, batching, exports and feature-signature coverage. The exported run
and prior matrices were also reproduced exactly from their frozen inputs.
Evidence is under `data/audit/atr_new_tab_features_20261106/`.

The [negative-results ledger](../NEGATIVE_RESULTS.md), particularly 025 and 046,
contains failed earlier ATR and RTV family additions. New source fields do not
overrule those results. This pack is implemented and materialised, **unselected
and not predictively evaluated**. A focused walk-forward comparison with a
matched noise baseline is needed before selecting it for a model or claiming
value beyond the market. No training, promotion or full feature-store rebuild
was performed here.
