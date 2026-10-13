# Feature-Engineering Taxonomy

A map of every `engineer_features` stage (`src/feature_engineer.py`) by **how it
computes**, because that — not what it means — determines what an *append* needs
and whether "build once, append forever" is possible.

This is the reference that makes the incremental migration estimable: convert
stages in class order, cheapest first, each behind a parity test.

## The six classes

| Class | Name | Depends on | Append needs | Cost to make incremental |
|---|---|---|---|---|
| **R** | Row-local | the row's own fields (+ static/external joins) | nothing | **free** (recompute per new row) |
| **W** | Within-race | all rows of the *current* race only | nothing | **free** (recompute per new race) |
| **L** | Lag / prev-value | the entity's previous row | last row per entity | tiny (degenerate B, N=1) |
| **B** | Bounded window | entity's last *N* races / last 365d | a bounded per-entity buffer | small |
| **C** | Cumulative | entity's *whole* prior history | a per-entity running accumulator | medium (ratings = template) |
| **F** | Forward-guarded | a future row, but date-guarded | bounded history | safe; flag explicitly |
| **G** | Global cross-time | the *entire dataset* | — | **forbidden** (breaks append + leaks) |

Correctness note: R/W/L/B/C/F are all pure functions of data **at-or-before** the
row's event time, so a settled row's features never change as later races
arrive — the invariant that makes append-only valid. **G** is the only class
that breaks it; an audit of the current pipeline found **none** (see the
append-safety test). Keep it that way.

## Stage-by-stage

Pipeline order (per `engineer_features` `stages` list). "Classes" lists every
pattern the stage uses; the **first** is dominant.

| Stage | Classes | Incremental state to carry |
|---|---|---|
| `add_horse_features` | C, B | per-horse: career counts/wins/places/pos-sum; rolling-N form buffer |
| `add_pedigree_features` | C | per-sire/dam/damsire: runs/wins/places (race-safe) |
| `add_medical_features` | L | per-horse: last gelding flag |
| `add_prize_money_features` | C | per-horse: cum prize, running-max prize |
| `add_collateral_form_features` | **F** | bounded history (rivals' next-out, date-guarded); *off by default* |
| `add_jockey_features` | C, B | per-jockey: career counts; rolling-N + 14/30d calendar buffers |
| `add_trainer_features` | C, B | per-trainer: race-safe counts; rolling-N + 14/30d buffers |
| `add_jockey_trainer_features` | C | per-(jockey×trainer) and per-trainer combo counts |
| `add_jockey_track_features` | C | per-(jockey×track) counts |
| `add_track_features` | C | per-(horse×track / horse×dist / horse×going) counts |
| `add_course_distance_features` | C | per-(horse×track×dist) counts |
| `add_class_change_features` | L | per-horse: previous class |
| `add_distance_change_features` | C, L | per-(horse×dist-cat) counts + running-max; prev distance |
| `add_target_encoded_features` | C, B | per-entity TE accumulators (lifetime) + 365d/EWMA buffers |
| `add_cross_entity_features` | C | per-(jockey×going, horse×going, trainer×class…) counts |
| `compute_elo_features` | C | per-horse Elo state — **already incremental** (`return_state`) |
| `compute_glicko_features` | C | per-horse Glicko/TrueSkill state — **already incremental** |
| `add_opposition_strength_features` | W, B | within-race elo aggregates; per-horse rolling of opposition |
| `add_headgear_features` | C, L | per-horse: ever-had-headgear; prev headgear |
| `add_surface_features` | C, L | per-(horse×surface) counts; prev surface |
| `add_trainer_specialisation_features` | C | per-(trainer×dist-cat, trainer×going) counts |
| `add_speed_figure_features` | B, W | per-horse rolling speed; within-race + per-class(backward) medians |
| `add_market_features` | R, W | none (odds math + per-race normalisation) |
| `add_trainer_runner_intent_features` | W | none (per-(race×trainer) stable pecking order) |
| `add_jockey_intent_features` | L | per-horse: prev jockey/elo/favourite/result |
| `add_race_context_features` | W, C, R | within-race ranks; per-(track×draw-third) counts; draw_pct |
| `add_pace_style_features` | W, B | within-race pace aggregates; per-horse weighted history |
| `add_field_quality_features` | W | none (field OR/Elo aggregates) |
| `add_track_config_features` | R | none (static venue reference + draw interactions) |
| `add_supplementary_features` | B, C, L | per-horse rolling consistency/class; best-class cummin; prev trainer |
| `add_race_relative_signal_features` | W | none (within-race relative families of base signals) |
| `add_interaction_features` | R, W | none (products of existing cols + a race avg) |
| `apply_conditional_feature_masks` | R | none (zero-out irrelevant features row-wise) |

## What this implies for the migration

Counting dominant classes:

- **Free now (R/W): ~11 stages** — recompute per new race, zero state. These can
  run on *just the new races* immediately (no history needed).
- **Cumulative (C): ~15 stages** — the bulk. Each needs a per-entity accumulator
  dict, exactly like the Elo/Glicko state already persisted by `FeatureStore`.
  Generic enough to share one "counts accumulator" helper across most of them.
- **Bounded window (B) + Lag (L): ~10 stages** — need a bounded per-entity buffer
  (the last *N* races / 365 days, or just the last row). One ring-buffer helper
  covers them.
- **Forward-guarded (F): 1 stage** (collateral form, off) — keep, flag, test.
- **Global (G): 0** — must stay 0; enforced by `tests/test_append_safety.py`.

So the append engine needs three reusable state objects — **counts accumulator**
(C), **bounded buffer** (B/L), and the existing **rating state** (C-ratings) —
plus the stateless R/W stages that already work on any slice. That is the whole
job, and it is additive: until a stage is converted it simply falls back to
"recompute over history", which is what happens today.
