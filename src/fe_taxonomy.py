"""Machine-readable feature-engineering stage taxonomy.

The prose version is ``docs/FE_TAXONOMY.md``; this is the enforced one. Each
``engineer_features`` pipeline stage maps to its **dominant compute class**,
which determines how it becomes incremental in an append (build-once /
append-forever; see ``docs/FE_REFACTOR_PLAN.md``).

``tests/test_fe_taxonomy.py`` keeps this in lock-step with the actual pipeline:
adding or removing a stage without classifying it fails the suite, and no stage
may be class ``G`` (whole-dataset / append-breaking / leaky).
"""

from __future__ import annotations

# Compute class -> what an incremental append needs for that stage.
CLASS_MEANINGS: dict[str, str] = {
    "R": "row-local: pure function of the row's own fields (+ static/external joins); no state",
    "W": "within-race: needs all rows of the current race only; no cross-race state",
    "L": "lag: needs the entity's previous row (degenerate bounded buffer, N=1)",
    "B": "bounded-window: needs the entity's last N races / last 365 days",
    "C": "cumulative: needs the entity's whole prior history (a running accumulator)",
    "F": "forward-guarded: references a future row but is date-guarded, so append-safe",
    "G": "global cross-time: depends on the whole dataset — FORBIDDEN (breaks append + leaks)",
}
VALID_CLASSES = set(CLASS_MEANINGS)
FORBIDDEN_CLASSES = {"G"}

# Dominant class per pipeline stage. (Many stages mix classes — see the doc for
# the full breakdown; the dominant class drives the append strategy.)
STAGE_TAXONOMY: dict[str, str] = {
    "add_horse_features": "C",
    "add_hidden_form_features": "R",
    "add_pedigree_features": "C",
    "add_medical_features": "C",  # wind_ops_prior cumulative (gelding part is L)
    "add_prize_money_features": "C",
    "add_collateral_form_features": "F",
    "add_jockey_features": "C",
    "add_trainer_features": "C",
    "add_jockey_trainer_features": "C",
    "add_jockey_track_features": "C",
    "add_track_features": "C",
    "add_course_distance_features": "C",
    "add_class_change_features": "L",
    "add_distance_change_features": "C",
    "add_target_encoded_features": "C",
    "add_cross_entity_features": "C",
    "compute_elo_features": "C",
    "compute_glicko_features": "C",
    "add_opposition_strength_features": "W",
    "add_headgear_features": "C",
    "add_surface_features": "C",
    "add_trainer_specialisation_features": "C",
    "add_speed_figure_features": "B",
    "add_atr_features": "B",
    "add_market_features": "R",
    "add_trainer_runner_intent_features": "W",
    "add_jockey_intent_features": "L",
    "add_race_context_features": "W",
    "add_pace_style_features": "W",
    "add_field_quality_features": "W",
    "add_track_config_features": "R",
    "add_course_dynamics_features": "C",
    "add_supplementary_features": "B",
    "add_race_relative_signal_features": "W",
    "add_interaction_features": "R",
    "apply_conditional_feature_masks": "R",
}

# Stages that need NO history at all — they can run on just the new races in an
# append (the free Phase-1 win). Derived from the taxonomy so it stays in sync.
STATELESS_CLASSES = {"R", "W"}


def stateless_stages() -> set[str]:
    """Stage names that depend only on the current race (classes R/W)."""
    return {name for name, cls in STAGE_TAXONOMY.items() if cls in STATELESS_CLASSES}


def stages_by_class() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {c: [] for c in VALID_CLASSES}
    for name, cls in STAGE_TAXONOMY.items():
        out[cls].append(name)
    return out
