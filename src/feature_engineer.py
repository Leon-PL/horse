"""Feature engineering: runs every feature stage over a processed frame, in order.

The stages live in ``src/features/`` (one module per family) and in the
source modules ``ratings``, ``atr_features``, ``rtv_features`` and
``bha_features``. ``engineer_features`` owns their order, the process
split and the final clean-up.
"""

import logging
import os

import numpy as np
import pandas as pd

import config
from src.bha_features import add_bha_features
from src.features.common import _add_result_history_flags, _defrag, _event_sort_key, _without
from src.features.connections import (
    add_cross_entity_features,
    add_jockey_features,
    add_jockey_intent_features,
    add_jockey_track_features,
    add_jockey_trainer_features,
    add_target_encoded_features,
    add_trainer_features,
    add_trainer_runner_intent_features,
    add_trainer_specialisation_features,
)
from src.features.history import counts_state_scope
from src.features.horse import (
    add_class_change_features,
    add_collateral_form_features,
    add_course_distance_features,
    add_distance_change_features,
    add_established_form_features,
    add_headgear_features,
    add_hidden_form_features,
    add_horse_features,
    add_prior_market_features,
    add_prize_money_features,
    add_speed_figure_features,
    add_supplementary_features,
    add_surface_features,
    add_track_features,
)
from src.features.pedigree import add_pedigree_features
from src.features.race import (
    add_course_dynamics_features,
    add_field_quality_features,
    add_interaction_features,
    add_market_features,
    add_opposition_strength_features,
    add_race_context_features,
    add_track_config_features,
    apply_conditional_feature_masks,
)
from src.features.relative import add_race_relative_signal_features
from src.features.sectionals import add_atr_features
from src.ratings import compute_elo_features, compute_glicko_features
from src.rtv_features import add_rtv_features
from src.source_dataset import dataset_build

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataset_build
def engineer_features(
    df: pd.DataFrame,
    save: bool = True,
    *,
    skip_ratings: bool = False,
    counts_state: dict | None = None,
    defer_global_imputation: bool = False,
    half: str | None = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: Processed DataFrame from data_processor
        save: Whether to save the feature-engineered data
        skip_ratings: When True, the Elo/Glicko rating *sweeps* are not run
            and the rating columns are expected to already be present on
            ``df`` (attached by the caller from persisted state). Used by
            the incremental feature store to avoid re-sweeping ratings over
            the full history on every append (the sweeps are ~47% of FE).
            Downstream rating-dependent stages then read those columns
            directly.
        half: Lets two processes share a build. ``"first"`` stops after
            ``add_atr_features`` and returns the working frame; ``"second"``
            takes that frame and finishes it. Each half carries its own
            ``counts_state``, so no entity may be tracked on both sides.

    Returns:
        DataFrame with all engineered features
    """
    if half != "second":
        logger.info(f"Starting feature engineering on {len(df)} records...")

        # Ensure race_date is proper datetime (CSV round-trip turns it to str)
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

        # Sort once — every sub-function needs chronological order for
        # cumsum / rolling / shift.  Sorting here avoids 12 redundant sorts.
        df["_event_dt"] = _event_sort_key(df)
        df = df.sort_values(["_event_dt", "race_id"]).copy()
        df = _add_result_history_flags(df)

    # Pipeline stages run in order; each must see the columns its
    # predecessors added (dependency notes kept beside the entries).
    # _defrag between stages keeps the BlockManager from fragmenting as
    # each stage inserts its columns one-by-one.
    stages = [
        add_horse_features,
        add_established_form_features, # same-scale marks and discipline evidence
        add_hidden_form_features,      # external/unseen prior form (foreign raiders)
        add_pedigree_features,         # sire/dam/damsire edge
        add_prize_money_features,      # class shifts / purse jumps
        add_collateral_form_features,  # prev-race rivals' next-out form
        add_jockey_features,
        add_trainer_features,
        add_jockey_trainer_features,
        add_jockey_track_features,
        add_track_features,
        add_course_distance_features,
        add_class_change_features,
        add_distance_change_features,
        add_target_encoded_features,   # smoothed entity baselines
        add_cross_entity_features,     # jockey×going, horse×going, etc.
    ]
    # Every stage sees the same event order (``_event_dt``: date plus off
    # time) whether the rating sweeps run here or the caller supplies ratings.
    if not skip_ratings:
        stages.append(compute_elo_features)
        if getattr(config, "GLICKO_ENABLED", True):
            stages.append(compute_glicko_features)  # rating + uncertainty (RD)
    stages += [
        add_opposition_strength_features,  # needs horse_elo
        add_headgear_features,
        add_surface_features,
        add_trainer_specialisation_features,  # needs dist_category
        add_speed_figure_features,
        add_atr_features,
        add_market_features,
        add_trainer_runner_intent_features,  # stable pecking-order / intent
        add_jockey_intent_features,        # needs is_favourite + jockey_elo
        add_race_context_features,
        add_field_quality_features,
        add_track_config_features,         # static venue data
        add_course_dynamics_features,      # course experience/handedness/pace-bias/travel
        add_supplementary_features,        # consistency, class, campaign, seasonal
        add_race_relative_signal_features,
        add_interaction_features,
        apply_conditional_feature_masks,   # zero irrelevant features
        add_prior_market_features,         # independent prior-day history, appended after existing stages
        add_rtv_features,                  # quality-filtered mature prior runs and observed pre-race context
        add_bha_features,                  # prior BHA reports under the configured availability policy
    ]
    if half is not None:
        handover = stages.index(add_atr_features) + 1
        stages = stages[:handover] if half == "first" else stages[handover:]
    _profile = os.environ.get("FE_PROFILE")
    # A seeded counts_state (FeatureStore append) makes class-C stages resume
    # from carried state instead of recomputing history; None = full build.
    with counts_state_scope(counts_state):
        if _profile:
            import time as _t
            _timings = []
            for stage in stages:
                _t0 = _t.perf_counter()
                df = stage(df)
                df = _defrag(df)
                _timings.append((stage.__name__, _t.perf_counter() - _t0))
            _timings.sort(key=lambda x: -x[1])
            _total = sum(t for _, t in _timings)
            logger.warning("FE stage profile (rows=%d, total=%.2fs):", len(df), _total)
            for _name, _sec in _timings:
                logger.warning("  %6.2fs  %5.1f%%  %s", _sec, 100 * _sec / _total, _name)
        else:
            for stage in stages:
                df = stage(df)
                df = _defrag(df)

    if half == "first":
        return df

    # Final consolidation before the NaN-handling block below.
    df = _defrag(df, max_blocks=0)

    # ── Feature-specific NaN handling ────────────────────────────
    # Prefer semantic defaults where they truly exist, and otherwise leave
    # missing values intact so the tree models can learn from cold starts
    # and unknown context directly.
    # Unknown speed history stays missing. Full-frame medians include future
    # results and make old rows change when history is appended; tree models
    # can use missingness directly. The legacy defer_global_imputation argument
    # remains accepted for callers, but no global imputation is performed.

    # Odds-based columns: fill with field median or neutral values
    odds_fill = {"implied_prob": 0.05, "norm_implied_prob": 0.05,
                 "log_odds": np.log1p(20.0), "odds_vs_field": 1.0,
                 "odds_rank": 5.0, "overround": 1.0,
                 "trainer_runner_rank_by_odds": 1.0,
                 "trainer_odds_vs_stablemate_best": 0.0,
                 "trainer_stable_market_share": 1.0,
                 "trainer_string_disagreement_or_odds": 0.0,
                 "trainer_string_disagreement_elo_odds": 0.0}
    for c, fill_val in odds_fill.items():
        if c in df.columns:
            df[c] = df[c].fillna(fill_val)

    # History columns (days since a run or win, raw rates, form) stay
    # missing for a horse without that history: a placeholder such as 60
    # days or a 0 rate reads as a real, specific record.

    # Stored weather observations pass through as recorded: a missing
    # reading stays missing rather than becoming a UK-average placeholder.
    # (The weather stage was removed on 2026-10-13; no feature reads them.)

    # Drop temporary sort helper.
    df = _without(df, [
        "_event_dt",
        "_finished_result",
        "_non_finisher_result",
        "_result_start",
        "_finish_pos_for_history",
        "_norm_pos_for_history",
        "_won_for_history",
        "_placed_for_history",
        "_result_start_for_history",
        "_non_finisher_for_history",
    ])

    if save:
        output_path = os.path.join(
            config.PROCESSED_DATA_DIR, "featured_races.parquet"
        )
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info(f"Saved feature-engineered data to {output_path}")

    logger.info(
        f"Feature engineering complete: {df.shape[1]} features, "
        f"{len(df)} records"
    )
    return df


if __name__ == "__main__":
    from src.data_processor import process_data

    processed = process_data()
    featured = engineer_features(processed)
    print(f"\nFeatured dataset shape: {featured.shape}")
    print(f"\nAll features:\n{list(featured.columns)}")
