"""Sire, dam and damsire progeny records.

Moved out of ``src/feature_engineer.py`` on 2026-11-07 (research 065).
"""

import logging

import numpy as np
import pandas as pd

from src.features.common import _bayesian_shrink, _defrag, _MID_STAGE_BLOCKS, _rate, _without
from src.features.history import _cumulative, _entity_day_start_view

logger = logging.getLogger(__name__)


def add_pedigree_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sire, dam, and damsire historical stats (Bayesian shrunk)."""
    logger.info("Engineering pedigree features...")
    df = df.copy()

    # Pre-compute condition masks for specialty pedigree features
    is_soft = df["going"].fillna("Good").isin(["Soft", "Heavy", "Yielding"])
    from src.surface_features import synthetic_surface
    is_aw = synthetic_surface(df["surface"]).eq(1)
    dist_f = df["distance_furlongs"].fillna(8.0)
    is_sprint = dist_f <= 7.0
    is_stay_route = dist_f >= 11.5

    result_start_flag = df["_result_start"].fillna(0.0)
    won_flag = df["won"].fillna(0.0)
    placed_flag = df["_placed_for_history"].fillna(0.0)

    for col in ["sire", "dam", "damsire"]:
        if col not in df.columns:
            continue

        # Race-safe metrics tracking each sire's/dam's progeny performance
        # We must exclude the current race to prevent data leakage from stable-mates
        # competing in the same race.
        # Use existing race-safe cumulative functions from the module!
        _ped = _cumulative(
            df[col].to_numpy(), df["race_id"].to_numpy(),
            {
                # Use the fillna(0) flags (as the conditional path below does) —
                # the raw _placed_for_history is NaN on result-less rows (live
                # cards, non-finishers), which propagated NaN into _safe_*_places
                # and left {col}_place_rate_shrunk NaN for every live runner while
                # _wins (from `won`, set to 0 on cards) stayed fine.
                f"_safe_{col}_runs": result_start_flag.to_numpy(),
                f"_safe_{col}_wins": won_flag.to_numpy(),
                f"_safe_{col}_places": placed_flag.to_numpy(),
            },
        )
        df[f"_safe_{col}_runs"] = _ped[f"_safe_{col}_runs"]
        df[f"_safe_{col}_wins"] = _ped[f"_safe_{col}_wins"]
        df[f"_safe_{col}_places"] = _ped[f"_safe_{col}_places"]

        df[f"{col}_win_rate"] = _rate(df[f"_safe_{col}_wins"], df[f"_safe_{col}_runs"])
        # Bayesian shrink towards typical population prior
        df[f"{col}_win_rate_shrunk"] = _bayesian_shrink(
            df[f"{col}_win_rate"], df[f"_safe_{col}_runs"],
            prior_rate=0.10, prior_strength=20.0,
        )

        df[f"{col}_place_rate"] = _rate(df[f"_safe_{col}_places"], df[f"_safe_{col}_runs"])
        df[f"{col}_place_rate_shrunk"] = _bayesian_shrink(
            df[f"{col}_place_rate"], df[f"_safe_{col}_runs"],
            prior_rate=0.30, prior_strength=20.0,
        )

        # Cleanup basic stats temporary columns
        temp_cols = [f"_safe_{col}_runs", f"_safe_{col}_wins", f"_safe_{col}_places", 
                     f"{col}_win_rate", f"{col}_place_rate"]

        # Only compute detailed conditional preferences for sire and damsire to avoid noise/fragmentation
        if col != "dam":
            conds = {
                "soft": is_soft,
                "aw": is_aw,
                "sprint": is_sprint,
                "stay": is_stay_route
            }
            for cond_name, mask in conds.items():
                temp_runs = f"_safe_{col}_{cond_name}_runs"
                temp_wins = f"_safe_{col}_{cond_name}_wins"
                temp_places = f"_safe_{col}_{cond_name}_places"

                # Conditional (going/surface/trip) progeny counts, race-safe and
                # seed-aware via _cumulative; metric keys are unique per (col, cond).
                _pc = _cumulative(
                    df[col].to_numpy(), df["race_id"].to_numpy(),
                    {
                        temp_runs: result_start_flag.where(mask, 0.0).to_numpy(),
                        temp_wins: won_flag.where(mask, 0.0).to_numpy(),
                        temp_places: placed_flag.where(mask, 0.0).to_numpy(),
                    },
                )
                df[temp_runs] = _pc[temp_runs]
                df[temp_wins] = _pc[temp_wins]
                df[temp_places] = _pc[temp_places]

                wr_raw = np.where(df[temp_runs] > 0, df[temp_wins] / df[temp_runs], 0.0)
                pr_raw = np.where(df[temp_runs] > 0, df[temp_places] / df[temp_runs], 0.0)

                # Shrunk rates
                df[f"{col}_{cond_name}_win_rate_shrunk"] = _bayesian_shrink(
                    wr_raw, df[temp_runs],
                    prior_rate=0.10, prior_strength=10.0,
                )
                df[f"{col}_{cond_name}_place_rate_shrunk"] = _bayesian_shrink(
                    pr_raw, df[temp_runs],
                    prior_rate=0.30, prior_strength=10.0,
                )

                # Specialisation edge (relative to overall class rating)
                df[f"{col}_{cond_name}_edge"] = (
                    df[f"{col}_{cond_name}_win_rate_shrunk"] - df[f"{col}_win_rate_shrunk"]
                )

                temp_cols.extend([temp_runs, temp_wins, temp_places])
                df = _defrag(df, max_blocks=_MID_STAGE_BLOCKS)

            # Drop temporary conditional indicator columns
            df = _without(df, ["_temp_runs_cond", "_temp_wins_cond", "_temp_places_cond"])

        df = _without(df, temp_cols)
        # All cards for a day are predicted before any of that day's results
        # exist. Progeny in later races must share the day's opening history.
        pedigree_cols = [c for c in df.columns if c.startswith(f"{col}_")
                         and (c.endswith("_shrunk") or c.endswith("_edge"))]
        df = _entity_day_start_view(df, [col], pedigree_cols)

    return df
