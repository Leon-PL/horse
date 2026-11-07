"""Declared within-race transforms of base signals and their composites.

Moved out of ``src/feature_engineer.py`` on 2026-11-07 (research 065).
"""

import logging

import numpy as np
import pandas as pd

import config
from src.features.common import _with_columns, _without

logger = logging.getLogger(__name__)


#: Within-race transforms of a base signal, keyed by base. The first field is
#: the base's orientation: ``higher`` or ``lower`` is better, or ``signed`` for
#: a context shift with no better direction. The second lists the transforms
#: to emit, one letter each:
#:
#: * ``F`` ``<base>_vs_field``: signed gap to the race mean (positive = better);
#: * ``Z`` ``<base>_z_in_race``: that gap over the race's standard deviation;
#: * ``R`` ``<base>_rank_pct_in_race``: percentile rank, best runner lowest;
#: * ``B`` ``<base>_vs_best_in_race``: signed gap to the best *other* runner;
#: * ``A`` ``<base>_abs_rank_pct_in_race``: percentile rank of the size of a
#:   signed shift, largest lowest (``signed`` bases only).
#:
#: Only transforms some recorded recipe in ``feature_sets/`` uses are emitted;
#: the other 302 of the earlier 537 were dropped on 2026-11-07 (research 065).
#: ``{te_window}`` is ``config.TE_WINDOW_DAYS``. A missing base value stays
#: missing in every transform.
_RACE_RELATIVE: dict[str, tuple[str, str]] = {
    "horse_prev_races": ("higher", "FZR"),
    "horse_going_win_rate": ("higher", "B"),
    "horse_going_wr_shrunk": ("higher", "B"),
    "horse_dist_cat_wr_shrunk": ("higher", "B"),
    "horse_surface_wr_shrunk": ("higher", "B"),
    "horse_rt_specialist_edge": ("higher", "F"),
    "jockey_elo": ("higher", "FZB"),
    "jockey_win_rate_shrunk": ("higher", "FZB"),
    "jockey_place_rate_shrunk": ("higher", "FZB"),
    "jockey_track_win_rate": ("higher", "B"),
    "jockey_dist_cat_runs": ("higher", "Z"),
    "jt_prev_runs": ("higher", "FB"),
    "trainer_prev_runs": ("higher", "R"),
    "trainer_freq": ("higher", "RB"),
    "trainer_elo": ("higher", "FZB"),
    "trainer_win_rate_shrunk": ("higher", "F"),
    "trainer_place_rate_shrunk": ("higher", "B"),
    "trainer_pr_30d": ("higher", "FB"),
    "trainer_pr_30d_delta": ("higher", "F"),
    "trainer_track_win_rate": ("higher", "FZB"),
    "trainer_class_runs": ("higher", "F"),
    "trainer_class_wr": ("higher", "FZB"),
    "trainer_dist_cat_wr": ("higher", "B"),
    "trainer_going_wr": ("higher", "FB"),
    "class_change": ("higher", "F"),
    "field_avg_elo_5": ("higher", "Z"),
    "field_avg_elo_10": ("higher", "FZB"),
    "field_avg_elo_ewma": ("higher", "B"),
    "horse_elo_momentum": ("higher", "FB"),
    "horse_margin_elo_momentum": ("higher", "FZB"),
    "combined_elo": ("higher", "FZB"),
    "horse_elo_surface_edge": ("higher", "F"),
    "horse_elo_rt_edge": ("higher", "Z"),
    "atr_finishing_speed_pct_avg_3": ("higher", "FZRB"),
    "atr_finishing_speed_pct_avg_5": ("higher", "FZRB"),
    "atr_finishing_speed_pct_best_3": ("higher", "FZRB"),
    "atr_finishing_speed_pct_trend": ("higher", "FZRB"),
    "atr_max_speed_avg_3": ("higher", "FZRB"),
    "atr_max_speed_avg_5": ("higher", "FZRB"),
    "atr_max_speed_best_3": ("higher", "FZRB"),
    "atr_max_speed_trend": ("higher", "FZRB"),
    "atr_closing_speed_avg_3": ("higher", "FZRB"),
    "atr_closing_speed_avg_5": ("higher", "FZRB"),
    "atr_closing_speed_best_3": ("higher", "FZRB"),
    "atr_closing_speed_trend": ("higher", "FZRB"),
    "atr_turn_of_foot_avg_3": ("higher", "FZRB"),
    "atr_turn_of_foot_avg_5": ("higher", "FZRB"),
    "atr_turn_of_foot_best_3": ("higher", "FZRB"),
    "atr_efficiency_score_avg_3": ("higher", "FZRB"),
    "atr_efficiency_score_avg_5": ("higher", "FZRB"),
    "atr_efficiency_score_best_3": ("higher", "FZRB"),
    "atr_energy_eff_avg_3": ("higher", "FZRB"),
    "atr_energy_eff_avg_5": ("higher", "FZRB"),
    "atr_peak_stride_length_avg_3": ("higher", "FZRB"),
    "atr_peak_stride_length_avg_5": ("higher", "FZRB"),
    "atr_peak_stride_freq_avg_3": ("higher", "FZRB"),
    "atr_upgrade_clean_avg_3": ("higher", "FZRB"),
    "atr_upgrade_clean_avg_5": ("higher", "FZRB"),
    "atr_upgrade_clean_best_3": ("higher", "FZRB"),
    "atr_closing_in_hand_avg_3": ("higher", "FZRB"),
    "atr_closing_in_hand_avg_5": ("higher", "FZRB"),
    "atr_early_pace_avg_3": ("higher", "FZRB"),
    "atr_cmt_trouble_rate_5": ("higher", "FZRB"),
    "atr_cmt_stayed_on_rate_5": ("higher", "FZRB"),
    "atr_cmt_weakened_rate_5": ("higher", "FZRB"),
    "atr_cmt_outpaced_rate_5": ("higher", "FZRB"),
    "atr_cmt_front_rate_5": ("higher", "FZRB"),
    "atr_cmt_held_up_rate_5": ("higher", "FZRB"),
    "atr_cmt_green_rate_5": ("higher", "FZRB"),
    "atr_cmt_nh_error_rate_5": ("higher", "FZRB"),
    "horse_avg_position": ("lower", "B"),
    "horse_ewma_lb_3": ("lower", "B"),
    "speed_fig_best_5": ("lower", "FB"),
    "speed_fig_adj_best_5": ("lower", "F"),
    "speed_fig_rel_avg_5": ("lower", "ZB"),
    "days_since_last_race": ("lower", "Z"),
    "days_log": ("lower", "F"),
    "days_since_last_win": ("lower", "B"),
    "age_decline_curve": ("lower", "F"),
    "jockey_te_norm_pos": ("lower", "B"),
    "jockey_te_ewma": ("lower", "B"),
    "trainer_te_norm_pos": ("lower", "B"),
    "trainer_te_ewma": ("lower", "FZB"),
    "jt_te_norm_pos": ("lower", "ZB"),
    "jockey_te_norm_pos_{te_window}d": ("lower", "B"),
    "trainer_te_norm_pos_{te_window}d": ("lower", "FB"),
    "distance_change": ("signed", "FZA"),
    "or_change": ("signed", "FZ"),
    "jockey_elo_change": ("signed", "FZ"),
}


#: Mean of within-race z-scores. Inputs share the composite's orientation and
#: are computed for it even when their own z column is not emitted.
_RELATIVE_COMPOSITES: dict[str, tuple[str, tuple[str, ...]]] = {
    "connections_relative_score": ("higher", (
        "jockey_elo", "jockey_wr_14d", "trainer_wr_14d", "trainer_win_rate_shrunk",
        "jockey_track_win_rate", "trainer_track_win_rate")),
    "freshness_relative_score": ("lower", (
        "days_since_last_race", "days_since_last_win", "years_from_peak_abs", "age_decline_curve")),
}


_RELATIVE_SUFFIX = {"F": "vs_field", "Z": "z_in_race", "R": "rank_pct_in_race",
                    "B": "vs_best_in_race", "A": "abs_rank_pct_in_race"}


def race_relative_outputs() -> list[str]:
    """Every column :func:`add_race_relative_signal_features` can emit."""
    window = int(getattr(config, "TE_WINDOW_DAYS", 365))
    names = []
    for base, (_, codes) in _RACE_RELATIVE.items():
        if "{te_window}" in base:
            if window <= 0:
                continue
            base = base.format(te_window=window)
        names += [f"{base}_{_RELATIVE_SUFFIX[code]}" for code in codes]
    return names + list(_RELATIVE_COMPOSITES)


def add_race_relative_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the declared within-race transforms (``_RACE_RELATIVE``) and composites."""
    logger.info("Engineering race-relative signal features...")

    if "race_id" not in df.columns:
        return df

    window = int(getattr(config, "TE_WINDOW_DAYS", 365))
    emit: dict[str, tuple[str, str]] = {}
    for base, spec in _RACE_RELATIVE.items():
        if "{te_window}" in base:
            if window <= 0:
                continue
            base = base.format(te_window=window)
        emit[base] = spec
    composite_inputs = {base: orientation for orientation, bases in _RELATIVE_COMPOSITES.values()
                        for base in bases}

    race_ids = df["race_id"]
    relative_cols: dict[str, pd.Series | np.ndarray] = {}
    z_scores: dict[str, pd.Series] = {}

    for orientation in ("higher", "lower", "signed"):
        bases = [b for b, (o, _) in emit.items() if o == orientation]
        bases += [b for b, o in composite_inputs.items() if o == orientation and b not in emit]
        available = [b for b in bases if b in df.columns]
        if not available:
            continue
        numeric_frame = (
            df[available]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        grouped = numeric_frame.groupby(race_ids)
        race_mean = grouped.transform("mean")
        race_std = grouped.transform("std").fillna(0.0)
        higher = orientation != "lower"
        signed_vs_field = numeric_frame - race_mean if higher else race_mean - numeric_frame
        wanted = "".join(emit.get(b, ("", ""))[1] for b in available)

        if orientation == "signed":
            if "A" in wanted:
                abs_rank = numeric_frame.abs().groupby(race_ids).rank(
                    ascending=False, pct=True, method="average")
        else:
            if "B" in wanted:
                # ``_vs_best_in_race`` compares each runner with the best *other*
                # runner, so the leader keeps its margin over the next best
                # instead of reading 0 like everyone level with it.
                oriented = numeric_frame if higher else -numeric_frame
                race_top = oriented.groupby(race_ids).transform("max")
                is_top = oriented.eq(race_top)
                sole_top = is_top & is_top.groupby(race_ids).transform("sum").eq(1)
                runner_up = oriented.where(~is_top).groupby(race_ids).transform("max")
                best_other = race_top.where(~sole_top, runner_up)
                signed_vs_best = (oriented - best_other).astype("float64")
            if "R" in wanted:
                rank_pct = numeric_frame.groupby(race_ids).rank(
                    ascending=not higher, pct=True, method="average")

        for col in available:
            codes = emit.get(col, ("", ""))[1]
            if "Z" in codes or col in composite_inputs:
                z_scores[col] = pd.Series(
                    np.where(race_std[col] > 1e-12, signed_vs_field[col] / race_std[col], 0.0),
                    index=df.index,
                ).where(numeric_frame[col].notna())
            for code in codes:
                name = f"{col}_{_RELATIVE_SUFFIX[code]}"
                if code == "F":
                    relative_cols[name] = signed_vs_field[col]
                elif code == "Z":
                    relative_cols[name] = z_scores[col]
                elif code == "R":
                    relative_cols[name] = rank_pct[col]
                elif code == "B":
                    relative_cols[name] = signed_vs_best[col]
                else:
                    relative_cols[name] = abs_rank[col]

    if relative_cols:
        overlap = [col for col in relative_cols if col in df.columns]
        if overlap:
            df = _without(df, overlap)
        df = _with_columns(df, pd.DataFrame(relative_cols, index=df.index))

    composite_cols: dict[str, pd.Series] = {}
    for name, (_, bases) in _RELATIVE_COMPOSITES.items():
        available = [b for b in bases if b in z_scores]
        if available:
            inputs = pd.DataFrame({f"{b}_z_in_race": z_scores[b] for b in available}, index=df.index)
            composite_cols[name] = inputs.mean(axis=1)
    if composite_cols:
        df = _with_columns(df, pd.DataFrame(composite_cols, index=df.index))

    return df
