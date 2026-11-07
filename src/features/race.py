"""Race-level stages: market, field quality, draw and age context, venue
configuration, course dynamics, interactions and conditional masks.

Moved out of ``src/feature_engineer.py`` on 2026-11-07 (research 065).
"""

import logging

import numpy as np
import pandas as pd

from src.features.common import _add_result_history_flags, _horse_key, _with_columns, _without
from src.features.history import _cumulative, _entity_day_start_view, _lag, _rolling
from src.track_config import direction_code, get_track_config, shape_code

logger = logging.getLogger(__name__)


def add_track_config_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge static track configuration data (direction, shape, gradients, etc.).

    These are fixed physical properties of each racecourse that provide
    genuinely new information not derivable from race results.
    """
    if "track" not in df.columns:
        return df

    logger.info("Engineering track configuration features...")
    unique_tracks = pd.Index(df["track"].astype(str).unique())
    config_map = {track: get_track_config(track) for track in unique_tracks}

    df["track_direction"] = df["track"].map(
        lambda t: direction_code(config_map[str(t)]["direction"])
    )
    df["track_shape"] = df["track"].map(
        lambda t: shape_code(config_map[str(t)]["shape"])
    )
    df["track_is_aw"] = df["track"].map(lambda t: config_map[str(t)]["is_aw"])
    df["track_circumference"] = df["track"].map(
        lambda t: config_map[str(t)]["circumference_f"]
    )
    df["track_draw_bias"] = df["track"].map(
        lambda t: config_map[str(t)]["draw_bias"]
    )

    # --- Shape one-hot (more useful than ordinal for trees) ---
    for shape_name, code in [("Galloping", 0), ("Tight", 1), ("Sharp", 2),
                              ("Undulating", 3), ("Stiff", 4)]:
        df[f"track_shape_{shape_name.lower()}"] = (
            df["track_shape"] == code
        ).astype(int)

    # --- Draw × draw-bias interaction ---
    # Strong draw bias + unfavourable draw = big disadvantage
    if "draw_pct" in df.columns:
        _dp = df["draw_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.5)
        df["draw_x_bias"] = _dp * df["track_draw_bias"]

        # New Draw × soft ground turf interaction: outside draws favored in heavy/soft turf
        _is_soft_going = df["going"].fillna("Good").isin(["Soft", "Heavy", "Yielding"]).astype(float)
        _is_turf = (~df["track_is_aw"].fillna(0).astype(bool)).astype(float)
        _is_soft_turf = _is_soft_going * _is_turf
        df["draw_pct_x_soft_turf"] = _dp * _is_soft_turf
        df["draw_bias_x_soft_turf"] = df["track_draw_bias"] * _is_soft_turf

    # --- Tight/sharp track × draw ---
    # On tight tracks, inside draws matter more
    if "draw_pct" in df.columns:
        df["tight_track_x_draw"] = (
            (df["track_shape_tight"] | df["track_shape_sharp"]).astype(float)
            * _dp
        )

    return df


# Static per-shape "quirkiness": how much an idiosyncratic layout rewards
# course-craft / experience. Galloping, fair tracks ≈ 0; tight/sharp ≈ 1.
# Codes match shape_code(): Galloping 0, Tight 1, Sharp 2, Undulating 3, Stiff 4.
_SHAPE_QUIRK = {0: 0.0, 1: 0.8, 2: 1.0, 3: 0.4, 4: 0.5}


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km (NaN-safe)."""
    R = 6371.0
    lat1r = np.radians(np.asarray(lat1, dtype=float))
    lat2r = np.radians(np.asarray(lat2, dtype=float))
    dlat = np.radians(np.asarray(lat2, dtype=float) - np.asarray(lat1, dtype=float))
    dlon = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_course_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Course angles that go beyond the static venue config.

    All features are strictly race-safe (prior runs only, via the seed-aware
    cumulative / lag primitives). Themes:

    * **Course experience × quirkiness** — field-relative course experience and
      proven course form weighted by a static quirk score (tight Chester,
      switchback Goodwood).
    * **Travel** — distance shipped from the last run: mild fatigue but also a
      trainer-intent signal.

    Handedness, pace-bias and going-at-this-course angles were retired on
    2026-11-07 (research 065): no recorded recipe used them.

    Must run after ``add_track_config_features`` (needs track_direction/shape/
    draw_bias/circumference) and ``add_track_features`` (horse_track_runs / wr).
    """
    if "track" not in df.columns:
        return df
    logger.info("Engineering course-dynamics features...")
    hkey = _horse_key(df)

    # ── Static course descriptors (0..1) ─────────────────────────
    shape = df.get("track_shape")
    draw_bias = pd.to_numeric(df.get("track_draw_bias"), errors="coerce").fillna(0.0) \
        if "track_draw_bias" in df.columns else pd.Series(0.0, index=df.index)
    circ = pd.to_numeric(df.get("track_circumference"), errors="coerce").fillna(10.0) \
        if "track_circumference" in df.columns else pd.Series(10.0, index=df.index)
    if shape is not None:
        quirk = shape.map(_SHAPE_QUIRK).fillna(0.0).astype(float)
    else:
        quirk = pd.Series(0.0, index=df.index)
    short_circ = ((circ > 0) & (circ < 9.0)).astype(float)
    df["course_quirk_score"] = (quirk + 0.15 * draw_bias + 0.2 * short_circ).clip(0.0, 1.0)

    # ── #1 Course experience × quirk ─────────────────────────────
    if "horse_track_runs" in df.columns:
        runs = df["horse_track_runs"].fillna(0.0)
        # Field-relative course experience (race-safe: prior-run counts only).
        df["course_exp_vs_field"] = runs - df.groupby("race_id")["horse_track_runs"].transform("mean")
    if "horse_track_wr_shrunk" in df.columns:
        df["course_form_x_quirk"] = (
            df["horse_track_wr_shrunk"].fillna(0.0) * df["course_quirk_score"]
        )

    # ── #6 Travel shipped from the last run ──────────────────────
    cfg = {t: get_track_config(str(t)) for t in df["track"].astype(str).unique()}
    df["track_lat"] = df["track"].astype(str).map(lambda t: cfg[t]["lat"])
    df["track_lon"] = df["track"].astype(str).map(lambda t: cfg[t]["lon"])
    prev_lat = _lag(df, hkey, "track_lat")
    prev_lon = _lag(df, hkey, "track_lon")
    df["travel_from_last_km"] = _haversine_km(
        prev_lat.to_numpy(), prev_lon.to_numpy(),
        df["track_lat"].to_numpy(), df["track_lon"].to_numpy())
    df = _without(df, ["track_lat", "track_lon"])

    return df


# add_weather_features and add_trainer_travel_features were removed on
# 2026-10-13. Both families measured dead (ledger 2g: every weather signal
# scores 0.0000 because weather is race-level; ledger 6: travel is noise), and
# both made the store depend on more than the database: the weather stage
# called the Open-Meteo API at build time for rows with no stored weather and
# silently filled failures with constants, and the travel stage read an
# unsigned CSV. A parity check on 106,467 rows showed the removal drops
# exactly their 10 columns and leaves the other 1,058 bit-identical.


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add explicit interaction features that capture well-known racing angles.

    Tree models can discover interactions via splits, but explicit
    features help them learn faster and more reliably.
    """
    logger.info("Engineering interaction features...")
    interaction_cols: dict[str, pd.Series | np.ndarray | int | float] = {}

    # 3) jockey_elo × is_favourite
    #    Strong rider booking confirmed by market.
    if "jockey_elo" in df.columns and "is_favourite" in df.columns:
        # Centre jockey_elo around baseline so the interaction is meaningful
        interaction_cols["jockey_elo_x_fav"] = (
            (df["jockey_elo"] - 1500) * df["is_favourite"]
        )

    # 5) speed_fig × distance match
    #    Good speed figure is more predictive at the proven distance.
    # speed_fig_avg_5 is lengths behind (0 = always won, median ~2), so a
    # clip at 2 gave half the field the same speed; 1/(1+x) keeps the order.
    if "speed_fig_avg_5" in df.columns:
        _speed_quality = 1.0 / (1.0 + df["speed_fig_avg_5"])

    # 6) OR × handicap — official rating matters most in handicaps.
    # Observed marks only: an unrated runner's race-median stand-in is not a
    # mark, and averaging it in pulls the field mean toward the median.
    if "official_rating" in df.columns and "handicap" in df.columns:
        from src.feature_semantics import observed_rating
        _mark = observed_rating(df)
        race_avg_or = _mark.groupby(df["race_id"]).transform("mean")
        interaction_cols["or_vs_field_x_hcap"] = (
            ((_mark - race_avg_or) * df["handicap"]).where(df["handicap"].ne(0), 0.0)
        )

    # 8) age × going — younger horses struggle more on soft/heavy
    if "age" in df.columns and "going_Soft" in df.columns:
        soft_heavy = df.get("going_Soft", 0)
        if "going_Heavy" in df.columns:
            soft_heavy = soft_heavy + df["going_Heavy"]

    # 10) Improving flag — both speed and form trending better
    #     form_slope > 0 means improving (see add_horse_features);
    #     speed_fig_trend < 0 means improving (lower = faster).
    if "speed_fig_trend" in df.columns and "form_slope" in df.columns:
        interaction_cols["improving_flag"] = (
            (df["speed_fig_trend"] < 0) & (df["form_slope"] > 0)
        ).astype(int)

    # 11) Trainer intent score — class drop + jockey upgrade + first-time headgear
    intent = pd.Series(0, index=df.index)
    if "class_dropped" in df.columns:
        intent = intent + df["class_dropped"]
    if "jockey_upgrade" in df.columns:
        intent = intent + df["jockey_upgrade"]
    if "first_time_headgear" in df.columns:
        intent = intent + df["first_time_headgear"]

    # 12) distance_change × class_change — trip + class shift combo
    if "distance_change" in df.columns and "class_change" in df.columns:
        interaction_cols["dist_x_class_change"] = df["distance_change"] * df["class_change"]

    # ── Market-residual interactions ─────────────────────────────
    # These help the model learn *when* the market misprices a horse
    # by pairing market signals with fundamental ability indicators.

    # 15) norm_implied_prob × horse_win_rate_shrunk — market vs proven ability
    if "norm_implied_prob" in df.columns and "horse_win_rate_shrunk" in df.columns:
        interaction_cols["mkt_x_win_rate"] = df["norm_implied_prob"] - df["horse_win_rate_shrunk"]

    # 16) log_odds × horse_elo — price level vs ability rating
    if "log_odds" in df.columns and "horse_elo" in df.columns:
        interaction_cols["logodds_x_elo"] = df["log_odds"] * (df["horse_elo"] - 1500) / 400

    # 17) odds_vs_field × class_dropped — overlooked class droppers
    if "odds_vs_field" in df.columns and "class_dropped" in df.columns:
        interaction_cols["odds_field_x_dropped"] = df["odds_vs_field"] * df["class_dropped"]

    # 18) norm_implied_prob × speed_fig_avg_5 — market vs recent speed
    if "norm_implied_prob" in df.columns and "speed_fig_avg_5" in df.columns:
        interaction_cols["mkt_x_speed"] = df["norm_implied_prob"] - _speed_quality

    # 19) odds_vs_field × jockey_elo — price vs rider quality
    if "odds_vs_field" in df.columns and "jockey_elo" in df.columns:
        interaction_cols["odds_field_x_jock_elo"] = df["odds_vs_field"] * (df["jockey_elo"] - 1500) / 400

    # 20) Context-adjusted Elo interactions — let the model learn when a high
    # overall Elo should be trusted only in the horse's preferred conditions.
    if "horse_elo" in df.columns:
        _elo_centered = (df["horse_elo"] - 1500) / 400
        if "horse_rt_specialist_edge" in df.columns:
            interaction_cols["elo_x_rt_edge"] = _elo_centered * df["horse_rt_specialist_edge"]

    # ── Additional strategic interactions ─────────────────────────

    # 21) Elo × recent form — high-rated horse currently in good form
    if "horse_elo" in df.columns and "horse_avg_pos_3" in df.columns:
        _elo_c = (df["horse_elo"] - 1500) / 400
        # avg_pos_3 is a field share (0 = won, 1 = last) → invert so positive = good
        _form_quality = 1.0 - df["horse_avg_pos_3"]
        interaction_cols["elo_x_recent_form"] = _elo_c * _form_quality

    # 24) Form slope × post-break — freshened horses with improving trend
    if "form_slope" in df.columns and "days_since_last_race" in df.columns:
        _is_fresh = (df["days_since_last_race"] > 30).astype(np.float32)
        interaction_cols["form_slope_x_fresh"] = df["form_slope"] * _is_fresh

    if interaction_cols:
        interaction_frame = pd.DataFrame(interaction_cols, index=df.index)
        overlapping_cols = [col for col in interaction_frame.columns if col in df.columns]
        if overlapping_cols:
            df = _without(df, overlapping_cols)
        df = _with_columns(df, interaction_frame)

    return df


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from betting odds (market indicators).

    Odds encode crowd wisdom about a horse's chances.

    .. warning::

       For historical results the ``odds`` column is the **Starting Price
       (SP)** — the final market price at race-off.  SP is significantly
       more informative than the early/ante-post odds available when a
       punter would actually place a bet.  This means backtest results
       using odds-derived features will be **optimistic** relative to
       live performance where only early morning or pre-race odds are
       available.
    """
    logger.info("Engineering market features...")
    df = df.copy()

    if "odds" not in df.columns:
        return df

    # Implied probability from odds
    df["implied_prob"] = 1.0 / df["odds"]

    # Normalised probability within race
    race_prob_sum = df.groupby("race_id")["implied_prob"].transform("sum")
    df["norm_implied_prob"] = df["implied_prob"] / race_prob_sum

    # Odds rank within race (1 = favourite)
    df["odds_rank"] = df.groupby("race_id")["odds"].rank(method="min")

    # Is favourite?
    df["is_favourite"] = (df["odds_rank"] == 1).astype(int)

    # Log odds (more linear relationship with probability)
    df["log_odds"] = np.log1p(df["odds"])

    # Odds relative to field average
    race_avg_odds = df.groupby("race_id")["odds"].transform("mean")
    df["odds_vs_field"] = df["odds"] / race_avg_odds

    # Market overround (sum of implied probs — higher = less efficient market)
    df["overround"] = race_prob_sum

    return df


def add_race_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features about the race context."""
    logger.info("Engineering race context features...")
    _rs = (df["_result_start"] if "_result_start" in df
           else _add_result_history_flags(df.copy())["_result_start"]).to_numpy()

    # Draw advantage as a share of the highest stall drawn in the race.
    # Jumps/bumpers have no starting stalls, so draw == 0 is "no draw", NOT the
    # innermost stall (real flat draws are >= 1). Mark these missing so the
    # model isn't told every jumps horse is best-drawn, and so the per-track
    # draw-third stats below are computed over stalls races only (NaN draw_pct
    # falls outside the low/mid/high bins). Dividing by ``num_runners`` (the
    # field after non-runners) put draws above 1 in 6.8% of drawn rows.
    _stall = pd.to_numeric(df["draw"], errors="coerce")
    _stall = _stall.where(_stall > 0)
    df["draw_pct"] = _stall / _stall.groupby(df["race_id"]).transform("max")

    # ── Track-specific draw bias ─────────────────────────────────
    # Per-track draw-position win-rate (point-in-time, no leakage).
    # Tracks like Chester, Beverley, Ascot have extreme draw biases.
    # No-draw races (NaN draw_pct) are excluded from the bins, so these
    # stats reflect flat-with-stalls form only and don't pollute mixed
    # flat/jumps tracks (Ascot, Newcastle, Doncaster, ...).
    if "track" in df.columns:

        # Create draw-position bins (low / mid / high third)
        df["_draw_third"] = pd.cut(
            df["draw_pct"].clip(0.01, 0.99),
            bins=[0, 0.33, 0.66, 1.0],
            labels=["low", "mid", "high"],
            include_lowest=True,
        ).astype(str)

        for draw_pos in ["low", "mid", "high"]:
            mask = (df["_draw_third"] == draw_pos).to_numpy()
            # Exclude every runner in the current race, including earlier
            # rows (results are commonly stored in finishing order).
            _td = _cumulative(
                df["track"].to_numpy()[mask],
                df["race_id"].to_numpy()[mask],
                {
                    f"_td_wins_{draw_pos}": df["won"].to_numpy()[mask],
                    f"_td_runs_{draw_pos}": _rs[mask],
                },
            )
            df.loc[mask, f"_td_wins_{draw_pos}"] = _td[f"_td_wins_{draw_pos}"]
            df.loc[mask, f"_td_runs_{draw_pos}"] = _td[f"_td_runs_{draw_pos}"]

        # Recombine: the horse's draw-third win-rate at this track
        df["track_draw_wr_wins"] = 0.0
        df["track_draw_wr_runs"] = 0.0
        for draw_pos in ["low", "mid", "high"]:
            mask = df["_draw_third"] == draw_pos
            w_col = f"_td_wins_{draw_pos}"
            r_col = f"_td_runs_{draw_pos}"
            if w_col in df.columns:
                df.loc[mask, "track_draw_wr_wins"] = df.loc[mask, w_col]
                df.loc[mask, "track_draw_wr_runs"] = df.loc[mask, r_col]

        # Missing below 6 prior runs: a 0 would read as "this draw never wins".
        df["track_draw_win_rate"] = np.where(
            df["track_draw_wr_runs"] > 5,
            df["track_draw_wr_wins"] / df["track_draw_wr_runs"],
            np.nan,
        )

        # Match the pre-race card view: exclude earlier races on this day.
        df = _entity_day_start_view(
            df, ["track", "_draw_third"], ["track_draw_win_rate"]
        )

        # Clean up
        drop_cols = ["_draw_third", "track_draw_wr_wins", "track_draw_wr_runs"]
        drop_cols += [c for c in df.columns if c.startswith("_td_")]
        df = _without(df, drop_cols)

    # ── Draw × distance × field size interactions ────────────────
    # Draw matters enormously in sprints (especially ≤6f) on tight
    # turning tracks with big fields, but barely at all over 2m+.
    _dp = df.get("draw_pct", pd.Series(0.5, index=df.index))
    _dp = _dp.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    _dist = df["distance_furlongs"].fillna(8.0)
    _nr = df["num_runners"].fillna(10)

    _is_sprint = (_dist <= 7.0).astype(float)   # sprints ≤ 7f
    _large = (_nr >= 14).astype(float)

    # Core interaction: draw advantage only matters for short distances
    # with big fields (where getting boxed in is a real risk).
    df["draw_x_sprint_x_large"] = _dp * _is_sprint * _large

    # Draw × inverse distance — draw impact fades with distance
    df["draw_x_inv_dist"] = _dp * (1.0 / _dist.clip(lower=4.0))

    # Draw × field size (normalized) — more runners = more draw impact
    df["draw_x_field_size"] = _dp * (_nr / 20.0).clip(0, 2)

    # Track bias × sprint — track draw bias is primarily a sprint phenomenon
    if "track_draw_bias" in df.columns:
        df["draw_bias_x_sprint"] = df["track_draw_bias"] * _is_sprint

    # ── Age-performance curve by race type ────────────────────────
    # Flat horses peak at 3-4yo; NH horses at 7-9yo. Modelling the
    # distance from peak age lets the model learn the decline curve.
    _age = df["age"].fillna(4).astype(float)
    from src.feature_semantics import jumps_mask
    _is_nh = jumps_mask(df).astype(float)

    # Peak age: 3.5 for Flat, 8.0 for NH
    _peak = np.where(_is_nh > 0.5, 8.0, 3.5)
    df["years_from_peak"] = _age - _peak
    df["years_from_peak_abs"] = np.abs(df["years_from_peak"])
    # Quadratic decline — penalty grows with distance from peak
    df["age_decline_curve"] = df["years_from_peak"] ** 2 / 25.0
    # Is the horse past peak? (different signal from approaching it)
    # Age appropriateness for race type
    # Young flat horse (2-3yo) — still improving

    # Weight relative to field
    race_avg_weight = df.groupby("race_id")["weight_lbs"].transform("mean")
    df["weight_vs_field"] = df["weight_lbs"] - race_avg_weight

    # Age relative to field
    race_avg_age = df.groupby("race_id")["age"].transform("mean")
    df["age_vs_field"] = df["age"] - race_avg_age

    # Field size categories

    # ── Field-size-conditioned base rates ─────────────────────────
    # Explicit prior: a random runner wins 1/n and places k/n.
    # Gives the model an anchor that shifts with field size so it can
    # learn calibration offsets rather than raw rates.
    _nr_safe = _nr.clip(lower=1)
    df["base_win_rate"] = 1.0 / _nr_safe

    # places_paid follows standard UK EW rules
    _hcap = df.get("handicap", pd.Series(0, index=df.index)).fillna(0).astype(bool)
    _pp = np.where(
        _nr <= 4, np.minimum(_nr, 3),  # match the trained <=4-runner top-three target
        np.where(_nr <= 7, 2,
        np.where(_nr <= 15, 3,
        np.where(_hcap, 4, 3)))
    )
    df["base_place_rate"] = _pp / _nr_safe.values

    # How far above baseline the market thinks this horse is (log scale)
    if "implied_prob" in df.columns:
        _ip = df["implied_prob"].clip(lower=1e-4)
        df["implied_prob_vs_base"] = np.log(_ip / df["base_win_rate"])

    # Prize money rank (proxy for race quality)

    # Weight per furlong — at longer distances weight matters more
    if "distance_furlongs" in df.columns:
        safe_dist = df["distance_furlongs"].replace(0, np.nan).fillna(8.0)
        df["weight_per_furlong"] = df["weight_lbs"] / safe_dist

    # Handicap features
    if "handicap" in df.columns:
        df["handicap"] = df["handicap"].fillna(0).astype(int)
        # OR x handicap interaction — OR is most meaningful in handicaps.
        # Missing for an unrated handicapper, not the race-median stand-in.
        if "official_rating" in df.columns:
            from src.feature_semantics import observed_rating
            df["or_x_handicap"] = (
                (observed_rating(df) * df["handicap"]).where(df["handicap"].ne(0), 0.0)
            )

    # --- Odds coefficient of variation (field competitiveness) ---
    if "odds" in df.columns:
        race_std_odds = df.groupby("race_id")["odds"].transform("std").fillna(0)
        race_mean_odds = df.groupby("race_id")["odds"].transform("mean")
        df["odds_cv"] = np.where(
            race_mean_odds > 0, race_std_odds / race_mean_odds, 0
        )

    # --- OR vs top-rated in race (observed marks only) ---
    if "official_rating" in df.columns:
        from src.feature_semantics import observed_rating
        _mark = observed_rating(df)
        race_max_or = _mark.groupby(df["race_id"]).transform("max")
        df["or_vs_top"] = _mark - race_max_or

    # --- Horse Elo vs field average ---
    if "horse_elo" in df.columns:
        field_avg = df.groupby("race_id")["horse_elo"].transform("mean")
        df["horse_elo_vs_field"] = df["horse_elo"] - field_avg

    return df


def add_field_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features describing the quality and competitiveness of the field.

    These contextualise a horse's abilities relative to the strength of
    the opposition it faces — beating a Group 1 field is very different
    from beating a Class 6 seller.
    """
    logger.info("Engineering field quality features...")

    # Official-rating context of the race field, from observed marks only.
    # data_processor gives an unrated runner the race-median OR; ranking and
    # spreading those stand-ins described the field as rated when it is not.
    # Missing where the race has too few rated runners to say.
    if "official_rating" in df.columns:
        from src.feature_semantics import observed_rating
        _mark = observed_rating(df)
        _by_race = _mark.groupby(df["race_id"])
        df["field_std_or"] = _by_race.transform("std")
        _or_rnk = _by_race.rank(ascending=False, method="first")
        race_second_or = _mark.where(_or_rnk.eq(2)).groupby(df["race_id"]).transform("max")
        df["or_rank_in_race"] = _by_race.rank(ascending=False, method="min")
        df["or_vs_second_top"] = _mark - race_second_or

    return df


def add_opposition_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """Strength-of-opposition features derived from Elo ratings.

    For each runner we compute three types of signal:

    1. **Current-race opposition** — average ``horse_elo`` of all
       co-runners *excluding* the horse itself.  This tells the model
       how tough *this* race is before it even starts.

    2. **Rolling prior-field quality** — per-horse rolling mean and EWMA
       of that per-race opposition Elo over its last N result races.
       This answers: "has this horse been competing in strong fields
       recently?"  A horse that finished 3rd against Group 1 quality
       is treated differently from one that won in a weak maiden.

    3. **Elo vs opposition** — horse's own Elo minus the field average
       (already computed by compute_elo_features as ``horse_elo_vs_field``,
       but here we add a *prior-race* EWMA version so the model can see
       whether the horse has been consistently above or below field level).

    Requires ``horse_elo`` to already be present (call after
    ``compute_elo_features``).
    """
    if "horse_elo" not in df.columns:
        return df

    logger.info("Engineering opposition-strength features…")
    hkey = _horse_key(df)

    # ── 1. Per-race opposition mean Elo (excluding self) ──────────────
    # race_elo_sum / race_n give us the field total.
    # field_elo_excl_self = (sum - self) / (n - 1)
    race_elo_sum = df.groupby("race_id")["horse_elo"].transform("sum")
    race_n = df.groupby("race_id")["horse_elo"].transform("count")
    # Guard against single-runner races (n==1) to avoid division by zero
    df["field_avg_elo"] = np.where(
        race_n > 1,
        (race_elo_sum - df["horse_elo"]) / (race_n - 1),
        df["horse_elo"],   # fallback: own Elo (no opponents)
    )

    # ── 2. Rolling prior-field quality per horse ──────────────────────
    # field_avg_elo reflects THIS race's opposition, but for training
    # signal we need the PRIOR races' field quality (shifted by 1).
    # Use _fast_grouped_rolling which applies shift(1) internally.
    _opp_ops = [
        ("field_avg_elo_5",   "mean",  5),
        ("field_avg_elo_10",  "mean", 10),
    ]
    # Add EWMA half-life 7 (≈ last 10 races with decent weight)
    _ewma_hl = 7
    _ewma_alpha = 1.0 - np.exp(-np.log(2) / _ewma_hl)
    _opp_ops.append(("field_avg_elo_ewma", "ewma", _ewma_alpha))

    for name, arr in _rolling(df,hkey, "field_avg_elo", _opp_ops).items():
        df[name] = arr

    # ── 3. Prior-race Elo-vs-field EWMA ──────────────────────────────
    # How consistently has this horse outrated / underrated its fields?
    df["_elo_vs_opp"] = df["horse_elo"] - df["field_avg_elo"]
    _vs_ops = [("elo_vs_field_ewma", "ewma", _ewma_alpha)]
    for name, arr in _rolling(df,hkey, "_elo_vs_opp", _vs_ops).items():
        df[name] = arr
    df = _without(df, ["_elo_vs_opp"])

    # Leave cold-start opposition features missing. Tree learners can
    # branch on missingness directly, which is preferable to inventing a
    # synthetic neutral field-quality history for unraced horses.

    return df


# ── Conditional feature masking ──────────────────────────────────
# Some features are only meaningful for certain race types.
# We only gate the hard structural case here: National Hunt
# (Hurdle / Chase) races have no stall draws, so draw-derived
# features are not merely low-signal there, they are inapplicable.
#
# Tree models in this project can already branch on missingness, so
# use NaN for not-applicable continuous features instead of inventing
# a numeric value that might carry unintended meaning.

# Draw-derived features that are inapplicable for NH races.
_NH_INAPPLICABLE_DRAW_FEATURES = [
    "draw_pct",
    "draw_x_bias",
    "draw_x_sprint_x_large",
    "draw_x_inv_dist",
    "draw_x_field_size",
    "draw_bias_x_sprint",
    "track_draw_win_rate",
    "tight_track_x_draw",
    "track_draw_bias",
    "draw_pct_x_soft_turf",
    "draw_bias_x_soft_turf",
]


def apply_conditional_feature_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Mask structurally inapplicable features for certain race types.

    Draw-related features are set to NaN for Hurdle / Chase because
    NH races have no stall draws. This prevents the model from
    learning from random or backfilled draw artefacts while preserving
    the information that the feature is not applicable for that row.

    Returns:
        DataFrame with masked features.
    """
    logger.info("Applying conditional feature masks (race-type gating)…")
    df = df.copy()

    from src.feature_semantics import jumps_mask
    is_nh = jumps_mask(df)
    no_stall = pd.to_numeric(df["draw"], errors="coerce").isna() | df["draw"].le(0) if "draw" in df else is_nh
    draw_inapplicable = is_nh | no_stall

    n_nh = int(is_nh.sum())

    masked_count = 0
    for col in _NH_INAPPLICABLE_DRAW_FEATURES:
        if col in df.columns:
            df.loc[draw_inapplicable, col] = np.nan
            masked_count += 1

    logger.info(
        f"  Masked {masked_count} draw-derived features "
        f"(set to NaN for {n_nh:,} NH rows)"
    )
    return df
