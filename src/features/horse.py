"""Horse form stages: career record, recent form, distance/going/surface/course
records, class and distance changes, equipment, speed proxies, prize money.

Moved out of ``src/feature_engineer.py`` on 2026-11-07 (research 065).
"""

import logging

import numpy as np
import pandas as pd

import config
from src.features.common import _bayesian_shrink, _horse_key, _rate, _without, dist_category
from src.features.history import (
    _carried,
    _carry,
    _composite_key,
    _cumulative,
    _expanding_prev,
    _ffill_prev,
    _lag,
    _rolling,
    _running_extreme,
    _time_window,
)

logger = logging.getLogger(__name__)


def add_horse_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate horse-specific historical features.

    For each horse at each race, these features represent its performance
    in PRIOR races only (no data leakage).
    """
    logger.info("Engineering horse features...")

    hkey = _horse_key(df)

    # --- Cumulative horse stats (shifted to avoid leakage) ---
    result_start_flag = df["_result_start"]
    finished_flag = df["_finished_result"]
    non_finisher_flag = df["_non_finisher_result"]
    placed_hist = df["_placed_for_history"].fillna(0.0)
    norm_pos_hist = df["_norm_pos_for_history"].fillna(0.0)

    # Cumulative (prior-only, race-safe) career counts via the incremental
    # primitive. With state=None (a full build) this is identical to the
    # historical ``groupby(hkey)[v].cumsum() - v`` formula; the FeatureStore
    # append path can seed it from carried state so it never recomputes
    # history. Class C — see docs/architecture/FE_TAXONOMY.md. (Horses run once per race, so
    # race-safe == plain cumsum-minus-current here.)
    _counts = _cumulative(
        df[hkey].to_numpy(), df["race_id"].to_numpy(),
        {
            "horse_prev_races": result_start_flag.to_numpy(),
            "horse_prev_finishes": finished_flag.to_numpy(),
            "horse_cum_wins": df["won"].to_numpy(),
            "horse_cum_places": placed_hist.to_numpy(),
            "horse_cum_non_finishes": non_finisher_flag.to_numpy(),
            "horse_cum_npos_sum": norm_pos_hist.to_numpy(),
        },
    )
    df["horse_prev_races"] = _counts["horse_prev_races"].astype(np.int64)
    df["horse_prev_finishes"] = _counts["horse_prev_finishes"].astype(np.int64)

    # Cumulative win rate (vectorised — avoids lambda overhead)
    df["horse_cum_wins"] = _counts["horse_cum_wins"].astype(np.int64)
    df["horse_win_rate"] = _rate(df["horse_cum_wins"], df["horse_prev_races"])
    # Bayesian-shrunk win rate (reliable even with few runs)
    df["horse_win_rate_shrunk"] = _bayesian_shrink(
        df["horse_win_rate"], df["horse_prev_races"],
        prior_rate=0.10, prior_strength=8.0,
    )

    # Cumulative place rate (top 3)
    df["horse_cum_places"] = _counts["horse_cum_places"].astype(np.float32)
    df["horse_place_rate"] = _rate(df["horse_cum_places"], df["horse_prev_races"])
    # Bayesian-shrunk place rate

    df["horse_cum_non_finishes"] = _counts["horse_cum_non_finishes"].astype(np.int64)
    df["horse_non_finish_rate"] = _rate(df["horse_cum_non_finishes"], df["horse_prev_races"])
    df["horse_non_finish_rate_shrunk"] = _bayesian_shrink(
        df["horse_non_finish_rate"], df["horse_prev_races"],
        prior_rate=0.05, prior_strength=10.0,
    )
    df["horse_last_run_non_finish"] = (
        _lag(df, hkey, "_non_finisher_result").fillna(0).astype(np.int8)
    )

    # Override the static scraped lifetime stats with properly computed
    # point-in-time values so the model sees cumulative counts that
    # only reflect prior races, not future ones.
    df["horse_runs"] = df["horse_prev_races"].astype(int)
    df["horse_wins"] = df["horse_cum_wins"].astype(int)
    df["horse_places"] = df["horse_cum_places"].astype(int)

    # Career average position as a share of the field (0 = won, 1 = last,
    # non-finish = last); missing for a debutant.
    df["horse_cum_npos_sum"] = _counts["horse_cum_npos_sum"].astype(np.float64)
    df["horse_avg_position"] = _rate(df["horse_cum_npos_sum"], df["horse_prev_races"])

    # --- Rolling form (Numba-accelerated) ---
    # All on the field-share position (non-finish = last).
    _pos_ops = [(f"horse_avg_pos_{w}", "mean", w) for w in config.ROLLING_WINDOWS]
    _pos_ops.append(("horse_best_pos_5", "min", 5))
    for name, arr in _rolling(df, hkey, "_norm_pos_for_history", _pos_ops).items():
        df[name] = arr

    # Rolling form over no prior runs stays missing: a placeholder (std 3.0,
    # win share 0) is indistinguishable from a real record.

    # --- Days since last race (missing for a debutant, not 60) ---
    df["horse_last_race_date"] = _lag(df, hkey, "race_date")
    df["days_since_last_race"] = (
        (df["race_date"] - df["horse_last_race_date"]).dt.days
    )

    # --- Days since last run — non-linear buckets ---
    df["days_log"] = np.log1p(df["days_since_last_race"])

    # --- Race-type-aware horse features ---
    if "race_type" in df.columns:
        result_start_hist = df["_result_start"].fillna(0.0)
        won_hist = df["_won_for_history"].fillna(0.0)
        _rt = _cumulative(
            _composite_key(df, hkey, "race_type"), df["race_id"].to_numpy(),
            {"horse_rt_runs": result_start_hist.to_numpy(), "horse_rt_wins": won_hist.to_numpy()},
        )
        df["horse_rt_runs"] = _rt["horse_rt_runs"]
        df["horse_rt_wins"] = _rt["horse_rt_wins"]
        df["horse_rt_win_rate"] = _rate(df["horse_rt_wins"], df["horse_rt_runs"])
        df["horse_rt_wr_shrunk"] = _bayesian_shrink(
            df["horse_rt_win_rate"], df["horse_rt_runs"],
            prior_rate=0.10, prior_strength=6.0,
        )
        df["horse_rt_specialist_edge"] = (
            df["horse_rt_wr_shrunk"] - df["horse_win_rate_shrunk"]
        )
        df = _without(df, ["horse_rt_wins"])

    # --- Weight change from previous run ---
    if "weight_lbs" in df.columns:
        df["_observed_weight"] = df["weight_lbs"].where(
            df["has_observed_weight"].eq(1)) if "has_observed_weight" in df else df["weight_lbs"]
        df["weight_change"] = df["_observed_weight"] - _lag(df, hkey, "_observed_weight")
        # Weight-change relative to the field: did THIS horse's burden shift
        # more or less than the race average?  Captures apprentice allowances
        # and handicapping adjustments that narrow/widen vs the competition.
        race_mean_wc = df.groupby("race_id")["weight_change"].transform("mean")
        df["weight_change_vs_field"] = df["weight_change"] - race_mean_wc
        # Binary flag: a drop of 3+ lbs is considered a meaningful relief.
        df = _without(df, ["_observed_weight"])

    # --- Age × form-trend interaction ---
    # Young horses (≤3yo) that are *improving* on recent form represent one
    # of the most reliable angles in racing.  We encode both the trend and
    # the interaction so the model can learn each component separately.
    if "age" in df.columns:
        _age = df["age"].fillna(4).values.astype(np.float32)
        # form_slope > 0 means improving: avg position over last 10 is
        # *worse* (larger) than over last 3 → horse is finishing better recently
        if "horse_avg_pos_3" in df.columns and "horse_avg_pos_10" in df.columns:
            _slope = (
                df["horse_avg_pos_10"].values.astype(np.float32)
                - df["horse_avg_pos_3"].values.astype(np.float32)
            )
            df["form_slope"] = _slope
        # Experience-adjusted age: young horse with many races is unusual
        # (can mean high-quality) vs young horse with few outings.
        df["age_x_prev_races"] = (
            _age * np.log1p(df["horse_prev_races"].fillna(0).values.astype(np.float32))
        )

    # --- Form-string features (if parsed in data_processor) ---
    for col in ["form_last_pos", "form_avg", "form_wins_count", "form_length"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # --- Is maiden (has never won) ---

    # --- Runs since last win (missing for a horse that has never won) ---
    # Starts up to and including each win, carried forward to later runs.
    df["_starts_through_win"] = (df["horse_prev_races"] + 1).where(df["_won_for_history"].eq(1))

    # --- Days since last win (missing for a maiden, not 365) ---
    df["_last_win_date"] = df["race_date"].where(df["won"] == 1)
    df["_last_win_date"] = _ffill_prev(df, hkey, "_last_win_date")
    df["days_since_last_win"] = (
        (df["race_date"] - df["_last_win_date"]).dt.days
    )

    # --- Average lengths behind (Numba-accelerated) ---
    # Non-finishers (PU/F/UR) have NaN lengths_behind — exclude them
    # from rolling aggregates so they don't pollute form signals.
    if "lengths_behind" in df.columns:
        from src.result_margins import margin_values
        _lb_finished = margin_values(df).copy()
        # Mask non-finishers: finish_position <= 0 ⇒ NaN
        if "finish_position" in df.columns:
            _lb_finished = _lb_finished.where(
                df["finish_position"].fillna(0).astype(int) >= 1
            )

        df["_lb_finished"] = _lb_finished

        alpha = 1 - np.exp(-np.log(2) / 3)
        _lb_ops = [("horse_ewma_lb_3", "ewma", alpha)]
        for name, arr in _rolling(df, hkey, "_lb_finished", _lb_ops).items():
            df[name] = arr

        df = _without(df, ["_close_finish", "_lb_finished"])

    # Clean up intermediate columns
    df = _without(df, [
            "horse_cum_wins",
            "horse_cum_places",
            "horse_cum_non_finishes",
            "horse_cum_npos_sum",
            "horse_last_race_date",
            "_last_win_date",
            "_starts_through_win",
        ])

    return df


def add_established_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Comparable marks and evidence from today's actual racing discipline."""
    from src.established_form_features import STATE_KEY, established_form_features
    values, advanced = established_form_features(df, state=_carried(STATE_KEY), return_state=True)
    for column in values:
        df[column] = values[column]
    _carry(STATE_KEY, advanced)
    return df


_PRIOR_MARKET_STATE_KEY = "prior_market::v1"


def add_prior_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Earlier market opinions and rated opposition, with no current-price input."""
    from src.prior_market_features import FEATURES, prior_market_features
    # Direct minimal-frame callers can lack measurements. Keep those unknown;
    # do not add fabricated measurement columns to the caller's dataframe.
    missing = {c: np.nan for c in ("odds", "official_rating", "won", "finish_position", "num_runners") if c not in df}
    work = df.assign(**missing) if missing else df
    values, advanced = prior_market_features(work, state=_carried(_PRIOR_MARKET_STATE_KEY), return_state=True)
    for column in FEATURES:
        df[column] = values[column]
    _carry(_PRIOR_MARKET_STATE_KEY, advanced)
    return df


def add_hidden_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag horses whose pre-race record shows prior runs NOT in our database.

    The classic case: a foreign raider (a French horse at Cheltenham) or a
    point-to-point graduate first appears with ``horse_prev_races == 0`` — so
    every form / rating feature is null and the model treats it as a blank-slate
    debutant and under-rates it (confirmed in the 2026-06 error analysis).

    **Leakage:** only PRE-RACE source fields are used. ``form`` is the record
    coming INTO the race — verified empirically that today's result is NOT in it
    (its most-recent character matches today's finishing position only at chance,
    ~12%). ``days_since_last_run`` is the raw source gap before today (the
    engineered ``days_since_last_race`` is DB-derived and defaults to 60 for
    DB-debutants, so it is unsuitable here). The raw lifetime ``horse_runs`` is
    deliberately NOT used — it is a *retroactive* career total (constant across a
    horse's rows; see line ~1087 where it is overwritten) and would leak the
    future. ``form`` parsing counts position characters (digits + P/F/U/… codes),
    requiring >=2 to clear the single-char placeholder that true debutants carry.
    """
    if "form" not in df.columns and "days_since_last_run" not in df.columns:
        return df
    logger.info("Engineering hidden-form (external-experience) features...")

    if "form" in df.columns:
        prior_form_runs = (df["form"].fillna("").astype(str)
                           .str.replace(r"[^0-9A-Za-z]", "", regex=True).str.len())
    else:
        prior_form_runs = pd.Series(0, index=df.index)
    df["prior_form_runs"] = prior_form_runs.astype(np.int16)

    src_days = (pd.to_numeric(df["days_since_last_run"], errors="coerce").fillna(0.0)
                if "days_since_last_run" in df.columns else pd.Series(0.0, index=df.index))
    db_prior = pd.to_numeric(df.get("horse_prev_races", 0), errors="coerce").fillna(0)

    # "Ran before per the source": a real gap since last run, or >=2 form chars
    # (>=2 avoids the single-char placeholder that true debutants carry).
    ran_before = (src_days > 0) | (df["prior_form_runs"] >= 2)
    # Appears as a DB debutant but has external history -> form we don't hold.
    # Recent runs the form book shows beyond our DB record (also catches
    # lightly-raced-but-experienced horses); capped by the ~6-run form window.
    excess = np.clip(df["prior_form_runs"].to_numpy() - db_prior.to_numpy(), 0, None)
    df["hidden_form_runs"] = np.where(ran_before.to_numpy(), excess, 0).astype(np.int16)
    # Experience corrected for hidden form, so raiders aren't treated as debutants.
    return df


def add_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate track-specific features for each horse."""
    logger.info("Engineering track features...")
    hkey = _horse_key(df)

    # Horse's track / distance / going records — class C, seed-aware so the
    # FeatureStore append carries state (see docs/architecture/FE_TAXONOMY.md). One row per
    # (horse, dim, race), so race-safe == plain cumsum-minus-current.
    result_start_hist = df["_result_start"].fillna(0.0)
    won_hist = df["_won_for_history"].fillna(0.0)
    _race = df["race_id"].to_numpy()
    _rs = result_start_hist.to_numpy()
    _wh = won_hist.to_numpy()

    _trk = _cumulative(_composite_key(df, hkey, "track"), _race,
        {"horse_track_runs": _rs, "horse_track_wins": _wh})
    df["horse_track_runs"] = _trk["horse_track_runs"]
    df["horse_track_wins"] = _trk["horse_track_wins"]
    df["horse_track_win_rate"] = _rate(df["horse_track_wins"], df["horse_track_runs"])
    df["horse_track_wr_shrunk"] = _bayesian_shrink(
        df["horse_track_win_rate"], df["horse_track_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )

    # Distances carry yards (7f, 7.06f, 7.16f), so group on the nearest half
    # furlong, as the course-and-distance record does.
    df["_dist_half_f"] = (df["distance_furlongs"] * 2).round() / 2
    _dst = _cumulative(_composite_key(df, hkey, "_dist_half_f"), _race,
        {"horse_dist_runs": _rs, "horse_dist_wins": _wh})
    df["horse_dist_runs"] = _dst["horse_dist_runs"]
    df["horse_dist_wins"] = _dst["horse_dist_wins"]

    _gng = _cumulative(_composite_key(df, hkey, "going"), _race,
        {"horse_going_runs": _rs, "horse_going_wins": _wh})
    df["horse_going_runs"] = _gng["horse_going_runs"]
    df["horse_going_wins"] = _gng["horse_going_wins"]
    df["horse_going_win_rate"] = _rate(df["horse_going_wins"], df["horse_going_runs"])
    df["horse_going_wr_shrunk"] = _bayesian_shrink(
        df["horse_going_win_rate"], df["horse_going_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )

    df = _without(df, [
            "horse_track_wins",
            "horse_dist_wins",
            "horse_going_wins",
            "_dist_half_f",
        ])
    return df


def add_class_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect class changes between a horse's consecutive races.

    In UK racing, Class 1 is the highest and Class 7 the lowest.
    A horse *dropping* in class (higher number) is often a positive signal;
    being *raised* (lower number) suggests the connections think it's improved
    but the horse faces tougher competition.
    """
    logger.info("Engineering class change features...")
    hkey = _horse_key(df)

    # Extract numeric class value  ("Class 3" → 3, "3" → 3)
    if "race_class" in df.columns:
        from src.feature_semantics import class_number
        df["_class_num"] = class_number(df["race_class"])
    else:
        df["_class_num"] = np.nan

    df["_prev_class"] = _lag(df, hkey, "_class_num")

    # Positive = dropping (easier), Negative = rising (harder)
    df["class_change"] = df["_class_num"] - df["_prev_class"]
    df["class_dropped"] = (df["class_change"] > 0).astype(int)

    # Official-rating change from previous run
    if "official_rating" in df.columns:
        from src.feature_semantics import observed_rating
        df["_observed_mark"] = observed_rating(df)
        df["or_change"] = df["_observed_mark"] - _lag(df, hkey, "_observed_mark")
        # Flat, hurdle and chase marks belong to separate rating scales.
        # Some Flat runners also have distinct turf and all-weather marks.
        # Without both published marks, a surface switch is not a proven rise.
        if "race_type" in df:
            from src.established_form_features import mark_scale
            df["_observed_mark_scale"] = mark_scale(df)
            previous_scale = _lag(df, hkey, "_observed_mark_scale")
            df["or_change"] = df["or_change"].where(df["_observed_mark_scale"].eq(previous_scale))
    else:
        df["or_change"] = np.nan

    df = _without(df, ["_class_num", "_prev_class", "_observed_mark", "_observed_mark_scale"])
    return df


def add_course_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Course & Distance (C&D) winner flag and stats.

    A horse that has previously won at the same track AND distance is a
    proven C&D winner — one of the most respected angles in racing.
    """
    logger.info("Engineering course & distance features...")
    hkey = _horse_key(df)

    # Round distance to nearest 0.5 f for grouping (minor variations)
    df["_dist_rounded"] = (df["distance_furlongs"] * 2).round() / 2

    # --- C&D (same track + same distance) ---
    result_start_hist = df["_result_start"].fillna(0.0)
    won_hist = df["_won_for_history"].fillna(0.0)
    _race = df["race_id"].to_numpy()
    _cd = _cumulative(
        _composite_key(df, hkey, "track", "_dist_rounded"), _race,
        {"horse_cd_runs": result_start_hist.to_numpy(), "horse_cd_wins": won_hist.to_numpy()},
    )
    df["horse_cd_runs"] = _cd["horse_cd_runs"]
    df["horse_cd_wins"] = _cd["horse_cd_wins"]
    df["horse_cd_win_rate"] = _rate(df["horse_cd_wins"], df["horse_cd_runs"])

    # --- Course only (already partially in add_track_features, but
    #     add a "has won at course before" binary for clarity) ---
    df["horse_course_wins"] = _cumulative(
        _composite_key(df, hkey, "track"), _race,
        {"horse_course_wins": won_hist.to_numpy()},
    )["horse_course_wins"]

    df = _without(df, ["_dist_rounded", "horse_cd_wins", "horse_course_wins"])
    return df


def add_distance_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features related to changes in trip distance between races.

    Stepping up or down in distance is one of the most important angles
    in racing — some horses improve dramatically with a trip change.
    """
    logger.info("Engineering distance change features...")
    hkey = _horse_key(df)

    # --- Distance change from previous race ---
    df["_prev_dist"] = _lag(df, hkey, "distance_furlongs")
    # Missing for a debutant: 0 would read as "same trip as last time".
    df["distance_change"] = df["distance_furlongs"] - df["_prev_dist"]

    # --- Distance category ---
    df["dist_category"] = dist_category(df["distance_furlongs"])

    # --- Horse win rate at this distance category ---
    df["_dist_cat_str"] = df["dist_category"].astype(str)
    result_start_hist = df["_result_start"].fillna(0.0)
    won_hist = df["_won_for_history"].fillna(0.0)
    _race = df["race_id"].to_numpy()
    _dc = _cumulative(
        _composite_key(df, hkey, "_dist_cat_str"), _race,
        {"horse_dist_cat_runs": result_start_hist.to_numpy(), "_dc_wins": won_hist.to_numpy()},
    )
    df["horse_dist_cat_runs"] = _dc["horse_dist_cat_runs"]
    df["_dc_wins"] = _dc["_dc_wins"]
    df["horse_dist_cat_wr"] = _rate(df["_dc_wins"], df["horse_dist_cat_runs"])
    df["horse_dist_cat_wr_shrunk"] = _bayesian_shrink(
        df["horse_dist_cat_wr"], df["horse_dist_cat_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )

    # Surface × distance-category record captures horses that are, for example,
    # good AW sprinters but mediocre on turf or over longer trips.
    if "surface" in df.columns:
        _sdc = _cumulative(
            _composite_key(df, hkey, "surface", "_dist_cat_str"), _race,
            {"horse_surface_dist_cat_runs": result_start_hist.to_numpy(), "_sdc_wins": won_hist.to_numpy()},
        )
        df["horse_surface_dist_cat_runs"] = _sdc["horse_surface_dist_cat_runs"]
        df["_sdc_wins"] = _sdc["_sdc_wins"]

    # --- At preferred distance? (best win rate distance category) ---
    # For each horse, find the distance category where they've won most.
    # Use cummax() (not transform("max")) to avoid look-ahead: at each
    # point in time we only know the horse's wins up to NOW, not future.
    # Identical formula to _dc_wins above — reuse to avoid a second sweep and a
    # duplicate state key.
    df["_cum_dist_cat_wins"] = df["_dc_wins"]
    # Running max of cumulative wins within each (horse, dist_cat) group
    df["_hdc_key"] = _composite_key(df, hkey, "_dist_cat_str")
    df["_dc_running_max"] = _running_extreme(df, "_hdc_key", "_cum_dist_cat_wins", kind="max", prev=False)
    # Running max across ALL distance categories for each horse
    df["_best_dc"] = _running_extreme(df, hkey, "_dc_running_max", kind="max", prev=False)
    df = _without(df, ["_hdc_key"])

    df = _without(df, ["_prev_dist", "_dist_cat_str", "_dc_wins",
                 "_cum_dist_cat_wins", "_dc_running_max", "_best_dc", "_sdc_wins"])
    return df


def add_headgear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from headgear information.

    Headgear (blinkers, visors, cheekpieces, etc.) can sharpen a horse's
    focus. *First-time headgear* is one of the most powerful signals in
    racing — it often produces immediate improvement.
    """
    if "headgear" not in df.columns:
        logger.info("No headgear column found — skipping headgear features.")
        # Still create the columns with zeros so the model doesn't break
        df["has_headgear"] = 0
        df["first_time_headgear"] = 0
        return df

    logger.info("Engineering headgear features...")
    hg = df["headgear"].fillna("").astype(str).str.strip().str.lower()

    # Binary: wearing any headgear?
    df["has_headgear"] = (
        hg.ne("") & hg.ne("none") & hg.ne("0") & hg.ne("nan")
    ).astype(int)

    # First-time headgear detection
    hkey = _horse_key(df)
    df["_prev_hg"] = _lag(df, hkey, "has_headgear").fillna(0)
    df["_ever_had_hg"] = _cumulative(
        df[hkey].to_numpy(), df["race_id"].to_numpy(),
        {"_ever_had_hg": df["has_headgear"].to_numpy()},
    )["_ever_had_hg"]
    # First-time = wearing headgear now AND never worn it in any prior race
    df["first_time_headgear"] = (
        (df["has_headgear"] == 1) & (df["_ever_had_hg"] == 0)
    ).astype(int)

    # Headgear removed (wore it last time but not now)

    df = _without(df, ["_prev_hg", "_ever_had_hg"])
    return df


def add_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """Horse performance on the current surface (Turf vs AW/Polytrack)."""
    if "surface" not in df.columns:
        return df

    logger.info("Engineering surface features...")
    hkey = _horse_key(df)

    result_start_hist = df["_result_start"].fillna(0.0)
    won_hist = df["_won_for_history"].fillna(0.0)
    _race = df["race_id"].to_numpy()
    _sf = _cumulative(
        _composite_key(df, hkey, "surface"), _race,
        {"horse_surface_runs": result_start_hist.to_numpy(), "horse_surface_wins": won_hist.to_numpy()},
    )
    df["horse_surface_runs"] = _sf["horse_surface_runs"]
    df["horse_surface_wins"] = _sf["horse_surface_wins"]
    df["horse_surface_win_rate"] = _rate(df["horse_surface_wins"], df["horse_surface_runs"])
    df["horse_surface_wr_shrunk"] = _bayesian_shrink(
        df["horse_surface_win_rate"], df["horse_surface_runs"],
        prior_rate=0.10, prior_strength=6.0,
    )

    # First time on this surface type (e.g. turf horse switching to AW)

    # Surface switch from previous run — captures adaptation cost
    df["_prev_surface"] = _lag(df, hkey, "surface")

    # Combined horse context: surface × race type
    if "race_type" in df.columns:
        _srt = _cumulative(
            _composite_key(df, hkey, "surface", "race_type"), _race,
            {"horse_surface_rt_runs": result_start_hist.to_numpy(), "_surface_rt_wins": won_hist.to_numpy()},
        )
        df["horse_surface_rt_runs"] = _srt["horse_surface_rt_runs"]
        df["_surface_rt_wins"] = _srt["_surface_rt_wins"]
    df = _without(df, ["_prev_surface", "horse_surface_wins", "_surface_rt_wins"])
    return df


def add_speed_figure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive proxy speed figures from ``lengths_behind`` and distance.

    These legacy inputs are margin-based performance proxies, not measured
    speed. Winning-time data exists separately. Use cumulative distance
    behind the winner, never the raw gap to the preceding finisher.

    All rolling windows use ``shift(1)`` to avoid data leakage.
    """
    if "lengths_behind" not in df.columns:
        return df

    logger.info("Engineering speed figure features...")
    hkey = _horse_key(df)

    # Only use lengths_behind from horses that actually finished
    from src.result_margins import margin_values
    _lb_for_speed = margin_values(df).copy()
    if "finish_position" in df.columns:
        _lb_for_speed = _lb_for_speed.where(
            df["finish_position"].fillna(0).astype(int) >= 1
        )

    # Lengths normalized for distance with non-linear power-law scaling (2c)
    # This reflects that margin dispersion does not scale linearly with distance;
    # 1 length in a 5f sprint requires more energy/speed output difference than in a 24f chase.
    safe_dist = df["distance_furlongs"].replace(0, np.nan).fillna(8.0)
    df["_lb_per_furlong"] = _lb_for_speed.fillna(np.nan) / np.power(safe_dist, 0.6)

    # Per-race relative performance: horse vs race median (neg = better)
    race_median_lpf = df.groupby("race_id")["_lb_per_furlong"].transform("median")
    df["_rel_perf"] = df["_lb_per_furlong"] - race_median_lpf

    # --- Numba-accelerated rolling speed figures ---
    _speed_ops = [
        ("speed_fig_avg_5", "mean", 5),
        ("speed_fig_avg_10", "mean", 10),
        ("speed_fig_best_5", "min", 5),
    ]
    for name, arr in _rolling(df,hkey, "_lb_per_furlong", _speed_ops).items():
        df[name] = arr

    # Trend: recent avg minus long-term avg (negative = improving)
    df["speed_fig_trend"] = df["speed_fig_avg_5"] - df["speed_fig_avg_10"]

    # Relative performance rolling avg (negative = above average)
    _rel_ops = [("speed_fig_rel_avg_5", "mean", 5)]
    for name, arr in _rolling(df,hkey, "_rel_perf", _rel_ops).items():
        df[name] = arr

    # ── Class-adjusted speed figures ─────────────────────────────
    if "race_class" in df.columns:
        from src.feature_semantics import class_number
        _class_num = class_number(df["race_class"])
        _quality = (1.0 - (_class_num - 1) * 0.1).clip(0.4, 1.0)
        # Additive class-based penalty instead of multiplicative adjustment (2a)
        # Multiplicative adjustment leaves all winners (lengths_behind=0.0) at 0.0,
        # which fails to distinguish a Class 1 winner from a Class 7 winner.
        # We apply an additive penalty proportional to the quality gap (max penalty 0.5 units).
        # A non-finisher or unavailable margin is not a zero-length defeat.
        # Preserve missingness so it cannot become the horse's best speed run.
        df["_adj_lpf"] = df["_lb_per_furlong"] + (1.0 - _quality) * 0.5

        _adj_ops = [("speed_fig_adj_best_5", "min", 5)]
        for name, arr in _rolling(df,hkey, "_adj_lpf", _adj_ops).items():
            df[name] = arr
        df = _without(df, ["_adj_lpf"])

        # ── Class-relative Z-score speed figures ─────────────────
        # shift(1) so each row's own lb_per_furlong is excluded from the class
        # median / std (avoids current-race outcome leakage). Seed-aware via
        # _expanding_prev (full per-class history — option A, exact).
        df["_class_key"] = _class_num.astype("Int64").astype("string").fillna("Unknown")
        _exp = _expanding_prev(df, "_class_key", "_lb_per_furlong", ["median", "std"])
        # Every performance on a date uses the same prior-day class scale.
        # A row-wise shift alone lets earlier finishers from this very race
        # enter the scale, so shuffling source result rows changes future
        # horse features. Gather by position (not first non-null value).
        _positions = pd.Series(np.arange(len(df)), index=df.index)
        _first = _positions.groupby(
            [df["_class_key"], pd.to_datetime(df["race_date"]).dt.normalize()], sort=False,
        ).transform("min").to_numpy(dtype=int)
        _class_median = pd.Series(_exp["median"].to_numpy()[_first], index=df.index)
        _class_std = pd.Series(_exp["std"].to_numpy()[_first], index=df.index).replace(0, np.nan).fillna(1.0)
        df["_speed_z_class"] = ((df["_lb_per_furlong"] - _class_median) / _class_std).where(_class_num.notna())
        df = _without(df, ["_class_key"])

        # Rolling Z-score averages (Numba-accelerated)
        _z_ops = [("speed_z_class_best_5", "min", 5)]
        for name, arr in _rolling(df,hkey, "_speed_z_class", _z_ops).items():
            df[name] = arr
        df = _without(df, ["_speed_z_class"])

    df = _without(df, ["_lb_per_furlong", "_rel_perf"])
    return df


def add_prize_money_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate prize money features capturing class intent and competition level."""
    if "prize_money" not in df.columns:
        return df

    logger.info("Engineering prize money features...")
    hkey = _horse_key(df)

    # Fill NaN with 0 (before grouping so the groupby sees clean values)
    df["prize_money"] = df["prize_money"].fillna(0)

    # Maximum purse the horse has CONTESTED, excluding the current race;
    # missing for a debutant.
    # NB: the shift MUST be inside transform — a bare
    # groupby().cummax().shift(1) shifts globally across horses and, with
    # raw rows in finish order, leaks finish position (CLAUDE.md rule 1).
    df["horse_max_prize_contested"] = _running_extreme(
        df, hkey, "prize_money", kind="max", prev=True
    )
    
    # Cumsum / count to avoid expanding().mean() issues:
    df["_safe_cum_prize"] = _cumulative(
        df[hkey].to_numpy(), df["race_id"].to_numpy(),
        {"_safe_cum_prize": df["prize_money"].to_numpy()},
    )["_safe_cum_prize"]
    hp_races = df.get("horse_prev_races", pd.Series(0, index=df.index))
    df["horse_avg_prize_contested"] = _rate(df["_safe_cum_prize"], hp_races)

    # Change vs typical class on a log scale. Purses are heavily skewed
    # (median about £5k, 99th percentile about £94k), so a £ difference is
    # dominated by the few big races. An unknown purse (0) leaves it missing.
    current = df["prize_money"].where(df["prize_money"] > 0)
    typical = df["horse_avg_prize_contested"].where(df["horse_avg_prize_contested"] > 0)
    df["prize_money_change_vs_avg"] = np.log1p(current) - np.log1p(typical)
    
    # Flag for severe class jumps based on money
    
    df = _without(df, ["_safe_cum_prize"])
    return df


def add_collateral_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Collateral form: how did the rivals from this horse's previous race
    perform in their own NEXT start?

    A horse beaten in a race whose other runners went on to win next time
    out ran better than the bare result suggests ("key race"). Uses only
    rival next-out results dated strictly BEFORE the current race, so no
    future information leaks despite the per-horse forward shift.

    Columns:
    - ``collateral_n_rivals``        — rivals with a known next-out result
    - ``collateral_next_win_rate``   — their next-out win rate
    - ``collateral_next_place_rate`` — their next-out top-3 rate
    - ``collateral_next_win_rate_vs_field`` — relative to today's field
    """
    if not getattr(config, "COLLATERAL_FORM", True):
        return df
    if "finish_position" not in df.columns or "race_id" not in df.columns:
        return df
    logger.info("Engineering collateral-form features...")

    hkey = _horse_key(df)
    groups = df.groupby(hkey, sort=False)

    # Each runner row annotated with that horse's NEXT run outcome.
    # Forward-looking by construction — only consumed under a date guard.
    next_date = groups["race_date"].shift(-1)
    next_fp = pd.to_numeric(groups["finish_position"].shift(-1), errors="coerce")
    rivals = pd.DataFrame({
        "race_id": df["race_id"].values,
        "rival_key": df[hkey].values,
        "next_date": next_date.values,
        "next_won": (next_fp == 1).astype(float).values,
        "next_placed": next_fp.between(1, 3).astype(float).values,
    })

    prev_race_id = groups["race_id"].shift(1)
    cur = pd.DataFrame({
        "row": np.arange(len(df)),
        "prev_race_id": prev_race_id.values,
        "self_key": df[hkey].values,
        "cur_date": df["race_date"].values,
    }).dropna(subset=["prev_race_id"])

    merged = cur.merge(rivals, left_on="prev_race_id", right_on="race_id")
    merged = merged[
        (merged["rival_key"] != merged["self_key"])
        & merged["next_date"].notna()
        & (merged["next_date"] < merged["cur_date"])
    ]
    agg = merged.groupby("row").agg(
        collateral_n_rivals=("next_won", "size"),
        collateral_next_win_rate=("next_won", "mean"),
        collateral_next_place_rate=("next_placed", "mean"),
    )

    out = agg.reindex(np.arange(len(df)))
    n_rivals = out["collateral_n_rivals"].fillna(0).astype(np.float32)
    win_rate = out["collateral_next_win_rate"].astype(np.float32)
    place_rate = out["collateral_next_place_rate"].astype(np.float32)

    race_ids = df["race_id"].values
    wr_ser = pd.Series(win_rate.values, index=df.index)
    wr_field = wr_ser.groupby(race_ids).transform("mean")

    df = pd.concat([df, pd.DataFrame({
        "collateral_n_rivals": n_rivals.values,
        "collateral_next_win_rate": win_rate.values,
        "collateral_next_place_rate": place_rate.values,
        "collateral_next_win_rate_vs_field": (wr_ser - wr_field).to_numpy(),
    }, index=df.index)], axis=1)
    return df


def add_supplementary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Additional derived features: consistency, class trajectory,
    campaign intensity, seasonal, field composition, market intelligence."""
    logger.info("Engineering supplementary features...")
    hkey = _horse_key(df)

    # ── Campaign Intensity ─────────────────────────────────────
    if "race_date" in df.columns:
        count_source = "_result_start_for_history" if "_result_start_for_history" in df.columns else "_ones"
        if count_source == "_ones":
            df["_ones"] = 1.0
        # Sum the start flags rather than count rows: a non-start is not a run.
        starts_14, _ = _time_window(df, hkey, "race_date", count_source, 14)
        starts_30, _ = _time_window(df, hkey, "race_date", count_source, 30)
        df = _without(df, ["_ones"])

    # ── Trainer Switch ──
    if "trainer" in df.columns:
        df["_prev_trainer"] = _lag(df, hkey, "trainer")
        df = _without(df, ["_prev_trainer"])

    # ── Same-trainer runners in race ──
    if "trainer" in df.columns:
        df["same_trainer_runners"] = df.groupby(
            ["race_id", "trainer"]
        )["trainer"].transform("count")

    # ── Market vs Elo mispricing ──
    if "odds_rank" in df.columns and "horse_elo_rank" in df.columns:
        # Positive = market rates horse lower than Elo suggests
        df["odds_vs_elo_rank"] = df["horse_elo_rank"] - df["odds_rank"]

    # ── Beaten-lengths consistency ───────────────────────────────
    if "lengths_behind" in df.columns:
        _lb_cons = _rolling(
            df, hkey, "beaten_lengths" if "beaten_lengths" in df else "lengths_behind",
            [("horse_lb_stddev_5", "std", 5)],
        )
        df["horse_lb_stddev_5"] = _lb_cons["horse_lb_stddev_5"]

    return df
