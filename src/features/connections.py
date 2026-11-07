"""Jockey and trainer stages: records, recent form, combinations,
target encodings, stable intent and booking signals.

Moved out of ``src/feature_engineer.py`` on 2026-11-07 (research 065).
"""

import ast
import logging

import numpy as np
import pandas as pd

import config
from src.features.common import _bayesian_shrink, _horse_key, _rate, _shrink_to_entity, _without
from src.features.history import (
    _composite_key,
    _cumulative,
    _entity_day_start_view,
    _lag,
    _rolling,
    _running_extreme,
    _time_window,
)

logger = logging.getLogger(__name__)


def add_jockey_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate jockey-specific historical features."""
    logger.info("Engineering jockey features...")

    result_start_flag = df["_result_start"]
    placed_hist = df["_placed_for_history"].fillna(0.0)

    # Cumulative (prior-only, race-safe) counts via the incremental primitive
    # (state=None == inline cumsum-minus-current; seed-capable for append).
    # A jockey rides once per race, so race-safe == plain. Class C.
    _counts = _cumulative(
        df["jockey"].to_numpy(), df["race_id"].to_numpy(),
        {
            "jockey_prev_rides": result_start_flag.to_numpy(),
            "jockey_cum_wins": df["won"].to_numpy(),
            "jockey_cum_places": placed_hist.to_numpy(),
        },
    )

    # Number of previous rides
    df["jockey_prev_rides"] = _counts["jockey_prev_rides"].astype(np.int64)

    # Cumulative win rate (vectorised)
    df["jockey_cum_wins"] = _counts["jockey_cum_wins"].astype(np.int64)
    df["jockey_win_rate"] = _rate(df["jockey_cum_wins"], df["jockey_prev_rides"])
    df["jockey_win_rate_shrunk"] = _bayesian_shrink(
        df["jockey_win_rate"], df["jockey_prev_rides"],
        prior_rate=0.10, prior_strength=10.0,
    )

    # Rolling form  (Numba-accelerated)
    _jock_won_ops = []
    _jock_pos_ops = []
    for w in [10, 20]:
        _jock_won_ops.append((f"jockey_wins_{w}", "sum", w))
        _jock_pos_ops.append((f"jockey_avg_pos_{w}", "mean", w))

    for col, ops in [("_won_for_history", _jock_won_ops), ("_norm_pos_for_history", _jock_pos_ops)]:
        results = _rolling(df,"jockey", col, ops)
        for name, arr in results.items():
            df[name] = arr

    # Place rate (top 3)
    df["jockey_cum_places"] = _counts["jockey_cum_places"].astype(np.float32)
    df["jockey_place_rate"] = _rate(df["jockey_cum_places"], df["jockey_prev_rides"])
    df["jockey_place_rate_shrunk"] = _bayesian_shrink(
        df["jockey_place_rate"], df["jockey_prev_rides"],
        prior_rate=0.30, prior_strength=10.0,
    )

    # ── Time-based windows (14d, 30d) ────────────────────────────
    # Ride-count windows can span weeks for busy jockeys but months
    # for quiet ones.  Calendar windows capture "hot streaks" better.
    df["race_date"] = pd.to_datetime(df["race_date"])
    for days in [14, 30]:
        _wins_col = f"jockey_wins_{days}d"
        _rides_col = f"jockey_rides_{days}d"
        _wr_col = f"jockey_wr_{days}d"
        _places_col = f"jockey_places_{days}d"
        _pr_col = f"jockey_pr_{days}d"

        w_arr, _ = _time_window(df, "jockey", "race_date", "_won_for_history", days)
        r_arr, _ = _time_window(df, "jockey", "race_date", "_result_start_for_history", days)
        p_arr, _ = _time_window(df, "jockey", "race_date", "_placed_for_history", days)
        df[_wins_col] = w_arr
        df[_rides_col] = r_arr
        df[_places_col] = p_arr
        with np.errstate(invalid="ignore"):
            df[_wr_col] = np.where(r_arr > 2, w_arr / r_arr, 0)
            df[_pr_col] = np.where(r_arr > 2, p_arr / r_arr, 0)

    for days in [14, 30]:
        df[f"jockey_wr_{days}d_delta"] = (
            df[f"jockey_wr_{days}d"] - df["jockey_win_rate_shrunk"]
        )
        df[f"jockey_pr_{days}d_delta"] = (
            df[f"jockey_pr_{days}d"] - df["jockey_place_rate_shrunk"]
        )

    df["jockey_hot_vs_30d"] = df["jockey_wr_14d"] - df["jockey_wr_30d"]
    df = _without(df, ["jockey_cum_wins", "jockey_cum_places"])
    df = _entity_day_start_view(df, ["jockey"], [
        "jockey_prev_rides", "jockey_win_rate", "jockey_win_rate_shrunk",
        "jockey_wins_10", "jockey_wins_20", "jockey_avg_pos_10", "jockey_avg_pos_20",
        "jockey_place_rate", "jockey_place_rate_shrunk",
        "jockey_wins_14d", "jockey_rides_14d", "jockey_places_14d",
        "jockey_wr_14d", "jockey_pr_14d",
        "jockey_wins_30d", "jockey_rides_30d", "jockey_places_30d",
        "jockey_wr_30d", "jockey_pr_30d",
        "jockey_wr_14d_delta", "jockey_pr_14d_delta",
        "jockey_wr_30d_delta", "jockey_pr_30d_delta", "jockey_hot_vs_30d",
    ])
    return df


def add_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trainer-specific historical features."""
    logger.info("Engineering trainer features...")

    placed_hist = df["_placed_for_history"].fillna(0.0)

    # Race-safe: trainers often have multiple runners in the same race.
    # All same-race runners must see identical prior-race stats. Routed through
    # the seed-aware cumulative primitive (class C; see docs/architecture/FE_TAXONOMY.md) so
    # the FeatureStore append can carry state instead of recomputing history.
    _tc = _cumulative(
        df["trainer"].to_numpy(), df["race_id"].to_numpy(),
        {
            "trainer_prev_runs": df["_result_start"].to_numpy(),
            "trainer_cum_wins": df["won"].to_numpy(),
            "trainer_cum_places": placed_hist.to_numpy(),
            # Every earlier entry, settled or not (non-runners and pending
            # cards included). Stablemates share one count; the store carries
            # it like the other counts, so an append or a live card continues
            # it. Only its within-race relatives are model inputs.
            "trainer_freq": np.ones(len(df)),
        },
    )
    df["trainer_freq"] = _tc["trainer_freq"] + 1
    df["trainer_prev_runs"] = _tc["trainer_prev_runs"]
    df["trainer_cum_wins"] = _tc["trainer_cum_wins"]
    df["trainer_win_rate"] = _rate(df["trainer_cum_wins"], df["trainer_prev_runs"])
    df["trainer_win_rate_shrunk"] = _bayesian_shrink(
        df["trainer_win_rate"], df["trainer_prev_runs"],
        prior_rate=0.10, prior_strength=10.0,
    )

    # Rolling form  (Numba-accelerated)
    _trainer_won_ops = []
    for w in [10, 20]:
        _trainer_won_ops.append((f"trainer_wins_{w}", "sum", w))
    results = _rolling(df,"trainer", "won", _trainer_won_ops)
    for name, arr in results.items():
        df[name] = arr

    # Place rate
    df["trainer_cum_places"] = _tc["trainer_cum_places"]
    df["trainer_place_rate"] = _rate(df["trainer_cum_places"], df["trainer_prev_runs"])
    df["trainer_place_rate_shrunk"] = _bayesian_shrink(
        df["trainer_place_rate"], df["trainer_prev_runs"],
        prior_rate=0.30, prior_strength=10.0,
    )

    # ── Time-based windows (14d, 30d) ────────────────────────────
    df["race_date"] = pd.to_datetime(df["race_date"])
    for days in [14, 30]:
        _wins_col = f"trainer_wins_{days}d"
        _runs_col = f"trainer_runs_{days}d"
        _wr_col = f"trainer_wr_{days}d"
        _places_col = f"trainer_places_{days}d"
        _pr_col = f"trainer_pr_{days}d"

        w_arr, _ = _time_window(df, "trainer", "race_date", "_won_for_history", days)
        r_arr, _ = _time_window(df, "trainer", "race_date", "_result_start_for_history", days)
        p_arr, _ = _time_window(df, "trainer", "race_date", "_placed_for_history", days)
        df[_wins_col] = w_arr
        df[_runs_col] = r_arr
        df[_places_col] = p_arr
        with np.errstate(invalid="ignore"):
            df[_wr_col] = np.where(r_arr > 2, w_arr / r_arr, 0)
            df[_pr_col] = np.where(r_arr > 2, p_arr / r_arr, 0)

    for days in [14, 30]:
        df[f"trainer_wr_{days}d_delta"] = (
            df[f"trainer_wr_{days}d"] - df["trainer_win_rate_shrunk"]
        )
        df[f"trainer_pr_{days}d_delta"] = (
            df[f"trainer_pr_{days}d"] - df["trainer_place_rate_shrunk"]
        )

    df["trainer_hot_vs_30d"] = df["trainer_wr_14d"] - df["trainer_wr_30d"]

    df = _without(df, ["trainer_cum_wins", "trainer_cum_places"])
    df = _entity_day_start_view(df, ["trainer"], [
        "trainer_prev_runs", "trainer_win_rate", "trainer_win_rate_shrunk",
        "trainer_wins_10", "trainer_wins_20",
        "trainer_place_rate", "trainer_place_rate_shrunk",
        "trainer_wins_14d", "trainer_runs_14d", "trainer_places_14d",
        "trainer_wr_14d", "trainer_pr_14d",
        "trainer_wins_30d", "trainer_runs_30d", "trainer_places_30d",
        "trainer_wr_30d", "trainer_pr_30d",
        "trainer_wr_14d_delta", "trainer_pr_14d_delta",
        "trainer_wr_30d_delta", "trainer_pr_30d_delta", "trainer_hot_vs_30d",
    ])

    # --- Trainer at track (race-safe) ---
    if "track" in df.columns:
        # Create a combined key so _race_safe helpers can group correctly
        df["_tt_key"] = df["trainer"].astype(str) + "||" + df["track"].astype(str)
        _ttc = _cumulative(
            df["_tt_key"].to_numpy(), df["race_id"].to_numpy(),
            {"trainer_track_runs": df["_result_start"].to_numpy(), "trainer_track_wins": df["won"].to_numpy()},
        )
        df["trainer_track_runs"] = _ttc["trainer_track_runs"]
        df["trainer_track_wins"] = _ttc["trainer_track_wins"]
        df["trainer_track_win_rate"] = _rate(df["trainer_track_wins"], df["trainer_track_runs"])
        df = _entity_day_start_view(
            df, ["_tt_key"], ["trainer_track_runs", "trainer_track_win_rate"]
        )
        df["trainer_track_win_rate"] = _shrink_to_entity(
            df, "trainer_track_win_rate", "trainer_track_runs", "trainer"
        )
        df = _without(df, ["trainer_track_wins", "_tt_key"])

    return df


def add_jockey_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate jockey performance at specific tracks."""
    if "track" not in df.columns:
        return df

    logger.info("Engineering jockey-at-track features...")
    df["_jt_key"] = df["jockey"].astype(str) + "||" + df["track"].astype(str)
    _jtkc = _cumulative(
        df["_jt_key"].to_numpy(), df["race_id"].to_numpy(),
        {"jockey_track_runs": df["_result_start"].to_numpy(), "jockey_track_wins": df["won"].to_numpy()},
    )
    df["jockey_track_runs"] = _jtkc["jockey_track_runs"]
    df["jockey_track_wins"] = _jtkc["jockey_track_wins"]
    df["jockey_track_win_rate"] = _rate(df["jockey_track_wins"], df["jockey_track_runs"])
    df = _entity_day_start_view(
        df, ["_jt_key"], ["jockey_track_runs", "jockey_track_win_rate"]
    )
    df["jockey_track_win_rate"] = _shrink_to_entity(
        df, "jockey_track_win_rate", "jockey_track_runs", "jockey"
    )
    df = _without(df, ["jockey_track_wins", "_jt_key"])
    return df


def add_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate jockey–trainer combination statistics.

    Certain jockey-trainer partnerships outperform their individual stats.
    This captures the synergy (or lack of) between the two.
    """
    logger.info("Engineering jockey-trainer combo features...")
    df["_jt_combo"] = df["jockey"].astype(str) + " | " + df["trainer"].astype(str)
    placed_hist = df["_placed_for_history"].fillna(0.0)

    # Race-safe: a jockey-trainer combo can (rarely) have multiple
    # runners in the same race (different jockeys for same trainer,
    # but the *combo* key makes this less frequent).  Stay consistent.
    # Seed-aware cumulative primitive (class C; see docs/architecture/FE_TAXONOMY.md).
    _jtc = _cumulative(
        df["_jt_combo"].to_numpy(), df["race_id"].to_numpy(),
        {
            "jt_prev_runs": df["_result_start"].to_numpy(),
            "jt_cum_wins": df["won"].to_numpy(),
            "jt_cum_places": placed_hist.to_numpy(),
        },
    )
    df["jt_prev_runs"] = _jtc["jt_prev_runs"]
    df["jt_cum_wins"] = _jtc["jt_cum_wins"]
    df["jt_win_rate"] = _rate(df["jt_cum_wins"], df["jt_prev_runs"])
    df["jt_win_rate_shrunk"] = _bayesian_shrink(
        df["jt_win_rate"], df["jt_prev_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )

    # Combo place rate (top 3)
    df["jt_cum_places"] = _jtc["jt_cum_places"]
    df["jt_place_rate"] = _rate(df["jt_cum_places"], df["jt_prev_runs"])
    df["jt_place_rate_shrunk"] = _bayesian_shrink(
        df["jt_place_rate"], df["jt_prev_runs"],
        prior_rate=0.30, prior_strength=5.0,
    )

    df = _entity_day_start_view(df, ["_jt_combo"], [
        "jt_prev_runs", "jt_win_rate", "jt_win_rate_shrunk",
        "jt_place_rate", "jt_place_rate_shrunk",
    ])
    df = _without(df, ["_jt_combo", "jt_cum_wins", "jt_cum_places"])

    # --- Is‑stable‑jockey booking signal ---
    # Trainers have a preferred jockey — the one they book most often.
    # When that jockey rides again it is a "stable jockey" booking, which
    # correlates with higher trainer intent (they chose their best pilot).
    # We derive this from prior ride history only (no look-ahead leakage).
    if "trainer" in df.columns and "jockey" in df.columns:
        # Count prior rides per (trainer, jockey) pair — race-safe so that
        # a trainer's multiple runners in the same race share identical priors.
        df["_tjkey"] = df["trainer"].astype(str) + "||" + df["jockey"].astype(str)
        df["_tj_prior_rides"] = _cumulative(
            df["_tjkey"].to_numpy(), df["race_id"].to_numpy(),
            {"_tj_prior_rides": df["_result_start"].to_numpy()},
        )["_tj_prior_rides"]

        # For each race row, find the jockey with the most prior rides for
        # this trainer.  We do this by: (a) for every row record the
        # (trainer, jockey, prior_rides) triple, then (b) for each trainer
        # take the rolling argmax prior to the current race.
        # Building a per-trainer "most‑used jockey so far" requires a
        # careful implementation to avoid leakage.  We achieve it by:
        #   1. groupby trainer, sort by race index
        #   2. expanding cummax of _tj_prior_rides within trainer
        #   3. is_stable_jockey = 1 when own prior rides == trainer's cummax
        trainer_max_tj = _running_extreme(df, "trainer", "_tj_prior_rides", kind="max", prev=False)
        df["is_stable_jockey"] = (
            (df["_tj_prior_rides"] == trainer_max_tj) & (df["_tj_prior_rides"] > 0)
        ).astype(np.int8)

        df = _without(df, ["_tjkey", "_tj_prior_rides"])

    return df


def add_jockey_intent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features capturing jockey switches, booking upgrades, and
    equipment-change signals.

    Connections booking a higher-rated jockey signals intent to win.
    Equipment changes (blinkers first time, tongue strap first time)
    are among the most predictive single-race signals in racing.
    """
    logger.info("Engineering jockey intent features...")
    hkey = _horse_key(df)

    # --- Jockey switch (different jockey from last run) ---
    df["_prev_jockey"] = _lag(df, hkey, "jockey")
    df["jockey_switch"] = (
        df["_prev_jockey"].notna() & (df["jockey"] != df["_prev_jockey"])
    ).astype(int)

    # --- Jockey upgrade (current jockey Elo > previous jockey Elo) ---
    if "jockey_elo" in df.columns:
        df["_prev_jockey_elo"] = _lag(df, hkey, "jockey_elo")
        df["jockey_elo_change"] = (
            (df["jockey_elo"] - df["_prev_jockey_elo"]).fillna(0)
        )
        df["jockey_upgrade"] = (
            (df["jockey_switch"] == 1) & (df["jockey_elo_change"] > 50)
        ).astype(int)
        df = _without(df, ["_prev_jockey_elo"])

    # --- Beaten favourite last time (bounce-back angle) ---
    if "is_favourite" in df.columns:
        df["_prev_fav"] = _lag(df, hkey, "is_favourite").fillna(0)
        df["_prev_won"] = _lag(df, hkey, "won").fillna(0)
        df["beaten_fav_last"] = (
            (df["_prev_fav"] == 1) & (df["_prev_won"] == 0)
        ).astype(int)
        df = _without(df, ["_prev_fav", "_prev_won"])

    # ── Granular equipment signals ───────────────────────────────
    # The headgear column is stored as a JSON-like list of dicts:
    # [{"symbol": "b", "name": "Blinkers", "count": 1}, ...]
    # count=1 means first-time application — the strongest signal.
    if "headgear" in df.columns:
        hg_raw = df["headgear"].fillna("").astype(str)

        # Parse the structured headgear data
        def _parse_hg(val):
            """Extract (set of symbols, dict of symbol→count)."""
            if not val or val in ("nan", "None", "", "0"):
                return set(), {}
            try:
                items = ast.literal_eval(val)
                if isinstance(items, list):
                    symbols = {d.get("symbol", "") for d in items if isinstance(d, dict)}
                    counts = {
                        d.get("symbol", ""): d.get("count", 999)
                        for d in items if isinstance(d, dict)
                    }
                    return symbols, counts
            except (ValueError, SyntaxError):
                pass
            return set(), {}

        unique_hg = pd.Index(hg_raw.unique())
        parsed_map = {val: _parse_hg(val) for val in unique_hg}
        parsed = hg_raw.map(parsed_map)
        symbols_series = parsed.map(lambda x: x[0])

        # Blinkers first time (count == 1)

        # Tongue strap first time

        # Cheekpieces first time

        # Visor first time

        # Any first-time equipment application

        # Total number of equipment items worn
        df["equip_count"] = symbols_series.map(len).astype(int)

    df = _without(df, ["_prev_jockey"])
    return df


def add_trainer_specialisation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trainer win rates by distance category and going.

    Some trainers specialise heavily — e.g. sprint specialists,
    trainers who target soft ground, etc.
    """
    logger.info("Engineering trainer specialisation features...")
    # --- Trainer × distance category (race-safe) ---
    if "dist_category" in df.columns:
        df["_dc_str"] = df["dist_category"].astype(str)
        df["_tdc_key"] = df["trainer"].astype(str) + "||" + df["_dc_str"]
        _tdcc = _cumulative(
            df["_tdc_key"].to_numpy(), df["race_id"].to_numpy(),
            {"trainer_dist_cat_runs": df["_result_start"].to_numpy(), "_tdc_wins": df["won"].to_numpy()},
        )
        df["trainer_dist_cat_runs"] = _tdcc["trainer_dist_cat_runs"]
        df["_tdc_wins"] = _tdcc["_tdc_wins"]
        df["trainer_dist_cat_wr"] = _rate(df["_tdc_wins"], df["trainer_dist_cat_runs"])
        df = _entity_day_start_view(
            df, ["_tdc_key"], ["trainer_dist_cat_runs", "trainer_dist_cat_wr"]
        )
        df["trainer_dist_cat_wr"] = _shrink_to_entity(
            df, "trainer_dist_cat_wr", "trainer_dist_cat_runs", "trainer"
        )
        df = _without(df, ["_dc_str", "_tdc_wins", "_tdc_key"])

    # --- Trainer × going (race-safe) ---
    if "going" in df.columns:
        df["_tg_key"] = df["trainer"].astype(str) + "||" + df["going"].astype(str)
        _tgc = _cumulative(
            df["_tg_key"].to_numpy(), df["race_id"].to_numpy(),
            {"trainer_going_runs": df["_result_start"].to_numpy(), "_tg_wins": df["won"].to_numpy()},
        )
        df["trainer_going_runs"] = _tgc["trainer_going_runs"]
        df["_tg_wins"] = _tgc["_tg_wins"]
        df["trainer_going_wr"] = _rate(df["_tg_wins"], df["trainer_going_runs"])
        df = _entity_day_start_view(
            df, ["_tg_key"], ["trainer_going_runs", "trainer_going_wr"]
        )
        df["trainer_going_wr"] = _shrink_to_entity(
            df, "trainer_going_wr", "trainer_going_runs", "trainer"
        )
        df = _without(df, ["_tg_wins", "_tg_key"])

    return df


def add_trainer_runner_intent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trainer multi-runner and first-string intent features."""
    logger.info("Engineering trainer multi-runner intent features...")
    df = df.copy()

    required = {"race_id", "trainer"}
    if not required.issubset(df.columns):
        return df

    trainer_runners = df.groupby(["race_id", "trainer"])["trainer"].transform("count")
    df["same_trainer_runners"] = trainer_runners
    df["trainer_has_multiple_runners"] = (trainer_runners > 1).astype(np.int8)

    if "official_rating" in df.columns:
        # Observed marks only (an unrated runner has no place in the OR order).
        from src.feature_semantics import observed_rating
        _mark = observed_rating(df)
        _stable = [df["race_id"], df["trainer"]]
        df["trainer_runner_rank_by_or"] = _mark.groupby(_stable).rank(
            ascending=False,
            method="min",
        )
        df["trainer_first_string_by_or"] = (
            (df["trainer_has_multiple_runners"] == 1)
            & (df["trainer_runner_rank_by_or"] == 1)
        ).astype(np.int8)
        stable_best_or = _mark.groupby(_stable).transform("max")
        df["trainer_or_vs_best_stablemate"] = _mark - stable_best_or

    if "horse_elo" in df.columns:
        df["trainer_runner_rank_by_elo"] = df.groupby(["race_id", "trainer"])["horse_elo"].rank(
            ascending=False,
            method="min",
        )
        df["trainer_first_string_by_elo"] = (
            (df["trainer_has_multiple_runners"] == 1)
            & (df["trainer_runner_rank_by_elo"] == 1)
        ).astype(np.int8)

    if "odds" in df.columns:
        stable_group = df.groupby(["race_id", "trainer"], sort=False)
        df["trainer_runner_rank_by_odds"] = stable_group["odds"].rank(
            ascending=True,
            method="min",
        )
        df["trainer_first_string_by_odds"] = (
            (df["trainer_has_multiple_runners"] == 1)
            & (df["trainer_runner_rank_by_odds"] == 1)
        ).astype(np.int8)

        stable_min_odds = stable_group["odds"].transform("min")
        df["trainer_odds_vs_stablemate_best"] = df["odds"] - stable_min_odds

        stable_implied = (1.0 / df["odds"].replace(0, np.nan)).fillna(0.0)
        stable_total_implied = stable_implied.groupby([df["race_id"], df["trainer"]]).transform("sum")
        df["trainer_stable_market_share"] = np.divide(
            stable_implied,
            stable_total_implied,
            out=np.zeros(len(df), dtype=np.float64),
            where=stable_total_implied > 0,
        )

        if "official_rating" in df.columns:
            df["trainer_first_string_odds_or_agree"] = (
                (df["trainer_first_string_by_odds"] == 1)
                & (df.get("trainer_first_string_by_or", 0) == 1)
            ).astype(np.int8)
            df["trainer_string_disagreement_or_odds"] = (
                df["trainer_runner_rank_by_odds"]
                - df["trainer_runner_rank_by_or"]
            ).abs()

        if "horse_elo" in df.columns:
            df["trainer_first_string_odds_elo_agree"] = (
                (df["trainer_first_string_by_odds"] == 1)
                & (df.get("trainer_first_string_by_elo", 0) == 1)
            ).astype(np.int8)
            df["trainer_string_disagreement_elo_odds"] = (
                df["trainer_runner_rank_by_odds"]
                - df["trainer_runner_rank_by_elo"]
            ).abs()

        if "is_stable_jockey" in df.columns:
            df["trainer_first_string_stable_jockey"] = (
                df["trainer_first_string_by_odds"] * df["is_stable_jockey"]
            ).astype(np.int8)

    # ── Jockey-booking strength ──────────────────────────────────────
    # Is the booked jockey stronger than the horse's market position implies
    # ("top jockey on an outsider" — connections don't waste a first-choice
    # booking)?
    if "jockey_elo" in df.columns:
        if "odds" in df.columns:
            # Percentile ranks within the race: 1.0 = best jockey / most
            # fancied horse. Positive gap = rider outranks the horse's
            # market slot.
            jockey_pct = df.groupby("race_id")["jockey_elo"].rank(
                ascending=True, pct=True, method="average"
            )
            implied = 1.0 / df["odds"].replace(0, np.nan)
            market_pct = implied.groupby(df["race_id"]).rank(
                ascending=True, pct=True, method="average"
            )
            df["jockey_booking_vs_market"] = (jockey_pct - market_pct).fillna(0.0)

    return df


# ── Smoothed target encoding ────────────────────────────────────
def add_target_encoded_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smoothed target encoding for high-cardinality categorical entities.

    Encodes each entity's cumulative *mean normalised position*
    (0 = winner, 1 = last) with Bayesian shrinkage toward the global
    mean.  This captures entity-specific baselines that rolling stats
    and win-rate shrinkage don't fully cover — e.g. "this trainer's
    runners tend to finish in the top quarter regardless of conditions."

    All computations use prior-only (cumsum - current) to avoid leakage.
    """
    logger.info("Engineering target-encoded features …")
    hkey = _horse_key(df)

    # Normalised position: 0 = winner, 1 = last, a non-finish counted as
    # last (see ``_add_result_history_flags``). Leaving non-finishes out made
    # a jockey or trainer whose runners keep falling look better than one
    # whose runners finish down the field.
    if "_norm_pos_for_history" in df.columns:
        df["_norm_pos"] = df["_norm_pos_for_history"].astype(np.float32)
    else:
        nr = df["num_runners"].replace(0, 1).values.astype(np.float32)
        df["_norm_pos"] = (
            (df["finish_position"].values.astype(np.float32) - 1)
            / np.maximum(nr - 1, 1)
        )
    df["_norm_pos_safe"] = df["_norm_pos"].fillna(0.0)
    df["_norm_pos_weight"] = df["_norm_pos"].notna().astype(np.float32)

    # Use the expanding mean (prior-only) as the global prior so it
    # doesn't peek at future races.  The expanding mean converges
    # quickly (~0.45-0.50) so the prior is very stable.
    global_mean = 0.45  # neutral prior — close to the true dataset mean
    prior_strength = 15.0  # stronger shrinkage than win-rate (0.1 prior)

    # ---- Helper: target-encode one entity column ----
    def _te(entity_col: str, prefix: str, race_safe: bool = False):
        """Add ``{prefix}_te_norm_pos`` target-encoded feature.

        Lifetime (class-C) encoding routed through the seed-aware _cumulative
        primitive so the FeatureStore append can carry its state.
        cumulative_counts is always race-safe, which equals the plain
        cumsum-minus-current for one-row-per-race entities (horse/jockey); the
        ``race_safe`` flag is retained for call-site clarity but no longer
        changes the result.
        """
        _c = _cumulative(
            df[entity_col].to_numpy(), df["race_id"].to_numpy(),
            {
                f"{prefix}_te_sum": df["_norm_pos_safe"].to_numpy(),
                f"{prefix}_te_n": df["_norm_pos_weight"].to_numpy(),
            },
        )
        cum_sum = _c[f"{prefix}_te_sum"]
        cum_n = _c[f"{prefix}_te_n"]
        observed = np.full(np.shape(cum_sum), global_mean, dtype=float)
        np.divide(cum_sum, cum_n, out=observed, where=(cum_n > 0))
        # Bayesian shrinkage: (n * observed + m * prior) / (n + m)
        df[f"{prefix}_te_norm_pos"] = (
            (cum_n * observed
             + prior_strength * global_mean)
            / (cum_n + prior_strength)
        )

    # Core entities
    _te(hkey, "horse")
    if "jockey" in df.columns:
        _te("jockey", "jockey")
    if "trainer" in df.columns:
        _te("trainer", "trainer", race_safe=True)
    if "track" in df.columns:
        _te("track", "track", race_safe=True)
    if "going" in df.columns:
        _te("going", "going", race_safe=True)

    # Interaction: jockey × trainer combo
    if "jockey" in df.columns and "trainer" in df.columns:
        df["_jt_combo"] = df["jockey"].astype(str) + "_" + df["trainer"].astype(str)
        _te("_jt_combo", "jt", race_safe=True)
        df = _without(df, ["_jt_combo"])

    # ── Windowed target encodings (recent-form baselines) ────────────────
    # These complement the lifetime cumulative encodings above.  Under
    # concept drift (long training windows) career averages shift away from
    # current form; a rolling 365-day window tracks the recent baseline.
    _te_w = getattr(config, "TE_WINDOW_DAYS", 365)
    if _te_w > 0 and "race_date" in df.columns:

        def _te_windowed(entity_col: str, prefix: str,
                         race_safe: bool = False) -> None:
            """Add ``{prefix}_te_norm_pos_{_te_w}d`` windowed encoding.

            When *race_safe* is True, aggregates to one observation per
            (entity, race_id) so that same-race runners sharing the
            entity value (e.g. all runners share the same going) cannot
            leak each other's outcomes.
            """
            if race_safe and "race_id" in df.columns:
                _agg = df.groupby([entity_col, "race_id"], sort=False).agg(
                    _np_sum=("_norm_pos_safe", "sum"),
                    _np_cnt=("_norm_pos_weight", "sum"),
                    race_date=("race_date", "first"),
                ).reset_index()
                # Sum-of-individual-norm_pos in window
                _ws, _ = _time_window(
                    _agg, entity_col, "race_date", "_np_sum", _te_w,
                )
                # Count-of-individual-runners in window
                _wc, _ = _time_window(
                    _agg, entity_col, "race_date", "_np_cnt", _te_w,
                )
                _agg["_val_sum"] = _ws
                _agg["_val_cnt"] = _wc
                _lookup = _agg.set_index([entity_col, "race_id"])[
                    ["_val_sum", "_val_cnt"]
                ]
                _mi = pd.MultiIndex.from_arrays(
                    [df[entity_col], df["race_id"]]
                )
                _mapped = _lookup.reindex(_mi)
                val_sum = _mapped["_val_sum"].values.astype(float)
                val_cnt = _mapped["_val_cnt"].values.astype(float)
            else:
                val_sum, _ = _time_window(
                    df, entity_col, "race_date", "_norm_pos_safe", _te_w,
                )
                val_cnt, _ = _time_window(
                    df, entity_col, "race_date", "_norm_pos_weight", _te_w,
                )
            observed = np.full(np.shape(val_sum), global_mean, dtype=float)
            np.divide(val_sum, val_cnt, out=observed, where=(val_cnt > 0))
            df[f"{prefix}_te_norm_pos_{_te_w}d"] = (
                (val_cnt * observed
                 + prior_strength * global_mean)
                / (val_cnt + prior_strength)
            )

        _te_windowed(hkey, "horse")
        if "jockey" in df.columns:
            _te_windowed("jockey", "jockey")
        if "trainer" in df.columns:
            _te_windowed("trainer", "trainer", race_safe=True)
        if "going" in df.columns:
            _te_windowed("going", "going", race_safe=True)

    # ── EWMA target encodings (smooth decay, no calendar cliff-edge) ─────
    # A race-count EWMA with half-life ≈ 10 races avoids the hard cutoff of
    # the 365d window while still down-weighting stale history — important
    # for horses/jockeys whose form drifts gradually over seasons.
    _te_hl = getattr(config, "TE_EWMA_HALF_LIFE_RACES", 10)
    if _te_hl > 0:
        _ewma_alpha = float(1 - np.exp(-np.log(2) / _te_hl))

        def _te_ewma(entity_col: str, prefix: str,
                     race_safe: bool = False) -> None:
            """Add ``{prefix}_te_ewma`` EWMA target-encoded feature.

            When *race_safe* is True, aggregates to one observation per
            (entity, race_id) first, computes the EWMA on that per-race
            series, then maps back — preventing same-race runners from
            leaking each other's outcomes.
            """
            if race_safe and "race_id" in df.columns:
                _dt_col = (
                    "_event_dt" if "_event_dt" in df.columns else "race_date"
                )
                _agg = (
                    df.groupby([entity_col, "race_id"], sort=False)
                    .agg(
                        _np_mean=("_norm_pos", "mean"),
                        _event_dt=(_dt_col, "first"),
                    )
                    .reset_index()
                )
                # Seed-aware EWMA over the per-(entity, race) series via the
                # rolling primitive (carries an EWMA-converging tail), then map
                # the per-race value back onto each runner.
                _eo = _rolling(
                    _agg, entity_col, "_np_mean",
                    [(f"_{prefix}_te_ewma_raw", "ewma", _ewma_alpha)],
                )
                _agg["_ewma"] = _eo[f"_{prefix}_te_ewma_raw"]
                _lookup = _agg.set_index([entity_col, "race_id"])["_ewma"]
                _mi = pd.MultiIndex.from_arrays(
                    [df[entity_col], df["race_id"]]
                )
                raw = _lookup.reindex(_mi).values
            else:
                results = _rolling(
                    df, entity_col, "_norm_pos",
                    [(f"_{prefix}_te_ewma_raw", "ewma", _ewma_alpha)],
                )
                raw = results[f"_{prefix}_te_ewma_raw"]
            # Prior weighted-run count (class C, seed-aware) drives the shrink.
            cum_n = _cumulative(
                df[entity_col].to_numpy(), df["race_id"].to_numpy(),
                {f"_{prefix}_te_ewma_n": df["_norm_pos_weight"].to_numpy()},
            )[f"_{prefix}_te_ewma_n"]
            # Shrink toward prior for low-count entities
            df[f"{prefix}_te_ewma"] = (
                (cum_n * np.where(cum_n > 0, raw, global_mean)
                 + prior_strength * global_mean)
                / (cum_n + prior_strength)
            )

        _te_ewma(hkey, "horse")
        if "jockey" in df.columns:
            _te_ewma("jockey", "jockey")
        if "trainer" in df.columns:
            _te_ewma("trainer", "trainer", race_safe=True)
        if "going" in df.columns:
            _te_ewma("going", "going", race_safe=True)

    # Date-strict views: entities that appear several times per day
    # (jockeys/trainers/combos — and shared context keys like track and
    # going) must not see same-day earlier results in their encodings.
    _te_w_str = str(getattr(config, "TE_WINDOW_DAYS", 365))
    _suffixes = ["_te_norm_pos", f"_te_norm_pos_{_te_w_str}d", "_te_ewma"]

    def _te_cols(prefix: str) -> list[str]:
        return [f"{prefix}{s}" for s in _suffixes]

    if "jockey" in df.columns:
        df = _entity_day_start_view(df, ["jockey"], _te_cols("jockey"))
    if "trainer" in df.columns:
        df = _entity_day_start_view(df, ["trainer"], _te_cols("trainer"))
    if "jockey" in df.columns and "trainer" in df.columns:
        df["_jt_combo"] = df["jockey"].astype(str) + "_" + df["trainer"].astype(str)
        df = _entity_day_start_view(df, ["_jt_combo"], _te_cols("jt"))
        df = _without(df, ["_jt_combo"])
    if "track" in df.columns:
        df = _entity_day_start_view(df, ["track"], _te_cols("track"))
    if "going" in df.columns:
        df = _entity_day_start_view(df, ["going"], _te_cols("going"))

    df = _without(df, ["_norm_pos", "_norm_pos_safe", "_norm_pos_weight"])
    return df


# ── Cross-entity target encodings ────────────────────────────────
def add_cross_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target-encoded interaction features for entity × context combos
    not already covered by the per-entity functions.

    All use cumulative (prior-only) computation with shift(1) to
    avoid data leakage.
    """
    logger.info("Engineering cross-entity features...")
    hkey = _horse_key(df)
    # Shared inputs for the class-C cross-entity counts (seed-aware _cumulative).
    _rs = df["_result_start"].fillna(0.0).to_numpy()
    _wh = df["_won_for_history"].fillna(0.0).to_numpy()
    _race = df["race_id"].to_numpy()

    # --- Jockey × going ---
    if "going" in df.columns:
        _jg = _cumulative(
            _composite_key(df, "jockey", "going"), _race,
            {"jockey_going_runs": _rs, "_jg_w": _wh},
        )
        df["jockey_going_runs"] = _jg["jockey_going_runs"]
        df["_jg_w"] = _jg["_jg_w"]
        df["jockey_going_wr"] = _rate(df["_jg_w"], df["jockey_going_runs"])
        df = _entity_day_start_view(
            df, ["jockey", "going"], ["jockey_going_runs", "jockey_going_wr"]
        )
        df["jockey_going_wr"] = _shrink_to_entity(
            df, "jockey_going_wr", "jockey_going_runs", "jockey"
        )
        df = _without(df, ["_jg_w"])

    # --- Jockey × distance category ---
    if "dist_category" in df.columns:
        df["_dc_str"] = df["dist_category"].astype(str)
        _jdc = _cumulative(
            _composite_key(df, "jockey", "_dc_str"), _race,
            {"jockey_dist_cat_runs": _rs, "_jdc_w": _wh},
        )
        df["jockey_dist_cat_runs"] = _jdc["jockey_dist_cat_runs"]
        df["_jdc_w"] = _jdc["_jdc_w"]
        df["jockey_dist_cat_wr"] = _rate(df["_jdc_w"], df["jockey_dist_cat_runs"])
        df = _entity_day_start_view(
            df, ["jockey", "_dc_str"], ["jockey_dist_cat_runs", "jockey_dist_cat_wr"]
        )
        df["jockey_dist_cat_wr"] = _shrink_to_entity(
            df, "jockey_dist_cat_wr", "jockey_dist_cat_runs", "jockey"
        )
        df = _without(df, ["_dc_str", "_jdc_w"])

    # --- Horse × going (win rate on this going type) ---
    if "going" in df.columns:
        # Tagged state key — this recomputes horse_going_runs (also produced in
        # add_track_features); a distinct key keeps the two seed states separate.
        _hg = _cumulative(
            _composite_key(df, hkey, "going"), _race,
            {"horse_going_runs__xent": _rs, "_hg_w": _wh},
        )
        df["horse_going_runs"] = _hg["horse_going_runs__xent"]
        df["_hg_w"] = _hg["_hg_w"]
        df = _without(df, ["_hg_w"])

    # --- Horse × going × distance rounded (C&D&G specialisation) ---
    if "going" in df.columns and "distance_furlongs" in df.columns:
        df["_dist_r2"] = (df["distance_furlongs"] * 2).round() / 2
        _hgd = _cumulative(
            _composite_key(df, hkey, "going", "_dist_r2"), _race,
            {"horse_going_dist_runs": _rs, "_hgd_w": _wh},
        )
        df["horse_going_dist_runs"] = _hgd["horse_going_dist_runs"]
        df["_hgd_w"] = _hgd["_hgd_w"]
        df = _without(df, ["_dist_r2", "_hgd_w"])

    # --- Trainer × race class (race-safe) ---
    if "race_class" in df.columns:
        df["_tc_key"] = df["trainer"].astype(str) + "||" + df["race_class"].astype(str)
        _tc = _cumulative(
            df["_tc_key"].to_numpy(), _race,
            {"trainer_class_runs": df["_result_start"].to_numpy(), "_tc_w": df["won"].to_numpy()},
        )
        df["trainer_class_runs"] = _tc["trainer_class_runs"]
        df["_tc_w"] = _tc["_tc_w"]
        df["trainer_class_wr"] = _rate(df["_tc_w"], df["trainer_class_runs"])
        df = _entity_day_start_view(
            df, ["_tc_key"], ["trainer_class_runs", "trainer_class_wr"]
        )
        df["trainer_class_wr"] = _shrink_to_entity(
            df, "trainer_class_wr", "trainer_class_runs", "trainer"
        )
        df = _without(df, ["_tc_w", "_tc_key"])

    return df
