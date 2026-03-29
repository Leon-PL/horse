"""
Walk-Forward Validation (Backtesting)
=====================================
Train on months 1-3, predict month 4, retrain on 1-4, predict 5, etc.

Gives a realistic view of how the model would have performed over time
using an expanding training window -- just like deploying it live.

Outputs per-fold ranking metrics (NDCG, top-1 accuracy), per-race P&L
under multiple betting strategies, and cumulative profit curves.

Usage::

    from src.backtester import walk_forward_validation

    report = walk_forward_validation(featured_df)
    print(report["summary"])

CLI::

    python -m src.backtester
    python -m src.backtester --min-train-months 2
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import config
from src.model import (
    get_feature_columns, make_relevance_labels,
    normalise_implied_prob_by_race,
    TripleEnsemblePredictor,
)
from src.each_way import get_ew_terms, ew_value as _ew_value

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _off_time_to_seconds(off_time: pd.Series) -> pd.Series:
    """Parse off-time strings to seconds after midnight with safe fallbacks."""
    s = off_time.astype(str).str.strip()
    parts = s.str.extract(r"(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?")
    h = pd.to_numeric(parts["h"], errors="coerce")
    m = pd.to_numeric(parts["m"], errors="coerce")
    sec = pd.to_numeric(parts["s"], errors="coerce").fillna(0)

    bad = h.isna() | m.isna()
    if bad.any():
        try:
            dt = pd.to_datetime(s[bad], errors="coerce", format="mixed")
        except TypeError:
            dt = pd.to_datetime(s[bad], errors="coerce")
        h.loc[bad] = dt.dt.hour
        m.loc[bad] = dt.dt.minute
        sec.loc[bad] = dt.dt.second

    h = h.fillna(12).clip(0, 23)
    m = m.fillna(0).clip(0, 59)
    sec = sec.fillna(0).clip(0, 59)
    return h * 3600 + m * 60 + sec


def _event_sort_key(df: pd.DataFrame) -> pd.Series:
    """Build event-time key using race_date + off_time when available."""
    race_dt = pd.to_datetime(df["race_date"], errors="coerce")
    if "off_time" not in df.columns:
        return race_dt
    off_secs = _off_time_to_seconds(df["off_time"]).astype(float)
    return race_dt + pd.to_timedelta(off_secs, unit="s")


# -----------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------

@dataclass
class FoldResult:
    """Metrics and P&L for a single walk-forward fold."""

    fold: int
    train_start: str
    train_end: str
    test_period: str
    train_size: int
    test_size: int
    test_races: int

    # Ranking / calibration metrics
    brier_score: float = 0.0
    ndcg_at_1: float = 0.0
    top1_accuracy: float = 0.0
    win_in_top3: float = 0.0

    # Betting P&L (1-unit flat stakes)
    top_pick_bets: int = 0
    top_pick_winners: int = 0
    top_pick_pnl: float = 0.0

    value_bets: int = 0
    value_winners: int = 0
    value_pnl: float = 0.0

    ew_bets: int = 0
    ew_winners: int = 0
    ew_placed: int = 0
    ew_pnl: float = 0.0


# -----------------------------------------------------------------------
# Core walk-forward engine
# -----------------------------------------------------------------------

def walk_forward_validation(
    df: pd.DataFrame,
    model_type: str = "triple_ensemble",
    min_train_months: int = 2,
    test_window_months: int = 1,
    value_threshold: float = 0.05,
    frameworks: dict[str, str] | None = None,
    params: dict[str, dict] | None = None,
    progress_callback=None,
    fast_fold: bool = True,
    ew_min_place_edge: float | None = None,
) -> dict:
    """
    Run walk-forward (expanding-window) validation.

    Args:
        df: Feature-engineered DataFrame with ``race_date`` column.
        model_type: ``"triple_ensemble"``.
        min_train_months: Minimum number of months to use for the first
                         training fold.
        test_window_months: How many months each test fold covers.
        value_threshold: Minimum ``model_prob - implied_prob`` to count
                        as a value bet.
        progress_callback: Optional ``(message, pct)`` for UI updates.

    Returns:
        dict with keys:
            ``folds``   – list[FoldResult]
            ``summary`` – DataFrame summarising each fold
            ``bets``    – DataFrame of every individual bet placed
            ``curves``  – DataFrame with cumulative P&L per bet
    """

    def _cb(msg: str, pct: float = 0.0) -> None:
        if progress_callback is not None:
            progress_callback(msg, pct)
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["_event_dt"] = _event_sort_key(df)
    # Sort by horse_name within each race to break finish_position
    # ordering from the raw data — prevents row-position from leaking
    # the outcome when model scores are degenerate (all-equal).
    _sort_cols = ["_event_dt", "race_id"]
    if "horse_name" in df.columns:
        _sort_cols.append("horse_name")
    df = df.sort_values(_sort_cols).reset_index(drop=True)

    # Only use actual results (finish_position > 0)
    df = df[df["finish_position"] > 0].copy()

    feature_cols = get_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found in the data.")

    # --- Build month boundaries ---
    df["_ym"] = df["race_date"].dt.to_period("M")
    months = sorted(df["_ym"].unique())

    if len(months) < min_train_months + 1:
        raise ValueError(
            f"Need at least {min_train_months + 1} months of data for "
            f"walk-forward validation, but only found {len(months)} "
            f"({months[0]} – {months[-1]})."
        )

    # --- Pre-count folds for progress reporting ---
    _total_folds = max(1, (len(months) - min_train_months + test_window_months - 1) // test_window_months)
    _cb(f"Walk-forward: {len(months)} months, ~{_total_folds} folds", 0.0)

    # --- Walk forward ---
    folds: list[FoldResult] = []
    all_bets: list[dict] = []

    purge_days = getattr(config, "PURGE_DAYS", 7)
    fold_idx = 0
    test_start = min_train_months  # index into `months`

    while test_start < len(months):
        test_end = min(test_start + test_window_months, len(months))
        train_months = months[:test_start]
        test_months = months[test_start:test_end]

        # Purge gap: remove training rows within PURGE_DAYS of the
        # first test date to prevent form-feature leakage.
        test_first_date = df.loc[df["_ym"].isin(test_months), "race_date"].min()
        purge_cutoff = test_first_date - pd.Timedelta(days=purge_days)

        train_mask = df["_ym"].isin(train_months) & (df["race_date"] <= purge_cutoff)
        test_mask = df["_ym"].isin(test_months)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) < 50 or len(test_df) < 10:
            test_start = test_end
            continue

        fold_idx += 1
        train_period = f"{train_months[0]} → {train_months[-1]}"
        test_period = " / ".join(str(m) for m in test_months)

        _fold_pct = fold_idx / max(_total_folds, 1)
        _cb(
            f"Fold {fold_idx}/{_total_folds}: train {train_period} · "
            f"test {test_period} ({len(train_df):,} train / {len(test_df):,} test)",
            _fold_pct,
        )

        logger.info(
            f"\n{'='*60}\n"
            f"  Fold {fold_idx}: train {train_period}  |  test {test_period}\n"
            f"  Train: {len(train_df):,} runners  |  Test: {len(test_df):,} runners\n"
            f"{'='*60}"
        )

        # --- Prepare & train ---
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values

        # Tree models don't require feature scaling.
        X_train_s = X_train
        X_test_s = X_test

        # Relevance labels for ranking metrics.
        y_train_rel = make_relevance_labels(
            train_df["finish_position"].values.astype(int),
        )
        groups_train = train_df.groupby("race_id", sort=False).size().values

        groups_test_arr = test_df.groupby("race_id", sort=False).size().values
        _te = TripleEnsemblePredictor(frameworks=frameworks)
        win_probs_cal, place_probs = _te.train_on_fold(
            X_train_s, X_test_s, train_df,
            groups_train, groups_test_arr, feature_cols,
            params=params,
            return_place_probs=True,
            fast_fold=fast_fold,
            test_df=test_df,
        )
        # train_on_fold returns calibrated probabilities
        y_prob = win_probs_cal
        raw_scores = win_probs_cal  # use win probs for ranking metrics

        # --- Ranking metrics (NDCG@1, Top-1, Winner-in-Top-3) ---
        from sklearn.metrics import ndcg_score as _ndcg

        y_test_rel = make_relevance_labels(
            test_df["finish_position"].values.astype(int),
        )

        ndcg1_list, top1_ok, win3_ok, n_eval = [], 0, 0, 0
        for race_id in test_df["race_id"].unique():
            rm = test_df["race_id"].values == race_id
            rs = raw_scores[rm]
            rl = y_test_rel[rm]
            if len(rs) < 2 or rl.max() == rl.min() or rs.max() == rs.min():
                continue
            n_eval += 1
            try:
                ndcg1_list.append(_ndcg([rl], [rs], k=1))
            except ValueError:
                pass
            if np.argmax(rs) == np.argmax(rl):
                top1_ok += 1
            if np.argmax(rl) in set(np.argsort(rs)[-3:]):
                win3_ok += 1

        fold_ndcg1 = float(np.mean(ndcg1_list)) if ndcg1_list else 0.0
        fold_top1 = top1_ok / n_eval if n_eval else 0.0
        fold_win3 = win3_ok / n_eval if n_eval else 0.0

        # Brier Score (calibration quality)
        y_won_test = (test_df["won"].values if "won" in test_df.columns
                      else (test_df["finish_position"].values == 1)).astype(int)
        probs_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
        fold_brier = float(brier_score_loss(y_won_test, probs_clipped))

        # --- P&L simulation (flat 1-unit stakes) ---
        test_with_probs = test_df.copy()
        test_with_probs["model_prob"] = y_prob
        test_with_probs["implied_prob"] = normalise_implied_prob_by_race(test_with_probs)
        test_with_probs["value_score"] = (
            test_with_probs["model_prob"] - test_with_probs["implied_prob"]
        )
        if place_probs is not None:
            test_with_probs["place_prob"] = place_probs

        # --- Strategy 1: Top Pick (highest win probability, vectorised) ---
        # Skip races where model_prob is constant (model has no opinion).
        _tp_col = "model_prob"
        _tp_range = test_with_probs.groupby("race_id", sort=False)[_tp_col]
        _tp_diff = _tp_range.transform("max") - _tp_range.transform("min")
        _tp_valid = test_with_probs[_tp_diff > 0].copy()
        if not _tp_valid.empty:
            _tp_idx = _tp_valid.groupby("race_id", sort=False)[_tp_col].idxmax()
            _tp = test_with_probs.loc[_tp_idx]
        else:
            _tp = pd.DataFrame()
        _tp_won = _tp["won"].values.astype(int) if not _tp.empty else np.array([], dtype=int)
        _tp_pnl_arr = np.where(_tp_won == 1, _tp["odds"].values - 1.0, -1.0) if not _tp.empty else np.array([])
        top_pick_bets = len(_tp)
        top_pick_winners = int(_tp_won.sum()) if len(_tp_won) else 0
        top_pick_pnl = float(_tp_pnl_arr.sum()) if len(_tp_pnl_arr) else 0.0

        for _ti, (_tridx, _tr) in enumerate(_tp.iterrows()):
            all_bets.append({
                "fold": fold_idx,
                "test_period": test_period,
                "race_id": _tr["race_id"],
                "race_date": str(_tr["race_date"].date()),
                "track": _tr.get("track", ""),
                "strategy": "top_pick",
                "horse_name": _tr["horse_name"],
                "model_prob": _tr["model_prob"],
                "odds": _tr["odds"],
                "won": int(_tr["won"]),
                "pnl": round(_tp_pnl_arr[_ti], 2),
            })

        # --- Strategy 2: Value Bets (vectorised) ---
        _dyn_thresh = value_threshold * np.sqrt(test_with_probs["odds"] / 3.0)
        _vb_mask = test_with_probs["value_score"] > _dyn_thresh
        _vb = test_with_probs[_vb_mask]
        _vb_won = _vb["won"].values.astype(int)
        _vb_pnl_arr = np.where(_vb_won == 1, _vb["odds"].values - 1.0, -1.0)
        value_bets_n = len(_vb)
        value_winners = int(_vb_won.sum())
        value_pnl = float(_vb_pnl_arr.sum())

        for _vi, (_, vp) in enumerate(_vb.iterrows()):
            all_bets.append({
                "fold": fold_idx,
                "test_period": test_period,
                "race_id": vp["race_id"],
                "race_date": str(vp["race_date"].date()),
                "track": vp.get("track", ""),
                "strategy": "value",
                "horse_name": vp["horse_name"],
                "model_prob": vp["model_prob"],
                "odds": vp["odds"],
                "won": int(vp["won"]),
                "pnl": round(_vb_pnl_arr[_vi], 2),
            })

        # --- Strategy 3: Each-Way Value Bets ---
        ew_bets_n = 0
        ew_winners = 0
        ew_placed_n = 0
        ew_pnl = 0.0

        if place_probs is not None and "place_prob" in test_with_probs.columns:
            from src.each_way import adjust_place_probs_for_race as _adj_pp

            for race_id, race_group in test_with_probs.groupby("race_id"):
                _nr = int(race_group["num_runners"].iloc[0]) if "num_runners" in race_group.columns else len(race_group)
                _hcap = bool(race_group.get("handicap", pd.Series(0)).iloc[0]) if "handicap" in race_group.columns else False
                ew_terms = get_ew_terms(_nr, is_handicap=_hcap)
                if ew_terms.eligible:
                    # Normalise model P(place) for this race's places_paid
                    _raw_pp = race_group["place_prob"].values
                    _win_pp = race_group["model_prob"].values
                    _adj_place = _adj_pp(_raw_pp, _win_pp, ew_terms.places_paid)

                    for _ew_i, (_, ep) in enumerate(race_group.iterrows()):
                        if ep["odds"] < 4.0 or ep["odds"] > 51.0:
                            continue
                        ev_result = _ew_value(
                            ep["model_prob"], float(_adj_place[_ew_i]),
                            ep["odds"], ew_terms,
                        )
                        _ew_base = ew_min_place_edge if ew_min_place_edge is not None else value_threshold
                        _ew_dyn_thresh = _ew_base * np.sqrt(ep["odds"] / 3.0)
                        if ev_result["place_edge"] > _ew_dyn_thresh and ev_result["place_ev"] > 0:
                            ew_bets_n += 1
                            # EW bet = 2 units (1 win + 1 place)
                            fp_val = int(ep["finish_position"])
                            won_flag = int(ep["won"])
                            placed_flag = int(fp_val <= ew_terms.places_paid)
                            pnl_ew = -2.0  # cost: 2 units
                            if won_flag:
                                pnl_ew += ep["odds"]  # win leg returns
                                pnl_ew += ev_result["place_odds"]  # place leg returns
                                ew_winners += 1
                                ew_placed_n += 1
                            elif placed_flag:
                                pnl_ew += ev_result["place_odds"]  # place leg only
                                ew_placed_n += 1
                            ew_pnl += pnl_ew

                            all_bets.append({
                                "fold": fold_idx,
                                "test_period": test_period,
                                "race_id": race_id,
                                "race_date": str(ep["race_date"].date()),
                                "track": ep.get("track", ""),
                                "strategy": "each_way",
                                "horse_name": ep["horse_name"],
                                "model_prob": ep["model_prob"],
                                "odds": ep["odds"],
                                "won": won_flag,
                                "pnl": round(pnl_ew, 2),
                            })

        test_races = test_with_probs["race_id"].nunique()

        fold_result = FoldResult(
            fold=fold_idx,
            train_start=str(train_months[0]),
            train_end=str(train_months[-1]),
            test_period=test_period,
            train_size=len(train_df),
            test_size=len(test_df),
            test_races=test_races,
            brier_score=round(fold_brier, 6),
            ndcg_at_1=round(fold_ndcg1, 4),
            top1_accuracy=round(fold_top1, 4),
            win_in_top3=round(fold_win3, 4),
            top_pick_bets=top_pick_bets,
            top_pick_winners=top_pick_winners,
            top_pick_pnl=round(top_pick_pnl, 2),
            value_bets=value_bets_n,
            value_winners=value_winners,
            value_pnl=round(value_pnl, 2),
            ew_bets=ew_bets_n,
            ew_winners=ew_winners,
            ew_placed=ew_placed_n,
            ew_pnl=round(ew_pnl, 2),
        )
        folds.append(fold_result)

        logger.info(
            f"  Brier: {fold_brier:.6f}  |  NDCG@1: {fold_ndcg1:.4f}  |  Top-1: {fold_top1:.4f}\n"
            f"  Top-Pick: {top_pick_winners}/{top_pick_bets} winners, "
            f"P&L = {top_pick_pnl:+.2f} units\n"
            f"  Value:    {value_winners}/{value_bets_n} winners, "
            f"P&L = {value_pnl:+.2f} units\n"
            f"  EW:       {ew_placed_n}/{ew_bets_n} placed ({ew_winners} won), "
            f"P&L = {ew_pnl:+.2f} units"
        )

        _cb(
            f"Fold {fold_idx}/{_total_folds} done — "
            f"NDCG@1 {fold_ndcg1:.3f} · Top-1 {fold_top1:.1%} · "
            f"TP {top_pick_pnl:+.1f}u · Val {value_pnl:+.1f}u · EW {ew_pnl:+.1f}u",
            fold_idx / max(_total_folds, 1),
        )

        test_start = test_end

    _cb(f"Walk-forward complete — {len(folds)} folds", 1.0)

    # --- Build summary DataFrame ---
    summary = pd.DataFrame([vars(f) for f in folds])

    # --- Build cumulative P&L curves ---
    bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()
    curves = _build_cumulative_curves(bets_df) if not bets_df.empty else pd.DataFrame()

    # --- Overall summary ---
    if folds:
        _print_overall_summary(folds, summary)

    return {
        "folds": folds,
        "summary": summary,
        "bets": bets_df,
        "curves": curves,
    }


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _build_cumulative_curves(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative P&L series for each strategy."""
    rows = []
    for strategy in ["top_pick", "value", "each_way"]:
        strat_bets = bets_df[bets_df["strategy"] == strategy].copy()
        if strat_bets.empty:
            continue
        strat_bets = strat_bets.sort_values("race_date").reset_index(drop=True)
        strat_bets["cum_pnl"] = strat_bets["pnl"].cumsum()
        strat_bets["bet_number"] = range(1, len(strat_bets) + 1)
        # EW bets cost 2 units each, so normalise ROI accordingly
        _stake_per_bet = 2.0 if strategy == "each_way" else 1.0
        strat_bets["cum_roi_pct"] = (
            strat_bets["cum_pnl"] / (strat_bets["bet_number"] * _stake_per_bet) * 100
        )
        rows.append(strat_bets)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _print_overall_summary(folds: list[FoldResult], summary: pd.DataFrame):
    """Print an overall summary to the console."""
    avg_ndcg = summary["ndcg_at_1"].mean()
    avg_top1 = summary["top1_accuracy"].mean()
    avg_brier = summary["brier_score"].mean()
    total_tp_pnl = summary["top_pick_pnl"].sum()
    total_tp_bets = summary["top_pick_bets"].sum()
    total_tp_wins = summary["top_pick_winners"].sum()
    total_v_pnl = summary["value_pnl"].sum()
    total_v_bets = summary["value_bets"].sum()
    total_v_wins = summary["value_winners"].sum()
    total_ew_pnl = summary["ew_pnl"].sum()
    total_ew_bets = summary["ew_bets"].sum()
    total_ew_wins = summary["ew_winners"].sum()
    total_ew_placed = summary["ew_placed"].sum()

    tp_sr = (total_tp_wins / total_tp_bets * 100) if total_tp_bets else 0
    tp_roi = (total_tp_pnl / total_tp_bets * 100) if total_tp_bets else 0
    v_sr = (total_v_wins / total_v_bets * 100) if total_v_bets else 0
    v_roi = (total_v_pnl / total_v_bets * 100) if total_v_bets else 0
    ew_place_sr = (total_ew_placed / total_ew_bets * 100) if total_ew_bets else 0
    ew_roi = (total_ew_pnl / (total_ew_bets * 2) * 100) if total_ew_bets else 0

    logger.info(
        f"\n{'='*65}\n"
        f"  WALK-FORWARD VALIDATION -- OVERALL RESULTS ({len(folds)} folds)\n"
        f"{'='*65}\n"
        f"  Avg Brier Score:  {avg_brier:.6f}\n"
        f"  Avg NDCG@1:       {avg_ndcg:.4f}\n"
        f"  Avg Top-1 Acc:    {avg_top1:.4f}\n"
        f"{'─'*65}\n"
        f"  TOP-PICK strategy (1 bet per race):\n"
        f"    Bets: {total_tp_bets}  |  Winners: {total_tp_wins}  |  "
        f"Strike rate: {tp_sr:.1f}%\n"
        f"    P&L: {total_tp_pnl:+.2f} units  |  ROI: {tp_roi:+.1f}%\n"
        f"{'─'*65}\n"
        f"  VALUE strategy (model_prob > implied_prob + threshold):\n"
        f"    Bets: {total_v_bets}  |  Winners: {total_v_wins}  |  "
        f"Strike rate: {v_sr:.1f}%\n"
        f"    P&L: {total_v_pnl:+.2f} units  |  ROI: {v_roi:+.1f}%\n"
        f"{'─'*65}\n"
        f"  EACH-WAY strategy (EW value, 2-unit stakes):\n"
        f"    Bets: {total_ew_bets}  |  Placed: {total_ew_placed}  |  Won: {total_ew_wins}  |  "
        f"Place rate: {ew_place_sr:.1f}%\n"
        f"    P&L: {total_ew_pnl:+.2f} units  |  ROI: {ew_roi:+.1f}%\n"
        f"{'='*65}"
    )


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Walk-forward backtest of the horse racing model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="triple_ensemble",
        choices=["triple_ensemble"],
        help="Model type (default: triple_ensemble)",
    )
    parser.add_argument(
        "--min-train-months",
        type=int,
        default=2,
        help="Minimum months of data before first prediction (default: 2)",
    )
    parser.add_argument(
        "--value-threshold",
        type=float,
        default=0.05,
        help="Minimum value edge for value betting strategy (default: 0.05)",
    )
    args = parser.parse_args()

    # Load featured data
    pq_path = os.path.join(
        config.PROCESSED_DATA_DIR, "featured_races.parquet"
    )
    csv_path = os.path.join(
        config.PROCESSED_DATA_DIR, "featured_races.csv"
    )
    featured_path = pq_path if os.path.exists(pq_path) else csv_path
    if not os.path.exists(featured_path):
        print(
            "❌ No featured data found. Run training first:\n"
            "   python train.py --source database --days-back 90"
        )
        raise SystemExit(1)

    print("📂 Loading featured data...")
    df = pd.read_parquet(featured_path, engine="pyarrow") if featured_path.endswith(".parquet") else pd.read_csv(featured_path)
    print(f"   {len(df)} records loaded\n")

    report = walk_forward_validation(
        df,
        model_type=args.model,
        min_train_months=args.min_train_months,
        value_threshold=args.value_threshold,
    )

    # Save results
    output_dir = os.path.join(config.DATA_DIR, "backtest")
    os.makedirs(output_dir, exist_ok=True)

    report["summary"].to_csv(
        os.path.join(output_dir, "fold_summary.csv"), index=False
    )
    if not report["bets"].empty:
        report["bets"].to_csv(
            os.path.join(output_dir, "all_bets.csv"), index=False
        )
    if not report["curves"].empty:
        report["curves"].to_csv(
            os.path.join(output_dir, "cumulative_pnl.csv"), index=False
        )
    print(f"\n💾 Results saved to {output_dir}/")
