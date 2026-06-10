"""
Test-Set Betting Analysis
=========================
Runs the three-strategy betting simulation (top pick / win value /
each-way value) over a held-out test set and produces the bets, PnL
curves, per-strategy stats and calibration tables consumed by the
Experiments UI and the run store.

Selection and settlement rules come from ``src.bet_settlement`` — the
shared source of truth with the walk-forward backtester and live
settlement.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.bet_settlement import (
    EW_STAKE_UNITS,
    ew_bet_selected,
    ew_odds_in_band,
    ew_placed_flag,
    settle_ew_bet,
    settle_win_bet,
    value_bet_selection as _value_bet_selection,
)

logger = logging.getLogger(__name__)


def max_drawdown(pnl_series: np.ndarray) -> float:
    """Calculate maximum drawdown from a P&L array."""
    if len(pnl_series) == 0:
        return 0.0
    cum = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max())


# Backwards-compatible private alias
_max_drawdown = max_drawdown


def analyse_test_set(
    win_probs: np.ndarray,
    groups_test: np.ndarray,
    test_df: pd.DataFrame,
    value_threshold: float = 0.05,
    staking_mode: str = "flat",
    kelly_fraction: float = 0.25,
    bankroll: float = 100.0,
    place_probs: np.ndarray | None = None,
    ew_min_place_edge: float | None = None,
) -> dict:
    """
    Run betting simulation on the held-out test set.

    Args:
        win_probs: Pre-calibrated P(win) from the win classifier.
        groups_test: Runners-per-race arrays.
        test_df: Test DataFrame (needs race_id, race_date, horse_name, odds, won).
        value_threshold: Minimum edge for value bets.
        staking_mode: "flat" or "kelly".
        kelly_fraction: Fraction of full Kelly.
        bankroll: Starting bankroll (Kelly only).
        place_probs: Pre-calibrated P(place) from place classifier.

    Returns:
        dict with bets, curves, stats, calibration, etc.
    """
    analysis_df = test_df.copy().reset_index(drop=True)

    # Probabilities are already calibrated — use directly
    analysis_df["model_prob"] = win_probs

    has_odds = "odds" in analysis_df.columns
    has_won = "won" in analysis_df.columns

    if has_odds:
        # Use raw implied probability (1/odds) — consistent with strategy
        # calibrator.  Overround correction was inflating apparent edge, making
        # marginal bets look like value and producing lower real-world ROI.
        analysis_df["implied_prob"] = 1.0 / analysis_df["odds"]
        analysis_df["value_score"] = (
            analysis_df["model_prob"] - analysis_df["implied_prob"]
        )
        _value_sel = _value_bet_selection(
            analysis_df["model_prob"].values,
            analysis_df["odds"].values,
            value_threshold,
        )
        analysis_df["value_edge"] = _value_sel["edge"]
        analysis_df["value_dyn_threshold"] = _value_sel["dyn_threshold"]
        analysis_df["value_clv"] = _value_sel["clv"]
        analysis_df["value_expected_roi"] = _value_sel["expected_roi"]
        analysis_df["is_value_bet"] = _value_sel["mask"]
    else:
        analysis_df["implied_prob"] = 0.0
        analysis_df["value_score"] = 0.0
        analysis_df["value_edge"] = 0.0
        analysis_df["value_dyn_threshold"] = 0.0
        analysis_df["value_clv"] = np.nan
        analysis_df["value_expected_roi"] = np.nan
        analysis_df["is_value_bet"] = False

    if not has_won:
        analysis_df["won"] = (
            analysis_df["finish_position"] == 1
        ).astype(int)

    # Attach place probability if supplied
    if place_probs is not None and len(place_probs) == len(analysis_df):
        analysis_df["place_prob"] = place_probs

    # ── Betting simulation ────────────────────────────────────────
    from src.utils import kelly_criterion as _kelly_fn
    from src.each_way import get_ew_terms, ew_value as _ew_value_fn

    use_kelly = staking_mode == "kelly"
    _current_bankroll = float(bankroll) if use_kelly else 0.0

    all_bets: list[dict] = []

    # Sort races chronologically so Kelly bankroll accumulates correctly
    _race_order = (
        analysis_df.groupby("race_id")["race_date"]
        .first()
        .sort_values()
        .index
    )

    for race_id in _race_order:
        race_group = analysis_df[analysis_df["race_id"] == race_id]

        # Strategy 1 — Top Pick: highest win probability.
        # Skip when the model cannot genuinely distinguish
        # runners (all scores identical) — otherwise idxmax() picks by
        # row order, inflating strike rate via data-ordering artefacts.
        if race_group["model_prob"].max() == race_group["model_prob"].min():
            continue
        best = race_group.loc[race_group["model_prob"].idxmax()]
        odds_val = float(best["odds"]) if has_odds else 0.0
        pnl = settle_win_bet(odds_val, best["won"] == 1)

        rd = best["race_date"]
        rd_str = str(rd.date()) if hasattr(rd, "date") else str(rd)[:10]

        all_bets.append({
            "race_date": rd_str,
            "race_id": race_id,
            "track": best.get("track", ""),
            "strategy": "top_pick",
            "horse_name": best["horse_name"],
            "model_prob": round(float(best["model_prob"]), 4),
            "odds": round(odds_val, 2),
            "won": int(best["won"]),
            "stake": 1.0,
            "pnl": round(pnl, 2),
            "clv": round(float(best["model_prob"]) * odds_val, 4) if has_odds else None,
        })

        # Strategy 2 — Value Bets (odds-dependent threshold)
        # Tighter threshold at short odds (where calibration is better),
        # looser at long odds (where small edges are masked by variance).
        # Uses raw implied probability (1/odds) — same as strategy calibrator.
        # Any bet passing this check already has CLV > 1, so no extra gate needed.
        if has_odds:
            value_picks = race_group[race_group["is_value_bet"]]
            for _, vp in value_picks.iterrows():
                vp_odds = float(vp["odds"])
                vp_prob = float(vp["model_prob"])

                # CLV > 1 is guaranteed by positive raw edge + threshold;
                # no separate gate needed.
                vp_clv = round(float(vp.get("value_clv", vp_prob * vp_odds)), 4)

                # Determine stake
                if use_kelly and _current_bankroll > 0:
                    k_frac = _kelly_fn(vp_prob, vp_odds, fraction=kelly_fraction)
                    stake = round(max(k_frac * _current_bankroll, 0), 4)
                    stake = min(stake, _current_bankroll)
                else:
                    stake = 1.0

                if stake < 0.01:
                    continue  # skip negligible bets

                pnl_v = settle_win_bet(vp_odds, vp["won"] == 1, stake)

                if use_kelly:
                    _current_bankroll += pnl_v
                    _current_bankroll = max(_current_bankroll, 0)  # can't go negative

                vd = vp["race_date"]
                vd_str = str(vd.date()) if hasattr(vd, "date") else str(vd)[:10]
                all_bets.append({
                    "race_date": vd_str,
                    "race_id": race_id,
                    "track": vp.get("track", ""),
                    "strategy": "value",
                    "horse_name": vp["horse_name"],
                    "model_prob": round(vp_prob, 4),
                    "odds": round(vp_odds, 2),
                    "won": int(vp["won"]),
                    "stake": round(stake, 4),
                    "pnl": round(pnl_v, 4),
                    "clv": vp_clv,
                })

            # Strategy 3 — Each-Way Value Bets
            if has_odds and "place_prob" in race_group.columns:
                _nr = int(race_group["num_runners"].iloc[0]) if "num_runners" in race_group.columns else len(race_group)
                _hcap = bool(race_group.get("handicap", pd.Series(0)).iloc[0]) if "handicap" in race_group.columns else False
                ew_terms = get_ew_terms(_nr, is_handicap=_hcap)
                if ew_terms.eligible:
                    # Adjust P(top-3) → P(top-places_paid) + per-race normalise
                    from src.each_way import adjust_place_probs_for_race as _adj_pp
                    _raw_pp = race_group["place_prob"].values
                    _win_pp = race_group["model_prob"].values
                    _adj_place = _adj_pp(_raw_pp, _win_pp, ew_terms.places_paid)

                    for _ew_i, (_, ep) in enumerate(race_group.iterrows()):
                        if not ew_odds_in_band(ep["odds"]):
                            continue
                        ev_result = _ew_value_fn(
                            ep["model_prob"], float(_adj_place[_ew_i]),
                            ep["odds"], ew_terms,
                        )
                        # Use dedicated EW edge threshold (sidebar "Min place edge")
                        # with the same dynamic odds scaling as value bets.
                        _ew_base = ew_min_place_edge if ew_min_place_edge is not None else value_threshold
                        if ew_bet_selected(ev_result["place_edge"], ev_result["place_ev"], ep["odds"], _ew_base):
                            won_flag = int(ep["won"])
                            placed_flag = ew_placed_flag(ep.get("finish_position"), ew_terms.places_paid)
                            pnl_ew = settle_ew_bet(ep["odds"], ev_result["place_odds"], won_flag, placed_flag)

                            ed = ep["race_date"]
                            ed_str = str(ed.date()) if hasattr(ed, "date") else str(ed)[:10]
                            all_bets.append({
                                "race_date": ed_str,
                                "race_id": race_id,
                                "track": ep.get("track", ""),
                                "strategy": "each_way",
                                "horse_name": ep["horse_name"],
                                "model_prob": round(float(ep["model_prob"]), 4),
                                "odds": round(float(ep["odds"]), 2),
                                "won": won_flag,
                                "placed": int(placed_flag or won_flag),
                                "stake": EW_STAKE_UNITS,
                                "pnl": round(pnl_ew, 4),
                                "clv": round(float(ep["model_prob"]) * float(ep["odds"]), 4),
                            })

    bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()

    # ── Cumulative P&L curves ────────────────────────────────────
    curve_parts: list[pd.DataFrame] = []
    for strategy in ["top_pick", "value", "each_way"]:
        strat = bets_df[bets_df["strategy"] == strategy].copy() if not bets_df.empty else pd.DataFrame()
        if strat.empty:
            continue
        strat = strat.sort_values("race_date").reset_index(drop=True)
        strat["cum_pnl"] = strat["pnl"].cumsum()
        strat["cum_staked"] = strat["stake"].cumsum()
        strat["bet_number"] = range(1, len(strat) + 1)
        strat["cum_roi_pct"] = (
            strat["cum_pnl"] / strat["cum_staked"] * 100
        ).fillna(0)
        curve_parts.append(strat)
    curves_df = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()

    # ── Summary stats per strategy ───────────────────────────────
    stats: dict = {}
    for strategy in ["top_pick", "value", "each_way"]:
        strat = (
            bets_df[bets_df["strategy"] == strategy]
            if not bets_df.empty else pd.DataFrame()
        )
        n_bets = len(strat)
        n_wins = int(strat["won"].sum()) if n_bets else 0
        n_placed = int(strat["placed"].sum()) if (n_bets and "placed" in strat.columns) else n_wins
        total_pnl = float(strat["pnl"].sum()) if n_bets else 0.0
        total_staked = float(strat["stake"].sum()) if n_bets else 0.0
        strike_rate = n_wins / n_bets * 100 if n_bets else 0.0
        place_rate = n_placed / n_bets * 100 if n_bets else 0.0
        roi = total_pnl / total_staked * 100 if total_staked > 0 else 0.0
        avg_odds_win = float(strat[strat["won"] == 1]["odds"].mean()) if n_wins else 0.0
        avg_odds_all = float(strat["odds"].mean()) if n_bets else 0.0
        avg_stake = total_staked / n_bets if n_bets else 0.0
        max_dd = _max_drawdown(strat["pnl"].values) if n_bets else 0.0
        # CLV = model_prob × odds; > 1 means we beat the closing market
        avg_clv = float(strat["clv"].mean()) if (n_bets and "clv" in strat.columns and strat["clv"].notna().any()) else None

        stats[strategy] = {
            "bets": n_bets,
            "winners": n_wins,
            "placed": n_placed,
            "total_staked": round(total_staked, 2),
            "pnl": round(total_pnl, 2),
            "strike_rate": round(strike_rate, 1),
            "place_rate": round(place_rate, 1),
            "roi": round(roi, 1),
            "avg_stake": round(avg_stake, 4),
            "avg_odds_winners": round(avg_odds_win, 2),
            "avg_odds_all": round(avg_odds_all, 2),
            "max_drawdown": round(max_dd, 2),
            "avg_clv": round(avg_clv, 4) if avg_clv is not None else None,
        }

    if has_odds and "is_value_bet" in analysis_df.columns and "value" in stats:
        value_subset = analysis_df[analysis_df["is_value_bet"]].copy()
        if not value_subset.empty:
            _vb_y = value_subset["won"].astype(int).values
            _vb_probs = np.clip(
                value_subset["model_prob"].values.astype(np.float64),
                1e-15,
                1 - 1e-15,
            )
            stats["value"]["avg_edge"] = round(float(value_subset["value_edge"].mean()), 4)
            stats["value"]["avg_clv"] = round(float(value_subset["value_clv"].mean()), 4)
            stats["value"]["expected_roi"] = round(float(value_subset["value_expected_roi"].mean()) * 100.0, 1)
            try:
                stats["value"]["selected_brier"] = round(float(brier_score_loss(_vb_y, _vb_probs)), 6)
            except Exception:
                stats["value"]["selected_brier"] = None
            try:
                stats["value"]["selected_log_loss"] = round(float(log_loss(_vb_y, _vb_probs)), 4)
            except Exception:
                stats["value"]["selected_log_loss"] = None
        else:
            stats["value"]["avg_edge"] = None
            stats["value"]["avg_clv"] = None
            stats["value"]["expected_roi"] = None
            stats["value"]["selected_brier"] = None
            stats["value"]["selected_log_loss"] = None

    # ── Daily P&L (for bar chart) ────────────────────────────────
    for strategy in ["top_pick", "value", "each_way"]:
        strat = (
            bets_df[bets_df["strategy"] == strategy].copy()
            if not bets_df.empty else pd.DataFrame()
        )
        if not strat.empty:
            daily = strat.groupby("race_date").agg(
                daily_pnl=("pnl", "sum"),
                daily_bets=("pnl", "count"),
                daily_wins=("won", "sum"),
            ).reset_index()
            daily["cum_pnl"] = daily["daily_pnl"].cumsum()
            stats[f"{strategy}_daily"] = daily.to_dict("records")

    # ── Value bets by odds band ──────────────────────────────────
    if not bets_df.empty and has_odds:
        value_bets = bets_df[bets_df["strategy"] == "value"].copy()
        if not value_bets.empty:
            bins = [0, 3, 5, 8, 12, 20, 9999]
            labels = ["1-3", "3-5", "5-8", "8-12", "12-20", "20+"]
            value_bets["odds_band"] = pd.cut(
                value_bets["odds"], bins=bins, labels=labels,
            )
            band = value_bets.groupby("odds_band", observed=True).agg(
                bets=("pnl", "count"),
                winners=("won", "sum"),
                pnl=("pnl", "sum"),
            ).reset_index()
            band["strike_rate"] = (
                band["winners"] / band["bets"] * 100
            ).round(1)
            band["roi"] = (
                band["pnl"] / band["bets"] * 100
            ).round(1)
            stats["value_by_odds_band"] = band.to_dict("records")

    # ── Calibration: model prob bucket vs actual win rate ────────
    cal_bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
    cal_labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-50%", "50%+"]
    analysis_df["prob_bucket"] = pd.cut(
        analysis_df["model_prob"], bins=cal_bins, labels=cal_labels,
    )
    cal = analysis_df.groupby("prob_bucket", observed=True).agg(
        runners=("won", "count"),
        actual_wins=("won", "sum"),
        avg_model_prob=("model_prob", "mean"),
    ).reset_index()
    cal["actual_win_rate"] = (
        cal["actual_wins"] / cal["runners"] * 100
    ).round(1)
    cal["avg_model_pct"] = (cal["avg_model_prob"] * 100).round(1)
    calibration = cal.to_dict("records")

    # ── Test-set date range ──────────────────────────────────────
    test_dates = pd.to_datetime(analysis_df["race_date"])
    date_range = (
        str(test_dates.min().date()),
        str(test_dates.max().date()),
    )

    logger.info(
        f"\nTest-Set Analysis ({date_range[0]} → {date_range[1]}):\n"
        f"  Top-Pick: {stats.get('top_pick', {}).get('bets', 0)} bets, "
        f"SR {stats.get('top_pick', {}).get('strike_rate', 0):.1f}%, "
        f"ROI {stats.get('top_pick', {}).get('roi', 0):+.1f}%, "
        f"P&L {stats.get('top_pick', {}).get('pnl', 0):+.2f}\n"
        f"  Value:    {stats.get('value', {}).get('bets', 0)} bets, "
        f"SR {stats.get('value', {}).get('strike_rate', 0):.1f}%, "
        f"ROI {stats.get('value', {}).get('roi', 0):+.1f}%, "
        f"P&L {stats.get('value', {}).get('pnl', 0):+.2f}\n"
        f"  Each-Way: {stats.get('each_way', {}).get('bets', 0)} bets, "
        f"SR {stats.get('each_way', {}).get('strike_rate', 0):.1f}%, "
        f"ROI {stats.get('each_way', {}).get('roi', 0):+.1f}%, "
        f"P&L {stats.get('each_way', {}).get('pnl', 0):+.2f}"
    )

    _value_config = {
        "value_threshold": value_threshold,
        "staking_mode": staking_mode,
        "kelly_fraction": kelly_fraction,
        "starting_bankroll": bankroll,
        "final_bankroll": round(_current_bankroll, 2) if use_kelly else None,
    }

    # ── Slim predictions cache (for strategy calibration) ────────
    # Save the per-runner model/place probabilities so the calibration
    # page can skip re-running predict_race() on subsequent visits.
    _pred_cols = ["race_id", "horse_name"]
    for _c in ["odds", "jockey", "trainer", "race_date"]:
        if _c in analysis_df.columns:
            _pred_cols.append(_c)
    _pred_cols.append("model_prob")
    if "place_prob" in analysis_df.columns:
        _pred_cols.append("place_prob")
    predictions_df = analysis_df[_pred_cols].copy()

    return {
        "bets": bets_df,
        "curves": curves_df,
        "predictions": predictions_df,
        "stats": stats,
        "calibration": calibration,
        "test_date_range": date_range,
        "test_races": analysis_df["race_id"].nunique(),
        "test_runners": len(analysis_df),
        "value_config": _value_config,
    }
