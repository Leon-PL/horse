"""
Strategy Calibrator — grid-search over betting parameters on the test set.

Precomputes all per-runner columns once (EW terms, place edges, P&Ls),
then evaluates each parameter combination with fast vectorized masking.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from src.each_way import (
    get_ew_terms,
    adjust_place_probs_for_race,
)

logger = logging.getLogger(__name__)


# ── Parameter grid defaults ──────────────────────────────────────────

DEFAULT_GRID = {
    "value_threshold": [0.02, 0.03, 0.05, 0.07, 0.10, 0.15],
    "min_place_edge": [0.02, 0.03, 0.05, 0.07, 0.10, 0.15],
    "min_odds": [2.0, 3.0, 4.0, 5.0, 6.0],
    "max_odds": [21.0, 31.0, 51.0, 101.0],
    "kelly_fraction": [0.10, 0.15, 0.20, 0.25, 0.33, 0.50],
}


# ── Helpers ──────────────────────────────────────────────────────────

def _max_drawdown(pnls: np.ndarray) -> float:
    if len(pnls) == 0:
        return 0.0
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max())


def _sharpe(pnls: np.ndarray) -> float:
    if len(pnls) < 2:
        return 0.0
    s = float(np.std(pnls))
    return float(np.mean(pnls) / s) if s > 1e-9 else 0.0


# ── Precomputation (runs once) ───────────────────────────────────────

def precompute_analysis(analysis_df: pd.DataFrame, ew_fraction: float = 0.25) -> pd.DataFrame:
    """
    Enrich analysis_df with all per-runner columns needed for fast
    vectorized grid search.  Should be called ONCE before run_grid_search.

    Adds columns:
        normalized_edge, value_pnl_flat, ew_eligible, place_edge,
        place_odds, ew_pnl_flat, ew_is_value, ew_placed
    """
    df = analysis_df.copy()
    N = len(df)

    odds = df["odds"].values.astype(np.float64)
    prob = df["model_prob"].values.astype(np.float64)
    won = df["won"].values.astype(np.int8)

    # ── Value-bet precomputation (fully vectorized) ──────────
    implied = np.where(odds > 0, 1.0 / odds, 0.0)
    edge = prob - implied
    odds_scale = np.sqrt(np.maximum(odds / 3.0, 1e-9))
    df["normalized_edge"] = edge / odds_scale
    df["value_pnl_flat"] = np.where(won == 1, odds - 1.0, -1.0)

    # ── EW precomputation (per-race via numpy arrays) ────────
    ew_eligible = np.zeros(N, dtype=np.bool_)
    place_edge_arr = np.zeros(N, dtype=np.float64)
    place_odds_arr = np.zeros(N, dtype=np.float64)
    ew_pnl_arr = np.zeros(N, dtype=np.float64)
    ew_is_value_arr = np.zeros(N, dtype=np.bool_)
    ew_placed_arr = np.zeros(N, dtype=np.bool_)
    ew_won_arr = np.zeros(N, dtype=np.bool_)

    has_place = "place_prob" in df.columns
    if has_place:
        place_prob_all = df["place_prob"].values.astype(np.float64)
        fp_all = df["finish_position"].values.astype(np.int32)
        nr_col = df["num_runners"].values if "num_runners" in df.columns else None
        hcap_col = df["handicap"].values if "handicap" in df.columns else None

        # Build group boundaries via race_id
        race_ids = df["race_id"].values
        # Use pandas groupby to get start/end positions efficiently
        grp = df.groupby("race_id", sort=False)
        starts = grp.cumcount().values == 0
        group_starts = np.where(starts)[0]
        group_sizes = grp.size().values

        for gi in range(len(group_starts)):
            s = group_starts[gi]
            n = group_sizes[gi]
            e = s + n

            nr = int(nr_col[s]) if nr_col is not None else n
            is_hcap = bool(hcap_col[s]) if hcap_col is not None else False
            ew_terms = get_ew_terms(nr, is_handicap=is_hcap, fraction_override=ew_fraction)
            if not ew_terms.eligible:
                continue

            ew_eligible[s:e] = True

            raw_pp = place_prob_all[s:e]
            win_pp = prob[s:e]
            adj_place = adjust_place_probs_for_race(raw_pp, win_pp, ew_terms.places_paid)

            race_odds = odds[s:e]
            p_odds = 1.0 + (race_odds - 1.0) * ew_terms.fraction
            imp_place = np.where(p_odds > 0, 1.0 / p_odds, 0.0)
            p_edge = adj_place - imp_place

            win_ev = win_pp * race_odds - 1.0
            place_ev = adj_place * p_odds - 1.0
            ew_ev = (win_ev + place_ev) / 2.0
            is_ew_val = (ew_ev > 0) | (place_ev > 0)

            race_fp = fp_all[s:e]
            race_won = won[s:e]
            placed = (race_fp > 0) & (race_fp <= ew_terms.places_paid)

            pnl = np.full(n, -2.0)
            w_mask = race_won == 1
            p_only = placed & ~w_mask
            pnl[w_mask] = -2.0 + race_odds[w_mask] + p_odds[w_mask]
            pnl[p_only] = -2.0 + p_odds[p_only]

            place_edge_arr[s:e] = p_edge
            place_odds_arr[s:e] = p_odds
            ew_pnl_arr[s:e] = pnl
            ew_is_value_arr[s:e] = is_ew_val
            ew_placed_arr[s:e] = placed | w_mask
            ew_won_arr[s:e] = w_mask

    df["ew_eligible"] = ew_eligible
    df["place_edge"] = place_edge_arr
    df["place_odds"] = place_odds_arr
    df["ew_pnl_flat"] = ew_pnl_arr
    df["ew_is_value"] = ew_is_value_arr
    df["ew_placed"] = ew_placed_arr
    df["ew_won"] = ew_won_arr

    return df


# ── Fast vectorized evaluation ───────────────────────────────────────

def _eval_flat(
    pre: pd.DataFrame,
    vt: float, mpe: float, mino: float, maxo: float,
) -> dict:
    """Evaluate one param combo with flat staking — fully vectorized."""

    # Value bets (apply same odds range filter as EW)
    odds_vals = pre["odds"].values
    v_mask = (
        (pre["normalized_edge"].values > vt)
        & (odds_vals >= mino)
        & (odds_vals <= maxo)
    )
    v_pnl_arr = pre["value_pnl_flat"].values[v_mask]
    v_odds_arr = odds_vals[v_mask]
    v_won_arr = pre["won"].values[v_mask]
    v_n = int(v_mask.sum())

    # EW bets
    ew_mask = (
        pre["ew_eligible"].values
        & pre["ew_is_value"].values
        & (pre["place_edge"].values > mpe)
        & (pre["odds"].values >= mino)
        & (pre["odds"].values <= maxo)
    )
    e_pnl_arr = pre["ew_pnl_flat"].values[ew_mask]
    e_odds_arr = pre["odds"].values[ew_mask]
    e_won_arr = pre["ew_won"].values[ew_mask]
    e_placed_arr = pre["ew_placed"].values[ew_mask]
    e_n = int(ew_mask.sum())

    # Value stats
    v_pnl = float(v_pnl_arr.sum()) if v_n else 0.0
    v_winners = int(v_won_arr.sum()) if v_n else 0
    v_staked = float(v_n)
    v_roi = (v_pnl / v_staked * 100) if v_staked > 0 else 0.0
    v_strike = (v_winners / v_n * 100) if v_n > 0 else 0.0
    v_avg = float(v_odds_arr.mean()) if v_n > 0 else 0.0

    # EW stats
    e_pnl = float(e_pnl_arr.sum()) if e_n else 0.0
    e_winners = int(e_won_arr.sum()) if e_n else 0
    e_placed = int(e_placed_arr.sum()) if e_n else 0
    e_staked = float(e_n * 2.0)
    e_roi = (e_pnl / e_staked * 100) if e_staked > 0 else 0.0
    e_place_rate = (e_placed / e_n * 100) if e_n > 0 else 0.0
    e_avg = float(e_odds_arr.mean()) if e_n > 0 else 0.0

    # Combined
    c_pnl = v_pnl + e_pnl
    c_staked = v_staked + e_staked
    c_roi = (c_pnl / c_staked * 100) if c_staked > 0 else 0.0

    all_pnl = np.concatenate([v_pnl_arr, e_pnl_arr]) if (v_n or e_n) else np.array([])

    return {
        "value_threshold": vt,
        "min_place_edge": mpe,
        "min_odds": mino,
        "max_odds": maxo,
        "kelly_fraction": 0.0,
        "staking_mode": "flat",
        "value_bets": v_n,
        "value_winners": v_winners,
        "value_pnl": round(v_pnl, 4),
        "value_staked": v_staked,
        "value_roi": round(v_roi, 2),
        "value_strike": round(v_strike, 2),
        "value_avg_odds": round(v_avg, 2),
        "value_max_dd": round(_max_drawdown(v_pnl_arr), 2),
        "value_sharpe": round(_sharpe(v_pnl_arr), 4),
        "ew_bets": e_n,
        "ew_winners": e_winners,
        "ew_placed": e_placed,
        "ew_pnl": round(e_pnl, 4),
        "ew_staked": e_staked,
        "ew_roi": round(e_roi, 2),
        "ew_place_rate": round(e_place_rate, 2),
        "ew_avg_odds": round(e_avg, 2),
        "ew_max_dd": round(_max_drawdown(e_pnl_arr), 2),
        "ew_sharpe": round(_sharpe(e_pnl_arr), 4),
        "combined_pnl": round(c_pnl, 4),
        "combined_staked": c_staked,
        "combined_roi": round(c_roi, 2),
        "combined_sharpe": round(_sharpe(all_pnl), 4),
    }


# ── Grid search ──────────────────────────────────────────────────────

def _kelly_value_loop(cand_odds, cand_prob, cand_won, kf, bankroll):
    """Run sequential Kelly bankroll simulation on pre-filtered candidates.
    
    Uses pre-computed numpy arrays for speed.
    Stakes are capped at ``max_stake_mult`` × initial bankroll per bet
    to prevent unrealistic exponential compounding.
    """
    n = len(cand_odds)
    if n == 0:
        return (
            np.array([], dtype=np.float64),
            0,
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    # Precompute Kelly fractions and bet returns vectorized
    b = cand_odds - 1.0
    raw_kelly = np.where(b > 0, ((b * cand_prob - (1.0 - cand_prob)) / b) * kf, 0.0)
    raw_kelly = np.maximum(raw_kelly, 0.0)

    max_stake = bankroll * 0.5  # Cap any single bet at 50% of *initial* bankroll

    # Sequential bankroll simulation — tight loop, pure scalars
    bank = bankroll
    pnls = []
    stakes = []
    bet_odds_list = []
    winners = 0
    for j in range(n):
        k = raw_kelly[j]
        if k <= 0 or bank <= 0:
            continue
        stake = min(round(k * bank, 4), bank, max_stake)
        if stake < 0.01:
            continue
        if cand_won[j]:
            pnl = stake * b[j]
            winners += 1
        else:
            pnl = -stake
        stakes.append(stake)
        pnls.append(pnl)
        bet_odds_list.append(cand_odds[j])
        bank += pnl

    return (
        np.array(pnls, dtype=np.float64),
        winners,
        np.array(stakes, dtype=np.float64),
        np.array(bet_odds_list, dtype=np.float64),
    )


def run_grid_search(
    analysis_df: pd.DataFrame,
    grid: dict[str, list] | None = None,
    staking_mode: str = "flat",
    bankroll: float = 100.0,
    ew_fraction: float = 0.25,
    progress_fn=None,
) -> pd.DataFrame:
    """
    Evaluate all parameter combos with fast vectorised operations (flat)
    or efficient sequential (Kelly).

    If ``analysis_df`` already has the precomputed columns
    (``normalized_edge``, ``ew_pnl_flat``, etc.) from
    ``precompute_analysis()``, they are reused.  Otherwise
    ``precompute_analysis()`` is called automatically.
    """
    if grid is None:
        grid = DEFAULT_GRID.copy()

    # Skip precompute if already done
    if "normalized_edge" in analysis_df.columns:
        pre = analysis_df
    else:
        logger.info("Precomputing per-runner data …")
        pre = precompute_analysis(analysis_df, ew_fraction=ew_fraction)
        logger.info("Precomputation done.")

    vt_vals = grid.get("value_threshold", DEFAULT_GRID["value_threshold"])
    mpe_vals = grid.get("min_place_edge", DEFAULT_GRID["min_place_edge"])
    mino_vals = grid.get("min_odds", DEFAULT_GRID["min_odds"])
    maxo_vals = grid.get("max_odds", DEFAULT_GRID["max_odds"])

    use_kelly = staking_mode == "kelly"
    kf_vals = grid.get("kelly_fraction", DEFAULT_GRID["kelly_fraction"]) if use_kelly else [0.0]

    results: list[dict] = []
    counter = 0

    if not use_kelly:
        # ── Flat staking: fully vectorized ───────────────────
        combos = list(itertools.product(vt_vals, mpe_vals, mino_vals, maxo_vals))
        total = len(combos)
        for i, (vt, mpe, mino, maxo) in enumerate(combos):
            if mino >= maxo:
                continue
            results.append(_eval_flat(pre, vt, mpe, mino, maxo))
            counter += 1
            if progress_fn and counter % 100 == 0:
                progress_fn(counter, total)
        if progress_fn:
            progress_fn(total, total)
    else:
        # ── Kelly: value bet results only depend on (vt, kf), ──
        # ── EW only depends on (mpe, mino, maxo)              ──
        odds_all = pre["odds"].values.astype(np.float64)
        prob_all = pre["model_prob"].values.astype(np.float64)
        won_all = pre["won"].values.astype(np.int8)
        norm_edge_all = pre["normalized_edge"].values.astype(np.float64)

        ew_elig = pre["ew_eligible"].values
        ew_isval = pre["ew_is_value"].values
        ew_pedge = pre["place_edge"].values
        ew_pnl_flat = pre["ew_pnl_flat"].values
        ew_won_v = pre["ew_won"].values
        ew_placed_v = pre["ew_placed"].values

        _valid_odds_pairs = sum(
            1 for a in mino_vals for b in maxo_vals if a < b
        )
        total = len(vt_vals) * len(mpe_vals) * _valid_odds_pairs * len(kf_vals)

        # 1) Cache value Kelly results per (vt, mino, maxo, kf)
        value_cache: dict[tuple, dict] = {}
        for vt in vt_vals:
          for mino, maxo in itertools.product(mino_vals, maxo_vals):
            if mino >= maxo:
                continue
            cand_mask = (norm_edge_all > vt) & (odds_all >= mino) & (odds_all <= maxo)
            cand_odds = odds_all[cand_mask].copy()
            cand_prob = prob_all[cand_mask].copy()
            cand_won = won_all[cand_mask].copy()

            for kf in kf_vals:
                v_arr, v_winners, v_stakes_arr, v_odds_arr = _kelly_value_loop(
                    cand_odds, cand_prob, cand_won, kf, bankroll,
                )
                v_n = len(v_arr)
                v_pnl = float(v_arr.sum()) if v_n else 0.0
                v_staked = float(v_stakes_arr.sum()) if v_n else 0.0
                value_cache[(vt, mino, maxo, kf)] = {
                    "v_arr": v_arr,
                    "v_n": v_n,
                    "v_pnl": v_pnl,
                    "v_staked": v_staked,
                    "v_winners": v_winners,
                    # Use actual placed-bet odds, not all candidate odds
                    "v_avg_odds": round(float(v_odds_arr.mean()), 2) if v_n > 0 else 0.0,
                    "v_max_dd": round(_max_drawdown(v_arr), 2),
                    "v_sharpe": round(_sharpe(v_arr), 4),
                    "v_roi": round(v_pnl / v_staked * 100, 2) if v_staked > 0 else 0.0,
                    "v_strike": round(v_winners / v_n * 100, 2) if v_n > 0 else 0.0,
                }

        # 2) Combine cached value results with vectorized EW
        for vt in vt_vals:
            for mpe, mino, maxo in itertools.product(mpe_vals, mino_vals, maxo_vals):
                if mino >= maxo:
                    continue

                ew_mask = ew_elig & ew_isval & (ew_pedge > mpe) & (odds_all >= mino) & (odds_all <= maxo)
                e_pnl_arr = ew_pnl_flat[ew_mask]
                e_odds_arr = odds_all[ew_mask]
                e_won_arr = ew_won_v[ew_mask]
                e_placed_arr = ew_placed_v[ew_mask]
                e_n = int(ew_mask.sum())
                e_pnl = float(e_pnl_arr.sum()) if e_n else 0.0
                e_staked = float(e_n * 2.0)

                for kf in kf_vals:
                    vc = value_cache[(vt, mino, maxo, kf)]
                    v_pnl = vc["v_pnl"]
                    v_staked = vc["v_staked"]

                    c_pnl = v_pnl + e_pnl
                    c_staked = v_staked + e_staked
                    all_pnl = np.concatenate([vc["v_arr"], e_pnl_arr]) if (vc["v_n"] or e_n) else np.array([])

                    results.append({
                        "value_threshold": vt,
                        "min_place_edge": mpe,
                        "min_odds": mino,
                        "max_odds": maxo,
                        "kelly_fraction": kf,
                        "staking_mode": "kelly",
                        "value_bets": vc["v_n"],
                        "value_winners": vc["v_winners"],
                        "value_pnl": round(v_pnl, 4),
                        "value_staked": round(v_staked, 4),
                        "value_roi": vc["v_roi"],
                        "value_strike": vc["v_strike"],
                        "value_avg_odds": vc["v_avg_odds"],
                        "value_max_dd": vc["v_max_dd"],
                        "value_sharpe": vc["v_sharpe"],
                        "ew_bets": e_n,
                        "ew_winners": int(e_won_arr.sum()) if e_n else 0,
                        "ew_placed": int(e_placed_arr.sum()) if e_n else 0,
                        "ew_pnl": round(e_pnl, 4),
                        "ew_staked": e_staked,
                        "ew_roi": round(e_pnl / e_staked * 100, 2) if e_staked > 0 else 0.0,
                        "ew_place_rate": round(int(e_placed_arr.sum()) / e_n * 100, 2) if e_n > 0 else 0.0,
                        "ew_avg_odds": round(float(e_odds_arr.mean()), 2) if e_n > 0 else 0.0,
                        "ew_max_dd": round(_max_drawdown(e_pnl_arr), 2),
                        "ew_sharpe": round(_sharpe(e_pnl_arr), 4),
                        "combined_pnl": round(c_pnl, 4),
                        "combined_staked": round(c_staked, 4),
                        "combined_roi": round(c_pnl / c_staked * 100, 2) if c_staked > 0 else 0.0,
                        "combined_sharpe": round(_sharpe(all_pnl), 4),
                    })

                    counter += 1
                    if progress_fn and counter % 100 == 0:
                        progress_fn(counter, total)

        if progress_fn:
            progress_fn(total, total)

    df = pd.DataFrame(results)
    df = df.sort_values("combined_pnl", ascending=False).reset_index(drop=True)
    return df


# ── Validated grid search (temporal OOS) ─────────────────────────────

def run_validated_grid_search(
    analysis_df: pd.DataFrame,
    grid: dict[str, list] | None = None,
    staking_mode: str = "flat",
    bankroll: float = 100.0,
    ew_fraction: float = 0.25,
    sort_by: str = "combined_pnl",
    min_bets: int = 10,
    val_fraction: float = 0.5,
    progress_fn=None,
) -> dict:
    """Run grid search on a calibration half, validate on a held-out half.

    Splits *analysis_df* temporally into a calibration set (first
    ``1 - val_fraction``) and a validation set (last ``val_fraction``),
    aligned to race boundaries.

    Returns a dict with:
        ``cal_results``  — full grid search DataFrame (calibration set)
        ``val_results``  — re-evaluated top combos on validation set
        ``best_cal``     — best row from calibration
        ``best_val``     — that same combo's performance on validation
        ``cal_races``    — number of races in calibration set
        ``val_races``    — number of races in validation set
        ``combos_tried`` — total parameter combos evaluated
    """
    df = analysis_df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date", "race_id"]).reset_index(drop=True)

    # Temporal split aligned to race boundary
    split_idx = int(len(df) * (1 - val_fraction))
    split_race = df.iloc[split_idx]["race_id"]
    while split_idx < len(df) and df.iloc[split_idx]["race_id"] == split_race:
        split_idx += 1

    cal_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    cal_races = int(cal_df["race_id"].nunique())
    val_races = int(val_df["race_id"].nunique())

    logger.info(
        f"Validated grid search: {cal_races} cal races, "
        f"{val_races} val races"
    )

    # Precompute per-runner columns for both halves
    cal_pre = precompute_analysis(cal_df, ew_fraction=ew_fraction)
    val_pre = precompute_analysis(val_df, ew_fraction=ew_fraction)

    # Run full grid search on calibration half
    cal_results = run_grid_search(
        cal_pre, grid=grid, staking_mode=staking_mode,
        bankroll=bankroll, ew_fraction=ew_fraction,
        progress_fn=progress_fn,
    )

    combos_tried = len(cal_results)

    # Filter and sort calibration results.
    # For combined metrics require EACH active leg to meet min_bets independently
    # (prevents combos with e.g. 20 win bets + 80 EW bets passing a min of 90).
    _ew_active = (cal_results["ew_bets"] > 0).any() if "ew_bets" in cal_results.columns else False
    if sort_by.startswith("value"):
        cal_mask = cal_results["value_bets"] >= min_bets
    elif sort_by.startswith("ew"):
        cal_mask = cal_results["ew_bets"] >= min_bets
    else:
        # combined_* — require each active leg to meet the threshold individually
        if _ew_active:
            cal_mask = (cal_results["value_bets"] >= min_bets) & (cal_results["ew_bets"] >= min_bets)
        else:
            cal_mask = cal_results["value_bets"] >= min_bets

    cal_filtered = cal_results[cal_mask].copy()
    cal_filtered = cal_filtered.sort_values(
        sort_by, ascending=False
    ).reset_index(drop=True)

    # Re-evaluate top combos on validation set
    param_cols = ["value_threshold", "min_place_edge", "min_odds", "max_odds"]
    if staking_mode == "kelly":
        param_cols.append("kelly_fraction")

    # Evaluate ALL filtered combos on OOS for flat staking (vectorized, fast).
    # For Kelly limit to 200 since each combo requires a sequential loop.
    n_top = len(cal_filtered) if staking_mode == "flat" else min(200, len(cal_filtered))
    val_rows: list[dict] = []

    for i in range(n_top):
        row = cal_filtered.iloc[i]
        vt = float(row["value_threshold"])
        mpe = float(row["min_place_edge"])
        mino = float(row["min_odds"])
        maxo = float(row["max_odds"])

        if staking_mode == "flat":
            val_row = _eval_flat(val_pre, vt, mpe, mino, maxo)
        else:
            kf = float(row["kelly_fraction"])
            # Re-run Kelly on validation set
            odds_v = val_pre["odds"].values.astype(np.float64)
            prob_v = val_pre["model_prob"].values.astype(np.float64)
            won_v = val_pre["won"].values.astype(np.int8)
            ne_v = val_pre["normalized_edge"].values.astype(np.float64)

            cand_mask = (ne_v > vt) & (odds_v >= mino) & (odds_v <= maxo)
            v_arr, v_winners, v_stakes_arr, v_odds_arr_val = _kelly_value_loop(
                odds_v[cand_mask], prob_v[cand_mask],
                won_v[cand_mask], kf, bankroll,
            )
            v_n = len(v_arr)
            v_pnl = float(v_arr.sum()) if v_n else 0.0
            v_staked = float(v_stakes_arr.sum()) if v_n else 0.0

            # EW part (flat within Kelly mode)
            ew_mask = (
                val_pre["ew_eligible"].values
                & val_pre["ew_is_value"].values
                & (val_pre["place_edge"].values > mpe)
                & (odds_v >= mino) & (odds_v <= maxo)
            )
            e_pnl_arr = val_pre["ew_pnl_flat"].values[ew_mask]
            e_odds_arr = odds_v[ew_mask]
            e_won_arr = val_pre["ew_won"].values[ew_mask]
            e_placed_arr = val_pre["ew_placed"].values[ew_mask]
            e_n = int(ew_mask.sum())
            e_pnl = float(e_pnl_arr.sum()) if e_n else 0.0
            e_staked = float(e_n * 2.0)
            e_winners = int(e_won_arr.sum()) if e_n else 0
            e_placed = int(e_placed_arr.sum()) if e_n else 0

            c_pnl = v_pnl + e_pnl
            c_staked = v_staked + e_staked
            all_pnl = np.concatenate([v_arr, e_pnl_arr]) if (v_n or e_n) else np.array([])

            val_row = {
                "value_threshold": vt, "min_place_edge": mpe,
                "min_odds": mino, "max_odds": maxo,
                "kelly_fraction": kf, "staking_mode": "kelly",
                "value_bets": v_n, "value_winners": v_winners,
                "value_pnl": round(v_pnl, 4),
                "value_staked": round(v_staked, 4),
                "value_roi": round(v_pnl / v_staked * 100, 2) if v_staked > 0 else 0.0,
                "value_strike": round(v_winners / v_n * 100, 2) if v_n > 0 else 0.0,
                "value_avg_odds": round(float(v_odds_arr_val.mean()), 2) if v_n > 0 else 0.0,
                "value_max_dd": round(_max_drawdown(v_arr), 2),
                "value_sharpe": round(_sharpe(v_arr), 4),
                "ew_bets": e_n,
                "ew_winners": e_winners,
                "ew_placed": e_placed,
                "ew_pnl": round(e_pnl, 4), "ew_staked": e_staked,
                "ew_roi": round(e_pnl / e_staked * 100, 2) if e_staked > 0 else 0.0,
                "ew_place_rate": round(e_placed / e_n * 100, 2) if e_n > 0 else 0.0,
                "ew_avg_odds": round(float(e_odds_arr.mean()), 2) if e_n > 0 else 0.0,
                "ew_max_dd": round(_max_drawdown(e_pnl_arr), 2),
                "ew_sharpe": round(_sharpe(e_pnl_arr), 4),
                "combined_pnl": round(c_pnl, 4),
                "combined_staked": round(c_staked, 4),
                "combined_roi": round(c_pnl / c_staked * 100, 2) if c_staked > 0 else 0.0,
                "combined_sharpe": round(_sharpe(all_pnl), 4),
            }

        val_rows.append(val_row)

    val_results = pd.DataFrame(val_rows) if val_rows else pd.DataFrame()

    # ── Apply min_bets filter to validation results ───────────────────
    if not val_results.empty:
        _ew_active_val = (val_results["ew_bets"] > 0).any() if "ew_bets" in val_results.columns else False
        if sort_by.startswith("value"):
            val_mask = val_results["value_bets"] >= min_bets
        elif sort_by.startswith("ew"):
            val_mask = val_results["ew_bets"] >= min_bets
        else:
            if _ew_active_val:
                val_mask = (val_results["value_bets"] >= min_bets) & (val_results["ew_bets"] >= min_bets)
            else:
                val_mask = val_results["value_bets"] >= min_bets
        val_filtered = val_results[val_mask].reset_index(drop=True)
    else:
        val_filtered = val_results

    # ── Build merged IS+OOS table with stability columns ─────────────
    # Join on parameter columns so each row shows IS metrics alongside
    # its OOS counterpart.  Stability = IS_roi - OOS_roi (lower = less overfit).
    _param_join = ["value_threshold", "min_place_edge", "min_odds", "max_odds"]
    if staking_mode == "kelly":
        _param_join.append("kelly_fraction")

    merged_results: pd.DataFrame = pd.DataFrame()
    if not val_results.empty:
        _oos_rename = {
            c: f"oos_{c}"
            for c in val_results.columns
            if c not in _param_join and not c.startswith("staking")
        }
        _oos_sub = val_results[_param_join + list(_oos_rename.keys())].rename(
            columns=_oos_rename
        )
        merged_results = cal_filtered.merge(_oos_sub, on=_param_join, how="left")
        # Stability columns
        merged_results["roi_decay"] = (
            merged_results["combined_roi"] - merged_results.get("oos_combined_roi", 0)
        ).round(2)
        # OOS / IS ratio — clipped to [-2, 2] to avoid inf on near-zero IS
        _is_roi = merged_results["combined_roi"].replace(0, np.nan)
        merged_results["stability_ratio"] = (
            (merged_results.get("oos_combined_roi", 0) / _is_roi)
            .clip(-2, 2)
            .round(3)
        )

    # IS-best: top of cal_filtered (sorted by IS metric)
    best_cal = cal_filtered.iloc[0].to_dict() if len(cal_filtered) > 0 else {}
    # IS-best's OOS performance
    best_val = val_filtered.iloc[0].to_dict() if len(val_filtered) > 0 else {}

    # OOS-best: sort val_filtered by the same metric and pick top
    best_oos_params: dict = {}
    best_oos_is: dict = {}
    if not val_filtered.empty and sort_by in val_filtered.columns:
        val_sorted = val_filtered.sort_values(sort_by, ascending=False).reset_index(drop=True)
        best_oos_params = val_sorted.iloc[0].to_dict()
        # Find the IS metrics for that same combo
        if not merged_results.empty:
            _oos_match = merged_results
            for _pc in _param_join:
                _oos_match = _oos_match[
                    _oos_match[_pc] == best_oos_params.get(_pc)
                ]
            if len(_oos_match) > 0:
                best_oos_is = _oos_match.iloc[0].to_dict()

    return {
        "cal_results": cal_results,
        "cal_filtered": cal_filtered,
        "val_results": val_results,
        "merged_results": merged_results,
        "best_cal": best_cal,
        "best_val": best_val,
        "best_oos_params": best_oos_params,
        "best_oos_is": best_oos_is,
        "cal_races": cal_races,
        "val_races": val_races,
        "combos_tried": combos_tried,
    }
