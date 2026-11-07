"""Closing-line value (CLV) against Betfair BSP.
=================================================

CLV measures how a bet's taken price compares to the market's closing price —
here the **Betfair Starting Price** (BSP), the sharpest available estimate of a
runner's true chance. Beating the close consistently is the single best proxy
for a real, durable edge, so the training pipeline reports CLV alongside
calibration (Brier/log-loss) and backtest ROI.

``compute_clv`` takes a bets frame (as produced by ``bet_analysis.analyse_test_
set``) and joins it to the selected dataset's ``betfair/bsp`` table through
``src.source_prices`` (saved canonical IDs; one shared rule for runners that
only have names), then summarises CLV overall and per strategy. Bets whose race has no BSP row (e.g.
Irish-only races, or dates beyond BSP coverage) are excluded from the CLV
denominator and reported as ``coverage``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def summarise_clv(clv) -> dict:
    """Summarise ``taken_odds / bsp`` ratios. Shared by every CLV report.

    .. warning::
       **Never judge CLV on the mean alone.** The mean of a price *ratio* is
       upward-biased, and badly so at long odds, because the upside is
       unbounded while the downside is not: a 25.0 shot closing at 12.5 scores
       2.0, closing at 50.0 scores 0.5. Averaging those gives 1.25 for a pair
       of moves that are symmetric in probability terms.

       Measured on real selections this is the difference between a positive
       and a negative verdict. At odds 25-50 one set of picks scored mean CLV
       **1.0229** while its median was **0.9691** and its geometric mean
       **0.9594** — the mean said the picks beat the close, the other two said
       they drifted. ``median_clv``, ``geometric_mean_clv`` and
       ``pct_beat_close`` are the honest readings; prefer them, and treat a
       mean that disagrees with them as evidence of a long-price tail rather
       than of an edge.
    """
    clv = np.asarray(clv, dtype=np.float64).ravel()
    clv = clv[np.isfinite(clv) & (clv > 0)]
    if clv.size == 0:
        return {"n": 0}
    log_clv = np.log(clv)
    return {
        "n": int(clv.size),
        "mean_clv": float(np.mean(clv)),            # biased high; see the warning
        "median_clv": float(np.median(clv)),
        "geometric_mean_clv": float(np.exp(np.mean(log_clv))),
        "mean_log_clv": float(np.mean(log_clv)),
        "pct_beat_close": float(np.mean(clv > 1.0)),
    }


#: Backwards-compatible private alias.
_summarise = summarise_clv


def compute_clv(bets: pd.DataFrame, ref_df: pd.DataFrame | None = None) -> dict:
    """Summarise closing-line value of *bets* against the selected dataset's BSP.

    Args:
        bets: bets frame with ``odds`` and either ``race_id`` + ``horse_id``
            or ``race_date``, ``horse_name``, ``off_time`` and (``track`` or
            ``track_key``); ``strategy`` is used when present. ``race_id`` lets
            us recover ``off_time`` from *ref_df* when the bets frame lacks it.
        ref_df: optional frame (e.g. the featured/test set) carrying
            ``race_id`` + ``off_time`` so the off-time key can be filled in.

    Returns:
        ``{"overall": {...}, "by_strategy": {name: {...}}, "coverage": float,
        "n_bets": int, "n_matched": int, "bsp_available": bool}``.
    """
    from src.source_dataset import read_table, source_path
    from src.source_prices import attach_bsp
    out: dict = {"n_bets": int(len(bets)), "n_matched": 0, "coverage": 0.0,
                 "bsp_available": source_path("betfair", "bsp").exists(), "overall": {}, "by_strategy": {}}
    if bets is None or bets.empty or not out["bsp_available"] or "odds" not in bets.columns:
        return out

    b = bets.copy()
    # off_time: prefer the bets frame, else map race_id -> off_time from ref_df.
    if "off_time" not in b.columns and ref_df is not None and "off_time" in ref_df.columns \
            and "race_id" in b.columns and "race_id" in ref_df.columns:
        off_map = ref_df.drop_duplicates("race_id").set_index("race_id")["off_time"]
        b["off_time"] = b["race_id"].map(off_map)
    if "track" not in b:
        b["track"] = b["track_key"]
    if "off_time" not in b:
        b["off_time"] = ""
    bsp = read_table("betfair", "bsp", linked=True)
    merged = attach_bsp(b.drop(columns="bsp", errors="ignore"), bsp, cols=("bsp",))
    numeric_bsp = pd.to_numeric(merged.bsp, errors="coerce")
    merged["bsp"] = numeric_bsp.where(np.isfinite(numeric_bsp) & numeric_bsp.gt(1.0))
    matched = merged.dropna(subset=["bsp"])
    out["n_matched"] = int(len(matched))
    out["coverage"] = float(len(matched) / len(merged)) if len(merged) else 0.0
    if matched.empty:
        return out

    clv = pd.to_numeric(matched["odds"], errors="coerce").to_numpy(dtype=float) / \
        pd.to_numeric(matched["bsp"], errors="coerce").to_numpy(dtype=float)
    out["overall"] = _summarise(clv)
    if "strategy" in matched.columns:
        for strat, grp in matched.groupby("strategy"):
            s_clv = pd.to_numeric(grp["odds"], errors="coerce").to_numpy(float) / \
                pd.to_numeric(grp["bsp"], errors="coerce").to_numpy(float)
            out["by_strategy"][str(strat)] = _summarise(s_clv)
    return out
