"""Where does the value filter lose money?

The market-combination walk-forward established something specific: the
combination **beats the model alone** (-3.55% vs -4.49% ROI @ BSP) yet still
**loses to backing everything in the band** (-2.68%). That is a statement about
the *selection rule*, not the blend — picking runners where
``combo_prob > mkt_imp`` chooses bets that are worse than average inside the
very band they are drawn from. A filter that is worse than no filter is doing
real work in the wrong direction, and nothing in this project has ever looked
at where.

The leading hypothesis is that the value filter is a **longshot filter in
disguise**. A model is least well calibrated at long odds, so that is where it
most often disagrees with the price; and long odds carry the worst effective
exchange return. If so, the filter concentrates stakes exactly where both the
model is weakest and the price is worst, and the fix is a price-aware rule
rather than a better model.

This script tests that, plus the obvious alternatives, on the per-bet rows
dumped by the walk-forward:

    python scripts/wf_benter_combination.py --max-folds 8 --dump data/processed/benter_bets.parquet
    python scripts/audit_selection_rule.py --bets data/processed/benter_bets.parquet

It is pure analysis — no model fitting — so a rule can be re-cut in seconds.
Every ROI is settled by the shared exchange formula and every interval is
race-clustered (``src/wf_eval.py``), so the numbers are comparable with the
walk-forward that produced the rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.wf_eval import bootstrap_roi_ci, fold_summary

DEFAULT_BETS = os.path.join("data", "processed", "benter_bets.parquet")

#: Price buckets. Exchange returns and model calibration both vary sharply with
#: price, so this is the first cut to make.
ODDS_BANDS = [(1.5, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 25.0), (25.0, 50.0)]

#: How far the combination's probability exceeds the market's, as a ratio.
EDGE_BUCKETS = [(1.0, 1.1), (1.1, 1.25), (1.25, 1.6), (1.6, 2.5), (2.5, np.inf)]

FIELD_BUCKETS = [(0, 6), (6, 9), (9, 13), (13, 100)]


def _roi(frame: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Race-clustered ROI for a set of bets."""
    if frame.empty:
        return {"n": 0, "roi": None, "ci_low": None, "ci_high": None, "races": 0}
    ci = bootstrap_roi_ci(
        frame["pnl_bsp"].to_numpy(),
        clusters=frame["race_id"].to_numpy(),
        n_boot=n_boot,
    )
    return {"n": ci["n"], "roi": ci["mean"], "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"], "races": ci["n_clusters"]}


def _clv(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    ratio = frame["entry_odds"].to_numpy() / frame["close_odds"].to_numpy()
    return round(float(np.nanmean(ratio)), 4)


def load_bets(path: str) -> pd.DataFrame:
    bets = pd.read_parquet(path)
    # The band is the universe every strategy is drawn from; outside it nothing
    # is tradeable, so it is not part of this comparison.
    bets = bets[bets["in_band"].astype(bool)].copy()
    bets["race_id"] = bets["race_id"].astype(str)
    bets["edge_ratio"] = bets["combo_prob"] / bets["mkt_imp"].clip(lower=1e-9)
    return bets


def by_band(bets: pd.DataFrame) -> list[dict]:
    """Selected vs unselected vs everything, per price bucket.

    The hypothesis stands or falls here: if the filter is a longshot filter,
    its share of stakes rises with price while its ROI gap to the band's own
    average goes negative at the long end.
    """
    rows = []
    for lo, hi in ODDS_BANDS:
        band = bets[(bets["entry_odds"] >= lo) & (bets["entry_odds"] < hi)]
        if band.empty:
            continue
        selected = band[band["sel_combo"].astype(bool)]
        rest = band[~band["sel_combo"].astype(bool)]
        sel_roi, all_roi = _roi(selected), _roi(band)
        rows.append({
            "band": f"{lo:g}-{hi:g}",
            "runners": len(band),
            "selected": len(selected),
            "select_rate": round(len(selected) / len(band), 4),
            "share_of_bets": None,                    # filled in by the caller
            "roi_selected": sel_roi["roi"],
            "ci": [sel_roi["ci_low"], sel_roi["ci_high"]],
            "roi_band": all_roi["roi"],
            "roi_unselected": _roi(rest)["roi"],
            "gap_vs_band": (round(sel_roi["roi"] - all_roi["roi"], 2)
                            if sel_roi["roi"] is not None and all_roi["roi"] is not None
                            else None),
            "clv_selected": _clv(selected),
            "mean_combo_prob": round(float(selected["combo_prob"].mean()), 4) if len(selected) else None,
            "actual_win_rate": round(float(selected["won"].mean()), 4) if len(selected) else None,
        })
    total_selected = sum(r["selected"] for r in rows) or 1
    for r in rows:
        r["share_of_bets"] = round(r["selected"] / total_selected, 4)
    return rows


def by_bucket(bets: pd.DataFrame, column: str, buckets, label: str) -> list[dict]:
    """ROI of the selected bets within arbitrary buckets of *column*."""
    selected = bets[bets["sel_combo"].astype(bool)]
    rows = []
    for lo, hi in buckets:
        chunk = selected[(selected[column] >= lo) & (selected[column] < hi)]
        if chunk.empty:
            continue
        stats = _roi(chunk)
        rows.append({
            label: f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+",
            "n": stats["n"], "roi": stats["roi"],
            "ci": [stats["ci_low"], stats["ci_high"]],
            "clv": _clv(chunk),
            "mean_combo_prob": round(float(chunk["combo_prob"].mean()), 4),
            "actual_win_rate": round(float(chunk["won"].mean()), 4),
        })
    return rows


def calibration_by_band(bets: pd.DataFrame) -> list[dict]:
    """Is the combination over-predicting where it selects?

    A value rule is only as good as the probability behind it. If predicted
    beats actual in the buckets the rule likes, the rule is buying its own
    calibration error.
    """
    rows = []
    for lo, hi in ODDS_BANDS:
        band = bets[(bets["entry_odds"] >= lo) & (bets["entry_odds"] < hi)]
        if band.empty:
            continue
        selected = band[band["sel_combo"].astype(bool)]
        if selected.empty:
            continue
        predicted = float(selected["combo_prob"].mean())
        actual = float(selected["won"].mean())
        rows.append({
            "band": f"{lo:g}-{hi:g}",
            "n": len(selected),
            "predicted": round(predicted, 4),
            "actual": round(actual, 4),
            "over_prediction": round(predicted - actual, 4),
            "market_implied": round(float(selected["mkt_imp"].mean()), 4),
        })
    return rows


def candidate_rules(bets: pd.DataFrame) -> list[dict]:
    """Would a price-aware rule have done better than the flat one?

    Each candidate is only a re-cut of rows already scored, so this is a
    like-for-like comparison and cheap enough to iterate on. It is a *diagnostic*
    on one dataset, not a validated strategy: anything promising here must be
    re-run through the walk-forward before it means anything.
    """
    band_all = _roi(bets)
    rules = {
        "bet all in band (baseline)": pd.Series(True, index=bets.index),
        "combo_prob > mkt_imp (current)": bets["sel_combo"].astype(bool),
        "model_prob > mkt_imp": bets["sel_model"].astype(bool),
        "current + odds < 12": bets["sel_combo"].astype(bool) & (bets["entry_odds"] < 12),
        "current + odds < 6": bets["sel_combo"].astype(bool) & (bets["entry_odds"] < 6),
        "current + edge > 10%": bets["sel_combo"].astype(bool) & (bets["edge_ratio"] > 1.10),
        "current + edge > 25%": bets["sel_combo"].astype(bool) & (bets["edge_ratio"] > 1.25),
        "odds < 6 + edge > 10%": (bets["sel_combo"].astype(bool)
                                  & (bets["entry_odds"] < 6)
                                  & (bets["edge_ratio"] > 1.10)),
        # The inverse: if the filter is anti-predictive, its complement should
        # beat the band. A sharp result here says the rule has real but
        # inverted information.
        "INVERSE of current": ~bets["sel_combo"].astype(bool),
    }
    out = []
    for name, mask in rules.items():
        chunk = bets[mask]
        stats = _roi(chunk)
        out.append({
            "rule": name, "n": stats["n"], "races": stats["races"],
            "roi": stats["roi"], "ci": [stats["ci_low"], stats["ci_high"]],
            "clv": _clv(chunk),
            "vs_bet_all": (round(stats["roi"] - band_all["roi"], 2)
                           if stats["roi"] is not None else None),
            "beats_bet_all": (stats["roi"] is not None
                              and stats["roi"] > band_all["roi"]),
        })
    return out


def audit(path: str) -> dict:
    bets = load_bets(path)
    selected = bets[bets["sel_combo"].astype(bool)]
    return {
        "source": path,
        "rows_in_band": len(bets),
        "selected": len(selected),
        "folds": sorted(bets["fold"].unique().tolist()),
        "overall": {
            "bet_all": _roi(bets),
            "selected": _roi(selected),
            "per_fold_selected": fold_summary([
                float(g["pnl_bsp"].mean() * 100)
                for _, g in selected.groupby("fold", sort=True)
            ]),
        },
        "by_price_band": by_band(bets),
        "calibration_by_band": calibration_by_band(bets),
        "by_edge_size": by_bucket(bets, "edge_ratio", EDGE_BUCKETS, "edge_ratio"),
        "by_field_size": by_bucket(bets, "num_runners", FIELD_BUCKETS, "field"),
        "candidate_rules": candidate_rules(bets),
    }


def render(report: dict) -> str:
    lines = [
        "",
        "  === Selection-rule audit: where does the value filter lose? ===",
        f"  {report['rows_in_band']:,} runners in band over {len(report['folds'])} folds; "
        f"{report['selected']:,} selected",
        "",
    ]
    bet_all, sel = report["overall"]["bet_all"], report["overall"]["selected"]
    lines += [
        f"  bet-all  ROI {bet_all['roi']:+.2f}%  CI[{bet_all['ci_low']:+.2f}, {bet_all['ci_high']:+.2f}]  n={bet_all['n']:,}",
        f"  selected ROI {sel['roi']:+.2f}%  CI[{sel['ci_low']:+.2f}, {sel['ci_high']:+.2f}]  n={sel['n']:,}",
        f"  the filter is worth {sel['roi'] - bet_all['roi']:+.2f} points vs no filter",
        "",
        "  -- by price band (does the filter chase longshots?) --",
        f"  {'band':>10} {'runners':>8} {'sel':>7} {'sel%':>6} {'stake%':>7} "
        f"{'ROI sel':>9} {'ROI band':>9} {'gap':>7} {'CLV':>7}",
    ]
    for r in report["by_price_band"]:
        lines.append(
            f"  {r['band']:>10} {r['runners']:>8,} {r['selected']:>7,} "
            f"{r['select_rate']*100:>5.1f}% {r['share_of_bets']*100:>6.1f}% "
            f"{r['roi_selected']:>+8.2f}% {r['roi_band']:>+8.2f}% "
            f"{(r['gap_vs_band'] if r['gap_vs_band'] is not None else 0):>+6.2f} "
            f"{(r['clv_selected'] if r['clv_selected'] is not None else float('nan')):>7.4f}"
        )
    lines += ["", "  -- calibration of the selected bets --",
              f"  {'band':>10} {'n':>8} {'predicted':>10} {'actual':>8} {'over':>8} {'mkt':>8}"]
    for r in report["calibration_by_band"]:
        lines.append(
            f"  {r['band']:>10} {r['n']:>8,} {r['predicted']:>10.4f} "
            f"{r['actual']:>8.4f} {r['over_prediction']:>+8.4f} {r['market_implied']:>8.4f}"
        )
    lines += ["", "  -- candidate rules (diagnostic only; re-validate in the WF) --",
              f"  {'rule':<32} {'n':>8} {'ROI':>9} {'vs bet-all':>11} {'CLV':>8}"]
    for r in report["candidate_rules"]:
        if r["roi"] is None:
            continue
        mark = " *" if r["beats_bet_all"] else ""
        lines.append(
            f"  {r['rule']:<32} {r['n']:>8,} {r['roi']:>+8.2f}% "
            f"{r['vs_bet_all']:>+10.2f} {(r['clv'] or float('nan')):>8.4f}{mark}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bets", default=DEFAULT_BETS,
                        help="per-bet parquet from wf_benter_combination --dump")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    if not os.path.isfile(args.bets):
        print(f"No per-bet rows at {args.bets}.\nGenerate them with:\n"
              f"  python scripts/wf_benter_combination.py --max-folds 8 --dump {args.bets}")
        return 2
    report = audit(args.bets)
    print(json.dumps(report, indent=2, default=str) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
