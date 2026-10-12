"""Where does the value filter lose money?

The market-combination walk-forward established something specific: the
combination **beats the model alone** (-3.55% vs -4.49% ROI @ BSP) yet still
**loses to backing everything in the band** (-2.68%). That is a statement about
the *selection rule*, not the blend — picking runners where
``combo_prob > mkt_imp`` chooses bets that are worse than average inside the
very band they are drawn from. A filter that is worse than no filter is doing
real work in the wrong direction, and nothing in this project has ever looked
at where.

The hypothesis going in was that the value filter is a **longshot filter in
disguise** — a model is least calibrated at long prices, so that is where it
most often disagrees with the market, and long prices pay worst. **The data
did not support that.** On 196,664 runners over eight folds the selection rate
barely rises with price (45% to 55%), and at the *longest* prices (25-50) the
filter actually beats the band by +1.49 points. The loss is concentrated in
the **middle**: -3.54 points at 12-25 and -1.21 at 6-12, together 59% of
stakes. The model's calibration is in fact excellent everywhere (predicted
within 0.0023 of actual in every band); what distinguishes the losing bands is
that the *market* is closer to the outcome than the model is.

This script produced that finding and remains the tool for re-cutting it, on
the per-bet rows dumped by the walk-forward:

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

from src.wf_eval import bootstrap_roi_ci, clv_summary, fold_summary

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


def _clv(frame: pd.DataFrame) -> dict | None:
    """CLV on all measures. The mean alone is upward-biased at long odds."""
    if frame.empty:
        return None
    out = clv_summary(frame["entry_odds"], frame["close_odds"])
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def _clv_median(frame: pd.DataFrame) -> float | None:
    summary = _clv(frame)
    return summary.get("median_clv") if summary else None


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
            "clv_median_selected": _clv_median(selected),
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
    """Does the filter add value *within the universe it bets in*?

    Each candidate declares two masks: the bets it strikes, and the **universe**
    those bets are drawn from. The comparison that matters is against backing
    everything in that same universe — not against backing everything in the
    whole band.

    This distinction is not pedantic; getting it wrong inverts the conclusion.
    "current + odds < 12" scores -1.29% against a whole-band baseline of
    -2.68%, which looks like the filter gaining 1.39 points. But backing
    *everything* at odds < 12 scores -1.07%, so inside its own universe the
    filter is 0.22 points **worse** than no filter at all. Essentially the
    entire apparent gain was avoiding long prices, which is a statement about
    the price band, not about the rule.

    Diagnostic only: these are re-cuts of one dataset, and anything promising
    has to go back through the walk-forward before it means anything.
    """
    everything = pd.Series(True, index=bets.index)
    selected = bets["sel_combo"].astype(bool)
    short_12 = bets["entry_odds"] < 12
    short_6 = bets["entry_odds"] < 6
    edge_10 = bets["edge_ratio"] > 1.10
    edge_25 = bets["edge_ratio"] > 1.25

    # (name, bets struck, universe drawn from)
    rules = [
        ("bet all in band (baseline)", everything, everything),
        ("combo_prob > mkt_imp (current)", selected, everything),
        ("model_prob > mkt_imp", bets["sel_model"].astype(bool), everything),
        ("current, odds < 12", selected & short_12, short_12),
        ("current, odds < 6", selected & short_6, short_6),
        ("current, edge > 10%", selected & edge_10, everything),
        ("current, edge > 25%", selected & edge_25, everything),
        ("current, odds<6 & edge>10%", selected & short_6 & edge_10, short_6 & edge_10),
        # If the filter were anti-predictive its complement would beat the
        # band. Note this is arithmetically forced once the filter trails
        # bet-all, so it is a consistency check rather than a finding.
        ("INVERSE of current", ~selected, everything),
    ]
    out = []
    whole_band = _roi(bets)
    for name, mask, universe in rules:
        chunk = bets[mask]
        stats = _roi(chunk)
        matched = _roi(bets[universe])
        out.append({
            "rule": name,
            "n": stats["n"], "races": stats["races"], "roi": stats["roi"],
            "ci": [stats["ci_low"], stats["ci_high"]],
            "clv": _clv(chunk),
            "universe_n": matched["n"],
            "roi_matched_baseline": matched["roi"],
            "vs_matched_baseline": (round(stats["roi"] - matched["roi"], 2)
                                    if None not in (stats["roi"], matched["roi"]) else None),
            "vs_whole_band": (round(stats["roi"] - whole_band["roi"], 2)
                              if stats["roi"] is not None else None),
            "beats_matched_baseline": (None not in (stats["roi"], matched["roi"])
                                       and stats["roi"] > matched["roi"]),
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
        "  -- by price band: filter vs backing the whole band --",
        f"  {'band':>10} {'runners':>8} {'sel%':>6} {'stake%':>7} "
        f"{'ROI sel':>9} {'ROI band':>9} {'gap':>7} {'CLVmed':>8}",
    ]
    for r in report["by_price_band"]:
        clv = r["clv_median_selected"]
        lines.append(
            f"  {r['band']:>10} {r['runners']:>8,} "
            f"{r['select_rate']*100:>5.1f}% {r['share_of_bets']*100:>6.1f}% "
            f"{r['roi_selected']:>+8.2f}% {r['roi_band']:>+8.2f}% "
            f"{(r['gap_vs_band'] if r['gap_vs_band'] is not None else 0):>+6.2f} "
            f"{(clv if clv is not None else float('nan')):>8.4f}"
        )
    lines += ["", "  -- calibration of the selected bets --",
              f"  {'band':>10} {'n':>8} {'predicted':>10} {'actual':>8} {'over':>8} {'mkt':>8}"]
    for r in report["calibration_by_band"]:
        lines.append(
            f"  {r['band']:>10} {r['n']:>8,} {r['predicted']:>10.4f} "
            f"{r['actual']:>8.4f} {r['over_prediction']:>+8.4f} {r['market_implied']:>8.4f}"
        )
    lines += [
        "",
        "  -- candidate rules (diagnostic only; re-validate in the WF) --",
        "  'matched' = backing EVERYTHING in the same universe the rule bets in.",
        "  Only the matched column says whether the filter itself adds value.",
        f"  {'rule':<30} {'n':>8} {'ROI':>9} {'matched':>9} {'vs match':>9} {'CLVmed':>8}",
    ]
    for r in report["candidate_rules"]:
        if r["roi"] is None:
            continue
        mark = " *" if r["beats_matched_baseline"] else ""
        clv = (r["clv"] or {}).get("median_clv", float("nan"))
        lines.append(
            f"  {r['rule']:<30} {r['n']:>8,} {r['roi']:>+8.2f}% "
            f"{r['roi_matched_baseline']:>+8.2f}% {r['vs_matched_baseline']:>+8.2f} "
            f"{clv:>8.4f}{mark}"
        )
    lines += [
        "",
        "  CLV shown as the MEDIAN ratio. The mean of entry/close is biased high",
        "  at long prices (upside unbounded, downside not) - see src/clv.py.",
        "",
    ]
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
