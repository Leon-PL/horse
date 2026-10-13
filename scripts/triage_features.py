"""Stage-1 triage: does a candidate feature carry information the market lacks?

The bar in this project is not "is it predictive?" but "is it **orthogonal to
what the market already prices?**" — because the Betfair price is sharp and
seven consecutive data acquisitions have died on redundancy
(``docs/NEGATIVE_RESULTS.md``). This script is the cheap first gate: it scores
candidates against the **market residual** (``won - mkt_imp``) so that a CLV
walk-forward, which is expensive, only ever runs on a shortlist.

Three refinements over a naive correlation, each of which changes the answer:

* **Partial correlation, controlling for the market probability.** Almost any
  ability feature correlates with price, and the residual has real structure
  against price (favourite-longshot bias). A feature can therefore show a
  non-zero raw correlation with the residual purely by proxying the price
  level. When the raw correlation is non-zero and the partial correlation is
  not, the feature is a price proxy and nothing more.

* **Within-race centring.** Racing outcomes are competitive: only the
  *relative* standing of a runner in its own field can pay. A feature that is
  high for every runner in a strong race says nothing. Both the feature and
  the residual are demeaned within race before scoring.

* **Effect size first, then significance.** With 100k+ rows a correlation of
  0.006 is "statistically significant" and economically worthless, so the
  ranking is by magnitude. But an effect-size gate alone is not enough either,
  because the noise floor moves with sample size: at n=12,000 pure noise
  clears 0.01 routinely. A candidate must therefore be **both** large enough
  to matter and distinguishable from noise after a Benjamini-Hochberg
  correction across every column in the run (216 at once for ATR).

Coverage matters too. ATR sectionals are GB-only and start partway through the
history, so the scoring is restricted to rows that actually have the data —
otherwise "has a value here" proxies for region and era, which correlate with
everything.

Usage::

    python scripts/triage_features.py --prefix atr_
    python scripts/triage_features.py --prefix atr_ --json
    python scripts/triage_features.py --prefix pace_ --min-coverage 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from src.betfair import attach_bsp
from src.wf_eval import development_mask
from src.feature_store import FeatureStore

logger = logging.getLogger("triage")

DIAG = ["race_id", "race_date", "won", "track", "off_time", "horse_name",
        "num_runners", "race_type"]

#: A partial correlation with the market residual below this is treated as
#: "the market already prices it". Calibrated against the recorded dead
#: results: sale price scored 0.0000 overall and 0.005 on debutants, and wind
#: surgery had a residual t of -0.33. Anything under 0.01 has never survived.
SHORTLIST_ABS_CORR = 0.01

#: ...and the effect must also be distinguishable from noise. An effect-size
#: gate alone is not enough, because the noise floor depends on sample size:
#: the standard error of a correlation is ~1/sqrt(n), so at n=12,000 pure
#: noise clears 0.01 routinely (1/sqrt(12000) = 0.009) while at n=200,000 it
#: cannot (0.002). Requiring both keeps the gate honest at either scale —
#: economic relevance from the effect size, statistical distinguishability
#: from the Benjamini-Hochberg q-value across every column in the run.
SHORTLIST_MAX_Q = 0.05


def _family(name: str) -> str:
    """Collapse a feature name to its underlying signal.

    216 ATR columns are ~21 signals in different windows and framings
    (avg_3/avg_5/best_3/trend, each also z/rank/vs-best within race). Ranking
    individual columns would bury the structure.
    """
    base = re.sub(r"_(z_in_race|rank_pct_in_race|vs_best_in_race|vs_field)$", "", name)
    base = re.sub(r"_(avg|best)_\d+$", "", base)
    base = re.sub(r"_rate_\d+$", "", base)
    base = re.sub(r"_trend$", "", base)
    return base


def _within_race_demean(values: np.ndarray, race_codes: np.ndarray) -> np.ndarray:
    """Subtract each race's mean, ignoring NaN."""
    series = pd.Series(values)
    return (series - series.groupby(race_codes).transform("mean")).to_numpy()


def market_controls(market_prob: np.ndarray, race_codes: np.ndarray) -> np.ndarray:
    """Design matrix for "everything the price already says".

    Controlling for the market probability **linearly** is not enough. Ability
    features are roughly linear in *log-odds*, not in probability, so a
    feature that merely restates ``log(price)`` keeps a sizeable partial
    correlation after a linear control and gets shortlisted as a discovery.
    (A test does exactly this: a pure ``log(mkt)`` proxy survives a linear
    control at 0.039, four times the shortlist threshold.)

    So the control is a small basis — probability, log-odds and a quadratic
    term — each demeaned within race, which absorbs monotone re-expressions of
    the price as well as mild curvature.
    """
    m = np.clip(np.asarray(market_prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    columns = [m, np.log(m / (1.0 - m)), m ** 2]
    return np.column_stack([_within_race_demean(c, race_codes) for c in columns])


def _residualise(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Least-squares residual of *values* on *controls* (with an intercept)."""
    design = np.column_stack([np.ones(len(values)), controls])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float, int]:
    """Correlation of *x* and *y* with the controls *z* partialled out.

    *z* may be a single column or a design matrix from :func:`market_controls`.
    Returns ``(raw_corr_xy, partial_corr, n)`` over rows finite throughout.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z).all(axis=1)
    n = int(ok.sum())
    if n < 100:
        return float("nan"), float("nan"), n
    xs, ys, zs = x[ok], y[ok], z[ok]
    if xs.std() == 0 or ys.std() == 0:
        return float("nan"), float("nan"), n

    r_xy = float(np.corrcoef(xs, ys)[0, 1])
    if not np.any(zs.std(axis=0) > 0):
        return r_xy, r_xy, n
    xr, yr = _residualise(xs, zs), _residualise(ys, zs)
    if xr.std() == 0 or yr.std() == 0:
        return r_xy, float("nan"), n
    return r_xy, float(np.corrcoef(xr, yr)[0, 1]), n


def _t_stat(corr: float, n: int) -> float:
    if not np.isfinite(corr) or n < 4 or abs(corr) >= 1:
        return float("nan")
    return float(corr * np.sqrt((n - 2) / max(1e-12, 1 - corr ** 2)))


def _bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaNs preserved)."""
    out = np.full(len(p_values), np.nan)
    finite = np.isfinite(p_values)
    if not finite.any():
        return out
    p = p_values[finite]
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(len(p))
    adjusted[order] = np.clip(ranked, 0, 1)
    out[finite] = adjusted
    return out


def _two_sided_p(t: np.ndarray) -> np.ndarray:
    """Normal approximation; n is large enough here that it is exact enough."""
    from math import erfc, sqrt
    return np.array([
        erfc(abs(v) / sqrt(2)) if np.isfinite(v) else np.nan for v in t
    ])


def load_candidates(prefix: str) -> tuple[pd.DataFrame, list[str]]:
    """Store rows with the candidate columns, the market price, and the label."""
    store = FeatureStore()
    if not store.exists():
        raise SystemExit("No feature store — build it on the training app's Data page.")
    available = pq.read_schema(store.featured_path).names
    candidates = sorted(c for c in available if c.startswith(prefix))
    if not candidates:
        raise SystemExit(f"No columns start with {prefix!r}.")

    coverage_flags = [c for c in available if c.startswith("has_") and prefix[:3] in c]
    cols = [c for c in dict.fromkeys(DIAG + coverage_flags + candidates) if c in available]
    df = pd.read_parquet(store.featured_path, columns=cols)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(np.int8)
    # Never screen on the golden hold-out. Today it has no T-5 price anyway,
    # but that protection would vanish the day its BASIC data is downloaded.
    df = df[development_mask(df["race_date"])].copy()

    df = attach_bsp(df, cols=("bsp", "ltp_300s"))
    entry = pd.to_numeric(df["ltp_300s"], errors="coerce")
    df = df[entry > 1].copy()
    implied = 1.0 / pd.to_numeric(df["ltp_300s"], errors="coerce")
    df["mkt_imp"] = (implied / implied.groupby(df["race_id"]).transform("sum")).astype(np.float64)
    return df.reset_index(drop=True), candidates


def triage(prefix: str, *, min_coverage: float = 0.05) -> dict:
    df, candidates = load_candidates(prefix)

    # Restrict to rows that actually carry the data, so "present" cannot
    # proxy for region or era.
    flag = next((c for c in df.columns if c.startswith("has_") and prefix[:3] in c), None)
    if flag is not None:
        covered = pd.to_numeric(df[flag], errors="coerce").fillna(0) > 0
        scope = df[covered].copy()
    else:
        scope = df
    if len(scope) < 1000:
        raise SystemExit(f"Only {len(scope)} rows carry {prefix} data; nothing to triage.")

    race_codes = pd.factorize(scope["race_id"])[0]
    residual = _within_race_demean(
        (scope["won"].to_numpy(dtype=np.float64) - scope["mkt_imp"].to_numpy()), race_codes,
    )
    market = market_controls(scope["mkt_imp"].to_numpy(), race_codes)

    rows = []
    for col in candidates:
        values = pd.to_numeric(scope[col], errors="coerce").to_numpy(dtype=np.float64)
        coverage = float(np.isfinite(values).mean())
        if coverage < min_coverage:
            rows.append({"feature": col, "family": _family(col), "coverage": round(coverage, 4),
                         "raw_corr": None, "partial_corr": None, "t": None, "n": 0,
                         "verdict": "too sparse"})
            continue
        centred = _within_race_demean(values, race_codes)
        raw, partial, n = _partial_corr(centred, residual, market)
        rows.append({
            "feature": col, "family": _family(col), "coverage": round(coverage, 4),
            "raw_corr": None if not np.isfinite(raw) else round(raw, 5),
            "partial_corr": None if not np.isfinite(partial) else round(partial, 5),
            "t": None if not np.isfinite(partial) else round(_t_stat(partial, n), 2),
            "n": n, "verdict": None,
        })

    # Multiple testing across every column scored in this run.
    t_values = np.array([r["t"] if r["t"] is not None else np.nan for r in rows])
    q_values = _bh_qvalues(_two_sided_p(t_values))
    for row, q in zip(rows, q_values):
        row["q"] = None if not np.isfinite(q) else round(float(q), 5)
        if row["verdict"] == "too sparse":
            continue
        pc = row["partial_corr"]
        if pc is None:
            row["verdict"] = "undefined"
        elif abs(pc) < SHORTLIST_ABS_CORR:
            row["verdict"] = "priced"
        elif row["q"] is None or row["q"] > SHORTLIST_MAX_Q:
            # Big enough to matter, not big enough to believe.
            row["verdict"] = "noise"
        else:
            row["verdict"] = "shortlist"

    # Family view: the strongest column per underlying signal.
    families: dict[str, dict] = {}
    for row in rows:
        if row["partial_corr"] is None:
            continue
        best = families.get(row["family"])
        if best is None or abs(row["partial_corr"]) > abs(best["partial_corr"]):
            families[row["family"]] = row
    family_rows = sorted(families.values(), key=lambda r: -abs(r["partial_corr"]))

    shortlist = [r for r in family_rows if r["verdict"] == "shortlist"]
    return {
        "prefix": prefix,
        "columns_scored": sum(1 for r in rows if r["partial_corr"] is not None),
        "columns_total": len(candidates),
        "rows_with_data": len(scope),
        "rows_in_store": len(df),
        "coverage_flag": flag,
        "date_range": [str(scope["race_date"].min().date()), str(scope["race_date"].max().date())],
        "shortlist_threshold": SHORTLIST_ABS_CORR,
        "shortlist_max_q": SHORTLIST_MAX_Q,
        "families": family_rows,
        "shortlist": shortlist,
        "all_columns": sorted(rows, key=lambda r: -abs(r["partial_corr"] or 0)),
    }


def render(report: dict) -> str:
    lines = [
        "",
        f"  === Orthogonality triage: {report['prefix']}* ===",
        f"  {report['columns_scored']}/{report['columns_total']} columns scored on "
        f"{report['rows_with_data']:,} rows that carry the data "
        f"(of {report['rows_in_store']:,} priced rows)",
        f"  coverage flag: {report['coverage_flag']}   "
        f"dates {report['date_range'][0]} .. {report['date_range'][1]}",
        "",
        "  Scored against the market residual (won - mkt_imp), both demeaned",
        "  within race, with the market probability partialled out.",
        "",
        f"  {'signal (best column of family)':<44} {'cover':>6} {'raw':>8} {'partial':>8} {'t':>7} {'verdict':>10}",
    ]
    for r in report["families"]:
        lines.append(
            f"  {r['family'][:44]:<44} {r['coverage']*100:>5.0f}% "
            f"{(r['raw_corr'] if r['raw_corr'] is not None else float('nan')):>+8.4f} "
            f"{(r['partial_corr'] if r['partial_corr'] is not None else float('nan')):>+8.4f} "
            f"{(r['t'] if r['t'] is not None else float('nan')):>+7.1f} {r['verdict']:>10}"
        )
    lines += ["", f"  Shortlist (|partial| >= {report['shortlist_threshold']} "
                  f"AND q <= {report['shortlist_max_q']}): "
                  f"{len(report['shortlist'])} of {len(report['families'])} signals"]
    if report["shortlist"]:
        for r in report["shortlist"]:
            lines.append(f"    {r['feature']}  partial={r['partial_corr']:+.4f}  t={r['t']:+.1f}")
        lines += [
            "",
            "  Next step for the shortlist ONLY: CLV at BSP under the locked",
            "  walk-forward, with the ROI-vs-CLV tripwire. Correlation here is",
            "  necessary, not sufficient.",
        ]
    else:
        lines += [
            "",
            "  Nothing clears the threshold: the market already prices this family.",
            "  Record it in docs/NEGATIVE_RESULTS.md and keep the data.",
        ]
    lines += [
        "",
        "  NOTE: with this many rows, significance is not the bar - a correlation",
        "  of 0.006 is 'significant' and worthless. Judge on magnitude.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", default="atr_", help="feature-name prefix to triage")
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=None, help="also write the JSON report here")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    report = triage(args.prefix, min_coverage=args.min_coverage)
    print(json.dumps(report, indent=2, default=str) if args.json else render(report))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"  report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
