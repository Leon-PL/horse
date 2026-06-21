"""Feature-quality audit over the feature store.

Two reports in one pass (no eyeballing 1000 columns):

1. BURN-IN — per-month warm-up curves (debut share, median prior races,
   Glicko-init share, overall feature coverage) and a recommended
   BURN_IN_MONTHS where the curves plateau.

2. RED FLAGS — one row per feature with coverage, constant/degenerate,
   range-sanity (by name), inf, biggest month-to-month distribution break
   (and its date), and target correlation (leak smell). Sorted worst-first
   to data/processed/feature_audit.csv; the worst rows print to stdout.

Usage:
    python scripts/feature_audit.py                 # full store
    python scripts/feature_audit.py --frac 0.3      # sample races (faster, lighter)
    python scripts/feature_audit.py --top 40        # print N worst features
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import config  # noqa: E402
from src.model import get_feature_columns  # noqa: E402

OUT = os.path.join(config.PROCESSED_DATA_DIR, "feature_audit.csv")


def _load(frac: float) -> pd.DataFrame:
    from src.feature_store import FeatureStore
    store = FeatureStore()
    if not store.exists():
        raise SystemExit("Feature store not built.")
    df = store.load_featured()
    if frac < 1.0 and "race_id" in df.columns:
        rng = np.random.default_rng(0)
        rids = df["race_id"].unique()
        keep = set(rng.choice(rids, size=int(len(rids) * frac), replace=False))
        df = df[df["race_id"].isin(keep)].copy()
    # downcast floats to halve memory for the wide scan
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["_ym"] = df["race_date"].dt.to_period("M")
    return df


# ── burn-in ────────────────────────────────────────────────────────────
def burn_in_report(df: pd.DataFrame, feats: list[str]) -> None:
    print("\n" + "=" * 72)
    print("BURN-IN  (when do form features warm up?)")
    print("=" * 72)
    g = df.groupby("_ym", sort=True)
    rows = pd.DataFrame(index=sorted(df["_ym"].dropna().unique()))
    rows["races"] = g["race_id"].nunique() if "race_id" in df.columns else g.size()
    if "horse_prev_races" in df.columns:
        hp = pd.to_numeric(df["horse_prev_races"], errors="coerce")
        rows["debut%"] = (hp == 0).groupby(df["_ym"]).mean() * 100
        rows["med_prev"] = hp.groupby(df["_ym"]).median()
    if "horse_glicko_rd" in df.columns:
        rows["glicko_init%"] = (pd.to_numeric(df["horse_glicko_rd"], errors="coerce") >= 349.9) \
            .groupby(df["_ym"]).mean() * 100
    # overall coverage = mean non-null fraction across features, per month
    nn = 1.0 - df[feats].isna().mean(axis=1)
    rows["coverage%"] = nn.groupby(df["_ym"]).mean() * 100

    # plateau detection on debut share (falls from ~100% to a floor)
    rec = None
    if "debut%" in rows.columns and len(rows) > 6:
        v = rows["debut%"].to_numpy(dtype=float)
        floor = float(np.nanmean(v[-12:])) if len(v) >= 12 else float(np.nanmin(v))
        thresh = floor + 5.0  # within 5 percentage points of the steady floor
        below = v <= thresh
        idx = next((i for i in range(len(below)) if below[i:].all()), None)
        if idx is not None:
            rec = idx  # months from start

    # compact print: quarterly
    show = rows.iloc[::3].copy()
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(show.round(1).to_string())
    print(f"\ncurrent config.BURN_IN_MONTHS = {getattr(config, 'BURN_IN_MONTHS', '?')}")
    if rec is not None:
        print(f"recommended burn-in ≈ {rec} months "
              f"(debut share reaches its steady floor and stays there)")
    else:
        print("debut share has not clearly plateaued — keep the current burn-in "
              "or extend the dataset.")


# ── per-feature red flags ──────────────────────────────────────────────
def _expected_range(name: str) -> tuple[float, float] | None:
    n = name.lower()
    if any(k in n for k in ("implied_prob", "win_probability", "place_probability")) or \
       n.endswith("_prob") or n.endswith("_rate") or n.endswith("_wr_shrunk") or n.endswith("_pr_shrunk"):
        return (0.0, 1.0)
    if n == "age":
        return (2.0, 15.0)
    if n.endswith("_rank"):
        return (1.0, 40.0)
    return None


def red_flags(df: pd.DataFrame, feats: list[str], top: int) -> None:
    print("\n" + "=" * 72)
    print("RED FLAGS  (per feature; review worst-first)")
    print("=" * 72)
    X = df[feats]
    desc = X.agg(["min", "max", "mean", "std"]).T
    nullp = X.isna().mean()
    # monthly means → biggest month-to-month break, scored as a robust ratio:
    # how anomalous the largest single jump is vs the TYPICAL monthly move.
    # (Normalising by overall std hides step-changes — a step inflates the std.)
    mm = X.groupby(df["_ym"]).mean()
    diffs = mm.diff().abs()
    dmax = diffs.max()
    dwhen = diffs.idxmax()
    typ = diffs.median()                       # typical month-to-month move
    break_score = (dmax / (typ + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # target correlation (leak smell)
    if "won" in df.columns:
        tcorr = X.corrwith(pd.to_numeric(df["won"], errors="coerce")).abs().fillna(0.0)
    else:
        tcorr = pd.Series(0.0, index=feats)

    rep = pd.DataFrame(index=feats)
    rep["null%"] = (nullp * 100).round(2)
    rep["min"] = desc["min"]
    rep["max"] = desc["max"]
    rep["constant"] = (desc["min"] == desc["max"]) | desc["std"].fillna(0).eq(0)
    rep["has_inf"] = np.isinf(desc["min"].to_numpy()) | np.isinf(desc["max"].to_numpy())
    rep["break_x"] = break_score.round(1)      # biggest jump ÷ typical monthly move
    rep["break_at"] = [str(dwhen.get(f, "")) for f in feats]
    rep["tgt_corr"] = tcorr.round(3)

    # range violations by name
    rng_flag = []
    for f in feats:
        er = _expected_range(f)
        bad = False
        if er is not None:
            lo, hi = er
            mn, mx = desc.loc[f, "min"], desc.loc[f, "max"]
            tol = 1e-4
            bad = (pd.notna(mn) and mn < lo - tol) or (pd.notna(mx) and mx > hi + tol)
        rng_flag.append(bad)
    rep["range_bad"] = rng_flag

    # severity score
    sev = (
        rep["constant"].astype(int) * 40
        + (rep["null%"] > 90).astype(int) * 35
        + ((rep["null%"] > 50) & (rep["null%"] <= 90)).astype(int) * 20
        + rep["has_inf"].astype(int) * 30
        + rep["range_bad"].astype(int) * 35
        + (rep["break_x"] > 6).astype(int) * 20
        + (rep["break_x"] > 15).astype(int) * 15
        + (rep["tgt_corr"] > 0.8).astype(int) * 50
        + ((rep["tgt_corr"] > 0.5) & (rep["tgt_corr"] <= 0.8)).astype(int) * 30
    )
    rep["severity"] = sev.astype(int)
    rep = rep.sort_values("severity", ascending=False)
    rep.to_csv(OUT)

    n_const = int(rep["constant"].sum())
    n_range = int(rep["range_bad"].sum())
    n_hinull = int((rep["null%"] > 90).sum())
    n_inf = int(rep["has_inf"].sum())
    n_leak = int((rep["tgt_corr"] > 0.5).sum())
    print(f"features audited: {len(feats)}")
    print(f"  constant/degenerate : {n_const}")
    print(f"  >90% null           : {n_hinull}")
    print(f"  range violations     : {n_range}")
    print(f"  contain inf          : {n_inf}")
    print(f"  high target-corr(>.5): {n_leak}  (possible current-race leak — check these)")
    print(f"\nfull report → {OUT}")
    print(f"\nTop {top} by severity:")
    cols = ["severity", "null%", "constant", "range_bad", "has_inf", "break_x", "break_at", "tgt_corr", "min", "max"]
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_colwidth", 28):
        print(rep[cols].head(top).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    config.FEATURE_ALLOWLIST = None          # audit ALL features, not a cut set
    df = _load(args.frac)
    feats = get_feature_columns(df)
    print(f"store: {len(df):,} rows × {len(feats)} features "
          f"({df['race_date'].min().date()} → {df['race_date'].max().date()})")
    burn_in_report(df, feats)
    red_flags(df, feats, args.top)


if __name__ == "__main__":
    main()
