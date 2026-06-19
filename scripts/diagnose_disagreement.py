"""Model-vs-market disagreement diagnostic.

For a saved run, joins the holdout predictions (model_prob, odds) to the
outcomes + race context, derives the overround-normalised market
probability, and dissects WHERE and WHETHER the (form) model's
disagreements with the market are right.

Usage:
    python scripts/diagnose_disagreement.py [run_id] [--top N] [--csv]

Defaults to the most recent run under data/runs/. The holdout predictions
are out-of-sample (the test tail), so this is an honest read on where the
model goes wrong.

Reads only the columns it needs from the (large) featured parquet.
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

RUNS_DIR = os.path.join(config.DATA_DIR, "runs")

# Context columns pulled from the featured parquet for segmentation.
_CONTEXT_COLS = [
    "race_id", "horse_name", "won", "finish_position", "num_runners",
    "handicap", "race_type", "going", "race_class", "distance_furlongs",
]


def _latest_run() -> str:
    runs = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]
    if not runs:
        raise SystemExit("No runs found under data/runs/")
    return sorted(runs)[-1]


def _market_prob(df: pd.DataFrame) -> np.ndarray:
    """Overround-normalised implied win prob per race (matches model.py)."""
    odds = pd.to_numeric(df["odds"], errors="coerce")
    ip = (1.0 / odds.where(odds > 1.0)).fillna(0.0).clip(0.0, 1.0).to_numpy(float)
    over = df.assign(_ip=ip).groupby("race_id")["_ip"].transform("sum").to_numpy(float)
    return ip / np.maximum(over, 1e-9)


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def load_run(run_id: str) -> pd.DataFrame:
    run_dir = os.path.join(RUNS_DIR, run_id)
    preds = pd.read_csv(os.path.join(run_dir, "predictions.csv"))
    preds["race_id"] = preds["race_id"].astype(str)

    feat_path = os.path.join(run_dir, "featured_races.parquet")
    have = [c for c in _CONTEXT_COLS if c]
    # Read only needed columns; tolerate any that are absent.
    import pyarrow.parquet as pq
    avail = set(pq.ParquetFile(feat_path).schema.names)
    cols = [c for c in have if c in avail]
    feat = pd.read_parquet(feat_path, columns=cols, engine="pyarrow")
    feat["race_id"] = feat["race_id"].astype(str)

    df = preds.merge(feat, on=["race_id", "horse_name"], how="inner")
    if "won" not in df.columns and "finish_position" in df.columns:
        df["won"] = (pd.to_numeric(df["finish_position"], errors="coerce") == 1).astype(int)
    df = df[pd.to_numeric(df["won"], errors="coerce").notna()].copy()
    df["won"] = df["won"].astype(int)

    df["market_prob"] = _market_prob(df)
    df = df[df["market_prob"] > 0].copy()          # need a usable market price
    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    df = df[df["model_prob"].notna()].copy()
    df["disagree"] = df["model_prob"] - df["market_prob"]
    df["log_ratio"] = np.log(np.clip(df["model_prob"], 1e-9, None) /
                             np.clip(df["market_prob"], 1e-9, None))
    return df


def _bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    nr = pd.to_numeric(out.get("num_runners"), errors="coerce")
    out["field_band"] = pd.cut(nr, [0, 7, 11, 15, 99], labels=["1-7", "8-11", "12-15", "16+"]).astype(str)
    dist = pd.to_numeric(out.get("distance_furlongs"), errors="coerce")
    out["dist_band"] = pd.cut(dist, [0, 6.5, 8.5, 11.5, 99], labels=["Sprint", "Mile", "Middle", "Staying"]).astype(str)
    out["hcap"] = np.where(pd.to_numeric(out.get("handicap"), errors="coerce").fillna(0) > 0, "Handicap", "Non-Hcap")
    out["odds_band"] = pd.cut(
        pd.to_numeric(out["odds"], errors="coerce"),
        [0, 3, 6, 11, 21, 1e9], labels=["<3", "3-6", "6-11", "11-21", "21+"],
    ).astype(str)
    return out


def report(df: pd.DataFrame, top: int, save_csv: bool, run_id: str) -> None:
    y = df["won"].to_numpy()
    mdl, mkt = df["model_prob"].to_numpy(), df["market_prob"].to_numpy()
    n_races = df["race_id"].nunique()

    print(f"\n{'='*72}\nDISAGREEMENT DIAGNOSTIC — run {run_id}")
    print(f"{len(df):,} holdout runners · {n_races:,} races · base win rate {y.mean():.3f}")
    print('='*72)

    print("\n## 1. Headline: form model vs market (out-of-sample)")
    print(f"  {'':14}{'log_loss':>10}{'brier':>10}")
    print(f"  {'model':14}{_logloss(mdl,y):>10.4f}{_brier(mdl,y):>10.4f}")
    print(f"  {'market':14}{_logloss(mkt,y):>10.4f}{_brier(mkt,y):>10.4f}")
    _d = _logloss(mdl, y) - _logloss(mkt, y)
    print(f"  → model {'BEATS' if _d < 0 else 'LOSES TO'} market by {abs(_d):.4f} log-loss")
    print(f"  corr(model, market) = {np.corrcoef(mdl, mkt)[0,1]:.4f}  "
          f"(low ⇒ form model is saying something different)")

    print("\n## 2. Where they disagree — deciles of (model_prob − market_prob)")
    print("    Negative bucket = model rates LOWER than market; positive = HIGHER.")
    print("    'right' if actual win rate tracks the model, not the market.")
    q = pd.qcut(df["disagree"], 10, duplicates="drop")
    g = df.groupby(q, observed=True)
    tbl = pd.DataFrame({
        "n": g.size(),
        "mean_disagree": g["disagree"].mean(),
        "model_p": g["model_prob"].mean(),
        "market_p": g["market_prob"].mean(),
        "actual": g["won"].mean(),
    })
    tbl["model_err"] = (tbl["model_p"] - tbl["actual"]).abs()
    tbl["market_err"] = (tbl["market_p"] - tbl["actual"]).abs()
    tbl["model_wins?"] = np.where(tbl["model_err"] < tbl["market_err"], "model", "market")
    print(tbl.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n## 3. Directional edge — does backing model-OVERS / model-UNDERS pay?")
    print("    Compares realised win rate to model & market probs by edge size.")
    for label, mask in [
        ("model >> market (model likes)", df["disagree"] > 0),
        ("model << market (model dislikes)", df["disagree"] < 0),
    ]:
        sub = df[mask]
        edge_band = pd.cut(sub["disagree"].abs(), [0, 0.02, 0.05, 0.1, 1.0],
                           labels=["0-2%", "2-5%", "5-10%", "10%+"])
        gg = sub.groupby(edge_band, observed=True)
        t = pd.DataFrame({
            "n": gg.size(), "model_p": gg["model_prob"].mean(),
            "market_p": gg["market_prob"].mean(), "actual": gg["won"].mean(),
        })
        print(f"\n  {label}:")
        print(t.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n## 4. Segments where the model trails the market most (brier gap)")
    print("    model_brier − market_brier; positive = model worse. Min 200 runners.")
    dfb = _bands(df)
    rows = []
    for col in ["field_band", "dist_band", "hcap", "odds_band", "race_type", "going", "race_class"]:
        if col not in dfb.columns:
            continue
        for val, sub in dfb.groupby(col, observed=True):
            if len(sub) < 200 or str(val) in ("nan", "Unknown"):
                continue
            yy = sub["won"].to_numpy()
            rows.append({
                "segment": f"{col}={val}", "n": len(sub),
                "model_brier": _brier(sub["model_prob"].to_numpy(), yy),
                "market_brier": _brier(sub["market_prob"].to_numpy(), yy),
            })
    seg = pd.DataFrame(rows)
    if not seg.empty:
        seg["gap"] = seg["model_brier"] - seg["market_brier"]
        seg = seg.sort_values("gap", ascending=False)
        print(seg.head(15).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(f"\n## 5. Worst individual disagreements (|model−market|), top {top}")
    print("    A) model confidently HIGHER than market, but LOST:")
    a = df[(df["disagree"] > 0) & (df["won"] == 0)].nlargest(top, "disagree")
    print(a[["race_id", "horse_name", "odds", "model_prob", "market_prob", "finish_position"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n    B) model confidently LOWER than market, but WON (missed winners):")
    b = df[(df["disagree"] < 0) & (df["won"] == 1)].nsmallest(top, "disagree")
    print(b[["race_id", "horse_name", "odds", "model_prob", "market_prob", "finish_position"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n## 6. Top-pick disagreement — model's top pick vs market favourite")
    idx_m = df.groupby("race_id")["model_prob"].idxmax()
    idx_k = df.groupby("race_id")["market_prob"].idxmax()
    mp = df.loc[idx_m, ["race_id", "horse_name", "won"]].set_index("race_id")
    kp = df.loc[idx_k, ["race_id", "horse_name", "won"]].set_index("race_id")
    joined = mp.join(kp, lsuffix="_model", rsuffix="_mkt")
    disagree_races = joined[joined["horse_name_model"] != joined["horse_name_mkt"]]
    agree_races = joined[joined["horse_name_model"] == joined["horse_name_mkt"]]
    print(f"  model top pick == market fav in {len(agree_races):,}/{len(joined):,} races "
          f"({len(agree_races)/max(len(joined),1):.1%})")
    if len(disagree_races):
        print(f"  When they DISAGREE on the favourite ({len(disagree_races):,} races):")
        print(f"     model's pick wins : {disagree_races['won_model'].mean():.3f}")
        print(f"     market's fav wins : {disagree_races['won_mkt'].mean():.3f}")
    print(f"  When they AGREE, that pick wins: {agree_races['won_model'].mean():.3f}")

    if save_csv:
        out_path = os.path.join(RUNS_DIR, run_id, "disagreement.csv")
        keep = ["race_id", "horse_name", "race_date", "odds", "model_prob",
                "market_prob", "disagree", "log_ratio", "won", "finish_position"]
        df[[c for c in keep if c in df.columns]].sort_values(
            "disagree", key=lambda s: s.abs(), ascending=False
        ).to_csv(out_path, index=False)
        print(f"\nSaved per-runner disagreement table → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?", default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--csv", action="store_true", help="save per-runner disagreement.csv")
    args = ap.parse_args()
    run_id = args.run_id or _latest_run()
    df = load_run(run_id)
    report(df, top=args.top, save_csv=args.csv, run_id=run_id)


if __name__ == "__main__":
    main()
