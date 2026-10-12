"""Walk-forward re-validation of the Benter market-combination edge.

Does combining a clean fundamental model with the market beat the market,
after costs, out of sample?

**Rewritten 2026-10-12.** The previous version carried three real
measurement defects. Fixing them did **not** change the verdict: eight purged
quarterly folds give combination -3.55% CI[-5.42,-1.55] @ BSP, model-alone
-4.49%, bet-all -2.68%, mean CLV 1.0076, 1/8 folds positive — the same
pattern the original reported (-2.64% / -3.65% / -1.83%). The market
combination is not a bankable edge. The defects were:

1. **The blend was fitted on in-sample model predictions.** The win model was
   trained on the fold's training rows, then the market blend was fitted on
   that same model's predictions *over those same rows*. In-sample
   probabilities are far sharper than out-of-sample ones, so the blend learned
   to over-weight the model and under-weight the market — then met a much
   weaker signal on the test quarter. Fixed with
   ``wf_eval.oof_predict_expanding``.
2. **It used a binary logistic regression on (log p_model, log p_market)**,
   which is not the per-race conditional logit the Benter combination is, and
   not what ``src/model.py`` uses in production. Now calls the production
   ``fit_market_anchor`` / ``apply_market_anchor``, so research and production
   share one estimator (hard rule #2).
3. **No purge gap**, while the production training path purges
   ``config.PURGE_DAYS``. Now uses the shared purged folds.

Two further defects: the model-alone and bet-all baselines were only
accumulated on folds where the combination happened to find a bet (so the
comparison ran over different sets of quarters), and the win model was given
the *place* model's hyperparameters.

What the corrected run does establish, and what to do with it:

* **alpha is positive in 8/8 folds** (mean +0.046, beta ~0.97). The
  fundamental model holds genuine information the price lacks — roughly 20x
  less than the market, far too little to clear commission on its own.
* **The weak link is the selection rule, not the blend.** The combination
  beats the model alone in every measurement yet still loses to backing
  everything in the band, so ``combo_prob > mkt_imp`` is selecting bets that
  are *worse than average* inside the band. See
  ``scripts/audit_selection_rule.py`` (feed it ``--dump``).

A nesting tripwire (``check_combination_nesting``) now fails the run if the
anchored probabilities score worse on race log loss than the market they
nest, since that would mean a broken fit rather than an absent signal. Note
this is a probability-scoring check: losing to "bet everything in the band"
is a staking comparison and says nothing about the fit.

Usage::

    .venv/Scripts/python.exe scripts/wf_benter_combination.py
    .venv/Scripts/python.exe scripts/wf_benter_combination.py --max-folds 3
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from src.bet_settlement import settle_win_exchange_bets
from src.betfair import attach_bsp
from src.feature_store import FeatureStore
from src.model import (
    apply_market_anchor,
    fit_market_anchor,
    get_feature_columns,
    market_probs_from_odds,
)
from src.wf_eval import (
    bootstrap_roi_ci,
    check_combination_nesting,
    clv_summary,
    expanding_period_folds,
    fold_summary,
    oof_predict_expanding,
    race_logloss,
)

logger = logging.getLogger("wf_benter")

DIAG = ["race_id", "race_date", "won", "track", "off_time", "horse_name",
        "num_runners", "race_type"]

#: Columns written by --dump, for scripts/audit_selection_rule.py. Keeping the
#: expensive fold refits separate from the analysis means a selection rule can
#: be re-cut in seconds instead of re-running the walk-forward.
DUMP_COLS = ["race_id", "race_date", "track", "horse_name", "num_runners",
             "race_type", "won", "entry_odds", "close_odds", "mkt_imp",
             "model_prob", "combo_prob", "pnl_bsp", "in_band",
             "sel_combo", "sel_model", "fold"]

#: Win-model hyperparameters. The previous version used the PLACE model's.
WIN_PARAMS = dict(
    n_estimators=600, num_leaves=35, max_depth=6, learning_rate=0.03,
    min_child_samples=50, subsample=0.6, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=3.0, objective="binary",
    subsample_freq=1, n_jobs=-1, verbose=-1, random_state=config.RANDOM_SEED,
)

#: Only strike inside a tradeable price band.
BAND = (1.5, 50.0)


def _race_blocks(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Sort into contiguous race blocks and return the group sizes.

    ``apply_market_anchor`` / ``fit_market_anchor`` reduce over contiguous
    per-race slices (``np.add.reduceat``), so rows must be grouped first or
    probabilities get normalised across the wrong runners.
    """
    ordered = frame.sort_values("race_id", kind="stable")
    groups = ordered.groupby("race_id", sort=False).size().to_numpy().astype(np.int64)
    return ordered, groups


def _normalise_per_race(values, race_ids) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64)).reset_index(drop=True)
    race = pd.Series(np.asarray(race_ids)).reset_index(drop=True)
    totals = series.groupby(race).transform("sum").replace(0, np.nan)
    return (series / totals).to_numpy()


def _anchor(model_probs, odds, frame: pd.DataFrame, anchor: dict) -> np.ndarray:
    """Apply a fitted (alpha, beta), returning values in *frame* row order."""
    work = frame.assign(_p=np.asarray(model_probs), _o=np.asarray(odds))
    ordered, groups = _race_blocks(work)
    market, valid = market_probs_from_odds(ordered["_o"].to_numpy(), groups)
    anchored = apply_market_anchor(
        ordered["_p"].to_numpy(), market, groups,
        anchor["alpha"], anchor["beta"], race_valid=valid,
    )
    return pd.Series(anchored, index=ordered.index).reindex(frame.index).to_numpy()


def _fit_win_model(train: pd.DataFrame, predict: pd.DataFrame, features: list[str]):
    model = lgb.LGBMClassifier(**WIN_PARAMS)
    model.fit(train[features], train["won"])
    return model.predict_proba(predict[features])[:, 1]


def load_frame() -> pd.DataFrame:
    store = FeatureStore()
    available = set(pq.read_schema(store.featured_path).names)
    allow = [
        line.strip() for line in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    cols = [c for c in sorted(set(allow) | set(DIAG)) if c in available]
    df = pd.read_parquet(store.featured_path, columns=cols)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int)
    df = attach_bsp(df, cols=("bsp", "ltp_300s"))

    entry = pd.to_numeric(df["ltp_300s"], errors="coerce")
    close = pd.to_numeric(df["bsp"], errors="coerce")
    df = df[(entry > 1) & (close > 1)].copy()
    df["_entry"] = pd.to_numeric(df["ltp_300s"], errors="coerce")
    df["_close"] = pd.to_numeric(df["bsp"], errors="coerce")
    # Tradeable market probability, overround-corrected per race.
    df["mkt_imp"] = _normalise_per_race(1.0 / df["_entry"], df["race_id"])
    return df.reset_index(drop=True)


def run(df: pd.DataFrame | None = None, *, max_folds: int | None = None,
        n_boot: int = 2000, min_train_rows: int = 5000,
        dump_path: str | None = None) -> dict:
    df = load_frame() if df is None else df.reset_index(drop=True)
    features = [
        c for c in get_feature_columns(df)
        if c in df.columns
        and c not in ("bsp", "ltp_300s", "_entry", "_close", "mkt_imp")
    ]
    folds = expanding_period_folds(
        df["race_date"], freq="Q", min_train_periods=8, min_train_rows=min_train_rows,
    )
    if max_folds:
        folds = folds[:max_folds]
    print(f"rows {len(df):,}  features {len(features)}  folds {len(folds)}", flush=True)

    pnl = {"combo": [], "model": [], "all": []}
    races = {"combo": [], "model": [], "all": []}
    clv_chunks, fold_rois, per_fold = [], [], []
    anchored_oos, market_oos, won_oos, race_oos = [], [], [], []
    dump_frames: list[pd.DataFrame] = []

    for fold in folds:
        train = df[fold.train].copy()
        test = df[fold.test].copy()

        # ── Stage 1: fundamental model, OUT-OF-FOLD on the train window ──
        # The blend below must never see in-sample model probabilities.
        oof = oof_predict_expanding(
            train,
            fit_predict=lambda tr, pr: _fit_win_model(tr, pr, features),
            n_splits=4, min_train_rows=min(2000, min_train_rows),
        )
        usable = np.isfinite(oof)
        if usable.sum() < 1000:
            print(f"fold {fold.label}: too little OOF history, skipped", flush=True)
            continue
        blend_rows = train[usable].copy()
        blend_rows["_oof"] = _normalise_per_race(oof[usable], blend_rows["race_id"])

        # ── Stage 2: fit (alpha, beta) on those OOF predictions ──────────
        ordered, groups = _race_blocks(blend_rows)
        market_tr, valid_tr = market_probs_from_odds(ordered["_entry"].to_numpy(), groups)
        anchor = fit_market_anchor(
            ordered["_oof"].to_numpy(), market_tr, groups,
            ordered["won"].to_numpy(), race_valid=valid_tr,
        )
        if anchor is None:
            print(f"fold {fold.label}: anchor fit failed, skipped", flush=True)
            continue

        # ── Apply to the held-out quarter ────────────────────────────────
        test["_mp"] = _normalise_per_race(
            _fit_win_model(train, test, features), test["race_id"],
        )
        test["_combo"] = _anchor(test["_mp"], test["_entry"], test, anchor)

        won = test["won"].to_numpy().astype(bool)
        pnl_bsp = settle_win_exchange_bets(test["_close"].to_numpy(), won)
        band = test["_entry"].between(*BAND).to_numpy()
        race_ids = test["race_id"].to_numpy()

        sel_combo = (test["_combo"].to_numpy() > test["mkt_imp"].to_numpy()) & band
        sel_model = (test["_mp"].to_numpy() > test["mkt_imp"].to_numpy()) & band

        # Baselines accumulate on EVERY fold, not only the folds where the
        # combination found a bet — otherwise the comparison runs over
        # different sets of quarters.
        for key, mask in (("combo", sel_combo), ("model", sel_model), ("all", band)):
            if mask.any():
                pnl[key].append(pnl_bsp[mask])
                races[key].append(race_ids[mask])

        if dump_path:
            dump_frames.append(pd.DataFrame({
                "race_id": test["race_id"].to_numpy().astype(str),
                "race_date": test["race_date"].to_numpy(),
                "track": test.get("track", pd.Series(index=test.index, dtype=object)).to_numpy(),
                "horse_name": test.get("horse_name", pd.Series(index=test.index, dtype=object)).to_numpy(),
                "num_runners": pd.to_numeric(
                    test.get("num_runners", pd.Series(index=test.index, dtype=float)),
                    errors="coerce").to_numpy(),
                "race_type": test.get("race_type", pd.Series(index=test.index, dtype=object)).astype(str).to_numpy(),
                "won": test["won"].to_numpy().astype(np.int8),
                "entry_odds": test["_entry"].to_numpy(),
                "close_odds": test["_close"].to_numpy(),
                "mkt_imp": test["mkt_imp"].to_numpy(),
                "model_prob": test["_mp"].to_numpy(),
                "combo_prob": test["_combo"].to_numpy(),
                "pnl_bsp": pnl_bsp,
                "in_band": band,
                "sel_combo": sel_combo,
                "sel_model": sel_model,
                "fold": fold.label,
            }))

        anchored_oos.append(test["_combo"].to_numpy()[band])
        market_oos.append(test["mkt_imp"].to_numpy()[band])
        won_oos.append(test["won"].to_numpy()[band])
        race_oos.append(race_ids[band])

        if sel_combo.any():
            clv_chunks.append(np.column_stack([
                test["_entry"].to_numpy()[sel_combo],
                test["_close"].to_numpy()[sel_combo],
            ]))
            roi = float(pnl_bsp[sel_combo].mean() * 100)
            fold_rois.append(roi)
            per_fold.append({
                "quarter": fold.label,
                "alpha": round(anchor["alpha"], 4),
                "beta": round(anchor["beta"], 4),
                "combo_bets": int(sel_combo.sum()),
                "roi_bsp_combo": round(roi, 2),
                "roi_bsp_model": (round(float(pnl_bsp[sel_model].mean() * 100), 2)
                                  if sel_model.any() else None),
                "purged_rows": fold.n_purged,
            })
        print(
            f"fold {fold.label}: alpha={anchor['alpha']:+.3f} "
            f"beta={anchor['beta']:+.3f} combo_bets={int(sel_combo.sum())}",
            flush=True,
        )

    if not pnl["combo"]:
        return {"error": "no fold produced a combination bet"}

    def summarise(key):
        return bootstrap_roi_ci(
            np.concatenate(pnl[key]),
            clusters=np.concatenate(races[key]),   # resample races, not bets
            n_boot=n_boot,
        )

    anchored_all = np.concatenate(anchored_oos)
    market_all = np.concatenate(market_oos)
    won_all = np.concatenate(won_oos)
    race_all = np.concatenate(race_oos)
    nesting = check_combination_nesting(
        race_logloss(anchored_all, won_all, race_all),
        race_logloss(market_all, won_all, race_all),
    )
    alphas = [f["alpha"] for f in per_fold]

    if dump_path and dump_frames:
        pd.concat(dump_frames, ignore_index=True)[DUMP_COLS].to_parquet(
            dump_path, index=False, engine="pyarrow",
        )
        print(f"wrote per-bet rows -> {dump_path}", flush=True)

    return {
        "combo": summarise("combo"),
        "model_only": summarise("model"),
        "bet_all": summarise("all"),
        # Full CLV summary: the mean of entry/close is biased high at long
        # prices, so a mean above 1 with a median below it means a price tail,
        # not closing-line value. See src/clv.summarise_clv.
        "clv": ({k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in clv_summary(*np.vstack(clv_chunks).T).items()}
                if clv_chunks else None),
        "folds": fold_summary(fold_rois),
        "nesting_check": nesting,
        "alpha": {
            "mean": round(float(np.mean(alphas)), 4) if alphas else None,
            "positive_folds": int(sum(a > 0 for a in alphas)),
            "n_folds": len(alphas),
        },
        "per_fold": per_fold,
    }


def render(report: dict) -> str:
    if "error" in report:
        return f"\n  {report['error']}\n"
    combo, model_only, bet_all = report["combo"], report["model_only"], report["bet_all"]
    folds, nesting = report["folds"], report["nesting_check"]
    clv = (report.get("clv") or {}).get("median_clv")
    lines = [
        "",
        "  === Benter combination, walk-forward @ BSP ===",
        "  blend fitted on out-of-fold predictions; race-clustered CIs; purged folds",
        "",
        f"  COMBINATION  ROI {combo['mean']:+.2f}%  "
        f"CI[{combo['ci_low']:+.2f}, {combo['ci_high']:+.2f}]  "
        f"bets={combo['n']:,} races={combo['n_clusters']:,}"
        + (f"  medianCLV {clv:.4f}" if clv is not None else ""),
        f"  MODEL ALONE  ROI {model_only['mean']:+.2f}%  "
        f"CI[{model_only['ci_low']:+.2f}, {model_only['ci_high']:+.2f}]  bets={model_only['n']:,}",
        f"  BET ALL      ROI {bet_all['mean']:+.2f}%  "
        f"CI[{bet_all['ci_low']:+.2f}, {bet_all['ci_high']:+.2f}]  bets={bet_all['n']:,}"
        "   (fair pool ~ -commission)",
        "",
        f"  Across folds: {folds['folds_positive']}/{folds['n_folds']} positive, "
        f"median {folds['median']:+.2f}%, range {folds['worst']:+.2f}% .. {folds['best']:+.2f}%",
        f"  Fitted alpha: mean {report['alpha']['mean']:+.4f}, positive in "
        f"{report['alpha']['positive_folds']}/{report['alpha']['n_folds']} folds",
        "",
        f"  Nesting check: {'PASS' if nesting['ok'] else 'FAIL'} - {nesting['reason']}",
    ]
    if not nesting["ok"]:
        lines.append("  >> Do NOT read this run as evidence about the signal.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-folds", type=int, default=None,
                        help="limit folds for a quick smoke run")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dump", default=None, metavar="PATH",
                        help="write per-bet rows for scripts/audit_selection_rule.py")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    report = run(max_folds=args.max_folds, n_boot=args.n_boot, dump_path=args.dump)
    print(json.dumps(report, indent=2, default=str) if args.json else render(report))
    # A failed nesting check means the measurement itself is untrustworthy.
    return 0 if report.get("nesting_check", {}).get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
