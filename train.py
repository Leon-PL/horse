"""Training CLI — a thin wrapper over ``src.training.train_run``.
================================================================

This is now a *shell*: all real logic lives in :mod:`src.training` so the CLI
and the Streamlit training app share one code path. It builds a
:class:`TrainingConfig` from the arguments, runs the pipeline, and prints the
headline metrics (calibration + backtest ROI + CLV vs BSP).

Usage:
    python train.py                          # full history from the feature store
    python train.py --no-refresh             # skip the catch-up scrape
    python train.py --window-days 600        # train on the last 600 days only
    python train.py --backtest               # add walk-forward validation
    python train.py --frameworks classifier=cat,place=lgbm
    python train.py --promote                # promote the new run to production
"""

import argparse
import logging
import os

# Limit numpy/BLAS threads to prevent ReleaseSemaphore parallel crashes.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import config
from src.run_store import promote
from src.training import TrainingConfig, train_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def _parse_frameworks(s: str | None) -> dict | None:
    if not s:
        return None
    valid_vals = {"xgb", "lgbm", "cat"}
    out: dict[str, str] = {}
    for pair in s.split(","):
        k, _, v = pair.strip().partition("=")
        if k not in config.SUB_MODEL_FRAMEWORKS:
            raise SystemExit(f"Unknown model role '{k}'. Choose classifier or place.")
        if v not in valid_vals:
            raise SystemExit(f"Invalid framework value '{v}'. Must be one of {valid_vals}.")
        out[k] = v
    return out


def _print_summary(run_id: str) -> None:
    from src.run_store import load_run_meta
    meta = load_run_meta(run_id)
    evals = (meta.get("training_config") or {}).get("evals", {})
    print("\n================ SUMMARY ================")
    print("run_id:", run_id)
    for name, m in (evals.get("calibration") or {}).items():
        if "brier_score" in m:
            print(f"  {name:16s} brier={m['brier_score']:.6f} logloss={m.get('log_loss', float('nan')):.4f} "
                  f"ece={m.get('ece', float('nan')):.4f} ndcg@1={m.get('ndcg_at_1', float('nan')):.4f}")
    for strat, s in (evals.get("roi") or {}).items():
        print(f"  {strat:10s} bets={s.get('bets')} strike={s.get('strike_rate')}% "
              f"roi={s.get('roi')}% pnl={s.get('pnl')}")
    clv = evals.get("clv") or {}
    if clv.get("overall"):
        o = clv["overall"]
        print(f"  CLV vs BSP: mean={o.get('mean_clv'):.3f} beat_close={100 * o.get('pct_beat_close', 0):.1f}% "
              f"(coverage {100 * clv.get('coverage', 0):.0f}%, n={o.get('n')})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the horse-racing model")
    ap.add_argument("--name", default="cli training run")
    ap.add_argument("--no-refresh", action="store_true", help="skip the catch-up scrape")
    ap.add_argument("--catchup-days", type=int, default=14)
    ap.add_argument("--window-days", type=int, default=None,
                    help="train on the last N days only (default: full history)")
    ap.add_argument("--frameworks", default=None,
                    help="e.g. 'classifier=cat,place=lgbm'")
    ap.add_argument("--blend-market-odds", action="store_true",
                    help="enable the Benter market anchor (SP known only at off — optimistic)")
    ap.add_argument("--win-calibration", choices=("platt_isotonic", "race_temperature"),
                    default=config.WIN_CALIBRATION_METHOD,
                    help="win probability calibration method, saved with the run")
    ap.add_argument("--backtest", action="store_true", help="run walk-forward validation")
    ap.add_argument("--promote", action="store_true", help="promote the new run to production")
    args = ap.parse_args()

    cfg = TrainingConfig(
        name=args.name,
        refresh_data=not args.no_refresh,
        catchup_days=args.catchup_days,
        train_window_days=args.window_days,
        frameworks=_parse_frameworks(args.frameworks) or dict(config.SUB_MODEL_FRAMEWORKS),
        blend_market_odds=args.blend_market_odds,
        win_calibration_method=args.win_calibration,
        run_walk_forward=args.backtest,
    )
    run_id = train_run(cfg)
    _print_summary(run_id)
    if args.promote:
        promote(run_id)
        print(f"\nPromoted {run_id} to production.")


if __name__ == "__main__":
    main()
