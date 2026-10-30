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
    python train.py --promote               # promote the new run to production
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
    from src.model import SUPPORTED_FRAMEWORKS

    valid_vals = set(SUPPORTED_FRAMEWORKS)
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
    split = (meta.get("training_config") or {}).get("split") or {}
    if split.get("mode") == "fit_as_of":
        print(f"  Parameters fitted through {split.get('training_last_date')} "
              f"(declared cutoff {split.get('as_of')}).")
        reference = (meta.get("training_config") or {}).get("validation_reference") or {}
        print(f"  Recipe validation: {reference.get('validation_run_id', 'unknown')}; "
              "this refit has no held-out scores.")
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
    ap.add_argument("--features", default=None, metavar="PATH",
                    help="explicit feature allow-list, e.g. feature_sets/win_with_rtv.txt")
    ap.add_argument("--window-days", type=int, default=None,
                    help="train on the last N days only (default: full history)")
    fitting = ap.add_mutually_exclusive_group()
    fitting.add_argument("--holdout-days", type=int, default=90,
                    help="reserve the latest N calendar days for evaluation; 0 uses the legacy percentage split")
    fitting.add_argument("--fit-as-of", default=None, metavar="YYYY-MM-DD",
                    help="fit through this settled cutoff with no held-out evaluation")
    ap.add_argument("--validation-report", default=None, metavar="PATH",
                    help="optional recipe certificate for --fit-as-of; its settings are reused and checked")
    ap.add_argument("--threads", type=int, default=None,
                    help="limit workers used by native model fitting")
    ap.add_argument("--frameworks", default=None,
                    help="e.g. 'classifier=lgbm,place=lgbm' (LightGBM is the only framework)")
    ap.add_argument("--blend-market-odds", action="store_true",
                    help="enable the Benter market anchor (SP known only at off — optimistic)")
    ap.add_argument("--win-calibration", choices=("platt_isotonic", "race_temperature"),
                    default=None,
                    help="win probability calibration method, saved with the run")
    ap.add_argument("--backtest", action="store_true", help="run walk-forward validation")
    ap.add_argument("--promote", action="store_true", help="promote the new run to production")
    args = ap.parse_args()
    if args.features:
        if args.validation_report:
            ap.error("Deployment features come from --validation-report; evaluate a changed feature list separately")
        feature_path = os.path.abspath(args.features)
        if not os.path.isfile(feature_path):
            ap.error(f"Feature allow-list does not exist: {feature_path}")
        config.FEATURE_ALLOWLIST = feature_path
    if args.holdout_days < 0:
        ap.error("--holdout-days must be non-negative")

    if args.validation_report and args.fit_as_of is None:
        ap.error("--validation-report applies only to --fit-as-of")
    if args.fit_as_of is not None and args.validation_report:
        if (args.frameworks is not None or args.win_calibration is not None
                or args.window_days is not None or args.blend_market_odds or args.backtest):
            ap.error("Deployment settings come from --validation-report; evaluate changed settings separately")
        cfg = TrainingConfig.deployment(args.validation_report, args.fit_as_of,
            name=args.name, refresh_data=not args.no_refresh,
            catchup_days=args.catchup_days, threads=args.threads)
    else:
        cfg = TrainingConfig(
            name=args.name,
            refresh_data=not args.no_refresh,
            catchup_days=args.catchup_days,
            train_window_days=args.window_days,
            holdout_days=args.holdout_days or None,
            fit_as_of=args.fit_as_of,
            frameworks=_parse_frameworks(args.frameworks) or dict(config.SUB_MODEL_FRAMEWORKS),
            blend_market_odds=args.blend_market_odds,
            win_calibration_method=args.win_calibration or config.WIN_CALIBRATION_METHOD,
            run_walk_forward=args.backtest,
            threads=args.threads,
        )
    run_id = train_run(cfg)
    _print_summary(run_id)
    if args.promote:
        if not promote(run_id):
            raise SystemExit(f"Run {run_id} was saved. Promotion refused; inspect its feature compatibility.")
        print(f"\nPromoted {run_id} to production.")


if __name__ == "__main__":
    main()
