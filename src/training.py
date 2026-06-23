"""Callable training core — one code path for the CLI and the training app.
============================================================================

Historically "train a model" was implemented three times (``train.py``, the
Streamlit Train & Tune page, and the ``scripts/train_600d*`` one-offs), which
drifted apart. This module is the single entry point: a typed
:class:`TrainingConfig` plus :func:`train_run`, which runs the real flow

    process_data -> get_featured_dataset (the store seam) -> RacePredictor.train
    -> evaluate (calibration + backtest ROI + CLV vs BSP) -> save a lean run

and returns the ``run_id``. Both the rewritten ``train.py`` CLI and the training
app call ``train_run`` so they can never diverge again. Runs are saved LEAN (no
per-run featured/processed parquet — that is reproducible from the feature store
via the data/fe signatures), and the model can be promoted to production with
``src.run_store.promote``.
"""

from __future__ import annotations

import dataclasses
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

import config
from src.clv import compute_clv
from src.data_processor import process_data
from src.featured_dataset import data_signature, fe_signature, get_featured_dataset
from src.model import RacePredictor, get_feature_columns
from src.run_store import save_run

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str, float], None]


DEFAULT_VALUE_CONFIG: dict = {
    "staking_mode": "flat",
    "value_threshold": 0.05,
    "value_min_odds": 1.0,
    "value_max_odds": 101.0,
    "kelly_fraction": 0.25,
    "bankroll": 100.0,
    "ew_enabled": True,
    "ew_fraction": 0.20,
    "ew_min_place_edge": 0.15,
    "ew_min_odds": 4.0,
    "ew_max_odds": 51.0,
}


@dataclass
class TrainingConfig:
    """Everything that defines a training run (saved verbatim for reproducibility)."""

    name: str = "training run"
    # ── data ──
    refresh_data: bool = True            # catch-up scrape recent days into the DB first
    catchup_days: int = 14
    train_window_days: int | None = None  # None = full history; else last N days
    # ── model ──
    frameworks: dict = field(default_factory=lambda: dict(config.SUB_MODEL_FRAMEWORKS))
    blend_market_odds: bool = False
    hyperparams: dict | None = None
    value_config: dict = field(default_factory=lambda: dict(DEFAULT_VALUE_CONFIG))
    # ── place-market strategy (opt-in, alongside the win/EW pipeline) ──
    train_place_market: bool = False        # also train the +EV exchange-place model
    place_market_features: bool = True       # form + pre-off win-market (else form-only)
    place_ev_threshold: float = 0.10
    # ── walk-forward (optional) ──
    run_walk_forward: bool = False
    wf_min_train_months: int = 3
    wf_fast_fold: bool = True
    # ── repro ──
    seed: int = config.RANDOM_SEED

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.BASE_DIR,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _apply_window(featured: pd.DataFrame, cfg: TrainingConfig) -> pd.DataFrame:
    """Restrict to the last ``train_window_days`` (a split, not a separate build)."""
    if not cfg.train_window_days:
        return featured
    rd = pd.to_datetime(featured["race_date"], errors="coerce")
    cutoff = rd.max() - pd.Timedelta(days=int(cfg.train_window_days))
    out = featured[rd >= cutoff].copy()
    logger.info("Training window: last %dd -> %d of %d rows",
                cfg.train_window_days, len(out), len(featured))
    return out


def evaluate(predictor: RacePredictor, metrics: dict, featured: pd.DataFrame) -> dict:
    """Bundle the three headline metric families from a trained predictor.

    * **calibration** — Brier / log-loss / ECE for the win & place classifiers
      (and the linear / market baselines) from ``metrics``.
    * **roi** — backtest betting stats per strategy from ``test_analysis``.
    * **clv** — closing-line value vs Betfair BSP on the test-set bets.
    """
    ta = predictor.test_analysis or {}
    stats = ta.get("stats", {}) or {}
    bets = ta.get("bets")

    calibration = {
        k: metrics.get(k) for k in (
            "win_classifier", "place_classifier", "baseline_win",
            "baseline_market", "baseline_place",
        ) if isinstance(metrics.get(k), dict)
    }
    roi = {s: stats.get(s) for s in ("top_pick", "value", "each_way") if stats.get(s)}

    clv: dict = {"bsp_available": False}
    if isinstance(bets, pd.DataFrame) and not bets.empty:
        try:
            clv = compute_clv(bets, ref_df=featured)
        except Exception as exc:  # CLV is a report, never fail a run over it
            logger.warning("CLV computation failed: %s", exc)
            clv = {"error": str(exc)}
    return {"calibration": calibration, "roi": roi, "clv": clv}


def train_run(cfg: TrainingConfig, *, progress_cb: ProgressCb | None = None) -> str:
    """Run the full pipeline for *cfg* and save a lean run. Returns the run_id."""
    t0 = time.time()

    def _tick(msg: str, pct: float) -> None:
        logger.info("[train_run] %s (%.0f%%)", msg, 100 * pct)
        if progress_cb:
            progress_cb(msg, pct)

    # 1) Data — optional catch-up scrape, then the full processed history.
    if cfg.refresh_data:
        _tick(f"Scraping last {cfg.catchup_days}d into the DB", 0.02)
        try:
            from src.database import sync_database
            sync_database(days_back=cfg.catchup_days)
        except Exception as exc:
            logger.warning("Catch-up scrape failed (%s); using DB as-is", exc)
    _tick("Processing history", 0.08)
    processed = process_data(save=False)

    # 2) Features — the store seam (build once / append new dates, never rebuild
    #    unless code/data signatures changed).
    _tick("Loading featured dataset (feature store)", 0.15)
    featured = get_featured_dataset(processed=processed)
    # Burn-in is an absolute boundary tied to the FULL history's start; capture
    # it before windowing so a short window isn't wrongly treated as cold-start.
    dataset_start = pd.to_datetime(featured["race_date"], errors="coerce").min()
    featured = _apply_window(featured, cfg)

    # 3) Train.
    _tick("Training models", 0.30)
    predictor = RacePredictor(frameworks=dict(cfg.frameworks))
    metrics = predictor.train(
        featured, params=cfg.hyperparams, value_config=cfg.value_config,
        blend_market_odds=cfg.blend_market_odds, dataset_start=dataset_start,
        progress_callback=(lambda m, p: _tick(m, 0.30 + 0.55 * p)) if progress_cb else None,
    )

    # 4) Evaluate (calibration + ROI + CLV).
    _tick("Evaluating (calibration / ROI / CLV)", 0.88)
    evals = evaluate(predictor, metrics, featured)

    # 4b) Optional place-market strategy (the validated +EV exchange-place model),
    #     trained ALONGSIDE the win/EW pipeline. Held-out eval block + artifact.
    place_market = None
    if cfg.train_place_market:
        _tick("Training place-market strategy (+EV @ place BSP)", 0.90)
        try:
            from src.place_market_model import train_and_evaluate as _pm_train
            pm_model, pm_meta, pm_eval = _pm_train(
                featured, use_market_features=cfg.place_market_features,
                ev_threshold=cfg.place_ev_threshold)
            evals["place_market"] = pm_eval
            place_market = (pm_model, pm_meta)
        except Exception as exc:  # never fail the whole run over the opt-in strategy
            logger.warning("Place-market strategy training failed: %s", exc)
            evals["place_market"] = {"error": str(exc)}

    # 5) Optional walk-forward.
    wf_report = None
    if cfg.run_walk_forward:
        _tick("Walk-forward validation", 0.92)
        from src.backtester import walk_forward_validation
        wf_report = walk_forward_validation(featured)

    # 6) Save a LEAN run (no featured/processed parquet — reproducible from the
    #    store via the recorded signatures).
    _tick("Saving run", 0.98)
    last = pd.to_datetime(processed["race_date"], errors="coerce").max()
    training_config = {
        "config": cfg.to_dict(),
        "evals": evals,
        "git_sha": _git_sha(),
        "seed": cfg.seed,
        "signatures": {
            "fe_signature": fe_signature(),
            "data_signature": data_signature(processed, cutoff=last),
        },
    }
    run_id = save_run(
        name=cfg.name,
        model_type="race_predictor",
        data_source="feature_store",
        data_rows=len(featured),
        n_features=len(get_feature_columns(featured)),
        elapsed_seconds=time.time() - t0,
        ensemble_weights=None,
        metrics=metrics,
        train_metrics=getattr(predictor, "train_metrics", None),
        test_analysis=predictor.test_analysis,
        training_config=training_config,
        wf_report=wf_report,
        featured_df=None,      # lean: do NOT snapshot the featured parquet
        processed_df=None,     # lean: do NOT snapshot the processed parquet
        place_market_model=place_market,  # (model, meta) or None
    )
    _tick(f"Done: run {run_id}", 1.0)
    return run_id
