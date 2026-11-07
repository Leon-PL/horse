"""Callable training core — one code path for the CLI and the training app.
============================================================================

Historically "train a model" was implemented three times (``train.py``, the
Streamlit Train & Tune page), which
drifted apart. This module is the single entry point: a typed
:class:`TrainingConfig` plus :func:`train_run`, which runs the real flow

    process_data -> get_training_dataset (the store refresh) -> RacePredictor.train
    -> evaluate (calibration + backtest ROI + CLV vs BSP) -> save a lean run

An explicit deployment cutoff instead reuses ``RacePredictor.fit_as_of`` and
links a separately evaluated recipe. That artifact has no held-out scores.

and returns the ``run_id``. Both the rewritten ``train.py`` CLI and the training
app call ``train_run`` so they can never diverge again. Runs are saved LEAN (no
per-run featured/processed parquet — that is reproducible from the feature store
via the data/fe signatures), and the model can be promoted to production with
``src.run_store.promote``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

import config
from src.bet_settlement import DEFAULT_VALUE_CONFIG
from src.clv import compute_clv
from src.data_processor import process_data
from src.featured_dataset import data_signature, fe_signature
from src.store_refresh import get_training_dataset
from src.model import RacePredictor
from src.run_store import save_run

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str, float], None]


@dataclass
class TrainingConfig:
    """Everything that defines a training run (saved verbatim for reproducibility)."""

    name: str = "training run"
    # ── data ──
    refresh_data: bool = True            # catch-up scrape recent days into the DB first
    catchup_days: int = 14
    train_window_days: int | None = None  # None = full history; else last N days
    holdout_days: int | None = 90        # None retains the legacy percentage split
    fit_as_of: str | None = None        # explicit settled cutoff; no evaluation holdout
    validation_report: str | None = None  # separately evaluated recipe certificate
    threads: int | None = None          # execution limit, not a statistical setting
    # ── model ──
    frameworks: dict = field(default_factory=lambda: dict(config.SUB_MODEL_FRAMEWORKS))
    win_calibration_method: str = field(default_factory=lambda: config.WIN_CALIBRATION_METHOD)
    blend_market_odds: bool = False
    hyperparams: dict | None = None
    value_config: dict = field(default_factory=lambda: dict(DEFAULT_VALUE_CONFIG))
    # ── walk-forward (optional) ──
    run_walk_forward: bool = False
    wf_min_train_months: int = 3
    wf_fast_fold: bool = True
    # ── repro ──
    seed: int = config.RANDOM_SEED

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def deployment(cls, report, cutoff, *, name="deployment refit", refresh_data=False,
                   catchup_days=14, threads=None):
        """Inherit the locked options instead of silently substituting defaults."""
        from src.deployment_validation import RECIPE_OPTIONS
        contents = json.loads(Path(report).read_text(encoding="utf-8"))
        options = contents["recipe"]["options"]
        if set(options) != set(RECIPE_OPTIONS):
            raise ValueError("Validation report has incomplete recipe options")
        return cls(**options, name=name, fit_as_of=cutoff, validation_report=str(report),
                   refresh_data=refresh_data, catchup_days=catchup_days, threads=threads)


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
    if cfg.threads is not None and (isinstance(cfg.threads, bool)
            or not isinstance(cfg.threads, int) or cfg.threads < 1):
        raise ValueError("threads must be a positive integer")

    from src.deployment_validation import load_validation_report, training_recipe
    if cfg.validation_report and cfg.fit_as_of is None:
        raise ValueError("validation_report requires an explicit fit_as_of cutoff")
    validation = None
    recipe = training_recipe(cfg)
    if cfg.fit_as_of is not None:
        try:
            cutoff = date.fromisoformat(cfg.fit_as_of)
        except (ValueError, TypeError) as exc:
            raise ValueError("fit_as_of must be a YYYY-MM-DD calendar date") from exc
        if cutoff.isoformat() != cfg.fit_as_of or cutoff >= date.today():
            raise ValueError("fit_as_of must be a settled date before today")
        if cfg.blend_market_odds or cfg.run_walk_forward:
            raise ValueError("Deployment refits require separate market correction and historical evaluation")
        # A certificate is optional: without one the refit simply has no
        # held-out score. With one, the recipe must match what was evaluated.
        if cfg.validation_report:
            validation = load_validation_report(cfg.validation_report, recipe)
            if validation["fe_signature"] != fe_signature():
                raise ValueError("Deployment features differ from the validated feature generation")

    def _tick(msg: str, pct: float) -> None:
        logger.info("[train_run] %s (%.0f%%)", msg, 100 * pct)
        if progress_cb:
            progress_cb(msg, pct)

    # 1) Data — optional catch-up scrape, then the full processed history.
    if cfg.refresh_data:
        _tick(f"Scraping last {cfg.catchup_days}d into the DB", 0.02)
        from src.ingestion.pipeline import refresh_sources, refresh_summary
        refresh_report = refresh_sources(days_back=cfg.catchup_days)
        _tick(refresh_summary(refresh_report), 0.07)
        if refresh_report["state"] != "ok":
            raise RuntimeError(refresh_summary(refresh_report) +
                ". Complete the source catch-up before training, or explicitly choose --no-refresh to use local history.")
    _tick("Processing history", 0.08)
    processed = process_data()

    # 2) Features — the store seam (build once / append new dates, never rebuild
    #    unless code/data signatures changed).
    _tick("Loading featured dataset (feature store)", 0.15)
    source_fe_signature = fe_signature()
    featured = get_training_dataset(processed=processed)
    if fe_signature() != source_fe_signature:
        raise RuntimeError("Feature-generation code changed while loading the dataset; restart training.")
    if validation and validation["fe_signature"] != source_fe_signature:
        raise ValueError("Deployment features differ from the validated feature generation")
    # Freeze provenance before the long fit. Another process may edit source
    # files or advance the store while this predictor trains on its own frame.
    last = pd.to_datetime(processed["race_date"], errors="coerce").max()
    source_signatures = {
        "fe_signature": source_fe_signature,
        "data_signature": data_signature(processed, cutoff=last),
    }
    source_git_sha = _git_sha()
    source_config = cfg.to_dict()
    # Burn-in is an absolute boundary tied to the FULL history's start; capture
    # it before windowing so a short window isn't wrongly treated as cold-start.
    dataset_start = pd.to_datetime(featured["race_date"], errors="coerce").min()
    if cfg.fit_as_of is not None:
        # Window against the declared historical cutoff, never the end of a
        # frame which might include future races.
        featured = featured.loc[pd.to_datetime(featured.race_date).le(cfg.fit_as_of)].copy()
    featured = _apply_window(featured, cfg)
    from src.forecast_evaluation import valid_race_mask
    if featured.empty:
        raise ValueError("Training requires at least one complete single-winner race")
    eligible = valid_race_mask(featured)
    rejected_fields = {"policy": "complete single-winner fields; genuine non-finishers retained",
                       "excluded_races": int(featured.loc[~eligible, "race_id"].nunique(dropna=False)),
                       "excluded_rows": int((~eligible).sum())}
    logger.info("Whole-field validation excluded %d races / %d rows",
                rejected_fields["excluded_races"], rejected_fields["excluded_rows"])
    featured = featured.loc[eligible].copy()
    if featured.empty:
        raise ValueError("Training requires at least one complete single-winner race")
    if validation:
        from src.model import get_feature_columns
        # Column subsampling makes training order part of the fitted recipe.
        # A store may write the same selected inputs in another schema order.
        expected = validation["features"]
        if set(get_feature_columns(featured)) != set(expected):
            raise ValueError("Deployment feature columns differ from the validated model")
        featured = featured.loc[:, expected + [c for c in featured if c not in expected]]

    # 3) Train.
    _tick("Training models", 0.30)
    predictor = RacePredictor(
        frameworks=dict(cfg.frameworks), random_seed=cfg.seed,
        win_calibration_method=cfg.win_calibration_method,
    )
    if cfg.threads is not None:
        predictor._autotune_njobs = cfg.threads
    if cfg.fit_as_of is not None:
        predictor.fit_as_of(featured, cfg.fit_as_of, params=cfg.hyperparams,
                            dataset_start=dataset_start)
        if validation and predictor.feature_cols != validation["features"]:
            raise ValueError("Deployment feature columns differ from the validated model")
        metrics = {}
        evals = {"scope": "Deployment refit has no held-out performance estimate",
                 "validation_reference": validation}
    else:
        metrics = predictor.train(
            featured, save=False, params=cfg.hyperparams, value_config=cfg.value_config,
            blend_market_odds=cfg.blend_market_odds, dataset_start=dataset_start,
            holdout_days=cfg.holdout_days,
            progress_callback=(lambda m, p: _tick(m, 0.30 + 0.55 * p)) if progress_cb else None,
        )
        _tick("Evaluating (calibration / ROI / CLV)", 0.88)
        evals = evaluate(predictor, metrics, featured)
    predictor.eval_split_info = dict(getattr(predictor, "eval_split_info", {}) or {})
    predictor.eval_split_info["input_field_validation"] = rejected_fields

    # 5) Optional walk-forward.
    wf_report = None
    if cfg.run_walk_forward:
        _tick("Walk-forward validation", 0.92)
        from src.backtester import walk_forward_validation
        vc = {**DEFAULT_VALUE_CONFIG, **cfg.value_config}
        wf_report = walk_forward_validation(
            featured,
            min_train_months=cfg.wf_min_train_months,
            frameworks=dict(cfg.frameworks),
            params=cfg.hyperparams,
            fast_fold=cfg.wf_fast_fold,
            blend_market_odds=cfg.blend_market_odds,
            random_seed=cfg.seed,
            win_calibration_method=cfg.win_calibration_method,
            value_threshold=vc["value_threshold"],
            value_min_odds=vc["value_min_odds"],
            value_max_odds=vc["value_max_odds"],
            ew_enabled=vc["ew_enabled"],
            ew_min_place_edge=vc["ew_min_place_edge"],
            ew_min_odds=vc["ew_min_odds"],
            ew_max_odds=vc["ew_max_odds"],
            progress_callback=(lambda m, p: _tick(m, 0.92 + 0.05 * p)) if progress_cb else None,
        )

    if training_recipe(cfg) != recipe:
        raise RuntimeError("Training recipe changed during fitting; refusing to publish the run")

    # 6) Save a LEAN run (no featured/processed parquet — reproducible from the
    #    store via the recorded signatures).
    _tick("Saving run", 0.98)
    training_config = {
        "config": source_config,
        "evals": evals,
        "git_sha": source_git_sha,
        "seed": source_config["seed"],
        "signatures": source_signatures,
        "split": dict(getattr(predictor, "eval_split_info", {})),
        "recipe": recipe,
        "validation_reference": validation,
    }
    run_id = save_run(
        predictor=predictor,
        name=cfg.name,
        model_type="race_predictor",
        data_source="feature_store",
        data_rows=len(featured),
        n_features=len(predictor.feature_cols),
        elapsed_seconds=time.time() - t0,
        ensemble_weights=None,
        metrics=metrics,
        train_metrics=getattr(predictor, "train_metrics", None),
        test_analysis=predictor.test_analysis,
        training_config=training_config,
        wf_report=wf_report,
        featured_df=None,      # lean: do NOT snapshot the featured parquet
        processed_df=None,     # lean: do NOT snapshot the processed parquet
    )
    _tick(f"Done: run {run_id}", 1.0)
    return run_id
