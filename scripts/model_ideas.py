"""Shared harness for the 057-060 win-model ideas (docs/research/MODEL_IDEAS_BRIEF.md).

Every arm is fitted on the same rows and folds as experiments 048/049/052: the
development cache of ``scripts/select_features.py`` (golden hold-out excluded),
purged expanding quarterly folds, and the production win recipe
(``feature_selection.fit_win_model``) on the production inputs
(``feature_sets/win_selected_056.txt``). Arms differ only in what each idea
changes. Development screens use the six selection quarters 2024Q1-2025Q2 at
screen capacity; confirmation uses 2025Q3-2026Q2 at native capacity.

Each fold's out-of-sample predictions are saved as soon as they exist, so a
stopped run resumes. Scores are race log loss of the actual winner after
per-race normalisation, with date-clustered bootstrap intervals on paired
per-race differences. Non-finishers stay in every field.

    python -m scripts.model_ideas baseline          # production arm, confirmation quarters
    python -m scripts.model_ideas baseline --dev    # production arm, selection quarters (screen)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.select_features import (CONFIRM_QUARTERS, OUT as CACHE_DIR, SELECTION_QUARTERS, Cache,  # noqa: E402
                                     read_names)
from src import feature_selection as fs  # noqa: E402
from src.feature_store import FeatureStore  # noqa: E402
from src.forecast_evaluation import paired_interval  # noqa: E402

OUT = ROOT / "data/research/model_ideas_057"
FEATURES = ROOT / "feature_sets/win_selected_056.txt"
THREADS = os.cpu_count() or 4
STAGES = {
    "dev": {"quarters": SELECTION_QUARTERS, "capacity": "screen"},
    "confirm": {"quarters": CONFIRM_QUARTERS, "capacity": "native"},
}

logger = logging.getLogger("model_ideas")


def production_inputs(cache: Cache) -> list[str]:
    names = read_names(FEATURES)
    missing = [n for n in names if n not in cache.index]
    if missing:
        raise SystemExit(f"Production inputs missing from the cache ({missing[:5]}); rerun "
                         "python -m scripts.select_features prepare on the current store.")
    return names


def store_columns(columns: list[str]) -> pd.DataFrame:
    """Store columns for every cached row, read by store row (as 049 reads race type)."""
    rows = pd.read_parquet(CACHE_DIR / "targets.parquet", columns=["store_row"]).store_row.to_numpy()
    table = pq.read_table(FeatureStore().featured_path, columns=columns).to_pandas()
    return table.iloc[rows].reset_index(drop=True)


def stage_dir(stage: str) -> Path:
    return OUT / stage


def predict_production(cache: Cache, names: list[str], train: np.ndarray, test: np.ndarray, *,
                       capacity: str, weight_multiplier: np.ndarray | None = None,
                       extra: np.ndarray | None = None) -> np.ndarray:
    """The production win recipe on one fold; optional row weight multipliers and extra inputs."""
    t = cache.targets

    def take(rows):
        x = cache.take(rows, names)
        return x if extra is None else np.hstack([x, extra[rows].astype(np.float32)])

    weights = fs.recency_weights(t.race_dates[train])
    if weight_multiplier is not None:
        weights = weights * weight_multiplier[train]
    model = fs.fit_win_model(take(train), t.won[train], t.soft_y[train], weights, t.race_ids[train],
                             t.race_dates[train], capacity=capacity, n_jobs=THREADS)
    raw = model.predict_proba(take(test))[:, 1]
    return fs.race_normalise(raw, fs.race_groups(t.race_ids[test]))


def run_arm(cache: Cache, stage: str, arm: str,
            predict: Callable[[np.ndarray, np.ndarray, str], np.ndarray | pd.DataFrame]) -> None:
    """Fit *arm* on every fold of *stage*, saving each fold's predictions (resumable).

    *predict(train_rows, test_rows, capacity)* returns the race-normalised win
    probability per test row, or a frame with a ``model`` column plus extras.
    """
    spec = STAGES[stage]
    directory = stage_dir(stage)
    directory.mkdir(parents=True, exist_ok=True)
    t = cache.targets
    for fold in cache.fold_data(spec["quarters"]):
        path = directory / f"{arm}_{fold.label}.parquet"
        if path.exists():
            continue
        started = time.monotonic()
        result = predict(fold.train_rows, fold.test_rows, spec["capacity"])
        frame = result.rename(columns={"p": "model"}) if isinstance(result, pd.DataFrame) else \
            pd.DataFrame({"model": result})
        test = fold.test_rows
        frame.insert(0, "row", test)
        frame.insert(1, "race_id", t.race_ids[test])
        frame.insert(2, "race_date", t.race_dates[test])
        frame.insert(3, "won", t.won[test])
        frame.insert(4, "market", t.market[test])
        frame.to_parquet(path, index=False)
        ll = race_losses(frame, "model").mean()
        logger.info("%s %s %s: race LL %.5f on %d races (%.0fs)", stage, arm, fold.label, ll,
                    frame.race_id.nunique(), time.monotonic() - started)


def load_arm(stage: str, arm: str) -> pd.DataFrame:
    quarters = STAGES[stage]["quarters"]
    paths = [stage_dir(stage) / f"{arm}_{q}.parquet" for q in quarters]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Arm {arm} has no saved predictions for {missing}; run it first.")
    return pd.concat([pd.read_parquet(p).assign(fold=q) for p, q in zip(paths, quarters)], ignore_index=True)


def race_losses(runners: pd.DataFrame, column: str) -> pd.Series:
    """Per-race log loss of the race-normalised *column*, indexed by race id."""
    groups = fs.race_groups(runners.race_id.to_numpy())
    p = fs.race_normalise(runners[column].to_numpy(np.float64), groups)
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    loss = -np.add.reduceat(runners.won.to_numpy() * np.log(np.clip(p, 1e-12, None)), starts)
    return pd.Series(loss, index=runners.race_id.to_numpy()[starts])


def compare(stage: str, arms: list[str], pairs: list[tuple[str, str]]) -> dict:
    """Pooled race log loss per arm and paired differences (challenger minus reference)."""
    frames = {arm: load_arm(stage, arm) for arm in arms}
    base = frames[arms[0]]
    races = base.drop_duplicates("race_id").set_index("race_id")[["race_date", "fold"]]
    market = race_losses(base, "market")
    losses = {}
    for arm, frame in frames.items():
        if not np.array_equal(frame.race_id.unique(), base.race_id.unique()) or len(frame) != len(base):
            raise ValueError(f"{arm} was not scored on the same runners as {arms[0]}")
        losses[arm] = race_losses(frame, "model")
    summary = {"stage": stage, "quarters": STAGES[stage]["quarters"], "races": int(len(races)),
               "runners": int(len(base)), "market_ll": float(market.mean()), "arms": {}, "pairs": []}
    for arm, loss in losses.items():
        per_quarter = loss.groupby(races.fold).mean()
        summary["arms"][arm] = {"race_ll": float(loss.mean()), "gap_to_market": float((loss - market).mean()),
                                "per_quarter": {q: round(float(v), 5) for q, v in per_quarter.items()}}
    for challenger, reference in pairs:
        delta = losses[challenger] - losses[reference]
        per_quarter = delta.groupby(races.fold).mean()
        ci = paired_interval(delta.to_numpy(), blocks=races.race_date.astype(str).to_numpy())["ci95"]
        summary["pairs"].append({
            "challenger": challenger, "reference": reference, "delta": float(delta.mean()), "ci95": ci,
            "quarters_better": f"{int((per_quarter < 0).sum())}/{len(per_quarter)}",
            "per_quarter": {q: round(float(v), 5) for q, v in per_quarter.items()},
        })
    return summary


def print_summary(summary: dict, title: str) -> None:
    print(f"\n{title}: {summary['races']:,} races, {summary['runners']:,} runners, "
          f"{', '.join(summary['quarters'])}; market race LL {summary['market_ll']:.4f}")
    for arm, row in summary["arms"].items():
        print(f"  {arm:>26}  race LL {row['race_ll']:.5f}  gap to market {row['gap_to_market']:+.5f}  "
              f"per quarter {row['per_quarter']}")
    for pair in summary["pairs"]:
        lo, hi = pair["ci95"]
        print(f"  {pair['challenger']} - {pair['reference']}: {pair['delta']:+.5f} [{lo:+.5f}, {hi:+.5f}], "
              f"better in {pair['quarters_better']} quarters {pair['per_quarter']}")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def baseline(cache: Cache, stage: str) -> None:
    names = production_inputs(cache)
    run_arm(cache, stage, "production",
            lambda train, test, capacity: predict_production(cache, names, train, test, capacity=capacity))


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("src.model").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["baseline"])
    parser.add_argument("--dev", action="store_true", help="selection quarters at screen capacity")
    args = parser.parse_args()
    setup_logging()
    stage = "dev" if args.dev else "confirm"
    baseline(Cache(), stage)
    summary = compare(stage, ["production"], [])
    print_summary(summary, f"Production recipe ({stage})")


if __name__ == "__main__":
    main()
