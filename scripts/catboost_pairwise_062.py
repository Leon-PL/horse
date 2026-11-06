"""CatBoost pairwise win model (experiment 062, a quick screen).

One arm on the 057-060 harness (``scripts/model_ideas.py``): the 129 production
inputs, purged quarterly folds 2025Q3-2026Q2, scored against the saved
production, exploded (057) and figure (058) arms on the same races.

The learner is CatBoost with its pairwise logistic loss (``PairLogit``, a
Bradley-Terry model of who beats whom). The pairs are the pairwise form of
057's registered exploded softmax: the winner against every other runner,
the second against the runners below it, the third likewise, weighted
(1, 0.25, 0.0625) times the production recency weight. Non-finishers are
beaten by every placed runner; a dead heat or a missing place ends a race's
sequence (``race_objectives.exploded_stages``). Nothing is tuned on the
confirmation quarters: the CatBoost settings below are fixed, the iteration
count comes from the production purged early-stopping split (then a refit on
every row), and win chances are a race softmax of the scores with one
temperature fitted on that early-stopping slab.

CatBoost is not a project dependency (LightGBM is the only production
framework). It is installed beside the locked env and put on the path here:

    UV_CACHE_DIR=data/tooling/uv-cache data/tooling/bootstrap/Scripts/uv.exe pip install \\
        --python .venv/Scripts/python.exe --target data/tooling/catboost --no-deps \\
        --only-binary :all: catboost==1.2.10 six==1.17.0

The first arm (``confirm``) is on results alone at fast settings. The second
(``soft``) learns as production does: its first stage is every ordered pair in
the race weighted by production's half-market soft target, at a learning rate of
0.05. Its readings were fixed before any of it ran: against production and 057,
blended with 057, and added to the 057 + figure blend (:func:`stacked`). Blend
weights are always fitted on earlier quarters. The soft arm was chosen after the
first arm had been scored on these quarters, so neither is a registered result.

    python -m scripts.catboost_pairwise_062 probe     # time 100 iterations on the first fold
    python -m scripts.catboost_pairwise_062 confirm   # results-only pairs, fast settings
    python -m scripts.catboost_pairwise_062 soft      # soft-target pairs, slower learning rate
    python -m scripts.catboost_pairwise_062 report
    python -m scripts.catboost_pairwise_062 report-soft
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data/tooling/catboost"))
import config  # noqa: E402
from scripts.model_ideas import (OUT, THREADS, Cache, compare, load_arm, print_summary,  # noqa: E402
                                 production_inputs, run_arm, save_json, setup_logging, store_columns)
from scripts.select_features import CONFIRM_QUARTERS  # noqa: E402
from src import feature_selection as fs  # noqa: E402
from src.forecast_evaluation import paired_interval  # noqa: E402
from src.model import RacePredictor  # noqa: E402
from src.race_objectives import exploded_stages, race_bounds, race_softmax  # noqa: E402

STAGE_WEIGHTS = (1.0, 0.25, 0.0625)
# Chosen for speed from timings on the first fold's training rows only: PairLogit's default of 10 Newton
# steps per leaf doubles the time per iteration (0.99 s against 0.42 s on 500k runners), and the higher
# learning rate keeps the early-stopped iteration count low. A slower setting could do better.
PARAMS = {"loss_function": "PairLogit", "eval_metric": "PairLogit", "learning_rate": 0.2, "depth": 6,
          "l2_leaf_reg": 3.0, "leaf_estimation_iterations": 1, "random_seed": 42, "allow_writing_files": False}
MAX_ITERATIONS = 1500
SOFT_PARAMS = {"learning_rate": 0.05}
SOFT_MAX_ITERATIONS = 3000

logger = logging.getLogger("catboost_pairwise_062")


def pairs_for(finish: np.ndarray, race_ids: np.ndarray, weights: np.ndarray,
              soft: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(pairs, pair_weights)``: each placed runner (1st-3rd) against every runner not yet chosen.

    With *soft* (production's half-market, half-result target), the first stage is every ordered pair
    (i, j) in the race weighted by ``soft[i]``: the pairwise form of a softmax on that target.
    """
    groups = fs.race_groups(race_ids)
    chosen = exploded_stages(finish, groups, len(STAGE_WEIGHTS))
    race = np.repeat(np.arange(len(groups)), groups)
    rows = np.arange(len(finish))
    pairs, pair_weights = [], []
    for k, stage_weight in enumerate(STAGE_WEIGHTS):
        if k == 0 and soft is not None:
            size = groups[race]
            first = np.repeat(rows, size)
            race_start = np.repeat(np.r_[0, np.cumsum(groups)[:-1]][race], size)
            other = race_start + np.arange(len(first)) - np.repeat(np.cumsum(size) - size, size)
            keep = (other != first) & (soft[first] > 0)
            pairs.append(np.column_stack([first[keep], other[keep]]))
            pair_weights.append(soft[first[keep]] * weights[first[keep]])
            continue
        winner = chosen[race, k]
        keep = winner >= 0
        for m in range(k + 1):
            keep &= rows != chosen[race, m]
        pairs.append(np.column_stack([winner[keep], rows[keep]]))
        pair_weights.append(stage_weight * weights[keep])
    return np.vstack(pairs), np.concatenate(pair_weights)


def _pool(x, race_ids, finish, weights, lo, hi, soft=None):
    from catboost import Pool
    pairs, pair_weights = pairs_for(finish[lo:hi], race_ids[lo:hi], weights[lo:hi],
                                    None if soft is None else soft[lo:hi])
    return Pool(x[lo:hi], label=np.zeros(hi - lo), group_id=pd.factorize(race_ids[lo:hi])[0], pairs=pairs,
                pairs_weight=pair_weights)


def _scale(scores: np.ndarray, won: np.ndarray, race_ids: np.ndarray) -> float:
    """One multiplier on the scores, by winner race log loss of the race softmax."""
    bounds = race_bounds(race_ids)

    def loss(log_c):
        p = race_softmax(np.exp(log_c) * scores, bounds)
        return -np.sum(won * np.log(np.clip(p, 1e-12, None))) / (len(bounds) - 1)
    return float(np.exp(minimize_scalar(loss, bounds=(-4.0, 3.0), method="bounded").x))


def fit_predict(x_train, x_test, race_ids, race_dates, finish, won, weights, test_race_ids,
                iterations: int | None = None, soft: np.ndarray | None = None, params: dict | None = None,
                max_iterations: int = MAX_ITERATIONS) -> tuple[np.ndarray, dict]:
    from catboost import CatBoostRanker
    params = {**PARAMS, **(params or {})}
    n = len(x_train)
    groups = fs.race_groups(race_ids)
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    end, validation = RacePredictor._early_stop_split(groups, pd.DatetimeIndex(race_dates)[starts],
                                                      purge_days=int(config.PURGE_DAYS))
    started = time.monotonic()
    selected = CatBoostRanker(**params, iterations=iterations or max_iterations, thread_count=THREADS,
                              od_type="Iter", od_wait=int(config.EARLY_STOPPING_ROUNDS), use_best_model=True,
                              verbose=250)
    selected.fit(_pool(x_train, race_ids, finish, weights, 0, end, soft),
                 eval_set=_pool(x_train, race_ids, finish, weights, validation, n, soft))
    rounds = int(selected.get_best_iteration()) + 1
    scale = _scale(selected.predict(x_train[validation:]), won[validation:], race_ids[validation:])
    info = {"rounds": rounds, "scale": scale, "selection_seconds": round(time.monotonic() - started),
            "slab_races": int(len(np.unique(race_ids[validation:])))}
    if iterations is not None:
        return np.empty(0), info
    model = CatBoostRanker(**params, iterations=rounds, thread_count=THREADS, verbose=0)
    model.fit(_pool(x_train, race_ids, finish, weights, 0, n, soft))
    info["total_seconds"] = round(time.monotonic() - started)
    return race_softmax(scale * model.predict(x_test), race_bounds(test_race_ids)), info


def make_predict(cache: Cache, names: list[str], finish: np.ndarray, info_path: Path,
                 soft: np.ndarray | None = None, **fit):
    t = cache.targets
    infos = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}

    def predict(train, test, _capacity):
        p, info = fit_predict(cache.take(train, names), cache.take(test, names), t.race_ids[train],
                              t.race_dates[train], finish[train], t.won[train],
                              fs.recency_weights(t.race_dates[train]), t.race_ids[test],
                              soft=None if soft is None else soft[train], **fit)
        label = str(pd.Period(pd.Timestamp(t.race_dates[test][0]), "Q"))
        infos[label] = info
        save_json(info_path, infos)
        logger.info("%s: %s", label, info)
        return p
    return predict


def probe(_args) -> None:
    cache = Cache()
    names = production_inputs(cache)
    finish = store_columns(["finish_position"]).finish_position.to_numpy(np.float64)
    t = cache.targets
    fold = cache.fold_data(CONFIRM_QUARTERS[:1])[0]
    train = fold.train_rows
    pairs, _ = pairs_for(finish[train], t.race_ids[train], np.ones(len(train)))
    logger.info("%d training runners, %d races, %d pairs", len(train), len(np.unique(t.race_ids[train])),
                len(pairs))
    _, info = fit_predict(cache.take(train, names), None, t.race_ids[train], t.race_dates[train], finish[train],
                          t.won[train], fs.recency_weights(t.race_dates[train]), None, iterations=100)
    print(f"100 iterations: {info['selection_seconds']} s (scale {info['scale']:.3f})")


def confirm(args) -> None:
    cache = Cache()
    finish = store_columns(["finish_position"]).finish_position.to_numpy(np.float64)
    run_arm(cache, "confirm", "catboost_pairwise",
            make_predict(cache, production_inputs(cache), finish, OUT / "catboost_info_062.json"))
    report(args)


def soft(args) -> None:
    cache = Cache()
    finish = store_columns(["finish_position"]).finish_position.to_numpy(np.float64)
    run_arm(cache, "confirm", "catboost_soft",
            make_predict(cache, production_inputs(cache), finish, OUT / "catboost_soft_info_062.json",
                         soft=cache.targets.soft_y, params=SOFT_PARAMS, max_iterations=SOFT_MAX_ITERATIONS))
    report_soft(args)


def _stack_loss(w, logs: np.ndarray, won: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per-race winner log loss of the geometric blend ``p ~ prod(p_k ^ w_k)``."""
    score = logs @ np.asarray(w)
    sizes = np.diff(np.r_[starts, len(score)])
    top = np.maximum.reduceat(score, starts)
    norm = np.log(np.add.reduceat(np.exp(score - np.repeat(top, sizes)), starts)) + top
    return -(np.add.reduceat(won * score, starts) - norm)


def _stack_inputs(frames: list[pd.DataFrame], mask: np.ndarray):
    rows = [f[mask] for f in frames]
    groups = fs.race_groups(rows[0].race_id.to_numpy())
    logs = np.column_stack([np.log(np.clip(fs.race_normalise(r.model.to_numpy(np.float64), groups), 1e-12, None))
                            for r in rows])
    return logs, rows[0].won.to_numpy(np.float64), np.r_[0, np.cumsum(groups)[:-1]]


def stacked(base: list[pd.DataFrame], extra: pd.DataFrame, production: pd.DataFrame) -> dict:
    """*base* plus *extra*, weights fitted on earlier quarters, against *base* alone and against *production*."""
    fold = production.fold.to_numpy()
    deltas: dict[str, list] = {"base": [], "production": []}
    dates, rows = [], []
    for k in range(1, len(CONFIRM_QUARTERS)):
        history, current = np.isin(fold, CONFIRM_QUARTERS[:k]), fold == CONFIRM_QUARTERS[k]

        def score(frames):
            args = _stack_inputs(frames, history)
            start = np.r_[1.0, np.zeros(len(frames) - 1)]
            fitted = minimize(lambda w: _stack_loss(w, *args).mean(), x0=start, method="BFGS").x
            return fitted, _stack_loss(fitted, *_stack_inputs(frames, current))

        weights, loss = score([*base, extra])
        _, base_loss = score(base)
        inputs = _stack_inputs([production], current)
        deltas["base"].append(loss - base_loss)
        deltas["production"].append(loss - _stack_loss([1.0], *inputs))
        dates.append(production[current].race_date.to_numpy()[inputs[2]])
        rows.append({"quarter": CONFIRM_QUARTERS[k], "weights": [round(float(w), 3) for w in weights],
                     "delta_vs_base": float(deltas["base"][-1].mean())})
    blocks = pd.Series(np.concatenate(dates)).astype(str).to_numpy()
    result: dict = {"races": int(len(blocks)), "per_quarter": rows}
    for name, parts in deltas.items():
        delta = np.concatenate(parts)
        result[f"vs_{name}"] = {"delta": float(delta.mean()), "ci95": paired_interval(delta, blocks=blocks)["ci95"],
                                "quarters_better": f"{sum(p.mean() < 0 for p in parts)}/{len(parts)}"}
    return result


def _report(arm: str, summary_name: str, title: str) -> dict:
    from scripts.thurstone_058 import blended
    arms = ["production", "exploded", arm]
    summary = compare("confirm", arms, [(arm, "production"), (arm, "exploded")])
    frames = {a: load_arm("confirm", a) for a in [*arms, "thurstone"]}
    summary["blends"] = {f"{a}+{arm}": blended(frames[a], frames[arm]) for a in ["production", "exploded"]}
    summary["stack"] = stacked([frames["exploded"], frames["thurstone"]], frames[arm], frames["production"])
    save_json(OUT / summary_name, summary)
    print_summary(summary, title)
    for name, b in summary["blends"].items():
        weights = [(round(q["production_weight"], 2), round(q["thurstone_weight"], 2)) for q in b["per_quarter"]]
        print(f"  blend {name} vs first alone: {b['delta_vs_production']:+.5f} "
              f"[{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}] on {b['races']} races, better {b['quarters_better']}, "
              f"weights {weights}")
    stack = summary["stack"]
    for name in ("base", "production"):
        row = stack[f"vs_{name}"]
        print(f"  exploded+figure+{arm} vs {name}: {row['delta']:+.5f} [{row['ci95'][0]:+.5f}, {row['ci95'][1]:+.5f}] "
              f"on {stack['races']} races, better {row['quarters_better']}")
    print(f"  stack weights {[q['weights'] for q in stack['per_quarter']]}")
    return summary


def report(_args) -> dict:
    return _report("catboost_pairwise", "summary_062.json", "062 CatBoost pairwise, results only, fast")


def report_soft(_args) -> dict:
    return _report("catboost_soft", "summary_062_soft.json", "062 CatBoost pairwise, soft target, rate 0.05")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["probe", "confirm", "soft", "report", "report-soft"])
    args = parser.parse_args()
    setup_logging()
    {"probe": probe, "confirm": confirm, "soft": soft, "report": report, "report-soft": report_soft}[args.stage](args)


if __name__ == "__main__":
    main()
