"""Corrections taught by the market's past prices instead of by winners alone.

    python -m research.error_anatomy.soft_corrections screen --groups NAME ... [--threads 6]

Every correction in the stack learns from win/lose outcomes: one bit per race.
The generalist itself is trained on a 75% market, 25% result target, because a
price says something about every runner. This module gives the corrections the
same teacher. Nothing else changes: a correction still sees only its own
inputs, starts from the stored out-of-fold forecast, is fitted on earlier
quarters only, and is scored on results.

Arms, declared before any was scored:

- ``won-short``: the 9 November recipe (results, rows from 2024Q3).
- ``soft-short``: the same recipe and rows, target 0.75 market + 0.25 result.
- ``soft-long``: soft target, rows from 2022Q3 (forecasts from ``extend_oof.py``).
- ``soft-long-rich``: as ``soft-long`` with more capacity (7 leaves, at least
  1,500 runners per leaf, 300 rounds), on the view that a quieter target can
  support it.

The market probability is the earlier race's own closing price (BSP, else SP)
and is used only as a training label for races before the scored quarter. No
price of a scored race enters its forecast.
"""
import argparse
import time

import numpy as np
import pandas as pd

from research.error_anatomy import entity_residuals as er
from research.error_anatomy.frame import OUT
from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s
from src.ingestion.storage import atomic_json, atomic_parquet

KEYS = s.KEYS
BLEND = .75
RICH = {"learning_rate": .02, "num_leaves": 7, "min_data_in_leaf": 1500, "lambda_l2": 200.,
        "bagging_fraction": .8, "bagging_freq": 1}
ARMS = {
    "won-short": {"window": "short", "target": "won", "params": sc.TIER1["params"], "rounds": 100},
    "soft-short": {"window": "short", "target": "y_soft", "params": sc.TIER1["params"], "rounds": 100},
    "soft-long": {"window": "long", "target": "y_soft", "params": sc.TIER1["params"], "rounds": 100},
    "soft-long-rich": {"window": "long", "target": "y_soft", "params": RICH, "rounds": 300},
}
SEEDS = (7, 8, 9)
TWIN_SEED = 11
ERROR_GROUPS = ("horse", "market-gap", "residual", "wave2")
STORE_GROUPS = ("tracking", "same-day-bias", "ability-lb", "own", "sale", "franking", "horse-state", "entity-recent",
                "horse-state-gap", "horse-state-result", "horse-state-lags", "market-level", "market-moves", "hidden-form", "owner", "newcomer", "foal-age", "foal-age-clean", "pace-comments", "matched-form", "plot", "filter", "market-rating", "debut-choice", "price-1100", "yard", "burden", "trainer-move", "trainer-move-twin", "comeback", "comeback-twin", "price-1100-hours", "price-3600", "price-300")


def soft_boost(columns, *, target, base="p_single", rounds=100, params=None, threads=4, seeds=SEEDS):
    """Fixed-round race-softmax corrections from the reference offset, towards ``target``."""
    import lightgbm as lgb
    info = []
    settings = {**s.BOOST, "num_threads": threads, **(params or {})}

    def fit_predict(train, test):
        started = time.perf_counter()
        x = train[columns].to_numpy(dtype=np.float32, na_value=np.nan)
        x_test = test[columns].to_numpy(dtype=np.float32, na_value=np.nan)
        offset = np.log(np.maximum(train[base].to_numpy(), s.TINY))
        bounds = s.bounds_of(train.race_id.to_numpy())
        data = lgb.Dataset(x, label=train[target].to_numpy(dtype=float), free_raw_data=False)
        correction, gain = np.zeros(len(test)), np.zeros(len(columns))
        for seed in seeds:
            booster = lgb.train({**settings, "seed": seed, "objective": s._softmax_objective(offset, bounds)}, data,
                                num_boost_round=rounds)
            correction += booster.predict(x_test, raw_score=True) / len(seeds)
            gain += booster.feature_importance("gain")
        info.append({"fold": test.quarter.iloc[0], "training_races": len(bounds) - 1,
                     "seconds": round(time.perf_counter() - started, 1),
                     "top_inputs": [columns[i] for i in np.argsort(-gain)[:8] if gain[i] > 0]})
        return correction
    fit_predict.info = info
    return fit_predict


def frames():
    """The short (2024Q3+) and long (2022Q3+) screening frames with the soft target."""
    long = er.long_frame()
    long["quarter"] = long.race_date.dt.to_period("Q").astype(str)
    short = sc.base_frame()
    long = s.attach(long, short[[*KEYS, "p_average"]])
    won = long.won.to_numpy(float)
    market = long.p_market.to_numpy(float)
    long["y_soft"] = np.where(np.isfinite(market), BLEND * market + (1 - BLEND) * won, won)
    # Other market weights for the correction's teacher (declared 15 November; joint_variants blend arms).
    for weight in (.5, .9):
        long[f"y_soft{int(weight * 100)}"] = np.where(np.isfinite(market), weight * market + (1 - weight) * won, won)
    short = s.attach(short, long[[*KEYS, "y_soft", "p_market"]])
    return {"short": short, "long": long}


def group_values(name, frame):
    if name in ERROR_GROUPS:
        values = pd.read_parquet(OUT / "long_inputs.parquet")
        _, groups = er.long_inputs_groups(values)
        return values[[*KEYS, *groups[name]]], groups[name]
    if name == "franking":
        from research.error_anatomy.franking import load
        return load(frame)
    if name.startswith("horse-state") or name == "entity-recent":
        from research.error_anatomy.horse_state import load
        return load(name, frame)
    if name in ("market-level", "market-moves"):
        from research.error_anatomy.market_history import load
        return load(name, frame)
    if name == "filter":
        from research.error_anatomy.horse_filter import load
        return load(frame)
    if name == "market-rating":
        from research.error_anatomy.market_rating import load
        return load(frame)
    if name == "debut-choice":
        from research.error_anatomy.debut_choice import load
        return load(frame)
    if name == "burden":
        from research.error_anatomy.burden import load
        return load(frame)
    if name == "yard":
        from research.error_anatomy.yard_change import load
        return load(frame)
    if name in ("comeback", "comeback-twin"):
        from research.comebacks_099.inputs import load
        return load(frame, twin=name.endswith("twin"))
    if name in ("trainer-move", "trainer-move-twin"):
        from research.trainer_moves_098.correction import load
        return load(frame, twin=name.endswith("twin"))
    if name in ("price-1100-hours", "price-3600", "price-300"):
        from research.error_anatomy.morning_price import load_timed
        return load_timed(frame, when={"price-1100-hours": "1100-hours", "price-3600": "3600s",
                                       "price-300": "300s"}[name])
    if name == "price-1100":
        from research.error_anatomy.morning_price import load
        return load(frame)
    if name == "plot":
        from research.error_anatomy.plot_races import load
        return load(frame)
    if name == "pace-comments":
        from research.error_anatomy.pace_comments import load
        return load(frame)
    if name == "matched-form":
        from research.error_anatomy.matched_form import load
        return load(frame)
    if name == "foal-age-clean":
        from research.error_anatomy.foal_age import load_clean
        return load_clean(frame)
    if name == "foal-age":
        from research.error_anatomy.foal_age import load
        return load(frame)
    if name == "newcomer":
        from research.error_anatomy.newcomer import load
        return load(frame)
    if name == "owner":
        from research.error_anatomy.owner import load
        return load(frame)
    if name == "hidden-form":
        from research.error_anatomy.hidden_form import load
        return load(frame)
    if name == "sale":
        from research.error_anatomy.sale_inputs import load
        return load(frame)
    if name == "own":
        # The generalist's own inputs: a gain here is under-fitting, not new information.
        columns = [c for c in sc.spec_features("full-063") if c not in frame]
        values = pd.read_parquet(s.STUDY / "input.parquet", columns=[*KEYS, *columns],
                                 filters=[("race_date", ">=", frame.race_date.min())])
        values.attrs = {}
        values[columns] = values[columns].astype("float32")
        return values, columns
    values, columns = sc.CANDIDATES[name]()(frame)
    return values[[*KEYS, *columns]], columns


def screen(names, threads, arms):
    windows = frames()
    target = OUT / "soft_corrections.json"
    report = __import__("json").loads(target.read_text()) if target.exists() else {}
    saved_path = OUT / "soft_corrections.parquet"
    scored = windows["short"].loc[windows["short"].quarter.isin(s.SCORED), KEYS].reset_index(drop=True)
    saved = pd.read_parquet(saved_path) if saved_path.exists() else scored.copy()
    for name in names:
        values, columns = group_values(name, windows["long"])
        data = {window: s.attach(frame, values) for window, frame in windows.items()}
        twins = {window: s.permute_within(frame, columns, seed=TWIN_SEED) for window, frame in data.items()}
        entry = report.get(name, {"inputs": len(columns)})
        for arm in arms:
            recipe = ARMS[arm]
            row = {}
            for label, source in (("real", data), ("twin", twins)):
                fit = soft_boost(columns, target=recipe["target"], rounds=recipe["rounds"],
                                 params=recipe["params"], threads=threads)
                frame = source[recipe["window"]]
                correction, _ = s.walk_forward(frame, fit)
                races = s.race_table(frame, correction, bases=("p_single", "p_average"), columns=columns)
                row[label] = races
                if label == "real":
                    entry[arm] = {"net": s.summarise(races, ("p_single", "p_average"))["p_average"],
                                  "top_inputs": fit.info[-1]["top_inputs"]}
                    rows = frame.quarter.isin(s.SCORED).to_numpy()
                    part = frame.loc[rows, KEYS].assign(c=correction[rows])
                    saved[f"{name}|{arm}"] = scored.merge(part, on=KEYS, how="left", validate="one_to_one").c.to_numpy()
            entry[arm]["vs_twin"] = s.versus(row["real"], row["twin"], ("p_average",))["p_average"]
            n, t = entry[arm]["net"], entry[arm]["vs_twin"]
            print(f"{name:<14} {arm:<15} net {n['mean']:+.5f} [{n['ci95'][0]:+.5f}, {n['ci95'][1]:+.5f}] "
                  f"{n['quarters_better']}/4   vs twin {t['mean']:+.5f} [{t['ci95'][0]:+.5f}, {t['ci95'][1]:+.5f}] "
                  f"{t['quarters_better']}/4", flush=True)
            report[name] = entry
            atomic_json(target, report)
            atomic_parquet(saved, saved_path)


def stack(names, threads, arm, tag=""):
    """Corrections of one arm for 2025Q1-2026Q2, combined on the pool with weights from earlier quarters.

    Compared with the 9 November stack (pool plus result-taught tracking and same-day corrections)
    and with the 10 November error stack. Every named group is refitted with ``arm``'s recipe, so
    naming tracking and same-day-bias replaces the old corrections instead of adding to them.
    """
    from research.meta_learner.specialists import CORRECTIONS, QUARTERS, linear_combiner, race_losses
    recipe = ARMS[arm]
    windows = frames()
    frame = windows[recipe["window"]]
    keep = frame.quarter.isin(QUARTERS).to_numpy()
    saved = frame.loc[keep, KEYS].reset_index(drop=True)
    path = OUT / f"soft_stack_{arm}.parquet"
    done = pd.read_parquet(path) if path.exists() else None
    for name in names:
        if done is not None and f"{name}:c" in done:
            saved[f"{name}:c"] = saved.merge(done[[*KEYS, f"{name}:c"]], on=KEYS, validate="one_to_one")[f"{name}:c"]
            continue
        values, columns = group_values(name, windows["long"])
        fit = soft_boost(columns, target=recipe["target"], rounds=recipe["rounds"], params=recipe["params"],
                         threads=threads)
        correction, _ = s.walk_forward(s.attach(frame, values), fit, scored=QUARTERS)
        saved[f"{name}:c"] = correction[keep]
        atomic_parquet(saved, path)
        print(f"{name}: fitted", flush=True)
    atomic_parquet(saved, path)
    new = [f"{name}:c" for name in names]
    pool = s.attach(sc.pool_frame(), pd.read_parquet(CORRECTIONS, columns=[*KEYS, "c_tracking", "c_same-day-bias"]))
    pool = s.attach(pool, saved)
    pool = pool.loc[pool.quarter.isin(QUARTERS)].reset_index(drop=True)
    combiner = linear_combiner(new, base="p_average")
    fitted, _ = s.walk_forward(pool, combiner, scored=s.SCORED)
    later = pool.quarter.isin(s.SCORED).to_numpy()
    scored = pool.loc[later].reset_index(drop=True)
    base = np.log(scored.p_pool.to_numpy())
    old = base + scored.c_tracking.to_numpy() + scored["c_same-day-bias"].to_numpy()
    arms = {"single": np.log(scored.p_single.to_numpy()), "average": np.log(scored.p_average.to_numpy()),
            "stack_9nov": old, "pool_plus_fitted": base + fitted[later],
            "pool_plus_summed": base + scored[new].to_numpy().sum(1)}
    bounds = s.bounds_of(scored.race_id.to_numpy())
    races = scored.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    loss = {k: race_losses(scored, v) for k, v in arms.items()}
    report = {"arm": arm, "groups": list(names), "coefficients": combiner.info,
              "race_log_loss": {k: float(v.mean()) for k, v in loss.items()},
              "fitted_vs": {k: s.paired(loss["pool_plus_fitted"], loss[k], races)
                            for k in ("single", "average", "stack_9nov")},
              "summed_vs_stack_9nov": s.paired(loss["pool_plus_summed"], loss["stack_9nov"], races),
              "evaluation_status": "retrospective screen on reused development quarters"}
    atomic_json(OUT / f"soft_stack_{arm}{tag}.json", report)
    runner = scored[[*KEYS, "race_date", "quarter"]].copy()
    runner["logp"] = s.race_log_softmax(arms["pool_plus_fitted"], bounds)
    atomic_parquet(runner, OUT / f"soft_stack_{arm}{tag}_forecast.parquet")
    print({k: round(v, 5) for k, v in report["race_log_loss"].items()})
    for k, r in report["fitted_vs"].items():
        print(f"fitted vs {k:<11} {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] {r['quarters_better']}/4")
    for row in combiner.info:
        print({k: v for k, v in row.items() if k == "fold" or k.endswith(":c")})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["screen", "stack"])
    parser.add_argument("--arm", default="soft-long-rich", choices=list(ARMS))
    parser.add_argument("--groups", nargs="+", required=True, choices=[*ERROR_GROUPS, *STORE_GROUPS])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--tag", default="", help="suffix for the stack's report and forecast, so a trial does not replace them")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "stack":
        stack(args.groups, args.threads, args.arm, args.tag)
    else:
        screen(args.groups, args.threads, args.arms)


if __name__ == "__main__":
    main()
