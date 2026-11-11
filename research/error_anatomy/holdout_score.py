"""Registration and the single scoring of the hold-out read (see ``holdout.py``)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from research.error_anatomy import entity_residuals as er
from research.error_anatomy import soft_corrections as soft
from research.error_anatomy.frame import OUT, race_normalised
from research.error_anatomy.franking import franking_inputs
from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s
from research.meta_learner.specialists import linear_combiner, race_losses
from src.ingestion.storage import atomic_json, atomic_parquet

ROOT = OUT / "holdout"
KEYS = s.KEYS
ARM = "soft-long-rich"
PRIMARY = ("horse", "market-gap")
SECONDARY = ("horse", "market-gap", "residual", "wave2", "franking", "same-day-bias", "tracking")
COMBINER_FROM = "2025Q1"          # corrections are made out of fold from here so weights have history
TEST_QUARTERS = ("2026Q2", "2026Q3", "2026Q4")
CODE = ("entity_residuals.py", "soft_corrections.py", "franking.py", "holdout.py", "holdout_score.py")


def holdout_start():
    return pd.Timestamp(config.GOLDEN_HOLDOUT_START)


def code_digests():
    folder = Path(__file__).parent
    return {name: hashlib.sha256((folder / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in CODE}


def protocol():
    return {
        "hypothesis": "Corrections built from the generalist's own past errors improve its race log loss on "
                      "races from the hold-out start that no choice in this study has seen.",
        "test_races": f"Every race from {holdout_start().date()} to the store's last date with a complete settled "
                      "field and a seed-42 forecast for every runner.",
        "base": "Full 063, seed 42, screen capacity, purged quarterly refits (the development protocol).",
        "inputs": "entity_residuals.long_inputs and franking.franking_inputs on out-of-fold forecasts from 2022Q3; "
                  "same-day-bias and tracking from screen_candidates.",
        "corrections": {"arm": ARM, "recipe": {k: v for k, v in soft.ARMS[ARM].items()}, "target_blend": soft.BLEND,
                        "seeds": list(soft.SEEDS), "fitted": "walk-forward, earlier quarters only, 7-day purge"},
        "weights": f"One coefficient per correction, race logit on the seed-42 forecast, fitted on corrections "
                   f"from {COMBINER_FROM} up to the quarter before the scored one.",
        "primary": {"groups": list(SECONDARY),
                    "statistic": "mean over test races of race log loss(base + weighted corrections) minus race "
                                 "log loss(base)",
                    "pass": "one-sided p < 0.01 for a mean below zero, standard error clustered by race date"},
        "secondary": {"groups": list(PRIMARY), "why": "the pair whose gain carried over in full to the "
                      "native-capacity forecast; same statistic, reported with its own p"},
        "descriptive": ["each group alone at unit weight", "by quarter", "by race type",
                        "gap to Betfair SP before and after"],
        "development_estimates": {"rehearsal_2026-01-01_to_2026-06-03": {"all_seven": -0.01146, "pair": -0.00475},
                                  "pair_on_native_2025Q3": -0.00355,
                                  "note": "chosen on these quarters, so the hold-out effect should be smaller"},
        "one_read": "score() refuses to run if receipt.json exists.",
        "code_sha256": code_digests(),
    }


def register():
    if (ROOT / "receipt.json").exists():
        raise RuntimeError("The hold-out has already been scored")
    target = ROOT / "registration.json"
    value = protocol()
    if target.exists() and json.loads(target.read_text()) != value:
        raise RuntimeError("A different registration already exists; remove nothing, investigate")
    atomic_json(target, value)
    print(json.dumps(value, indent=1))


def full_frame(development_only=False):
    """Every runner with a seed-42 out-of-fold forecast from 2022Q3 to the store's last date.

    With ``development_only`` nothing fitted on or forecast for a hold-out date is read.
    """
    early = pd.read_parquet(OUT / "early_oof_seed42.parquet")[[*KEYS, "probability"]]
    cached = s.cached_forecasts(s.STUDY, "full-063")[[*KEYS, "probability"]]
    parts, agreement = [early, cached], None
    if not development_only:
        late = pd.read_parquet(ROOT / "forecasts_seed42.parquet")[[*KEYS, "probability"]]
        late["race_id"] = late.race_id.astype(str)
        overlap = cached.merge(late, on=KEYS, suffixes=("_dev", "_new"))
        if len(overlap):
            agreement = float(np.abs(np.log(overlap.probability_dev) - np.log(overlap.probability_new)).max())
        parts.append(late.loc[~late.set_index(KEYS).index.isin(cached.set_index(KEYS).index)])
    forecasts = pd.concat(parts, ignore_index=True).rename(columns={"probability": "p_single"})
    labels = pd.read_parquet(ROOT / "input.parquet", columns=[*s.LABELS, "bsp", "odds"],
                             filters=[("race_date", ">=", pd.Timestamp("2022-07-01"))])
    labels.attrs = {}
    frame = labels.merge(forecasts, on=KEYS, how="inner", validate="one_to_one")
    size = frame.groupby("race_id").race_id.transform("size")
    frame = frame.loc[size.eq(frame.num_runners)]
    frame, _ = s.ordered_frame(frame)
    store = sc.read_store(["trainer", "jockey", "sire", "race_type", "days_since_last_run", "track", "going",
                           "damsire", "handicap", "surface", "distance_furlongs"],
                          start=frame.race_date.min()).drop(columns="race_date")
    frame = s.attach(frame, store)
    for column in ("bsp", "odds"):
        price = pd.to_numeric(frame[column], errors="coerce")
        whole = (price > 1).groupby(frame.race_id).transform("all")
        frame[f"_p_{column}"] = np.where(whole, race_normalised(1 / price.where(price > 1), frame.race_id), np.nan)
    frame["p_bsp"] = frame._p_bsp
    frame["p_market"] = frame._p_bsp.fillna(frame._p_odds)
    frame = frame.drop(columns=["_p_bsp", "_p_odds"])
    frame["quarter"] = frame.race_date.dt.to_period("Q").astype(str)
    won = frame.won.to_numpy(float)
    market = frame.p_market.to_numpy(float)
    frame["y_soft"] = np.where(np.isfinite(market), soft.BLEND * market + (1 - soft.BLEND) * won, won)
    return frame, agreement


def group_inputs(frame):
    values, groups = er.long_inputs(frame)
    history = sc.read_store(["finish_position", "won"])
    franking = franking_inputs(history, frame, market="p_market")
    franking = franking.loc[franking.race_id.isin(frame.race_id.unique())]
    groups["franking"] = [c for c in franking if c not in KEYS]
    franking["horse_id"] = franking.horse_id.astype(values.horse_id.dtype)
    values = values.merge(franking, on=KEYS, how="left", validate="one_to_one")
    for name in ("same-day-bias", "tracking"):
        part, columns = sc.CANDIDATES[name]()(frame)
        part = part[[*KEYS, *columns]].copy()
        part[columns] = part[columns].astype("float32")
        values = values.merge(part, on=KEYS, how="left", validate="one_to_one")
        groups[name] = columns
    return values, groups


def clustered_test(change, dates):
    """Mean, date-clustered standard error and one-sided p for a mean below zero."""
    from scipy.stats import norm
    table = pd.DataFrame({"d": change, "date": np.asarray(dates)})
    by_date = table.groupby("date").d.agg(["sum", "size"])
    n, mean = len(table), table.d.mean()
    scores = by_date["sum"] - mean * by_date["size"]
    clusters = len(by_date)
    se = float(np.sqrt(clusters / (clusters - 1) * (scores ** 2).sum()) / n)
    return {"mean": float(mean), "se": se, "z": float(mean / se), "p_one_sided": float(norm.cdf(mean / se)),
            "races": int(n), "dates": int(clusters)}


REHEARSAL_START = pd.Timestamp("2026-01-01")
REHEARSAL_QUARTERS = ("2026Q1", "2026Q2")


def rehearse(threads=6):
    """The whole scoring path on development races only (2026Q1 to 3 June 2026). Reads no hold-out race."""
    score(threads, rehearsal=True)


def score(threads=6, rehearsal=False):
    test_start, test_quarters, target = holdout_start(), TEST_QUARTERS, ROOT / "receipt.json"
    if rehearsal:
        test_start, test_quarters, target = REHEARSAL_START, REHEARSAL_QUARTERS, ROOT / "rehearsal.json"
    else:
        registration = json.loads((ROOT / "registration.json").read_text())
        if registration["code_sha256"] != code_digests():
            raise RuntimeError("Code changed since registration; register again before scoring")
        if target.exists():
            raise RuntimeError("The hold-out has already been scored")
    frame, agreement = full_frame(development_only=rehearsal)
    values, groups = group_inputs(frame)
    frame = s.attach(frame, values)
    quarters = sorted(q for q in frame.quarter.unique() if q >= COMBINER_FROM)
    recipe = soft.ARMS[ARM]
    for name in SECONDARY:
        fit = soft.soft_boost(groups[name], target=recipe["target"], rounds=recipe["rounds"],
                              params=recipe["params"], threads=threads)
        correction, _ = s.walk_forward(frame, fit, scored=quarters)
        frame[f"{name}:c"] = correction
        print(f"{name}: corrections fitted", flush=True)
    work = frame.loc[frame.quarter.isin(quarters)].reset_index(drop=True)
    fitted, weights = {}, {}
    for label, names in (("primary", PRIMARY), ("all", SECONDARY)):
        combiner = linear_combiner([f"{n}:c" for n in names], base="p_single")
        fitted[label] = s.walk_forward(work, combiner, scored=test_quarters)[0]
        weights[label] = combiner.info
    work["fitted_primary"], work["fitted_all"] = fitted["primary"], fitted["all"]
    test = work.loc[work.race_date.ge(test_start)].reset_index(drop=True)
    if not np.isfinite(test[["fitted_primary", "fitted_all"]].to_numpy()).all():
        raise ValueError("A test runner has no fitted correction")
    atomic_parquet(test[[*KEYS, "race_date", "quarter", "p_single", "p_bsp", "won", "fitted_primary", "fitted_all",
                         *[f"{n}:c" for n in SECONDARY]]],
                   ROOT / ("rehearsal_runners.parquet" if rehearsal else "scored_runners.parquet"))
    bounds = s.bounds_of(test.race_id.to_numpy())
    races = test.iloc[bounds[:-1]][["race_id", "race_date", "quarter", "race_type"]].reset_index(drop=True)
    base = np.log(test.p_single.to_numpy())
    loss = {"base": race_losses(test, base), "primary": race_losses(test, base + test.fitted_primary.to_numpy()),
            "all": race_losses(test, base + test.fitted_all.to_numpy())}
    receipt = {"registration_sha256": None if rehearsal else hashlib.sha256(
                   (ROOT / "registration.json").read_bytes()).hexdigest(),
               "rehearsal": rehearsal, "first_date": str(test.race_date.min().date()), "last_date": str(test.race_date.max().date()),
               "refit_agreement_max_log_ratio": agreement,
               "race_log_loss": {k: float(v.mean()) for k, v in loss.items()},
               "primary_all_seven": clustered_test(loss["all"] - loss["base"], races.race_date),
               "secondary_pair": clustered_test(loss["primary"] - loss["base"], races.race_date),
               "weights": weights, "each_group_unit_weight": {}, "by_quarter": {}, "by_race_type": {}}
    receipt["primary_all_seven"]["passes"] = bool(receipt["primary_all_seven"]["p_one_sided"] < 0.01)
    for name in SECONDARY:
        alone = race_losses(test, base + test[f"{name}:c"].to_numpy())
        receipt["each_group_unit_weight"][name] = clustered_test(alone - loss["base"], races.race_date)
    for key, column in (("by_quarter", "quarter"), ("by_race_type", "race_type")):
        for value, rows in races.groupby(column).indices.items():
            receipt[key][str(value)] = {"races": int(len(rows)),
                                        "pair": float((loss["primary"] - loss["base"])[rows].mean()),
                                        "all_seven": float((loss["all"] - loss["base"])[rows].mean())}
    priced = pd.Series(test.p_bsp.notna().to_numpy()).groupby(test.race_id.to_numpy(), sort=False).all().to_numpy()
    market = race_losses(test.assign(p_bsp=test.p_bsp.fillna(1.)), np.log(test.p_bsp.fillna(1.).to_numpy()))
    receipt["gap_to_bsp"] = {"races": int(priced.sum()), "bsp": float(market[priced].mean()),
                             **{k: float(v[priced].mean() - market[priced].mean()) for k, v in loss.items()}}
    atomic_json(target, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "weights"}, indent=1))
