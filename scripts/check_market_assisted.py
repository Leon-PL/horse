"""Time-ordered market correction with a reused December model; no tree fits."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.benchmark_forecasts import align_predictions, score_forecasts
from src.deployment_validation import file_sha256 as sha
from src.featured_dataset import fe_signature
from src.forecast_benchmark import match_market_fields
from src.forecast_evaluation import probability_metrics, valid_race_mask
from src.market_correction_research import apply_correction, fit_correction
from src.model import RacePredictor
from src.run_store import get_production_run_id

CASE = ROOT / "data/audit/market_assisted_036"
KEY = ["race_id", "horse_name", "race_date"]
SOURCES = ["scripts/check_market_assisted.py", "src/market_correction_research.py",
           "src/forecast_benchmark.py", "src/forecast_evaluation.py", "src/model.py",
           "src/probability_calibration.py", "src/betfair.py", "src/utils.py"]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def fresh_pair(frame, path, dates, plan):
    snapshot_meta = json.loads(path.with_suffix(".json").read_text())
    if snapshot_meta["output_sha256"] != sha(path) or snapshot_meta["seconds_before"] != plan["horizon_seconds"]:
        raise ValueError("Market snapshot fingerprint/horizon differs")
    snapshots = pd.read_parquet(path, filters=[("race_date", ">=", dates[0]), ("race_date", "<=", dates[1])])
    if len(snapshots) and (snapshots.race_date.ge(plan["reserved_boundary"]).any()):
        raise ValueError("Reserved dates entered the market assessment")
    bad = (snapshots.definition_ms.gt(snapshots.decision_ms)
           | (snapshots.scheduled_start_ms-snapshots.decision_ms).ne(plan["horizon_seconds"]*1000))
    excluded = set(snapshots.loc[bad, "market_id"])
    snapshots = snapshots.loc[~snapshots.market_id.isin(excluded)].copy()
    paired, coverage = match_market_fields(frame, snapshots)
    runner_age = (paired.decision_ms-paired.price_changed_ms)/1000
    market_age = (paired.decision_ms-paired.observed_ms)/1000
    valid = runner_age.between(0, plan["every_runner_change_max_age_seconds"]) & market_age.between(0, plan["market_observation_max_age_seconds"])
    complete = valid.groupby(paired.race_id).transform("all")
    coverage.update({"future_definition_or_horizon_rejects": len(excluded),
        "fresh_paired_races": int(paired.loc[complete, "race_id"].nunique()),
        "stale_price_or_market_races": int(paired.loc[~complete, "race_id"].nunique())})
    return paired.loc[complete].reset_index(drop=True), coverage


def prepare(plan):
    if (CASE / "preparation.json").exists():
        raise ValueError("Preserve completed source preparation")
    for name, digest in plan["input_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Registered input changed")
    before = {"production": get_production_run_id(), "fe_signature": fe_signature()}
    predictor = RacePredictor(); predictor.load(str(ROOT / plan["model"]))
    if predictor.eval_split_info["as_of"] != plan["model_training_cutoff"]:
        raise ValueError("Use the earlier frozen model, never current production weights")
    if len(predictor.feature_cols) != 154:
        raise ValueError("This experiment is the selected033 recipe")
    columns = list(dict.fromkeys([*predictor.feature_cols, *KEY, "won", "finish_position", "finish_pos_label",
        "num_runners", "handicap", "track", "off_time", "race_type", "horse_prev_races"]))
    store = ROOT / "data/processed/feature_store/featured.parquet"
    frames = {}
    coverages = {}
    replay_error = None
    for role, dates, snapshot_name in (("calibration", plan["calibration_dates"], plan["calibration_snapshots"]),
                                        ("test", plan["test_dates"], plan["test_snapshots"])):
        frame = pd.read_parquet(store, columns=columns,
            filters=[("race_date", ">=", pd.Timestamp(dates[0])), ("race_date", "<=", pd.Timestamp(dates[1]))])
        frame = frame.loc[valid_race_mask(frame)].sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
        if frame.race_date.ge(plan["reserved_boundary"]).any():
            raise ValueError("Reserved outcome dates entered the feature assessment")
        if role == "calibration":
            reference = pd.read_parquet(ROOT / plan["calibration_predictions"])
            labels = KEY + ["won", "finish_position", "num_runners", "handicap"]
            pd.testing.assert_frame_equal(frame[labels], reference[labels], check_dtype=False)
            frame["model_probability"] = reference[plan["model_probability_column"]].to_numpy()
            small = frame.loc[frame.race_id.isin(frame.race_id.drop_duplicates().head(20))]
            replay_error = float(np.max(np.abs(align_predictions(predictor, small)-small.model_probability.to_numpy())))
            if replay_error > 1e-12:
                raise ValueError("Frozen model does not reproduce its registered independent forecasts")
        paired, coverage = fresh_pair(frame, ROOT / snapshot_name, dates, plan)
        if role == "test" and len(paired):
            paired["model_probability"] = align_predictions(predictor, paired)
        keep = list(dict.fromkeys([*KEY, "won", "num_runners", "handicap", "horse_prev_races", "model_probability",
            "market_probability", "market_id", "selection_id", "decision_ms", "observed_ms", "price_changed_ms"]))
        if paired.empty:
            raise ValueError(f"No complete fresh {role} fields; inspect the source coverage")
        paired[keep].to_parquet(CASE / (role+".parquet"), index=False)
        frames[role] = paired
        coverages[role] = coverage
    if (frames["calibration"].race_id.nunique() < plan["minimum_calibration_fields"]
            or frames["test"].race_id.nunique() < plan["minimum_test_fields"]):
        raise ValueError("Insufficient complete fresh fields for the registered assessment")
    after = {"production": get_production_run_id(), "fe_signature": fe_signature()}
    if before != after:
        raise ValueError("The baseline generation changed during preparation")
    report = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "scope": plan["scope"],
        "coverage": coverages, "model_training_as_of": predictor.eval_split_info["as_of"],
        "native_fit": predictor.eval_split_info, "calibration_forecast_replay_error": replay_error,
        "source_sha256": {p: sha(ROOT / p) for p in SOURCES},
        "input_sha256": {name: sha(CASE / name) for name in ("calibration.parquet", "test.parquet", "plan.json")},
        "installed": after, "new_tree_fits": 0,
        "limitations": [plan["source_contract"], "The same December model ages across the assessment; this is not a quarterly refit simulation.",
                        "Existing final-starter features condition the comparison on unchanged fields, not fully as-issued historic cards.",
                        "Previously inspected development dates; no fresh independent confirmation or executable-profit claim."]}
    write(CASE / "preparation.json", report)
    print(json.dumps(report, indent=2), flush=True)


def arrays(frame):
    groups = frame.groupby("race_id", sort=False).size().to_numpy()
    if not np.array_equal(frame.race_id, np.repeat(frame.race_id.drop_duplicates(), groups)):
        raise ValueError("Whole contiguous fields required")
    return frame.model_probability.to_numpy(), frame.market_probability.to_numpy(), groups, frame.won.to_numpy()


def metrics(frame, parameters):
    model, market, groups, won = arrays(frame)
    probability = apply_correction(model, market, groups, parameters["alpha"], parameters["beta"])
    return probability_metrics(probability, won, groups)[0]


def fit(plan):
    if (CASE / "recipe.json").exists():
        raise ValueError("Preserve the frozen pre-test recipe")
    prepared = json.loads((CASE / "preparation.json").read_text())
    for name, digest in prepared["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Research source changed after its input audit")
    for name, digest in prepared["input_sha256"].items():
        if sha(CASE / name) != digest:
            raise ValueError("Frozen correction inputs changed")
    calibration = pd.read_parquet(CASE / "calibration.parquet")
    days = calibration.race_date.dt.normalize()
    unique = sorted(days.unique())
    boundary = pd.Timestamp(unique[int(len(unique)*plan["fit_fraction_dates"])])
    earlier = calibration.loc[days.le(boundary-pd.Timedelta(days=plan["purge_days"]))].copy()
    later = calibration.loc[days.ge(boundary)].copy()
    if earlier.race_id.nunique() < plan["minimum_fit_fields"] or later.race_id.nunique() < plan["minimum_validation_fields"]:
        raise ValueError("Insufficient chronological fitting/validation fields")
    fitted = {"market": {"alpha": 0., "beta": 1.},
        "market_calibrated": fit_correction(*arrays(earlier), model_allowed=False, ridge=plan["ridge_to_market"]),
        "market_combined": fit_correction(*arrays(earlier), model_allowed=True, ridge=plan["ridge_to_market"])}
    validation = {name: metrics(later, parameters) for name, parameters in fitted.items()}
    target_mask = later.groupby("race_id").horse_prev_races.transform("min").ge(5) & later.groupby("race_id").handicap.transform("min").eq(1)
    target = later.loc[target_mask]
    target_metrics = {name: metrics(target, parameters) for name, parameters in fitted.items()} if target.race_id.nunique() >= 50 else {}
    selected = "market"
    for name in ("market_calibrated", "market_combined"):
        overall_ok = (validation[name]["race_log_loss"] < validation[selected]["race_log_loss"]-1e-6
                      and validation[name]["brier"] <= validation["market"]["brier"])
        target_ok = not target_metrics or all(target_metrics[name][k] <= target_metrics["market"][k]
                                              for k in ("race_log_loss", "brier"))
        if overall_ok and target_ok:
            selected = name
    recipe = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selected": selected,
        "parameters": fitted, "fit_first_date": str(earlier.race_date.min().date()),
        "fit_last_date": str(earlier.race_date.max().date()), "fit_fields": int(earlier.race_id.nunique()),
        "validation_first_date": str(later.race_date.min().date()), "validation_last_date": str(later.race_date.max().date()),
        "validation_fields": int(later.race_id.nunique()), "validation_scores": validation,
        "validation_established_handicap_scores": target_metrics, "source_contract": plan["source_contract"]}
    write(CASE / "recipe.json", recipe)
    # Test probabilities/outcomes are not used above this saved decision.
    test = pd.read_parquet(CASE / "test.parquet")
    model, market, groups, won = arrays(test)
    forecasts = {"independent": model, "market": market}
    for name in ("market_calibrated", "market_combined"):
        parameters = fitted[name]
        forecasts[name] = apply_correction(model, market, groups, parameters["alpha"], parameters["beta"])
    forecasts["selected"] = forecasts[selected]
    report = score_forecasts(test, forecasts)
    established = test.groupby("race_id").horse_prev_races.transform("min").ge(5) & test.groupby("race_id").handicap.transform("min").eq(1)
    report["established_handicaps"] = score_forecasts(test.loc[established], {name: values[established.to_numpy()] for name, values in forecasts.items()})
    report.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), selected=selected, recipe=recipe,
        scope=plan["scope"], new_tree_fits=0, production_changed=False,
        limitations=prepared["limitations"], recipe_sha256=sha(CASE / "recipe.json"))
    for name, values in forecasts.items(): test[name+"_probability"] = values
    test.to_parquet(CASE / "test_predictions.parquet", index=False)
    write(CASE / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    plan = json.loads((CASE / "plan.json").read_text())
    prepare(plan) if args.prepare else fit(plan)
