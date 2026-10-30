"""A frozen place-head contribution to win forecasts; no new tree fits.

Prepare January–March forecasts, freeze coefficients using purged earlier/later
dates, then evaluate May–10June. Independent of the live installation rollout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.benchmark_independent_features import score_pair
from src.auxiliary_win_research import auxiliary_win_probabilities, fit_auxiliary_win
from src.deployment_validation import file_sha256 as sha
from src.forecast_evaluation import probability_metrics, valid_race_mask
from src.model import RacePredictor

CASE = ROOT / "data/audit/auxiliary_win_041"
KEY = ["race_id", "horse_name", "race_date"]
METADATA = [*KEY, "won", "finish_position", "num_runners", "handicap", "horse_prev_races"]
MODEL = "data/audit/prior_market_feasibility_033/installation/historical_model.joblib"
CALIBRATION = "data/audit/market_transfer_034/baseline/inputs.parquet"
TEST = "data/audit/live_feature_speed_20261029/installation/previous_store/featured.parquet"
TEST_META = "data/audit/live_feature_speed_20261029/installation/previous_store/meta.json"
REFERENCE = "data/audit/prior_market_feasibility_033/installation/historical_predictions.parquet"
MARKET = "data/audit/market_assisted_036/test.parquet"
SOURCES = ["scripts/check_auxiliary_win.py", "src/auxiliary_win_research.py",
    "src/market_correction_research.py", "scripts/benchmark_independent_features.py",
    "src/forecast_evaluation.py", "src/model.py", "src/probability_calibration.py",
    "src/betfair.py", "src/utils.py", "config.py", "tests/test_auxiliary_win_research.py"]


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify(plan):
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Registered source changed: {name}")
    for name, digest in plan["input_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")


def load_model():
    model = RacePredictor()
    model.load(str(ROOT / MODEL))
    if model.eval_split_info.get("as_of") != "2025-12-22" or len(model.feature_cols) != 154 or model.place_model is None:
        raise ValueError("Use the frozen December154-input win/place model")
    for head in (model.clf_model, model.place_model):
        underlying = getattr(head, "reg", head)
        underlying.set_params(n_jobs=2)
    return model


def load_fields(model, path, first, last):
    columns = list(dict.fromkeys([*model.feature_cols, *METADATA]))
    frame = pd.read_parquet(ROOT / path, columns=columns, filters=[
        ("race_date", ">=", pd.Timestamp(first)), ("race_date", "<=", pd.Timestamp(last))])
    if frame.race_date.ge("2026-06-11").any():
        raise ValueError("Reserved outcome dates cannot enter this investigation")
    frame = frame.loc[valid_race_mask(frame)].sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    if frame.duplicated(KEY).any() or frame.empty:
        raise ValueError("Complete unambiguous fields are required")
    return frame


def native_forecasts(model, frame):
    groups = frame.groupby("race_id", sort=False).size().to_numpy()
    if not np.array_equal(frame.race_id, np.repeat(frame.race_id.drop_duplicates(), groups)):
        raise ValueError("Whole fields must be contiguous")
    matrix = model.scaler.transform(frame[model.feature_cols].to_numpy())
    raw_win = model.clf_model.predict_proba(matrix)[:, 1]
    # Deliberately bypass the served place projection: it uses win floors and
    # would create a circular purportedly independent auxiliary channel.
    raw_place = model.place_model.predict_proba(matrix)[:, 1]
    win = model._calibrate_win_probs(raw_win, groups)
    auxiliary_win_probabilities(win, raw_place, groups)  # validates both channels
    return frame[METADATA].assign(win_probability=win, raw_place_probability=raw_place)


def arrays(frame):
    groups = frame.groupby("race_id", sort=False).size().to_numpy()
    if not np.array_equal(frame.race_id, np.repeat(frame.race_id.drop_duplicates(), groups)):
        raise ValueError("Whole contiguous fields required")
    return frame.win_probability.to_numpy(), frame.raw_place_probability.to_numpy(), groups


def target_mask(frame):
    group = frame.groupby("race_id", sort=False)
    return group.horse_prev_races.transform("min").ge(5) & group.handicap.transform("min").eq(1)


def predict(frame, parameters):
    return auxiliary_win_probabilities(*arrays(frame), alpha=parameters["alpha"], beta=parameters["beta"])


def metrics(frame, parameters):
    return probability_metrics(predict(frame, parameters), frame.won.to_numpy(), arrays(frame)[2])[0]


def paired(frame, baseline, candidate):
    return score_pair(frame.assign(baseline_probability=baseline, candidate_probability=candidate))


def prepare():
    if (CASE / "plan.json").exists():
        raise ValueError("Keep the registered procedure; do not repeat preparation")
    CASE.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / TEST_META).read_text())
    if meta["fe_signature"] != "11bf1232d575eea6":
        raise ValueError("Use the immutable pre-migration033/038 feature generation")
    plan = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "The already-trained raw place classifier may carry incremental win-ranking information absent from the win head.",
        "model_cutoff": "2025-12-22", "fit_dates": ["2026-01-01", "2026-02-21"],
        "validation_dates": ["2026-02-28", "2026-03-31"], "test_dates": ["2026-05-01", "2026-06-10"],
        "coefficient_bounds": {"alpha_place": [0, 1], "beta_win": [.5, 1.5]}, "ridge_to_identity": .001,
        "recipe": "softmax(beta*log(existing win probability) + alpha*log(auxiliary place-utility book)); identity alpha0/beta1",
        "validation_rule": "Select auxiliary only if alpha>1e-6 and overall LL beats both unchanged win and fitted win-temperature control by1e-6, Brier no worse than either, and both established-handicap scores no worse than either when at least50target fields. Temperature control cannot be selected alone. Freeze before reading/evaluating May outcomes.",
        "advance_rule": "Selected auxiliary must improve later overall and established-handicap LL with neither runner Brier worse, and overall date95%LL upper bound<0. Only then consider a separate native calibration-transfer check; no automatic installation.",
        "new_tree_fits": 0, "small_coefficient_fits": 2,
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "input_sha256": {name: sha(ROOT / name) for name in [MODEL, CALIBRATION, TEST, TEST_META, REFERENCE, MARKET]},
        "scope": "Previously inspected development; same frozen December heads across fitting/validation/test. No market odds in candidate. No production/store/recorder writes.",
        "limitations": ["Raw place log odds are auxiliary scores, not an exact inversion from place probabilities to win probabilities.",
                        "One frozen model ages across the evaluation; this is not a refitting simulation or fresh confirmation.",
                        "A positive result would still require testing coefficient transfer in the native calibration recipe before deployment."]}
    write(CASE / "plan.json", plan)
    with zipfile.ZipFile(CASE / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.write(ROOT / name, name)
    started = time.monotonic()
    model = load_model()
    frame = load_fields(model, CALIBRATION, "2026-01-01", "2026-03-31")
    forecast = native_forecasts(model, frame)
    reference = pd.read_parquet(ROOT / REFERENCE)
    columns = [*KEY, "won", "finish_position", "num_runners", "handicap"]
    joined = forecast.merge(reference[columns + ["candidate_probability"]], on=columns, how="outer", validate="one_to_one", indicator=True)
    if not joined.pop("_merge").eq("both").all() or frame.race_id.nunique() != 2591:
        raise ValueError("Historical native field/label population differs")
    error = float(np.max(np.abs(joined.win_probability - joined.candidate_probability)))
    if error > 1e-12:
        raise ValueError("Frozen native win forecasts do not replay")
    small = frame.loc[frame.race_id.isin(frame.race_id.drop_duplicates().head(20))]
    first = native_forecasts(model, small)
    poisoned = native_forecasts(model, small.assign(won=0, finish_position=0, odds=np.nan))
    for name in ("win_probability", "raw_place_probability"):
        np.testing.assert_array_equal(first[name], poisoned[name])
    forecast.to_parquet(CASE / "calibration.parquet", index=False)
    verify(plan)
    report = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "model_training_as_of": "2025-12-22",
        "fields": int(forecast.race_id.nunique()), "runners": len(forecast),
        "native_win_replay_max_error": error, "current_price_and_label_poison_max_error": 0.,
        "raw_place_is_unprojected": True, "new_tree_fits": 0,
        "calibration_sha256": sha(CASE / "calibration.parquet"), "plan_sha256": sha(CASE / "plan.json"),
        "elapsed_seconds": time.monotonic() - started}
    write(CASE / "preparation.json", report)
    print(json.dumps(report, indent=2), flush=True)


def fit():
    if (CASE / "recipe.json").exists():
        raise ValueError("The recipe is already frozen")
    plan = json.loads((CASE / "plan.json").read_text())
    verify(plan)
    prepared = json.loads((CASE / "preparation.json").read_text())
    if sha(CASE / "calibration.parquet") != prepared["calibration_sha256"]:
        raise ValueError("Calibration forecasts changed")
    frame = pd.read_parquet(CASE / "calibration.parquet")
    earlier = frame.loc[frame.race_date.between(*plan["fit_dates"])].copy()
    later = frame.loc[frame.race_date.between(*plan["validation_dates"])].copy()
    if earlier.race_id.nunique() < 200 or later.race_id.nunique() < 100:
        raise ValueError("Insufficient dated fitting/validation fields")
    parameters = {"identity": {"alpha": 0., "beta": 1.}}
    for name, allowed in [("temperature_control", False), ("auxiliary", True)]:
        parameters[name] = fit_auxiliary_win(*arrays(earlier), earlier.won.to_numpy(), allow_place=allowed,
                                             ridge=plan["ridge_to_identity"])
    validation = {name: metrics(later, values) for name, values in parameters.items()}
    target = later.loc[target_mask(later)]
    target_metrics = {name: metrics(target, values) for name, values in parameters.items()} if target.race_id.nunique() >= 50 else {}
    auxiliary = validation["auxiliary"]
    qualifies = parameters["auxiliary"]["alpha"] > 1e-6 and all(
        auxiliary["race_log_loss"] < validation[control]["race_log_loss"] - 1e-6
        and auxiliary["brier"] <= validation[control]["brier"]
        for control in ("identity", "temperature_control"))
    if target_metrics:
        qualifies = qualifies and all(target_metrics["auxiliary"][metric] <= target_metrics[control][metric]
            for control in ("identity", "temperature_control") for metric in ("race_log_loss", "brier"))
    recipe = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected": "auxiliary" if qualifies else "identity", "parameters": parameters,
        "fit_fields": int(earlier.race_id.nunique()), "validation_fields": int(later.race_id.nunique()),
        "target_validation_fields": int(target.race_id.nunique()),
        "validation_scores": validation, "validation_target_scores": target_metrics,
        "plan_sha256": sha(CASE / "plan.json"), "test_outcomes_used": False}
    write(CASE / "recipe.json", recipe)
    print(json.dumps(recipe, indent=2), flush=True)


def evaluate():
    if (CASE / "report.json").exists():
        raise ValueError("Preserve the completed readout")
    plan = json.loads((CASE / "plan.json").read_text())
    recipe = json.loads((CASE / "recipe.json").read_text())
    if recipe["plan_sha256"] != sha(CASE / "plan.json"):
        raise ValueError("Selected recipe belongs to another plan")
    verify(plan)
    model = load_model()
    test = native_forecasts(model, load_fields(model, TEST, *plan["test_dates"]))
    if test.race_id.nunique() != 1727:
        raise ValueError("Preserve the complete frozen May–10June field population")
    forecasts = {name: predict(test, values) for name, values in recipe["parameters"].items()}
    forecasts["selected"] = forecasts[recipe["selected"]]
    overall = {name: paired(test, forecasts["identity"], values) for name, values in forecasts.items() if name != "identity"}
    mask = target_mask(test)
    target = {name: paired(test.loc[mask], forecasts["identity"][mask], values[mask])
              for name, values in forecasts.items() if name != "identity"}
    #036 retained these labels but not numeric finishing positions.
    price_labels = [*KEY, "won", "num_runners", "handicap", "horse_prev_races"]
    price = pd.read_parquet(ROOT / MARKET, columns=price_labels + ["market_probability", "model_probability"])
    matching = test.merge(price, on=price_labels, how="inner", validate="one_to_one")
    if len(matching) != len(price) or matching.race_id.nunique() != 1351:
        raise ValueError("Exact existing fresh-market field/label overlap changed")
    np.testing.assert_allclose(matching.win_probability, matching.model_probability, atol=1e-12, rtol=0)
    market_reports = {name: paired(matching, matching.market_probability.to_numpy(), predict(matching, values))
                      for name, values in recipe["parameters"].items()}
    candidate = overall["selected"]
    selected_target = target["selected"]
    point = all(s["candidate"]["race_log_loss"] < s["baseline"]["race_log_loss"] and
                s["candidate"]["brier"] <= s["baseline"]["brier"] for s in (candidate, selected_target))
    precise = candidate["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]["ci95"][1] < 0
    for name, values in forecasts.items():
        test[name + "_probability"] = values
    test.to_parquet(CASE / "test_predictions.parquet", index=False)
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "selected": recipe["selected"],
        "overall": overall, "established_handicaps": target,
        "auxiliary_vs_temperature_control": paired(test, forecasts["temperature_control"], forecasts["auxiliary"]),
        "matched_market": market_reports, "new_tree_fits": 0,
        "advance_to_native_calibration_transfer_check": bool(recipe["selected"] == "auxiliary" and point and precise),
        "recipe_sha256": sha(CASE / "recipe.json"), "scope": plan["scope"],
        "production_writes": False, "limitations": plan["limitations"] + [
            "Market readout reuses036's fresh same-field BetfairT-5 LTP sample; not executable quotes."]}
    verify(plan)
    write(CASE / "report.json", report)
    print(json.dumps({"selected": report["selected"], "overall": report["overall"]["selected"],
        "target": report["established_handicaps"]["selected"],
        "advance": report["advance_to_native_calibration_transfer_check"], "new_tree_fits": 0}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "fit", "evaluate"])
    {"prepare": prepare, "fit": fit, "evaluate": evaluate}[parser.parse_args().action]()
