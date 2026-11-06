"""One date-local win/place calibration comparison; never promotes a model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import config
from src.deployment_validation import file_sha256
from src.forecast_evaluation import probability_metrics, paired_interval, valid_race_mask
from src.model import RacePredictor, prepare_multi_target_data
from src.recent_probability import calendar_split, SmoothWinMap, RecentPlaceMap

OUTPUT = ROOT / "data/audit/predictive_review_038/probability"
REFERENCE = ROOT / "data/runs/20261029_073522_043953/model.joblib"
INPUT = ROOT / "data/audit/prior_market_feasibility_033/installation/corrected_inputs.parquet"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = ["src/model.py", "src/recent_probability.py", "scripts/benchmark_recent_probability.py", "config.py"]
    plan = {"schema": 1, "historical_cutoff": "2025-12-22", "test": ["2026-01-01", "2026-03-31"],
        "calibration_days": 90, "purge_days": 7, "tree_fits": 2,
        "win_trees": config.CLASSIFIER_PARAMS["n_estimators"],
        "place_trees": config.PLACE_CLASSIFIER_PARAMS["n_estimators"],
        "candidate": "Recent purged90-day smooth race-log-loss win map; smooth place map, learned shrinkage and missing slots",
        "selection": "Win: lower overall/established-handicap LL, no worse Brier, overall date95% upper LLdelta<0. Place: overall/target Brier no worse, overall date95% upper delta<0.",
        "scope": "Previously inspected development; no fresh confirmation, final-estimator refit or automatic promotion",
        "source_sha256": {f: file_sha256(ROOT/f) for f in sources},
        "input_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in (REFERENCE, INPUT)}}
    registered = OUTPUT / "plan.json"
    if registered.exists():
        if json.loads(registered.read_text()) != plan:
            raise ValueError("Registered procedure or inputs changed; preserve the earlier experiment")
    else:
        write(registered, plan)
        with zipfile.ZipFile(OUTPUT/"source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
            for f in sources:
                z.write(ROOT/f, f)
    started = time.monotonic()
    reference = RacePredictor()
    reference.load(str(REFERENCE))
    frame = pd.read_parquet(INPUT)
    # The installation input already contains the same154 selected columns.
    if not set(reference.feature_cols).issubset(frame):
        projection = pd.read_parquet(ROOT/"data/audit/prior_market_feasibility_033/projection.parquet",
            columns=["race_id", "horse_name", "race_date", *[c for c in reference.feature_cols if c not in frame]])
        frame = frame.merge(projection, on=["race_id", "horse_name", "race_date"], validate="one_to_one")
    frame = frame.loc[valid_race_mask(frame)].copy()
    frame["race_date"] = pd.to_datetime(frame.race_date)
    frame = frame.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    history = frame.loc[frame.race_date.le(plan["historical_cutoff"])].copy()
    before, cal_mask = calendar_split(history.race_date, days=90, purge_days=7)
    calibration = history.loc[cal_mask].copy()
    base = history.loc[before].copy()
    # Force the exact saved feature order; feature subsampling makes it material.
    base = base[reference.feature_cols+[c for c in base if c not in reference.feature_cols]]
    data = prepare_multi_target_data(base, dataset_start=frame.race_date.min(), training_only=True)
    if data["feature_cols"] != reference.feature_cols:
        raise ValueError("Calibration learner input order differs from the frozen model")
    learner = RacePredictor()
    learner._autotune_njobs = 2
    learner.feature_cols, learner.scaler = data["feature_cols"], data["scaler"]
    for role, fn, target in (("win", learner._train_win_classifier, "won"),
                              ("place", learner._train_place_classifier, "placed")):
        destination = OUTPUT/(role+"_recent_base.joblib")
        if destination.exists():
            model = joblib.load(destination)
        else:
            print(json.dumps({"stage": "fit_recent_"+role, "base_through": str(base.race_date.max())[:10],
                              "calibration_first": str(calibration.race_date.min())[:10], "rows": len(data["X_train"])}), flush=True)
            kw = {"sample_weight": data["sample_weight_train"]}
            if role == "win":
                blend = config.WIN_SOFT_LABEL_BLEND
                teacher = data["win_teacher_train"]
                kw["soft_y"] = np.where(teacher > 0, blend*teacher+(1-blend)*data["y_train_won"], data["y_train_won"])
                kw["objective"] = "cross_entropy"  # the frozen reference model's learner
            model = fn(data["X_train"], data["y_train_"+target], **kw)
            joblib.dump(model, destination)
        setattr(learner, "clf_model" if role == "win" else "place_model", model)
    raw = learner.scaler.transform(calibration[learner.feature_cols].to_numpy())
    groups = calibration.groupby("race_id", sort=False).size().to_numpy()
    raw_win, raw_place = learner.clf_model.predict_proba(raw)[:,1], learner.place_model.predict_proba(raw)[:,1]
    win_map = SmoothWinMap.fit(raw_win, calibration.won, groups)
    def context(part):
        fields = part.groupby("race_id", sort=False)
        sizes = fields.size().to_numpy()
        hcap = fields.handicap.first().to_numpy().astype(bool)
        places = np.where(sizes <= 4, 3, np.where(sizes <= 7, 2, np.where((sizes >= 16)&hcap, 4, 3)))
        return sizes, np.minimum(places, sizes), fields.race_type.first().astype(str).to_numpy()
    groups, places, codes = context(calibration)
    # Place refinement is isolated from the changed win map for attribution.
    cal_win = reference._calibrate_win_probs(raw_win, groups)
    placed = ((calibration.finish_position.to_numpy() >= 1)
              & (calibration.finish_position.to_numpy() <= np.repeat(places, groups))).astype(int)
    place_map = RecentPlaceMap.fit(raw_place, cal_win, placed, calibration.finish_position, groups, places, codes)
    joblib.dump({"win": win_map, "place": place_map}, OUTPUT/"maps.joblib")
    test = frame.loc[frame.race_date.between(*plan["test"])].copy()
    groups, places, codes = context(test)
    matrix = reference.scaler.transform(test[reference.feature_cols].to_numpy())
    raw_win, raw_place = reference.clf_model.predict_proba(matrix)[:,1], reference.place_model.predict_proba(matrix)[:,1]
    baseline_win = reference._calibrate_win_probs(raw_win, groups)
    candidate_win = win_map.predict(raw_win, groups)
    baseline_place = reference._compute_place_probabilities(test, None, baseline_win, groups, raw_place_probs=raw_place)
    candidate_place = place_map.predict(raw_place, baseline_win, groups, places, codes)
    won = test.won.to_numpy()
    placed = ((test.finish_position.to_numpy() >= 1)&(test.finish_position.to_numpy() <= np.repeat(places, groups))).astype(int)
    fields = test.groupby("race_id", sort=False)
    days = fields.race_date.first().to_numpy()
    target_race = fields.horse_prev_races.min().ge(5).to_numpy() & fields.handicap.first().eq(1).to_numpy()
    target_row = np.repeat(target_race, groups)
    def win_scores(mask):
        row = np.repeat(mask, groups)
        b, bl, _ = probability_metrics(baseline_win[row], won[row], groups[mask])
        c, cl, _ = probability_metrics(candidate_win[row], won[row], groups[mask])
        return {"baseline": b, "candidate": c, "date_LL_delta_ci95": paired_interval(cl-bl, blocks=days[mask])}
    def place_scores(mask):
        row = np.repeat(mask, groups)
        delta = (candidate_place[row]-placed[row])**2-(baseline_place[row]-placed[row])**2
        sizes = groups[mask]
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        totals = np.add.reduceat(delta, starts)
        # Ratio-estimator influence preserves runner-weighted Brier under date clustering.
        mean = float(delta.mean())
        influence = (totals-mean*sizes)/sizes.mean()
        ci = paired_interval(influence, blocks=days[mask])["ci95"]
        return {"baseline_brier": float(np.mean((baseline_place[row]-placed[row])**2)),
                "candidate_brier": float(np.mean((candidate_place[row]-placed[row])**2)),
                "delta": mean, "date_delta_ci95": [float(x+mean) for x in ci]}
    all_races = np.ones(len(groups), bool)
    win = {"overall": win_scores(all_races), "established_handicaps": win_scores(target_race)}
    place = {"overall": place_scores(all_races), "established_handicaps": place_scores(target_race)}
    win_pass = all(v["candidate"]["race_log_loss"] < v["baseline"]["race_log_loss"] and v["candidate"]["brier"] <= v["baseline"]["brier"] for v in win.values()) and win["overall"]["date_LL_delta_ci95"]["ci95"][1] < 0
    place_pass = all(v["delta"] <= 0 for v in place.values()) and place["overall"]["date_delta_ci95"][1] < 0
    for name, digest in plan["source_sha256"].items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError("Registered probability procedure changed during fitting")
    test[["race_id", "horse_name", "race_date", "won", "finish_position", "num_runners", "handicap"]].assign(
        baseline_win=baseline_win, candidate_win=candidate_win, baseline_place=baseline_place,
        candidate_place=candidate_place).to_parquet(OUTPUT/"predictions.parquet", index=False)
    report = {"completed_at": datetime.now(timezone.utc).isoformat(), "win": win, "place": place,
        "win_selected": bool(win_pass), "place_selected": bool(place_pass),
        "win_map": asdict(win_map), "place_map": {**asdict(place_map), "deficits": {str(k): v for k,v in place_map.deficits.items()}},
        "calibration_first": str(calibration.race_date.min())[:10], "calibration_last": str(calibration.race_date.max())[:10],
        "base_last": str(base.race_date.max())[:10], "races": len(groups), "elapsed_seconds": time.monotonic()-started,
        "production_changed": False, "fresh_confirmation": False}
    write(OUTPUT/"report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
