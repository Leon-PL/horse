"""Complete038's readout from saved fits; repair cohort metadata only.

The original procedure and source capsule remain unchanged. Its two native
heads and fitted maps completed before a missing reporting-only experience
column stopped the readout. This reader never fits or changes a coefficient.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from scripts.benchmark_recent_probability import OUTPUT, REFERENCE, INPUT, write
from src.deployment_validation import file_sha256
from src.forecast_evaluation import probability_metrics, paired_interval, valid_race_mask
from src.model import RacePredictor

KEY = ["race_id", "horse_name", "race_date"]
COHORT = ROOT/"data/audit/prior_market_feasibility_033/native_check/predictions.parquet"


def align_cohort(frame, reference):
    frame, reference = frame.copy(), reference.copy()
    for part in (frame,reference):
        part["race_id"] = part.race_id.astype(str)
        part["race_date"] = pd.to_datetime(part.race_date)
    labels = KEY+["won", "finish_position", "num_runners", "handicap"]
    joined = frame.merge(reference[labels+["horse_prev_races", "candidate_probability"]], on=labels,
                         how="outer", validate="one_to_one", indicator=True)
    if not joined.pop("_merge").eq("both").all():
        raise ValueError("Saved cohort metadata does not match every original field/label")
    if not valid_race_mask(joined).all():
        raise ValueError("Cohort join changed a complete field")
    return joined.sort_values(KEY,kind="stable").reset_index(drop=True)


def run():
    plan = json.loads((OUTPUT/"plan.json").read_text())
    for name,digest in {**plan["source_sha256"],**plan["input_sha256"]}.items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError("Original038 frozen procedure or inputs changed")
    artifacts = [OUTPUT/(role+"_recent_base.joblib") for role in ("win","place")]+[OUTPUT/"maps.joblib",COHORT]
    receipt = {"scope":"Reporting-only cohort join repair; no new fits/maps or altered advancement rule",
        "original_plan_sha256":file_sha256(OUTPUT/"plan.json"),
        "reader_sha256":file_sha256(Path(__file__)),
        "artifacts_sha256":{str(p.relative_to(ROOT)):file_sha256(p) for p in artifacts}}
    path = OUTPUT/"readout_repair.json"
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError("Readout repair definitions changed")
    if not path.exists():
        write(path,receipt)
    model = RacePredictor()
    model.load(str(REFERENCE))
    maps = joblib.load(OUTPUT/"maps.joblib")
    frame = pd.read_parquet(INPUT, filters=[("race_date",">=",pd.Timestamp(plan["test"][0])),
                                           ("race_date","<=",pd.Timestamp(plan["test"][1]))])
    missing = [c for c in model.feature_cols if c not in frame]
    if missing:
        extra = pd.read_parquet(ROOT/"data/audit/prior_market_feasibility_033/projection.parquet",columns=KEY+missing)
        frame = frame.merge(extra,on=KEY,validate="one_to_one")
    frame = frame.loc[valid_race_mask(frame)].copy()
    cohort = pd.read_parquet(COHORT)
    frame = align_cohort(frame,cohort)
    fields = frame.groupby("race_id",sort=False)
    groups = fields.size().to_numpy()
    days = fields.race_date.first().to_numpy()
    hcap = fields.handicap.first().to_numpy().astype(bool)
    places = np.minimum(groups,np.where(groups<=4,3,np.where(groups<=7,2,np.where((groups>=16)&hcap,4,3))))
    codes = fields.race_type.first().astype(str).to_numpy()
    target = fields.horse_prev_races.min().ge(5).to_numpy()&hcap
    matrix = model.scaler.transform(frame[model.feature_cols].to_numpy())
    raw_win,raw_place = model.clf_model.predict_proba(matrix)[:,1],model.place_model.predict_proba(matrix)[:,1]
    win = model._calibrate_win_probs(raw_win,groups)
    error = float(np.max(np.abs(win-frame.candidate_probability.to_numpy())))
    if error > 1e-12:
        raise ValueError(f"Original154-input forecasts do not replay: {error}")
    candidate_win = maps["win"].predict(raw_win,groups)
    place = model._compute_place_probabilities(frame,None,win,groups,raw_place_probs=raw_place)
    candidate_place = maps["place"].predict(raw_place,win,groups,places,codes)
    won = frame.won.to_numpy()
    placed = ((frame.finish_position.to_numpy()>=1)&(frame.finish_position.to_numpy()<=np.repeat(places,groups))).astype(int)
    def score(mask):
        row = np.repeat(mask,groups)
        b,bl,_ = probability_metrics(win[row],won[row],groups[mask])
        c,cl,_ = probability_metrics(candidate_win[row],won[row],groups[mask])
        return {"baseline":b,"candidate":c,"date_LL_delta_ci95":paired_interval(cl-bl,blocks=days[mask])}
    def place_score(mask):
        row = np.repeat(mask,groups)
        delta = (candidate_place[row]-placed[row])**2-(place[row]-placed[row])**2
        sizes = groups[mask]
        starts = np.r_[0,np.cumsum(sizes)[:-1]]
        mean = float(delta.mean())
        influence = (np.add.reduceat(delta,starts)-mean*sizes)/sizes.mean()
        interval = paired_interval(influence,blocks=days[mask])["ci95"]
        return {"baseline_brier":float(np.mean((place[row]-placed[row])**2)),
            "candidate_brier":float(np.mean((candidate_place[row]-placed[row])**2)),
            "delta":mean,"date_delta_ci95":[float(x+mean) for x in interval]}
    overall = np.ones(len(groups),bool)
    win_report = {"overall":score(overall),"established_handicaps":score(target)}
    place_report = {"overall":place_score(overall),"established_handicaps":place_score(target)}
    win_pass = all(v["candidate"]["race_log_loss"]<v["baseline"]["race_log_loss"]
        and v["candidate"]["brier"]<=v["baseline"]["brier"] for v in win_report.values()) and win_report["overall"]["date_LL_delta_ci95"]["ci95"][1]<0
    place_pass = all(v["delta"]<=0 for v in place_report.values()) and place_report["overall"]["date_delta_ci95"][1]<0
    frame[KEY+["won","finish_position","num_runners","handicap"]].assign(
        baseline_win=win,candidate_win=candidate_win,baseline_place=place,candidate_place=candidate_place).to_parquet(OUTPUT/"predictions.parquet",index=False)
    report = {"completed_at":datetime.now(timezone.utc).isoformat(),"win":win_report,"place":place_report,
        "win_selected":bool(win_pass),"place_selected":bool(place_pass),"baseline_replay_max_error":error,
        "win_map":asdict(maps["win"]),"place_map":{**asdict(maps["place"]),"deficits":{str(k):v for k,v in maps["place"].deficits.items()}},
        "races":len(groups),"runners":len(frame),"target_races":int(target.sum()),
        "calibration_first":"2025-09-24","calibration_last":"2025-12-22","base_last":"2025-09-17",
        "tree_fits":2,"production_changed":False,"fresh_confirmation":False,"readout_repair":receipt}
    write(OUTPUT/"report.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    run()
