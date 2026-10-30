"""Qualify038's place layer and fit its current calibration; preserve win trees.

Default saves a candidate. --promote publishes that already verified candidate.
No existing run, source capsule or historical forecast is overwritten.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
import config
from src.deployment_validation import file_sha256 as sha, training_recipe, write_validation_report
from src.featured_dataset import fe_signature, data_signature
from src.feature_store import FeatureStore
from src.forecast_evaluation import valid_race_mask
from src.model import RacePredictor, prepare_multi_target_data
from src.probability_calibration import PlaceProbabilityRefinement, fit_recent_place_refinement
from src.run_store import save_run, load_run_meta, get_production_run_id, promote as publish, RUNS_DIR
from src.training import TrainingConfig

OUTPUT = ROOT/"data/audit/predictive_review_038/installation_v4"
RESEARCH = OUTPUT.parent/"probability"
PARENT = "20261029_085433_011356"
HISTORICAL = "20261029_073522_043953"
INPUT = ROOT/"data/audit/prior_market_feasibility_033/installation/corrected_inputs.parquet"
KEY = ["race_id","horse_name","race_date"]


def write(path,value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,default=str,allow_nan=False),encoding="utf-8")
    temp.replace(path)


def load_model(identifier):
    p = RacePredictor()
    p.load(str(Path(RUNS_DIR)/identifier/"model.joblib"))
    p._autotune_njobs = 2
    return p


def run(promote=False):
    started = time.monotonic()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    report = json.loads((RESEARCH/"report.json").read_text())
    if not report["place_selected"] or report["win_selected"] or report["baseline_replay_max_error"] != 0:
        raise ValueError("038 did not qualify this place-only update")
    if fe_signature() != "11bf1232d575eea6":
        raise ValueError("Preserve the installed feature generation for a place-only change")
    cfg = TrainingConfig(refresh_data=False,threads=2)
    current_input = OUTPUT/"current_inputs.parquet"
    parent_metadata = load_run_meta(PARENT)
    cutoff = parent_metadata["training_config"]["split"]["training_last_date"]
    parent_model = load_model(PARENT)
    if not current_input.exists():
        store = FeatureStore()
        import pyarrow.parquet as pq
        with store._access():
            store_meta = store.meta()
            processed = pd.read_parquet(store.processed_path)
            prefix = data_signature(processed,cutoff=cutoff)
            if prefix != parent_metadata["training_config"]["signatures"]["data_signature"]:
                raise ValueError("Current history prefix differs from the parent model's actual training data")
            del processed
            available = set(pq.ParquetFile(store.featured_path).schema.names)
            metadata = ["race_id","horse_name","race_date","won","finish_position","num_runners","handicap",
                        "race_type","track","off_time","odds","finish_pos_label","lengths_behind","beaten_lengths"]
            columns = list(dict.fromkeys(parent_model.feature_cols+[c for c in metadata if c in available]))
            current = pd.read_parquet(store.featured_path,columns=columns,filters=[("race_date","<=",pd.Timestamp(cutoff))])
        current.to_parquet(current_input,index=False)
        write(OUTPUT/"current_input_receipt.json",{"store_meta":store_meta,"parent_training_prefix":prefix,
            "cutoff":cutoff,"last_date":str(pd.to_datetime(current.race_date).max())[:10],"rows":len(current),
            "sha256":sha(current_input)})
    else:
        current = pd.read_parquet(current_input)
    current.race_date = pd.to_datetime(current.race_date)
    if str(current.race_date.max())[:10] != cutoff:
        raise ValueError("Current calibration snapshot does not reach the parent's settled cutoff")
    current = current.loc[valid_race_mask(current)].copy()
    current = current[parent_model.feature_cols+[c for c in current if c not in parent_model.feature_cols]]
    plan = {"parent":PARENT,"historical":HISTORICAL,"recipe":training_recipe(cfg),
        "scope":"Place refinement only; final win/place estimators and win calibration reused unchanged",
        "evidence_sha256":{str(p.relative_to(ROOT)):sha(p) for p in (
            RESEARCH/"report.json",RESEARCH/"predictions.parquet",RESEARCH/"maps.joblib",RESEARCH/"plan.json",
            Path(RUNS_DIR)/PARENT/"model.joblib",Path(RUNS_DIR)/HISTORICAL/"model.joblib",INPUT,current_input,OUTPUT/"current_input_receipt.json")},
        "controller_sha256":sha(Path(__file__))}
    manifest = OUTPUT/"manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != plan:
            raise ValueError("Registered place installation inputs/recipe changed")
    else:
        if get_production_run_id() != PARENT:
            raise ValueError("Production parent changed before038 installation")
        write(manifest,plan)
        with zipfile.ZipFile(OUTPUT/"source_snapshot.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
            for name in [*plan["recipe"]["source_sha256"],"scripts/adopt_recent_place.py","config.py"]:
                archive.write(ROOT/name,name)
    status_path = OUTPUT/"status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    previous = OUTPUT.parent/"installation_v3"/"status.json"
    if not status.get("qualification_run") and previous.exists():
        earlier = json.loads(previous.read_text())
        qualification = earlier.get("qualification_run")
        if qualification and load_run_meta(qualification)["training_config"]["recipe"] == plan["recipe"]:
            status.update(stage="qualified",qualification_run=qualification,
                qualification_reused_from=str(previous),reason="The historical qualification is valid; current calibration now uses the verified live-store prefix")
            write(status_path,status)
    historic = load_model(HISTORICAL)
    frame = pd.read_parquet(INPUT)
    missing = [c for c in historic.feature_cols if c not in frame]
    if missing:
        projection = pd.read_parquet(ROOT/"data/audit/prior_market_feasibility_033/projection.parquet",columns=KEY+missing)
        frame = frame.merge(projection,on=KEY,validate="one_to_one")
    frame = frame.loc[valid_race_mask(frame)].copy()
    frame.race_date = pd.to_datetime(frame.race_date)
    frame = frame[historic.feature_cols+[c for c in frame if c not in historic.feature_cols]]
    if not status.get("qualification_run"):
        prototype = joblib.load(RESEARCH/"maps.joblib")["place"]
        historic.place_refinement = PlaceProbabilityRefinement(**asdict(prototype))
        test = frame.loc[frame.race_date.between("2026-01-01","2026-03-31")].copy()
        test["race_id"] = test.race_id.astype(str)
        test = test.sort_values(KEY,kind="stable")
        actual = historic.predict_races(test)[KEY+["win_probability","place_probability"]]
        expected = pd.read_parquet(RESEARCH/"predictions.parquet")
        joined = actual.merge(expected,on=KEY,validate="one_to_one")
        if len(joined) != len(test) or len(joined) != 22824:
            raise ValueError("Qualification forecast join lost runners")
        np.testing.assert_array_equal(joined.win_probability,joined.baseline_win)
        np.testing.assert_array_equal(joined.place_probability,joined.candidate_place)
        parent_meta = load_run_meta(HISTORICAL)
        metrics = deepcopy(parent_meta["metrics"])
        ordered = test.sort_values(["race_date","race_id","horse_name"],kind="stable")
        sizes = ordered.groupby("race_id",sort=False).size().to_numpy()
        raw_place = historic.place_model.predict_proba(historic.scaler.transform(ordered[historic.feature_cols].to_numpy()))[:,1]
        raw_win = historic.clf_model.predict_proba(historic.scaler.transform(ordered[historic.feature_cols].to_numpy()))[:,1]
        win = historic._calibrate_win_probs(raw_win,sizes)
        place = historic._compute_place_probabilities(ordered,None,win,sizes,raw_place_probs=raw_place)
        metrics["place_classifier"] = historic._evaluate_place_model(place,raw_place,ordered,sizes)
        metrics["place_classifier"]["brier_calibrated"] = report["place"]["overall"]["candidate_brier"]
        historical_tc = deepcopy(parent_meta["training_config"])
        historical_tc.update(recipe=plan["recipe"],config=cfg.to_dict(),
            split={**historical_tc.get("split",{}),"mode":"historical_recipe_validation","holdout_races":2591,
                   "training_last_date":"2025-12-22","holdout_first_date":"2026-01-01","holdout_last_date":"2026-03-31"},
            evidence={"038_report":str(RESEARCH/"report.json"),"historical_win_max_change":0.,
                      "canonical_place_replay_max_error":0.,"fresh_confirmation":False})
        historical_tc["evals"] = {"calibration":metrics,"scope":"Previously inspected Q1 development; place refinement qualified, win unchanged"}
        historic.eval_split_info.update(mode="historical_recipe_validation",holdout_races=2591)
        historic.eval_split_info.setdefault("calibration",{})["place_refinement"] = {
            "method":"recent_projected","first_date":"2025-09-24","last_date":"2025-12-22","base_last_date":"2025-09-17"}
        qualification = save_run(predictor=historic,name="038 qualified recent place refinement; win unchanged",
            model_type="race_predictor",data_source="frozen033historical_fields",data_rows=parent_meta["data_rows"],
            n_features=154,elapsed_seconds=0,metrics=metrics,training_config=historical_tc)
        status.update(stage="qualified",qualification_run=qualification)
        write(status_path,status)
    certificate = OUTPUT/"validation.json"
    if not certificate.exists():
        write_validation_report(status["qualification_run"],certificate,evidence_paths=[manifest,RESEARCH/"report.json",RESEARCH/"predictions.parquet",OUTPUT/"source_snapshot.zip"])
    if not status.get("candidate_run"):
        parent = load_model(PARENT)
        parent_metadata = load_run_meta(PARENT)
        cutoff = parent.eval_split_info["training_last_date"]
        allowed = current.loc[current.race_date.le(cutoff)].copy()
        data = prepare_multi_target_data(allowed,dataset_start=allowed.race_date.min(),training_only=True)
        if data["feature_cols"] != parent.feature_cols:
            raise ValueError("Current calibration feature order changed")
        parent_scope = parent_metadata["training_config"]["split"]
        if (len(data["X_train"]) != parent_scope["training_runners"]
                or len(data["groups_train"]) != parent_scope["training_races"]
                or str(data["train_df"].race_date.max())[:10] != cutoff):
            raise ValueError("Current calibration rows/fields/cutoff differ from the parent training scope")
        groups = data["groups_train"]
        starts = np.r_[0,np.cumsum(groups)[:-1]]
        rows = data["train_df"]
        teacher = data["win_teacher_train"]
        soft = np.where(teacher>0,config.WIN_SOFT_LABEL_BLEND*teacher+(1-config.WIN_SOFT_LABEL_BLEND)*data["y_train_won"],data["y_train_won"])
        write(status_path,{**status,"stage":"fitting_current_calibration","base_model":PARENT})
        parent.place_refinement,detail = fit_recent_place_refinement(parent,data["X_train"],groups,
            data["y_train_won"],data["y_train_placed"],data["fp_train"],
            pd.DatetimeIndex(pd.to_datetime(rows.race_date).to_numpy()[starts]),soft_y=soft,
            race_types=rows.race_type.to_numpy() if "race_type" in rows else None,
            handicap=rows.handicap.to_numpy() if "handicap" in rows else None,
            checkpoint_directory=OUTPUT/"components")
        if parent.place_refinement is None:
            raise ValueError("Current place refinement has insufficient dated history")
        parent.eval_split_info.setdefault("calibration",{})["place_refinement"] = detail
        parent.eval_split_info["place_refinement_parent"] = PARENT
        parent.eval_split_info["performance_claim"] = "None for this current-data calibration; see separate historical qualification"
        tc = deepcopy(parent_metadata["training_config"])
        tc.update(recipe=plan["recipe"],split=parent.eval_split_info,validation_reference={
            "validation_run_id":status["qualification_run"],"report_path":str(certificate),"report_sha256":sha(certificate)},
            evals={"scope":"Current-data place calibration update; no new held-out scores"})
        candidate = save_run(predictor=parent,name="038 recent place refinement; original033win and final trees",
            model_type="race_predictor",data_source="feature_store",data_rows=parent_metadata["data_rows"],
            n_features=154,elapsed_seconds=time.monotonic()-started,metrics={},training_config=tc)
        status.update(stage="saved_candidate",candidate_run=candidate,calibration=detail)
        write(status_path,status)
    candidate = load_model(status["candidate_run"])
    original = load_model(PARENT)
    # Recent rows are operational checks, never retrospective accuracy evidence
    # for the October-trained model. Use full actual fields and both interfaces.
    sample = current.loc[current.race_date.ge(current.race_date.max()-pd.Timedelta(days=2))].copy()
    a,b = candidate.predict_races(sample),original.predict_races(sample)
    aligned = a.merge(b[KEY+["win_probability"]],on=KEY,suffixes=("","_parent"),validate="one_to_one")
    np.testing.assert_array_equal(aligned.win_probability,aligned.win_probability_parent)
    from scripts.verify_live_upgrade import check_outputs
    check_outputs(a,candidate)
    maximum = 0.
    for identity in sample.race_id.drop_duplicates().head(5):
        single = candidate.predict_race(sample.loc[sample.race_id.eq(identity)]).set_index("horse_name")
        batch = a.loc[a.race_id.eq(identity)].set_index("horse_name").reindex(single.index)
        difference = float(np.max(np.abs(single.place_probability-batch.place_probability)))
        maximum = max(maximum,difference)
        np.testing.assert_allclose(single.place_probability,batch.place_probability,atol=1e-12,rtol=0)
    if training_recipe(cfg) != plan["recipe"]:
        raise ValueError("Place recipe changed during calibration")
    status.update(stage="verified_candidate",recent_fields=int(sample.race_id.nunique()),
        win_max_change=0.,single_batch_place_max_error=maximum,model_sha256=sha(Path(RUNS_DIR)/status["candidate_run"]/"model.joblib"))
    if promote:
        if get_production_run_id() not in {PARENT,status["candidate_run"]} or not publish(status["candidate_run"]):
            raise ValueError("Promotion refused for this qualified place-only update")
        status["stage"] = "installed"
        status["installed_at"] = datetime.now(timezone.utc).isoformat()
    write(status_path,status)
    print(json.dumps(status,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote",action="store_true")
    args = parser.parse_args()
    run(args.promote)
