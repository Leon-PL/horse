"""One frozen comparison of audited surface inputs against the completed016 model.

Uses the native model/folds/scoring, with an explicitly versioned research
projection. It does not fabricate a compatible live feature-store generation
or promote its models. Only one native fit runs at a time.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import config
from scripts import benchmark_model_repair as native
from scripts.benchmark_independent_features import canonical_hash,META
from scripts.replay_market_snapshots import digest
from src.featured_dataset import fe_signature
from src.forecast_evaluation import valid_race_mask
from src.wf_eval import development_mask,expanding_period_folds

EXTRA_SOURCES=["src/audited_surface_features.py","scripts/benchmark_surface_repair.py",
    "scripts/build_surface_repair_projection.py","tests/test_audited_surface_features.py"]
native.SOURCES=list(dict.fromkeys([*native.SOURCES,*EXTRA_SOURCES]))
OUTPUT=ROOT/"data/audit/model_improvement/017_surface_family"
REFERENCE=ROOT/"data/audit/model_improvement/016_market_teacher"


def prepare():
    OUTPUT.mkdir(exist_ok=True)
    if (OUTPUT/"manifest.json").exists():
        raise ValueError("Already frozen; run without --prepare to resume")
    prior=json.loads((REFERENCE/"manifest.json").read_text())
    completion=json.loads((REFERENCE/"completion.json").read_text())
    for name,expected in completion["evidence_sha256"].items():
        if digest(REFERENCE/name)!=expected:raise ValueError(f"Reference evidence changed: {name}")
    research=ROOT/"data/audit/feature_review_017"
    receipt=json.loads((research/"surface_projection_receipt.json").read_text())
    for name,expected in receipt["source_sha256"].items():
        if digest(ROOT/name)!=expected:raise ValueError(f"Audited family source changed: {name}")
    if digest(research/"surface_features.parquet")!=receipt["projection_sha256"]:
        raise ValueError("Surface research projection changed")
    if digest(REFERENCE/"inputs.parquet")!=prior["inputs_sha256"]:
        raise ValueError("Reference projection changed")
    features=[*prior["features"],"race_is_synthetic","sire_aw_prior_runs_log1p"]
    spec={"experiment":"017 corrected AW pedigree plus current surface and support",
        "registered_at_utc":datetime.now(timezone.utc).isoformat(),
        "reference_directory":"data/audit/model_improvement/016_market_teacher",
        "reference_variant":"candidate","baseline_features":prior["features"],"candidate_features":features,
        "candidate_periods":[f["period"] for f in prior["folds"]],"market_teacher":True,"market_teacher_blend":.5,
        "place_validation_periods":["2026Q1"],"changed_existing_features":["sire_aw_edge"],
        "added_features":["race_is_synthetic","sire_aw_prior_runs_log1p"],
        "advancement_rule":"All3periods; paired LL upper95<0, Brier/binaryLL no worse, >=2quarters improve versus best016; no subgroup selection or test tuning",
        "scope":"Retrospective development; only inspected source/data validity before registering this family. Live FE/state integration and full serving parity required before any deployment."}
    native.write(OUTPUT/"registration.json",spec)
    frame=pd.read_parquet(REFERENCE/"inputs.parquet")
    overlay=pd.read_parquet(research/"surface_features.parquet",columns=["race_id","horse_name","race_date",
        "sire_aw_edge","race_is_synthetic","sire_aw_prior_runs_log1p"])
    keys=["race_id","horse_name","race_date"]
    original=frame[keys].copy()
    frame=frame.drop(columns="sire_aw_edge").merge(overlay,on=keys,how="left",sort=False,validate="one_to_one")
    if not original.equals(frame[keys]) or frame.race_is_synthetic.isna().any():
        raise ValueError("Projection merge lost or reordered an audited runner")
    frame=frame[list(dict.fromkeys([*features,*[c for c in frame if c not in features]]))]
    frame[features]=frame[features].astype(np.float32)
    temporary=OUTPUT/"inputs.parquet.tmp";frame.to_parquet(temporary,index=False,compression="zstd")
    temporary.replace(OUTPUT/"inputs.parquet")
    eligible=frame.loc[valid_race_mask(frame)&development_mask(frame.race_date)].reset_index(drop=True)
    folds={f.label:f for f in expanding_period_folds(eligible.race_date)}
    bounds=[{"period":p,"cutoff":str(eligible.loc[folds[p].train,"race_date"].max())[:10],
        "test_start":str(eligible.loc[folds[p].test,"race_date"].min())[:10],
        "test_end":str(eligible.loc[folds[p].test,"race_date"].max())[:10]} for p in spec["candidate_periods"]]
    if bounds!=prior["folds"]:raise ValueError("Chronological folds changed")
    audit={"passed":True,"kind":"Composed: original151 invariance checks plus audited repaired-family causality/native equivalence",
        "original_audit_sha256":prior["invariance_sha256"],"real_native_prefix_rows":receipt["real_native_prefix_rows"],
        "native_prefix_columns":receipt["native_prefix_columns_matched"],
        "family_test_log_sha256":digest(ROOT/"data/audit/feature-repairs-tests-017-final.log"),
        "selection_did_not_use_reserved_outcomes":True}
    native.write(OUTPUT/"input_invariance.json",audit)
    params=prior["params"]
    if dict(config.CLASSIFIER_PARAMS)!=params["classifier"]:raise ValueError("Win parameters changed")
    recipe,cfg=native.configured_recipe(spec,params)
    manifest={"registration":spec,"registration_sha256":digest(OUTPUT/"registration.json"),
        "source_sha256":native.source_hashes(),"params":params,"folds":bounds,"features":features,
        "baseline_features":prior["features"],"inputs_sha256":digest(OUTPUT/"inputs.parquet"),
        "base_feature_store_signature":fe_signature(),
        "candidate_feature_signature":canonical_hash({"base":fe_signature(),"overlay":receipt["projection_sha256"],"sources":receipt["source_sha256"]}),
        "dataset_start":prior["dataset_start"],"invariance_sha256":digest(OUTPUT/"input_invariance.json"),
        "reference_manifest_sha256":digest(REFERENCE/"manifest.json"),
        "reference_predictions_sha256":digest(REFERENCE/"paired_predictions.parquet"),"reference_variant":"candidate",
        "reference_models_sha256":{p:digest(REFERENCE/f"{p}_candidate.joblib") for p in spec["candidate_periods"]},
        "calibration":prior["calibration"],"recency_half_life_days":prior["recency_half_life_days"],
        "seed":42,"learner_threads":2,"place_validation_periods":spec["place_validation_periods"],
        "native_training_recipe":recipe,"native_training_config":cfg,
        "market_teacher_inputs_sha256":prior["market_teacher_inputs_sha256"],"secondary_references":{},
        "overlay_receipt_sha256":digest(research/"surface_projection_receipt.json"),
        "scope":spec["scope"],"live_compatible":False}
    native.write(OUTPUT/"manifest.json",manifest)
    with zipfile.ZipFile(OUTPUT/"source_snapshot.zip","x",compression=zipfile.ZIP_DEFLATED) as z:
        for name in native.SOURCES:z.write(ROOT/name,name)
    print(json.dumps({"stage":"registered","features":len(features),"manifest_sha256":canonical_hash(manifest)}),flush=True)
    del frame,eligible,overlay;gc.collect()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args()
    if args.prepare:return prepare()
    registration=OUTPUT/"registration.json"
    spec=json.loads(registration.read_text())
    for period in spec["candidate_periods"]:
        native.fit_fold(period,registration=str(registration))
    native.summarise(str(registration))


if __name__=="__main__":main()
