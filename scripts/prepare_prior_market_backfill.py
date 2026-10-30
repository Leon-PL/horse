"""Prepare only the three new columns/state; never alter the normal store.

Run while the native proposal is being checked. Adoption remains conditional
on that check. Existing033 projection/evidence stays immutable.
"""
from pathlib import Path
import json
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.deployment_validation import file_sha256
from src.featured_dataset import fe_signature
from src.feature_store import FeatureStore
from src.prior_market_research import FEATURES, prior_market_features

OUTPUT=ROOT / "data/audit/prior_market_feasibility_033/backfill"
KEY=["race_id","horse_name","race_date"]


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    store=FeatureStore()
    signature=fe_signature()
    meta=store.meta()
    if meta["fe_signature"] != signature:
        raise ValueError("Finish the032 ordinary store first")
    began=time.monotonic()
    columns=KEY+["horse_id","race_name","race_type","surface","won","finish_position","finish_pos_label",
        "num_runners","odds","official_rating","has_official_rating"]
    f=pd.read_parquet(store.featured_path,columns=columns)
    added,state=prior_market_features(f,return_state=True)
    values=pd.concat([f[KEY],added],axis=1)
    original=pd.read_parquet(ROOT / "data/audit/prior_market_feasibility_033/projection.parquet")
    paired=values.loc[values.race_date.le("2026-03-31")].merge(original,on=KEY,validate="one_to_one",suffixes=("_full","_screen"))
    if len(paired)!=len(original):
        raise ValueError("Full backfill has different historical runners")
    for c in FEATURES:
        np.testing.assert_array_equal(paired[c+"_full"].to_numpy(dtype=np.float32),paired[c+"_screen"].to_numpy(dtype=np.float32))
    if max(map(len,state["history"].values()),default=0)>3 or state["last_date"]!=meta["last_date"]:
        raise ValueError("Unexpected carried-history bounds/date")
    values.to_parquet(OUTPUT / "features.parquet",index=False)
    with (OUTPUT / "state.pkl").open("wb") as stream:
        pickle.dump(state,stream,protocol=pickle.HIGHEST_PROTOCOL)
    if fe_signature()!=signature:
        raise ValueError("Feature sources changed while preparing the backfill")
    report={"scope":"Prepared additions only; no ordinary source/store/model change or automatic adoption",
        "base_fe_signature":signature,"base_store_meta":meta,"runners":len(values),"features":list(FEATURES),
        "screen_prefix_float32_exact":True,"prefix_runners_checked":len(paired),"state_last_date":state["last_date"],
        "helper_sha256":file_sha256(ROOT / "src/prior_market_research.py"),
        "script_sha256":file_sha256(Path(__file__)),"base_featured_sha256":file_sha256(store.featured_path),
        "features_sha256":file_sha256(OUTPUT / "features.parquet"),"state_sha256":file_sha256(OUTPUT / "state.pkl"),
        "elapsed_seconds":time.monotonic()-began}
    (OUTPUT / "manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("runners","prefix_runners_checked","screen_prefix_float32_exact","state_last_date","elapsed_seconds")},indent=2),flush=True)


if __name__=="__main__":
    main()
