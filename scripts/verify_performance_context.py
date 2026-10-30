"""Verify real pilot context histories after the unknown-reason guard repair."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.performance_context import context_history, FEATURES
from src.deployment_validation import file_sha256

OUTPUT = ROOT/"data/audit/predictive_review_038/context"
KEY = ["race_id","horse_name","race_date"]


def run():
    plan = json.loads((OUTPUT/"plan.json").read_text())
    original = OUTPUT/"performance_context_at_pilot.py"
    if file_sha256(original) != plan["implementation_sha256"]["src/performance_context.py"]:
        raise ValueError("Original pilot implementation was not preserved exactly")
    old = pd.read_parquet(OUTPUT/"pilot_history.parquet")
    names = sorted(old.horse_name.unique())
    columns = [*KEY,"horse_id","off_time","finish_position","finish_pos_label"]
    frame = pd.read_parquet(ROOT/"data/audit/market_transfer_034/source_inputs.parquet",columns=columns,
                            filters=[("horse_name","in",names)])
    source = pd.read_parquet(OUTPUT/"source_observations.parquet",columns=KEY+["ride_description","nonfinish_reason"])
    for part in (frame,source,old):
        part["race_id"] = part.race_id.astype(str)
        part["race_date"] = pd.to_datetime(part.race_date)
    frame = frame.merge(source,on=KEY,how="left",validate="one_to_one")
    full = context_history(frame)
    cut = frame.race_date.le("2025-12-31")
    _,seed = context_history(frame.loc[cut],return_state=True)
    pd.testing.assert_frame_equal(context_history(frame.loc[~cut],state=seed),full.loc[~cut])
    changed = frame.copy()
    future = changed.race_date.ge("2026-02-01")
    changed.loc[future,"ride_description"] = "Hampered, slowly away, mistake"
    changed.loc[future,"nonfinish_reason"] = "PulledUp"
    pd.testing.assert_frame_equal(context_history(changed).loc[~future],full.loc[~future])
    shuffled = frame.sample(frac=1,random_state=42)
    pd.testing.assert_frame_equal(context_history(shuffled).sort_index(),full.sort_index())
    current = pd.concat([frame[KEY],full],axis=1)
    matched = current.merge(old,on=KEY,suffixes=("_new","_old"),validate="one_to_one")
    if len(matched) != len(old):
        raise ValueError("Repaired pilot lost historical identities")
    differences = {c:int((~np.isclose(matched[c+"_new"],matched[c+"_old"],equal_nan=True)).sum()) for c in FEATURES}
    report = {"rows":len(frame),"horses":len(names),"original_source_sha256":file_sha256(original),
        "current_source_sha256":file_sha256(ROOT/"src/performance_context.py"),
        "full_prefix_seed_equal":True,"future_outcome_and_commentary_independent":True,
        "row_permutation_equal":True,"changed_values_from_unknown_reason_guard":differences,
        "scope":"Real-source temporal/representation checks only; no learner fit or accuracy claim"}
    current.to_parquet(OUTPUT/"pilot_history_verified.parquet",index=False)
    (OUTPUT/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    run()
