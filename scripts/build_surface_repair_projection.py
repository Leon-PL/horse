"""Freeze a causal research projection of the audited surface repair."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.audit_selected_feature_semantics import sha
from src.audited_surface_features import pedigree_surface_history, synthetic_surface, MODEL_FEATURES
from src.feature_engineer import _add_result_history_flags, add_pedigree_features
from src.featured_dataset import fe_signature


def main():
    root=ROOT/"data/audit/feature_review_017"
    store=ROOT/"data/audit/model_improvement/015_nonfinisher_history/candidate_store"
    destination=root/"surface_features.parquet"
    if destination.exists():
        raise ValueError("Do not overwrite a frozen surface projection")
    columns=["race_id","horse_name","race_date","sire","damsire","surface","finish_position",
             "finish_pos_label","won","num_runners","going","distance_furlongs"]
    history=pd.read_parquet(store/"processed.parquet",columns=columns)
    # Native-stage equivalence on real data, including unknown parents.
    prefix=history.loc[history.race_date.le(history.race_date.min()+pd.Timedelta(days=10))].copy()
    independent=pedigree_surface_history(prefix,prefix)
    canonical=prefix.copy()
    canonical.loc[synthetic_surface(canonical.surface).eq(1),"surface"]="AW"
    native=add_pedigree_features(_add_result_history_flags(canonical))
    checked=[]
    for entity in ("sire","damsire"):
        for suffix in ("aw_win_rate_shrunk","aw_place_rate_shrunk","aw_edge"):
            name=f"{entity}_{suffix}"
            np.testing.assert_allclose(native[name],independent[name],atol=1e-12,rtol=0,equal_nan=True)
            checked.append(name)
    values=pedigree_surface_history(history,history)
    old=pd.read_parquet(store/"featured.parquet",columns=["sire_aw_edge"])
    changed=~np.isclose(values.sire_aw_edge.to_numpy(np.float32),old.sire_aw_edge.to_numpy(np.float32),atol=0,rtol=0,equal_nan=True)
    output=pd.concat([history[["race_id","horse_name","race_date"]],values],axis=1)
    temporary=destination.with_suffix(".parquet.tmp")
    output.to_parquet(temporary,index=False,compression="zstd");temporary.replace(destination)
    receipt={"rows":len(output),"races":int(output.race_id.nunique()),
        "changed_existing_sire_aw_edge_rows":int(changed.sum()),
        "synthetic_rows":int(values.race_is_synthetic.eq(1).sum()),
        "sire_aw_support_max":float(values.sire_aw_prior_runs.max()),
        "real_native_prefix_rows":len(prefix),"native_prefix_columns_matched":checked,
        "feature_store_fe_signature_unchanged":fe_signature(),
        "candidate_features":list(MODEL_FEATURES),
        "source_sha256":{n:sha(ROOT/n) for n in ("src/audited_surface_features.py","src/feature_engineer.py",
            "src/result_status.py","scripts/build_surface_repair_projection.py","tests/test_audited_surface_features.py")},
        "processed_sha256":sha(store/"processed.parquet"),"projection_sha256":sha(destination),
        "scope":"Research only; no active feature/store/model changes. Exact prior-day AW history plus current-surface context and prior-sample support; outcome validity only, no model score selection."}
    (root/"surface_projection_receipt.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
