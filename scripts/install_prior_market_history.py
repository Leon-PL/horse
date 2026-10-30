"""Adopt the passed033 family by extending the proved-unchanged032 store.

Append only independent columns/state, verify native generation on a real
prefix, then prepare the existing canonical qualification/deployment controller.
Never repeat the base history build or the completed native win estimators.
"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import ast
import gc
import json
import os
import pickle
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import config
from scripts.integrate_canonical_repairs import verified_historical_cache_scope
from src.deployment_validation import file_sha256 as sha
from src.feature_engineer import _PRIOR_MARKET_STATE_KEY, engineer_features
from src.featured_dataset import _FE_SOURCE_FILES, fe_signature, _model_load_columns
from src.feature_store import FeatureStore
from src.prior_market_research import FEATURES
from src.run_store import get_production_run_id
from src.training import TrainingConfig

BASE=ROOT / "data/audit/trip_suitability_032/installation"
RESEARCH=ROOT / "data/audit/prior_market_feasibility_033"
BACKFILL=RESEARCH / "backfill"
OUTPUT=RESEARCH / "installation"
KEY=["race_id","horse_name","race_date"]
PARENT="20261028_212411_292221"


def write(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2),encoding="utf-8")
    temporary.replace(path)


def prove_additive_sources():
    old=ast.parse((BACKFILL / "base_sources/feature_engineer.py").read_text(encoding="utf-8"))
    new=ast.parse((ROOT / "src/feature_engineer.py").read_text(encoding="utf-8"))
    removed=0
    class StripNewStage(ast.NodeTransformer):
        def visit_List(self,node):
            nonlocal removed
            kept=[e for e in node.elts if not (isinstance(e,ast.Name) and e.id=="add_prior_market_features")]
            removed+=len(node.elts)-len(kept)
            node.elts=kept
            return self.generic_visit(node)
    new.body=[n for n in new.body if not (
        isinstance(n,ast.FunctionDef) and n.name=="add_prior_market_features") and not (
        isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_PRIOR_MARKET_STATE_KEY" for t in n.targets))]
    new=StripNewStage().visit(new)
    if removed!=1 or ast.dump(old,include_attributes=False)!=ast.dump(new,include_attributes=False):
        raise ValueError("Existing feature code changed beyond the independent appended stage")
    old=ast.parse((BACKFILL / "base_sources/featured_dataset.py").read_text(encoding="utf-8"))
    new=ast.parse((ROOT / "src/featured_dataset.py").read_text(encoding="utf-8"))
    def strip_registry(tree):
        values=[]; kept=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_FE_SOURCE_FILES" for t in node.targets):
                values=list(ast.literal_eval(node.value))
            else:
                kept.append(node)
        tree.body=kept
        return values,ast.dump(tree,include_attributes=False)
    old_files,old_code=strip_registry(old);new_files,new_code=strip_registry(new)
    if old_code!=new_code or set(new_files)!=set(old_files)|{"src/prior_market_research.py","src/forecast_evaluation.py"}:
        raise ValueError("Feature registry changed beyond the new stage's actual dependencies")
    previous=json.loads((BASE / "rebuild_status.json").read_text())["source_sha256"]
    for name,digest in previous.items():
        if name not in ("src/feature_engineer.py","src/featured_dataset.py") and sha(ROOT/name)!=digest:
            raise ValueError(f"Existing source changed: {name}")


def check_native_prefix(store,features):
    prefix=pd.read_parquet(store.processed_path,filters=[("race_date","<=",pd.Timestamp("2020-09-20"))])
    rated,columns,_=store._seeded_rating_frame(prefix,None)
    work=prefix.merge(rated,on=KEY[:2],validate="one_to_one")
    fresh=engineer_features(work,save=False,skip_ratings=True,counts_state={})
    cached=pd.read_parquet(store.featured_path,columns=KEY+features,
        filters=[("race_date","<=",pd.Timestamp("2020-09-20"))])
    paired=fresh[KEY+features].merge(cached,on=KEY,validate="one_to_one",suffixes=("_native","_stored"))
    if len(paired)!=len(prefix):
        raise ValueError("Native prefix loses original runners")
    for c in features:
        np.testing.assert_allclose(paired[c+"_native"].to_numpy(np.float32),paired[c+"_stored"].to_numpy(np.float32),
            equal_nan=True,rtol=1e-6,atol=1e-6,err_msg=c)
    return len(prefix)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    result=json.loads((RESEARCH / "native_check/report.json").read_text())
    native=json.loads((RESEARCH / "native_check/manifest.json").read_text())
    prep=json.loads((BACKFILL / "manifest.json").read_text())
    if not result["advance_to_canonical"] or result["current_price_independence_100_races_max_error"]>1e-12:
        raise ValueError("The predeclared native win gate did not pass")
    if get_production_run_id()!=PARENT:
        raise ValueError("The installed parent changed")
    scope_status={"historical_component_cache_scope":native["component_cache_scope"],
        "historical_component_cache_semantics":native["learner_semantics"]}
    verified_historical_cache_scope(scope_status,TrainingConfig(refresh_data=False,threads=2))
    prove_additive_sources()
    if sha(ROOT / "src/prior_market_research.py")!=prep["helper_sha256"]:
        raise ValueError("The tested price-history calculation changed")
    for name,key in (("features.parquet","features_sha256"),("state.pkl","state_sha256")):
        if sha(BACKFILL/name)!=prep[key]:
            raise ValueError("The prepared feature/state addition changed")
    signature=fe_signature()
    source_hashes={p:sha(ROOT/p) for p in _FE_SOURCE_FILES}
    store=FeatureStore()
    current_meta=store.meta()
    old_root=BACKFILL / "previous_generation/feature_store"
    old_root.mkdir(parents=True,exist_ok=True)
    selected=native["features"]
    if selected[-3:]!=list(FEATURES):
        raise ValueError("Native gate belongs to another feature family")
    if current_meta["fe_signature"]!=signature:
        if current_meta!=prep["base_store_meta"] or sha(store.featured_path)!=prep["base_featured_sha256"]:
            raise ValueError("Base history changed after preparing the independent additions")
        with store._access():
            for name in store._FILES:
                original,target=Path(store.root)/name,old_root/name
                if not target.exists():
                    os.link(original,target)
                elif sha(original)!=sha(target):
                    raise ValueError("Previous generation is already frozen differently")
        values=pd.read_parquet(BACKFILL / "features.parquet").set_index(KEY,verify_integrity=True)
        with (BACKFILL / "state.pkl").open("rb") as stream:
            addition=pickle.load(stream)
        with (old_root / "counts_state.pkl").open("rb") as stream:
            counts=pickle.load(stream)
        if _PRIOR_MARKET_STATE_KEY in counts:
            raise ValueError("A price-history state already exists in the base generation")
        counts[_PRIOR_MARKET_STATE_KEY]=addition
        old_schema=pq.read_schema(store.featured_path)
        if set(FEATURES)&set(old_schema.names):
            raise ValueError("The new columns already exist")
        schema=old_schema
        for c in FEATURES:
            schema=schema.append(pa.field(c,pa.float64()))
        total=0
        with store._staged_update() as pending:
            with pq.ParquetWriter(pending.featured_path,schema,compression="snappy") as writer:
                for batch in pq.ParquetFile(old_root / "featured.parquet").iter_batches(batch_size=32768):
                    original=pa.Table.from_batches([batch])
                    index=pd.MultiIndex.from_frame(original.select(KEY).to_pandas())
                    positions=values.index.get_indexer(index)
                    if np.any(positions<0):
                        raise ValueError("Backfill lacks original runners")
                    extra=values.iloc[positions]
                    extended=original
                    for c in FEATURES:
                        extended=extended.append_column(c,pa.array(extra[c].to_numpy(),type=pa.float64(),from_pandas=True))
                    if not extended.drop(list(FEATURES)).equals(original,check_metadata=False):
                        raise ValueError("An existing feature value changed during extension")
                    writer.write_table(extended)
                    total+=len(extra)
            if total!=prep["runners"]:
                raise ValueError("Extended store population differs")
            for name in ("processed.parquet","ratings_state.pkl"):
                os.link(old_root/name,Path(pending.root)/name)
            pending._save_counts_state(counts)
            meta=deepcopy(current_meta)
            meta.update(fe_signature=signature,built_at=datetime.now(timezone.utc).isoformat(),
                independent_feature_backfill={"features":list(FEATURES),"base_fe_signature":prep["base_fe_signature"],
                    "base_featured_sha256":prep["base_featured_sha256"],"prefix_float32_exact":True,"old_columns_exact":True})
            with open(pending.meta_path,"w",encoding="utf-8") as stream:
                json.dump(meta,stream,indent=2)
            if fe_signature()!=signature or any(sha(ROOT/p)!=d for p,d in source_hashes.items()):
                raise ValueError("Feature sources changed during the independent extension")
            prefix_rows=check_native_prefix(pending,selected)
        write(BACKFILL / "commit.json",{"stage":"committed","fe_signature":signature,"rows":total,
            "columns":len(schema),"old_columns_bitwise_values_exact":True,"native_prefix_rows":prefix_rows,
            "base_sources_additive_ast_proof":True})
        del values,counts,addition
        gc.collect()
    else:
        prefix_rows=json.loads((BACKFILL / "commit.json").read_text())["native_prefix_rows"]
    # Selection is fixed only after the native gate and normal-history proof.
    feature_file=ROOT / "feature_sets/win.txt"
    feature_file.write_text("# 033 reviewed prior-market history plus unconditional verified source corrections\n"+
        "\n".join(selected)+"\n",encoding="utf-8")
    names=_model_load_columns(store)
    if names is None or not set(FEATURES).issubset(names) or not {"lengths_behind","beaten_lengths"}.issubset(names):
        raise ValueError("Normal model projection lacks the selected history/target fields")
    frame=pd.read_parquet(store.featured_path,columns=names,filters=[("race_date","<=",pd.Timestamp("2026-03-31"))])
    frame.to_parquet(OUTPUT / "corrected_inputs.parquet",index=False)
    for source,name in ((BASE/"baseline_2026Q1_predictions.parquet","baseline_2026Q1_predictions.parquet"),
        (RESEARCH/"native_check/manifest.json","native_check_manifest.json"),
        (RESEARCH/"native_check/report.json","native_check_report.json"),
        (RESEARCH/"native_check/predictions.parquet","native_win_predictions.parquet"),
        (BACKFILL/"commit.json","backfill_commit.json"), (BACKFILL/"manifest.json","backfill_preparation.json")):
        target=OUTPUT/name
        if target.exists() and sha(target)!=sha(source):
            raise ValueError("Do not overwrite another canonical evidence capsule")
        if not target.exists():
            shutil.copyfile(source,target)
    cache=OUTPUT / "components/historical"
    cache.mkdir(parents=True,exist_ok=True)
    components=list((RESEARCH / "native_check/components").glob("*.joblib"))
    if len(components)!=3:
        raise ValueError("Expected exactly the three native win estimators")
    for source in components:
        target=cache/source.name
        if not target.exists():
            shutil.copyfile(source,target)
        elif sha(target)!=sha(source):
            raise ValueError("Canonical estimator checkpoint differs")
    status={"stage":"rebuilt","label":"033 prior market history and032 verified source corrections",
        "fe_signature":signature,"source_sha256":source_hashes,"store_meta":store.meta(),"features":selected,
        "extra_source_paths":["scripts/install_prior_market_history.py","scripts/prepare_prior_market_backfill.py"],
        "extra_input_paths":["native_check_manifest.json","native_check_report.json","backfill_commit.json","backfill_preparation.json"],
        "reference_win_predictions":"native_win_predictions.parquet",
        "teacher_inputs_sha256":{"data/processed/betfair_bsp.parquet":sha(ROOT / "data/processed/betfair_bsp.parquet")},
        "adoption_policy":"Mandatory source corrections always apply. Prior-market inputs selected after their fixed cheap and native win checks improve overall and established-handicap log loss with no worse runner Brier. Same inspected Q1 development; no fresh confirmation, no retuning.",
        **scope_status}
    write(OUTPUT / "rebuild_status.json",status)
    print(json.dumps({"stage":"canonical_history_ready","features":len(selected),"fe_signature":signature,
        "native_prefix_rows":prefix_rows,"win_components_reused":3}),flush=True)


if __name__=="__main__":
    main()
