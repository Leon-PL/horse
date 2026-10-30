"""Install source-confirmed metadata and rebuild ordinary histories once.

Recompute every Elo component. Reuse Glicko/TrueSkill only after proving all
their inputs/code/settings unchanged and reproducing a native prefix exactly.
The normal staged builder still owns every other feature and carried state.
"""
from pathlib import Path
from copy import deepcopy
import gc
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config
from scripts.build_feature_store_batched import _build
from src.data_processor import process_data
from src.deployment_validation import file_sha256 as sha
from src.featured_dataset import _FE_SOURCE_FILES, fe_signature, _model_load_columns
from src.feature_store import FeatureStore
from src.rating_reuse import component_signature, freeze_rating_source
from src.ratings import compute_elo_features
from src.run_store import get_production_run_id
from src.verified_race_metadata import apply_verified_race_types

OUTPUT = ROOT / "data/audit/trip_suitability_032"
INSTALL = OUTPUT / "installation"
KEY = ["race_id", "horse_name"]
PARENT = "20261028_212411_292221"


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class RaceTypeRatingCache:
    def __init__(self, source):
        self.store = FeatureStore(str(source))
        self.receipt = json.loads((OUTPUT / "rating_source.json").read_text())

    def prepare(self, processed):
        if component_signature() != self.receipt["rating_component_signature"]:
            raise ValueError("Rating code/config/runtime changed")
        for name, digest in self.receipt["source_files_sha256"].items():
            if sha(Path(self.store.root)/name) != digest:
                raise ValueError("Frozen original rating source changed")
        if any(getattr(config, k, False) for k in
               ("GLICKO_JOCKEY_TRAINER", "GLICKO_DIMENSIONAL", "GLICKO_MARGIN", "GLICKO2_ENABLED")):
            raise ValueError("Additional Glicko dimensions require separate review")
        previous = self.store.load_processed()
        expected = apply_verified_race_types(previous.copy())
        for kind in ("Flat", "Hurdle", "Chase"):
            col = "race_type_"+kind
            if col in expected:
                expected[col] = expected.race_type.eq(kind).astype(previous[col].dtype)
        # No result, entity, margin, date, distance or other processed value
        # may change alongside these reviewed event-type replacements.
        a, b = expected.set_index(KEY).sort_index(), processed.set_index(KEY).sort_index()
        pd.testing.assert_index_equal(a.index, b.index, exact=False)
        if set(a) != set(b):
            raise ValueError("Processed schema changed beyond this metadata batch")
        for col in a:
            pd.testing.assert_series_equal(a[col], b[col], check_dtype=False, check_names=False)
        changed = previous.race_type.astype(str).ne(expected.race_type.astype(str))
        if changed.sum() != 430 or previous.loc[changed, "race_id"].nunique() != 52:
            raise ValueError("Correction population differs from the confirmed source audit")
        prefix = processed.loc[processed.race_date.le(processed.race_date.min()+pd.Timedelta(days=5))].copy()
        native, columns, _ = self.store._seeded_rating_frame(prefix, None)
        preserved = [c for c in columns if "glicko" in c or c.startswith("horse_ts_")]
        cached = self.store.load_featured(columns=KEY+preserved)
        check = native[KEY+preserved].merge(cached, on=KEY, validate="one_to_one", suffixes=("_new", "_old"))
        if len(check) != len(native):
            raise ValueError("Native rating prefix lost runners")
        for col in preserved:
            np.testing.assert_allclose(check[col+"_new"], check[col+"_old"], rtol=0, atol=1e-12, equal_nan=True)
        print(json.dumps({"stage":"recompute_all_elo", "rows":len(processed), "reused_glicko_trueskill":preserved}), flush=True)
        work = self.store._with_dist_category(processed.copy())
        base = set(work)
        rated, elo_state = compute_elo_features(work, return_state=True)
        elo_columns = [c for c in rated if c not in base and c != "_event_dt"]
        if set(elo_columns+preserved) != set(columns):
            raise ValueError("Unexpected rating output schema")
        ratings = rated[KEY+elo_columns].merge(cached, on=KEY, validate="one_to_one")
        self.ratings = ratings.set_index(KEY, verify_integrity=True)
        self.columns = columns
        self.final_state = {"elo":elo_state, "glicko":deepcopy(self.store.load_state()["glicko"])}
        return {"source_root":str(self.store.root), "rating_columns":columns,
            "validated_processed_rows":len(processed), "prefix_rows_checked":len(native),
            "preserved":preserved, "recomputed":elo_columns, "corrected_rows":430, "corrected_fields":52}

    def for_chunk(self, chunk):
        keys = pd.MultiIndex.from_frame(chunk[KEY])
        if not keys.isin(self.ratings.index).all():
            raise ValueError("Rating snapshot lacks runners")
        return self.ratings.reindex(keys).reset_index()[KEY+self.columns]


def main():
    INSTALL.mkdir(parents=True, exist_ok=True)
    if get_production_run_id() != PARENT:
        raise ValueError("Installed parent changed before this rebuild")
    if (INSTALL / "rebuild_status.json").exists():
        completed = json.loads((INSTALL / "rebuild_status.json").read_text())
        if completed["stage"] == "rebuilt" and completed["fe_signature"] == fe_signature():
            print("Corrected build already complete; no rebuild repeated", flush=True)
            return
    source = OUTPUT / "previous_generation/feature_store"
    source.mkdir(parents=True, exist_ok=True)
    store = FeatureStore()
    if not (OUTPUT / "rating_source.json").exists():
        with store._access():
            for name in store._FILES:
                original, backup = Path(store.root)/name, source/name
                if not backup.exists():
                    os.link(original, backup)
                elif sha(original) != sha(backup):
                    raise ValueError("Frozen previous generation differs")
        freeze_rating_source(FeatureStore(str(source)), OUTPUT / "rating_source.json")
    merged = OUTPUT / "verified_race_types_merged.json"
    current = ROOT / "reference/verified_race_types.json"
    if sha(current) != sha(merged):
        if sha(current) != sha(OUTPUT / "verified_race_types_before.json"):
            raise ValueError("Reference catalog changed independently")
        temporary = current.with_suffix(".json.tmp")
        shutil.copyfile(merged, temporary)
        temporary.replace(current)
    signature = fe_signature()
    hashes = {name:sha(ROOT/name) for name in _FE_SOURCE_FILES}
    write(INSTALL / "rebuild_status.json", {"stage":"rebuilding", "fe_signature":signature, "source_sha256":hashes})
    with store._staged_update() as pending:
        metadata = _build(pending, 60_000, ratings_cache=RaceTypeRatingCache(source))
        if fe_signature() != signature or any(sha(ROOT/name) != digest for name,digest in hashes.items()):
            raise ValueError("Feature sources changed during rebuild")
    gc.collect()
    names = _model_load_columns(store)
    if names is None or not {"lengths_behind", "beaten_lengths"}.issubset(names):
        raise ValueError("Ordinary model projection lacks native target metadata")
    frame = pd.read_parquet(store.featured_path, columns=names,
        filters=[("race_date", "<=", pd.Timestamp("2026-03-31"))])
    frame.to_parquet(INSTALL / "corrected_inputs.parquet", index=False)
    shutil.copyfile(ROOT / "data/audit/established_form_029/installation/historical_predictions.parquet",
        INSTALL / "baseline_2026Q1_predictions.parquet")
    features = [s.strip() for s in (ROOT / "feature_sets/win.txt").read_text().splitlines()
                if s.strip() and not s.lstrip().startswith("#")]
    write(INSTALL / "rebuild_status.json", {"stage":"rebuilt", "label":"032 verified race-type corrections",
        "fe_signature":signature, "source_sha256":hashes, "store_meta":metadata, "features":features,
        "extra_source_paths":["scripts/rebuild_race_type_repairs.py", "scripts/confirm_remaining_race_types.py"],
        "teacher_inputs_sha256":{"data/processed/betfair_bsp.parquet":sha(ROOT / "data/processed/betfair_bsp.parquet")}})
    print(json.dumps({"stage":"rebuilt", "fe_signature":signature, "rows":metadata["featured_rows"]}), flush=True)


if __name__ == "__main__":
    main()
