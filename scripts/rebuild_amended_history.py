"""Prepare and rebuild corrected result histories in an isolated generation.

No source-data installation, model fitting, recorder changes or promotion.
Winner amendments affect Elo/Glicko and subsequent form, so an old rating
cache cannot be reused. The complete native builder owns every calculation.
"""
from datetime import datetime, timezone
import argparse
import gc
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.data_processor import process_data
from src.deployment_validation import file_sha256 as sha
from src.feature_store import FeatureStore, _stringify_objects
from src.featured_dataset import _FE_SOURCE_FILES, data_signature, fe_signature

CASE = ROOT / "data/audit/result_amendments_042"
STAGE = CASE / "staged_history_v2"
OUTPUT = CASE / "history_rebuild"
BEFORE = ROOT / "data/audit/live_feature_speed_20261029/installation/previous_store/processed.parquet"
KEY = ["race_id", "horse_name"]
CHANGED = {"finish_position", "finish_pos_label", "won", "lengths_behind", "beaten_lengths"}


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify(plan):
    if fe_signature() != plan["fe_signature"]:
        raise ValueError("Feature sources/configuration changed")
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Pinned implementation changed: " + name)
    for name, digest in plan["input_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Pinned source evidence changed: " + name)


def prepare():
    OUTPUT.mkdir(exist_ok=False)
    qualification = json.loads((STAGE / "report.json").read_text())
    for name, digest in qualification["outputs_sha256"].items():
        if sha(STAGE / name) != digest:
            raise ValueError("Staged result evidence changed")
    paths = ["scripts/rebuild_amended_history.py", *_FE_SOURCE_FILES]
    plan = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "fe_signature": fe_signature(),
        "source_sha256": {name: sha(ROOT / name) for name in paths},
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in
            (STAGE / "race_results.csv", BEFORE, CASE / "proposed_repairs.parquet", ROOT / "data/atr_cache/atr_sectionals.parquet")},
        "policy": "Mandatory official-result correction, unchanged154 selected inputs; full native rating/form rebuild; no optional-feature trial or model score",
        "chunk_rows": 60000, "canonical_writes": False, "new_model_fits": 0}
    write(OUTPUT / "plan.json", plan)
    with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as capsule:
        for name in paths:
            capsule.write(ROOT / name, arcname=name)
    raw = pd.read_csv(STAGE / "race_results.csv")
    processed = _stringify_objects(process_data(df=raw, save=False))
    del raw
    gc.collect()
    previous = pd.read_parquet(BEFORE)
    a = previous.set_index(KEY).sort_index()
    b = processed.set_index(KEY).sort_index()
    pd.testing.assert_index_equal(a.index, b.index, exact=False)
    if set(a) != set(b):
        raise ValueError("Processed schema changed outside the source amendment")
    counts = {}
    for col in a:
        if col not in CHANGED:
            pd.testing.assert_series_equal(a[col], b[col], check_dtype=False, check_names=False, check_exact=True)
        else:
            same = a[col].eq(b[col]) | (a[col].isna() & b[col].isna())
            counts[col] = int((~same).sum())
    repaired_ids = pd.read_parquet(CASE / "proposed_repairs.parquet", columns=["race_id"]).race_id.astype(str).unique()
    unaffected = ~a.index.get_level_values("race_id").astype(str).isin(repaired_ids)
    pd.testing.assert_frame_equal(a.loc[unaffected], b.loc[unaffected, a.columns], check_dtype=False, check_exact=True)
    repaired = processed.loc[processed.race_id.astype(str).isin(repaired_ids)]
    audit = json.loads((CASE / "audit.json").read_text())
    if (len(repaired) != audit["eligible_field_runners"] or repaired.race_id.nunique() != audit["eligible_fields"] or repaired.won.sum() != audit["eligible_fields"]
            or not repaired.loc[repaired.won.eq(0), ["lengths_behind", "beaten_lengths"]].isna().all().all()
            or not repaired.loc[repaired.won.eq(1), ["lengths_behind", "beaten_lengths"]].eq(0).all().all()):
        raise ValueError("Normal processing changed the corrected label/margin contract")
    processed.to_parquet(OUTPUT / "processed.parquet", index=False)
    verify(plan)
    write(OUTPUT / "preparation.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(processed), "fields": int(processed.race_id.nunique()), "all_other_processed_values_exact": True,
        "all_unaffected_fields_exact": True, "changed_processed_cells": counts,
        "data_signature": data_signature(processed), "processed_sha256": sha(OUTPUT / "processed.parquet"),
        "plan_sha256": sha(OUTPUT / "plan.json"), "new_model_fits": 0})
    print("Prepared corrected history; all unrelated processed values and fields match exactly.", flush=True)


def rebuild():
    from scripts.build_feature_store_batched import _build
    plan = json.loads((OUTPUT / "plan.json").read_text())
    preparation = json.loads((OUTPUT / "preparation.json").read_text())
    if (OUTPUT / "rebuilt.json").exists():
        raise ValueError("Preserve the completed corrected history build")
    verify(plan)
    if sha(OUTPUT / "processed.parquet") != preparation["processed_sha256"]:
        raise ValueError("Prepared history changed")
    processed = pd.read_parquet(OUTPUT / "processed.parquet")
    store = FeatureStore(str(OUTPUT / "feature_store"))
    with store._staged_update() as pending:
        metadata = _build(pending, plan["chunk_rows"], processed=processed)
        verify(plan)
        if metadata["processed_rows"] != preparation["rows"] or metadata["featured_rows"] != preparation["rows"]:
            raise ValueError("Corrected history build lost source rows")
    write(OUTPUT / "rebuilt.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "store_meta": metadata, "fe_signature": fe_signature(), "source_plan_sha256": sha(OUTPUT / "plan.json"),
        "ratings_recomputed": True, "canonical_writes": False, "new_model_fits": 0})
    print("Corrected history generation built; no live files replaced and no model fitted.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=["prepare", "rebuild"])
    cli.add_argument("--case", type=Path, default=CASE)
    args = cli.parse_args()
    CASE = args.case.resolve()
    STAGE = CASE / "staged_history_v2"
    OUTPUT = CASE / "history_rebuild"
    {"prepare": prepare, "rebuild": rebuild}[args.phase]()
