"""Align the 044 repair with the ordinary CSV contract, excluding scraped_at.

This changes an operational fingerprint, never selected feature values or model
bytes. Preserve the original refit/publication and qualify the exact adjustment.
"""
from contextlib import closing
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "data/audit/rtv_repair_features_044"
OUT = CASE / "source_contract"

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.deployment_validation import file_sha256 as sha
from src.data_processor import process_data
from src.feature_store import FeatureStore, _stringify_objects
from src.featured_dataset import data_signature, fe_signature, _rtv_signature
from src.source_pipeline import _publish


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    if (OUT / "prepared.json").exists():
        raise ValueError("Preserve the completed contract preparation")
    OUT.mkdir(exist_ok=True)
    fit = read(CASE / "refit/fitted.json")
    artifact = ROOT / "data/runs" / fit["run_id"] / "model.joblib"
    if sha(artifact) != fit["model_sha256"]:
        raise ValueError("Native weights changed")
    current = FeatureStore()
    previous = FeatureStore(str(OUT / "previous_store"))
    with current._access():
        for name in current._FILES:
            if not (Path(previous.root) / name).exists():
                os.link(Path(current.root) / name, Path(previous.root) / name)
    metadata = previous.meta()
    old = previous.load_processed()
    if "scraped_at" not in old:
        raise ValueError("No internal timestamp exists in the installed processed contract")
    with closing(sqlite3.connect((ROOT / "data/races.db").as_uri() + "?mode=ro", uri=True)) as con:
        raw = pd.read_sql_query("SELECT * FROM results ORDER BY race_date", con).drop(columns=["scraped_at"])
    raw.to_csv(OUT / "canonical.csv", index=False)
    del raw
    normal = _stringify_objects(process_data(df=pd.read_csv(OUT / "canonical.csv"), save=False))
    expected = old.drop(columns=["scraped_at"])
    different = [c for c in normal if c not in expected or not normal[c].equals(expected[c])]
    try:
        pd.testing.assert_frame_equal(expected[normal.columns], normal, check_dtype=False, check_exact=True)
    except AssertionError:
        summary = {}
        for name in different:
            if name not in expected:
                summary[name] = {"new_column": True}
                continue
            same = expected[name].eq(normal[name]) | (expected[name].isna() & normal[name].isna())
            summary[name] = {"changed_rows": int((~same).sum()), "before_dtype": str(expected[name].dtype),
                "after_dtype": str(normal[name].dtype), "examples": pd.DataFrame({"before": expected.loc[~same,name],
                "after": normal.loc[~same,name]}).head(5).astype(str).to_dict("records")}
        _publish(OUT / "differences.json", summary)
        raise
    normal.to_parquet(OUT / "processed.parquet", index=False)
    normalized_signature = data_signature(normal)
    prefix = data_signature(normal, cutoff="2026-10-28")
    del normal, old, expected
    ready = FeatureStore(str(OUT / "ready_store"))
    source = pq.ParquetFile(previous.featured_path)
    schema = source.schema_arrow.remove(source.schema_arrow.get_field_index("scraped_at"))
    with ready._staged_update() as pending:
        with pq.ParquetWriter(pending.featured_path, schema) as writer:
            for batch in source.iter_batches(batch_size=30000):
                table = pa.Table.from_batches([batch]).drop_columns(["scraped_at"])
                writer.write_table(table)
        shutil.copyfile(OUT / "processed.parquet", pending.processed_path)
        for name in ("ratings_state.pkl", "counts_state.pkl"):
            os.link(Path(previous.root) / name, Path(pending.root) / name)
        metadata["data_signature"] = normalized_signature
        metadata["rtv_signature"] = _rtv_signature(metadata["last_date"])
        Path(pending.meta_path).write_text(json.dumps(metadata, indent=2))
    original_meta = ROOT / "data/runs" / fit["run_id"] / "meta.json"
    shutil.copyfile(original_meta, OUT / "model_meta_before.json")
    meta = read(original_meta)
    selected = meta["training_config"]["validation_reference"]["features"]
    if "scraped_at" in selected:
        raise ValueError("Internal timestamp was selected; a metadata-only correction is impossible")
    rows = 0
    for before, after in zip(pq.ParquetFile(previous.featured_path).iter_batches(batch_size=30000, columns=selected),
                             pq.ParquetFile(ready.featured_path).iter_batches(batch_size=30000, columns=selected), strict=True):
        pd.testing.assert_frame_equal(before.to_pandas(), after.to_pandas(), check_exact=True)
        rows += before.num_rows
    if rows != 755357:
        raise ValueError("Unexpected history population")
    migration = {"excluded_operational_column": "scraped_at", "previous_source_signature": fit["signatures"]["data_signature"],
        "normalized_source_signature": normalized_signature, "normalized_training_prefix_signature": prefix,
        "original_meta_path": str(OUT / "model_meta_before.json"), "original_meta_sha256": sha(OUT / "model_meta_before.json"),
        "retained_inputs_exact": len(selected), "rows_checked": rows, "new_model_fits": 0,
        "scope": "Ordinary load_from_database excludes the operational timestamp; every fitted input and model artifact is unchanged"}
    meta["training_config"]["signatures"]["data_signature"] = normalized_signature
    meta["training_config"]["source_contract_migration_044"] = migration
    _publish(OUT / "model_meta_ready.json", meta)
    _publish(OUT / "prepared.json", {"prepared_at": datetime.now(timezone.utc).isoformat(), "run_id": fit["run_id"],
        "model_sha256": fit["model_sha256"], "fe_signature": fe_signature(), "migration": migration,
        "raw_sha256": sha(OUT / "canonical.csv"), "store_meta": metadata,
        "model_meta_sha256": sha(OUT / "model_meta_ready.json"), "selected_inputs_exact": True})
    print({"prepared": True, "normalized_source_signature": normalized_signature, "selected_inputs_exact": len(selected), "rows": rows}, flush=True)


def publish():
    from scripts.install_rtv_generation_044 import replace_copy
    prepared = read(OUT / "prepared.json")
    original_meta = ROOT / "data/runs" / prepared["run_id"] / "meta.json"
    if sha(original_meta) != prepared["migration"]["original_meta_sha256"]:
        raise ValueError("Installed metadata changed")
    current, ready = FeatureStore(), FeatureStore(str(OUT / "ready_store"))
    # No source fact changes: require a fresh DB export to match the preview.
    with closing(sqlite3.connect(ROOT / "data/races.db")) as con:
        with con:
            con.execute("BEGIN IMMEDIATE")
            raw = pd.read_sql_query("SELECT * FROM results ORDER BY race_date", con).drop(columns=["scraped_at"])
            raw.to_csv(OUT / "canonical_committing.csv", index=False)
            if sha(OUT / "canonical_committing.csv") != prepared["raw_sha256"]:
                raise ValueError("Source facts changed during contract qualification")
            try:
                with current._staged_update() as pending:
                    for name in current._FILES:
                        shutil.copyfile(Path(ready.root) / name, Path(pending.root) / name)
                replace_copy(OUT / "canonical_committing.csv", ROOT / "data/raw/race_results.csv")
                replace_copy(OUT / "model_meta_ready.json", original_meta)
                _publish(OUT / "published.json", {**prepared, "published_at": datetime.now(timezone.utc).isoformat(),
                    "model_bytes_unchanged": True, "source_facts_unchanged": True})
            except Exception:
                previous = FeatureStore(str(OUT / "previous_store"))
                with current._staged_update() as pending:
                    for name in current._FILES:
                        shutil.copyfile(Path(previous.root) / name, Path(pending.root) / name)
                replace_copy(OUT / "model_meta_before.json", original_meta)
                raise
    print("Normal CSV/store fingerprint contract installed; native model bytes unchanged", flush=True)


def register():
    from src.forward_store import create_study
    from src.forward_readout import register_plan
    from src.forward_recorder import replace_json
    published = read(OUT / "published.json")
    qa = read(CASE / "installation/qa_contract_after/report.json")
    if not qa["passed"] or not qa["staged_installed_inputs_and_probabilities_exact"]:
        raise ValueError("Finish normalized-contract serving parity")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = create_study(ROOT / f"data/forecasts/forward/{stamp}_rtv_features_044_{published['run_id']}")
    register_plan(study, study.path / "readout_plan.json")
    study.verify()
    settings = {"study": str(study.path), "model_id": published["run_id"], "protocol_sha256": study.protocol_sha256,
        "first_eligible_date": study.protocol["first_eligible_date"], "configured_at": datetime.now(timezone.utc).isoformat(),
        "scheduled_task": "HorseForecastRecorder_038", "scheduled_task_verified": False}
    replace_json(ROOT / "data/forecasts/recorder.json", settings)
    _publish(OUT / "recorder_registration.json", settings)
    print(json.dumps(settings, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "publish", "register"])
    {"prepare": prepare, "publish": publish, "register": register}[parser.parse_args().phase]()
