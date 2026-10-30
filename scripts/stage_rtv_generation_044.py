"""Prepare the verified repair and native RTV feature generation in isolation."""
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
CASE = ROOT/"data/audit/rtv_repair_features_044"
PROJECT = CASE/"project"
sys.path.insert(0, str(PROJECT))

import pandas as pd
import config
from src.database import _export_results_snapshot
from src.deployment_validation import file_sha256 as sha
from src.source_pipeline import _publish
from src.verified_result_repair import apply_verified_results


def inputs():
    before = pd.read_parquet(CASE/"repair_before.parquet")
    after = pd.read_parquet(CASE/"repair_after.parquet")
    additional = pd.read_parquet(CASE/"new_fields.parquet")
    old = before.set_index(["race_id", "horse_name"])
    new = after.set_index(["race_id", "horse_name"])
    removed = old.index[~old.index.isin(new.index)].tolist()
    evidence = {"review_roster_sha256": sha(CASE/"review_roster.json"), "verification_sha256": sha(CASE/"verification.json"),
        "confirmed_nonstarters": [list(k) for k in removed],
        "input_sha256": {name: sha(CASE/(name+".parquet")) for name in ("repair_before", "repair_after", "new_fields")},
        "source_sha256": {p.name: sha(p) for p in (CASE/"sources").glob("*.json")},
        "source": "Fresh complete weighed-in SL fields agree with the independently parsed RTV official order"}
    return before, after, additional, evidence


def link_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        os.link(source, destination)


def prepare():
    verification = json.loads((CASE/"verification.json").read_text())
    if verification["counts"].get("verified_repair") != 75 or verification["counts"].get("verified_missing_field") != 14:
        raise ValueError("Finish all 75 official-order fields and 14 missing starter fields first")
    destination = PROJECT/"data/races.db"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with closing(sqlite3.connect(destination)) as prior:
            count = prior.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        if count:
            raise ValueError("Preserve the existing staged database")
        destination.rename(CASE/("empty_test_database_"+datetime.now().strftime("%H%M%S%f")+".db"))
    with closing(sqlite3.connect((ROOT/"data/races.db").as_uri()+"?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(CASE/"source_before.db")) as backup:
            source.backup(backup)
    shutil.copy2(CASE/"source_before.db", destination)
    before, after, additional, evidence = inputs()
    with closing(sqlite3.connect(destination)) as con:
        result = apply_verified_results(con, before, after, additional, repair_id="rtv_verified_results_044", evidence=evidence)
        source = pd.read_sql_query("SELECT * FROM results ORDER BY race_date", con)
        _export_results_snapshot(source.drop(columns=["scraped_at"], errors="ignore"), str(PROJECT/"data/raw/race_results.csv"))
    for relative in ("data/atr_cache", "data/reference"):
        for path in (ROOT/relative).rglob("*"):
            if path.is_file():
                link_file(path, PROJECT/path.relative_to(ROOT))
    for relative in ("data/processed/betfair_bsp.parquet", "data/trainer_locations.csv"):
        if (ROOT/relative).exists():
            link_file(ROOT/relative, PROJECT/relative)
    _publish(CASE/"prepared_source.json", {"prepared_at": datetime.now(timezone.utc).isoformat(), "application": result,
        "rows": len(source), "fields": int(source.race_id.nunique()), "raw_sha256": sha(PROJECT/"data/raw/race_results.csv"),
        "canonical_writes": False, "evidence": evidence})
    print(result, flush=True)


def materialise():
    from src.rtv_feature_collection import refresh_run_observations, refresh_pre_observations
    for year in (2021, 2023, 2024, 2025, 2026):
        for quarter in (1, 2, 3, 4):
            lo = f"{year}-{3*quarter-2:02d}-01"
            hi = str((pd.Timestamp(lo)+pd.offsets.QuarterEnd()).date())
            result = refresh_run_observations(lo, hi, database=PROJECT/"data/races.db", source_root=ROOT/"data/rtv_cache")
            print({"part": f"{year}-Q{quarter}", **result}, flush=True)
    result = refresh_pre_observations(raw_path=ROOT/"data/rtv_cache/rtv_raw.sqlite")
    _publish(CASE/"materialised.json", {"completed_at": datetime.now(timezone.utc).isoformat(), "pre": result,
        "run_sha256": sha(PROJECT/"data/rtv_cache/feature_runs.parquet"), "pre_sha256": sha(PROJECT/"data/rtv_cache/feature_pre.parquet")})
    print(result, flush=True)


def rebuild():
    from src.data_processor import process_data
    from src.feature_store import FeatureStore
    from scripts.build_feature_store_batched import _build
    from src.featured_dataset import fe_signature
    raw = pd.read_csv(PROJECT/"data/raw/race_results.csv")
    processed = process_data(df=raw, save=False)
    processed.to_parquet(CASE/"processed.parquet", index=False)
    store = FeatureStore()
    with store._staged_update() as pending:
        metadata = _build(pending, 60000, processed=processed)
    _publish(CASE/"rebuilt.json", {"completed_at": datetime.now(timezone.utc).isoformat(), "meta": metadata,
        "fe_signature": fe_signature(), "new_tree_fits": 0, "canonical_writes": False})
    print(metadata, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "materialise", "rebuild"])
    {"prepare": prepare, "materialise": materialise, "rebuild": rebuild}[parser.parse_args().phase]()
