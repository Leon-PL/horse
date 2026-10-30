"""Correct numeric casualty tail ranks without changing any retained model input."""
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "data/audit/rtv_repair_features_044"
PROJECT = CASE / "project"
OUT = CASE / "casualty_quality"
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.deployment_validation import file_sha256 as sha
from src.feature_store import FeatureStore
from src.featured_dataset import _rtv_signature, fe_signature
from src.rtv_features import add_rtv_features, build_run_observations, FEATURES
from src.rtv_feature_collection import observation_tables
from src.source_pipeline import canonical_rows, match_runners, _publish


def main():
    if (CASE / "casualty_quality.json").exists():
        raise ValueError("Preserve the completed quality refresh")
    OUT.mkdir(exist_ok=False)
    table_root = ROOT / "data/rtv_cache"
    bad_parts, parsed = [], {}
    for path in sorted((table_root / "tables/runners").glob("*.parquet")):
        frame = pd.read_parquet(path)
        bad = frame.state.eq("casualty") & frame.is_finisher
        if not bad.any():
            continue
        bad_parts.append(frame.loc[bad, ["race_id", "runner_id", "horse_name", "state", "finish_position"]])
        relative = path.relative_to(ROOT).as_posix()
        preserved = OUT / "parsed_before" / relative
        preserved.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, preserved)
        destination = PROJECT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.loc[bad, "is_finisher"] = False
        frame.to_parquet(destination, index=False)
        parsed[relative] = {"before_sha256": sha(path), "after_sha256": sha(destination), "corrected_flags": int(bad.sum())}
    bad = pd.concat(bad_parts, ignore_index=True)
    if len(bad) != 12:
        raise ValueError("The reviewed casualty population changed")
    for path in (table_root / "tables").rglob("*.parquet"):
        destination = PROJECT / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(path, destination)
    bad.to_parquet(OUT / "source_rows.parquet", index=False)
    bad_keys = pd.MultiIndex.from_frame(bad[["race_id", "runner_id"]])
    race_dates = []
    for path in (table_root / "tables/races").glob("*.parquet"):
        frame = pd.read_parquet(path, columns=["race_id", "race_date"])
        race_dates.extend(frame.loc[frame.race_id.isin(bad.race_id), "race_date"].tolist())
    runs_path = PROJECT / "data/rtv_cache/feature_runs.parquet"
    shutil.copyfile(runs_path, OUT / "feature_runs_before.parquet")
    original = pd.read_parquet(runs_path)
    replacements = []
    for day in sorted(set(race_dates)):
        tables = observation_tables(day, day, source_root=table_root)
        runners = tables["runners"]
        affected = pd.MultiIndex.from_frame(runners[["race_id", "runner_id"]]).isin(bad_keys)
        runners.loc[affected, "is_finisher"] = False
        links = match_runners(tables["races"], runners, canonical_rows(day, day, db_path=PROJECT / "data/races.db"))
        replacements.append(build_run_observations(tables["races"], runners, tables["metrics"], tables["sectionals"], tables["quality_flags"], links))
        print({"refreshed_day": day, "corrected_flags": int(affected.sum())}, flush=True)
    revised = pd.concat([original.loc[~pd.to_datetime(original.race_date).isin(pd.to_datetime(race_dates))], *replacements], ignore_index=True)
    revised = revised.sort_values(["race_date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)
    original = original.sort_values(["race_date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)
    keys = ["race_id", "horse_id", "race_date", "available_date"]
    pd.testing.assert_frame_equal(original[keys], revised[keys], check_dtype=False, check_exact=True)
    changes = {c: int((~(original[c].eq(revised[c]) | (original[c].isna() & revised[c].isna()))).sum()) for c in original if c not in keys}
    changes = {c: n for c, n in changes.items() if n}
    revised.to_parquet(OUT / "feature_runs_revised.parquet", index=False)
    os.replace(OUT / "feature_runs_revised.parquet", runs_path)
    del original, revised, replacements, frame
    gc.collect()
    store = FeatureStore()
    backup = FeatureStore(str(OUT / "previous_store"))
    with store._access():
        for name in store._FILES:
            os.link(Path(store.root) / name, Path(backup.root) / name)
    selected = json.loads((CASE / "refit/plan.json").read_text())["recipe"]["feature_allowlist"]
    if set(selected) & set(FEATURES):
        raise ValueError("A retained fitted input depends on the refreshed RTV facts")
    schema = pq.read_schema(backup.featured_path)
    metadata = backup.meta()
    rows = 0
    with store._staged_update() as pending:
        with pq.ParquetWriter(pending.featured_path, schema) as writer:
            for batch in pq.ParquetFile(backup.featured_path).iter_batches(batch_size=30000):
                before = batch.to_pandas()
                after = add_rtv_features(before)[before.columns]
                ordinary = [c for c in before if c not in FEATURES]
                pd.testing.assert_frame_equal(before[ordinary], after[ordinary], check_exact=True)
                writer.write_table(pa.Table.from_pandas(after, schema=schema, preserve_index=False))
                rows += len(after)
        for name in store._FILES:
            if name not in ("featured.parquet", "meta.json"):
                os.link(Path(backup.root) / name, Path(pending.root) / name)
        metadata["rtv_signature"] = _rtv_signature(metadata["last_date"])
        Path(pending.meta_path).write_text(json.dumps(metadata, indent=2))
    # Normal directory inheritance keeps the two newly committed files readable
    # by both the restricted development token and the user's recorder token.
    readable = OUT / "readable_store_files"
    readable.mkdir()
    for name in ("featured.parquet", "meta.json"):
        shutil.copyfile(Path(store.root) / name, readable / name)
        os.replace(readable / name, Path(store.root) / name)
    # Check the actual persisted selected inputs, not just the in-memory edit.
    for old, new in zip(pq.ParquetFile(backup.featured_path).iter_batches(batch_size=30000, columns=selected),
                        pq.ParquetFile(store.featured_path).iter_batches(batch_size=30000, columns=selected), strict=True):
        pd.testing.assert_frame_equal(old.to_pandas(), new.to_pandas(), check_exact=True)
    if rows != 755357 or metadata["fe_signature"] != fe_signature():
        raise ValueError("Quality refresh changed the canonical generation")
    result = {"completed_at": datetime.now(timezone.utc).isoformat(), "corrected_casualty_flags": len(bad),
        "parsed_files": parsed, "run_observation_changes": changes,
        "before_runs_sha256": sha(OUT / "feature_runs_before.parquet"), "after_runs_sha256": sha(runs_path),
        "store_meta": metadata, "rows_checked": rows, "retained_inputs": len(selected),
        "all_non_rtv_inputs_unchanged_exact": True, "persisted_selected_inputs_unchanged_exact": True,
        "processed_and_rating_state_bytes_unchanged": True, "new_model_fits": 0,
        "parser_sha256": sha(PROJECT / "src/rtv_scraper.py"),
        "scope": "External RTV fact/parser correction only; original trial cache and all retained fitted inputs preserved"}
    _publish(CASE / "casualty_quality.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
