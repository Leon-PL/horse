"""Cross-check local metadata against already cached public performances.

Read-only audit, with no requests, fitting, data repair or outcome-based cohort.
Keep source contradictions and unknown types explicit instead of guessing.
"""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.deployment_validation import file_sha256
from src.data_scraper import _infer_race_type as infer_race_type

OUTPUT = ROOT / "data/audit/cached_race_code_review_025"
CODE = {"FLAT": "Flat", "AW FLAT": "Flat", "CHASE": "Chase",
        "HURDLE": "Hurdle", "HURDL": "Hurdle", "NHF": "Flat", "NH FLAT": "Flat", "BUMPER": "Flat"}


def run(metadata=None):
    source = Path(metadata) if metadata else ROOT / "data/raw/race_results.csv"
    local = (pd.read_parquet(source) if metadata else
        pd.read_csv(source, usecols=["race_id", "horse_id", "horse_name", "race_date",
            "race_name", "race_type", "distance_furlongs", "age", "track"],
            dtype={"race_id": str, "horse_id": str}))
    local["horse_name"] = local.horse_name.str.strip().str.title()
    paths = sorted((ROOT / "data/research").glob("public_prior_results*/horses/*.json"))
    rows = []
    snapshot = hashlib.sha256()
    for path in paths:
        raw = path.read_bytes()
        snapshot.update(str(path.relative_to(ROOT)).encode()); snapshot.update(raw)
        cached = json.loads(raw)
        for result in cached["results"]:
            if str(result["horse_id"]) != str(cached["horse_id"]):
                raise ValueError("Cached performance identity differs")
            rows.append({"race_id": str(result["race_id"]), "horse_id": str(result["horse_id"]),
                "profile_date": str(result["date"])[:10], "run_type": str(result.get("run_type") or "").upper().strip()})
    profiles = pd.DataFrame(rows).drop_duplicates()
    if profiles.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Contradictory cached performances; inspect before classifying")
    compared = local.merge(profiles, on=["race_id", "horse_id"], validate="one_to_one")
    compared["structured_coarse_type"] = compared.run_type.map(CODE)
    compared["date_agrees"] = compared.profile_date.eq(compared.race_date.str[:10])
    compared["coarse_type_disagrees"] = compared.structured_coarse_type.notna() & compared.structured_coarse_type.ne(compared.race_type)
    compared["proposed_race_type"] = compared.race_name.map(infer_race_type)
    compared["proposed_type_disagrees"] = compared.structured_coarse_type.notna() & compared.structured_coarse_type.ne(compared.proposed_race_type)
    disagreements = compared.loc[~compared.date_agrees | compared.coarse_type_disagrees].copy()
    identity = local.groupby("horse_id", sort=False).horse_name.nunique()
    names = local.groupby("horse_name", sort=False).horse_id.nunique()
    shared_names = local.loc[local.horse_name.isin(names[names.gt(1)].index)].drop_duplicates(["horse_id", "horse_name"])
    multiple_names = local.loc[local.horse_id.isin(identity[identity.gt(1)].index)].drop_duplicates(["horse_id", "horse_name"])
    short_jumps = local.loc[local.race_type.isin(["Hurdle", "Chase"]) & local.distance_furlongs.lt(13)]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    projection = OUTPUT / "local_metadata.parquet"
    if projection.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(projection), local, check_dtype=False)
    else:
        local.to_parquet(projection, index=False, compression="zstd")
    for name, frame in (("source_disagreements", disagreements), ("shared_names", shared_names),
                        ("multiple_names_per_id", multiple_names), ("short_jump_races", short_jumps),
                        ("proposed_parser_disagreements", compared.loc[compared.proposed_type_disagrees])):
        frame.to_csv(OUTPUT / (name+".csv"), index=False)
    report = {"scope": "Source-consistency audit on fixed existing public caches; no source collection, data changes, model fit or gain claim",
        "local_rows": len(local), "profile_files": len(paths), "unique_cached_performances": len(profiles),
        "matched_local_performances": len(compared), "matched_local_races": int(compared.race_id.nunique()),
        "structured_type_counts": compared.run_type.value_counts().to_dict(),
        "unrecognised_type_counts": compared.loc[compared.structured_coarse_type.isna()].run_type.value_counts().to_dict(),
        "coarse_type_disagreeing_rows": int(compared.coarse_type_disagrees.sum()),
        "coarse_type_disagreeing_races": int(compared.loc[compared.coarse_type_disagrees].race_id.nunique()),
        "proposed_type_disagreeing_rows": int(compared.proposed_type_disagrees.sum()),
        "proposed_type_disagreeing_races": int(compared.loc[compared.proposed_type_disagrees].race_id.nunique()),
        "proposed_new_disagreements": int((~compared.coarse_type_disagrees & compared.proposed_type_disagrees).sum()),
        "date_disagreements": int((~compared.date_agrees).sum()),
        "bumper_rows_encoded_flat": int((compared.run_type.isin(["NHF", "NH FLAT", "BUMPER"]) & compared.race_type.eq("Flat")).sum()),
        "bumper_races_encoded_flat": int(compared.loc[compared.run_type.isin(["NHF", "NH FLAT", "BUMPER"]) & compared.race_type.eq("Flat")].race_id.nunique()),
        "short_jump_races_under13f": int(short_jumps.race_id.nunique()),
        "horse_ids_with_multiple_display_names": int(identity.gt(1).sum()),
        "display_names_with_multiple_ids": int(names.gt(1).sum()),
        "identity_interpretation": "Display-name variants or reused names are review leads, not proof of identity errors; the primary horse-history features already prefer horse IDs",
        "race_type_interpretation": "NH Flat/bumper races are intentionally collapsed to Flat by current ingestion; this is an information-loss hypothesis, not proof of an incorrect published result",
        "local_source_path": str(source), "local_source_sha256": file_sha256(source), "profile_cache_sha256": snapshot.hexdigest(),
        "source_sha256": file_sha256(Path(__file__)), "production_changed": False}
    report["proposed_parser_sha256"] = file_sha256(ROOT / "src/data_scraper.py")
    report["local_metadata_projection_sha256"] = file_sha256(projection)
    fields = local.drop_duplicates("race_id").copy()
    fields["proposed_race_type"] = fields.race_name.map(infer_race_type)
    changed = fields.loc[fields.race_type.ne(fields.proposed_race_type)].copy()
    changed.to_csv(OUTPUT / "proposed_type_changes_by_race.csv", index=False)
    report["proposed_changed_local_races"] = len(changed)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, help="Replay a frozen local_metadata.parquet after daily source data advance")
    run(parser.parse_args().metadata)
