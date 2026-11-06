"""Reproducible, offline comparison of RTV partitions with current SL results."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import pyarrow.parquet as pq

from src.ingestion import racing_tv as rtv
from src.ingestion.pipeline import canonical_rows, match_runners


def run(directory, destination, db_path=rtv.RESULTS_DB_PATH):
    directory, destination = Path(directory), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    checks = defaultdict(lambda: [0, 0])
    parts = []
    coverage = defaultdict(int)
    conflict_frames = []
    for path in sorted((directory / "races").glob("*.parquet")):
        races = pd.read_parquet(path)
        runners = pd.read_parquet(directory / "runners" / path.name)
        report = rtv.validate_against_db({"races": races, "runners": runners}, db_path=db_path)
        b = report.pop("_joined")
        conflicts = b.loc[b.is_finisher & b.db_pos.ge(1) & b.final_position.ne(b.db_pos),
            ["race_date", "track_name", "off_time_utc", "race_id", "runner_id", "db_race_id",
             "horse_name", "finish_position", "final_position", "db_pos", "state"]]
        conflict_frames.append(conflicts)
        links = match_runners(races, runners, canonical_rows(*report["date_range"], db_path=db_path))
        links.to_parquet(destination / (path.stem + "_runner_links.parquet"), index=False)
        coverage["strict_identity_links"] += len(links)
        coverage["all_runners_including_nonstarters"] += len(runners)
        for name in ("rtv_races", "rtv_races_matched", "db_races", "n_db_races_missing_in_rtv",
                     "runners_rtv", "runners_matched", "n_runners_only_db", "rtv_abandoned_races_excluded",
                     "rtv_arab_races_excluded"):
            counts[name] += report.get(name, 0)
        for name, eligible, agree in (
            ("official_position", b.is_finisher & b.db_pos.ge(1), b.final_position.eq(b.db_pos)),
            ("physical_position", b.is_finisher & b.db_pos.ge(1), b.finish_position.eq(b.db_pos)),
            ("age", b.age.notna() & b.age_db.notna(), b.age.eq(b.age_db)),
            ("weight", b.weight_lbs.notna() & b.weight_lbs_db.notna(), (b.weight_lbs-b.weight_lbs_db).abs().lt(.5)),
            ("sp", b.sp_decimal.notna() & b.odds.notna(), (b.sp_decimal-b.odds).abs().le(.01*b.odds)),
            ("nonfinisher", pd.Series(True, index=b.index), (~b.is_finisher).eq(b.db_pos.fillna(0).eq(0))),
        ):
            checks[name][0] += int((eligible & agree).sum())
            checks[name][1] += int(eligible.sum())
        for name in ("timeform_rating", "comment", "equipment_symbols"):
            if name in runners:
                coverage[name] += int(runners[name].notna().sum())
        for table in ("metrics", "sectionals"):
            target = directory / table / path.name
            if target.exists() and pq.read_metadata(target).num_rows:
                cols = ["race_id", "runner_id"] + (["metric", "value"] if table == "metrics" else [])
                df = pd.read_parquet(target, columns=cols)
                coverage[table + "_rows"] += len(df)
                coverage[table + "_runners"] += len(df[["race_id", "runner_id"]].drop_duplicates())
                coverage[table + "_races"] += df.race_id.nunique()
                if table == "metrics":
                    for key, group in df.groupby("metric"):
                        coverage["metric_" + key] += int(pd.to_numeric(group.value, errors="coerce").notna().sum())
        for table in ("quality_flags",):
            target = directory / table / path.name
            if target.exists() and pq.read_metadata(target).num_rows:
                flags = pd.read_parquet(target)
                for key, group in flags.groupby("check"):
                    coverage["flag_" + key] += len(group)
        report["part"] = path.stem
        (destination / (path.stem + ".json")).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        parts.append({k: v for k, v in report.items() if k in ("part", "date_range", "race_field_agreement", "runner_field_agreement")})
        print(json.dumps({"part": path.stem, "races": report["rtv_races_matched"], "runners": report["runners_matched"]}), flush=True)
    conflicts = pd.concat(conflict_frames, ignore_index=True)
    conflicts.to_csv(destination / "official_position_conflicts.csv", index=False)
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "database": str(db_path),
              "official_position_conflict_races": int(conflicts.db_race_id.nunique()),
              "counts": dict(counts), "agreement": {k: {"agree": a, "compared": n, "fraction": a/n if n else None}
              for k, (a, n) in checks.items()}, "coverage": dict(coverage), "parts": parts}
    (destination / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "agreement": result["agreement"], "coverage": result["coverage"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=rtv.TABLES_DIR)
    parser.add_argument("--destination", default=str(ROOT / "data/audit/rtv_pipeline_043/crosscheck"))
    parser.add_argument("--database", default=rtv.RESULTS_DB_PATH)
    args = parser.parse_args()
    run(args.directory, args.destination, db_path=args.database)
