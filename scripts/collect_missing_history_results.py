"""Extend the dated-results pilot to jumps and Cheltenham missing-history horses.

Membership uses pre-race fields only, ends before the reserved holdout and is
frozen before requests. This separate cache never rewrites the earlier Flat
pilot, local results, lifetime totals or ratings.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from threading import Event

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from scripts.collect_prior_horse_results import FIELDS, fetch_one
from src.ingestion.sporting_life import SportingLifeScraper
from src.feature_semantics import class_number


def cohort(frame, scope):
    experienced = frame.days_since_last_run.gt(0) | frame.prior_form_runs.ge(2)
    selected = frame.loc[frame.horse_prev_races.eq(0) & experienced
        & frame.race_date.between("2021-01-01", "2026-03-31")].copy()
    if scope == "cheltenham":
        selected = selected.loc[selected.track.str.casefold().eq("cheltenham")]
    elif scope == "jumps":
        selected = selected.loc[selected.race_type.isin(["Hurdle", "Chase", "NH Flat", "Nh Flat"])]
    elif scope == "jumps_class1":
        selected = selected.loc[selected.race_type.isin(["Hurdle", "Chase"]) & class_number(selected.race_class).eq(1)]
    else:
        raise ValueError("Unknown missing-history scope")
    selected = selected.sort_values(["race_date", "horse_id"], kind="stable").drop_duplicates("horse_id")
    return [{"horse_id": str(int(r.horse_id)), "horse_name": r.horse_name,
             "first_local_date": str(r.race_date)[:10], "selection_race_type": r.race_type,
             "selection_track": r.track} for r in selected.itertuples(index=False)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["cheltenham", "jumps", "jumps_class1"], default="cheltenham")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=2)
    args = parser.parse_args()
    suffix = f"_first_{args.limit}" if args.limit is not None else ""
    output = ROOT / f"data/research/public_prior_results_{args.scope}{suffix}"
    (output / "horses").mkdir(parents=True, exist_ok=True)
    registration_path = output / "cohort.json"
    if registration_path.exists():
        encoded = registration_path.read_text()
        registration = json.loads(encoded)
        if registration["scope"] != args.scope or registration["source_fields"] != list(FIELDS):
            raise ValueError("Existing cohort has a different contract")
    else:
        columns = ["horse_id", "horse_name", "race_date", "race_type", "track",
                   "horse_prev_races", "days_since_last_run", "prior_form_runs", "race_class"]
        frame = pd.read_parquet(ROOT / "data/processed/feature_store/featured.parquet", columns=columns)
        horses = cohort(frame, args.scope)
        if args.limit is not None:
            horses = horses[:args.limit]
        registration = {"scope": args.scope, "source_fields": list(FIELDS), "horses": horses,
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection": "First local start with positive pre-race source recency or >=2 form symbols; 2021-01-01..2026-03-31; selected venue/code only; no outcomes, prices or later career records"}
        encoded = json.dumps(registration, indent=2)
        registration_path.write_text(encoded)
    selected = registration["horses"]
    # The same first-local visitor can qualify at Cheltenham and in a Class1
    # race. Reuse its frozen dated source, preserving its original fetch time.
    for horse in selected:
        target = output / "horses" / f"{horse['horse_id']}.json"
        source = ROOT / "data/research/public_prior_results_cheltenham/horses" / target.name
        if not target.exists() and source.exists():
            saved = json.loads(source.read_text())
            if all(saved.get(k) == horse[k] for k in ("horse_id", "horse_name", "first_local_date")):
                shutil.copy2(source, target)
    print(json.dumps({"stage": "registered", "horses": len(selected),
        "cohort_sha256": hashlib.sha256(encoded.encode()).hexdigest()}), flush=True)
    scraper, stopped = SportingLifeScraper(), Event()
    done = populated = total_prior = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(fetch_one, scraper, horse, stopped, output_dir=output) for horse in selected]
        for future in as_completed(pending):
            saved, _ = future.result()
            done += 1
            if "error" in saved:
                failures.append(saved)
            else:
                earlier = [r for r in saved["results"] if str(r["date"])[:10] < saved["first_local_date"]]
                populated += int(bool(earlier))
                total_prior += len(earlier)
            if done % 20 == 0 or done == len(selected):
                report = {"stage": "complete" if done == len(selected) else "collection",
                    "completed": done, "total": len(selected), "errors": len(failures),
                    "horses_with_dated_prior_records": populated, "prior_records": total_prior,
                    "model_or_database_changed": False}
                (output / "progress.json").write_text(json.dumps(report, indent=2))
                print(json.dumps(report), flush=True)
    (output / "failures.json").write_text(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
