"""Cache public prior results for a fixed, outcome-free missing-history cohort.

This does not add partial fields to the race database, change pedigrees, or
consume current profile ratings/career totals. It is a separate data pilot.
Existing per-horse caches survive a shutdown; requests share the scraper's
global rate limiter. Future feature work must filter every record by date.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
from threading import Event

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.data_scraper import SportingLifeScraper
from src.utils import normalise_horse_key

OUTPUT = ROOT / "data/research/public_prior_results"
FIELDS = ("race_id", "horse_id", "horse_name", "date", "time", "course_name", "run_type",
          "distance", "position", "casualty", "runner_count", "going", "result_between_distance")


def sanitise(profile, horse_id, horse_name):
    reference = profile.get("horse_reference") or {}
    if str(reference.get("id")) != str(horse_id):
        raise ValueError("Profile identity does not match requested horse")
    if normalise_horse_key(profile.get("name", "")) != normalise_horse_key(horse_name):
        raise ValueError("Profile name does not match requested horse")
    rows = []
    seen = {}
    for original in profile.get("previous_results") or []:
        if str(original.get("horse_id")) != str(horse_id):
            raise ValueError("A performance belongs to a different horse")
        date = pd.Timestamp(original.get("date"))
        if pd.isna(date) or date.tzinfo is not None:
            raise ValueError("Performance date is unavailable or ambiguous")
        if date.normalize() > pd.Timestamp.now().normalize():
            raise ValueError("A claimed past result has a future date")
        row = {k: original.get(k) for k in FIELDS}
        key = str(row["race_id"])
        if key in seen and seen[key] != row:
            raise ValueError("Conflicting duplicate performances")
        if key not in seen:
            rows.append(row)
            seen[key] = row
    return sorted(rows, key=lambda r: (r["date"], str(r["race_id"])))


def cohort():
    columns = ["horse_id", "horse_name", "race_date", "race_type", "age",
               "horse_prev_races", "has_hidden_form", "prior_form_runs"]
    f = pd.read_parquet(ROOT / "data/processed/feature_store/featured.parquet", columns=columns)
    selected = f.loc[f.race_date.between("2021-01-01", "2026-03-31")
        & f.horse_prev_races.eq(0) & f.has_hidden_form.eq(1) & f.age.ge(4)
        & f.prior_form_runs.ge(3) & f.race_type.eq("Flat")]
    selected = selected.sort_values(["race_date", "horse_id"], kind="stable").drop_duplicates("horse_id")
    return [{"horse_id": str(int(r.horse_id)), "horse_name": r.horse_name,
             "first_local_date": str(r.race_date)[:10]}
            for r in selected.itertuples(index=False)]


def fetch_one(scraper, horse, stopped=None, *, output_dir=None):
    target = Path(output_dir or OUTPUT) / "horses" / f"{horse['horse_id']}.json"
    if target.exists():
        saved = json.loads(target.read_text())
        if saved["horse_id"] != horse["horse_id"] or saved["horse_name"] != horse["horse_name"]:
            raise ValueError("Cached identity differs")
        return saved, True
    if stopped is not None and stopped.is_set():
        return {**horse, "error": "Collection stopped after an access/rate-limit response"}, False
    url = f"https://www.sportinglife.com/racing/profiles/horse/{horse['horse_id']}"
    try:
        data = scraper._fetch_next_data(url)
        profile = (data or {}).get("props", {}).get("pageProps", {}).get("profile") or {}
        rows = sanitise(profile, horse["horse_id"], horse["horse_name"])
        saved = {**horse, "source_url": url, "fetched_at": datetime.now(timezone.utc).isoformat(),
                 "results": rows, "scope": "Historical results retrieved later; current profile ratings and lifetime statistics excluded"}
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(saved, separators=(",", ":")))
        temporary.replace(target)
        return saved, False
    except Exception as error:
        if stopped is not None and any(code in str(error) for code in ("403 Client Error", "429 Client Error")):
            stopped.set()
        return {**horse, "error": f"{type(error).__name__}: {error}"}, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="optional smaller outcome-free pilot")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "horses").mkdir(exist_ok=True)
    name = "cohort.json" if args.limit is None else f"cohort_first_{args.limit}.json"
    path = OUTPUT / name
    if path.exists():
        encoded = path.read_text()
        registration = json.loads(encoded)
        if registration["source_fields"] != list(FIELDS):
            raise ValueError("Cached source schema changed; preserve the earlier pilot")
        selected = registration["horses"]
    else:
        selected = cohort()
        if args.limit is not None:
            selected = selected[:args.limit]
        registration = {"selection": "First cohort-eligible Flat row in2021-01-01..2026-03-31, age>=4, source prior-form-count>=3, missing recorded history; no outcomes or prices used",
                        "horses": selected, "workers": args.workers, "source_fields": list(FIELDS)}
        encoded = json.dumps(registration, indent=2)
        path.write_text(encoded)
    print(json.dumps({"stage": "registered", "horses": len(selected), "cohort_sha256": hashlib.sha256(encoded.encode()).hexdigest()}), flush=True)
    scraper = SportingLifeScraper()
    stopped = Event()
    done, failures, before, populated, reused = 0, [], 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, scraper, horse, stopped) for horse in selected]
        for future in as_completed(futures):
            saved, cached = future.result()
            done += 1
            reused += int(cached)
            if "error" in saved:
                failures.append(saved)
            else:
                prior = [r for r in saved["results"] if str(r["date"])[:10] < saved["first_local_date"]]
                before += len(prior)
                populated += int(bool(prior))
            if done % 50 == 0 or done == len(selected):
                progress = {"stage": "collection", "completed": done, "total": len(selected),
                    "errors": len(failures), "reused": reused, "horses_with_prior_results": populated,
                    "results_before_first_local_start": before}
                (OUTPUT / "progress.json").write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
    report = {"stage": "complete", "horses": len(selected), "completed": done, "reused": reused,
              "horses_with_prior_results": populated, "results_before_first_local_start": before,
              "failures": failures, "model_features_changed": False}
    (OUTPUT / "coverage_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({**report, "failures": len(failures)}), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
