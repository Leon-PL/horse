"""Freeze and collect the outstanding RTV/SL result repair roster, read-only."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.data_scraper import SportingLifeScraper, _slugify
from src.source_pipeline import _publish
from src import rtv_scraper as rtv
from src.utils import normalise_off_time_key

CASE = ROOT / "data/audit/rtv_repair_features_044"


def main():
    CASE.mkdir(parents=True, exist_ok=True)
    receipts = CASE/"sources"
    receipts.mkdir(exist_ok=True)
    con = sqlite3.connect((ROOT/"data/races.db").as_uri()+"?mode=ro", uri=True)
    conflicts = pd.read_csv(ROOT/"data/audit/rtv_pipeline_043/crosscheck/official_position_conflicts.csv", dtype={"db_race_id": str})
    ids = conflicts.db_race_id.unique().tolist()
    db = pd.read_sql_query("SELECT * FROM results WHERE race_id IN ("+",".join("?" for _ in ids)+")", con, params=ids)
    db.to_parquet(CASE/"before_conflict_rows.parquet", index=False)
    roster = []
    for race_id, frame in db.groupby("race_id"):
        row = frame.iloc[0]
        roster.append({"kind": "position_conflict", "race_id": str(race_id), "race_date": row.race_date,
            "track": row.track, "off_time": row.off_time, "race_name": row.race_name,
            "url": f"https://www.sportinglife.com/racing/results/{row.race_date}/{_slugify(row.track)}/{race_id}/{_slugify(row.race_name)}"})
    races = pd.concat([pd.read_parquet(p, columns=["race_id", "race_date", "track_name", "off_time_utc", "track_slug", "url_hhmm", "is_abandoned", "is_arab_race", "res_title"])
                       for p in sorted((ROOT/"data/rtv_cache/tables/races").glob("*.parquet"))], ignore_index=True)
    canonical = pd.read_sql_query("SELECT DISTINCT race_date, track, off_time FROM results", con)
    for frame, track, off in ((races, "track_name", "off_time_utc"), (canonical, "track", "off_time")):
        frame["_track"] = frame[track].map(rtv._track_join_key)
        frame["_off"] = frame[off].map(normalise_off_time_key)
    keys = ["race_date", "_track", "_off"]
    gaps = races.loc[~races.is_abandoned & ~races.is_arab_race].merge(canonical[keys], on=keys, how="left", indicator=True)
    gaps = gaps.loc[gaps._merge.eq("left_only")].copy()
    gaps.drop(columns="_merge").to_parquet(CASE/"missing_races.parquet", index=False)
    pending = pd.read_sql_query("SELECT * FROM result_sync_races WHERE completed=0", con)
    con.close()
    scraper = SportingLifeScraper()
    listings = {}
    for row in gaps.itertuples():
        candidates = pending.loc[pending.race_date.eq(row.race_date)
            & pending.url.str.contains('/'+str(row.track_slug).replace("-park", "")+'/', regex=False)]
        if len(candidates) != 1:
            if row.race_date not in listings:
                listings[row.race_date] = scraper.get_results_urls(row.race_date)
            candidates = [x for x in listings[row.race_date] if rtv._track_join_key(x.get("track")) == rtv._track_join_key(row.track_name)
                          and normalise_off_time_key(x.get("time")) == normalise_off_time_key(row.off_time_utc)]
        else:
            candidates = candidates.to_dict("records")
        identity = str(candidates[0]["race_id"]) if len(candidates) == 1 else None
        roster.append({"kind": "missing_race", "race_id": identity, "rtv_race_id": int(row.race_id),
            "race_date": row.race_date, "track": row.track_name, "off_time": row.off_time_utc,
            "race_name": row.res_title, "url": candidates[0]["url"] if identity else None})
    _publish(CASE/"review_roster.json", {"created_at": datetime.now(timezone.utc).isoformat(), "roster": roster})
    def fetch(item):
        if not item["url"]:
            return {**item, "state": "not_listed_by_sl"}
        path = receipts/(str(item["race_id"])+".json")
        if path.exists():
            return {**item, "state": "cached"}
        page = scraper._fetch_next_data(item["url"])
        if not page:
            raise ValueError("Empty SL page: "+item["url"])
        _publish(path, {"url": item["url"], "observed_at": datetime.now(timezone.utc).isoformat(), "page": page})
        return {**item, "state": "collected"}
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(fetch, item): item for item in roster}
        for future in as_completed(jobs):
            item = jobs[future]
            try:
                result = future.result()
            except Exception as error:
                result = {**item, "state": "error", "error": str(error)}
            results.append(result)
            print(json.dumps({k: result.get(k) for k in ("kind", "race_id", "state", "error")}), flush=True)
    _publish(CASE/"collection.json", {"completed_at": datetime.now(timezone.utc).isoformat(), "counts": dict(Counter(x["state"] for x in results)), "results": results})


if __name__ == "__main__":
    main()
