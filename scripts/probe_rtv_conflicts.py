"""Read-only live SL spot checks of the RTV official-order disagreement queue."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.data_scraper import SportingLifeScraper
from src.source_pipeline import _publish, _read
from src.utils import normalise_horse_key


def main():
    root = ROOT / "data/audit/rtv_pipeline_043"
    conflicts = pd.read_csv(root/"crosscheck/official_position_conflicts.csv", dtype={"db_race_id": str})
    chosen = conflicts.loc[conflicts.race_date.ge("2026-10-01")].drop_duplicates("db_race_id").tail(5)
    scraper, reports = SportingLifeScraper(), []
    with sqlite3.connect((ROOT/"data/races.db").as_uri()+"?mode=ro", uri=True) as con:
        for row in chosen.itertuples():
            url = con.execute("SELECT url FROM result_sync_races WHERE race_id=?", (row.db_race_id,)).fetchone()
            if url:
                url = url[0]
            else:
                url = next(x["url"] for x in scraper.get_results_urls(row.race_date)
                           if str(x["race_id"]) == row.db_race_id)
            receipt = root/"spot_checks"/(row.db_race_id+".json")
            page = (_read(receipt)["page"] if receipt.exists() else scraper._fetch_next_data(url))
            if not receipt.exists():
                _publish(receipt, {"url": url, "observed_at": datetime.now(timezone.utc).isoformat(), "page": page})
            original = scraper._fetch_next_data
            scraper._fetch_next_data = lambda _, page=page: page
            try:
                current = pd.DataFrame(scraper.scrape_race_result(url, row.race_date))
            except Exception as error:
                reports.append({"date": row.race_date, "track": row.track_name, "race_id": row.db_race_id,
                    "url": url, "fresh_sl_parse_error": str(error), "source_receipt": str(receipt)})
                continue
            finally:
                scraper._fetch_next_data = original
            current["_horse"] = current.horse_name.map(normalise_horse_key)
            old = conflicts.loc[conflicts.db_race_id.eq(row.db_race_id)].copy()
            old["_horse"] = old.horse_name.map(normalise_horse_key)
            joined = old.merge(current[["_horse", "finish_position"]], on="_horse", suffixes=("", "_fresh_sl"), validate="one_to_one")
            reports.append({"date": row.race_date, "track": row.track_name, "race_id": row.db_race_id,
                "url": url, "compared_conflicts": len(joined),
                "fresh_sl_agrees_database": int(joined.finish_position_fresh_sl.eq(joined.db_pos).sum()),
                "fresh_sl_agrees_rtv_final": int(joined.finish_position_fresh_sl.eq(joined.final_position).sum()),
                "runners": joined[["horse_name", "db_pos", "final_position", "finish_position_fresh_sl"]].to_dict("records")})
    _publish(root/"spot_checks/summary.json", {"checks": reports, "canonical_writes": 0})
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
