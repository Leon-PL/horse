"""Qualify three cached explicit last-of-all-starters amendments; no requests."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.data_scraper import SportingLifeScraper
from src.database import _validate_race_results
from src.deployment_validation import file_sha256 as sha
from src.result_amendment_repair import corrected_rows

CASE = ROOT / "data/audit/result_amendments_042"
OUTPUT = CASE / "official_last_places"


def main():
    OUTPUT.mkdir(exist_ok=False)
    review = json.loads((CASE / "official_review/report.json").read_text())
    wanted = [x for x in review["all_decisions"] if x.get("reason") == "unsupported_or_nonunique_finishing_order"]
    if {x["race_id"] for x in wanted} != {"686057", "738488", "859314"}:
        raise ValueError("Use only the three reviewed explicit last-place cases")
    exchange = pd.read_parquet(CASE / "winner_census/disagreement_fields.parquet")
    exchange["_id"] = pd.to_numeric(exchange.horse_id).astype("int64")
    database = CASE / "staged_history_v2/original.db"
    sources = [CASE / "official_review/sources" / (x["race_id"] + ".json") for x in wanted]
    plan = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Explicit original winner demoted to N while genuine non-finishers retain0; normal settled parser and complete independent winner evidence; no guessed orders",
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        "implementation_sha256": {name: sha(ROOT / name) for name in
            ("scripts/qualify_disputed_last_places.py", "src/result_amendment_repair.py", "src/data_scraper.py")}}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    originals, proposals = [], []
    for decision, source in zip(wanted, sources):
        race_id = decision["race_id"]
        receipt = json.loads(source.read_text())
        page = receipt["payload"]
        race = page["props"]["pageProps"]["race"]
        summary = race["race_summary"]
        assert str(summary["race_summary_reference"]["id"]) == race_id
        parser = SportingLifeScraper()
        parser._fetch_next_data = lambda url: page
        final = pd.DataFrame(parser.scrape_race_result(decision["url"], decision["race_date"]))
        _validate_race_results(final, decision["race_date"], race_id)
        final["_id"] = pd.to_numeric(final.horse_id).astype("int64")
        with closing(sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)) as connection:
            before = pd.read_sql_query("SELECT * FROM results WHERE race_id=?", connection, params=[race_id])
        before["_id"] = pd.to_numeric(before.horse_id).astype("int64")
        if set(before._id) != set(final._id):
            raise ValueError("Changed starter population")
        original = {int(r["horse"]["horse_reference"]["id"]): r.get("finish_position") for r in race["rides"]}
        proposed = before.copy()
        aligned = final.set_index("_id").reindex(proposed._id)
        proposed["source_position"] = aligned.finish_position.to_numpy()
        proposed["source_finish_pos_label"] = aligned.finish_pos_label.to_numpy()
        proposed["result_amendment_json"] = aligned.result_amendment_json.to_numpy()
        # corrected_rows also verifies names, date, course and time against
        # these observed source values, not just the internally stored row.
        for col in ("horse_name", "race_date", "track", "off_time"):
            proposed[col] = aligned[col].to_numpy()
        proposed["num_runners"] = len(proposed)
        proposed["source_original_finish_position"] = proposed._id.map(original)
        book = exchange.loc[exchange.race_id.astype(str).eq(race_id)].set_index("_id")
        proposed["exchange_won"] = book.reindex(proposed._id).exchange_won.to_numpy()
        proposed["source_provider"] = "sporting_life_amendment"
        proposed["source_event_id"] = race_id
        proposed["source_stage"] = summary["race_stage"]
        proposed["source_receipt_sha256"] = sha(source)
        proposed["source_observed_at"] = receipt["observed_at_utc"]
        proposed = proposed.drop(columns="_id")
        before = before.drop(columns="_id")
        corrected_rows(before, proposed)
        originals.append(before)
        proposals.append(proposed)
    pd.concat(originals, ignore_index=True).to_parquet(OUTPUT / "before_db_rows.parquet", index=False)
    pd.concat(proposals, ignore_index=True).to_parquet(OUTPUT / "proposed_repairs.parquet", index=False)
    with (OUTPUT / "report.json").open("x", encoding="utf-8") as stream:
        json.dump({"completed_at_utc": datetime.now(timezone.utc).isoformat(), "eligible_fields": len(proposals),
            "eligible_runners": sum(map(len, proposals)), "new_requests": 0, "canonical_writes": False,
            "plan_sha256": sha(OUTPUT / "plan.json")}, stream, indent=2)
    print("Three explicit last-of-all-starters amendments qualified from cached official facts.", flush=True)


if __name__ == "__main__":
    main()
