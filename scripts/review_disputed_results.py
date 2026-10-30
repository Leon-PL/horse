"""Bounded official-source review of the remaining exchange disagreements.

Use the existing result provider and normal request limits. Keep only factual
result fields needed to verify amendments. Stop on access/rate-limit errors;
no challenge handling, provider switching or inferred settlement labels.
"""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.data_scraper import SportingLifeScraper, _NEXT_DATA_RE, _slugify
from src.database import _validate_race_results
from src.deployment_validation import file_sha256 as sha
from src.utils import normalise_horse_key, normalise_track_key, normalise_off_time_key

CASE = ROOT / "data/audit/result_amendments_042"
OUTPUT = CASE / "official_review"
DATABASE = CASE / "staged_history_v2/original.db"
SUMMARY_FIELDS = ("race_summary_reference", "date", "name", "course_name", "going", "distance", "race_class",
    "time", "off_time", "winning_time", "ride_count", "has_handicap", "course_surface", "run_type", "race_stage")
RIDE_FIELDS = ("ride_status", "finish_position", "amended_position", "finish_distance", "casualty")


def minimal(page):
    props = page["props"]["pageProps"]
    race = props["race"]
    summary = {key: race["race_summary"][key] for key in SUMMARY_FIELDS if key in race["race_summary"]}
    rides = []
    for ride in race["rides"]:
        row = {key: ride[key] for key in RIDE_FIELDS if key in ride}
        horse = ride.get("horse") or {}
        row["horse"] = {key: horse[key] for key in ("horse_reference", "name", "age", "sex") if key in horse}
        rides.append(row)
    return {"props": {"pageProps": {"race": {"race_summary": summary, "rides": rides},
                                   "isUkorIreMeeting": props.get("isUkorIreMeeting", True)}}}


def ids(frame):
    values = pd.to_numeric(frame.horse_id, errors="raise")
    if values.isna().any() or values.le(0).any() or values.mod(1).ne(0).any():
        raise ValueError("missing_or_invalid_source_horse_id")
    return values.astype("int64")


def main():
    OUTPUT.mkdir(exist_ok=True)
    sources_dir = OUTPUT / "sources"
    sources_dir.mkdir(exist_ok=True)
    if (OUTPUT / "report.json").exists():
        raise ValueError("Preserve the completed official review")
    roster = pd.read_csv(CASE / "winner_census/disagreement_fields.csv", dtype={"race_id": str})
    roster = roster.loc[roster.still_disagrees].sort_values(["race_date", "race_id"])
    if len(roster) != 97:
        raise ValueError("Use the fixed97-field residual roster")
    paths = ["scripts/review_disputed_results.py", "src/data_scraper.py", "src/database.py", "src/utils.py"]
    plan = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "race_ids": roster.race_id.tolist(),
        "maximum_new_requests": 97, "maximum_attempts_per_field": 1, "source_provider": "Sporting Life",
        "policy": "Full identical source IDs/names/date/course/time/starter population; weighed-in final order; original positions reproduce stored positions; final winner agrees with the complete independent exchange field. No canonical mutation.",
        "source_sha256": {name: sha(ROOT / name) for name in paths},
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in
            (DATABASE, CASE / "winner_census/disagreement_fields.parquet", CASE / "winner_census/disagreement_fields.csv")}}
    plan_path = OUTPUT / "plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if any(previous[key] != plan[key] for key in ("race_ids", "source_sha256", "input_sha256")):
            raise ValueError("Official-review roster or sources changed")
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    exchange = pd.read_parquet(CASE / "winner_census/disagreement_fields.parquet")
    exchange["race_id"] = exchange.race_id.astype(str)
    exchange["_id"] = ids(exchange)
    scraper = SportingLifeScraper()
    decisions, proposals, before_rows = [], [], []
    for row in roster.to_dict("records"):
        race_id = row["race_id"]
        with closing(sqlite3.connect(f"file:{DATABASE.as_posix()}?mode=ro", uri=True)) as connection:
            before = pd.read_sql_query("SELECT * FROM results WHERE race_id=?", connection, params=[race_id])
        before["race_id"] = before.race_id.astype(str)
        before["_id"] = ids(before)
        date = str(before.race_date.iloc[0])[:10]
        url = f"https://www.sportinglife.com/racing/results/{date}/{_slugify(before.track.iloc[0])}/{race_id}/{_slugify(before.race_name.iloc[0])}"
        path = sources_dir / (race_id + ".json")
        decision = {"race_id": race_id, "race_date": date, "url": url}
        try:
            if path.exists():
                receipt = json.loads(path.read_text())
            elif race_id == "601907":
                receipt = json.loads((ROOT / "tests/fixtures/amended_doncaster_20201211.json").read_text())
                path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            else:
                response = scraper._get(url, max_retries=1)
                match = _NEXT_DATA_RE.search(response.text)
                if match is None:
                    raise ValueError("missing_next_data")
                receipt = {"source_url": url, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "response_sha256": hashlib.sha256(response.content).hexdigest(),
                    "scope": "Minimal factual result fields; no profile/commentary text", "payload": minimal(json.loads(match.group(1)))}
                path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            payload = receipt["payload"]
            race = payload["props"]["pageProps"]["race"]
            summary = race["race_summary"]
            if str(summary["race_summary_reference"]["id"]) != race_id or str(summary["date"])[:10] != date:
                raise ValueError("source_race_or_date_mismatch")
            scraper._fetch_next_data = lambda url, payload=payload: payload
            final = pd.DataFrame(scraper.scrape_race_result(url, date_str=date))
            _validate_race_results(final, date, race_id)
            final["_id"] = ids(final)
            if len(final) != len(before) or final._id.duplicated().any() or set(final._id) != set(before._id):
                raise ValueError("different_or_partial_starter_population")
            original = pd.DataFrame([{"_id": int((r.get("horse") or {})["horse_reference"]["id"]),
                "source_original_finish_position": pd.to_numeric(r.get("finish_position"), errors="coerce")}
                for r in race["rides"] if (r.get("horse") or {}).get("horse_reference")])
            combined = before.merge(final[["_id", "horse_name", "track", "off_time", "finish_position", "finish_pos_label", "result_amendment_json"]],
                on="_id", suffixes=("", "_official"), validate="one_to_one")
            combined = combined.merge(original, on="_id", validate="one_to_one")
            if not combined.horse_name.map(normalise_horse_key).equals(combined.horse_name_official.map(normalise_horse_key)):
                raise ValueError("horse_name_disagrees_with_source_id")
            for column, normalize in (("track", normalise_track_key), ("off_time", normalise_off_time_key)):
                if not combined[column].map(normalize).equals(combined[column+"_official"].map(normalize)):
                    raise ValueError("course_or_time_mismatch")
            if not combined.finish_position.eq(combined.source_original_finish_position).all():
                raise ValueError("stored_order_differs_from_original_source_order")
            if combined.result_amendment_json.isna().any():
                raise ValueError("no_source_amendment")
            book = exchange.loc[exchange.race_id.eq(race_id), ["_id", "exchange_won"]]
            combined = combined.merge(book, on="_id", how="left", validate="one_to_one")
            if not combined.exchange_won.isin([0, 1]).all() or not combined.finish_position_official.eq(1).equals(combined.exchange_won.eq(1)):
                raise ValueError("official_winner_disagrees_with_exchange")
            for positions in (combined.finish_position, combined.finish_position_official):
                positive = positions.loc[positions.gt(0)].tolist()
                if positions.isna().any() or positions.lt(0).any() or sorted(positive) != list(range(1, len(positive)+1)):
                    raise ValueError("unsupported_or_nonunique_finishing_order")
            combined["source_position"] = combined.finish_position_official
            combined["source_finish_pos_label"] = combined.finish_pos_label_official
            combined["source_provider"] = "sporting_life_amendment"
            combined["source_event_id"] = race_id
            combined["source_stage"] = summary["race_stage"]
            combined["source_receipt_sha256"] = sha(path)
            combined["source_observed_at"] = receipt["observed_at_utc"]
            combined["corrected_won"] = combined.source_position.eq(1).astype(int)
            combined["corrected_finish_position"] = combined.source_position
            combined["corrected_lengths_behind"] = np.where(combined.corrected_won.eq(1), 0., np.nan)
            proposals.append(combined.drop(columns="_id"))
            before_rows.append(before.drop(columns="_id"))
            decision.update({"decision": "eligible", "runners": len(combined),
                "position_changes": int(combined.finish_position.ne(combined.source_position).sum()),
                "source_receipt_sha256": sha(path)})
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            decision.update({"decision": "quarantined", "reason": str(error)[:300], "http_status": status})
            if status in (401, 403, 429) or (status is not None and status >= 500):
                decisions.append(decision)
                print(json.dumps(decision), flush=True)
                break
        decisions.append(decision)
        print(json.dumps(decision), flush=True)
    pd.DataFrame(decisions).to_csv(OUTPUT / "field_decisions.csv", index=False)
    if proposals:
        pd.concat(proposals, ignore_index=True).to_parquet(OUTPUT / "proposed_repairs.parquet", index=False)
        pd.concat(before_rows, ignore_index=True).to_parquet(OUTPUT / "before_db_rows.parquet", index=False)
    if any(sha(ROOT / name) != digest for name, digest in plan["source_sha256"].items()):
        raise ValueError("Source changed during review")
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "planned_fields": 97,
        "reviewed_fields": len(decisions), "eligible_fields": len(proposals),
        "eligible_runners": sum(map(len, proposals)), "quarantined_fields": sum(d["decision"] != "eligible" for d in decisions),
        "all_decisions": decisions, "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources_dir.glob("*.json")},
        "plan_sha256": sha(plan_path), "canonical_writes": False, "new_model_fits": 0}
    with (OUTPUT / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ("all_decisions", "source_sha256")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
