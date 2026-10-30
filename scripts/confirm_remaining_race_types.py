"""Confirm remaining parser/distance leads against dated public performances.

No historical rewrite or training. One runner per event is chosen by ID,
without using its finish or prices. Raw profiles and receipts survive restart.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.data_scraper import SportingLifeScraper, _infer_race_type
from src.deployment_validation import file_sha256
from src.utils import normalise_horse_key

OUTPUT = ROOT / "data/audit/trip_suitability_032"
CODE = {"FLAT":"Flat", "AW FLAT":"Flat", "CHASE":"Chase", "HURDLE":"Hurdle",
        "HURDL":"Hurdle", "NH FLAT":"Flat", "NHF":"Flat", "BUMPER":"Flat"}


def confirmed_record(payload, row):
    profile = payload["props"]["pageProps"]["profile"]
    if str(profile["horse_reference"]["id"]) != row["horse_id"]:
        raise ValueError("Profile identity differs")
    if normalise_horse_key(profile["name"]) != normalise_horse_key(row["horse_name"]):
        raise ValueError("Profile name differs")
    matches = [r for r in profile.get("previous_results", []) if str(r.get("race_id")) == row["race_id"]]
    if not matches:
        raise ValueError("Requested event absent from returned historical performances")
    observations = []
    for result in matches:
        if str(result.get("horse_id")) != row["horse_id"] or str(result.get("date"))[:10] != row["date"]:
            raise ValueError("Performance identity/date differs")
        if str(result.get("course_name", "")).strip().casefold() != row["track"].strip().casefold():
            raise ValueError("Performance course differs")
        kind = CODE.get(str(result.get("run_type") or "").upper().strip())
        if kind is None:
            raise ValueError("No recognised structured historical race code")
        observations.append(kind)
    if len(set(observations)) != 1:
        raise ValueError("Historical performances conflict")
    return observations[0]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "profiles"
    cache.mkdir(exist_ok=True)
    f = pd.read_parquet(ROOT / "data/processed/feature_store/processed.parquet",
        columns=["race_id", "horse_id", "horse_name", "race_date", "race_name", "track", "race_type", "distance_furlongs"],
        filters=[("race_date", "<=", pd.Timestamp("2026-03-31"))])
    f["race_id"] = f.race_id.astype(str)
    f["horse_id"] = pd.to_numeric(f.horse_id, errors="raise").astype("Int64").astype(str)
    fields = f.sort_values(["race_id", "horse_id"], kind="stable").drop_duplicates("race_id").copy()
    fields["parser_type"] = fields.race_name.map(_infer_race_type)
    known = json.loads((ROOT / "reference/verified_race_types.json").read_text())["races"]
    selected = fields.loc[(fields.parser_type.ne(fields.race_type)
        | (fields.race_type.isin(["Hurdle", "Chase"]) & fields.distance_furlongs.lt(13)))
        & ~fields.race_id.isin(known)].copy()
    selected["date"] = selected.race_date.dt.strftime("%Y-%m-%d")
    roster = OUTPUT / "remaining_type_leads.csv"
    if roster.exists():
        selected = pd.read_csv(roster, dtype={"race_id":str, "horse_id":str})
    else:
        selected.to_csv(roster, index=False)
    scraper = SportingLifeScraper()
    receipts, corrections = [], {}
    for i, row in enumerate(selected.to_dict("records")):
        url = f"https://www.sportinglife.com/racing/profiles/horse/{row['horse_id']}"
        path = cache / (row["horse_id"]+".json")
        receipt = {"race_id":row["race_id"], "horse_id":row["horse_id"], "url":url,
            "read_at_utc":datetime.now(timezone.utc).isoformat(), "cache_hit":path.exists()}
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                payload = scraper._fetch_next_data(url)
                path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            kind = confirmed_record(payload, row)
            receipt.update(passed=True, source_type=kind, source_sha256=file_sha256(path))
            if kind != row["race_type"]:
                corrections[row["race_id"]] = {"date":row["date"], "track":row["track"],
                    "race_type":kind, "race_name":row["race_name"], "source_url":url,
                    "source_horse_id":row["horse_id"], "source_sha256":file_sha256(path)}
        except Exception as exc:
            receipt.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        receipts.append(receipt)
        (OUTPUT / "type_source_receipts.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
        (OUTPUT / "source_confirmed_race_types.json").write_text(json.dumps({"races":corrections}, indent=2), encoding="utf-8")
        print(json.dumps({"completed":i+1, "total":len(selected), **receipt}), flush=True)
        if not receipt["passed"] and any(c in receipt["error"] for c in ("403", "429")):
            break
    print(json.dumps({"leads":len(selected), "checked":len(receipts), "confirmed_corrections":len(corrections),
        "failed":sum(not r["passed"] for r in receipts), "production_changed":False}), flush=True)


if __name__ == "__main__":
    main()
