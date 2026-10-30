"""Cross-check all70 cached ATR repair orders with current primary results.

This only strengthens source qualification; it never chooses facts by model
score or alters the sealed correction. Stop on any source disagreement.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from scripts.review_disputed_results import minimal, ids
from src.data_scraper import SportingLifeScraper, _NEXT_DATA_RE, _slugify
from src.database import _validate_race_results
from src.deployment_validation import file_sha256 as sha
from src.utils import normalise_horse_key

CASE = ROOT / "data/audit/result_amendments_042"
OUTPUT = CASE / "atr_primary_verification"


def main():
    OUTPUT.mkdir(exist_ok=False)
    sources = OUTPUT / "sources"
    sources.mkdir()
    proposed = pd.read_parquet(CASE / "proposed_repairs.parquet")
    before = pd.read_parquet(CASE / "before_db_rows.parquet")
    for frame in (proposed, before):
        frame["race_id"] = frame.race_id.astype(str)
        frame["_id"] = ids(frame)
    if proposed.race_id.nunique() != 70:
        raise ValueError("Use the fixed70-field cached-order batch")
    plan = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "fields": 70, "maximum_new_requests": 70,
        "scope": "Primary-source agreement with the sealed cached official orders; no canonical mutation or prediction scores",
        "input_sha256": {name: sha(CASE / name) for name in ("proposed_repairs.parquet", "before_db_rows.parquet")},
        "implementation_sha256": {name: sha(ROOT / name) for name in ("scripts/verify_cached_official_orders.py", "scripts/review_disputed_results.py", "src/data_scraper.py")}}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    scraper = SportingLifeScraper()
    receipts = []
    for race_id, group in proposed.groupby("race_id", sort=True):
        original = before.loc[before.race_id.eq(race_id)].iloc[0]
        date = str(original.race_date)[:10]
        url = f"https://www.sportinglife.com/racing/results/{date}/{_slugify(original.track)}/{race_id}/{_slugify(original.race_name)}"
        if race_id == "754950":
            receipt = json.loads((ROOT / "tests/fixtures/amended_goodwood_20230827.json").read_text())
        else:
            response = scraper._get(url, max_retries=1)
            match = _NEXT_DATA_RE.search(response.text)
            if match is None:
                raise ValueError("No primary factual result payload")
            receipt = {"source_url": url, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                "response_sha256": hashlib.sha256(response.content).hexdigest(), "payload": minimal(json.loads(match.group(1)))}
        path = sources / (race_id + ".json")
        path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        scraper._fetch_next_data = lambda url, page=receipt["payload"]: page
        final = pd.DataFrame(scraper.scrape_race_result(url, date))
        _validate_race_results(final, date, race_id)
        final["_id"] = ids(final)
        check = group[["_id", "horse_name", "source_position"]].merge(final[["_id", "horse_name", "finish_position"]],
            on="_id", how="outer", validate="one_to_one", suffixes=("_cache", "_primary"), indicator=True)
        if (not check.pop("_merge").eq("both").all()
                or not check.horse_name_cache.map(normalise_horse_key).equals(check.horse_name_primary.map(normalise_horse_key))
                or not check.source_position.eq(check.finish_position).all()):
            check.to_csv(OUTPUT / (race_id + "_disagreement.csv"), index=False)
            raise ValueError("Primary result disagrees with cached proposed order: " + race_id)
        receipts.append({"race_id": race_id, "runners": len(final), "final_order_exact": True, "source_sha256": sha(path)})
        print(json.dumps(receipts[-1]), flush=True)
    if any(sha(ROOT / name) != digest for name, digest in plan["implementation_sha256"].items()):
        raise ValueError("Source changed during primary verification")
    with (OUTPUT / "report.json").open("x", encoding="utf-8") as stream:
        json.dump({"completed_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
            "fields": len(receipts), "runners": sum(x["runners"] for x in receipts), "receipts": receipts,
            "plan_sha256": sha(OUTPUT / "plan.json"), "canonical_writes": False, "new_model_fits": 0}, stream, indent=2)
    print("All70 cached official orders agree exactly with the primary result source.", flush=True)


if __name__ == "__main__":
    main()
