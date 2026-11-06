"""Verify structured nonfinish recovery against fixed cached full results.

No network collection, database writes, feature rebuild or model fitting.
Preserve the original parser separately before changing its source.
"""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.ingestion.sporting_life import SportingLifeScraper
from src.deployment_validation import file_sha256
from src.featured_dataset import fe_signature
from src.result_status import result_labels

OUTPUT = ROOT / "data/audit/nonfinish_source_037"
SOURCE = ROOT / "data/audit/handicap_weight_review_030/sources"


def main():
    if (OUTPUT / "report.json").exists():
        raise ValueError("Preserve completed audit evidence")
    old_path = OUTPUT / "data_scraper_before.py"
    spec = importlib.util.spec_from_file_location("scraper_before_037", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    parsers = [old.SportingLifeScraper(), SportingLifeScraper()]
    paths = sorted(SOURCE.glob("*.json"))
    if len(paths) != 48:
        raise ValueError("Use the fixed 48-source roster")
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): file_sha256(p) for p in paths}
    statuses = Counter()
    reasons = Counter()
    annotations, changed = [], []
    before_results, after_results = [], []
    card_runners = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        race = payload["props"]["pageProps"]["race"]
        summary = race["race_summary"]
        race_id = str(summary["race_summary_reference"]["id"])
        if race_id != path.stem:
            raise ValueError("Source race identity mismatch")
        date = str(summary["date"])[:10]
        for ride in race["rides"]:
            statuses[str(ride.get("ride_status"))] += 1
        # Exercise both ordinary public entry points with the exact same page.
        for parser in parsers:
            parser._fetch_next_data = lambda url, payload=payload: payload
        before = parsers[0].scrape_race_result(f"https://cached.test/{race_id}", date)
        after = parsers[1].scrape_race_result(f"https://cached.test/{race_id}", date)
        if len(before) != len(after) or not after:
            raise ValueError("Changed the started runner population")
        before_results.extend(before)
        after_results.extend(after)
        pd.testing.assert_frame_equal(pd.DataFrame(before).drop(columns="finish_pos_label"),
                                      pd.DataFrame(after).drop(columns="finish_pos_label"))
        for prior, current in zip(before, after):
            if prior["finish_pos_label"] != current["finish_pos_label"]:
                changed.append({"race_id": race_id, "horse_id": current["horse_id"],
                    "horse_name": current["horse_name"], "old_label": prior["finish_pos_label"],
                    "new_label": current["finish_pos_label"]})
        for ride in race["rides"]:
            horse = ride.get("horse") or {}
            casualty = ride.get("casualty")
            reason = casualty.get("reason") if isinstance(casualty, dict) else None
            if reason:
                reasons[str(reason)] += 1
            annotations.append({"race_id": race_id, "race_date": date,
                "horse_id": str((horse.get("horse_reference") or {}).get("id", "")),
                "horse_name": horse.get("name"), "ride_status": ride.get("ride_status"),
                "raw_finish_position": str(ride.get("finish_position")), "casualty_reason": reason,
                "ride_description": ride.get("ride_description"), "source_sha256": hashes[str(path.relative_to(ROOT)).replace("\\", "/")]})
        before_card = parsers[0].scrape_racecard(f"https://cached.test/{race_id}", date)
        after_card = parsers[1].scrape_racecard(f"https://cached.test/{race_id}", date)
        pd.testing.assert_frame_equal(pd.DataFrame(before_card), pd.DataFrame(after_card))
        card_runners += len(after_card)
    before = pd.DataFrame(before_results)
    after = pd.DataFrame(after_results)
    if after.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Ambiguous parsed identities")
    # Specific reasons must preserve the existing generic nonfinisher counts.
    before_nf = result_labels(before).ne("") & before.finish_position.eq(0)
    after_nf = result_labels(after).ne("") & after.finish_position.eq(0)
    pd.testing.assert_series_equal(before_nf, after_nf)
    if len(changed) != 37 or int(after_nf.sum()) != 37:
        raise ValueError("Unexpected source-recovery count")
    if after.loc[after.finish_position.eq(0), "finish_pos_label"].eq("").any():
        raise ValueError("Cached nonfinish reason was lost")
    pd.DataFrame(annotations).to_parquet(OUTPUT / "source_annotations.parquet", index=False)
    pd.DataFrame(changed).to_csv(OUTPUT / "label_changes.csv", index=False)
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "source_files": len(paths), "source_sha256": hashes,
        "raw_rides": len(annotations), "ride_status_counts": dict(statuses),
        "parsed_started_runners": len(after), "parsed_fields": int(after.race_id.nunique()),
        "structured_reason_counts": dict(reasons), "recovered_labels": len(changed),
        "unchanged_other_result_columns": True, "unchanged_card_runners": card_runners,
        "unchanged_card_values": True, "unchanged_nonfinish_counts": True,
        "fe_signature": fe_signature(),
        "parser_before_sha256": file_sha256(old_path),
        "parser_after_sha256": file_sha256(ROOT / "src/ingestion/sporting_life.py"),
        "audit_script_sha256": file_sha256(Path(__file__)),
        "scope": "Source preservation only; no historical DB edits, rebuild, fitting or prediction-gain claim",
        "annotation_scope": "Fixed development error-review sample, not representative; dated ride_description only, no profile/commentary inference"}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
