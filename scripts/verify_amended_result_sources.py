"""Replay fixed cached source pages before/after the amended-position fix.

No requests, canonical writes, feature builds or model fits.
"""
from collections import Counter
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.data_scraper import SportingLifeScraper
from src.deployment_validation import file_sha256 as sha

CASE = ROOT / "data/audit/result_amendments_042"


def main(destination):
    if destination.exists():
        raise ValueError("Preserve the completed parser replay")
    spec = importlib.util.spec_from_file_location("scraper_before_amendments_042", CASE / "data_scraper_before.py")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    old, new = previous.SportingLifeScraper(), SportingLifeScraper()
    paths = sorted((ROOT / "data/audit/handicap_weight_review_030/sources").glob("*.json"))
    paths += sorted((ROOT / "data/audit/predictive_review_038/context/sources").glob("*.json"))
    if len(paths) != 72:
        raise ValueError("The fixed cached source roster changed")
    stages, amended_values = Counter(), Counter()
    rows, cards, changed_fields = 0, 0, []
    hashes = {}
    for path in paths:
        page = json.loads(path.read_text(encoding="utf-8"))
        race = page["props"]["pageProps"]["race"]
        summary = race["race_summary"]
        identity = str(summary["race_summary_reference"]["id"])
        if identity != path.stem:
            raise ValueError("Cached source identity mismatch")
        date = str(summary["date"])[:10]
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        stages[str(summary.get("race_stage"))] += 1
        for ride in race["rides"]:
            amended_values[str(ride.get("amended_position"))] += 1
        for parser in (old, new):
            parser._fetch_next_data = lambda url, page=page: page
        before = pd.DataFrame(old.scrape_race_result("cached", date))
        after = pd.DataFrame(new.scrape_race_result("cached", date))
        if after.result_amendment_json.notna().any():
            changed_fields.append(identity)
        else:
            pd.testing.assert_frame_equal(before, after[before.columns], check_exact=True)
        before_card = pd.DataFrame(old.scrape_racecard("cached", date))
        after_card = pd.DataFrame(new.scrape_racecard("cached", date))
        pd.testing.assert_frame_equal(before_card, after_card[before_card.columns], check_exact=True)
        assert after_card.result_amendment_json.isna().all()
        rows += len(after)
        cards += len(after_card)
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "source_fields": len(paths), "started_result_rows": rows, "card_rows": cards,
        "stage_counts": dict(stages), "amended_value_counts": dict(amended_values),
        "reordered_cached_fields": changed_fields,
        "unamended_result_values_exact": True, "all_card_values_exact": True,
        "source_sha256": hashes, "parser_before_sha256": sha(CASE / "data_scraper_before.py"),
        "parser_after_sha256": sha(ROOT / "src/data_scraper.py"),
        "script_sha256": sha(Path(__file__)), "new_requests": 0, "new_model_fits": 0}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=CASE / "final_cached_source_replay.json")
    main(cli.parse_args().output)
