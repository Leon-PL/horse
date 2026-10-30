"""Collect exact primary-source identities for the two residual Thoroughbred cases."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_scraper import SportingLifeScraper
from src.rtv_scraper import RTVClient
from src.source_pipeline import _publish


def main():
    root = ROOT/"data/audit/rtv_repair_features_044/sources"
    client = RTVClient()
    status, body = client.get("/horses/unbreakable-duke-ireland")
    page = json.loads(body)
    _publish(root/"rtv_horse_494668_verified.json", {"status": status, "observed_at": datetime.now(timezone.utc).isoformat(), "page": page})
    horse = page.get("horse", {})
    print({k: v for k, v in horse.items() if "id" in k or "name" in k or "ref" in k})
    page = SportingLifeScraper()._fetch_next_data("https://www.sportinglife.com/racing/profiles/horse/1095679")
    _publish(root/"horse_1095679.json", page)
    profile = page["props"]["pageProps"].get("profile", {})
    found = [x for x in profile.get("previous_results", []) if str(x.get("date", "")).startswith("2024-07-05")]
    print(found)
    if found:
        identity = str(found[0]["race_id"])
        url = f"https://www.sportinglife.com/racing/results/2024-07-05/bellewstown/{identity}/dore-co-solicitors-handicap-div-1"
        page = SportingLifeScraper()._fetch_next_data(url)
        _publish(root/(identity+".json"), {"url": url, "observed_at": datetime.now(timezone.utc).isoformat(), "page": page})


if __name__ == "__main__":
    main()
