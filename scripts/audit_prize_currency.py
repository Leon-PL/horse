"""Verify purse units at every stored venue from early, middle and late races.

Selection uses venue/date and positive recorded purses only, never outcomes.
Only race identity and the source prizes are cached; no horse profile fields
or future results are imported. This script does not alter historical inputs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.data_scraper import SportingLifeScraper, _slugify

OUTPUT = ROOT / "data/audit/prize_currency_020"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(row, scraper):
    identity = str(row["race_id"])
    date = str(row["race_date"])[:10]
    url = f"https://www.sportinglife.com/racing/results/{date}/{_slugify(row['track'])}/{identity}/{_slugify(row['race_name'])}"
    target = OUTPUT / "source_races" / f"{identity}.json"
    if target.exists():
        result = json.loads(target.read_text())
    else:
        try:
            data = scraper._fetch_next_data(url)
            race = (data or {}).get("props", {}).get("pageProps", {}).get("race") or {}
            summary = race.get("race_summary") or {}
            result = {"requested_race_id": identity, "source_url": url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "race_id": str((summary.get("race_summary_reference") or {}).get("id")),
                "race_date": summary.get("date"), "track": summary.get("course_name"),
                "prizes": race.get("prizes")}
        except Exception as error:
            return {**row, "error": str(error), "source_url": url}
        if result["race_id"] != identity or result["race_date"] != date or _slugify(result["track"] or "") != _slugify(row["track"]):
            return {**row, "error": "Source race identity mismatch", "source_url": url}
        target.write_text(json.dumps(result, indent=2))
    if result["race_id"] != identity or result["race_date"] != date or _slugify(result["track"] or "") != _slugify(row["track"]):
        return {**row, "error": "Cached race identity mismatch", "source_url": url}
    prizes = result.get("prizes") or {}
    prizes = prizes.get("prize", []) if isinstance(prizes, dict) else prizes
    first = [p for p in prizes if str(p.get("position")) == "1"]
    if len(first) != 1:
        return {**row, "error": "Missing or ambiguous first prize", "source_url": url}
    text = str(first[0].get("prize", ""))
    match = re.fullmatch(r"\s*([0-9.,]+)\s+(GBP|EUR)\s*", text)
    if not match:
        return {**row, "error": f"Unsupported prize units: {text}", "source_url": url}
    amount = float(match.group(1).replace(",", ""))
    return {**row, "source_url": url, "source_sha256": sha(target), "currency": match.group(2),
            "source_first_prize": amount, "stored_first_prize": row["prize_money"],
            "amount_difference": amount-row["prize_money"]}


def main():
    started = time.perf_counter()
    (OUTPUT / "source_races").mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/processed/feature_store/processed.parquet"
    frame = pd.read_parquet(source, columns=["race_id", "race_date", "track", "race_name", "prize_money"])
    races = frame.loc[frame.prize_money.gt(0)].drop_duplicates("race_id").sort_values(["race_date", "race_id"])
    samples = []
    for _, group in races.groupby("track", sort=True):
        positions = sorted(set((0, len(group)//2, len(group)-1)))
        samples.extend(group.iloc[positions].to_dict("records"))
    selected = OUTPUT / "selected_races.json"
    encoded = json.dumps(samples, sort_keys=True, indent=2, default=str)
    if selected.exists() and selected.read_text() != encoded:
        raise ValueError("Source selection changed; preserve the existing audit")
    selected.write_text(encoded)
    scraper = SportingLifeScraper()
    result = []
    print(json.dumps({"stage": "selected", "venues": len({r['track'] for r in samples}), "source_races": len(samples)}), flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch, row, scraper) for row in samples]
        for future in as_completed(futures):
            result.append(future.result())
            if len(result) % 10 == 0:
                print(json.dumps({"completed": len(result), "errors": sum("error" in r for r in result)}), flush=True)
    table = pd.DataFrame(result).sort_values(["track", "race_date"])
    table.to_csv(OUTPUT / "source_currency_audit.csv", index=False)
    mapping, unresolved = {}, {}
    for track, group in table.groupby("track"):
        known = group.loc[group.currency.notna()] if "currency" in group else group.iloc[:0]
        if len(known) >= 2 and known.currency.nunique() == 1:
            mapping[track] = known.currency.iloc[0]
        else:
            unresolved[track] = group.to_dict("records")
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "source": "Sporting Life explicit first-prize currency strings",
        "scope": "Venue unit verification from early/middle/late source samples; not an exhaustive per-race currency census",
        "selection_sha256": sha(selected), "source_code_sha256": sha(Path(__file__)),
        "source_meta_sha256": sha(source.with_name("meta.json")),
        "sample_races": len(samples), "verified_venues": len(mapping), "currency_by_track": mapping,
        "unresolved": unresolved, "error_races": int(table.get("error", pd.Series(dtype=str)).notna().sum()),
        "amount_disagreements": int(table.get("amount_difference", pd.Series(dtype=float)).abs().gt(.01).sum()),
        "audit_table_sha256": sha(OUTPUT / "source_currency_audit.csv"), "elapsed_seconds": time.perf_counter()-started}
    (OUTPUT / "source_currency_audit.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
