"""
Build ``reference/connection_aliases.json``: one name per jockey and trainer.
============================================================================

Sporting Life changes how it displays a rider or trainer ("M P Sheehy" became
"Mikey Sheehy"; amateurs lose their "Mr"), and ``results`` stores the name
shown when each page was scraped. Jockey and trainer histories are keyed by
name, so one person's record can be split across two names used side by side
for years.

Racing TV publishes a numeric ID for every jockey and trainer. Runners are
linked to Sporting Life rows with the exact date/course/off-time/horse join
(``src.source_pipeline.match_runners``). RTV IDs are only stable within an
era: they were re-issued around January 2024 ("Luke Morris" is 22, then 1773)
and some riders had a temporary second ID in late 2025. So names and IDs are
joined into people (connected components), with three guards:

* a name-ID link needs ``MIN_RIDES`` rides and ``MIN_EDGE_SHARE`` of the name;
* a name whose IDs are active at the same time may be two people, so it is
  never used as a bridge and never remapped;
* a person is left alone unless every pair of their names shares a surname
  with compatible initials ("B W Harvey" fits "Ben Harvey"; "B J Murphy" and
  "B T Murphy" are two trainers that RTV filed under one ID). A joint licence
  passes when one holder matches ("I Mohammed" fits "I Mohammed & J Santos").

The canonical name of a person is the most recently used Sporting Life name,
so live cards (which show the current name) need no mapping.

Usage:
    python scripts/build_connection_aliases.py            # report only
    python scripts/build_connection_aliases.py --write    # write the reference file
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from src.source_pipeline import match_runners  # noqa: E402

OUT = ROOT / "reference" / "connection_aliases.json"
TABLES = Path(config.DATA_DIR) / "rtv_cache" / "tables"
MIN_RIDES = 3          # rides needed before a name-ID link counts
MIN_EDGE_SHARE = 0.05  # share of a name's linked rides a link must carry
OVERLAP_DAYS = 30      # IDs of one name active together for longer = two people
ROLES = {"jockey": ("res_jockey__id", "card_jockey_id"), "trainer": ("res_trainer__id", "card_trainer_id")}
# Groups that fail the initials rule but were checked by hand (one ID, names in sequence).
CONFIRMED = [{"Liz Doyle", "Miss E Doyle", "Miss Liz Doyle"}]  # Elizabeth Doyle, renamed 2023-05


def _read(name: str, columns: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(path, columns=columns) for path in sorted((TABLES / name).glob("*.parquet"))],
                     ignore_index=True)


def linked_rows() -> pd.DataFrame:
    races = _read("races", ["race_id", "race_date", "track_name", "off_time_utc", "is_abandoned", "is_arab_race"])
    runner_cols = ["race_id", "runner_id", "horse_name"] + [c for pair in ROLES.values() for c in pair]
    runners = _read("runners", runner_cols)
    with sqlite3.connect(Path(config.DATA_DIR) / "races.db") as con:
        db = pd.read_sql("SELECT race_id, horse_id, horse_name, race_date, track, off_time, jockey, trainer FROM results", con)
    links = match_runners(races, runners, db[["race_id", "horse_id", "horse_name", "race_date", "track", "off_time"]])
    links = links.merge(runners, left_on=["rtv_race_id", "rtv_runner_id"], right_on=["race_id", "runner_id"],
                        validate="one_to_one", suffixes=("", "_rtv"))
    db["race_id"], db["horse_id"] = db.race_id.astype(str), db.horse_id.astype(str)
    links["sl_race_id"], links["sl_horse_id"] = links.sl_race_id.astype(str), links.sl_horse_id.astype(str)
    return links.merge(db[["race_id", "horse_id", "race_date", "jockey", "trainer"]],
                       left_on=["sl_race_id", "sl_horse_id"], right_on=["race_id", "horse_id"],
                       validate="one_to_one", suffixes=("", "_db"))


TITLES = {"mr", "mrs", "miss", "ms", "dr", "capt", "major", "sir", "lady", "lord", "hon"}


def _holders(name: str) -> list[tuple[str, str]]:
    """(given-name initials, surname) of each licence holder in a possibly joint name."""
    holders = []
    for part in re.split(r"&| and ", name):
        words = [w for w in re.sub(r"[^a-z' -]", "", part.casefold()).replace("-", " ").split()
                 if w.strip(".") not in TITLES]
        if words:
            holders.append(("".join(w[0] for w in words[:-1]), words[-1]))
    return holders


def _compatible(a: str, b: str) -> bool:
    """Same surname, and one name's initials begin the other's ("B W" fits "Ben", "B J" does not fit "B T")."""
    return any(sa == sb and (ia.startswith(ib) or ib.startswith(ia))
               for ia, sa in _holders(a) for ib, sb in _holders(b))


def _concurrent(links: pd.DataFrame) -> bool:
    """True when two of a name's IDs were in use together for OVERLAP_DAYS."""
    spans = links.sort_values("first")[["first", "last"]].to_numpy()
    latest_end = spans[0][1]
    for start, end in spans[1:]:
        if latest_end - start > pd.Timedelta(days=OVERLAP_DAYS):
            return True
        latest_end = max(latest_end, end)
    return False


def aliases_for(rows: pd.DataFrame, role: str) -> tuple[dict[str, str], dict]:
    primary, fallback = ROLES[role]
    ids = pd.to_numeric(rows[primary], errors="coerce").fillna(pd.to_numeric(rows[fallback], errors="coerce"))
    frame = pd.DataFrame({"name": rows[role].astype("string").str.strip(), "rtv_id": ids,
                          "date": pd.to_datetime(rows["race_date_db"] if "race_date_db" in rows else rows["race_date"])}).dropna()
    frame = frame[frame.name.ne("") & frame.name.str.casefold().ne("non runner")]
    links = frame.groupby(["name", "rtv_id"]).date.agg(rides="size", first="min", last="max").reset_index()
    links["share"] = links.rides / links.groupby("name").rides.transform("sum")
    links = links[(links.rides >= MIN_RIDES) & (links.share >= MIN_EDGE_SHARE)]
    shared = {name for name, group in links.groupby("name") if len(group) > 1 and _concurrent(group)}
    links = links[~links.name.isin(shared)]

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for link in links.itertuples():
        parent[find(f"n:{link.name}")] = find(f"i:{link.rtv_id:g}")
    last_used = frame.groupby("name").date.max()
    people: dict[str, list[str]] = {}
    for name in links.name.unique():
        people.setdefault(find(f"n:{name}"), []).append(name)
    mapping, conflicting = {}, 0
    for names in people.values():
        if len(names) < 2:
            continue
        if set(names) not in CONFIRMED and not all(_compatible(a, b) for i, a in enumerate(names) for b in names[i + 1:]):
            conflicting += 1
            continue
        canonical = max(names, key=lambda n: (last_used[n], n))
        mapping.update({n: canonical for n in names if n != canonical})
    stats = {"linked_rides": int(len(frame)), "names": int(frame.name.nunique()),
             "names_mapped": len(mapping), "people_with_several_names": sum(len(v) > 1 for v in people.values()) - conflicting,
             "rides_remapped": int(frame.name.map(mapping).notna().sum()),
             "names_used_by_two_ids_at_once": len(shared), "people_skipped_for_name_conflict": conflicting}
    return mapping, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rows = linked_rows()
    payload = {"source": "Racing TV jockey/trainer IDs linked to Sporting Life rows (scripts/build_connection_aliases.py)",
               "min_rides": MIN_RIDES, "min_edge_share": MIN_EDGE_SHARE, "overlap_days": OVERLAP_DAYS}
    for role in ROLES:
        mapping, stats = aliases_for(rows, role)
        payload[role] = dict(sorted(mapping.items()))
        print(role, stats)
        for name in list(mapping)[:12]:
            print(f"    {name!r} -> {mapping[name]!r}")
    if args.write:
        OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
