"""Reconcile full fields using fresh SL pages and independently stored RTV orders."""
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src import rtv_scraper as rtv
from src.database import _validate_race_results
from src.source_pipeline import load_history_table, _publish
from src.utils import normalise_horse_key, normalise_off_time_key
from src.deployment_validation import file_sha256

CASE = ROOT/"data/audit/rtv_repair_features_044"


def main():
    spec = importlib.util.spec_from_file_location("rtv_staged_scraper", CASE/"project/src/data_scraper.py")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    roster = json.loads((CASE/"review_roster.json").read_text())["roster"]
    races = load_history_table("races", "2023-06-10", "2026-10-29")
    runners = load_history_table("runners", "2023-06-10", "2026-10-29")
    races["_track"] = races.track_name.map(rtv._track_join_key)
    races["_off"] = races.off_time_utc.map(normalise_off_time_key)
    decisions, before_parts, after_parts, new_parts = [], [], [], []
    with sqlite3.connect((ROOT/"data/races.db").as_uri()+"?mode=ro", uri=True) as con:
        for item in roster:
            decision = dict(item)
            if not item["race_id"]:
                decision.update(state="source_not_listed", reason="No exact SL event identity")
                decisions.append(decision)
                continue
            identity = item["race_id"]
            path = CASE/"sources"/(identity+".json")
            try:
                receipt = json.loads(path.read_text())
                page = receipt["page"]
                scraper = parser.SportingLifeScraper()
                scraper._fetch_next_data = lambda url, page=page: page
                final = pd.DataFrame(scraper.scrape_race_result(item["url"], item["race_date"], raise_if_not_run=True))
                _validate_race_results(final, item["race_date"], identity)
                candidates = races.loc[races.race_date.eq(item["race_date"])
                    & races._track.eq(rtv._track_join_key(item["track"])) & races._off.eq(normalise_off_time_key(item["off_time"]))]
                if len(candidates) != 1:
                    raise ValueError("Ambiguous RTV race")
                race = candidates.iloc[0]
                if race.is_abandoned or race.is_arab_race:
                    raise ValueError("Excluded RTV race")
                other = runners.loc[runners.race_id.eq(race.race_id) & runners.in_results.fillna(False).astype(bool)
                    & ~runners.is_nonrunner].copy()
                if identity == "871834":
                    # Independently reviewed rename: same three pedigree parents,
                    # foaling date and five source-profile participations. No
                    # finishing position or odds are used as identity evidence.
                    sl_profile = json.loads((CASE/"sources/horse_1188514.json").read_text())["props"]["pageProps"]["profile"]
                    tv_profile = json.loads((CASE/"sources/rtv_horse_494668_verified.json").read_text())["page"]["horse"]
                    parents = ((sl_profile["sire"]["name"], tv_profile["sire_name"]),
                        (sl_profile["dam"]["name"], tv_profile["dam_name"]), (sl_profile["damsire"]["name"], tv_profile["dam_sire_name"]))
                    sl_events = {(x["date"][:10], rtv._track_join_key(x["course_name"])) for x in sl_profile["previous_results"]}
                    tv_events = {(x["meeting_date"][:10], rtv._track_join_key(x["track_name"])) for x in tv_profile["performances"]}
                    if (not all(a==b for a,b in parents) or sl_profile["foaled"] != tv_profile["foaling_date"]
                            or len(sl_events & tv_events) < 4):
                        raise ValueError("Renamed horse profile evidence changed")
                    alias = other.horse_id.eq(494668)
                    if alias.sum() != 1 or not final.horse_id.astype(str).eq("1188514").any():
                        raise ValueError("Renamed horse stable ID absent")
                    other.loc[alias, "horse_name"] = "My Mars"
                    decision["profile_identity_events"] = len(sl_events & tv_events)
                for frame in (final, other):
                    frame["_horse"] = frame.horse_name.map(normalise_horse_key)
                if (other._horse.duplicated().any() or final._horse.duplicated().any()
                        or set(other._horse) != set(final._horse)):
                    decision["only_sl"] = sorted(set(final._horse)-set(other._horse))
                    decision["only_rtv"] = sorted(set(other._horse)-set(final._horse))
                    raise ValueError("SL/RTV starter population differs")
                joined = final.merge(other[["_horse", "runner_id", "finish_position", "final_position", "sp_decimal"]],
                    on="_horse", validate="one_to_one", suffixes=("", "_rtv"))
                if not joined.finish_position.eq(joined.final_position.fillna(0)).all():
                    bad = joined.loc[joined.finish_position.ne(joined.final_position.fillna(0)), ["horse_name", "finish_position", "final_position"]]
                    decision["disagreements"] = bad.to_dict("records")
                    raise ValueError("Fresh SL final order differs from RTV")
                before = pd.read_sql_query("SELECT * FROM results WHERE race_id=?", con, params=(identity,))
                if before.empty:
                    new_parts.append(final.drop(columns="_horse"))
                    decision.update(state="verified_missing_field", runners=len(final))
                else:
                    final_ids = pd.to_numeric(final.horse_id)
                    old_ids = pd.to_numeric(before.horse_id)
                    removed = set(old_ids)-set(final_ids)
                    source_nonstarters = {int(r["horse"]["horse_reference"]["id"]) for r in page["props"]["pageProps"]["race"]["rides"]
                        if str(r.get("ride_status", "")).replace("-", "").upper() in {"NONRUNNER", "WITHDRAWN"}}
                    if (set(final_ids)-set(old_ids) or not removed.issubset(source_nonstarters)
                            or not before.loc[old_ids.isin(removed), "finish_position"].eq(0).all()):
                        raise ValueError("Existing/source starter population differs")
                    decision["removed_confirmed_nonstarters"] = len(removed)
                    # Source horse IDs survive renaming. Preserve the database's
                    # race-day spelling while joining the current source name.
                    before["_horse"] = old_ids.map(pd.Series(final._horse.to_numpy(), index=final_ids))
                    merged = before.loc[before._horse.notna()].merge(final[["_horse", "horse_id", "finish_position", "finish_pos_label", "won", "lengths_behind"]],
                        on="_horse", suffixes=("", "_final"), validate="one_to_one")
                    if not pd.to_numeric(merged.horse_id).eq(pd.to_numeric(merged.horse_id_final)).all():
                        raise ValueError("Horse source IDs differ")
                    changed = merged.finish_position.ne(merged.finish_position_final)
                    if changed.any():
                        retained = before.loc[before._horse.notna()]
                        after = retained.drop(columns="_horse").copy()
                        final = final.set_index("_horse")
                        for col in ("finish_position", "finish_pos_label", "won", "lengths_behind"):
                            after[col] = retained._horse.map(final[col]).to_numpy()
                        before_parts.append(before.drop(columns="_horse"))
                        after_parts.append(after)
                        decision.update(state="verified_repair", runners=len(before), position_changes=int(changed.sum()))
                    else:
                        decision.update(state="already_matches_fresh_sl", runners=len(before))
                decision["sl_receipt_sha256"] = file_sha256(path)
                decision["rtv_race_id"] = int(race.race_id)
            except Exception as error:
                decision.update(state="review_required", error=str(error))
            decisions.append(decision)
            print(json.dumps({k: decision.get(k) for k in ("race_id", "state", "error", "runners", "position_changes")}), flush=True)
    for name, parts in (("repair_before", before_parts), ("repair_after", after_parts), ("new_fields", new_parts)):
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(CASE/(name+".parquet"), index=False)
    _publish(CASE/"verification.json", {"checked_at": datetime.now(timezone.utc).isoformat(),
        "counts": dict(Counter(x["state"] for x in decisions)), "decisions": decisions})


if __name__ == "__main__":
    main()
