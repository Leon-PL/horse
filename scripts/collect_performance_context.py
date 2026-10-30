"""Audit cached factual ride context and optionally collect at most24 prior fields.

Core results, feature values and production weights are never rewritten. The
optional normal-ingestion context table retains verified source observations.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.data_scraper import SportingLifeScraper
from src.database import DB_PATH, init_db, _get_conn, _insert_result_context
from src.deployment_validation import file_sha256
from src.performance_context import context_history, FEATURES, description_mentions
from src.utils import normalise_horse_key

OUTPUT = ROOT/"data/audit/predictive_review_038/context"
CACHED = ROOT/"data/audit/handicap_weight_review_030/sources"
HISTORY = ROOT/"data/audit/market_transfer_034/source_inputs.parquet"


def read_source(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    race = payload["props"]["pageProps"]["race"]
    summary = race["race_summary"]
    identity = str(summary["race_summary_reference"]["id"])
    date = str(summary["date"])[:10]
    parser = SportingLifeScraper()
    parser._fetch_next_data = lambda url: payload
    rows = parser.scrape_race_result("cached", date)
    if not rows or not all(str(row["race_id"]) == identity and row["race_date"] == date for row in rows):
        raise ValueError("Cached source identity differs from its parsed result")
    return pd.DataFrame(rows)


def run(collect_prior=False, persist=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    additional = OUTPUT/"sources"
    additional.mkdir(exist_ok=True)
    source_paths = sorted(CACHED.glob("*.json"))+sorted(additional.glob("*.json"))
    observations = pd.concat([read_source(p) for p in source_paths], ignore_index=True)
    # Bounded population chosen from the earlier random source sample, without
    # inspecting whether a particular runner subsequently won or lost.
    old = json.loads((CACHED.parent/"manifest.json").read_text())
    names = sorted(observations.loc[observations.race_id.astype(str).isin(old["random_fields"]), "horse_name"].unique())
    sampled = pd.Series(names).sample(n=min(12,len(names)), random_state=42).tolist()
    wanted = sorted(set(observations.horse_name)|set(sampled))
    columns = ["race_id", "horse_id", "horse_name", "race_date", "off_time", "track", "race_name", "finish_position", "finish_pos_label"]
    history = pd.read_parquet(HISTORY, columns=columns, filters=[("horse_name", "in", wanted)])
    history["race_id"] = history.race_id.astype(str)
    history["race_date"] = pd.to_datetime(history.race_date)
    plan = {"scope": "Coverage/causality pilot, not prediction evidence", "sample_seed":42,
        "sampled_horses": sampled, "maximum_additional_fields":24,
        "sources_sha256": {str(p.relative_to(ROOT)):file_sha256(p) for p in sorted(CACHED.glob("*.json"))},
        "implementation_sha256": {f:file_sha256(ROOT/f) for f in ("src/data_scraper.py", "src/performance_context.py", "scripts/collect_performance_context.py")}}
    plan_path = OUTPUT/"plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Preserve the fixed source pilot; definitions changed")
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    requests = []
    for name in sampled:
        first = pd.to_datetime(observations.loc[observations.horse_name.eq(name), "race_date"]).min()
        previous = history.loc[history.horse_name.eq(name)&history.race_date.lt(first)].sort_values(["race_date", "off_time", "race_id"]).tail(3)
        requests.extend(previous.to_dict("records"))
    requests = list({str(row["race_id"]):row for row in requests}.values())[:24]
    receipts = []
    if collect_prior:
        scraper = SportingLifeScraper()
        for row in requests:
            rid = str(row["race_id"])
            path = additional/(rid+".json")
            if (CACHED/(rid+".json")).exists() or path.exists():
                continue
            slug = lambda value: re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")
            date = row["race_date"].date().isoformat()
            url = f"https://www.sportinglife.com/racing/results/{date}/{slug(row['track'])}/{rid}/{slug(row['race_name'])}"
            receipt = {"race_id":rid, "url":url, "observed_at":datetime.now(timezone.utc).isoformat()}
            try:
                payload = scraper._fetch_next_data(url)
                summary = payload["props"]["pageProps"]["race"]["race_summary"]
                if str(summary["race_summary_reference"]["id"]) != rid or str(summary["date"])[:10] != date:
                    raise ValueError("Historical context source identity/date differs")
                path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
                receipts.append({**receipt, "sha256":file_sha256(path), "passed":True})
            except Exception as error:
                receipts.append({**receipt, "passed":False, "error":str(error)[:500]})
                # A blocked/failed source is a pilot limitation, never a reason
                # to launch an unbounded scrape or fabricate missing evidence.
                break
        source_paths = sorted(CACHED.glob("*.json"))+sorted(additional.glob("*.json"))
        observations = pd.concat([read_source(p) for p in source_paths], ignore_index=True)
    observations = observations.drop_duplicates(["race_id", "horse_name"])
    # The database is the identity contract; comments are never joined by name
    # alone, and the old canonical position/weight remain unchanged.
    with sqlite3.connect(f"file:{Path(DB_PATH).as_posix()}?mode=ro", uri=True) as connection:
        stored = pd.read_sql_query("SELECT race_id,horse_name,race_date,horse_id,weight_lbs FROM results", connection)
    key = ["race_id", "horse_name", "race_date"]
    stored.race_id = stored.race_id.astype(str)
    joined = observations.merge(stored, on=key, suffixes=("", "_stored"), validate="one_to_one")
    if len(joined) != len(observations) or not np.allclose(joined.weight_lbs, joined.weight_lbs_stored, equal_nan=True):
        raise ValueError("Context sources do not reproduce existing runner identities/declared weights")
    def canonical_id(series):
        return pd.to_numeric(series, errors="raise").astype("Int64")
    if not canonical_id(joined.horse_id).equals(canonical_id(joined.horse_id_stored)):
        raise ValueError("Context horse identity differs from the stored source")
    if persist:
        init_db()
        with _get_conn() as connection:
            _insert_result_context(connection, observations)
    observations.to_parquet(OUTPUT/"source_observations.parquet", index=False)
    # The temporal representation is evaluated only for the fixed source-pilot
    # horses; these coverage rates are not claimed for the full database.
    evidence = observations[[*key, "ride_description", "nonfinish_reason"]].copy()
    evidence.race_date = pd.to_datetime(evidence.race_date)
    contextual = history.merge(evidence, on=key, how="left", validate="one_to_one")
    features = context_history(contextual)
    prefix = contextual.race_date <= "2025-12-31"
    if prefix.any() and (~prefix).any():
        _, seed = context_history(contextual.loc[prefix], return_state=True)
        pd.testing.assert_frame_equal(context_history(contextual.loc[~prefix], state=seed), features.loc[~prefix])
    pd.concat([contextual[["race_id", "horse_name", "race_date"]], features], axis=1).to_parquet(OUTPUT/"pilot_history.parquet", index=False)
    report = {"source_fields":int(observations.race_id.nunique()), "source_runners":len(observations),
        "explicit_claims":int(observations.reported_jockey_claim_lbs.notna().sum()),
        "positive_claims":int(observations.reported_jockey_claim_lbs.gt(0).sum()),
        "unknown_claims":int(observations.reported_jockey_claim_lbs.isna().sum()),
        "penalties":int(observations.weight_penalty_lbs.notna().sum()),
        "descriptions":int(observations.ride_description.fillna("").str.strip().ne("").sum()),
        "mention_counts":{k:int(sum(description_mentions(v)[k] == 1 for v in observations.ride_description))
                          for k in ("impeded", "slow_start", "jump_error", "eased", "stayed_on")},
        "pilot_horses":len(wanted), "pilot_history_rows":len(contextual),
        "pilot_rows_with_prior_description":int(features.context_description_support_3.gt(0).sum()),
        "full_prefix_seed_equal":True, "metadata_persisted":persist, "core_results_changed":False,
        "new_network_requests":len(receipts), "receipts":receipts,
        "prediction_gain_measured":False, "decision":"Accumulate dated context through normal collection; this pilot is too sparse/outcome-selected for a native prediction claim"}
    (OUTPUT/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-prior", action="store_true")
    parser.add_argument("--persist-context", action="store_true")
    args = parser.parse_args()
    run(args.collect_prior, args.persist_context)
