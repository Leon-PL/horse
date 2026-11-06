"""Exercise the repaired live path on prospective copies of real complete cards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.data_processor import process_data
from src.database import read_results_snapshot
from src.feature_store import FeatureStore
from src.featured_dataset import fe_signature
from src.result_status import pending_card


def smoke(store_root, features, output):
    store = FeatureStore(str(store_root))
    before = store.meta()
    if before["fe_signature"] != fe_signature():
        raise ValueError("Live smoke requires current feature generation")
    last = pd.Timestamp(before["last_date"])
    history = read_results_snapshot()
    settled = history.loc[pd.to_datetime(history.race_date).eq(last)].reset_index(drop=True)
    cards = []
    for _, typed in settled.groupby("race_type", sort=False):
        for _, field in typed.groupby("race_id", sort=False):
            if field.num_runners.eq(len(field)).all() and field.horse_id.notna().all():
                cards.append(field.copy())
                break
    card = pd.concat(cards, ignore_index=True)
    # New IDs prevent a simulated pending card colliding with settled history.
    card["race_id"] = -pd.to_numeric(card.race_id)
    card["race_date"] = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    card = pending_card(card)
    processed = process_data(df=card)
    featured = store.featurize_live(processed)
    missing = sorted(set(features) - set(featured))
    if missing or len(featured) != len(card) or store.meta() != before:
        raise ValueError(f"Live feature smoke failed: missing={missing}")
    if not featured.won.eq(0).all() or not featured.finish_position.eq(0).all():
        raise ValueError("Synthetic pending cards retained their settled outcomes")
    if not featured.horse_prev_races.ge(1).all():
        raise ValueError("A horse seen in yesterday's complete field lost its known history")
    values = featured[features].to_numpy(float)
    if np.isinf(values).any():
        raise ValueError("Live inputs contain infinite values")
    output = Path(output)
    featured[list(dict.fromkeys(["race_id", "horse_name", "race_date", *features]))].to_parquet(
        output.with_suffix(".parquet"), index=False)
    report = {"passed": True, "scope": "Serving smoke only, no forecast performance estimate",
              "fe_signature": fe_signature(), "history_last_date": before["last_date"],
              "simulated_pending_date": str((last + pd.Timedelta(days=1)).date()),
              "races": card.race_id.nunique(), "runners": len(card),
              "race_types": sorted(card.race_type.unique().tolist()),
              "selected_inputs": features, "missing_inputs": missing, "store_metadata_unchanged": True}
    report["all_known_horses_matched_history"] = True
    report["minimum_previous_starts"] = int(featured.horse_prev_races.min())
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "selected_inputs"}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="data/audit/model_improvement/015_nonfinisher_history/candidate_store")
    parser.add_argument("--registration", default="data/audit/model_improvement/015_nonfinisher_history/registration.json")
    parser.add_argument("--output", default="data/audit/live_features_repair_015.json")
    args = parser.parse_args()
    features = json.loads(Path(args.registration).read_text())["candidate_features"]
    smoke(args.store, features, args.output)


if __name__ == "__main__":
    main()
