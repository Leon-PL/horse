"""Stage times of a from-scratch feature build over one year of history (research 109).

    python -m research.feature_pruning_109.build_profile [--start 2025-07-01 --end 2026-06-30]

Runs ``engineer_features`` (ratings included) on the processed results of the window, with every stage timed by
the same wrapper as ``stage_map``. A full store build is about six such years plus the rating sweep.
"""
import argparse
import json

import pandas as pd

from research.feature_pruning_109 import stage_map as sm
from src import feature_engineer as fe


def main(start, end):
    from src.data_processor import process_data
    processed = process_data()
    dates = pd.to_datetime(processed.race_date)
    window = processed.loc[(dates >= start) & (dates <= end)].reset_index(drop=True)
    record, originals = sm.instrument()
    try:
        fe.engineer_features(window, save=False)
    finally:
        sm.restore(originals)
    needed = set(sm.needed_columns())
    rows = [{"stage": r["stage"], "seconds": round(r["seconds"], 1), "added": len(r["added"]),
             "needed": len(set(r["added"]) & needed)} for r in record]
    total = sum(r["seconds"] for r in rows)
    for r in sorted(rows, key=lambda r: -r["seconds"]):
        print(f"{r['seconds']:7.1f}s {100 * r['seconds'] / total:5.1f}%  {r['added']:4d} cols {r['needed']:3d} needed  {r['stage']}")
    sm.OUT.mkdir(parents=True, exist_ok=True)
    (sm.OUT / "build_profile.json").write_text(json.dumps({"start": start, "end": end, "rows": len(window),
                                                           "stages": rows}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-07-01")
    parser.add_argument("--end", default="2026-06-30")
    args = parser.parse_args()
    main(args.start, args.end)
