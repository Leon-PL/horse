"""Which feature stages produce the columns the live models read, and what each costs (research 109).

    python -m research.feature_pruning_109.stage_map [--date 2026-11-16] [--ablate]

Consumers: the win model of the Matchbook paper record (run 20261114_122027_967139, 87 inputs) and the forecast
correction (tracking columns, entity and card columns). Every stage of ``src.feature_engineer.engineer_features``
is wrapped to record the columns it adds and its time on a live card (``features_for_live_cards``).

``--ablate`` then replaces every stage that adds no needed column with a pass-through and checks that the needed
columns come out identical: if they do, those stages feed nothing the models read.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import feature_engineer as fe
from src.correction import data as cd
from src.correction import inputs as ci

OUT = Path("data/research/feature_pruning_109")
WIN_RUN = "20261114_122027_967139"
STAGE_NAMES = None


def needed_columns():
    from src.run_store import load_run_predictor
    (name, member), = load_run_predictor(WIN_RUN).models.items()
    return sorted(set(member.spec.features) | set(cd.tracking_columns()) | set(ci.ENTITY_COLUMNS) | set(cd.CARD_COLUMNS))


def stage_functions():
    """Every stage function name that engineer_features may run, in module order."""
    source = Path(fe.__file__).read_text(encoding="utf-8")
    body = source[source.index("def engineer_features("):]
    body = body[:body.index("_profile = os.environ")]
    names = []
    for token in body.replace(",", " ").replace("(", " ").split():
        token = token.strip()
        if (token.startswith(("add_", "compute_", "_use_day_order")) and token.isidentifier()
                and callable(getattr(fe, token, None)) and token not in names):
            names.append(token)
    return names


def instrument(skip=()):
    record = []
    originals = {}
    for name in stage_functions():
        original = getattr(fe, name)
        originals[name] = original

        def wrapped(df, *a, _name=name, _original=original, **k):
            before = set(df.columns)
            started = time.perf_counter()
            out = df if _name in skip else _original(df, *a, **k)
            record.append({"stage": _name, "seconds": time.perf_counter() - started,
                           "added": sorted(set(out.columns) - before), "skipped": _name in skip})
            return out
        setattr(fe, name, wrapped)
    return record, originals


def restore(originals):
    for name, original in originals.items():
        setattr(fe, name, original)


def featurise(day, skip=()):
    from src.ingestion.pipeline import get_scraped_racecards
    from src.live_prediction import features_for_live_cards
    cards = get_scraped_racecards(date_str=day)
    record, originals = instrument(skip)
    try:
        frame = features_for_live_cards(cards, day, use_cache=False)
    finally:
        restore(originals)
    return frame, record


def main(day, ablate):
    OUT.mkdir(parents=True, exist_ok=True)
    needed = needed_columns()
    frame, record = featurise(day)
    table = []
    for r in record:
        made = set(r["added"])
        table.append({"stage": r["stage"], "seconds": round(r["seconds"], 2), "added": len(made),
                      "needed": sorted(made & set(needed))})
    produced = {c for r in record for c in r["added"]}
    report = {"date": day, "needed": len(needed), "store_columns_on_card": int(frame.shape[1]),
              "needed_not_made_by_a_stage": sorted(set(needed) - produced), "stages": table}
    for row in table:
        print(f"{row['seconds']:7.2f}s  {row['added']:4d} cols  {len(row['needed']):3d} needed  {row['stage']}")
    if ablate:
        idle = [r["stage"] for r in table if not r["needed"] and r["stage"] != "_use_day_order_after_ratings"]
        ablated, record2 = featurise(day, skip=set(idle))
        keys = ["race_id", "horse_id"]
        a = frame.set_index(keys)[[c for c in needed if c in frame]].sort_index()
        b = ablated.set_index(keys)[[c for c in needed if c in ablated]].sort_index()
        missing = sorted(set(a.columns) - set(b.columns))
        changed = [c for c in a.columns if c in b and not a[c].equals(b[c])
                   and not np.allclose(pd.to_numeric(a[c], errors="coerce"), pd.to_numeric(b[c], errors="coerce"),
                                       equal_nan=True)]
        report["ablation"] = {"skipped": idle, "seconds_without": round(sum(r["seconds"] for r in record2), 1),
                              "seconds_with": round(sum(r["seconds"] for r in record), 1),
                              "needed_missing": missing, "needed_changed": changed}
        print(json.dumps(report["ablation"], indent=1))
    (OUT / "stage_map.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-11-16")
    parser.add_argument("--ablate", action="store_true")
    args = parser.parse_args()
    main(args.date, args.ablate)
