"""Per-column fingerprints of the feature store and a live card, to prove a pipeline change left values unchanged.

    python -m research.feature_pruning_109.fingerprint save NAME [--card-date 2026-11-16]
    python -m research.feature_pruning_109.fingerprint compare BEFORE AFTER [--columns keep|needed|all]

Each store column is read in (race_id, horse_id) order and hashed (``pandas.util.hash_pandas_object``), so two
stores compare column by column without holding either in memory. The live card's features are saved whole.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = Path("data/research/feature_pruning_109/fingerprints")
KEYS = ["race_id", "horse_id"]


def store_digests():
    from src.feature_store import FeatureStore
    path = FeatureStore().featured_path
    names = pq.read_schema(path).names
    keys = pd.read_parquet(path, columns=KEYS)
    order = np.lexsort((keys.horse_id.astype("int64").to_numpy(), keys.race_id.astype("int64").to_numpy()))
    digests = {"__rows__": int(len(keys))}
    batch = 40
    for i in range(0, len(names), batch):
        part = pd.read_parquet(path, columns=names[i:i + batch])
        for column in part:
            values = part[column].iloc[order].reset_index(drop=True)
            if values.dtype == object:
                values = values.astype(str)
            digests[column] = str(int(pd.util.hash_pandas_object(values, index=False).to_numpy().astype("uint64")
                                      .sum(dtype="uint64")))
    return digests


def save(name, card_date):
    from src.ingestion.sporting_life import enrich_racecard_pedigree, load_cached_racecards
    from src.live_prediction import features_for_live_cards
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "store.json").write_text(json.dumps(store_digests(), indent=0))
    # The cached card file, never a fresh scrape: both sides must featurise the same declared runners.
    cards = enrich_racecard_pedigree(load_cached_racecards(card_date))
    card = features_for_live_cards(cards, card_date, use_cache=False)
    card.to_parquet(folder / "card.parquet", index=False)
    print(f"saved {folder}")


def compare(before, after, scope):
    from research.feature_pruning_109 import stage_map
    a = json.loads((OUT / before / "store.json").read_text())
    b = json.loads((OUT / after / "store.json").read_text())
    if scope == "needed":
        columns = stage_map.needed_columns()
    elif scope == "keep":
        from src.store_columns import KEEP
        columns = sorted(set(KEEP) & set(a))
    else:
        columns = sorted(set(a) - {"__rows__"})
    rows = (a["__rows__"], b["__rows__"])
    missing = [c for c in columns if c in a and c not in b]
    changed = [c for c in columns if c in a and c in b and a[c] != b[c]]
    ca = pd.read_parquet(OUT / before / "card.parquet").sort_values(KEYS).reset_index(drop=True)
    cb = pd.read_parquet(OUT / after / "card.parquet").sort_values(KEYS).reset_index(drop=True)
    card_changed = []
    for c in columns:
        if c in ca and c in cb:
            x, y = ca[c], cb[c]
            same = x.equals(y) or (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and
                                   np.allclose(x.astype(float), y.astype(float), equal_nan=True, rtol=0, atol=0))
            if not same:
                card_changed.append(c)
    card_missing = [c for c in columns if c in ca and c not in cb]
    result = {"scope": scope, "columns": len(columns), "rows": rows, "store_missing": missing,
              "store_changed": changed, "card_missing": card_missing, "card_changed": card_changed,
              "store_columns": (len(a) - 1, len(b) - 1), "card_columns": (ca.shape[1], cb.shape[1])}
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("save")
    s.add_argument("name")
    s.add_argument("--card-date", default="2026-11-16")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--columns", default="keep", choices=["keep", "needed", "all"])
    args = parser.parse_args()
    if args.command == "save":
        save(args.name, args.card_date)
    else:
        compare(args.before, args.after, args.columns)
