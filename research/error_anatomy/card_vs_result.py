"""Compare cached pre-race racecards with the stored results for the same runners.

    python -m research.error_anatomy.card_vs_result

The model is trained and scored on results-page rows. A forecast made before
the race only has the racecard. Any field that differs between the two is
either train/serve skew or, if the results value was fixed after the off,
information from the future. Reads ``data/racecards_cache`` and ``data/races.db``.
"""
import glob
import sqlite3

import numpy as np
import pandas as pd

from research.error_anatomy.frame import OUT
from src.ingestion.storage import atomic_json

FIELDS = ["off_time", "track", "race_class", "race_type", "distance_furlongs", "going", "prize_money", "num_runners",
          "jockey", "trainer", "age", "sex", "headgear", "weight_lbs", "draw", "form", "days_since_last_run",
          "official_rating", "horse_runs", "surface", "handicap"]
NUMERIC = {"race_class", "distance_furlongs", "prize_money", "num_runners", "age", "weight_lbs", "draw",
           "days_since_last_run", "official_rating", "horse_runs", "handicap"}


def text(values):
    return values.astype("string").str.strip().str.lower().replace({"nan": pd.NA, "none": pd.NA, "": pd.NA})


def main():
    cards = pd.concat([pd.read_csv(path, dtype={"race_id": str, "horse_id": str}) for path in
                       sorted(glob.glob("data/racecards_cache/racecards_*.csv"))], ignore_index=True)
    cards = cards.drop_duplicates(["race_id", "horse_id"], keep="last")
    connection = sqlite3.connect("file:data/races.db?mode=ro", uri=True)
    dates = sorted(cards.race_date.unique())
    marks = ",".join("?" * len(dates))
    results = pd.read_sql_query(f"select * from results where race_date in ({marks})", connection, params=dates)
    results["race_id"], results["horse_id"] = results.race_id.astype(str), results.horse_id.astype(str)
    both = cards.merge(results, on=["race_id", "horse_id"], suffixes=("_card", "_result"), how="outer", indicator=True)
    report = {"card_dates": len(dates), "card_runners": len(cards), "result_runners": len(results),
              "on_card_only": int(both._merge.eq("left_only").sum()), "in_result_only": int(both._merge.eq("right_only").sum()),
              "matched": int(both._merge.eq("both").sum()), "fields": {}}
    print({k: v for k, v in report.items() if k != "fields"})
    matched = both.loc[both._merge.eq("both")]
    print(f"{'field':<22}{'both present':>13}{'differ':>9}{'share':>8}{'card only':>11}{'result only':>13}")
    for field in FIELDS:
        a, b = matched.get(f"{field}_card"), matched.get(f"{field}_result")
        if a is None or b is None:
            continue
        if field in NUMERIC:
            a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
            if field in ("official_rating", "draw"):
                a, b = a.where(a > 0), b.where(b > 0)
            present = a.notna() & b.notna()
            differ = present & ~np.isclose(a.fillna(0), b.fillna(0), atol=.011)
        else:
            a, b = text(a), text(b)
            present = a.notna() & b.notna()
            differ = present & (a.fillna("") != b.fillna(""))
        row = {"both_present": int(present.sum()), "differ": int(differ.sum()),
               "share": float(differ.sum() / max(present.sum(), 1)),
               "card_only": int((a.notna() & b.isna()).sum()), "result_only": int((a.isna() & b.notna()).sum())}
        examples = matched.loc[differ, [f"{field}_card", f"{field}_result"]].astype(str)
        row["examples"] = examples.value_counts().head(8).reset_index().values.tolist()
        report["fields"][field] = row
        print(f"{field:<22}{row['both_present']:>13}{row['differ']:>9}{row['share']:>8.3f}{row['card_only']:>11}{row['result_only']:>13}")
    atomic_json(OUT / "card_vs_result.json", report)
    for field in ("num_runners", "going", "weight_lbs", "jockey", "distance_furlongs", "headgear", "days_since_last_run",
                  "form", "horse_runs", "official_rating", "race_class"):
        if field in report["fields"] and report["fields"][field]["differ"]:
            print(f"\n{field}: most common (card, result, count)")
            for example in report["fields"][field]["examples"]:
                print("   ", example)


if __name__ == "__main__":
    main()
