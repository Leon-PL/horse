"""Are the scraped per-race fields as-of-race, or as-of-scrape?

The `sex` column was the horse's sex on the day its results page was scraped,
which encoded its future career and inflated every model that used it
(ledger 15). No existing guard could see it: test_feature_time_boundaries
proves features do not change when *future rows* are appended, but a raw value
that was future-contaminated when scraped never changes either.

The tell is simple. A field that must change over a horse's career (age,
official rating, trainer, form, and biologically sex) will be nearly constant
per horse if it is recorded at scrape time. This audits every such field and
fails when one that should vary does not.

    python scripts/audit_field_timing.py
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import pandas as pd

#: Fields that must vary across a career, with the minimum share of horses
#: (10+ runs) that should show a change if the field is recorded per race.
#: Thresholds sit far below the measured as-of-race rates (2026-10-13:
#: trainer 0.47, OR 0.999, age 0.987, form 1.0) and far above a scrape-time
#: field (sex: 0.02, though every filly becomes a mare at five).
MUST_VARY = {
    "age": 0.80,
    "official_rating": 0.80,
    "form": 0.80,
    "days_since_last_run": 0.80,
    "weight_lbs": 0.80,
    "trainer": 0.20,
    "headgear": 0.30,
}
#: Fields recorded per scrape by design, kept out of features as raw values.
KNOWN_SCRAPE_TIME = {"sex": "collapsed to sex at birth in data_processor (ledger 15)"}


def audit(db_path: str = os.path.join("data", "races.db"), min_runs: int = 10) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cols = list(MUST_VARY) + list(KNOWN_SCRAPE_TIME)
    have = {r[1] for r in con.execute("pragma table_info(results)")}
    use = [c for c in cols if c in have]
    df = pd.read_sql(f"select horse_id, {', '.join(use)} from results where horse_id is not null", con)
    runs = df.groupby("horse_id").size()
    sub = df[df["horse_id"].isin(runs[runs >= min_runs].index)]
    changes = (sub.groupby("horse_id")[use].nunique() > 1).mean()
    rows = []
    for col in use:
        need = MUST_VARY.get(col)
        rows.append({"field": col, "share_changing": round(float(changes[col]), 3),
                     "required": need, "ok": (need is None) or bool(changes[col] >= need),
                     "note": KNOWN_SCRAPE_TIME.get(col, "")})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=os.path.join("data", "races.db"))
    args = ap.parse_args()
    report = audit(args.db)
    print(report.to_string(index=False))
    bad = report[~report["ok"]]
    if not bad.empty:
        print(f"\nFAIL: {', '.join(bad['field'])} barely change per horse - recorded at scrape time?")
        return 1
    print("\nOK: every career-varying field varies; known scrape-time fields are handled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
