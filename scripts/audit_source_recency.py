"""Check whether raw pre-race recency agrees with observed earlier starts.

Research receipt only. Does not replace active features or amend source data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(source, profiles):
    frame = pd.read_csv(source, usecols=["horse_id", "horse_name", "race_id", "race_date",
                                        "days_since_last_run", "race_type"])
    frame["race_date"] = pd.to_datetime(frame.race_date)
    frame["horse_id"] = pd.to_numeric(frame.horse_id, errors="raise").astype("int64")
    frame["source_days"] = pd.to_numeric(frame.days_since_last_run, errors="coerce")
    frame = frame.sort_values(["horse_id", "race_date", "race_id"], kind="stable")
    daily = frame[["horse_id", "race_date"]].drop_duplicates().copy()
    daily["previous_local_date"] = daily.groupby("horse_id", sort=False).race_date.shift()
    frame = frame.merge(daily, on=["horse_id", "race_date"], validate="many_to_one")
    frame["local_days"] = (frame.race_date - frame.previous_local_date).dt.days
    known = frame.local_days.gt(0) & frame.source_days.gt(0)

    def counts(part):
        comparable = part.local_days.gt(0) & part.source_days.gt(0)
        both = part.loc[comparable]
        return {"rows": len(part), "positive_source_gap": int(part.source_days.gt(0).sum()),
                "comparable_rows": len(both), "equal_gap": int(both.source_days.eq(both.local_days).sum()),
                "source_gap_shorter": int(both.source_days.lt(both.local_days).sum()),
                "source_gap_longer": int(both.source_days.gt(both.local_days).sum()),
                "source_missing_or_nonpositive": int((~part.source_days.gt(0)).sum())}

    public_checks = []
    snapshot = hashlib.sha256()
    for path in sorted(Path(profiles).glob("*.json")):
        snapshot.update(path.name.encode()); snapshot.update(path.read_bytes())
        cache = json.loads(path.read_text())
        day = pd.Timestamp(cache["first_local_date"])
        eligible = frame.loc[frame.horse_id.eq(int(cache["horse_id"])) & frame.race_date.eq(day)]
        # Evaluate availability only from the fixed cohort's selection date.
        prior_dates = [pd.Timestamp(row["date"]) for row in cache["results"]
                       if row.get("date") and pd.Timestamp(row["date"]) < day
                       and int(row.get("runner_count") or 0) >= 2
                       and int(row.get("position") or 0) > 0]
        if eligible.empty or not prior_dates:
            continue
        latest = max(prior_dates)
        for row in eligible.itertuples():
            public_checks.append({"horse_id": cache["horse_id"], "horse_name": cache["horse_name"],
                "race_id": str(row.race_id), "date": str(day.date()),
                "source_days": float(row.source_days) if pd.notna(row.source_days) else None,
                "known_public_gap_days": (day-latest).days})
    public = pd.DataFrame(public_checks)
    report = {"scope": "Retrospective data validation only; public profiles need not contain every start",
              "source_sha256": digest(source), "profile_cache_sha256": snapshot.hexdigest(),
              "all": counts(frame), "by_race_type": {str(k): counts(g) for k, g in frame.groupby("race_type")},
              "first_local_rows_with_positive_source_gap": int((frame.local_days.isna() & frame.source_days.gt(0)).sum()),
              "public_profile_comparison": {"rows": len(public),
                  "equal_gap": int(public.source_days.eq(public.known_public_gap_days).sum()),
                  "source_gap_shorter": int(public.source_days.lt(public.known_public_gap_days).sum()),
                  "source_gap_longer": int(public.source_days.gt(public.known_public_gap_days).sum())}}
    return report, public, frame.loc[known & frame.source_days.gt(frame.local_days)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/race_results.csv")
    parser.add_argument("--profiles", default="data/research/public_prior_results/horses")
    parser.add_argument("--output", default="data/audit/source_recency_20261027.json")
    args = parser.parse_args()
    report, public, conflicts = audit(args.source, args.profiles)
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2))
    public.to_csv(output.with_suffix(".public.csv"), index=False)
    conflicts.head(100).to_csv(output.with_suffix(".conflicts.csv"), index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
