"""Are the labels right? Cross-check the results DB against Betfair.

Every accuracy number, every ROI and every refutation in this project rests on
``won`` and ``finish_position`` being correct. Nothing has ever checked them.
``plan.txt`` Stage 1 asked for exactly this audit and it was never carried out.

It matters more than it sounds. A label wrong even 0.5% of the time puts a
ceiling on achievable accuracy, and — worse — it is not symmetric: a missing
winner makes a race unwinnable for *every* model, which flatters the market
baseline relative to the model only if the errors correlate with price. Either
way an unmeasured error rate means every reported gap carries an unknown
offset.

Betfair is a genuinely independent source: ``betfair_bsp.parquet`` carries its
own ``won`` and ``n_runners`` settled by the exchange, and
``betfair_place_bsp.parquet`` carries ``placed_bf`` and ``bf_places_paid``. So
the scrape can be checked against a party that had money on the outcome.

The joins reuse ``src.betfair.attach_bsp``, which owns the merge keys
(CLAUDE.md hard rule 3) — this script must never roll its own.

One trap worth knowing: ``attach_bsp`` assigns ``out[c] = merged[c]``, so
requesting ``cols=("won",)`` **overwrites the caller's own ``won`` column**
with Betfair's. An audit written that way would compare the exchange against
itself and report perfect agreement. Our label is renamed before the join.

    .venv/Scripts/python.exe scripts/audit_labels.py
    .venv/Scripts/python.exe scripts/audit_labels.py --json --since 2023-01-01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.betfair import attach_bsp, load_bsp

logger = logging.getLogger("audit_labels")

PLACE_BSP = os.path.join("data", "processed", "betfair_place_bsp.parquet")


def load_results(since: str | None = None) -> pd.DataFrame:
    from src.database import load_from_database

    df = load_from_database()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    if since:
        df = df[df["race_date"] >= pd.Timestamp(since)]
    for col in ("finish_position", "won", "num_runners"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def internal_checks(df: pd.DataFrame) -> dict:
    """Consistency the DB can be checked against on its own."""
    g = df.groupby("race_id", sort=False)
    winners = g["won"].sum()
    fp = df["finish_position"]

    # A dead heat legitimately produces two winners, so >1 is reported
    # separately from 0 rather than lumped in as "bad".
    n_races = int(len(winners))
    no_winner = winners[winners == 0]
    multi_winner = winners[winners > 1]

    # won should be exactly finish_position == 1.
    fp1 = (fp == 1)
    won1 = (df["won"] == 1)
    disagree = int((fp1 != won1).sum())

    # Declared field size versus rows actually stored.
    sizes = g.size()
    declared = g["num_runners"].first()
    both = pd.DataFrame({"stored": sizes, "declared": declared}).dropna()
    short = both[both["stored"] < both["declared"]]

    return {
        "races": n_races,
        "runners": int(len(df)),
        "races_without_a_winner": int(len(no_winner)),
        "races_without_a_winner_pct": round(100 * len(no_winner) / max(n_races, 1), 3),
        "races_with_multiple_winners": int(len(multi_winner)),
        "dead_heat_rate_pct": round(100 * len(multi_winner) / max(n_races, 1), 3),
        "won_vs_finish_position_disagreements": disagree,
        "non_finishers_pct": round(100 * float((fp.fillna(0) <= 0).mean()), 2),
        "races_with_fewer_rows_than_declared": int(len(short)),
        "median_rows_missing_when_short": (
            float((short["declared"] - short["stored"]).median()) if len(short) else 0.0
        ),
    }


def betfair_checks(df: pd.DataFrame) -> dict:
    """The real audit: our label against the exchange's settled result."""
    bsp = load_bsp()
    if bsp is None or bsp.empty:
        return {"available": False, "reason": "betfair_bsp.parquet not loadable"}

    # attach_bsp assigns `out[c] = merged[c]`, so asking it for "won" would
    # SILENTLY OVERWRITE our own label with Betfair's and the audit would
    # compare the exchange against itself, always agreeing. Rename ours first.
    left = df.rename(columns={"won": "won_db"})
    joined = attach_bsp(left, bsp, cols=("won", "n_runners"))
    ours = pd.to_numeric(joined["won_db"], errors="coerce")
    theirs = pd.to_numeric(joined["won"], errors="coerce")
    assert "won_db" in joined.columns, "our label must survive the join"

    matched = ours.notna() & theirs.notna()
    agree = (ours[matched] == theirs[matched])
    ours_only = int(((ours == 1) & (theirs == 0) & matched).sum())
    theirs_only = int(((ours == 0) & (theirs == 1) & matched).sum())
    return {
        "available": True,
        "runners_matched": int(matched.sum()),
        "match_rate_pct": round(100 * float(matched.mean()), 2),
        "label_disagreements": int((~agree).sum()),
        "label_disagreement_pct": round(100 * float((~agree).mean()), 4) if matched.sum() else None,
        "we_say_won_betfair_says_lost": ours_only,
        "betfair_says_won_we_say_lost": theirs_only,
    }


def place_checks(df: pd.DataFrame) -> dict:
    """Place labels against the exchange's settled place market."""
    if not os.path.isfile(PLACE_BSP):
        return {"available": False, "reason": f"{PLACE_BSP} missing"}
    from src.betfair import bf_horse_key, bf_track_key
    from src.utils import normalise_off_time_key

    place = pd.read_parquet(
        PLACE_BSP, columns=["race_date", "track", "off_key", "horse_key",
                            "bf_places_paid", "placed_bf"],
    )
    place["race_date"] = pd.to_datetime(place["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    place["track_key"] = place["track"].map(bf_track_key)
    right = place.drop_duplicates(["race_date", "track_key", "off_key", "horse_key"])

    left = pd.DataFrame({
        "race_date": pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "track_key": df["track"].map(bf_track_key),
        "off_key": df["off_time"].map(normalise_off_time_key),
        "horse_key": df["horse_name"].map(bf_horse_key),
        "fp": pd.to_numeric(df["finish_position"], errors="coerce"),
    })
    m = left.merge(right, on=["race_date", "track_key", "off_key", "horse_key"], how="left")
    ok = m["placed_bf"].notna() & m["bf_places_paid"].notna() & m["fp"].notna()
    derived = ((m["fp"] > 0) & (m["fp"] <= pd.to_numeric(m["bf_places_paid"], errors="coerce"))).astype(int)
    theirs = pd.to_numeric(m["placed_bf"], errors="coerce").fillna(0).astype(int)
    disagree = int((derived[ok] != theirs[ok]).sum())
    return {
        "available": True,
        "runners_matched": int(ok.sum()),
        "placed_disagreements": disagree,
        "placed_disagreement_pct": (round(100 * disagree / int(ok.sum()), 4)
                                    if int(ok.sum()) else None),
        "note": "derived = finish_position within Betfair's places-paid",
    }


def audit(since: str | None = None) -> dict:
    df = load_results(since)
    return {
        "since": since or str(df["race_date"].min().date()),
        "until": str(df["race_date"].max().date()),
        "internal": internal_checks(df),
        "vs_betfair_win": betfair_checks(df),
        "vs_betfair_place": place_checks(df),
    }


def render(report: dict) -> str:
    i, w, p = report["internal"], report["vs_betfair_win"], report["vs_betfair_place"]
    lines = [
        "",
        f"  === Label audit: {report['since']} .. {report['until']} ===",
        f"  {i['runners']:,} runners over {i['races']:,} races",
        "",
        "  -- internal consistency --",
        f"    races with no winner stored  : {i['races_without_a_winner']:,} "
        f"({i['races_without_a_winner_pct']}%)",
        f"    races with >1 winner         : {i['races_with_multiple_winners']:,} "
        f"({i['dead_heat_rate_pct']}%)   <- dead heats are legitimate",
        f"    won != (finish_position==1)  : {i['won_vs_finish_position_disagreements']:,}",
        f"    non-finishers (fp<=0)        : {i['non_finishers_pct']}% of runners",
        f"    races short of declared field: {i['races_with_fewer_rows_than_declared']:,}"
        f"   (median {i['median_rows_missing_when_short']:.0f} rows missing)",
        "",
        "  -- win label vs Betfair's settled result --",
    ]
    if w.get("available"):
        lines += [
            f"    matched                      : {w['runners_matched']:,} runners "
            f"({w['match_rate_pct']}%)",
            f"    label disagreements          : {w['label_disagreements']:,} "
            f"({w['label_disagreement_pct']}%)",
            f"      we say won, Betfair lost   : {w['we_say_won_betfair_says_lost']:,}",
            f"      Betfair won, we say lost   : {w['betfair_says_won_we_say_lost']:,}",
        ]
    else:
        lines.append(f"    unavailable: {w.get('reason')}")
    lines += ["", "  -- place label vs Betfair's settled place market --"]
    if p.get("available"):
        lines += [
            f"    matched                      : {p['runners_matched']:,} runners",
            f"    placed disagreements         : {p['placed_disagreements']:,} "
            f"({p['placed_disagreement_pct']}%)",
        ]
    else:
        lines.append(f"    unavailable: {p.get('reason')}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", default=None, help="ISO date to start from")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    report = audit(args.since)
    print(json.dumps(report, indent=2, default=str) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
