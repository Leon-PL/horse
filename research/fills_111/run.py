"""How much of the 11:00 price could actually be taken, and for how long (research 111).

    python -m research.fills_111.run [--day 2026-11-16]        # every paper day with stored prices, or one

The paper record takes each bet at the best Matchbook back at 11:00 and walks a £10 stake down three levels. That
says nothing about whether the price lasted, how much money stood at it later, or where the book closed. This reads
every stored Matchbook snapshot after 11:00 (``data/matchbook/snapshots``: half-hourly whole books plus one a minute
in the last 20 minutes before each race) and follows each runner of every eligible race from 11:00 to its off.

Per runner (``runners.parquet``):

- ``taken``: the 11:00 best back (the paper price); ``stake_1100``: money offered at ``taken`` or better at 11:00;
- ``held_share``: share of the time from 11:00 to the off that ``taken`` or better was still on offer, each snapshot
  standing until the next one; ``held_minutes``: minutes until it first went; ``held_to_close``: still there at the
  last pre-off quote;
- ``max_stake``: the most money ever seen at ``taken`` or better in one snapshot (the most one bet could have got);
- ``close_back`` / ``close_ratio``: the last pre-off best back and ``taken / close_back``; with a settled day,
  ``bsp`` and ``clv`` (taken / BSP) for the bets.

The summary (``summary.json``) separates the rule's bets from all eligible runners and splits by price band. Ratios
are read as geometric means and medians, never arithmetic means (see ``src.clv``). Read-only over the paper record.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.matchbook import paper, record

OUT = Path("data/research/fills_111")
LEVELS = (1, 2, 3)
BANDS = [1, 3, 6, 11, 21, 52]


def stake_at(rows, price):
    """Money offered at ``price`` or better (back odds at least ``price``) across the three back levels."""
    total = np.zeros(len(rows))
    for level in LEVELS:
        odds, size = rows[f"back{level}"].to_numpy(float), rows[f"back{level}_size"].to_numpy(float)
        total += np.where((odds >= price - 1e-9) & (size > 0), size, 0.)
    return total


def follow(path, taken, start, off):
    """One runner's quotes from ``start`` (11:00) to ``off``: how long and how much ``taken`` stayed available."""
    path = path.loc[(path.taken_at >= start) & (path.taken_at < off) & ~path.in_running].sort_values("taken_at")
    if path.empty or not taken > 1:
        return {}
    times = path.taken_at.reset_index(drop=True)
    ends = pd.concat([times.iloc[1:], pd.Series([off])], ignore_index=True)
    span = (ends - times).dt.total_seconds().to_numpy()   # each snapshot stands until the next, the last until off
    stake = stake_at(path, taken)
    held = stake > 0
    total = span.sum()
    gone = np.flatnonzero(~held)
    first_gone = (times.iloc[gone[0]] - start).total_seconds() / 60 if len(gone) else np.nan
    last = path.iloc[-1]
    return {"snapshots": int(len(path)), "stake_1100": float(stake[0]), "max_stake": float(stake.max()),
            "held_share": float((span * held).sum() / total) if total > 0 else np.nan,
            "held_minutes": first_gone if len(gone) else (off - start).total_seconds() / 60,
            "held_to_close": bool(held[-1]), "close_back": float(last.back1) if last.back1 > 1 else np.nan,
            "close_at": pd.Timestamp(last.taken_at).tz_convert(record.LONDON).strftime("%H:%M"),
            "close_market_volume": float(last.market_volume) if pd.notna(last.market_volume) else np.nan}


def study_day(day):
    folder = paper.day_dir(day)
    if not (folder / "forecast.parquet").exists():
        return pd.DataFrame()
    forecast = pd.read_parquet(folder / "forecast.parquet")
    forecast = forecast.loc[forecast.eligible_race & forecast.runner_id.notna()].reset_index(drop=True)
    quotes = record.read_day(day)
    if forecast.empty or quotes.empty:
        return pd.DataFrame()
    start = pd.Timestamp(forecast.snapshot_1100.iloc[0])
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize(record.LONDON).tz_convert("UTC")
    by_runner = {k: g for k, g in quotes.groupby("runner_id")}
    offs = quotes.drop_duplicates("runner_id").set_index("runner_id").start_utc
    rows = []
    for r in forecast.itertuples(index=False):
        path = by_runner.get(r.runner_id)
        off = offs.get(r.runner_id)
        if path is None or off is None or pd.isna(off):
            continue
        info = follow(path, r.back1, start, off)
        if info:
            rows.append({"race_date": day, "race_id": r.race_id, "horse_id": r.horse_id, "horse_name": r.horse_name,
                         "track": r.track, "off_time": r.off_time, "bet": bool(r.bet), "edge": r.edge, "p": r.p,
                         "share": r.share, "race_tight": r.race_tight, "taken": r.back1,
                         "fill_stake": r.fill_stake, "fill_price": r.fill_price, **info})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["close_ratio"] = out.taken / out.close_back
    if (folder / "settled.parquet").exists():
        settled = pd.read_parquet(folder / "settled.parquet")[["race_id", "horse_id", "bsp", "won"]]
        out = out.merge(settled, on=["race_id", "horse_id"], how="left")
        out["clv"] = out.taken / out.bsp
    return out


def _geo(x):
    x = pd.Series(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else None


def describe(frame):
    if frame.empty:
        return {"runners": 0}
    out = {"runners": int(len(frame)), "races": int(frame.race_id.nunique()),
           "median_stake_1100": float(frame.stake_1100.median()),
           "share_with_10_at_1100": float((frame.stake_1100 >= 10).mean()),
           "median_max_stake": float(frame.max_stake.median()),
           "median_held_share": float(frame.held_share.median()),
           "median_held_minutes": float(frame.held_minutes.median()),
           "share_held_to_close": float(frame.held_to_close.mean()),
           "geo_taken_over_close": _geo(frame.close_ratio),
           "median_taken_over_close": float(frame.close_ratio.median()) if frame.close_ratio.notna().any() else None}
    if "clv" in frame and frame.clv.notna().any():
        out["geo_clv_bsp"] = _geo(frame.clv)
        out["median_clv_bsp"] = float(frame.clv.median())
    return out


def summarise(frame):
    bands = pd.cut(frame.taken, BANDS, right=False)
    return {"days": sorted(frame.race_date.unique().tolist()), "bets": describe(frame.loc[frame.bet]),
            "eligible_runners": describe(frame),
            "positive_edge_not_bet": describe(frame.loc[~frame.bet & frame.edge.gt(0)]),
            "eligible_by_price": {str(b): describe(g) for b, g in frame.groupby(bands, observed=True)},
            "bets_tight_books": describe(frame.loc[frame.bet & frame.race_tight.ge(paper.TIGHT)]),
            "bets_thin_books": describe(frame.loc[frame.bet & frame.race_tight.lt(paper.TIGHT)])}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default=None)
    args = parser.parse_args(argv)
    days = [args.day] if args.day else sorted(p.name for p in paper.PAPER.glob("*") if (p / "forecast.parquet").exists())
    frames = [study_day(d) for d in days]
    frame = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) \
        else pd.DataFrame()
    if frame.empty:
        raise SystemExit("No paper day with an 11:00 forecast and stored prices")
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT / "runners.parquet", index=False)
    summary = summarise(frame)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    cols = ["race_date", "off_time", "track", "horse_name", "taken", "fill_stake", "stake_1100", "max_stake",
            "held_minutes", "held_share", "held_to_close", "close_back", "close_at"]
    print(frame.loc[frame.bet, [c for c in cols if c in frame]].round(2).to_string(index=False))
    print(json.dumps({"bets": summary["bets"], "eligible_runners": summary["eligible_runners"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
