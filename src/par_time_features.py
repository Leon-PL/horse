"""Time-based, going-corrected speed figures from Sporting Life winning times.

Motivation
----------
The existing ``speed_fig_*`` family (``feature_engineer.add_speed_figure_features``)
is a *proxy*: it derives a figure from ``lengths_behind`` per furlong only. With no
absolute clock it cannot compare a horse across different races, days or going — a
fast time on soft ground looks identical to a slow time on firm if the winning
margins match.

Sporting Life publishes an actual ``winning_time`` for every race. This module turns
that into a cross-race-comparable speed figure with the going removed, the way
par-time figures (Beyer / Timeform variant) are built:

1. **Par** ``par_spf[surface, dist]`` — robust median winning seconds-per-furlong over
   all history. A structural track/trip constant (not a per-horse quantity, so not a
   betting leak).
2. **Track variant / going allowance** ``variant[track, date, surface]`` — the
   *leave-one-race-out* mean deviation of that day's winners from par, in
   seconds-per-furlong. This is the measured ground speed for the day, independent of
   the described going word.
3. **Per-race figure** — each runner's time = ``winning_time + lengths_behind`` in
   seconds (lengths converted via the race's own furlongs-per-second), corrected by the
   day's variant and compared to par. Higher = faster.

The per-race figure is a *result* of that race; it is only ever consumed as a
**strictly-past** per-horse rolling feature (``shift(1)`` inside a grouped transform —
CLAUDE.md hard rule #1), exactly like the existing speed figures.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

# Geometry: 1 furlong = 220 yd; a horse length ~ 2.4 m ~ 2.625 yd -> ~83.8 lengths/furlong.
# Used only to convert a beaten distance (lengths) into seconds via each race's own pace,
# so the exact constant cancels to first order — kept explicit for clarity.
_LENGTHS_PER_FURLONG = 83.8
_FIG_SCALE = 100.0  # arbitrary points-per-(sec/furlong); makes figures human-readable

_WT_RE = re.compile(r"(?:(\d+)\s*m)?\s*([\d.]+)\s*s", re.I)


def parse_winning_time(raw) -> float:
    """'1m 13.24s' -> 73.24 ; '59.34s' -> 59.34 ; junk/empty -> NaN."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    m = _WT_RE.search(str(raw).strip())
    if not m:
        return np.nan
    mins = float(m.group(1)) if m.group(1) else 0.0
    secs = float(m.group(2))
    v = mins * 60.0 + secs
    return v if 5.0 < v < 1200.0 else np.nan


def load_winning_times(path: str = "data/raw/race_winning_times.csv") -> pd.DataFrame:
    """Backfill artifact -> tidy (race_id, winning_time_secs). Resilient if absent."""
    import os
    if not os.path.exists(path):
        return pd.DataFrame(columns=["race_id", "winning_time_secs"])
    wt = pd.read_csv(path, dtype={"race_id": str})
    if "winning_time_secs" not in wt.columns or wt["winning_time_secs"].isna().all():
        wt["winning_time_secs"] = wt["winning_time_raw"].map(parse_winning_time)
    wt = wt.dropna(subset=["winning_time_secs"])
    wt = wt.drop_duplicates("race_id", keep="first")
    return wt[["race_id", "winning_time_secs"]]


def add_time_speed_figure(
    df: pd.DataFrame,
    winning_times: pd.DataFrame | None = None,
    *,
    wt_path: str = "data/raw/race_winning_times.csv",
) -> pd.DataFrame:
    """Attach the per-race going-corrected time speed figure ``tspf``.

    Requires columns: race_id, surface, distance_furlongs, lengths_behind,
    finish_position, track, race_date. Rows with no winning time get ``tspf`` = NaN.
    """
    df = df.copy()
    if winning_times is None:
        winning_times = load_winning_times(wt_path)
    df["race_id"] = df["race_id"].astype(str)

    df = df.merge(winning_times, on="race_id", how="left")

    dist = pd.to_numeric(df["distance_furlongs"], errors="coerce")
    wt = pd.to_numeric(df["winning_time_secs"], errors="coerce")
    lb = pd.to_numeric(df["lengths_behind"], errors="coerce")

    # Pace of the winner (furlongs/sec) -> lengths/sec -> seconds per beaten length.
    lengths_per_sec = (dist * _LENGTHS_PER_FURLONG) / wt
    runner_extra = lb / lengths_per_sec          # seconds the runner finished behind
    runner_time = wt + runner_extra.fillna(0.0)  # winner: lb=0 -> winning_time
    df["_spf"] = runner_time / dist               # runner seconds-per-furlong (raw pace)

    # Par seconds-per-furlong per (surface, distance) from winners only.
    surf = df["surface"].astype(str).fillna("Turf")
    win_mask = pd.to_numeric(df["finish_position"], errors="coerce") == 1
    df["_surf"] = surf
    df["_distkey"] = dist.round(1)
    par = (
        df.loc[win_mask & wt.notna()]
        .groupby(["_surf", "_distkey"])["_spf"].median()
        .rename("_par_spf")
    )
    df = df.merge(par, on=["_surf", "_distkey"], how="left")

    # Track variant (going allowance), seconds/furlong, leave-one-RACE-out within
    # (track, date, surface): mean winner deviation from par over the OTHER races.
    df["_dev"] = df["_spf"] - df["_par_spf"]
    win = df[win_mask & df["_par_spf"].notna()].copy()
    win["_daykey"] = list(zip(win["track"].astype(str), win["race_date"].astype(str), win["_surf"]))
    grp = win.groupby("_daykey")["_dev"]
    g_sum = grp.transform("sum")
    g_cnt = grp.transform("count")
    # leave-one-out mean over the day's winners (excluding this race's winner)
    win["_variant"] = np.where(g_cnt > 1, (g_sum - win["_dev"]) / (g_cnt - 1), np.nan)
    # one variant per (track,date,surface): all races on the day share it; map back
    day_variant = win.groupby("_daykey")["_variant"].mean()
    df["_daykey"] = list(zip(df["track"].astype(str), df["race_date"].astype(str), df["_surf"]))
    df["_variant"] = df["_daykey"].map(day_variant)

    # Going-corrected figure: how much faster than par (sec/furlong) after removing the
    # day's ground speed. Positive = faster. Scaled to readable points.
    corrected = (df["_par_spf"] - (df["_spf"] - df["_variant"].fillna(0.0)))
    df["tspf"] = corrected * _FIG_SCALE
    # Only meaningful where we actually had a clock for the race.
    df.loc[wt.isna(), "tspf"] = np.nan

    df = df.drop(columns=["_spf", "_surf", "_distkey", "_par_spf", "_dev",
                          "_variant", "_daykey", "winning_time_secs"], errors="ignore")
    return df


def add_past_time_speed_features(
    df: pd.DataFrame, horse_key_col: str = "horse_name"
) -> pd.DataFrame:
    """Strictly-past rolling features from the per-race figure ``tspf``.

    Leakage-safe: ``shift(1)`` is applied *inside* the per-horse transform so a horse's
    figure for the current race is never used for that race (CLAUDE.md rule #1).
    Requires the frame sorted by date within horse; we sort defensively.
    """
    if "tspf" not in df.columns:
        df = add_time_speed_figure(df)
    df = df.copy()
    sort_cols = [c for c in ["race_date", "off_time", "race_id"] if c in df.columns]
    df = df.sort_values([horse_key_col, *sort_cols]).reset_index(drop=True)
    g = df.groupby(horse_key_col, sort=False)["tspf"]

    past = g.transform(lambda s: s.shift(1))
    df["tspf_avg_3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["tspf_avg_5"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["tspf_best_5"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).max())
    df["tspf_last"] = past
    df["tspf_trend"] = (
        g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        - g.transform(lambda s: s.shift(1).rolling(8, min_periods=2).mean())
    )
    df["tspf_n_past"] = g.transform(lambda s: s.shift(1).notna().cumsum())
    return df
