"""Matchbook price snapshots on disk.

``data/matchbook/snapshots/<UK date>/<HHMMSS>_<label>.parquet``, one file per snapshot, every listed British and
Irish race. ``watch`` keeps the near-off book: every minute it stores the races due off within ``window`` minutes,
plus a whole-day snapshot every half hour, until the day's last race has gone off. The last pre-off row of each
runner is its Matchbook close.
"""
import glob
import time
from pathlib import Path

import pandas as pd

from src.ingestion.storage import atomic_parquet
from src.matchbook import feed

ROOT = Path("data/matchbook")
SNAPSHOTS = ROOT / "snapshots"
LONDON = feed.LONDON


def uk_now():
    return pd.Timestamp.now(tz=LONDON)


def save(frame, label, when=None):
    when = when or uk_now()
    target = SNAPSHOTS / str(when.date()) / f"{when.strftime('%H%M%S')}_{label}.parquet"
    atomic_parquet(frame, target)
    return target


def snapshot(label, http=None):
    frame = feed.snapshot_frame(http)
    if frame.empty:
        return None, frame
    return save(frame, label, frame.taken_at.iloc[0].tz_convert(LONDON)), frame


def _paths(day, whole_only=False):
    paths = sorted(glob.glob(str(SNAPSHOTS / str(pd.Timestamp(day).date()) / "*.parquet")))
    return [p for p in paths if not (whole_only and Path(p).stem.endswith("_close"))]


def read_day(day):
    """Every stored snapshot row taken on a UK date."""
    frames = [pd.read_parquet(p) for p in _paths(day)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def at(day, clock):
    """The latest whole snapshot (not a near-off one) taken at or before ``clock`` (UK time) on ``day``."""
    cut = str(pd.Timestamp(f"{pd.Timestamp(day).date()} {clock}").strftime("%H%M%S"))
    paths = [p for p in _paths(day, whole_only=True) if Path(p).stem[:6] <= cut]
    return pd.read_parquet(paths[-1]) if paths else pd.DataFrame()


def closes(day):
    """Each runner's last stored row before its race's scheduled start, not in running."""
    rows = read_day(day)
    if rows.empty:
        return rows
    rows = rows.loc[(rows.taken_at < rows.start_utc) & ~rows.in_running]
    rows = rows.sort_values("taken_at").groupby(["event_id", "runner_id"], as_index=False).tail(1)
    return rows.reset_index(drop=True)


def watch(*, window=20, every=60, full_every=1800, until=None, http=None, log=print):
    """Record until the last listed race of today has started (or ``until``, a UK clock time)."""
    http = http or feed.session()
    last_full = 0.
    while True:
        frame = feed.snapshot_frame(http)
        now = uk_now()
        today = frame.loc[frame.race_date.eq(str(now.date()))] if not frame.empty else frame
        if not frame.empty:
            if time.time() - last_full >= full_every:
                save(frame, "full", now)
                last_full = time.time()
            due = today.start_utc - pd.Timestamp.now(tz="UTC")
            near = today.loc[(due > pd.Timedelta(0)) & (due <= pd.Timedelta(minutes=window))]
            if not near.empty:
                save(near, "close", now)
        pending = today.loc[today.start_utc > pd.Timestamp.now(tz="UTC")] if not today.empty else today
        stop = until and now >= pd.Timestamp(f"{now.date()} {until}").tz_localize(LONDON)
        if pending.empty or stop:
            log(f"{now:%H:%M} watch finished: no race of today still to start")
            return
        time.sleep(every)
