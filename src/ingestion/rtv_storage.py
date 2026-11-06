"""Published Racing TV generations and compatible historical partitions."""
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import pandas as pd
import config
from src.ingestion.storage import read_json

_pinned = ContextVar("rtv_generation", default=None)


def root():
    return Path(config.DATA_DIR)/"rtv_cache"


def generation(source_root=None):
    if source_root is None and _pinned.get() is not None:
        return _pinned.get()
    return read_json(Path(source_root or root())/"pipeline/generation.json")


@contextmanager
def pin_generation():
    token = _pinned.set(generation())
    try:
        yield
    finally:
        _pinned.reset(token)


def history_manifest(source_root=None):
    base = Path(source_root or root())
    return generation(base).get("history", read_json(base/"pipeline/history.json"))


def materialised_path(name, fallback):
    base = root()
    relative = generation().get("files", {}).get(name)
    return base/relative if relative else Path(fallback)


def partition_paths(directory, start=None, end=None):
    """Read old explicit ranges as well as quarters, newest replacement first."""
    paths = []
    for path in Path(directory).glob("*.parquet"):
        quarter = re.fullmatch(r"(\d{4})-Q([1-4])", path.stem)
        interval = re.fullmatch(r"(first|\d{4}-\d{2}-\d{2})_(last|\d{4}-\d{2}-\d{2})", path.stem)
        if quarter:
            year, q = map(int, quarter.groups())
            lo = f"{year}-{3*q-2:02d}-01"
            hi = str((pd.Timestamp(lo)+pd.offsets.QuarterEnd()).date())
        elif interval:
            lo, hi = interval.groups()
            lo = "0000" if lo == "first" else lo
            hi = "9999" if hi == "last" else hi
        else:
            raise ValueError(f"Unsupported RTV partition {path.name}; rebuild with python -m src.ingestion.racing_tv build")
        if (not start or hi >= start) and (not end or lo <= end):
            paths.append(path)
    return sorted(paths, key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)


def accepted_days(source_root=None, *, manifest=None):
    base = Path(source_root or root())
    days = (manifest if manifest is not None else history_manifest(base)).get("days", {})
    return {day: entry for day, entry in days.items()
            if entry.get("complete", True) and (base/"pipeline"/entry["directory"]/"races.parquet").exists()}
