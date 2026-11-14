"""Owners: the model's past errors by who owns the horse.

The fourth pass (``fourth_pass.py``) puts the whole remaining loss to the 08:00
exchange price in races full of unraced and lightly raced horses, mostly flat
non-handicaps. For those horses the model has pedigree, trainer and jockey,
and nothing about the owner, though the market reads a great deal into the
colours a newcomer carries. Owner names are in the Racing TV archive from
September 2020 and were never joined to the results.

``owner``, declared before scoring: the same shrunk, 365-day-decayed residual
and market-gap means as the trainer and sire inputs (``entity_residuals``),
keyed by owner, owner with trainer, and owner with the horse's state (debut,
back from a break, fit). Strictly earlier dates only. The name is the one on
the result or card of that race, so a later sale cannot reach back.
"""
import re

import numpy as np
import pandas as pd

from research.error_anatomy import entity_residuals as er
from research.error_anatomy.frame import OUT
from research.meta_learner import screening as s
from src.ingestion.pipeline import canonical_rows, match_runners
from src.ingestion.rtv_observations import observation_tables
from src.ingestion.storage import atomic_parquet

KEYS = s.KEYS
SAVED = OUT / "owners.parquet"
ENTITIES = {"owner": ["owner"], "owner_trainer": ["owner", "trainer"], "owner_fresh": ["owner", "_fresh"]}
TITLES = re.compile(r"\b(mr|mrs|ms|miss|dr|sir|lady|lord|the|and|hh|sheikh|shaikh|h h)\b")


def owner_key(names):
    """Lower-case owner name without titles or punctuation, so 'Mr J. Smith' and 'J Smith' agree."""
    text = pd.Series(names, dtype="string").str.lower().str.replace("&", " and ", regex=False)
    text = text.str.replace(r"[^a-z0-9 ]", " ", regex=True).str.replace(TITLES, " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    return text.where(text.str.len() > 0)


def linked_runner_columns(columns, start="2020-09-01", end="2026-06-30"):
    """Named Racing TV runner columns for every linked starter, by this project's race and horse IDs."""
    race_cols = ["race_id", "race_date", "track_name", "off_time_utc", "is_abandoned", "is_arab_race"]
    runner_cols = ["race_id", "runner_id", "horse_name", "in_results", "is_nonrunner", *columns]
    parts = []
    for year in range(int(start[:4]), int(end[:4]) + 1):  # a year at a time keeps memory flat
        lo, hi = max(start, f"{year}-01-01"), min(end, f"{year}-12-31")
        tables = observation_tables(lo, hi, names=("runners",), column_overrides={"races": race_cols, "runners": runner_cols})
        runners = tables["runners"].reindex(columns=runner_cols)
        runners = runners.loc[runners.in_results.eq(True) & ~runners.is_nonrunner.fillna(False).astype(bool)]
        links = match_runners(tables["races"], runners, canonical_rows(lo, hi))
        linked = links.merge(runners[["race_id", "runner_id", *columns]], left_on=["rtv_race_id", "rtv_runner_id"],
                             right_on=["race_id", "runner_id"], validate="one_to_one")
        part = pd.DataFrame({"race_id": linked.sl_race_id.astype(str),
                             "horse_id": pd.to_numeric(linked.sl_horse_id).astype("int64")})
        for column in columns:
            part[column] = linked[column].to_numpy()
        parts.append(part)
        print(year, len(part), flush=True)
    out = pd.concat(parts, ignore_index=True)
    return out.loc[~out.duplicated(KEYS, keep=False)].reset_index(drop=True)


def build(start="2020-09-01", end="2026-06-30"):
    """Owner of every linked starter, by this project's race and horse IDs."""
    out = linked_runner_columns(["owner"], start, end)
    out["owner"] = owner_key(out.owner)
    atomic_parquet(out, SAVED)
    return out


def owner_inputs(frame):
    owners = pd.read_parquet(SAVED) if SAVED.exists() else build()
    owners["horse_id"] = owners.horse_id.astype(frame.horse_id.dtype)
    work = frame.merge(owners, on=KEYS, how="left", validate="one_to_one")
    work.index = frame.index
    return er.entity_inputs(work, market="p_market", half_life=er.LONG_HALF_LIFE, entities=ENTITIES, prefix="ow",
                            extra_keys={"owner": work.owner.to_numpy(object)})


def load(frame):
    values = owner_inputs(frame)
    return values, [c for c in values if c not in KEYS]


if __name__ == "__main__":
    table = build()
    print(len(table), "runners,", table.owner.nunique(), "owners,", f"{table.owner.notna().mean():.3f} named")
    print(table.owner.value_counts().head(12).to_string())
