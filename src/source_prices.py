"""Evaluation prices from the selected dataset, attached by saved runner links.

Model fitting (the BSP teacher) and CLV read only the selected dataset's
``betfair`` tables. Prices join by canonical race and horse ID; a target with
only names is first resolved to IDs by the shared identity rule. The
collector's own name matcher (``src.betfair.attach_bsp``) is not used here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.source_dataset import current_manifest, read_table, source_path
from src.source_contracts import _ids, resolve_runners

PRICE_COLUMNS = ("bsp", "ltp_preoff", "ltp_60s", "ltp_300s", "ltp_last")
_LINKS = ("sl_race_id", "sl_horse_id", "match_status")


def load_bsp():
    """The selected dataset's linked BSP rows, or None when it has none."""
    frame = read_table("betfair", "bsp", linked=True)
    return frame if not frame.empty else None


def _canonical_target_keys(frame, options):
    manifest = current_manifest()
    if manifest is None or "sl" not in manifest["sources"]:
        return None
    columns = [options.get(name+"_col", default) for name, default in
               (("date", "race_date"), ("track", "track"), ("off", "off_time"), ("horse", "horse_name"))]
    if not set(columns).issubset(frame):
        return None
    keys = frame[columns].copy().reset_index(drop=True)
    keys.columns = ["race_date", "track", "off_time", "horse_name"]
    unique = keys.drop_duplicates().reset_index(drop=True)
    days = pd.to_datetime(unique.race_date, errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique().tolist()
    reference = pd.read_parquet(source_path("sl", "results"),
        columns=["race_id", "horse_id", "race_date", "track", "off_time", "horse_name"], filters=[("race_date", "in", days)])
    links = resolve_runners(unique, reference, provider="targets")
    mapping = pd.concat([unique, links[["sl_race_id", "sl_horse_id"]]], axis=1)
    resolved = keys.merge(mapping, on=list(keys.columns), how="left", validate="many_to_one")[["sl_race_id", "sl_horse_id"]]
    for column, key in (("race_id", "sl_race_id"), ("horse_id", "sl_horse_id")):
        if column in frame:
            given = _ids(frame[column]).reset_index(drop=True)
            conflict = given.notna() & given.ne(resolved[key]).fillna(True)
            resolved.loc[conflict, ["sl_race_id", "sl_horse_id"]] = pd.NA
    return resolved


def attach_bsp(df, bsp=None, *, cols=PRICE_COLUMNS, overwrite=False, **kwargs):
    """Add *cols* from linked price rows; unmatched runners get NaN.

    *bsp* defaults to the selected dataset's BSP table. A supplied frame must
    carry its saved links (``sl_race_id``, ``sl_horse_id``, ``match_status``).
    """
    if not overwrite and set(cols) & set(df):
        raise ValueError("attach_bsp would replace existing columns")
    bsp = load_bsp() if bsp is None else bsp
    out = df.copy()
    if bsp is None or bsp.empty:
        for column in cols:
            out[column] = np.nan
        return out
    if not set(_LINKS).issubset(bsp):
        raise ValueError("Prices must carry their saved runner links (sl_race_id, sl_horse_id, match_status)")
    source = bsp.loc[bsp.match_status.eq("matched"), ["sl_race_id", "sl_horse_id", *cols]].copy()
    keys = ["sl_race_id", "sl_horse_id"]
    for key in keys:
        source[key] = _ids(source[key])
    source = source.dropna(subset=keys)
    # Ambiguous source links remain unknown; never multiply target runners.
    source = source.loc[~source.duplicated(keys, keep=False)]
    if {"race_id", "horse_id"}.issubset(df):
        left = pd.DataFrame({"sl_race_id": _ids(df.race_id), "sl_horse_id": _ids(df.horse_id)})
    else:
        left = _canonical_target_keys(df, kwargs)
        if left is None:
            for column in cols:
                out[column] = np.nan
            return out
    joined = left.merge(source, on=keys, how="left", validate="many_to_one")
    for column in cols:
        out[column] = joined[column].to_numpy() if column in joined else np.nan
    return out
