"""Evaluation prices selected by the training manifest, with persisted links.

The collector's native price matcher stays unchanged while its backfill runs.
Research callers without a manifest retain the explicit native-cache interface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.betfair import attach_bsp as native_attach_bsp, load_bsp as native_load_bsp
from src.source_dataset import current_manifest, read_table, source_path
from src.source_contracts import _ids, resolve_runners


def load_bsp(path=None):
    if path is not None or current_manifest() is None:
        return native_load_bsp(path)
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


def attach_bsp(df, bsp=None, *, cols=("bsp", "ltp_preoff", "ltp_60s", "ltp_300s", "ltp_last"), overwrite=False, **kwargs):
    bsp = load_bsp() if bsp is None else bsp
    if bsp is None and current_manifest() is not None:
        if not overwrite and set(cols) & set(df):
            raise ValueError("attach_bsp would replace existing columns")
        out = df.copy()
        for column in cols:
            out[column] = np.nan
        return out
    if bsp is None or not {"sl_race_id", "sl_horse_id", "match_status"}.issubset(bsp):
        return native_attach_bsp(df, bsp, cols=cols, overwrite=overwrite, **kwargs)
    if not overwrite and set(cols) & set(df):
        raise ValueError("attach_bsp would replace existing columns")
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
            return native_attach_bsp(df, pd.DataFrame(), cols=cols, overwrite=overwrite, **kwargs)
    joined = left.merge(source, on=keys, how="left", validate="many_to_one")
    out = df.copy()
    for column in cols:
        out[column] = joined[column].to_numpy() if column in joined else np.nan
    return out
