"""Read accepted French and UAE races (complete fields) for history calculations."""
from functools import lru_cache
import hashlib

import pandas as pd

from src.source_dataset import require_manifest, source_path


@lru_cache(maxsize=3)
def _history(path):
    frame = pd.read_parquet(path)
    if not frame.country.isin(["fr", "ae"]).all():
        raise ValueError("Unexpected country in the foreign-form history")
    frame["race_date"] = pd.to_datetime(frame.race_date, errors="raise")
    available = pd.to_datetime(frame.available_date, errors="raise")
    if not available.eq(frame.race_date + pd.Timedelta(days=1)).all():
        raise ValueError("Invalid availability in the foreign-form history")
    # With no reliable intraday publication timestamp, release every start
    # at the close of its race date, strictly before the following day's cards.
    frame["_event_dt"] = available - pd.Timedelta(nanoseconds=1)
    return frame


def version():
    table = require_manifest()["sources"].get("sl", {}).get("tables", {}).get("foreign_form")
    return table["sha256"] if table else None


def history(*, through=None, after=None):
    if version() is None:
        return pd.DataFrame()
    frame = _history(source_path("sl", "foreign_form"))
    if through is not None:
        frame = frame.loc[frame._event_dt.lt(pd.Timestamp(through).normalize())]
    if after is not None:
        frame = frame.loc[frame._event_dt.ge(pd.Timestamp(after).normalize())]
    return frame


def signature(cutoff):
    frame = history(through=cutoff)
    if frame.empty:
        return None
    # Every field a form, trip or rating calculation reads. Ratings group by
    # race, so the race identity and the whole field are part of the input.
    fields = ["horse_id", "race_date", "provider", "native_race_id", "race_type", "surface",
              "distance_furlongs", "finish_position", "finish_pos_label", "num_runners", "beaten_lengths"]
    return hashlib.sha256(pd.util.hash_pandas_object(frame[fields], index=False).values.tobytes()).hexdigest()
