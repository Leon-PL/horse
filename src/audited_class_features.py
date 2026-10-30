"""Research class transitions which preserve an unavailable class as unknown."""
from __future__ import annotations

import numpy as np
import pandas as pd


def uk_class_number(values):
    """UK class1..7 only; group/grade numbers are not UK class numbers."""
    text=pd.Series(values,copy=False).astype("string").str.strip()
    number=text.str.extract(r"(?i)^(?:class\s*)?([1-7])(?:\.0+)?$",expand=False)
    return pd.to_numeric(number,errors="coerce").astype(float)


def class_transition_features(cards,history):
    """Latest actual prior start, never the latest convenient nonmissing class."""
    from src.feature_engineer import _add_result_history_flags

    def ids(values):
        number=pd.to_numeric(values,errors="coerce")
        if (number.notna() & number.mod(1).ne(0)).any():
            raise ValueError("Class history requires integral source horse IDs")
        return number.astype("Int64").astype("string")

    cols=["horse_id","race_date","race_id","race_class","finish_position","won"]
    cols += [c for c in ("finish_pos_label","num_runners") if c in history]
    past=_add_result_history_flags(history[cols].copy())
    past=past.loc[past._result_start.eq(1)].copy()
    past["_horse"]=ids(past.horse_id)
    past["_day"]=pd.to_datetime(past.race_date,errors="raise").dt.normalize()
    past["_class"]=uk_class_number(past.race_class)
    daily=past.loc[past._horse.notna()].groupby(["_horse","_day"])["_class"].agg(
        first="first",known="count",entries="size",distinct="nunique").reset_index()
    daily["previous_race_class"]=daily["first"].where(daily.known.eq(daily.entries)&daily.distinct.eq(1))
    left=pd.DataFrame({"_horse":ids(cards.horse_id).to_numpy(),
        "_day":pd.to_datetime(cards.race_date,errors="raise").dt.normalize().to_numpy(),
        "_row":np.arange(len(cards))})
    left["_horse"]=left["_horse"].astype("string")
    previous=np.full(len(cards),np.nan)
    known=left._horse.notna()
    if known.any() and not daily.empty:
        aligned=pd.merge_asof(left.loc[known].sort_values("_day",kind="stable"),
            daily[["_horse","_day","previous_race_class"]].sort_values("_day",kind="stable"),
            by="_horse",on="_day",direction="backward",allow_exact_matches=False)
        previous[aligned._row.to_numpy()]=aligned.previous_race_class.to_numpy()
    current=uk_class_number(cards.race_class).to_numpy()
    delta=current-previous
    out=pd.DataFrame({"current_class_known":np.isfinite(current).astype(np.int8),
        "previous_class_known":np.isfinite(previous).astype(np.int8),
        "class_transition_known":np.isfinite(delta).astype(np.int8),
        "previous_race_class":previous,"class_change_missing_aware":delta},index=cards.index)
    mean=out.class_change_missing_aware.groupby(cards.race_id.to_numpy()).transform("mean")
    out["class_change_vs_field_missing_aware"]=(out.class_change_missing_aware-mean).fillna(0.)
    if "distance_change" in cards:
        out["dist_x_class_change_missing_aware"]=cards.distance_change.to_numpy()*delta
    return out
