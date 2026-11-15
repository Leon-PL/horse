"""The correction with the race's own 11:00 price: the forecast behind the Matchbook paper record.

The production correction (``src.correction``, groups ``GROUPS``) plus six inputs read from the 11:00 price of the
race itself (research 104/107, ``research/error_anatomy/morning_price.py``): log share of the field, its gap to the
win model's forecast, log price against the favourite, price rank, the move from 08:00 (each centred within the
race) and the share of the field priced. Research 107's two Betfair trade-activity inputs (seconds since the last
trade, number of price changes) are left out because Matchbook's feed has no trades (research 108).

Fitted on Betfair last traded prices at 08:00 and 11:00; served on Matchbook's back/lay middle
(``feed.reference_price``) at the same clock times. Races off at or before 11:00 have the inputs blanked.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.correction import data as cd
from src.correction import model as cm
from src.ingestion.storage import atomic_json

KEYS = cm.KEYS
PRICE_INPUTS = ("pm_logshare", "pm_gap", "pm_vs_favourite", "pm_rank", "pm_move_0800", "pm_field_priced")
FORMAT = "matchbook_1100_correction"


def price_inputs(race_id, p_single, price_1100, price_0800):
    """The six inputs from row-aligned arrays; a runner without an 11:00 price has them all NaN."""
    race = np.asarray(race_id)
    price = pd.Series(np.asarray(price_1100, float))
    price = price.where(price > 1)
    inverse = 1 / price
    share = np.log(inverse / inverse.groupby(race).transform("sum"))
    out = pd.DataFrame(index=price.index)
    out["pm_logshare"] = share
    gap = share - np.log(np.asarray(p_single, float))
    out["pm_gap"] = gap - gap.groupby(race).transform("mean")
    out["pm_vs_favourite"] = np.log(price / price.groupby(race).transform("min"))
    out["pm_rank"] = price.groupby(race).rank(method="average")
    early = pd.Series(np.asarray(price_0800, float))
    move = np.log(early.where(early > 1) / price)
    out["pm_move_0800"] = move - move.groupby(race).transform("mean")
    out["pm_field_priced"] = price.notna().groupby(race).transform("mean").to_numpy()
    return out[list(PRICE_INPUTS)].astype("float32")


def _betfair_prices(frame, manifest):
    """Betfair 08:00 and 11:00 last traded prices for the frame's runners, 11:00 blanked for races off by 11:00."""
    from src.source_dataset import load_manifest, read_table, use_manifest
    from src.source_prices import attach_bsp
    with use_manifest(load_manifest(manifest["id"], root=manifest.get("root"))):
        table = read_table("betfair", "timing", linked=True)
        out = attach_bsp(frame[KEYS].copy(), table, cols=("ltp_0800", "ltp_1100", "off_ms", "race_date"))
    eleven = (pd.to_datetime(out.race_date) + pd.Timedelta(hours=11)).dt.tz_localize(
        "Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    off = pd.to_datetime(out.off_ms, unit="ms", utc=True)
    out.loc[~(off > eleven).to_numpy(), "ltp_1100"] = np.nan
    return out.reset_index(drop=True)


def fit(as_of, output, *, store_root=None, threads=6):
    """Fit on every race with a forecast and an 11:00 Betfair price up to ``as_of``; save to ``output``."""
    frame, prices, tracking, receipt = cd.assemble(store_root, development=False, end=as_of)
    values, columns = cm.build_inputs(frame, prices=prices, tracking=tracking)
    rows = frame.p_base.notna().to_numpy()
    frame, values = frame.loc[rows].reset_index(drop=True), values.loc[rows].reset_index(drop=True)
    betfair = _betfair_prices(frame, receipt["dataset_manifest"])
    last_priced = pd.to_datetime(frame.race_date[betfair.ltp_1100.notna().to_numpy()]).max()
    keep = (pd.to_datetime(frame.race_date) <= last_priced).to_numpy()
    frame, values, betfair = (x.loc[keep].reset_index(drop=True) for x in (frame, values, betfair))
    extra = price_inputs(frame.race_id, frame.p_base, betfair.ltp_1100, betfair.ltp_0800)
    values = pd.concat([values, extra], axis=1)
    columns = [*columns, *PRICE_INPUTS]
    model = cm.CorrectionModel(columns).fit(frame, values, threads=threads)
    model.lineage.update({"as_of": str(pd.Timestamp(as_of).date()), "last_priced_race_date": str(last_priced.date()),
                          "fe_signature": receipt["fe_signature"], "dataset_manifest": receipt["dataset_manifest"]["id"],
                          "price_inputs": list(PRICE_INPUTS), "races": int(frame.race_id.nunique()),
                          "evidence": "research 107 (registered rule, hold-out pass); research 108 (without trade inputs)"})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"format": FORMAT, "version": 1, "model": model}, out / "correction.joblib")
    atomic_json(out / "lineage.json", model.lineage)
    return model


def load(folder):
    contents = joblib.load(Path(folder) / "correction.joblib")
    if contents.get("format") != FORMAT or contents.get("version") != 1:
        raise ValueError("Not a Matchbook 11:00 correction")
    return contents["model"]


def card_extra(prices):
    """The ``extra`` hook for ``src.correction.serve``: appends the six inputs from day prices.

    ``prices`` holds KEYS, ``price_1100`` and ``price_0800`` for card runners (missing runners get NaN).
    """
    def extra(part, values, columns):
        day = part[KEYS].merge(prices.astype({k: part[k].dtype for k in KEYS}), on=KEYS, how="left",
                               validate="one_to_one")
        made = price_inputs(part.race_id, part.p_base, day.price_1100, day.price_0800)
        if not values[KEYS].reset_index(drop=True).equals(part[KEYS].reset_index(drop=True)):
            raise ValueError("Card inputs are not in card order")
        return pd.concat([values.reset_index(drop=True), made], axis=1), [*columns, *PRICE_INPUTS]
    return extra


