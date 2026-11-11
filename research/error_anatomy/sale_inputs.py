"""Public auction prices as model inputs.

Sale price was dropped from the project because the market fully prices it
(ledger: residual correlation 0.0000 against the market). That is a betting
verdict. The model does not have the input at all, and it is weakest on
first-time and lightly raced runners, whose price is most of what is publicly
known about them. Judged by model accuracy it is an open question.

Sales are matched to horses by sire, dam and foaling year, and only sales held
before the horse could have raced are used: foal, yearling and breeze-up sales
no later than the year the horse turned two, and never a sale later than the
race. Horses-in-training sales are excluded, because their prices reflect
racecourse form after the fact.

Inputs per runner: its own log price and that price relative to its sale and
year; the sire's median log price at earlier years' sales (available for every
runner by that sire, bought at auction or not); and each relative to the field.
"""
import numpy as np
import pandas as pd

from research.error_anatomy.entity_residuals import KEYS

SALES = "data/processed/tattersalls_sales.parquet"
PRE_RACING = {"DEF": 0, "DEY": 1, "OC1": 1, "OC2": 1, "OC3": 1, "OC4": 1, "SOM": 1, "AUY": 1,
              "GBU": 2, "CBU": 2}  # sale code -> age of the horse in the sale year
BREEZE_UP_READY = "-06-01"  # breeze-up sales are held in April and May


def _name(values):
    return values.astype(str).str.lower().str.replace(r"\s*\([a-z]{2,3}\)\s*$", "", regex=True).str.replace(
        r"[^a-z0-9]", "", regex=True)


def sale_inputs(history, sales=None):
    """``history`` needs race_id, horse_id, race_date, sire, dam and foaled."""
    sales = pd.read_parquet(SALES) if sales is None else sales
    sales = sales.loc[sales.sale.isin(PRE_RACING) & sales.price_guineas.gt(0)].copy()
    sales["age_at_sale"] = sales.sale.map(PRE_RACING)
    sales["foal_year"] = sales.sale_year - sales.age_at_sale
    sales["log_price"] = np.log(sales.price_guineas)
    sales["relative"] = sales.log_price - sales.groupby(["sale", "sale_year"]).log_price.transform("median")
    sales["_sire"], sales["_dam"] = _name(sales.sire), _name(sales.dam)

    h = history[[*KEYS, "race_date", "sire", "dam", "foaled"]].copy()
    h["_sire"], h["_dam"] = _name(h.sire), _name(h.dam)
    foaled = pd.to_datetime(h.foaled, errors="coerce", dayfirst=True)
    h["foal_year"] = foaled.dt.year
    # A horse sold more than once before racing keeps its latest pre-racing sale (the freshest opinion).
    own = sales.sort_values("sale_year").drop_duplicates(["_sire", "_dam", "foal_year"], keep="last")
    h = h.merge(own[["_sire", "_dam", "foal_year", "sale", "sale_year", "log_price", "relative"]],
                on=["_sire", "_dam", "foal_year"], how="left", validate="many_to_one")
    available = np.where(h.sale.isin(["GBU", "CBU"]),
                         pd.to_datetime(h.sale_year.astype("Int64").astype(str) + BREEZE_UP_READY, errors="coerce"),
                         pd.to_datetime((h.sale_year + 1).astype("Int64").astype(str) + "-01-01", errors="coerce"))
    usable = h.log_price.notna() & (h.race_date >= pd.Series(available, index=h.index))
    out = h[KEYS].copy()
    out["sale_log_price"] = h.log_price.where(usable).astype("float32")
    out["sale_relative"] = h.relative.where(usable).astype("float32")

    # Sire's median yearling-sale log price over sale years strictly before the race year.
    yearly = sales.groupby(["_sire", "sale_year"]).agg(total=("relative", "sum"), lots=("relative", "size")).reset_index()
    yearly = yearly.sort_values("sale_year")
    yearly[["total", "lots"]] = yearly.groupby("_sire")[["total", "lots"]].cumsum()
    look = pd.DataFrame({"_sire": h._sire, "sale_year": h.race_date.dt.year.astype("int64")}).reset_index()
    joined = pd.merge_asof(look.sort_values("sale_year"), yearly, on="sale_year", by="_sire",
                           allow_exact_matches=False).set_index("index").reindex(h.index)
    out["sire_sale_relative"] = (joined.total / (joined.lots + 5.)).astype("float32")  # shrunk mean
    out["sire_sale_lots"] = joined.lots.fillna(0).astype("float32")
    race = history.race_id
    for column in ("sale_relative", "sire_sale_relative"):
        out[f"{column}_vs_field"] = (out[column] - out[column].groupby(race).transform("mean")).astype("float32")
    out["sale_priced_share_of_field"] = out.sale_relative.notna().groupby(race).transform("mean").astype("float32")
    return out


def load(frame):
    """Candidate loader in the cheap-screen convention: (values, columns) for the frame's races."""
    from research.meta_learner import screen_candidates as sc
    history = sc.read_store(["sire", "dam", "foaled"])
    values = sale_inputs(history)
    values = values.loc[values.race_id.isin(frame.race_id.unique())]
    return values, [c for c in values if c not in KEYS]
