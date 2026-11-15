"""Corrected forecasts for an unsettled card.

The card's features come from the existing live path
(``src.live_prediction.features_for_live_cards``). The win model forecasts
it; the card is then placed after the settled history so its records are read
from strictly earlier dates, and the correction adjusts the forecast.
"""
import numpy as np
import pandas as pd

from src.correction import data, history
from src.correction import inputs as ci
from src.correction import model as cm

KEYS = ci.KEYS
#: Records older than this against the card make the correction's inputs stale without any error.
MAX_HISTORY_LAG_DAYS = 7


def load_correction(path):
    import joblib
    contents = joblib.load(path)
    if contents.get("format") != "forecast_correction" or contents.get("version") != 1:
        raise ValueError("Unsupported forecast correction artifact")
    return contents["model"]


def win_member(candidate):
    models = getattr(candidate, "models", None)
    if not models or len(models) != 1:
        raise ValueError("The forecast history follows one win model; this run is not a single-model candidate")
    return next(iter(models.values()))


def card_rows(featured, forecast, columns):
    """The card as history-frame rows: no result, no price, whatever the card's placeholder columns say."""
    missing = [c for c in columns if c not in featured]
    if missing:
        raise ValueError(f"The card lacks columns the correction reads: {missing[:8]}")
    rows = featured[list(columns)].copy()
    rows["p_base"] = forecast
    for column in ("won", "finish_position", "p_market", "bsp", "odds"):
        rows[column] = np.nan
    rows["race_date"] = pd.to_datetime(rows.race_date).dt.normalize()
    return rows.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)


def correct_card(settled, prices, card, model, tracking, extra=None):
    """Corrected probabilities for ``card`` rows placed after the ``settled`` history frame.

    The settled rows are history only; the wide input table is built for the card's rows alone.
    ``extra(card_rows, values, columns)`` may append inputs known only on the day (returns values, columns).
    """
    if card.race_date.min() <= settled.race_date.max():
        raise ValueError("The card must be dated after every settled race in the history frame")
    if set(card.race_id.astype(str)) & set(settled.race_id.astype(str)):
        raise ValueError("A card race is already in the settled history")
    card = card.astype({"race_id": settled.race_id.dtype, "horse_id": settled.horse_id.dtype})
    shared = [c for c in card.columns if c in settled or c in tracking]
    frame = pd.concat([settled[[c for c in shared if c in settled]], card[shared]], ignore_index=True)
    empty = pd.DataFrame(np.nan, index=range(len(card)), columns=prices.columns)
    empty = empty.assign(**{k: card[k].to_numpy() for k in KEYS})
    all_prices = pd.concat([prices, empty[prices.columns]], ignore_index=True)
    rows = np.r_[np.zeros(len(settled), bool), np.ones(len(card), bool)]
    values, columns = cm.build_inputs(frame, prices=all_prices, tracking=tracking, only=rows)
    part = frame.loc[rows].reset_index(drop=True)
    if extra is not None:
        values, columns = extra(part, values, columns)
    if columns != model.columns:
        raise ValueError("The correction was fitted on different inputs than this store and history provide")
    out = part[[*KEYS, "race_date", "p_base"]].copy()
    out["adjustment"] = model.adjustment(values)
    out["probability"] = cm.corrected(out.p_base.to_numpy(float), out.adjustment.to_numpy(), part.race_id.to_numpy())
    return out


def forecast(featured, candidate, model, *, store_root=None, extra=None):
    """Win-model and corrected forecasts for a featurised card (one date)."""
    member = win_member(candidate)
    held = history.read()
    if held.empty or set(held.recipe_signature) != {member.spec.signature}:
        raise ValueError("The forecast history was made by a different recipe than this win model")
    card_date = pd.to_datetime(featured.race_date).dt.normalize()
    lag = (card_date.min() - held.race_date.max()).days
    if lag > MAX_HISTORY_LAG_DAYS:
        raise ValueError(f"The forecast history ends {lag} days before the card; run `python -m src.correction extend`")
    base = candidate.predict(featured)
    settled, prices, store_last = data.settled_frame(store_root, forecasts=held)
    tracking = [c for c in data.tracking_columns() if c in featured]
    columns = list(dict.fromkeys([*KEYS, "race_date", "horse_name", *ci.ENTITY_COLUMNS, *data.CARD_COLUMNS, *tracking]))
    card = card_rows(featured, base, columns)
    out = correct_card(settled, prices, card, model, tracking, extra=extra)
    names = card[[*KEYS, "horse_name"]].astype({"race_id": out.race_id.dtype, "horse_id": out.horse_id.dtype})
    out = out.merge(names, on=KEYS, validate="one_to_one")
    return out, {"history_last_date": str(held.race_date.max().date()), "store_last_date": store_last,
                 "correction_as_of": model.lineage.get("as_of"), "win_model_as_of": candidate.lineage.get("as_of")}
