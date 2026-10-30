"""Research-only transfer and recency of strictly earlier market assessments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.established_form_features import _horse_keys, mark_scale
from src.prior_market_research import past_market_observations
from src.result_status import NONSTARTER_LABELS, result_labels

FEATURES = (
    "horse_market_all_scale_logrel_3_vs_field",
    "horse_market_recent_price_change_vs_field",
    "horse_market_all_scale_support_3",
    "horse_market_scale_changed",
    "horse_market_family_changed",
)


def market_transfer_features(frame, *, state=None, return_state=False):
    """Keep the last three actual starts, across scales, for at most365 days.

    Prices express relative opinion within their original field, not pounds or
    absolute ability. Explicit scale/family flags allow a learner to distinguish
    transfer from continuity. No official ratings are pooled or converted.
    A missing book advances the window as unknown. The trend requires a priced
    latest start and at least one earlier priced start; it never skips an
    unpriced latest start and pretends an old trend is current. Current-day
    prices/outcomes are excluded, and input state is not mutated.
    """
    dates = pd.to_datetime(frame.race_date, errors="raise").dt.normalize()
    state = state or {}
    if dates.isna().any() or (len(frame) and state.get("last_date")
                            and dates.min() <= pd.Timestamp(state["last_date"])):
        raise ValueError("Transfer seed must end on an earlier complete date")
    days = dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    identities = _horse_keys(frame)
    scales = mark_scale(frame).to_numpy(dtype=object, na_value=None)
    # Shared complete-book validation; the unused opposition calculation must
    # not require an observed mark for this price-only proposal.
    work = frame if "official_rating" in frame else frame.assign(official_rating=np.nan)
    opinions = past_market_observations(work).logrel.to_numpy()
    finish = pd.to_numeric(frame.finish_position, errors="coerce").to_numpy(float)
    labels = result_labels(frame)
    starts = (finish > 0) | (labels.ne("") & ~labels.isin(NONSTARTER_LABELS)).to_numpy()
    known = np.array([key is not None and scale is not None
                      for key, scale in zip(identities, scales)], dtype=bool)
    actual = pd.DataFrame({"identity": identities, "day": days})
    if actual.loc[starts & known].duplicated(["identity", "day"]).any():
        raise ValueError("Multiple actual starts for one identity/day make latest price ambiguous")

    output = {column: np.full(len(frame), np.nan) for column in FEATURES}
    history = {key: list(values) for key, values in state.get("history", {}).items()}
    order = np.lexsort((frame.race_id.astype(str).to_numpy(), days))
    boundaries = np.r_[0, np.flatnonzero(np.diff(days[order])) + 1, len(frame)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        rows = order[lo:hi]
        for i in rows:
            if not known[i]:
                continue
            recent = [entry for entry in history.get(identities[i], [])
                      if days[i] - entry[0] <= 365]
            prices = [price for _, _, price in recent if np.isfinite(price)]
            output[FEATURES[2]][i] = len(prices)
            if prices:
                output[FEATURES[0]][i] = sum(prices) / len(prices)
            if recent:
                _, previous_scale, latest = recent[-1]
                output[FEATURES[3]][i] = float(previous_scale != scales[i])
                output[FEATURES[4]][i] = float(
                    previous_scale.startswith("Flat|") != scales[i].startswith("Flat|"))
                earlier = [price for _, _, price in recent[:-1] if np.isfinite(price)]
                if np.isfinite(latest) and earlier:
                    output[FEATURES[1]][i] = latest - sum(earlier) / len(earlier)
        for i in rows:
            if starts[i] and known[i]:
                entry = (int(days[i]), scales[i], float(opinions[i]))
                history[identities[i]] = (history.get(identities[i], []) + [entry])[-3:]

    result = pd.DataFrame(output, index=frame.index)
    for column in FEATURES[:2]:
        result[column] -= result[column].groupby(frame.race_id, sort=False).transform("mean")
    if return_state:
        return result, {"last_date": str(dates.max().date()) if len(frame) else state.get("last_date"),
                        "history": history}
    return result
