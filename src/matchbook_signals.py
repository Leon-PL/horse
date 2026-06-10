from __future__ import annotations

import hashlib
import re
from typing import Any

import pandas as pd

from src.utils import (
    normalise_horse_key as _normalise_horse_key,
    normalise_off_time_key as _normalise_off_time_key,
    normalise_track_key as _normalise_track_key,
)


def event_date_str(event: dict[str, Any]) -> str:
    start = str(event.get("start") or "")
    return start[:10] if len(start) >= 10 else ""


def event_track_and_off_time(event: dict[str, Any]) -> tuple[str, str]:
    name = str(event.get("name") or "").strip()
    match = re.match(r"^(\d{1,2}:\d{2})\s+(.+)$", name)
    if match:
        return match.group(2).strip(), match.group(1)
    start = str(event.get("start") or "")
    off_time = start[11:16] if len(start) >= 16 else ""
    return name, off_time


def _best_price(prices: list[dict[str, Any]], sides: set[str]) -> tuple[float | None, float | None]:
    best_odds = None
    best_amount = None
    for price in prices or []:
        side = str(price.get("side") or "").lower()
        if side not in sides:
            continue
        odds = price.get("odds")
        amount = price.get("available-amount")
        if odds is None:
            continue
        odds_val = float(odds)
        if best_odds is None:
            best_odds = odds_val
            best_amount = float(amount) if amount is not None else None
            continue
        if "back" in sides or "win" in sides:
            if odds_val > best_odds:
                best_odds = odds_val
                best_amount = float(amount) if amount is not None else None
        else:
            if odds_val < best_odds:
                best_odds = odds_val
                best_amount = float(amount) if amount is not None else None
    return best_odds, best_amount


def build_fake_prediction_frame(
    event: dict[str, Any],
    markets: list[dict[str, Any]],
) -> pd.DataFrame:
    track, off_time = event_track_and_off_time(event)
    race_date = event_date_str(event)
    race_name = str(event.get("name") or "")

    win_market = next(
        (
            market
            for market in markets or []
            if str(market.get("name") or "").upper() == "WIN"
        ),
        None,
    )
    if not isinstance(win_market, dict):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for runner in win_market.get("runners") or []:
        prices = runner.get("prices") or []
        best_back, _ = _best_price(prices, {"back", "win"})
        best_lay, _ = _best_price(prices, {"lay", "lose"})
        anchor_odds = best_back or best_lay
        if anchor_odds is None or float(anchor_odds) <= 1.0:
            base_prob = 0.01
        else:
            base_prob = 1.0 / float(anchor_odds)

        # Deterministic name-based tilt so the fake output is stable but not
        # identical to raw market probabilities.
        seed = int(hashlib.sha1(str(runner.get("name") or "").encode("utf-8")).hexdigest()[:8], 16)
        tilt = 1.0 + (((seed % 2001) - 1000) / 20000.0)
        rows.append({
            "track": track,
            "off_time": off_time,
            "race_date": race_date,
            "race_name": race_name,
            "horse_name": runner.get("name"),
            "win_probability_raw": base_prob * tilt,
        })

    fake_df = pd.DataFrame(rows)
    if fake_df.empty:
        return fake_df

    total = float(fake_df["win_probability_raw"].sum())
    if total <= 0:
        fake_df["win_probability"] = 1.0 / len(fake_df)
    else:
        fake_df["win_probability"] = fake_df["win_probability_raw"] / total
    fake_df["predicted_rank"] = fake_df["win_probability"].rank(ascending=False, method="min").astype(int)
    return fake_df.drop(columns=["win_probability_raw"]).sort_values(["predicted_rank", "horse_name"], kind="stable").reset_index(drop=True)


def build_signal_frame(
    predictions: pd.DataFrame,
    event: dict[str, Any],
    markets: list[dict[str, Any]],
    *,
    min_back_edge: float,
    min_lay_edge: float,
    min_liquidity: float,
) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()

    track, off_time = event_track_and_off_time(event)
    event_track_key = _normalise_track_key(track)
    event_off_time_key = _normalise_off_time_key(off_time)

    race_preds = predictions.copy()
    race_preds["_trk"] = race_preds.get("track", "").map(_normalise_track_key)
    race_preds["_ot"] = race_preds.get("off_time", "").map(_normalise_off_time_key)
    race_preds["_hn"] = race_preds["horse_name"].map(_normalise_horse_key)
    race_preds = race_preds[(race_preds["_trk"] == event_track_key) & (race_preds["_ot"] == event_off_time_key)].copy()
    if race_preds.empty:
        return pd.DataFrame()

    win_market = next(
        (
            market
            for market in markets or []
            if str(market.get("name") or "").upper() == "WIN"
        ),
        None,
    )
    if not isinstance(win_market, dict):
        return pd.DataFrame()

    runner_rows: list[dict[str, Any]] = []
    for runner in win_market.get("runners") or []:
        prices = runner.get("prices") or []
        best_back, back_available = _best_price(prices, {"back", "win"})
        best_lay, lay_available = _best_price(prices, {"lay", "lose"})
        runner_rows.append({
            "horse_name": runner.get("name"),
            "_hn": _normalise_horse_key(runner.get("name")),
            "best_back_odds": best_back,
            "best_back_available": back_available,
            "best_lay_odds": best_lay,
            "best_lay_available": lay_available,
            "matchbook_status": runner.get("status"),
        })

    market_df = pd.DataFrame(runner_rows)
    if market_df.empty:
        return pd.DataFrame()

    merged = race_preds.merge(
        market_df,
        on="_hn",
        how="left",
        suffixes=("", "_matchbook"),
    )
    merged["fair_odds"] = 1.0 / merged["win_probability"].clip(lower=1e-9)
    merged["implied_back_prob"] = 1.0 / pd.to_numeric(merged["best_back_odds"], errors="coerce")
    merged["implied_lay_prob"] = 1.0 / pd.to_numeric(merged["best_lay_odds"], errors="coerce")
    merged["back_edge_pct"] = merged["win_probability"] - merged["implied_back_prob"]
    merged["lay_edge_pct"] = merged["implied_lay_prob"] - merged["win_probability"]
    merged["spread_pct"] = (
        pd.to_numeric(merged["best_lay_odds"], errors="coerce")
        / pd.to_numeric(merged["best_back_odds"], errors="coerce")
        - 1.0
    )
    merged["back_candidate"] = (
        merged["back_edge_pct"].fillna(-1.0) >= float(min_back_edge)
    ) & (
        pd.to_numeric(merged["best_back_available"], errors="coerce").fillna(0.0) >= float(min_liquidity)
    )
    merged["lay_candidate"] = (
        merged["lay_edge_pct"].fillna(-1.0) >= float(min_lay_edge)
    ) & (
        pd.to_numeric(merged["best_lay_available"], errors="coerce").fillna(0.0) >= float(min_liquidity)
    )
    merged["signal"] = ""
    merged.loc[merged["back_candidate"], "signal"] = "BACK"
    merged.loc[merged["lay_candidate"], "signal"] = merged.loc[merged["lay_candidate"], "signal"].replace("", "LAY")
    merged.loc[merged["back_candidate"] & merged["lay_candidate"], "signal"] = "BACK / LAY"

    columns = [
        "predicted_rank",
        "horse_name",
        "win_probability",
        "fair_odds",
        "best_back_odds",
        "best_back_available",
        "best_lay_odds",
        "best_lay_available",
        "back_edge_pct",
        "lay_edge_pct",
        "spread_pct",
        "signal",
        "matchbook_status",
    ]
    present_cols = [col for col in columns if col in merged.columns]
    return merged[present_cols].sort_values(["predicted_rank", "horse_name"], kind="stable").reset_index(drop=True)