from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

import config
from src.utils import (
    normalise_horse_key as _normalise_horse_key,
    normalise_off_time_key as _normalise_off_time_key,
    normalise_track_key as _normalise_track_key,
)


PAPER_TRADES_DIR = os.path.join(config.DATA_DIR, "paper_trades")
PAPER_TRADES_CSV = os.path.join(PAPER_TRADES_DIR, "paper_trades.csv")


def _ensure_store() -> None:
    os.makedirs(PAPER_TRADES_DIR, exist_ok=True)


def load_paper_trades() -> pd.DataFrame:
    _ensure_store()
    if not os.path.exists(PAPER_TRADES_CSV):
        return pd.DataFrame()
    df = pd.read_csv(PAPER_TRADES_CSV)
    return df


def save_paper_trades(df: pd.DataFrame) -> None:
    _ensure_store()
    df.to_csv(PAPER_TRADES_CSV, index=False)


def append_paper_trades(new_rows: pd.DataFrame) -> int:
    if new_rows is None or new_rows.empty:
        return 0
    existing = load_paper_trades()
    combined = pd.concat([existing, new_rows], ignore_index=True, sort=False) if not existing.empty else new_rows.copy()
    save_paper_trades(combined)
    return int(len(new_rows))


def build_paper_trades_from_signals(
    signal_df: pd.DataFrame,
    *,
    event: dict,
    stake: float = 1.0,
    source: str = "fake_model",
    log_backs: bool = True,
    log_lays: bool = True,
) -> pd.DataFrame:
    if signal_df is None or signal_df.empty:
        return pd.DataFrame()

    side_mask = pd.Series(False, index=signal_df.index)
    signal_labels = signal_df["signal"].fillna("").astype(str)
    if log_backs:
        side_mask |= signal_labels.str.contains("BACK", regex=False)
    if log_lays:
        side_mask |= signal_labels.str.contains("LAY", regex=False)
    chosen = signal_df.loc[side_mask].copy()
    if chosen.empty:
        return pd.DataFrame()

    event_name = str(event.get("name") or "")
    race_date = str(event.get("start") or "")[:10]
    off_time = event_name.split(" ", 1)[0] if " " in event_name else ""
    track = event_name.split(" ", 1)[1] if " " in event_name else event_name
    created_at = datetime.now().isoformat(timespec="seconds")

    rows: list[dict] = []
    for _, row in chosen.iterrows():
        label = str(row.get("signal") or "")
        sides: list[str] = []
        if log_backs and "BACK" in label:
            sides.append("BACK")
        if log_lays and "LAY" in label:
            sides.append("LAY")
        for side in sides:
            entry_odds = float(row["best_back_odds"] if side == "BACK" else row["best_lay_odds"])
            available = float(row["best_back_available"] if side == "BACK" else row["best_lay_available"])
            rows.append({
                "logged_at": created_at,
                "trade_id": f"{created_at}|{event.get('id')}|{row['horse_name']}|{side}",
                "status": "OPEN",
                "source": source,
                "event_id": event.get("id"),
                "event_name": event_name,
                "race_date": race_date,
                "track": track,
                "off_time": off_time,
                "horse_name": row["horse_name"],
                "side": side,
                "stake": float(stake),
                "entry_odds": entry_odds,
                "entry_available": available,
                "win_probability": float(row["win_probability"]),
                "fair_odds": float(row["fair_odds"]),
                "back_edge_pct": float(row.get("back_edge_pct", 0.0) or 0.0),
                "lay_edge_pct": float(row.get("lay_edge_pct", 0.0) or 0.0),
                "spread_pct": float(row.get("spread_pct", 0.0) or 0.0),
                "result_finish_position": None,
                "result_won": None,
                "pnl": None,
                "_trk": _normalise_track_key(track),
                "_ot": _normalise_off_time_key(off_time),
                "_hn": _normalise_horse_key(row["horse_name"]),
            })
    return pd.DataFrame(rows)


def settle_paper_trades(trades_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty or results_df is None or results_df.empty:
        return trades_df.copy() if isinstance(trades_df, pd.DataFrame) else pd.DataFrame()

    out = trades_df.copy()
    open_mask = out["status"].fillna("OPEN") == "OPEN"
    if not open_mask.any():
        return out

    results = results_df.copy()
    results["race_date"] = pd.to_datetime(results["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    results["_trk"] = results["track"].map(_normalise_track_key)
    results["_ot"] = results["off_time"].map(_normalise_off_time_key)
    results["_hn"] = results["horse_name"].map(_normalise_horse_key)
    keep_cols = [c for c in ["race_date", "_trk", "_ot", "_hn", "finish_position", "won"] if c in results.columns]
    results = results[keep_cols].drop_duplicates(["race_date", "_trk", "_ot", "_hn"], keep="last")

    merged = out.loc[open_mask].merge(
        results,
        on=["race_date", "_trk", "_ot", "_hn"],
        how="left",
        suffixes=("", "_result"),
    )
    if merged.empty:
        return out

    settled_mask = merged["finish_position"].notna()
    if not settled_mask.any():
        return out

    merged.loc[settled_mask, "result_finish_position"] = pd.to_numeric(merged.loc[settled_mask, "finish_position"], errors="coerce")
    merged.loc[settled_mask, "result_won"] = pd.to_numeric(merged.loc[settled_mask, "won"], errors="coerce").fillna(0).astype(int)
    back_mask = settled_mask & (merged["side"] == "BACK")
    lay_mask = settled_mask & (merged["side"] == "LAY")
    merged.loc[back_mask, "pnl"] = merged.loc[back_mask].apply(
        lambda row: row["stake"] * (row["entry_odds"] - 1.0) if int(row["result_won"]) == 1 else -row["stake"],
        axis=1,
    )
    merged.loc[lay_mask, "pnl"] = merged.loc[lay_mask].apply(
        lambda row: -row["stake"] * (row["entry_odds"] - 1.0) if int(row["result_won"]) == 1 else row["stake"],
        axis=1,
    )
    merged.loc[settled_mask, "status"] = "SETTLED"
    merged.loc[settled_mask, "settled_at"] = datetime.now().isoformat(timespec="seconds")

    updates = merged.set_index("trade_id")
    out = out.set_index("trade_id")
    for col in ["status", "result_finish_position", "result_won", "pnl", "settled_at"]:
        if col in updates.columns:
            out.loc[updates.index, col] = updates[col]
    return out.reset_index()