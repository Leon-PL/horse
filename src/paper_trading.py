"""
Place-market paper trading
==========================
Forward (paper) validation of the validated place-market edge: the form +
pre-off win-market place model, +EV-selected against the live PLACE price, logged
for later settlement. Built for **Matchbook** live odds (Betfair API is paid);
per the operator, MB↔Betfair price differences are arbitraged away so MB prices
stand in for the bettable price. MB liquidity is thin, so selections are placed
*manually* on Betfair — this module's job is to produce and track the daily
selection list, capturing the live price AND available size at decision time.

Two phases:
  * ``score_and_select`` — join the day's form-featured runners to live MB
    win+place prices, build the win-market features, score P(place), and pick
    the +EV bets (EV threshold; the validated ROI↔volume / liquidity knob).
  * ``settle`` — once results are in, mark placed (finish <= places paid) and
    book P&L at the *logged* price, maintaining a running ledger.

The selection / settlement maths is the single source of truth and is pure +
unit-tested; the live MB fetch and feature build wrap it.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import config
from src.utils import normalise_horse_key, normalise_track_key, normalise_off_time_key
from src.bet_settlement import settle_place_exchange_bets

# The win-market features the place model was trained with (see
# scripts/train_place_paper_model.py). Built identically at train (BF pre-off
# price) and inference (live MB win price) via build_market_features.
MARKET_FEATS = ["win_ltp300", "mkt_open", "mkt_win_prob", "mkt_log300", "mkt_openlog", "mkt_oddsrank"]

DEFAULT_LEDGER = os.path.join(config.PROCESSED_DATA_DIR, "paper_trades.csv")
SELECTION_COLS = [
    "logged_at", "race_date", "track", "off_time", "horse_name", "p_place",
    "mb_win_odds", "mb_place_odds", "mb_place_available", "places_paid",
    "ev", "stake", "settled", "finish_position", "placed", "pnl",
]


def build_market_features(df: pd.DataFrame, win_col: str, open_col: str | None = None) -> pd.DataFrame:
    """Win-market features from a win-odds column, per race (``race_id``).

    Identical recipe at train time (BF pre-off ``ltp_300s`` / ``opening_odds``)
    and inference (live MB win price). When *open_col* is missing the win price
    doubles as the opening proxy (MB↔BF mismatch is treated as immaterial).
    """
    o = pd.to_numeric(df[win_col], errors="coerce")
    op = pd.to_numeric(df[open_col], errors="coerce") if (open_col and open_col in df.columns) else o
    grp = df["race_id"]
    ip = 1.0 / o
    out = pd.DataFrame(index=df.index)
    out["win_ltp300"] = o
    out["mkt_open"] = op
    out["mkt_win_prob"] = ip / ip.groupby(grp).transform("sum")
    out["mkt_log300"] = np.log(o.clip(lower=1.01))
    out["mkt_openlog"] = np.log(op.clip(lower=1.01))
    out["mkt_oddsrank"] = o.groupby(grp).rank()
    return out


def _merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "track_key": df["track"].map(normalise_track_key),
        "off_key": df["off_time"].map(normalise_off_time_key),
        "horse_key": df["horse_name"].map(normalise_horse_key),
    })


def score_and_select(
    featured: pd.DataFrame,
    mb_prices: pd.DataFrame,
    model,
    feature_cols: list[str],
    *,
    ev_threshold: float = 0.10,
    stake: float = 1.0,
    jumps_only: bool = False,
) -> pd.DataFrame:
    """Join form-featured runners to live MB prices, score P(place), pick +EV bets.

    *featured*  – the day's runners with the model's form features + race meta
                  (track, off_time, horse_name, race_date, num_runners, race_type).
    *mb_prices* – per-runner live MB prices (track, off_time, horse_name,
                  mb_win_odds, mb_place_odds, mb_place_available, places_paid).
    *model*     – fitted classifier with ``predict_proba``; *feature_cols* is the
                  exact training column order (form features + MARKET_FEATS).

    Returns the selected bets (EV > threshold), at flat *stake*, ready to log.
    """
    feat = featured.copy()
    mb = mb_prices.copy()
    fk, mk = _merge_keys(feat), _merge_keys(mb)
    feat = pd.concat([feat.reset_index(drop=True), fk.reset_index(drop=True)], axis=1)
    mb = pd.concat([mb.reset_index(drop=True), mk.reset_index(drop=True)], axis=1)
    cols = ["track_key", "off_key", "horse_key", "mb_win_odds", "mb_place_odds",
            "mb_place_available", "places_paid"]
    j = feat.merge(mb[cols].drop_duplicates(["track_key", "off_key", "horse_key"]),
                   on=["track_key", "off_key", "horse_key"], how="inner")
    if j.empty:
        return pd.DataFrame(columns=SELECTION_COLS)

    # a race_id for per-race normalisation of the win-market features
    j["race_id"] = j["track_key"] + "|" + j["off_key"]
    mfeat = build_market_features(j, "mb_win_odds")
    for c in MARKET_FEATS:
        j[c] = mfeat[c].to_numpy()

    # Every model feature must be genuinely present — live features are never
    # guessed or imputed at scoring time. The lookahead cache is built via
    # FeatureStore.featurize_live (correct ratings + completed one-hot schema), so
    # a missing column here is a real FE/serving bug, not a benign absence. Fail
    # loudly rather than silently 0-fill.
    missing = [c for c in feature_cols if c not in j.columns]
    if missing:
        raise KeyError(
            f"score_and_select: {len(missing)}/{len(feature_cols)} model features "
            f"absent from the live featurised card — rebuild the lookahead cache "
            f"(scripts/build_lookahead.py). Missing: {missing}")
    X = j[feature_cols]
    j["p_place"] = model.predict_proba(X)[:, 1]
    j["mb_place_odds"] = pd.to_numeric(j["mb_place_odds"], errors="coerce")
    j["ev"] = j["p_place"] * j["mb_place_odds"] - 1.0

    sel = j[(j["ev"] > ev_threshold) & j["mb_place_odds"].between(1.01, 1000)].copy()
    if jumps_only and "race_type" in sel.columns:
        sel = sel[sel["race_type"].astype(str).str.contains("Hurdle|Chase", case=False, na=False)]
    sel["logged_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    sel["stake"] = float(stake)
    for c in ("settled", "placed"):
        sel[c] = 0
    sel["finish_position"] = np.nan
    sel["pnl"] = np.nan
    keep = [c for c in SELECTION_COLS if c in sel.columns]
    return sel[keep].sort_values(["off_time", "ev"], ascending=[True, False]).reset_index(drop=True)


def log_selections(selections: pd.DataFrame, ledger_path: str = DEFAULT_LEDGER) -> int:
    """Append selections to the ledger (de-duped on race/horse/date). Returns rows added."""
    if selections is None or selections.empty:
        return 0
    new = selections.copy()
    key = ["race_date", "track", "off_time", "horse_name"]
    if os.path.exists(ledger_path):
        old = pd.read_csv(ledger_path)
        combined = pd.concat([old, new], ignore_index=True)
        before = len(old)
        combined = combined.drop_duplicates(subset=key, keep="first")
        added = len(combined) - before
    else:
        combined = new.drop_duplicates(subset=key, keep="first")
        added = len(combined)
    combined.to_csv(ledger_path, index=False)
    return int(added)


def settle(results: pd.DataFrame, ledger_path: str = DEFAULT_LEDGER, commission: float = 0.02) -> dict:
    """Settle un-settled ledger rows against *results* and book P&L at the logged
    MB place odds. *results* needs race_date, track, off_time/horse_name,
    finish_position. Place leg pays (odds-1)*(1-commission) if placed, else -stake.
    Returns a summary dict and rewrites the ledger.
    """
    if not os.path.exists(ledger_path):
        return {"error": "no ledger"}
    led = pd.read_csv(ledger_path)
    r = results.copy()
    for f in (led, r):
        kk = _merge_keys(f)
        f["track_key"], f["off_key"], f["horse_key"] = kk["track_key"], kk["off_key"], kk["horse_key"]
        f["race_date"] = pd.to_datetime(f["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    rr = r[["race_date", "track_key", "off_key", "horse_key", "finish_position"]].drop_duplicates(
        ["race_date", "track_key", "off_key", "horse_key"])
    m = led.merge(rr, on=["race_date", "track_key", "off_key", "horse_key"], how="left", suffixes=("", "_res"))
    fp = pd.to_numeric(m["finish_position_res"], errors="coerce")
    has = fp.notna() & (m["settled"].fillna(0).astype(int) == 0)
    placed = (fp > 0) & (fp <= pd.to_numeric(m["places_paid"], errors="coerce"))
    odds = pd.to_numeric(m["mb_place_odds"], errors="coerce")
    stake = pd.to_numeric(m["stake"], errors="coerce").fillna(1.0)
    pnl = settle_place_exchange_bets(odds, placed, stake=stake, commission=commission)
    m.loc[has, "finish_position"] = fp[has]
    m.loc[has, "placed"] = placed[has].astype(int)
    m.loc[has, "pnl"] = pnl[has]
    m.loc[has, "settled"] = 1
    m = m.drop(columns=[c for c in m.columns if c.endswith("_res") or c in ("track_key", "off_key", "horse_key")])
    m.to_csv(ledger_path, index=False)
    s = m[m["settled"].fillna(0).astype(int) == 1]
    n = len(s); pnl_tot = float(pd.to_numeric(s["pnl"], errors="coerce").sum())
    staked = float(pd.to_numeric(s["stake"], errors="coerce").sum())
    return {
        "newly_settled": int(has.sum()), "total_settled": n,
        "placed": int(pd.to_numeric(s["placed"], errors="coerce").sum()),
        "pnl": round(pnl_tot, 2), "staked": round(staked, 2),
        "roi_pct": round(100 * pnl_tot / staked, 2) if staked else 0.0,
    }
