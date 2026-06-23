"""
Place-market strategy — the validated +EV place model
======================================================
Single source of truth for the project's one statistically-positive edge: an
odds-free form place model (optionally **+ pre-off win-market features**) bet
+EV against the EXCHANGE place price (Betfair place BSP), settled at the
operator's commission. Walk-forward validated at +9.31% (form-only) / +15.24%
(form + win-market) ROI @ place BSP, 2% commission, 12/12 folds (see the
place-market-edge notes).

Used by:
  * scripts/train_place_paper_model.py — the paper-trading artifact, and
  * src/training.py (train_run) — as an opt-in strategy ALONGSIDE the existing
    win/each-way pipeline.

Three pieces, all here so train-time and live agree:
  * ``prepare_place_data``      — build X (form [+ market]), the placed target,
                                  and attach the exchange settlement columns.
  * ``train_place_market_model``— fit the LGBM place classifier.
  * ``evaluate_place_market``   — +EV selection @ place BSP, ROI + bootstrap CI.
``train_and_evaluate`` wires them: held-out-by-date eval + a final model on all
matched rows for the artifact.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

import config
from src.betfair import attach_bsp, bf_track_key, bf_horse_key
from src.utils import normalise_off_time_key
from src.paper_trading import build_market_features, MARKET_FEATS
from src.bet_settlement import settle_place_exchange_bets, PLACE_EXCHANGE_COMMISSION

logger = logging.getLogger(__name__)

PLACE_BSP_PATH = os.path.join(config.PROCESSED_DATA_DIR, "betfair_place_bsp.parquet")

# Same LGBM config the edge was validated with (scripts/model_lab.BASE_PARAMS).
PLACE_PARAMS = dict(
    n_estimators=2000, num_leaves=35, max_depth=6, learning_rate=0.01,
    min_child_samples=50, subsample=0.6, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=3.0, objective="binary",
    subsample_freq=1, n_jobs=-1, verbose=-1,
)

DEFAULT_EV_THRESHOLD = 0.10


def _d10(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.strftime("%Y-%m-%d")


def prepare_place_data(featured: pd.DataFrame, *, use_market_features: bool = True):
    """Attach the place target + exchange settlement columns (+ win-market
    features) to *featured*. Returns ``(df, feature_cols, form_cols)``.

    Adds: ``bf_places_paid`` (Betfair numberOfWinners — the placed cutoff AND
    settlement count), ``place_bsp``/``placed_bf`` (exchange settlement), and the
    six MARKET_FEATS (when requested), built from the BF pre-off win price
    (``ltp_300s``) — identical recipe to live MB win prices at inference.
    """
    from src.model import get_feature_columns

    if not os.path.exists(PLACE_BSP_PATH):
        raise FileNotFoundError(
            f"Place-market strategy needs {PLACE_BSP_PATH} — run "
            "scripts/extract_betfair_place_bsp.py first.")

    df = featured.copy()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    # Win pre-off price (BF ltp_300s) for the market features.
    df = attach_bsp(df, cols=("ltp_300s",))
    df["win_ltp300_raw"] = pd.to_numeric(df.get("ltp_300s"), errors="coerce")

    # Place market: places-paid (target) + place BSP / placed flag (settlement).
    pb = pd.read_parquet(PLACE_BSP_PATH, columns=[
        "race_date", "track", "off_key", "horse_key",
        "bf_places_paid", "place_bsp", "placed_bf"])
    pb["race_date"] = _d10(pb["race_date"])
    pb["track_key"] = pb["track"].map(bf_track_key)
    R = pb[["race_date", "track_key", "off_key", "horse_key",
            "bf_places_paid", "place_bsp", "placed_bf"]].drop_duplicates(
        ["race_date", "track_key", "off_key", "horse_key"])

    L = pd.DataFrame({
        "race_date": _d10(df["race_date"]),
        "track_key": df["track"].map(bf_track_key),
        "off_key": df["off_time"].map(normalise_off_time_key),
        "horse_key": df["horse_name"].map(bf_horse_key),
    })
    merged = L.merge(R, on=["race_date", "track_key", "off_key", "horse_key"], how="left")
    for col in ("bf_places_paid", "place_bsp", "placed_bf"):
        df[col] = pd.to_numeric(merged[col].to_numpy(), errors="coerce")

    form = [c for c in get_feature_columns(df) if c in df.columns]
    if use_market_features:
        mf = build_market_features(df, "win_ltp300_raw", "opening_odds")
        for c in MARKET_FEATS:
            df[c] = mf[c].to_numpy()
        feature_cols = form + MARKET_FEATS
    else:
        feature_cols = list(form)
    return df, feature_cols, form


def _placed_target(df: pd.DataFrame) -> pd.Series:
    """Placed = finished within Betfair's paid places. Prefer the exchange's own
    placed flag, else derive from finish position vs places-paid."""
    if "placed_bf" in df.columns and df["placed_bf"].notna().any():
        pl = pd.to_numeric(df["placed_bf"], errors="coerce")
        if pl.notna().any():
            return pl.fillna(0).astype(int)
    fp = pd.to_numeric(df["finish_position"], errors="coerce")
    pp = pd.to_numeric(df["bf_places_paid"], errors="coerce")
    return ((fp > 0) & (fp <= pp)).astype(int)


def train_place_market_model(df: pd.DataFrame, feature_cols: list[str], *, params: dict | None = None):
    """Fit the LGBM place classifier on the matched rows of *df* (those with a
    real Betfair places-paid — no fillna). Returns the fitted model."""
    import lightgbm as lgb

    train = df[df["bf_places_paid"].notna()].copy()
    y = _placed_target(train)
    model = lgb.LGBMClassifier(**(params or PLACE_PARAMS))
    model.fit(train[feature_cols], y)
    logger.info("Place model trained: %d rows, %d features (placed rate %.3f)",
                len(train), len(feature_cols), float(y.mean()))
    return model


def evaluate_place_market(df: pd.DataFrame, model, feature_cols: list[str], *,
                          ev_threshold: float = DEFAULT_EV_THRESHOLD,
                          commission: float = PLACE_EXCHANGE_COMMISSION,
                          jumps_only: bool = False, n_boot: int = 2000,
                          seed: int = 0) -> dict:
    """+EV place selection @ exchange place BSP on *df*, with bootstrap CI.

    Selection: ``ev = p_place * place_bsp - 1 > ev_threshold`` among runners with
    a valid place BSP and known places-paid. PnL settled by the shared exchange
    formula. Reports overall and (always) the jumps subset. Returns a dict; never
    raises on an empty selection (returns ``n_bets=0``).
    """
    d = df.copy()
    X = d[feature_cols]
    d["p_place"] = model.predict_proba(X)[:, 1]
    price = pd.to_numeric(d["place_bsp"], errors="coerce")
    pp = pd.to_numeric(d["bf_places_paid"], errors="coerce")
    valid = price.notna() & (price > 1.01) & pp.notna()
    d["ev"] = d["p_place"] * price - 1.0

    def _stats(mask: pd.Series) -> dict:
        sel = d[mask & valid & (d["ev"] > ev_threshold)]
        n = len(sel)
        if n == 0:
            return {"n_bets": 0, "roi_pct": None, "ci_low": None, "ci_high": None,
                    "total_pnl": 0.0, "hit_rate": None, "avg_odds": None}
        placed = _placed_target(sel).to_numpy().astype(bool)
        odds = pd.to_numeric(sel["place_bsp"], errors="coerce").to_numpy()
        pnl = settle_place_exchange_bets(odds, placed, stake=1.0, commission=commission)
        rng = np.random.default_rng(seed)
        boots = pnl[rng.integers(0, n, size=(n_boot, n))].mean(axis=1) * 100.0
        return {
            "n_bets": int(n),
            "roi_pct": round(float(pnl.mean() * 100.0), 2),
            "ci_low": round(float(np.percentile(boots, 2.5)), 2),
            "ci_high": round(float(np.percentile(boots, 97.5)), 2),
            "total_pnl": round(float(pnl.sum()), 1),
            "hit_rate": round(float(placed.mean()), 3),
            "avg_odds": round(float(odds.mean()), 2),
        }

    all_true = pd.Series(True, index=d.index)
    is_jumps = (d["race_type"].astype(str).str.contains("Hurdle|Chase", case=False, na=False)
                if "race_type" in d.columns else pd.Series(False, index=d.index))
    out = _stats(all_true)
    out.update({"ev_threshold": ev_threshold, "commission": commission,
                "n_candidates": int((valid).sum()), "jumps": _stats(is_jumps)})
    return out


def train_and_evaluate(featured: pd.DataFrame, *, use_market_features: bool = True,
                       ev_threshold: float = DEFAULT_EV_THRESHOLD,
                       commission: float = PLACE_EXCHANGE_COMMISSION,
                       test_frac: float = 0.2, params: dict | None = None):
    """Held-out-by-date eval + a final model on ALL matched rows (the artifact).

    Splits matched rows by race date (last *test_frac* of dates = test), trains
    an eval model on the earlier dates only and scores the held-out tail (honest
    ROI), then refits on every matched row for the saved model. Returns
    ``(final_model, meta, eval_block)``.
    """
    df, feature_cols, form = prepare_place_data(featured, use_market_features=use_market_features)
    matched = df[df["bf_places_paid"].notna()].copy()
    if matched.empty:
        raise ValueError("No rows with a Betfair places-paid match — cannot train place model.")

    dates = np.sort(matched["race_date"].dropna().unique())
    cutoff = dates[int(len(dates) * (1.0 - test_frac))] if len(dates) > 4 else dates[-1]
    tr = matched[matched["race_date"] < cutoff]
    te = matched[matched["race_date"] >= cutoff]
    logger.info("Place-market split @ %s: train %d / test %d",
                pd.Timestamp(cutoff).date(), len(tr), len(te))

    eval_model = train_place_market_model(tr, feature_cols, params=params)
    eval_block = evaluate_place_market(te, eval_model, feature_cols,
                                       ev_threshold=ev_threshold, commission=commission)
    eval_block.update({
        "use_market_features": bool(use_market_features),
        "test_cutoff": pd.Timestamp(cutoff).strftime("%Y-%m-%d"),
        "test_rows": int(len(te)), "train_rows": int(len(tr)),
    })

    final_model = train_place_market_model(matched, feature_cols, params=params)
    meta = {
        "feature_cols": feature_cols, "form": form,
        "market": MARKET_FEATS if use_market_features else [],
        "use_market_features": bool(use_market_features),
        "ev_threshold": ev_threshold, "commission": commission,
        "trained_rows": int(len(matched)),
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    return final_model, meta, eval_block


def walk_forward_place_market(featured: pd.DataFrame, *, use_market_features: bool = True,
                              ev_threshold: float = DEFAULT_EV_THRESHOLD,
                              commission: float = PLACE_EXCHANGE_COMMISSION,
                              min_train_quarters: int = 8, jumps_only: bool = False,
                              params: dict | None = None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Expanding-window walk-forward of the place-market strategy.

    For each quarter after a *min_train_quarters* burn-in, train on all earlier
    matched rows and forward-test on that quarter only (+EV selection @ place BSP,
    settled at *commission*). Pools every fold's bets for one ROI + bootstrap CI,
    and reports per-fold ROIs and how many folds were positive — the robust,
    across-time read (no single-split luck). This is the method behind the
    project's headline place-edge numbers.
    """
    df, feature_cols, _ = prepare_place_data(featured, use_market_features=use_market_features)
    matched = df[df["bf_places_paid"].notna()].copy()
    if matched.empty:
        raise ValueError("No rows with a Betfair places-paid match.")
    if jumps_only and "race_type" in matched.columns:
        matched = matched[matched["race_type"].astype(str).str.contains("Hurdle|Chase", case=False, na=False)]
    matched["_q"] = matched["race_date"].dt.to_period("Q")
    quarters = sorted(matched["_q"].unique())

    pnl_chunks, per_fold = [], []
    for q in quarters[min_train_quarters:]:
        tr = matched[matched["_q"] < q]
        te = matched[matched["_q"] == q]
        if len(tr) < 1000 or te.empty:
            continue
        model = train_place_market_model(tr, feature_cols, params=params)
        p = model.predict_proba(te[feature_cols])[:, 1]
        price = pd.to_numeric(te["place_bsp"], errors="coerce").to_numpy()
        pp = pd.to_numeric(te["bf_places_paid"], errors="coerce").to_numpy()
        ev = p * price - 1.0
        sel = np.isfinite(price) & (price > 1.01) & np.isfinite(pp) & (ev > ev_threshold)
        if not sel.any():
            continue
        te_sel = te[sel]
        placed = _placed_target(te_sel).to_numpy().astype(bool)
        pnl = settle_place_exchange_bets(price[sel], placed, commission=commission)
        pnl_chunks.append(pnl)
        per_fold.append({"quarter": str(q), "n_bets": int(sel.sum()),
                         "roi_pct": round(float(pnl.mean() * 100.0), 2)})
        logger.info("WF fold %s: %d bets, ROI %.2f%%", q, int(sel.sum()), float(pnl.mean() * 100))

    if not pnl_chunks:
        return {"n_folds": 0, "n_bets": 0, "roi_pct": None}
    pnl_all = np.concatenate(pnl_chunks)
    n = len(pnl_all)
    rng = np.random.default_rng(seed)
    boots = pnl_all[rng.integers(0, n, size=(n_boot, n))].mean(axis=1) * 100.0
    return {
        "n_folds": len(per_fold), "n_bets": int(n),
        "roi_pct": round(float(pnl_all.mean() * 100.0), 2),
        "ci_low": round(float(np.percentile(boots, 2.5)), 2),
        "ci_high": round(float(np.percentile(boots, 97.5)), 2),
        "total_pnl": round(float(pnl_all.sum()), 1),
        "folds_positive": int(sum(f["roi_pct"] > 0 for f in per_fold)),
        "use_market_features": bool(use_market_features), "jumps_only": bool(jumps_only),
        "ev_threshold": ev_threshold, "commission": commission,
        "per_fold": per_fold,
    }
