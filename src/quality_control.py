"""
Data-Quality Gates for Betting
==============================
Single source of truth for the pre-bet **data-quality** checks. These gates
*suppress* bets where the model's inputs are known to be unreliable — they are
**not** value / edge rules (those live in :mod:`src.bet_settlement`). Keeping
them in one declared registry means the rules are explicit, tunable and
auditable in the app rather than ad-hoc filters scattered through the UI that
get added once and forgotten.

How it works
------------
Each check is declared in :data:`QC_CHECKS` with a stable ``id``, a human
description, the config keys it reads, and a function returning a per-row
boolean mask of *flagged* runners. :func:`assess_quality` runs every enabled
check and rolls the flags up to a **race-level block**: a race is blocked for
betting when it contains a flagged runner, because a single mispriced /
unpriced contender distorts every runner's race-normalised probability — so no
bet in that race can be trusted, not just the flagged horse's.

Current checks
--------------
* ``thin_history`` — a market *contender* (short price) with too few prior runs
  in our database. The classic case is a foreign raider (e.g. a French horse at
  Cheltenham) whose form lives in races we do not scrape: the model has no
  signal for it and systematically under-rates it. Confirmed in the 2026-06
  error analysis (market-fancied ≤1-run winners: model 0.275 vs market 0.337).

Adding a check
--------------
Append a spec to :data:`QC_CHECKS` and (optionally) defaults to
``config.QC_DEFAULTS``. The app panel and the betting suppression pick it up
automatically — no other wiring required.
"""

from __future__ import annotations

import pandas as pd

# Defaults live here so a fresh install has sane gates even without config
# overrides; ``config.QC_DEFAULTS`` (if present) and per-call cfg override them.
QC_DEFAULTS: dict = {
    # Off by default: quantification (2026-06-22) showed the thin_history gate
    # does not improve betting ROI on the current model — it is kept as opt-in,
    # auditable infrastructure (toggle in the Picks app's Quality Control panel).
    "enabled": False,
    # thin_history: a "contender" is a runner at or below this SP ...
    "thin_history_contender_max_odds": 8.0,
    # ... and "thin" means at most this many prior runs in our database.
    "thin_history_min_prev_races": 1,
}


def _resolve_cfg(cfg: dict | None) -> dict:
    """Merge call-time cfg over config.QC_DEFAULTS over module defaults."""
    merged = dict(QC_DEFAULTS)
    try:
        import config
        merged.update(config.QC_DEFAULTS or {})
    except Exception:
        pass
    if cfg:
        merged.update(cfg)
    return merged


def _check_thin_history(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Flag market contenders (short price) with too little DB form to judge."""
    if "horse_prev_races" not in df.columns or "odds" not in df.columns:
        return pd.Series(False, index=df.index)
    prev = pd.to_numeric(df["horse_prev_races"], errors="coerce")
    odds = pd.to_numeric(df["odds"], errors="coerce")
    min_prev = float(cfg.get("thin_history_min_prev_races", 1))
    max_odds = float(cfg.get("thin_history_contender_max_odds", 8.0))
    return (prev.fillna(0) <= min_prev) & odds.notna() & (odds <= max_odds)


# The registry. Order is display order. ``fn`` returns a per-row flag mask.
QC_CHECKS: list[dict] = [
    {
        "id": "thin_history",
        "label": "Thin-history contender",
        "description": (
            "A market contender (short price) with too few prior runs in our "
            "database — e.g. a foreign raider whose form we don't scrape. The "
            "model can't see its ability, under-rates it, and the whole race is "
            "mispriced as a result."
        ),
        "fn": _check_thin_history,
        "config_keys": ["thin_history_contender_max_odds", "thin_history_min_prev_races"],
    },
]


def assess_quality(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Run all checks and roll flags up to a race-level betting block.

    Returns a dict with:

    * ``flags``        — DataFrame of one boolean column per check (df index).
    * ``runner_flagged`` — bool Series: this runner tripped a check.
    * ``race_blocked`` — bool Series: this runner's race contains a flagged
      runner (so no bet in it is trustworthy).
    * ``reason``       — string Series: the labels of the checks each runner
      tripped (empty when none), for display.
    * ``blocked_race_ids`` — set of blocked race ids.
    * ``summary``      — per-check counts + totals, for the app panel.
    * ``config``       — the resolved config actually used.
    """
    cfg = _resolve_cfg(cfg)
    flags = pd.DataFrame(index=df.index)
    summary: dict = {"checks": {}}

    if not df.empty and cfg.get("enabled", True):
        for chk in QC_CHECKS:
            mask = chk["fn"](df, cfg).reindex(df.index).fillna(False).astype(bool)
            flags[chk["id"]] = mask.to_numpy()
            summary["checks"][chk["id"]] = {
                "label": chk["label"],
                "n_runners": int(mask.sum()),
            }
    else:
        for chk in QC_CHECKS:
            flags[chk["id"]] = False
            summary["checks"][chk["id"]] = {"label": chk["label"], "n_runners": 0}

    runner_flagged = flags.any(axis=1) if not flags.empty else pd.Series(False, index=df.index)

    # Reason string per runner: the labels of the checks it tripped.
    label_by_id = {c["id"]: c["label"] for c in QC_CHECKS}
    reason = pd.Series("", index=df.index, dtype=object)
    for cid in flags.columns:
        hit = flags[cid]
        reason = reason.mask(hit, (reason + "; " + label_by_id[cid]).str.lstrip("; "))

    # Race-level block: any flagged runner contaminates its whole race.
    if "race_id" in df.columns and runner_flagged.any():
        blocked_ids = set(df.loc[runner_flagged, "race_id"].unique())
        race_blocked = df["race_id"].isin(blocked_ids)
    else:
        blocked_ids = set()
        race_blocked = runner_flagged.copy()

    summary["n_runners_flagged"] = int(runner_flagged.sum())
    summary["n_races_blocked"] = len(blocked_ids)
    summary["n_runners_in_blocked_races"] = int(race_blocked.sum())

    return {
        "flags": flags,
        "runner_flagged": runner_flagged,
        "race_blocked": race_blocked,
        "reason": reason,
        "blocked_race_ids": blocked_ids,
        "summary": summary,
        "config": cfg,
    }


def apply_betting_gate(df: pd.DataFrame, bet_flag_cols, cfg: dict | None = None) -> dict:
    """Set the given bet-flag columns to ``False`` for blocked races, in place.

    *bet_flag_cols* is an iterable of column names (e.g. ``["is_value",
    "is_ew_value"]``). Returns the :func:`assess_quality` result so callers can
    render what was gated. Columns that don't exist are skipped.
    """
    qc = assess_quality(df, cfg)
    blocked = qc["race_blocked"]
    for col in bet_flag_cols:
        if col in df.columns:
            df.loc[blocked, col] = False
    return qc
