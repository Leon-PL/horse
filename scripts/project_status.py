"""One command that answers "can I trust anything this system says right now?"

The pipeline has four links that must agree before a pick means anything:

    results database -> feature store -> trained model -> promoted model

Each link carries its own staleness, and each was previously only visible by
hand-reading a JSON file, a parquet footer and a run directory. When they
silently disagree the system still produces confident numbers — the 2026-06
"+15% place edge" was a stale store, and the trainer_freq leak inflated
ROI@BSP by 56 points. Both were found late and by accident.

This prints the state of every link and a plain verdict. Read-only: it opens
metadata and parquet footers, never the 2 GB of feature rows. By default it
also recomputes the data signature from the database (about a minute), which
is the only way to see results backfilled beneath the store; ``--quick``
skips that.

    .venv/Scripts/python.exe scripts/project_status.py
    .venv/Scripts/python.exe scripts/project_status.py --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

OK, WARN, BAD, DASH = "OK", "WARN", "STALE", "--"
_MARK = {OK: "[ ok ]", WARN: "[warn]", BAD: "[STALE]", DASH: "[  - ]"}

#: Results older than this make rolling-form features describe a stale horse.
FRESH_DATA_DAYS = 14


def _days_behind(value) -> int | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None
    return (date.today() - parsed).days


def _database() -> dict:
    try:
        from src.database import db_stats
        stats = db_stats()
    except Exception as exc:  # pragma: no cover - missing/locked db
        return {"state": BAD, "detail": f"unreadable ({exc})"}
    behind = _days_behind(stats.get("latest_date"))
    # Form features roll over the last ~20 runs, so a database months behind
    # makes every history feature describe a different horse than the one
    # running today. That is a blocker, not a warning.
    if behind is None:
        state = WARN
    elif behind <= 2:
        state = OK
    elif behind <= FRESH_DATA_DAYS:
        state = WARN
    else:
        state = BAD
    return {
        "state": state,
        "latest_date": stats.get("latest_date"),
        "days_behind": behind,
        "races": stats.get("total_races"),
        "runners": stats.get("total_runners"),
        "detail": (
            f"{stats.get('total_races'):,} races to {stats.get('latest_date')}"
            + (f" ({behind} days behind today)" if behind is not None else "")
        ),
    }


def store_findings(meta: dict, *, live_fe_sig: str, db_latest=None,
                   live_data_sig: str | None = None, atr_now: str | None = None) -> tuple[str, list[str]]:
    """Judge the store against everything that can make it stale.

    The FE signature alone missed three ways a store goes stale while its code
    is unchanged: results backfilled beneath it (``data_signature``), results
    added after it (the store lags the database), and the ATR sectional cache
    changing beneath it (read inside FE, so no other signature covers it).
    """
    state, notes = OK, []
    stored_sig = meta.get("fe_signature")
    if stored_sig != live_fe_sig:
        state = BAD
        notes.append(f"built by FE {stored_sig} but code is now {live_fe_sig} -> REBUILD")
    else:
        notes.append(f"FE code matches ({live_fe_sig})")
    if live_data_sig is not None:
        if live_data_sig != meta.get("data_signature"):
            state = BAD
            notes.append("results changed beneath the store (data_signature) -> REBUILD")
        else:
            notes.append("data matches")
    last = meta.get("last_date")
    if db_latest is not None and last and str(db_latest)[:10] > str(last)[:10]:
        state = BAD
        notes.append(f"database has results to {str(db_latest)[:10]} not yet in the store -> refresh")
    atr = meta.get("atr_signature")
    if isinstance(atr, dict) and atr.get("sig"):
        if atr_now is not None and atr_now != atr["sig"]:
            state = BAD
            notes.append("ATR sectional cache changed beneath the store -> REBUILD")
    elif state == OK:
        state = WARN
        notes.append("sectional cache unverified (store predates the ATR stamp)")
    return state, notes


def _feature_store(deep: bool = True) -> dict:
    from src.feature_store import FeatureStore
    from src.featured_dataset import data_signature, fe_signature

    store = FeatureStore()
    if not store.exists():
        return {"state": BAD, "detail": "not built - build it on the training app's Data page"}
    meta = store.meta()
    live_sig = fe_signature()
    db_latest = live_data_sig = atr_now = None
    try:
        from src.database import get_latest_date
        db_latest = get_latest_date()
    except Exception:  # pragma: no cover - reported by the database probe
        pass
    if deep and meta.get("last_date"):
        # Recompute from the database: catches a backfill beneath the store.
        from src.data_processor import process_data
        live_data_sig = data_signature(process_data(save=False), cutoff=meta["last_date"])
    atr = meta.get("atr_signature")
    if isinstance(atr, dict) and atr.get("cutoff"):
        from src.feature_inputs import atr_cache_signature
        atr_now = atr_cache_signature(atr["cutoff"])
    state, notes = store_findings(meta, live_fe_sig=live_sig, db_latest=db_latest,
                                  live_data_sig=live_data_sig, atr_now=atr_now)
    behind = _days_behind(meta.get("last_date"))
    return {
        "state": state,
        "fe_signature": meta.get("fe_signature"),
        "live_fe_signature": live_sig,
        "code_matches": meta.get("fe_signature") == live_sig,
        "data_checked": live_data_sig is not None,
        "last_date": meta.get("last_date"),
        "days_behind": behind,
        "rows": meta.get("featured_rows"),
        "built_at": meta.get("built_at"),
        "detail": (
            f"{meta.get('featured_rows', 0):,} rows to {meta.get('last_date')}"
            + (f" ({behind} days behind)" if behind is not None else "")
            + "; " + "; ".join(notes)
        ),
    }


def _runs() -> dict:
    from src.run_store import list_runs, run_signatures

    try:
        runs = list_runs()
    except Exception as exc:  # pragma: no cover
        return {"state": WARN, "detail": f"unreadable ({exc})"}
    if not runs:
        return {"state": BAD, "detail": "no training runs recorded"}
    newest = runs[0]
    rid = newest["run_id"]
    signed = sum(1 for r in runs if run_signatures(r["run_id"]).get("fe_signature"))
    return {
        "state": OK,
        "count": len(runs),
        "newest": rid,
        "newest_timestamp": newest.get("timestamp"),
        "with_signatures": signed,
        "detail": (
            f"{len(runs)} run(s), newest {rid}; "
            f"{signed}/{len(runs)} record a feature signature"
        ),
    }


def _production() -> dict:
    from src.run_store import (
        GENERATION_MATCH, GENERATION_MISMATCH, GENERATION_NO_MODEL,
        feature_generation_status,
    )

    status = feature_generation_status()
    state = {
        GENERATION_MATCH: OK,
        GENERATION_MISMATCH: BAD,
        GENERATION_NO_MODEL: BAD,
    }.get(status["state"], WARN)
    return {
        "state": state,
        "generation_state": status["state"],
        "run_id": status["run_id"],
        "detail": status["message"],
    }


def collect(deep: bool = True) -> dict:
    """Every link's state, worst-first in the printed order."""
    sections: dict[str, dict] = {}
    for name, probe in (
        ("database", _database),
        ("feature_store", (lambda: _feature_store(deep=False)) if not deep
                          else (lambda: _feature_store())),
        ("training_runs", _runs),
        ("production_model", _production),
    ):
        try:
            sections[name] = probe()
        except Exception as exc:  # pragma: no cover - probe must never crash the report
            sections[name] = {"state": WARN, "detail": f"check failed: {exc}"}

    blockers = [
        f"{name}: {body['detail']}"
        for name, body in sections.items()
        if body["state"] == BAD
    ]
    # Every link must hold: a promoted model matching a current store, built
    # from results recent enough for rolling-form features to mean anything.
    servable = not blockers
    return {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "sections": sections,
        "servable": servable,
        "blockers": blockers,
    }


def render(report: dict) -> str:
    lines = [
        "",
        f"  Horse racing pipeline status - {report['checked_at']}",
        "  " + "-" * 68,
    ]
    labels = {
        "database": "Results database",
        "feature_store": "Feature store",
        "training_runs": "Training runs",
        "production_model": "Promoted model",
    }
    for name, body in report["sections"].items():
        lines.append(f"  {_MARK[body['state']]:>7}  {labels[name]:<18} {body['detail']}")
    lines.append("  " + "-" * 68)
    if report["servable"]:
        lines.append("  VERDICT: ready - the promoted model matches the current features.")
    else:
        lines.append("  VERDICT: not ready to serve picks. Blocking:")
        lines += [f"           - {b}" for b in report["blockers"]]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--quick", action="store_true",
                        help="skip recomputing data_signature from the database (~1 min)")
    args = parser.parse_args()
    # A status report is the output; module start-up chatter is not.
    logging.disable(logging.INFO)

    report = collect(deep=not args.quick)
    print(json.dumps(report, indent=2, default=str) if args.json else render(report))
    return 0 if report["servable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
