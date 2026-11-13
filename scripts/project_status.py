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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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
        state = BAD
    elif behind <= 2:
        state = OK
    elif behind <= FRESH_DATA_DAYS:
        state = WARN
    else:
        state = BAD
    pending = stats.get("recent_pending_races", 0)
    unverified = stats.get("recent_unverified_dates", 0)
    if pending or unverified:
        state = BAD
    return {
        "state": state,
        "latest_date": stats.get("latest_date"),
        "days_behind": behind,
        "races": stats.get("total_races"),
        "runners": stats.get("total_runners"),
        "pending_races": stats.get("pending_races", 0),
        "recent_pending_races": pending,
        "recent_unverified_dates": unverified,
        "detail": (
            f"{stats.get('total_races'):,} races to {stats.get('latest_date')}"
            + (f" ({behind} days behind today)" if behind is not None else "")
            + (f"; {pending} recent pending races, {unverified} unverified dates" if pending or unverified else "")
        ),
    }


def store_findings(meta: dict, *, live_fe_sig: str, db_latest=None,
                   live_data_sig: str | None = None, atr_now: str | None = None, rtv_now=None) -> tuple[str, list[str]]:
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
    if (isinstance(atr, dict) and atr.get("sig")
            and atr.get("cutoff") == last and last):
        if atr_now is not None and atr_now != atr["sig"]:
            state = BAD
            notes.append("ATR sectional cache changed beneath the store -> REBUILD")
    else:
        state = BAD
        notes.append("sectional cache unverified (missing stamp or cutoff differs from store)")
    if not meta.get("data_signature") or _days_behind(last) is None:
        state = BAD
        notes.append("store provenance incomplete -> rebuild")
    if rtv_now is not None:
        recorded = meta.get("rtv_signature")
        if (recorded is not None or any(rtv_now["sig"].values())) and recorded != rtv_now:
            state = BAD
            notes.append("RTV observations changed beneath the store -> REBUILD")
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
    # Compare against the dataset the next refresh would select.
    from src.source_dataset import committed_dataset, use_manifest
    dataset = committed_dataset()
    rtv_now = None
    if dataset is not None:
        with use_manifest(dataset):
            if deep and meta.get("last_date"):
                # Recompute from the database: catches a backfill beneath the store.
                # process_data() alone reads the published snapshot, which can still be old
                # after a DB repair or an interrupted snapshot publication.
                from src.data_processor import load_database_history, process_data
                live_data_sig = data_signature(process_data(df=load_database_history()), cutoff=meta["last_date"])
            atr = meta.get("atr_signature")
            if isinstance(atr, dict) and atr.get("cutoff"):
                from src.feature_inputs import atr_cache_signature
                atr_now = atr_cache_signature(atr["cutoff"])
            from src.featured_dataset import _rtv_signature
            rtv_now = _rtv_signature(meta["last_date"]) if meta.get("last_date") else None
    state, notes = store_findings(meta, live_fe_sig=live_sig, db_latest=db_latest,
                                  live_data_sig=live_data_sig, atr_now=atr_now, rtv_now=rtv_now)
    if dataset is None:
        state = BAD
        notes.append("no committed source dataset -> python -m src.ingestion --publish-sources")
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
        return {"state": BAD, "detail": f"unreadable ({exc})"}
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
        feature_generation_status, load_run_meta,
    )

    status = feature_generation_status()
    state = {
        GENERATION_MATCH: OK,
        GENERATION_MISMATCH: BAD,
        GENERATION_NO_MODEL: BAD,
    }.get(status["state"], BAD)
    # A recently created artifact can still contain parameters fitted months
    # earlier. Only the saved fitting split establishes that cutoff; neither
    # the artifact timestamp nor the feature store's end date establishes it.
    training_last_date = None
    if status["run_id"]:
        try:
            meta = load_run_meta(status["run_id"])
            training = meta.get("training_config", {})
            split = training.get("split", {}) if isinstance(training, dict) else {}
            value = split.get("training_last_date") if isinstance(split, dict) else None
            if isinstance(value, str):
                training_last_date = date.fromisoformat(value).isoformat()
        except (FileNotFoundError, OSError, ValueError):
            pass
    age = _days_behind(training_last_date)
    detail = status["message"]
    if status["run_id"]:
        detail += (
            f" Parameters fitted through {training_last_date} ({age} days ago)."
            if training_last_date else " Parameter fitting cutoff unknown."
        )
    return {
        "state": state,
        "generation_state": status["state"],
        "store_state": status.get("store_state"),
        "run_id": status["run_id"],
        "training_last_date": training_last_date,
        "parameter_age_days": age,
        "detail": detail,
    }


def _rtv_collection():
    """Sidecar health is visible but cannot invalidate an unchanged model."""
    from pathlib import Path
    from datetime import timezone
    root = Path(config.DATA_DIR) / "rtv_cache/pipeline"
    values = {}
    for kind in ("history", "snapshot"):
        path = root / (kind + "_status.json")
        if path.exists():
            values[kind] = json.loads(path.read_text(encoding="utf-8"))
    if not values:
        return {"state": WARN, "detail": "RTV has not been refreshed by the source pipeline"}
    healthy = True
    for kind in ("history", "snapshot"):
        value = values.get(kind, {})
        observed = value.get("checked_at")
        age = ((datetime.now(timezone.utc)-datetime.fromisoformat(observed)).total_seconds() if observed else float("inf"))
        healthy &= value.get("state") == "ok" and 0 <= age <= (86400 if kind == "history" else 3600)
    return {"state": OK if healthy else WARN, "detail": "; ".join(
        f"{kind}: {values.get(kind, {}).get('state', 'missing')}" for kind in ("history", "snapshot")), **values}


def _source_refresh():
    from src.ingestion.pipeline import DEFAULT_SOURCES
    path = Path(config.DATA_DIR)/"source_refresh/status.json"
    if not path.exists():
        return {"state": WARN, "detail": "No unified source coverage report yet"}
    report = json.loads(path.read_text(encoding="utf-8"))
    observed = report.get("checked_at")
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(observed)).total_seconds() if observed else float("inf")
    values = report.get("sources", {})
    if report.get("readiness"):
        from src.source_health import consumer_readiness
        readiness = consumer_readiness(values, purpose="serving", requested_end=report.get("end"))
        # A subset check cannot refresh the mandatory providers' check times.
        for provider in ("sl", "fx"):
            checked = values.get(provider, {}).get("checked_at") or observed
            elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(checked)).total_seconds() if checked else float("inf")
            if not 0 <= elapsed <= 86400:
                readiness["blockers"].append(f"{provider}: coverage report is stale")
        readiness["ready"] = not readiness["blockers"] and not report.get("publication_error")
        detail = [f"{k}: {v.get('state')}" for k, v in values.items()]
        return {"state": OK if readiness["ready"] and not readiness["limitations"] else WARN if readiness["ready"] else BAD,
                "detail": "; ".join(detail+readiness["blockers"]), "readiness": readiness, "report": report}
    missing = sorted(set(DEFAULT_SOURCES)-values.keys())
    issues = []
    if missing:
        issues.append("refresh did not cover " + ", ".join(missing))
    if not 0 <= age <= 86400:
        issues.append("coverage report is stale")
    if str(report.get("end") or "") < (date.today()-timedelta(days=1)).isoformat():
        issues.append("refresh range ends before yesterday")
    complete = (report.get("state") == "ok" and not issues
                and all(v.get("state") in {"ok", "cached"} for v in values.values()))
    detail = [f"{k}: {v.get('state')}" for k, v in values.items()]
    return {"state": OK if complete else BAD, "detail": "; ".join(detail+issues), "report": report}


#: The correction's records go stale silently when the forecast history stops following results.
FORECAST_HISTORY_LAG_DAYS = 7


def _forecast_history(db_latest=None) -> dict:
    from src.correction.history import status

    meta = status()
    if not meta:
        return {"state": DASH, "detail": "none (the forecast correction is not installed)"}
    detail = f"{meta['races']:,} races to {meta['last_date']}"
    if len(meta.get("recipe_signatures", [])) != 1:
        return {"state": WARN, "detail": detail + "; forecasts from more than one recipe"}
    try:
        lag = (datetime.fromisoformat(str(db_latest)[:10]) - datetime.fromisoformat(meta["last_date"])).days
    except (TypeError, ValueError):
        return {"state": WARN, "detail": detail + "; results date unknown"}
    if lag > FORECAST_HISTORY_LAG_DAYS:
        return {"state": WARN, "detail": detail + f"; {lag} days behind results -> python -m src.correction extend"}
    return {"state": OK, "detail": detail}


def collect(deep: bool = True) -> dict:
    """Every link's state, worst-first in the printed order."""
    sections: dict[str, dict] = {}
    for name, probe in (
        ("database", _database),
        ("source_refresh", _source_refresh),
        ("feature_store", (lambda: _feature_store(deep=False)) if not deep
                          else (lambda: _feature_store())),
        ("training_runs", _runs),
        ("production_model", _production),
    ):
        try:
            sections[name] = probe()
        except Exception as exc:  # pragma: no cover - probe must never crash the report
            # A failed check establishes nothing about readiness. Keep the
            # report usable, but do not publish a success exit code.
            sections[name] = {"state": BAD, "detail": f"check failed: {exc}"}

    blockers = [
        f"{name}: {body['detail']}"
        for name, body in sections.items()
        if body["state"] == BAD
    ]
    try:
        sections["rtv_collection"] = _rtv_collection()
    except Exception as exc:
        sections["rtv_collection"] = {"state": WARN, "detail": f"RTV monitoring failed: {exc}"}
    try:
        sections["forecast_history"] = _forecast_history(sections.get("database", {}).get("latest_date"))
    except Exception as exc:
        sections["forecast_history"] = {"state": WARN, "detail": f"forecast history check failed: {exc}"}
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
        "source_refresh": "Source coverage",
        "feature_store": "Feature store",
        "training_runs": "Training runs",
        "production_model": "Promoted model",
        "rtv_collection": "Racing TV source",
        "forecast_history": "Forecast history",
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
