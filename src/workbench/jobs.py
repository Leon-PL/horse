"""Durable project jobs; existing owners retain locks and policies."""
from datetime import date, datetime, timedelta, timezone
import os
import re
import subprocess
import sys
from uuid import uuid4

import psutil

from src.ingestion.storage import atomic_json, writer_lock
from src.workbench.data import read_json, date_range
from src.workbench.modelling import MODELLING_JOBS, Modelling


def alive(pid, created_at=None):
    if not pid:
        return False
    try:
        process = psutil.Process(int(pid))
        return ("python" in process.name().lower() and process.is_running()
                and (not created_at or process.create_time() >= datetime.fromisoformat(created_at).timestamp()-15))
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


class Jobs:
    def __init__(self, catalog, modelling=None):
        self.catalog = catalog
        self.root = catalog.root/"workbench/jobs"
        self.modelling = modelling or Modelling(catalog)

    def list(self):
        output = []
        for path in sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
            value = read_json(path)
            if value.get("state") in {"queued", "running"} and not alive(value.get("pid"), value.get("created_at")):
                age = (datetime.now(timezone.utc)-datetime.fromisoformat(value["created_at"])).total_seconds()
                if age > 30:
                    value = {**value, "state": "interrupted", "error": "Worker exited before recording completion; inspect its log"}
            output.append(value)
        return output

    def validate(self, kind, params):
        if not isinstance(params, dict):
            raise ValueError("Operation parameters must be a JSON object")
        if isinstance(kind, str) and kind in MODELLING_JOBS:
            return self.modelling.validate_job(kind, params)
        if kind not in {"refresh", "replay", "publish", "validate", "verify", "atr_resume", "feature_update", "feature_check", "foaling_backfill"}:
            raise ValueError("Unknown workbench operation")
        if kind in {"feature_update", "feature_check"}:
            from src.workbench.features import Features
            service = Features(self.catalog)
            if kind == "feature_update":
                manifest = self.catalog.manifest(params.get("dataset"))
                if not manifest.get("id"):
                    raise ValueError("Publish and select an accepted dataset before building features")
                if not isinstance(params.get("rebuild", False), bool):
                    raise ValueError("Rebuild must be a boolean")
                return {"dataset": manifest["id"], "rebuild": params.get("rebuild", False)}
            first, last = date_range(params["start"], params["end"], 31)
            selected = service.selection(params.get("feature_set"))
            family = params.get("family", "")
            from src.workbench.feature_catalog import FAMILIES
            if family and family not in FAMILIES:
                raise ValueError("Choose a known feature family")
            return {"start": first.isoformat(), "end": last.isoformat(),
                    "feature_set": selected["id"] if selected else "all", "family": family,
                    "feature_set_signature": selected["signature"] if selected else None}
        if kind in {"refresh", "replay", "validate"}:
            yesterday = date.today()-timedelta(days=1)
            start = params.get("start") or (yesterday-timedelta(days=6)).isoformat()
            end = params.get("end") or yesterday.isoformat()
            first, last = date_range(start, end, 31)
            if last >= date.today():
                raise ValueError("Historical ingestion requires a settled past date range")
            params = {**params, "start": first.isoformat(), "end": last.isoformat()}
        if kind == "refresh":
            from src.ingestion.pipeline import DEFAULT_SOURCES
            sources = params.get("sources", list(DEFAULT_SOURCES))
            if not isinstance(sources, list) or not sources or set(sources)-set(DEFAULT_SOURCES):
                raise ValueError("Choose valid ingestion sources, including bha for BHA stewards")
            busy = [w["label"] for w in self.catalog.workers() if w["source"] in sources and (w["active"] or w.get("process_alive") and w["status"].get("state") in {"running", "waiting"})]
            if busy:
                raise ValueError(", ".join(busy)+" already owns its collector; choose other sources while it runs")
            if params.get("enrich") and not {"sl", "rtv"}.intersection(sources):
                raise ValueError("Pedigree enrichment requires Sporting Life or Racing TV")
            if params.get("weather") and "sl" not in sources:
                raise ValueError("Weather collection requires Sporting Life")
            params = {"start": params["start"], "end": params["end"], "sources": sources,
                      "recheck": bool(params.get("recheck")), "atr_max_races": min(max(int(params.get("atr_max_races", 50)), 1), 500),
                      "enrich": bool(params.get("enrich")), "weather": bool(params.get("weather"))}
        if kind == "atr_resume":
            status = read_json(self.catalog.root/"atr_cache/rebuild_v2/status.json")
            if alive(status.get("pid")) and status.get("state") in {"running", "waiting"}:
                raise ValueError("An ATR worker is already running")
            launch = read_json(self.catalog.root/"atr_cache/rebuild_v2/launch_options.json")
            delay = float(params.get("request_delay", launch.get("request_delay", 1.5)))
            if not 1.5 <= delay <= 30:
                raise ValueError("ATR request delay must be between 1.5 and 30 seconds")
            params = {"request_delay": delay}
        if kind == "verify":
            manifest = self.catalog.manifest(params.get("dataset"))
            if not manifest.get("id"):
                raise ValueError("No accepted dataset is available to verify")
            params = {"dataset": manifest["id"]}
        if kind in {"publish", "foaling_backfill"}:
            params = {}
        return params

    def start(self, kind, params):
        params = self.validate(kind, params)
        self.root.mkdir(parents=True, exist_ok=True)
        with writer_lock(self.root/".manager.lock") as acquired:
            if not acquired:
                raise ValueError("Job submission is busy; try again")
            active = [j for j in self.list() if j["state"] in {"queued", "running"}]
            # The regular coordinator also publishes; serialize those owners.
            conflicts = {"refresh", "replay", "publish", "foaling_backfill"}
            if any(j["kind"] == kind or {j["kind"], kind}.issubset(conflicts)
                   or {j["kind"], kind}.issubset(MODELLING_JOBS) for j in active):
                raise ValueError("A conflicting operation is already queued or running")
            identifier = uuid4().hex
            path = self.root/(identifier+".json")
            value = {"id": identifier, "kind": kind, "params": params, "state": "queued",
                     "created_at": datetime.now(timezone.utc).isoformat(), "data_dir": str(self.catalog.root)}
            atomic_json(path, value)
            with path.with_suffix(".log").open("wb") as log:
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                try:
                    process = subprocess.Popen([sys.executable, "-u", "-m", "src.workbench.worker", str(path)],
                                               cwd=self.catalog.base, stdout=log, stderr=log,
                                               stdin=subprocess.DEVNULL, creationflags=flags,
                                               start_new_session=os.name != "nt")
                except Exception as exc:
                    atomic_json(path, {**value, "state": "failed", "error": str(exc)})
                    raise
            # Do not overwrite a worker that has already recorded its own PID.
            with writer_lock(self.root/(identifier+".lock")) as free:
                if free:
                    current = read_json(path)
                    if current.get("state") == "queued":
                        atomic_json(path, {**current, "pid": process.pid})
            return read_json(path)

    def stop_atr(self):
        status = read_json(self.catalog.root/"atr_cache/rebuild_v2/status.json")
        if not alive(status.get("pid")) or status.get("state") not in {"running", "waiting"}:
            raise ValueError("No running ATR worker was found")
        path = self.catalog.root/"atr_cache/rebuild_v2/STOP"
        path.write_text("Graceful stop requested from Racing Workbench\n", encoding="utf-8")
        return {"state": "stop_requested", "pid": status["pid"], "message": "The worker will finish its current request and flush exports"}

    def log(self, identifier):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Invalid job ID")
        path = self.root/(identifier+".log")
        if not path.exists():
            return {"text": "No log output yet"}
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size-64000))
            text = stream.read().decode("utf-8", errors="replace")
        text = re.sub(r"(?i)(bearer\s+|(?:password|token|app[_-]?key)\s*[=:]\s*)[^\s,'\"}]+", r"\1[redacted]", text)
        return {"text": text}
