"""Execute one persisted workbench command outside the HTTP process."""
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time

from src.ingestion.storage import atomic_json, writer_lock
from src.workbench.data import Catalog, connection, read_json


def run(path):
    path = Path(path).resolve()
    job = read_json(path)
    if path.name != job.get("id", "")+".json":
        raise ValueError("Invalid job receipt")
    import config
    if Path(job["data_dir"]).resolve() != Path(config.DATA_DIR).resolve():
        raise ValueError("Background jobs must use the configured project data directory")

    def stamp(**extra):
        job.update(extra, updated_at=datetime.now(timezone.utc).isoformat())
        with writer_lock(path.with_suffix(".lock")) as acquired:
            if not acquired:
                raise RuntimeError("Job receipt is busy")
            atomic_json(path, job)

    stamp(state="running", pid=os.getpid())
    catalog = Catalog(config.DATA_DIR, config.BASE_DIR)
    kind, params = job["kind"], job["params"]
    try:
        if kind == "feature_update":
            from src.workbench.features import update_store
            def progress(**values):
                stamp(progress={**job.get("progress", {}), **values})
            class FeatureProgress(logging.Handler):
                last = 0
                def emit(self, record):
                    if record.name not in {"src.feature_engineer", "src.store_build", "src.ratings"} or time.monotonic()-self.last < 2:
                        return
                    self.last = time.monotonic()
                    try:
                        progress(phase="building", message=record.getMessage()[:320])
                    except Exception:
                        self.handleError(record)
            handler = FeatureProgress()
            logging.getLogger().addHandler(handler)
            try:
                result = update_store(catalog, params["dataset"], params["rebuild"], progress=progress)
            finally:
                logging.getLogger().removeHandler(handler)
            state = "complete"
        elif kind in {"model_benchmark", "model_fit"}:
            from src.workbench.modelling import Modelling, run_modelling_job
            try:
                result = run_modelling_job(Modelling(catalog), job, lambda **values: stamp(progress=values))
            except SystemExit as exc:
                raise ValueError("The modelling command rejected the settings. See the job log.") from exc
            state = "complete"
        elif kind == "feature_check":
            from src.workbench.features import Features
            service = Features(catalog)
            selection = service.selection(params["feature_set"])
            if (selection["signature"] if selection else None) != params["feature_set_signature"]:
                raise ValueError("The feature set changed after this check was queued; queue it again")
            stamp(progress={"phase": "checking", "message": "Scanning the bounded stored cohort"})
            result = service.quality(params["start"], params["end"], params["feature_set"], params["family"])
            state = "partial" if result["missing_columns"] or result["duplicate_keys"] or any(r["infinite"] for r in result["rows"]) else "complete"
        elif kind in {"refresh", "replay"}:
            from src.ingestion.pipeline import refresh_sources
            result = refresh_sources(start=params["start"], end=params["end"],
                sources=("sl",) if kind == "replay" else tuple(params["sources"]),
                offline=kind == "replay", recheck=params.get("recheck", False),
                enrich=params.get("enrich", False), weather=params.get("weather", False),
                atr_max_races=params.get("atr_max_races", 50),
                progress_callback=lambda source, body: stamp(progress={"source": source, **body}))
            state = "complete" if result["state"] in {"ok", "cached"} else "partial"
        elif kind == "foaling_backfill":
            from src.ingestion.pedigree import backfill_rtv_foaling_and_publish

            def profile_progress(body):
                stamp(progress={**body, "source": "Racing TV",
                                "message": f"{body['ok']:,} profiles fetched · {body['pending']:,} pending"})

            result = backfill_rtv_foaling_and_publish(progress=profile_progress)
            state = "complete" if result["state"] in {"ok", "cached"} else "partial"
        elif kind == "publish":
            from src.source_dataset import capture_dataset
            manifest = capture_dataset(include_prices=True)
            result, state = {"dataset_id": manifest["id"], "sources": list(manifest["sources"])}, "complete"
        elif kind == "verify":
            from src.source_dataset import load_manifest
            stamp(progress={"source": "selected dataset", "state": "verifying content hashes"})
            manifest = load_manifest(params["dataset"])
            result, state = {"dataset_id": manifest["id"], "verified_sources": list(manifest["sources"])}, "complete"
        elif kind == "validate":
            result = catalog.issues(params["start"], params["end"])
            state = "complete" if not any(r["severity"] == "error" for r in result["rows"]) else "partial"
        elif kind == "atr_resume":
            from src.ingestion import atr_rebuild as atr
            root = catalog.root/"atr_cache/rebuild_v2"
            with connection(root/"rebuild.sqlite") as con:
                pinned = json.loads(con.execute("SELECT value FROM control WHERE key='code_hashes'").fetchone()[0])
            if pinned != atr.code_hashes():
                raise ValueError("ATR collector code differs from the checkpoint; review the change before resuming")
            from src.workbench.jobs import alive
            previous = read_json(root/"status.json")
            if previous.get("state") in {"running", "waiting"} and alive(previous.get("pid")):
                raise ValueError("An ATR collector is already running")
            (root/"STOP").unlink(missing_ok=True)
            original_status = atr.publish_status
            recorded = False
            def record_status(*args, **kwargs):
                nonlocal recorded
                body = original_status(*args, **kwargs)
                if not recorded:
                    atomic_json(root/"launch_options.json", {"launcher_pid": os.getpid(), "started_at": job["created_at"],
                        "request_delay": params["request_delay"], "code_hashes": pinned,
                        "purpose": "Resumed from Racing Workbench", "job_id": job["id"]})
                    recorded = True
                return body
            atr.publish_status = record_status
            try:
                atr.run(root, request_delay=params["request_delay"])
            finally:
                atr.publish_status = original_status
            result = read_json(root/"status.json")
            state = "complete" if result.get("state") in {"complete", "stopped", "batch_complete"} else "partial"
        else:
            raise ValueError("Unsupported job")
        stamp(state=state, result=result, finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        logging.exception("Workbench operation failed")
        stamp(state="failed", error=str(exc), finished_at=datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(sys.argv[1])
