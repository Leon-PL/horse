"""Immutable local forecast studies, observations and first-eligible selection."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile
from uuid import uuid4
from zoneinfo import ZoneInfo

import config
from src.deployment_validation import file_sha256
from src.featured_dataset import _FE_SOURCE_FILES, fe_signature

ROOT = Path(config.BASE_DIR)
SOURCES = list(dict.fromkeys([*_FE_SOURCE_FILES, "config.py", "src/model.py",
    "src/probability_calibration.py", "src/each_way.py", "src/live_prediction.py", "src/betfair.py",
    "src/bet_settlement.py", "src/run_store.py", "feature_sets/win.txt",
    "src/matchbook_client.py", "src/matchbook_signals.py", "src/forward_store.py",
    "src/forward_capture.py", "src/forward_timing.py", "src/forward_assessment.py", "src/forecast_evaluation.py",
    "src/deployment_validation.py", "src/data_scraper.py", "src/feature_inputs.py",
    "src/parquet_io.py", "scripts/record_forecasts.py", "scripts/assess_forward_forecasts.py",
    "src/issued_evaluation.py", "src/forward_recorder.py", "scripts/run_forecast_recorder.py",
    "src/database.py", "scripts/start_forecast_recorder.ps1", "scripts/install_forecast_recorder.ps1",
    "src/forward_readout.py", "scripts/configure_forecast_recorder.py",
    "src/source_pipeline.py", "src/rtv_scraper.py", "src/rtv_feature_collection.py"]))


def utcnow():
    return datetime.now(timezone.utc)


def timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("A timezone-aware observation timestamp is required")
    return parsed.astimezone(timezone.utc)


def numeric_id(value):
    text = str(value)
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("Exchange IDs must be exact decimal strings or integers")
    return text


def publish_json(path, value):
    """Publish a complete file exclusively; concurrent writers cannot replace it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def create_study(path, *, model_id=None, now=None):
    from src.run_store import RUNS_DIR, get_production_run_id, load_run_meta, feature_generation_status, GENERATION_MATCH

    moment = timestamp(now or utcnow())
    model_id = model_id or get_production_run_id()
    if not model_id or not re.fullmatch(r"[A-Za-z0-9_-]+", model_id):
        raise ValueError("A saved model is required")
    generation = feature_generation_status(model_id)
    if generation["state"] != GENERATION_MATCH or generation.get("run_fe_signature") != fe_signature():
        raise ValueError("Model and current feature generation must match")
    metadata = load_run_meta(model_id)
    recipe = (metadata.get("training_config") or {}).get("recipe") or {}
    if not recipe or recipe.get("options", {}).get("blend_market_odds"):
        raise ValueError("Use a recorded independent recipe without a market blend")
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(RUNS_DIR) / model_id / "model.joblib", path / "model.joblib")
    shutil.copy2(Path(RUNS_DIR) / model_id / "meta.json", path / "model_meta.json")
    sources = {name: file_sha256(ROOT / name) for name in SOURCES}
    with zipfile.ZipFile(path / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(ROOT / name, name)
    value = {
        "schema_version": 1, "created_at": moment.isoformat(), "model_id": model_id,
        "model_sha256": file_sha256(path / "model.joblib"),
        "model_meta_sha256": file_sha256(path / "model_meta.json"),
        "fe_signature": fe_signature(), "recipe": recipe, "source_sha256": sources,
        "source_snapshot_sha256": file_sha256(path / "source_snapshot.zip"),
        "market_endpoint": config.MATCHBOOK_EDGE_URL.rstrip("/"),
        "first_eligible_date": (moment.astimezone(ZoneInfo("Europe/London")).date()+timedelta(days=1)).isoformat(),
        "model_update_rule": "Fixed pinned artifact; daily history may advance under unchanged FE code",
        "record_place_probabilities": True,
        "place_reference": "Same fitted place head under retained legacy calibration/projection; original issued top-k definition",
        "reference": "Matchbook WIN normalised best-back decimal prices, GBP; separate from Betfair historical LTP",
        "horizon_seconds": 300, "capture_window_seconds": 30,
        "timed_max_wait_seconds": 600,
        "timed_capture_rule": "Optional bounded wait after native inference, before unchanged-card recheck and fresh market reads; all publication gates unchanged",
        "schedule_rule": "First schedule observed by this recorder, known before the capture window; later changes reject eligibility",
        "max_runner_price_age_seconds": 300, "max_market_age_seconds": 90,
        "max_card_age_seconds": 90, "clock_skew_tolerance_seconds": 2,
        "max_history_lag_days": 2, "minimum_available_gbp": 2.0,
        "card_recheck_rule": "Fresh card after native inference; all input columns, dtypes and values must match the initial card exactly",
        "selection_rule": "First eligible complete paired capture per event; keep every attempt and later withdrawals separately",
        "assessment_scope": "Forward baseline monitoring; formal confirmation sample size is not registered, so no automatic parity or improvement declaration",
        "parity_margins_for_future_confirmation": {"race_log_loss": .002, "runner_brier": .0001},
        "runtime": {"python": __import__("sys").version, "packages": {name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "pyarrow", "lightgbm", "scikit-learn")}},
    }
    publish_json(path / "protocol.json", value)
    publish_json(path / "protocol.sha256.json", {"sha256": file_sha256(path / "protocol.json")})
    return ForwardStudy(path)


class ForwardStudy:
    def __init__(self, path):
        self.path = Path(path).resolve()
        raw = (self.path / "protocol.json").read_bytes()
        self.protocol = json.loads(raw)
        self.protocol_sha256 = hashlib.sha256(raw).hexdigest()
        if json.loads((self.path / "protocol.sha256.json").read_text())["sha256"] != self.protocol_sha256:
            raise ValueError("Study protocol changed after registration")

    def verify(self):
        p = self.protocol
        if file_sha256(self.path / "protocol.json") != self.protocol_sha256:
            raise ValueError("Study protocol changed")
        if file_sha256(self.path / "model.joblib") != p["model_sha256"]:
            raise ValueError("Pinned model changed")
        if file_sha256(self.path / "model_meta.json") != p["model_meta_sha256"]:
            raise ValueError("Pinned model metadata changed")
        if file_sha256(self.path / "source_snapshot.zip") != p["source_snapshot_sha256"]:
            raise ValueError("Frozen implementation archive changed")
        if fe_signature() != p["fe_signature"]:
            raise ValueError("Feature generation changed; use a new study")
        if config.MATCHBOOK_EDGE_URL.rstrip("/") != p["market_endpoint"]:
            raise ValueError("Market data endpoint changed")
        if any(file_sha256(ROOT / name) != expected for name, expected in p["source_sha256"].items()):
            raise ValueError("Forecast implementation changed; use a new study")
        effective = json.loads(json.dumps({name: getattr(config, name, None)
                      for name in p["recipe"]["model_config"]}, default=str, sort_keys=True))
        if effective != p["recipe"]["model_config"]:
            raise ValueError("Model configuration changed")

    def observe_event(self, event, observed_at):
        event_id = numeric_id(event["id"])
        value = {"event_id": event_id, "first_seen_at": timestamp(observed_at).isoformat(),
                 "first_start": timestamp(event["start"]).isoformat(),
                 "name": event.get("name"), "protocol_sha256": self.protocol_sha256}
        path = self.path / "events" / f"{event_id}.json"
        try:
            publish_json(path, value)
        except FileExistsError:
            value = json.loads(path.read_text())
            if (value["event_id"] != event_id or value["protocol_sha256"] != self.protocol_sha256
                    or timestamp(value["first_seen_at"]) > timestamp(observed_at)):
                raise ValueError("Stored first observation is inconsistent")
        return value

    def selected(self, event_id):
        path = self.path / "eligible" / f"{numeric_id(event_id)}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def begin_attempt(self):
        capture_id = utcnow().strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:12]
        path = self.path / "records" / capture_id
        path.mkdir(parents=True, exist_ok=False)
        return capture_id, path

    def finish_attempt(self, capture_id, value, *, clock=utcnow):
        path = self.path / "records" / capture_id
        required = {"forecasts.parquet", "model_inputs.parquet", "market_source.json", "racecard.parquet",
                    "feature_vintage.json", "event.json", "first_observation.json", "racecard_source.json",
                    "racecard_refresh_source.json", "racecard_refresh.parquet"}
        if value.get("capture_mode") == "timed":
            required |= {"timing_intent.json", "timing_receipt.json"}
        if value.get("benchmark_eligible") and (not value.get("paired_available")
                or value.get("errors") or not required <= {p.name for p in path.iterdir()}):
            raise ValueError("An eligible forecast requires complete paired evidence")
        value = {**value, "capture_id": capture_id, "protocol_sha256": self.protocol_sha256,
                 "files_sha256": {p.name: file_sha256(p) for p in path.iterdir() if p.is_file()}}
        publish_json(path / "capture.json", value)
        published = timestamp(clock())
        decision = timestamp(value["first_start"])-timedelta(seconds=self.protocol["horizon_seconds"]) if value.get("first_start") else None
        timing_ok = bool(value.get("benchmark_eligible") and decision is not None
                         and decision-timedelta(seconds=self.protocol["capture_window_seconds"]) <= published <= decision)
        receipt = {"capture_sha256": file_sha256(path / "capture.json"),
                   "published_at": published.isoformat(), "eligible_after_publication": timing_ok,
                   "reason": "published_after_window" if value.get("benchmark_eligible") and not timing_ok else None}
        publish_json(path / "publication.json", receipt)
        if timing_ok:
            try:
                publish_json(self.path / "eligible" / f"{numeric_id(value['event_id'])}.json",
                    {"capture_id": capture_id, "capture_sha256": file_sha256(path / "capture.json"),
                     "publication_sha256": file_sha256(path / "publication.json"),
                     "capture_published_at": published.isoformat(),
                     "protocol_sha256": self.protocol_sha256})
            except FileExistsError:
                pass  # The earlier eligible record remains authoritative.
        selected = self.selected(value["event_id"])
        return {**value, "selected_for_benchmark": bool(selected and selected["capture_id"] == capture_id),
                "published_at": published.isoformat(),
                "reasons": value.get("reasons", []) + ([receipt["reason"]] if receipt["reason"] else [])}

    def verify_record(self, capture_id):
        if not re.fullmatch(r"[A-Za-z0-9_]+", capture_id):
            raise ValueError("Invalid capture ID")
        path = self.path / "records" / capture_id
        value = json.loads((path / "capture.json").read_text())
        receipt = json.loads((path / "publication.json").read_text())
        if receipt["capture_sha256"] != file_sha256(path / "capture.json"):
            raise ValueError("Capture manifest changed after publication")
        if value["protocol_sha256"] != self.protocol_sha256:
            raise ValueError("Capture belongs to another study")
        if any(Path(name).name != name or file_sha256(path / name) != expected
               for name, expected in value["files_sha256"].items()):
            raise ValueError("Recorded forecast inputs changed")
        return {**value, "published_at": receipt["published_at"]}

    def selected_records(self):
        for path in sorted((self.path / "eligible").glob("*.json")):
            index = json.loads(path.read_text())
            record = self.verify_record(index["capture_id"])
            folder = self.path / "records" / index["capture_id"]
            if (index["protocol_sha256"] != self.protocol_sha256
                    or index["capture_sha256"] != file_sha256(folder / "capture.json")
                    or index["publication_sha256"] != file_sha256(folder / "publication.json")
                    or path.stem != record["event_id"] or not record["benchmark_eligible"]
                    or not json.loads((folder / "publication.json").read_text())["eligible_after_publication"]):
                raise ValueError("Eligible forecast index is inconsistent")
            decision = timestamp(record["first_start"])-timedelta(seconds=self.protocol["horizon_seconds"])
            if not decision-timedelta(seconds=self.protocol["capture_window_seconds"]) <= timestamp(record["published_at"]) <= decision:
                raise ValueError("Eligible forecast was published outside its window")
            yield record
