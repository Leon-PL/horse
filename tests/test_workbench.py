"""Behavioural tests for local inspection, evidence and action boundaries."""
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zlib

import pandas as pd
import pytest

from src.source_contracts import TableContract
from src.source_dataset import publish_source, save_manifest
from src.workbench.data import Catalog, fingerprint, records
from src.workbench.jobs import Jobs
from src.workbench.review import ReviewStore
from src.workbench.server import WorkbenchServer


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    import config
    root = tmp_path/"data"
    root.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", str(root))
    frame = pd.DataFrame([
        {"race_id": "r1", "race_date": "2026-06-01", "track": "Ascot", "off_time": "13:00", "race_name": "Test field",
         "horse_id": "h1", "horse_name": "Name <script>alert(1)</script>", "finish_position": 1, "won": 1,
         "num_runners": 2, "distance_furlongs": 8., "surface": "Turf", "going": "Good"},
        {"race_id": "r1", "race_date": "2026-06-01", "track": "Ascot", "off_time": "13:00", "race_name": "Test field",
         "horse_id": "h2", "horse_name": "Second", "finish_position": 2, "won": 0,
         "num_runners": 2, "distance_furlongs": 8., "surface": "Turf", "going": "Good"},
    ])
    frame["race_type"] = "Flat"
    with sqlite3.connect(root/"races.db") as con:
        frame.to_sql("results", con, index=False)
        con.execute("CREATE INDEX races_date ON results(race_date)")
    receipt = publish_source("sl", {"results": frame}, contracts={"results": TableContract("canonical runner", ("race_id", "horse_id"), ("race_id", "horse_id"))})
    save_manifest({"sl": receipt})
    return Catalog(root, tmp_path)


def test_browsing_does_not_collect_publish_or_create_review_store(catalog, monkeypatch):
    monkeypatch.setattr("src.source_dataset.capture_dataset", lambda **kw: pytest.fail("Read attempted publication"))
    before = (catalog.sources/"datasets/latest.json").read_bytes()
    assert catalog.overview()["database"]["races"] == 1
    found = catalog.races("2026-06-01", "2026-06-01", "<script>")
    assert found["total"] == 1
    assert catalog.race("r1")["header"]["off_time_local"] == "14:00 BST"
    assert (catalog.sources/"datasets/latest.json").read_bytes() == before
    assert not (catalog.root/"workbench").exists()


def test_absent_accepted_source_does_not_fall_back_to_native(catalog):
    queue = catalog.root/"atr_cache/rebuild_v2"
    queue.mkdir(parents=True)
    with sqlite3.connect(queue/"rebuild.sqlite") as con:
        con.execute("CREATE TABLE runners(race_id TEXT,payload TEXT)")
        con.execute("INSERT INTO runners VALUES (?,?)", ("r1", json.dumps({"horse_name": "Second", "atr_finishing_speed_pct": 99})))
    accepted = catalog.race_source("r1", "atr")
    native = catalog.race_source("r1", "atr", "native")
    assert accepted["tables"] == {}
    assert native["tables"]["sectionals"]["count"] == 1


def test_dataset_selection_remains_explicit_and_manifest_content_is_checked(catalog):
    old = catalog.manifest()
    frame = catalog.canonical("r1")
    frame["going"] = "Soft"
    receipt = publish_source("sl", {"results": frame}, contracts={"results": TableContract("canonical runner", ("race_id",), ("race_id", "horse_id"))})
    new = save_manifest({"sl": receipt})
    assert old["id"] != new["id"]
    assert catalog.race_source("r1", "sl", dataset=old["id"])["tables"]["results"]["rows"][0]["going"] == "Good"
    assert catalog.race_source("r1", "sl")["tables"]["results"]["rows"][0]["going"] == "Soft"
    change = catalog.compare("r1", "sl", old["id"], new["id"])
    assert len(change["changes"]) == 2
    assert "going" in change["changes"][0]["fields"]
    path = catalog.sources/"datasets"/(old["id"]+".json")
    altered = json.loads(path.read_text())
    altered["purpose"] = "altered"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="content"):
        catalog.manifest(old["id"])


def test_bha_notes_use_unique_local_time_and_never_claim_runner_links(catalog):
    report = pd.DataFrame([{"report_key": "2026:1", "course": "ASCOT", "race_date": "2026-06-01"}])
    notes = pd.DataFrame([{"report_key": "2026:1", "race_number": 1, "off_time_local": "14:00", "text": "A source-native note"}])
    race = catalog.canonical("r1")
    result, message = catalog.qualify_bha({"reports": report, "race_notes": notes}, race)
    assert len(result["race_notes"]) == 1
    assert "No runner attribution" in message
    duplicate = race.copy()
    duplicate["race_id"] = "ambiguous"
    with sqlite3.connect(catalog.db) as con:
        duplicate.to_sql("results", con, if_exists="append", index=False)
    result, message = catalog.qualify_bha({"reports": report, "race_notes": notes}, race)
    assert result["race_notes"].empty
    assert len(result["unassigned_race_notes"]) == 1
    assert "Ambiguous" in message


def test_wrong_course_bha_report_is_not_attached(catalog):
    result, reason = catalog.qualify_bha({"reports": pd.DataFrame([{"report_key": "other", "course": "York"}])}, catalog.canonical("r1"))
    assert result == {}
    assert "No BHA fixture" in reason


def test_coverage_preserves_grains_and_marks_reference_sources(catalog):
    result = catalog.coverage("2026-06-01", "2026-06-02")
    sl = next(s for s in result["sources"] if s["source"] == "sl")
    fx = next(s for s in result["sources"] if s["source"] == "fx")
    assert sl["cells"][0]["count"] == 2
    assert sl["cells"][1]["state"] == "no_canonical_races"
    assert fx["cells"][0]["state"] == "reference"
    with pytest.raises(ValueError):
        catalog.coverage("2020-01-01", "2026-06-01")


def test_evidence_is_provider_scoped_and_hash_checked(catalog):
    folder = catalog.root/"sl_cache"
    folder.mkdir()
    raw = b"<script>alert('source text, not executable UI')</script>"
    sha = hashlib.sha256(raw).hexdigest()
    with sqlite3.connect(folder/"raw.sqlite") as con:
        con.execute("CREATE TABLE payloads(sha256 TEXT PRIMARY KEY,body BLOB)")
        con.execute("INSERT INTO payloads VALUES (?,?)", (sha, zlib.compress(raw)))
    assert catalog.evidence("sl", sha) == (raw, "text/plain; charset=utf-8")
    with pytest.raises(FileNotFoundError):
        catalog.evidence("bha", sha)
    with pytest.raises(ValueError):
        catalog.evidence("sl", "../../secret")


def test_notes_decisions_and_validated_proposals_survive_restart(catalog):
    path = catalog.root/"workbench/review.sqlite"
    store = ReviewStore(path)
    store.note("r1", "Check the original report", True)
    store.decide("issue-1", "investigating", "Evidence needed")
    before = records(catalog.canonical("r1"))
    proposal = store.propose(catalog, "r1", [{"field": "going", "value": "Soft"}], "Source correction", "sha256: evidence", fingerprint(before))
    assert proposal["validation_passed"]
    assert len(proposal["changes"]) == 2
    assert ReviewStore(path).read()["notes"][0]["bookmarked"] == 1
    assert ReviewStore(path).proposal(proposal["id"])["reason"] == "Source correction"
    assert records(catalog.canonical("r1")) == before
    with pytest.raises(ValueError, match="changed"):
        store.propose(catalog, "r1", [{"field": "going", "value": "Soft"}], "Reason", "Evidence", "stale")
    with pytest.raises(ValueError, match="not supported"):
        store.propose(catalog, "r1", [{"field": "horse_id", "value": "wrong"}], "Reason", "Evidence", fingerprint(before))


def test_invalid_result_proposal_is_saved_with_failed_validation(catalog):
    store = ReviewStore(catalog.root/"review.sqlite")
    frame = catalog.canonical("r1")
    result = store.propose(catalog, "r1", [{"horse_id": "h1", "field": "won", "value": 0}], "Review result", "Original result", fingerprint(records(frame)))
    assert not result["validation_passed"]
    assert result["findings"]
    assert catalog.canonical("r1").won.sum() == 1


def test_impossible_distance_is_reported_and_does_not_pass_proposal_validation(catalog):
    store = ReviewStore(catalog.root/"review.sqlite")
    frame = catalog.canonical("r1")
    result = store.propose(catalog, "r1", [{"field": "distance_furlongs", "value": -3}], "Check source", "Evidence", fingerprint(records(frame)))
    assert not result["validation_passed"]
    assert any(f["code"] == "invalid_distance_furlongs" for f in result["findings"])


def test_busy_bha_is_excluded_from_refresh_and_pedigree_needs_canonical_source(catalog, monkeypatch):
    monkeypatch.setattr(catalog, "workers", lambda: [{"source": "bha", "label": "BHA stewards", "active": True}])
    jobs = Jobs(catalog)
    with pytest.raises(ValueError, match="BHA stewards"):
        jobs.validate("refresh", {"start": "2026-06-01", "end": "2026-06-02", "sources": ["bha"]})
    with pytest.raises(ValueError, match="Sporting Life"):
        jobs.validate("refresh", {"start": "2026-06-01", "end": "2026-06-02", "sources": ["fx"], "enrich": True})
    assert jobs.validate("refresh", {"start": "2026-06-01", "end": "2026-06-02", "sources": ["rtv"], "enrich": True})["enrich"]
    assert jobs.validate("foaling_backfill", {"start": "2026-06-01", "end": "2026-06-02"}) == {}
    assert jobs.validate("refresh", {"start": "2026-06-01", "end": "2026-06-02", "sources": ["sl"], "weather": True})["weather"]


def test_atr_resume_preserves_last_recorded_request_delay(catalog):
    root = catalog.root/"atr_cache/rebuild_v2"
    root.mkdir(parents=True)
    (root/"launch_options.json").write_text(json.dumps({"request_delay": 2.5}))
    assert Jobs(catalog).validate("atr_resume", {})["request_delay"] == 2.5


def test_old_running_betfair_receipt_is_stale_not_an_active_collector(catalog):
    root = catalog.root/"betfair/history"
    root.mkdir(parents=True)
    (root/"status.json").write_text(json.dumps({"state": "running", "checked_at": "2020-01-01T00:00:00+00:00", "files": {"published": 12}}))
    worker = next(w for w in catalog.workers() if w["source"] == "betfair")
    assert worker["stale"]
    assert not worker["active"]
    assert worker["status"]["files"]["published"] == 12


def test_jobs_reject_duplicate_active_atr_and_oversized_refresh(catalog, monkeypatch):
    monkeypatch.setattr(catalog, "workers", lambda: [{"source": "atr", "label": "ATR", "active": True}])
    jobs = Jobs(catalog)
    with pytest.raises(ValueError, match="already owns"):
        jobs.validate("refresh", {"start": "2026-06-01", "end": "2026-06-02", "sources": ["atr"]})
    with pytest.raises(ValueError, match="31 days"):
        jobs.validate("refresh", {"start": "2020-01-01", "end": "2026-06-02", "sources": ["sl"]})
    assert jobs.validate("refresh", {"start": "2026-06-01", "end": "2026-06-02", "sources": ["bha"]})["sources"] == ["bha"]
    with pytest.raises(ValueError, match="between"):
        jobs.validate("atr_resume", {"request_delay": 0})


@pytest.fixture
def http_server(catalog):
    server = WorkbenchServer(("127.0.0.1", 0), catalog, Path(__file__).parents[1]/"ui/dist")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def request(server, path, body=None, headers=None):
    headers = headers or {}
    if body is not None:
        headers = {"Content-Type": "application/json", **headers}
    req = Request(f"http://127.0.0.1:{server.server_port}"+path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
    return urlopen(req, timeout=15)


def test_http_requires_local_host_session_and_origin_for_actions(http_server):
    server = http_server
    with request(server, "/api/session") as response:
        session = json.load(response)
    for headers in ({}, {"X-Workbench-Token": session["token"], "Origin": "https://other.example"}, {"Host": "other.example"}):
        with pytest.raises(HTTPError) as error:
            request(server, "/api/review/note", {"race_id": "r1", "note": "Forbidden"}, headers)
        assert error.value.code == 403
    with request(server, "/api/review/note", {"race_id": "r1", "note": "Saved"}, {"X-Workbench-Token": session["token"]}) as response:
        assert json.load(response)["saved"]
    with request(server, "/api/review") as response:
        assert json.load(response)["notes"][0]["note"] == "Saved"


def test_http_returns_real_records_downloads_and_blocks_traversal(http_server):
    server = http_server
    with request(server, "/api/races/r1") as response:
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        assert "<script>" in json.load(response)["rows"][0]["horse_name"]
    with request(server, "/api/races/r1/export?format=csv") as response:
        assert "attachment" in response.headers["Content-Disposition"]
        assert b"horse_name" in response.read()
    with pytest.raises(HTTPError) as error:
        request(server, "/../.env")
    assert error.value.code == 404
    with request(server, "/") as response:
        assert b"Racing Workbench" in response.read()
