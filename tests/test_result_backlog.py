"""Bounded recovery uses dated listing evidence without losing pending fields."""
# ruff: noqa: F811 -- imported pytest fixture is injected by name
from datetime import timedelta
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from test_database_sync import sync_env, runners, TODAY, YESTERDAY  # noqa: F401
from src import database
from src.ingestion import sporting_life as data_scraper, pipeline as source_pipeline
from src.ingestion import results

OLD = "2026-08-01"


@pytest.fixture
def published_sources(monkeypatch):
    # These recovery fixtures intentionally omit canonical IDs. Publication has
    # its own contracts/tests; report-path checks must not adopt native caches.
    from src import source_dataset
    calls = []
    def capture(**kwargs):
        calls.append(kwargs)
        return {"id": "f"*64}
    monkeypatch.setattr(source_dataset, "capture_dataset", capture)
    return calls


def queue(identity, *, attempted=None, error="Empty or incomplete result response"):
    database.init_db()
    with database._get_conn() as con:
        con.execute("INSERT INTO result_sync_races(race_date,race_id,url,attempts,last_attempt_at,last_error) "
                    "VALUES (?,?,?,1,?,?)", (OLD, identity, f"https://example.test/{identity}", attempted, error))


def evidence(reason="DORMANT", **extra):
    return {"source_date": OLD, "reason": reason, "has_results": False,
            "payload_sha256": "a"*64,
            "source_url": f"https://www.sportinglife.com/racing/results/{OLD}", **extra}


def test_backlog_budget_does_not_limit_current_collection(sync_env):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["new-a", "new-b"]
    queue("old-a")
    queue("old-b")
    result = results.sync_database(1, backlog_max_races=1)
    assert fake.fetched == {"new-a": 1, "new-b": 1, "old-a": 1}
    assert result["pending_races"] == 0 and result["older_pending_races"] == 1
    assert result["backlog"]["attempted"] == result["backlog"]["completed"] == 1
    assert set(database.load_from_database().race_id) == {"new-a", "new-b", "old-a"}


def test_listing_proves_many_terminal_jobs_without_race_page_requests(sync_env):
    for identity in ("a", "b", "c"):
        queue(identity)
    fake = sync_env.scraper
    original = fake.get_results_urls
    def listing(day, uk_only):
        links = original(day, uk_only)
        fake.result_listing_evidence = {OLD: {"a": evidence(), "b": evidence("ABANDONED"), "c": evidence("ARABIAN")}}
        return links
    fake.get_results_urls = listing
    report = results.sync_database(0, backlog_max_races=3)
    assert fake.fetched == {} and fake.listed == [OLD]
    assert report["listing_resolved_races"] == 3 and report["requests"] == 0
    assert report["backlog"]["completed"] == 3 and report["older_pending_races"] == 0
    with database._get_conn() as con:
        assert con.execute("SELECT COUNT(*) FROM result_terminal_evidence WHERE source_url IS NOT NULL").fetchone()[0] == 3
        assert con.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0


@pytest.mark.parametrize("changes", [{"source_date": "2026-08-02"}, {"source_date": ""},
                                     {"has_results": True}, {"reason": "INQUIRY"}, {"payload_sha256": None}])
def test_unproved_or_conflicting_listing_falls_back_to_the_full_page(sync_env, changes):
    queue("a")
    sync_env.scraper.result_listing_evidence = {OLD: {"a": evidence(**changes)}}
    report = results.sync_database(0, backlog_max_races=1)
    assert sync_env.scraper.fetched == {"a": 1}
    assert report["listing_resolved_races"] == 0 and report["new_rows"] == 2


def test_listing_cannot_close_a_job_with_accepted_results(sync_env):
    queue("a")
    database.insert_results(pd.DataFrame(runners("a", OLD)))
    sync_env.scraper.result_listing_evidence = {OLD: {"a": evidence("ABANDONED")}}
    report = results.sync_database(0, backlog_max_races=1)
    assert sync_env.scraper.fetched == {"a": 1} and report["listing_resolved_races"] == 0
    assert len(database.load_from_database()) == 2


def test_recent_dormant_listing_does_not_close_the_job(sync_env):
    queue("a")
    with database._get_conn() as con:
        con.execute("UPDATE result_sync_races SET race_date=?", (YESTERDAY,))
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    fake.result_listing_evidence = {YESTERDAY: {"a": evidence(source_date=YESTERDAY)}}
    fake.responses["a"] = data_scraper.RaceNotRun("DORMANT")
    report = results.sync_database(1)
    assert report["pending_races"] == 1 and fake.fetched == {"a": 1}


def test_failed_backlog_job_yields_to_other_due_jobs(sync_env):
    queue("a")
    queue("b")
    fake = sync_env.scraper
    fake.responses["a"] = RuntimeError("temporary failure")
    first = results.sync_database(0, backlog_max_races=1)
    second = results.sync_database(0, backlog_max_races=1)
    assert first["backlog"]["unresolved_attempts"] == 1
    assert fake.fetched == {"a": 1, "b": 1}
    assert second["backlog"]["completed"] == 1


def test_backlog_retains_review_cases_and_recent_attempts(sync_env):
    queue("review", error="Review required: accepted correction differs")
    queue("identity", error="Missing or duplicate runner names")
    queue("recent", attempted=TODAY.isoformat())
    queue("due", attempted=(TODAY-timedelta(days=2)).isoformat())
    report = results.sync_database(0, backlog_max_races=10)
    assert sync_env.scraper.fetched == {"due": 1}
    assert report["backlog"]["remaining_races"] == 3
    assert report["backlog"]["review_required_races"] == 2
    assert report["backlog"]["retry_waiting_races"] == 1
    assert report["backlog"]["due_races"] == 0


def test_source_coordinator_can_recover_backlog_without_a_recent_window(sync_env, monkeypatch, published_sources):
    monkeypatch.setattr(source_pipeline.config, "DATA_DIR", str(sync_env.root))
    queue("a")
    report = source_pipeline.refresh_sources(0, sources=("sl",), sl_backlog_max_races=1)
    assert report["sources"]["sl"]["backlog"]["completed"] == 1
    assert report["dataset_manifest_id"] == "f"*64
    assert published_sources == [{"include_prices": True}]


def test_recovery_report_does_not_replace_the_current_all_source_report(sync_env, monkeypatch, published_sources):
    monkeypatch.setattr(source_pipeline.config, "DATA_DIR", str(sync_env.root))
    path = sync_env.root/"source_refresh/status.json"
    path.parent.mkdir()
    path.write_text('{"existing_report": true}', encoding="utf-8")
    queue("a")
    source_pipeline.refresh_sources(0, sources=("sl",), sl_backlog_max_races=1)
    assert json.loads(path.read_text()) == {"existing_report": True}
    assert json.loads(path.with_name("backlog_status.json").read_text())["state"] == "ok"
    assert published_sources == [{"include_prices": True}]


def test_provider_refusal_stops_historical_fallback_page_requests(sync_env):
    queue("a")
    class ProviderError(RuntimeError):
        response = SimpleNamespace(status_code=403)
    sync_env.scraper.listings[OLD] = ProviderError("provider refused")
    report = results.sync_database(0, backlog_max_races=1)
    assert report["backlog"]["aborted_blocked"]
    assert sync_env.scraper.fetched == {} and report["older_pending_races"] == 1


def listing_payload(day=OLD, *, stage="DORMANT", name="Ordinary race"):
    return {"props": {"pageProps": {"meetings": [{"meeting_summary": {
        "date": day, "course": {"name": "Kempton", "country": {"long_name": "England"}}},
        "races": [{"race_summary_reference": {"id": 123}, "date": day, "name": name, "race_stage": stage}]}]}}}


def test_scraper_retains_dated_evidence_for_excluded_arabian_races(monkeypatch):
    client = data_scraper.SportingLifeScraper()
    monkeypatch.setattr(client, "_fetch_next_data", lambda url: listing_payload(name="Handicap (Arab Race)"))
    assert client.get_results_urls(OLD) == []
    assert client.result_listing_evidence[OLD]["123"]["reason"] == "ARABIAN"


def test_wrong_listing_date_cannot_provide_terminal_evidence(monkeypatch):
    client = data_scraper.SportingLifeScraper()
    monkeypatch.setattr(client, "_fetch_next_data", lambda url: listing_payload(day="2026-08-02"))
    with pytest.raises(ValueError, match="another meeting date"):
        client.get_results_urls(OLD)
    assert OLD not in client.result_listing_evidence


def test_provider_error_is_not_a_confirmed_empty_listing(monkeypatch):
    client = data_scraper.SportingLifeScraper()
    monkeypatch.setattr(client, "_fetch_next_data", lambda url: {"props": {"pageProps": {"meetings": [], "hasError": True}}})
    with pytest.raises(ValueError, match="Source reported an error"):
        client.get_results_urls(OLD)
    assert OLD not in client.empty_result_dates


def test_terminal_evidence_points_to_the_exact_retained_response(tmp_path, monkeypatch):
    client = data_scraper.SportingLifeScraper()
    client.raw_store_path = tmp_path/"raw.sqlite"
    body = ('<script id="__NEXT_DATA__">'+json.dumps(listing_payload())+'</script>').encode()
    monkeypatch.setattr(client, "_get", lambda url: SimpleNamespace(content=body, text=body.decode(), status_code=200))
    client.get_results_urls(OLD)
    expected = hashlib.sha256(body).hexdigest()
    assert client.result_listing_evidence[OLD]["123"]["payload_sha256"] == expected
    client.offline = True
    monkeypatch.setattr(client, "_get", lambda url: pytest.fail("HTTP during replay"))
    client.get_results_urls(OLD)
    assert client.result_listing_evidence[OLD]["123"]["payload_sha256"] == expected


@pytest.mark.parametrize("second", ["SIERRA NEVADA", " Sierra Nevada ", "Sierra  Nevada"])
def test_ingestion_rejects_names_that_collapse_under_shared_normalisation(second):
    rows = runners("collision", OLD)
    rows[0].update(horse_name="Sierra Nevada")
    rows[1].update(horse_name=second)
    with pytest.raises(ValueError, match="duplicate runner names"):
        database._validate_race_results(pd.DataFrame(rows), OLD, "collision")
