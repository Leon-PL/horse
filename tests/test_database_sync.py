"""Offline coverage for durable, resumable historical result ingestion."""

from collections import Counter
from datetime import datetime, timedelta
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

from src import database
from src import data_scraper


TODAY = datetime(2026, 10, 13, 12)
YESTERDAY = "2026-10-12"


def runners(race_id, date=YESTERDAY):
    return [
        {"race_id": race_id, "race_date": date, "horse_name": f"Horse {i}",
         "finish_position": position, "won": int(position == 1), "num_runners": 2}
        for i, position in enumerate([1, 99])
    ]


@pytest.fixture
def sync_env(tmp_path, monkeypatch):
    clock = [TODAY]

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls):
            return clock[0]

    class FakeScraper:
        def __init__(self):
            self.listings = {}
            self.responses = {}
            self.listed = []
            self.fetched = Counter()

        def get_results_urls(self, date, uk_only):
            assert uk_only
            self.listed.append(date)
            ids = self.listings.get(date, [])
            if isinstance(ids, Exception):
                raise ids
            return [{"race_id": race_id, "url": f"https://example.test/{race_id}"}
                    for race_id in ids]

        def scrape_race_result(self, url, date_str):
            race_id = url.rsplit("/", 1)[-1]
            self.fetched[race_id] += 1
            response = self.responses.get(race_id, runners(race_id, date_str))
            if isinstance(response, Exception):
                raise response
            return response

    fake = FakeScraper()
    enrichments = []
    monkeypatch.setattr(database, "datetime", FrozenDatetime)
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "races.db"))
    monkeypatch.setattr(database.config, "RAW_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(data_scraper, "SportingLifeScraper", lambda: fake)
    monkeypatch.setitem(sys.modules, "src.weather", SimpleNamespace(
        backfill_weather_in_db=lambda dates: enrichments.append(("weather", dates))))
    monkeypatch.setitem(sys.modules, "src.pedigree_backfill", SimpleNamespace(
        backfill_pedigree=lambda: enrichments.append(("pedigree", None))))
    monkeypatch.setattr("requests.sessions.Session.request",
                        lambda *a, **kw: pytest.fail("Network is forbidden in sync tests"))
    return SimpleNamespace(scraper=fake, root=tmp_path, enrichments=enrichments, clock=clock)


def ledger():
    with database._get_conn() as conn:
        return conn.execute("""
            SELECT race_id, completed, attempts FROM result_sync_races ORDER BY race_id
        """).fetchall()


def test_partial_failure_is_retried_without_refetching_successes(sync_env):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a", "b"]
    fake.responses["b"] = RuntimeError("temporary failure")
    first = database.sync_database(days_back=1)
    assert set(first.race_id) == {"a"}
    assert ledger() == [("a", 1, 1), ("b", 0, 1)]

    del fake.responses["b"]
    second = database.sync_database(days_back=1)
    assert len(second) == 4
    assert fake.fetched == {"a": 1, "b": 2}
    assert ledger() == [("a", 1, 1), ("b", 1, 2)]

    database.sync_database(days_back=1)
    assert fake.fetched == {"a": 1, "b": 2}
    assert len(database.load_from_database()) == 4
    assert len(sync_env.enrichments) == 4  # Only the two successful inserts.


def test_listing_has_no_arbitrary_race_cap_and_deduplicates_links(sync_env):
    sync_env.scraper.listings[YESTERDAY] = [str(i) for i in range(57)] + ["0"]
    result = database.sync_database(days_back=1)
    assert result.race_id.nunique() == 57
    assert len(result) == 114
    assert set(sync_env.scraper.fetched.values()) == {1}


@pytest.mark.parametrize("listing_error", [[], RuntimeError("listing unavailable")])
def test_persisted_race_is_retried_even_when_relisting_fails(sync_env, listing_error):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    fake.responses["a"] = []
    database.sync_database(days_back=1)
    fake.listings[YESTERDAY] = listing_error
    del fake.responses["a"]
    assert len(database.sync_database(days_back=1)) == 2
    assert ledger() == [("a", 1, 2)]
    with database._get_conn() as conn:
        assert conn.execute("SELECT listing_succeeded FROM result_sync_dates").fetchone() == (0,)


@pytest.mark.parametrize("invalid", [
    [],
    [{**row, "finish_position": 0} for row in runners("a")],
    [{**row, "finish_position": 99} for row in runners("a")],
    runners("a")[:1],
    [runners("a")[0], runners("a")[0]],
    runners("wrong-race"),
    runners("a", "2026-10-11"),
    [{**row, "horse_name": ""} for row in runners("a")],
])
def test_bad_responses_remain_pending_until_full_settled_field(sync_env, invalid):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    fake.responses["a"] = invalid
    assert database.sync_database(days_back=1).empty
    assert ledger() == [("a", 0, 1)]
    del fake.responses["a"]
    result = database.sync_database(days_back=1)
    assert list(result.finish_position) == [1, 99]  # Retain genuine non-finishers.
    assert ledger() == [("a", 1, 2)]


def test_inserts_and_completion_commit_atomically(sync_env, monkeypatch):
    sync_env.scraper.listings[YESTERDAY] = ["a"]
    real_insert = database._insert_results

    def interrupted_insert(conn, frame):
        real_insert(conn, frame.iloc[:1])
        raise RuntimeError("interrupted before completion")

    monkeypatch.setattr(database, "_insert_results", interrupted_insert)
    assert database.sync_database(days_back=1).empty
    assert ledger() == [("a", 0, 1)]
    monkeypatch.setattr(database, "_insert_results", real_insert)
    assert len(database.sync_database(days_back=1)) == 2
    assert ledger() == [("a", 1, 2)]


def test_legacy_migration_rechecks_recent_dates_only_and_repairs_fragments(sync_env, caplog):
    old = "2026-10-04"
    database.insert_results(pd.DataFrame(runners("a") + runners("b")[:1] + runners("old", old)))
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a", "b", "c"]
    fake.listings[old] = ["old", "previously-missing"]
    with caplog.at_level("INFO"):
        result = database.sync_database(days_back=10)
    assert old not in fake.listed
    assert fake.fetched == {"b": 1, "c": 1}
    assert len(result) == 8
    assert ledger() == [("a", 1, 0), ("b", 1, 1), ("c", 1, 1)]
    assert "older legacy dates without a completeness audit" in caplog.text


def test_history_audit_relists_legacy_dates_and_fetches_only_the_gaps(sync_env):
    """The old sync kept 50 races a day and skipped failed pages silently."""
    old = "2026-10-04"
    database.insert_results(pd.DataFrame(runners("kept", old)))
    fake = sync_env.scraper
    fake.listings[old] = ["kept", "truncated"]
    fake.listings[YESTERDAY] = []

    assert database.queue_history_audit() == 1
    database.sync_database(days_back=10)

    assert old in fake.listed
    assert fake.fetched == {"truncated": 1}  # the stored race is adopted
    assert database.queue_history_audit() == 0  # now ledger-managed
    with database._get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM results WHERE race_date = ?",
                            (old,)).fetchone()[0] == 4


def test_recent_relisting_discovers_late_races(sync_env):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    database.sync_database(days_back=1)
    fake.listings[YESTERDAY] = ["a", "b"]
    assert len(database.sync_database(days_back=1)) == 4
    assert fake.fetched == {"a": 1, "b": 1}


def test_empty_listing_can_recover_after_leaving_recent_window(sync_env):
    fake = sync_env.scraper
    database.sync_database(days_back=1)
    sync_env.clock[0] += timedelta(days=8)
    fake.listings[YESTERDAY] = ["a"]
    assert len(database.sync_database(days_back=9)) == 2
    assert fake.fetched == {"a": 1}


def test_pending_dates_outside_requested_window_are_not_fetched(sync_env):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    fake.responses["a"] = []
    database.sync_database(days_back=1)
    sync_env.clock[0] += timedelta(days=8)
    database.sync_database(days_back=1)
    assert fake.fetched == {"a": 1}


def test_up_to_date_path_returns_and_exports_full_history(sync_env):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    database.sync_database(days_back=1)
    database.insert_results(pd.DataFrame(runners("old", "2020-01-01")))
    sync_env.clock[0] += timedelta(days=8)
    fake.listed.clear()
    result = database.sync_database(days_back=9)
    assert YESTERDAY not in fake.listed  # Complete and outside the recent relist window.
    assert set(result.race_id) == {"a", "old"}
    snapshot = pd.read_csv(sync_env.root / "race_results.csv")
    assert set(snapshot.race_id) == {"a", "old"}
    assert fake.fetched == {"a": 1}
