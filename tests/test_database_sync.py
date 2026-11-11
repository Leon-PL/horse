"""Offline coverage for durable, resumable historical result ingestion."""

from collections import Counter
from datetime import datetime, timedelta
from types import SimpleNamespace
import sqlite3
import sys

import pandas as pd
import pytest

from src import database
from src.ingestion import sporting_life as data_scraper
from src.ingestion import results


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

        def scrape_race_result(self, url, date_str, raise_if_not_run=False):
            race_id = url.rsplit("/", 1)[-1]
            self.fetched[race_id] += 1
            response = self.responses.get(race_id, runners(race_id, date_str))
            if isinstance(response, data_scraper.RaceNotRun) and not raise_if_not_run:
                return []
            if isinstance(response, Exception):
                raise response
            return response

    fake = FakeScraper()
    enrichments = []
    monkeypatch.setattr(database, "datetime", FrozenDatetime)
    monkeypatch.setattr(results, "datetime", FrozenDatetime)
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "races.db"))
    monkeypatch.setattr(database.config, "RAW_DATA_DIR", str(tmp_path))
    # These fixtures name races with letters; real race IDs are numbers.
    monkeypatch.setitem(database.RESULTS_SNAPSHOT_COLUMNS, "race_id", "text")
    monkeypatch.setattr(data_scraper, "SportingLifeScraper", lambda: fake)
    monkeypatch.setitem(sys.modules, "src.ingestion.weather", SimpleNamespace(
        backfill_weather_in_db=lambda dates: enrichments.append(("weather", dates))))
    monkeypatch.setitem(sys.modules, "src.ingestion.pedigree", SimpleNamespace(
        backfill_pedigree=lambda: enrichments.append(("pedigree", None))))
    monkeypatch.setattr("requests.sessions.Session.request",
                        lambda *a, **kw: pytest.fail("Network is forbidden in sync tests"))
    return SimpleNamespace(scraper=fake, root=tmp_path, enrichments=enrichments, clock=clock)


def sync(*args, **kwargs):
    """Run the collector and return the stored history these tests inspect."""
    results.sync_database(*args, **kwargs)
    return database.load_from_database()


def ledger():
    with database._get_conn() as conn:
        return conn.execute("""
            SELECT race_id, completed, attempts FROM result_sync_races ORDER BY race_id
        """).fetchall()


def test_claims_are_published_and_follow_result_revisions(sync_env):
    race_id = "claim"
    sync_env.scraper.listings[YESTERDAY] = [race_id]
    field = runners(race_id)
    field[0].update(weight_lbs=149, reported_jockey_claim_lbs=5)
    field[1].update(weight_lbs=144, reported_jockey_claim_lbs=0)
    sync_env.scraper.responses[race_id] = field
    sync(days_back=1)
    path = sync_env.root / "race_results.parquet"
    snapshot = database.read_results_snapshot(path).sort_values("horse_name")
    assert snapshot.reported_jockey_claim_lbs.tolist() == [5, 0]
    assert snapshot.weight_lbs.tolist() == [149, 144]

    field[0]["reported_jockey_claim_lbs"] = None
    sync(days_back=1, recheck=True)
    snapshot = database.read_results_snapshot(path).sort_values("horse_name")
    assert snapshot.reported_jockey_claim_lbs.isna().tolist() == [True, False]


def test_results_only_preserves_full_export_without_enrichment(sync_env):
    old = "2026-09-01"
    database.insert_results(pd.DataFrame(runners("old", old)))
    sync_env.scraper.listings[YESTERDAY] = ["new"]
    result = sync(days_back=1, enrich=False)
    assert set(result.race_id) == {"old", "new"}
    exported = pd.read_parquet(sync_env.root / "race_results.parquet")
    assert set(exported.race_id) == {"old", "new"}
    assert ledger() == [("new", 1, 1)]
    assert sync_env.enrichments == []


def test_partial_failure_is_retried_without_refetching_successes(sync_env):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a", "b"]
    fake.responses["b"] = RuntimeError("temporary failure")
    first = sync(days_back=1)
    assert set(first.race_id) == {"a"}
    assert ledger() == [("a", 1, 1), ("b", 0, 1)]

    del fake.responses["b"]
    second = sync(days_back=1)
    assert len(second) == 4
    assert fake.fetched == {"a": 1, "b": 2}
    assert ledger() == [("a", 1, 1), ("b", 1, 2)]

    sync(days_back=1)
    assert fake.fetched == {"a": 1, "b": 2}
    assert len(database.load_from_database()) == 4
    assert sync_env.enrichments == []  # Ordinary result refreshes do not run optional enrichment.


def test_listing_has_no_arbitrary_race_cap_and_deduplicates_links(sync_env):
    sync_env.scraper.listings[YESTERDAY] = [str(i) for i in range(57)] + ["0"]
    result = sync(days_back=1)
    assert result.race_id.nunique() == 57
    assert len(result) == 114
    assert set(sync_env.scraper.fetched.values()) == {1}


@pytest.mark.parametrize("listing_error", [[], RuntimeError("listing unavailable")])
def test_persisted_race_is_retried_even_when_relisting_fails(sync_env, listing_error):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    fake.responses["a"] = []
    sync(days_back=1)
    fake.listings[YESTERDAY] = listing_error
    del fake.responses["a"]
    assert len(sync(days_back=1)) == 2
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
    runners("a") + [{**runners("a")[1], "horse_name": "obsolete non-runner"}],
    [{**row, "num_runners": 0} for row in runners("a")],
    [{**row, "num_runners": None} for row in runners("a")],
    [runners("a")[0], {**runners("a")[1], "num_runners": 3}],
    [runners("a")[0], {**runners("a")[1], "finish_position": None}],
    [runners("a")[0], {**runners("a")[1], "finish_position": float("inf")}],
    [runners("a")[0], {**runners("a")[1], "finish_position": 1.5}],
    [runners("a")[0], {**runners("a")[1], "won": 1}],
])
def test_bad_responses_remain_pending_until_full_settled_field(sync_env, invalid):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    fake.responses["a"] = invalid
    assert sync(days_back=1).empty
    assert ledger() == [("a", 0, 1)]
    del fake.responses["a"]
    result = sync(days_back=1)
    assert list(result.finish_position) == [1, 99]  # Retain genuine non-finishers.
    assert ledger() == [("a", 1, 2)]


def test_inserts_and_completion_commit_atomically(sync_env, monkeypatch):
    sync_env.scraper.listings[YESTERDAY] = ["a"]
    real_insert = database._replace_race_results

    def interrupted_insert(conn, frame):
        real_insert(conn, frame)
        raise RuntimeError("interrupted before completion")

    monkeypatch.setattr(database, "_replace_race_results", interrupted_insert)
    assert sync(days_back=1).empty
    assert ledger() == [("a", 0, 1)]
    monkeypatch.setattr(database, "_replace_race_results", real_insert)
    assert len(sync(days_back=1)) == 2
    assert ledger() == [("a", 1, 2)]


def test_legacy_migration_rechecks_recent_dates_only_and_repairs_fragments(sync_env, caplog):
    old = "2026-10-04"
    database.insert_results(pd.DataFrame(runners("a") + runners("b")[:1] + runners("old", old)))
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a", "b", "c"]
    fake.listings[old] = ["old", "previously-missing"]
    with caplog.at_level("INFO"):
        result = sync(days_back=10)
    assert old not in fake.listed
    assert fake.fetched == {"a": 1, "b": 1, "c": 1}  # Recent complete fields are revision-checked too.
    assert len(result) == 8
    assert ledger() == [("a", 1, 1), ("b", 1, 1), ("c", 1, 1)]
    assert "older legacy dates without a completeness audit" in caplog.text


def test_history_audit_relists_legacy_dates_and_fetches_only_the_gaps(sync_env):
    """The old sync kept 50 races a day and skipped failed pages silently."""
    old = "2026-10-04"
    database.insert_results(pd.DataFrame(runners("kept", old)))
    fake = sync_env.scraper
    fake.listings[old] = ["kept", "truncated"]
    fake.listings[YESTERDAY] = []

    assert database.queue_history_audit() == 1
    sync(days_back=10)

    assert old in fake.listed
    assert fake.fetched == {"truncated": 1}  # the stored race is adopted
    assert database.queue_history_audit() == 0  # now ledger-managed
    with database._get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM results WHERE race_date = ?",
                            (old,)).fetchone()[0] == 4


def test_recent_relisting_discovers_late_races(sync_env):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    sync(days_back=1)
    fake.listings[YESTERDAY] = ["a", "b"]
    assert len(sync(days_back=1)) == 4
    assert fake.fetched == {"a": 1, "b": 1}


def test_empty_listing_can_recover_after_leaving_recent_window(sync_env):
    fake = sync_env.scraper
    sync(days_back=1)
    sync_env.clock[0] += timedelta(days=8)
    fake.listings[YESTERDAY] = ["a"]
    assert len(sync(days_back=9)) == 2
    assert fake.fetched == {"a": 1}


def test_pending_dates_outside_requested_window_are_not_fetched(sync_env):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    fake.responses["a"] = []
    sync(days_back=1)
    sync_env.clock[0] += timedelta(days=8)
    sync(days_back=1)
    assert fake.fetched == {"a": 1}


def test_up_to_date_path_returns_and_exports_full_history(sync_env):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    sync(days_back=1)
    database.insert_results(pd.DataFrame(runners("old", "2020-01-01")))
    sync_env.clock[0] += timedelta(days=8)
    fake.listed.clear()
    result = sync(days_back=9)
    assert YESTERDAY not in fake.listed  # Complete and outside the recent relist window.
    assert set(result.race_id) == {"a", "old"}
    snapshot = pd.read_parquet(sync_env.root / "race_results.parquet")
    assert set(snapshot.race_id) == {"a", "old"}
    assert fake.fetched == {"a": 1}


def test_races_that_never_ran_are_closed_not_retried(sync_env):
    """Abandoned/void races used to stay pending forever (3,445 of them after
    the 2026-10-13 history audit), refetched on every wide sync."""
    old = "2026-09-01"
    fake = sync_env.scraper
    fake.listings[old] = ["ran", "abandoned"]
    fake.responses["abandoned"] = data_scraper.RaceNotRun("ABANDONED")
    sync(days_back=60)
    sync(days_back=60)
    assert fake.fetched == {"ran": 1, "abandoned": 1}
    with database._get_conn() as conn:
        row = conn.execute("SELECT completed, last_error FROM result_sync_races "
                           "WHERE race_id = 'abandoned'").fetchone()
    assert row == (1, "not run: ABANDONED")


def test_recent_dormant_race_stays_open(sync_env):
    """A DORMANT page from yesterday may just not be updated yet."""
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["late"]
    fake.responses["late"] = data_scraper.RaceNotRun("DORMANT")
    sync(days_back=1)
    fake.responses["late"] = runners("late")
    sync(days_back=1)
    assert fake.fetched == {"late": 2}
    assert ledger() == [("late", 1, 2)]


@pytest.mark.parametrize("already_completed", [False, True])
def test_overfull_stored_race_is_replaced_and_enrichment_preserved(sync_env, already_completed):
    old = runners("a") + [{**runners("a")[1], "horse_name": "obsolete non-runner"}]
    database.insert_results(pd.DataFrame(old))
    with database._get_conn() as conn:
        conn.execute("UPDATE results SET odds = 5, weather_temp_max = 17 WHERE race_id = 'a'")
        if already_completed:
            conn.execute("INSERT INTO result_sync_dates (race_date, listing_succeeded) VALUES (?, 1)",
                         (YESTERDAY,))
            conn.execute("INSERT INTO result_sync_races (race_date, race_id, url, completed) "
                         "VALUES (?, 'a', 'https://example.test/a', 1)", (YESTERDAY,))
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    fake.responses["a"] = [{**row, "odds": 12} for row in runners("a")]
    result = sync(days_back=1)
    assert len(result) == 2
    assert set(result.horse_name) == {"Horse 0", "Horse 1"}
    assert result.odds.eq(12).all()
    assert result.weather_temp_max.eq(17).all()
    assert fake.fetched == {"a": 1}
    assert ledger() == [("a", 1, 1)]


@pytest.mark.parametrize("failure", ["invalid_response", "transaction_failure"])
def test_failed_replacement_preserves_entire_previous_race(sync_env, monkeypatch, failure):
    old = runners("a") + [{**runners("a")[1], "horse_name": "obsolete non-runner"}]
    database.insert_results(pd.DataFrame(old))
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    if failure == "invalid_response":
        fake.responses["a"] = runners("a")[:1]
    else:
        original = database._replace_race_results

        def interrupted(conn, frame):
            original(conn, frame)
            raise RuntimeError("interrupted after replacement")

        monkeypatch.setattr(database, "_replace_race_results", interrupted)
    result = sync(days_back=1)
    assert len(result) == 3
    assert set(result.horse_name) == {"Horse 0", "Horse 1", "obsolete non-runner"}
    assert ledger() == [("a", 0, 1)]


def test_inconsistent_stored_counts_are_not_adopted_as_complete(sync_env):
    old = [runners("a")[0], {**runners("a")[1], "num_runners": 1}]
    database.insert_results(pd.DataFrame(old))
    sync_env.scraper.listings[YESTERDAY] = ["a"]
    result = sync(days_back=1)
    assert result.num_runners.eq(2).all()
    assert sync_env.scraper.fetched == {"a": 1}


def test_interrupted_csv_write_preserves_previous_whole_file(tmp_path, monkeypatch):
    target = tmp_path / "results.csv"
    original = pd.DataFrame(runners("old"))
    original.to_csv(target, index=False)
    previous_bytes = target.read_bytes()

    def fail_replace(*args):
        raise OSError("injected publish failure")

    monkeypatch.setattr(database.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        database.write_csv_atomically(pd.DataFrame(runners("new")), str(target))
    assert target.read_bytes() == previous_bytes
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("interrupted", [False, True])
def test_database_operation_always_releases_handle_and_transaction(sync_env, interrupted):
    database.init_db()
    try:
        with database._get_conn() as conn:
            database._insert_results(conn, pd.DataFrame(runners("a")))
            if interrupted:
                raise RuntimeError("interrupted operation")
    except RuntimeError:
        assert interrupted
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")
    with database._get_conn() as fresh:
        actual = fresh.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    assert actual == (0 if interrupted else 2)
