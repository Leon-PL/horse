"""Source recovery, publication and retention regressions from the stage-one audit."""
# ruff: noqa: F811 -- imported pytest fixture is injected by its registered name
from datetime import date, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from test_database_sync import sync_env, runners, YESTERDAY  # noqa: F401
from src import database
from src.ingestion import pipeline as sources, sporting_life as data_scraper
from src.ingestion import pedigree
from src.ingestion import results
from src.ingestion.storage import RawPayloadStore, atomic_json, writer_lock


def test_unchanged_refresh_neither_loads_history_nor_rewrites_snapshot(sync_env, monkeypatch):
    sync_env.scraper.listings[YESTERDAY] = ["a"]
    first = results.sync_database(1)
    before = (sync_env.root/"race_results.parquet").stat().st_mtime_ns
    monkeypatch.setattr(database, "load_from_database", lambda: pytest.fail("Full history loaded on no-op"))
    monkeypatch.setattr(database, "_export_results_snapshot", lambda *a: pytest.fail("Unchanged history exported"))
    second = results.sync_database(1)
    assert second["snapshot"]["state"] == "cached"
    assert second["snapshot"]["revision"] == first["snapshot"]["revision"]
    assert (sync_env.root/"race_results.parquet").stat().st_mtime_ns == before


def test_direct_repair_invalidates_snapshot_even_without_new_rows(sync_env):
    sync_env.scraper.listings[YESTERDAY] = ["a"]
    first = results.sync_database(1)
    with database._get_conn() as con:
        con.execute("UPDATE results SET odds=12 WHERE race_id='a'")
    second = results.sync_database(0)
    assert second["snapshot"]["state"] == "published"
    assert second["snapshot"]["revision"] > first["snapshot"]["revision"]
    assert pd.read_parquet(sync_env.root/"race_results.parquet").odds.eq(12).all()


def test_completed_race_is_revision_checked_when_due(sync_env):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    results.sync_database(1)
    revised = runners("a")
    revised[0].update(finish_position=2, won=0)
    revised[1].update(finish_position=1, won=1)
    fake.responses["a"] = revised
    sync_env.clock[0] += timedelta(hours=7)
    result = results.sync_database(1)
    assert fake.fetched["a"] == 2
    assert result["changed_races"] == 1 and result["changed_from"] == YESTERDAY
    assert database.load_from_database().finish_position.tolist() == [2, 1]


def test_revision_cannot_overwrite_a_reviewed_result(sync_env):
    fake = sync_env.scraper
    fake.listings[YESTERDAY] = ["a"]
    results.sync_database(1)
    with database._get_conn() as con:
        con.execute("CREATE TABLE verified_result_rows(race_id TEXT)")
        con.execute("INSERT INTO verified_result_rows VALUES ('a')")
    revised = runners("a")
    revised[0].update(finish_position=2, won=0)
    revised[1].update(finish_position=1, won=1)
    fake.responses["a"] = revised
    result = results.sync_database(1, recheck=True)
    assert result["state"] == "partial" and result["pending_races"] == 1
    assert database.load_from_database().finish_position.tolist() == [1, 99]


def test_database_refresh_lock_prevents_duplicate_publishers(sync_env):
    with writer_lock(Path(database.DB_PATH).with_suffix(".source.lock")) as acquired:
        assert acquired
        with pytest.raises(RuntimeError, match="already running"):
            results.sync_database(0)


def test_source_race_times_are_retained_outside_model_columns(sync_env):
    frame = pd.DataFrame([{**r, "winning_time_secs": 91.5, "off_time_actual": "12:03", "prize_currency": "EUR"}
                          for r in runners("a")])
    database.ingest_results(frame)
    with database._get_conn() as con:
        evidence = json.loads(con.execute("SELECT payload_json FROM result_race_observations").fetchone()[0])
    assert evidence == {"winning_time_secs": 91.5, "off_time_actual": "12:03", "prize_currency": "EUR"}
    assert "winning_time_secs" not in database.load_from_database()


def test_raw_source_revisions_can_be_replayed_without_http(tmp_path, monkeypatch):
    scraper = data_scraper.SportingLifeScraper()
    scraper.raw_store_path = tmp_path/"raw.sqlite"
    url = "https://example.test/result"
    for value in (1, 2):
        page = '<script id="__NEXT_DATA__">'+json.dumps({"revision": value})+'</script>'
        monkeypatch.setattr(scraper, "_get", lambda url, p=page: SimpleNamespace(status_code=200, text=p, content=p.encode()))
        assert scraper._fetch_next_data(url)["revision"] == value
    scraper.offline = True
    monkeypatch.setattr(scraper, "_get", lambda *a: pytest.fail("HTTP during offline replay"))
    assert scraper._fetch_next_data(url) == {"revision": 2}
    with RawPayloadStore(scraper.raw_store_path).connection() as con:
        assert con.execute("SELECT COUNT(*) FROM payloads").fetchone()[0] == 2


def test_pedigree_failures_are_retryable_and_do_not_become_success_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path/"races.db"))
    database.insert_results(pd.DataFrame([{**r, "horse_id": str(i+1)} for i, r in enumerate(runners("a"))]))
    monkeypatch.setattr(pedigree, "fetch_horse_pedigree", lambda *a: None)
    assert pedigree.backfill_pedigree() == 0
    with database._get_conn() as con:
        assert con.execute("SELECT COUNT(*) FROM horse_pedigree").fetchone()[0] == 0
        con.execute("UPDATE pedigree_fetch_state SET next_retry_at='2000-01-01'")
    assert len(pedigree.pending_horse_ids()) == 2


@pytest.mark.parametrize("source", ["sample", "databsae"])
def test_non_real_sources_are_rejected_before_writing(tmp_path, monkeypatch, source):
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    canonical = tmp_path/"race_results.parquet"
    canonical.write_text("real history", encoding="utf-8")
    with pytest.raises(ValueError, match="sources from"):
        sources.refresh_sources(sources=(source,))
    assert canonical.read_text(encoding="utf-8") == "real history"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["race_results.parquet"]


def _fake_rtv(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    requested = []
    monkeypatch.setattr(sources.rtv, "RawStore", lambda *a: SimpleNamespace(close=lambda: None, day_status=lambda day: "complete"))
    monkeypatch.setattr(sources.rtv, "RTVClient", lambda: None)
    monkeypatch.setattr(sources.rtv, "scrape_days", lambda days, *a, **kw: requested.extend(days) or {d: {} for d in days})
    monkeypatch.setattr(sources.rtv, "build_tables", lambda *a, **kw: {"races": pd.DataFrame(
        columns=["race_id", "is_abandoned", "is_arab_race"]), "runners": pd.DataFrame()})
    monkeypatch.setattr(sources, "canonical_rows", lambda *a: pd.DataFrame())
    monkeypatch.setattr(sources, "_refresh_rtv_features", lambda *a, **kw: {"files": {}})
    return requested


def test_rtv_catchup_recovers_more_than_ten_days(tmp_path, monkeypatch):
    requested = _fake_rtv(tmp_path, monkeypatch)
    result = sources.refresh_rtv_history(15, today=date(2026, 11, 4))
    assert result["state"] == "ok"
    assert len(requested) == 15 and min(requested) == "2026-10-20"


def test_rtv_materialisation_failure_preserves_accepted_generation(tmp_path, monkeypatch):
    _fake_rtv(tmp_path, monkeypatch)
    pointer = tmp_path/"rtv_cache/pipeline/generation.json"
    before = {"history": {"days": {}}, "files": {"runs": "old/runs.parquet"}}
    atomic_json(pointer, before)
    def fail(*a, **kw):
        raise RuntimeError("interrupted before all files were ready")
    monkeypatch.setattr(sources, "_refresh_rtv_features", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        sources.refresh_rtv_history(1, today=date(2026, 11, 4))
    assert json.loads(pointer.read_text()) == before


def test_successful_rtv_dates_publish_while_failed_dates_keep_their_prior_view(tmp_path, monkeypatch):
    _fake_rtv(tmp_path, monkeypatch)
    monkeypatch.setattr(sources.rtv, "RawStore", lambda *a: SimpleNamespace(close=lambda: None,
        day_status=lambda day: "partial" if day == "2026-11-03" else "complete"))
    result = sources.refresh_rtv_history(2, today=date(2026, 11, 4))
    assert result["state"] == "partial" and result["pending_dates"] == ["2026-11-03"]
    accepted = sources._read(sources.pipeline_dir()/"history.json")["days"]
    assert list(accepted) == ["2026-11-02"]


def test_missing_rtv_listing_on_an_empty_canonical_day_is_recorded(tmp_path, monkeypatch):
    _fake_rtv(tmp_path, monkeypatch)
    monkeypatch.setattr(sources.rtv, "RawStore", lambda *a: SimpleNamespace(close=lambda: None, day_status=lambda day: "list_404"))
    monkeypatch.setattr(sources.rtv, "scrape_days", lambda *a, **kw: {})
    result = sources.refresh_rtv_history(1, today=date(2026, 11, 4))
    assert result["state"] == "ok"
    accepted = sources._read(sources.pipeline_dir()/"history.json")["days"]
    assert accepted["2026-11-03"]["collection"]["state"] == "confirmed_empty"


def test_empty_rtv_response_does_not_claim_a_known_canonical_race_is_complete(tmp_path, monkeypatch):
    _fake_rtv(tmp_path, monkeypatch)
    monkeypatch.setattr(sources, "canonical_rows", lambda *a: pd.DataFrame({"race_id": ["a"]}))
    result = sources.refresh_rtv_history(1, today=date(2026, 11, 4))
    assert result["state"] == "partial" and result["pending_dates"] == ["2026-11-03"]


def test_partial_daily_manifest_preserves_complete_base_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    root = sources.pipeline_dir()
    (root.parent/"tables/races").mkdir(parents=True)
    pd.DataFrame({"race_id": [1], "race_date": ["2026-10-29"]}).to_parquet(root.parent/"tables/races/2026-Q4.parquet")
    atomic_json(root/"history.json", {"days": {"2026-10-29": {"complete": False, "directory": "broken"}}})
    assert sources.load_history_table("races", "2026-10-29", "2026-10-29").race_id.tolist() == [1]


def test_old_range_parts_are_readable_and_overlap_does_not_duplicate_races(tmp_path):
    from src.ingestion.rtv_observations import observation_tables
    path = tmp_path/"tables/races"
    path.mkdir(parents=True)
    frame = pd.DataFrame({"race_id": [1], "race_date": ["2026-10-29"]})
    frame.to_parquet(path/"2026-Q4.parquet")
    frame.to_parquet(path/"2026-10-01_2026-10-29.parquet")
    assert observation_tables("2026-10-01", "2026-10-29", source_root=tmp_path, names=("races",))["races"].race_id.tolist() == [1]


def test_readers_pin_one_rtv_generation(tmp_path, monkeypatch):
    from src.ingestion import rtv_storage
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    pointer = tmp_path/"rtv_cache/pipeline/generation.json"
    atomic_json(pointer, {"files": {"runs": "first/runs.parquet", "cards": "first/cards.parquet"}})
    with rtv_storage.pin_generation():
        assert "first" in str(rtv_storage.native_path("runs"))
        atomic_json(pointer, {"files": {"runs": "second/runs.parquet", "cards": "second/cards.parquet"}})
        assert "first" in str(rtv_storage.native_path("cards"))
    assert "second" in str(rtv_storage.native_path("cards"))


def test_failed_source_is_returned_and_saved_as_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(results, "sync_database", lambda **kw: {"state": "ok"})
    def unavailable(**kw):
        raise RuntimeError("provider unavailable")
    monkeypatch.setattr(sources, "refresh_rtv_history", unavailable)
    report = sources.refresh_sources(1, sources=("sl", "rtv"))
    assert report["state"] == "partial" and report["sources"]["rtv"]["state"] == "error"
    assert json.loads((tmp_path/"source_refresh/status.json").read_text())["state"] == "partial"


def test_fixed_par_is_fitted_once_per_input_generation(tmp_path, monkeypatch):
    from src.ingestion import rtv_observations as collection
    races = pd.DataFrame({"race_id": [1], "win_time_s": [90.]})
    monkeypatch.setattr(collection, "observation_tables", lambda *a, **kw: {"races": races})
    fitted = []
    monkeypatch.setattr(collection.features, "fit_time_par", lambda frame: fitted.append(len(frame)) or {"test": True})
    assert collection.time_par(source_root=tmp_path) == {"test": True}
    assert collection.time_par(source_root=tmp_path) == {"test": True}
    assert fitted == [1]
    races.loc[0, "win_time_s"] = 91.
    collection.time_par(source_root=tmp_path)
    assert fitted == [1, 1]


def test_ordinary_currency_processing_does_not_fetch_references(tmp_path, monkeypatch):
    from src import prize_currency_features as currency
    path = tmp_path/"rates.csv"
    pd.DataFrame({"observation_date": ["2020-01-01"], "gbp_per_eur": [.8]}).to_csv(path, index=False)
    monkeypatch.setattr(currency, "_paths", lambda: (tmp_path/"units.json", path))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("Network in ordinary processing"))
    assert currency._load_rates().gbp_per_eur.tolist() == [.8]
    assert not hasattr(currency, "urlopen")


def test_exchange_rate_refresh_fetches_once_a_day_and_publishes(tmp_path, monkeypatch):
    import io
    from src.ingestion import fx
    path = tmp_path/"rates.csv"
    pd.DataFrame({"observation_date": ["2020-01-01"], "gbp_per_eur": [.8]}).to_csv(path, index=False)
    latest = (pd.Timestamp.now().normalize()-pd.Timedelta(days=1)).date().isoformat()
    document = (f'<e xmlns="x"><Cube><Cube time="2020-01-01"><Cube currency="GBP" rate="0.8"/></Cube>'
                f'<Cube time="{latest}"><Cube currency="USD" rate="1.1"/><Cube currency="GBP" rate="0.9"/></Cube>'
                f'</Cube></e>').encode()
    requests = []
    class Reply(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(fx, "reference_paths", lambda: (tmp_path/"units.json", path))
    monkeypatch.setattr(fx, "urlopen", lambda url, timeout: requests.append(url) or Reply(document))
    report = fx.refresh_prize_references()
    assert report == {"state": "ok", "latest_observation": latest, "checked_today": True}
    assert pd.read_csv(path).gbp_per_eur.tolist() == [.8, .9]
    assert fx.refresh_prize_references()["state"] == "ok"
    assert len(requests) == 1


def test_partial_materialisation_never_overwrites_accepted_files(tmp_path, monkeypatch):
    from src import rtv_features, rtv_text
    from src.ingestion import rtv_observations as collection
    from src.ingestion.rtv_storage import native_path
    monkeypatch.setattr(sources.config, "DATA_DIR", str(tmp_path))
    base = tmp_path/"rtv_cache"
    base.mkdir()
    values = pd.DataFrame({"race_id": ["a"], "horse_id": [1], "race_date": pd.to_datetime(["2026-10-29"]), "value": [1.]})
    values.to_parquet(base/"feature_runs.parquet", index=False)
    values.to_parquet(base/"feature_cards.parquet", index=False)
    text = pd.DataFrame({"race_id": ["a"], "horse_id": [1], "race_date": ["2026-10-29"], "tf_comment": ["old"], "tf_verdict": ["A"]})
    text.to_parquet(base/"feature_text.parquet", index=False)
    values.to_parquet(base/"feature_text_scores.parquet", index=False)
    monkeypatch.setattr(collection, "observation_tables", lambda *a, **kw: {
        name: pd.DataFrame() for name in ("races", "runners", "metrics", "sectionals", "quality_flags")})
    monkeypatch.setattr(collection, "canonical_rows", lambda *a, **kw: pd.DataFrame())
    monkeypatch.setattr(collection, "match_runners", lambda *a: pd.DataFrame())
    monkeypatch.setattr(collection, "time_par", lambda **kw: None)
    monkeypatch.setattr(rtv_features, "build_run_observations", lambda *a, **kw: values.assign(value=2.))
    monkeypatch.setattr(rtv_features, "build_card_observations", lambda *a: values.assign(value=2.))
    monkeypatch.setattr(rtv_text, "build_text_observations", lambda *a: text.assign(tf_comment="new"))
    monkeypatch.setattr(collection, "ensure_text_models", lambda *a, **kw: [])
    def fail(*a, **kw):
        raise RuntimeError("score materialisation interrupted")
    monkeypatch.setattr(collection, "refresh_text_scores", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        collection.refresh_run_observations("2026-10-29", "2026-10-29")
    assert pd.read_parquet(native_path("runs")).value.tolist() == [1.]
    assert pd.read_parquet(native_path("cards")).value.tolist() == [1.]
    assert pd.read_parquet(native_path("text")).tf_comment.tolist() == ["old"]
    def finish(start, end, *, destination, **kw):
        values.assign(value=2.).to_parquet(destination, index=False)
        return {"text_scores": 1}
    monkeypatch.setattr(collection, "refresh_text_scores", finish)
    collection.refresh_run_observations("2026-10-29", "2026-10-29")
    assert pd.read_parquet(native_path("runs")).value.tolist() == [2.]
    assert pd.read_parquet(native_path("text")).tf_comment.tolist() == ["new"]
