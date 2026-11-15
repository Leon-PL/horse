"""Publication, replay and cross-provider input-boundary regressions."""
import hashlib
import json
import sqlite3
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pytest

from src.source_contracts import TableContract, coverage_report, observation_metadata, resolve_runners
from src import source_dataset as datasets
from src.ingestion import publication


@pytest.fixture(scope="module")
def module_dataset():
    return datasets.current_manifest()


def test_module_fixtures_also_use_the_isolated_dataset(module_dataset, reference_dataset):
    assert module_dataset is reference_dataset


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.config, "DATA_DIR", str(tmp_path))
    with datasets.use_manifest(None):
        yield tmp_path


def canonical():
    return pd.DataFrame({"race_id": ["r1", "r1", "r2"], "horse_id": [1, 2, 1],
        "horse_name": ["Alpha", "Beta", "Alpha"], "race_date": ["2025-01-01", "2025-01-01", "2025-01-04"],
        "track": ["Kempton"]*3, "off_time": ["12:00"]*3})


def publish(value):
    return datasets.publish_source("test", {"runners": pd.DataFrame({"id": [1], "value": [value]})},
        contracts={"runners": TableContract("runner", ("id", "value"), ("id",))})


def _worker_read(manifest, root):
    datasets.config.DATA_DIR = root
    datasets.install_manifest(manifest)
    return datasets.read_table("test", "runners").value.tolist()


def test_publication_is_idempotent_and_old_manifests_replay(sources):
    first = publish(10)
    assert publish(10)["version"] == first["version"]
    manifest = datasets.save_manifest({"test": first})
    publish(20)
    with datasets.use_manifest(manifest):
        assert datasets.read_table("test", "runners").value.tolist() == [10]
    assert len([p for p in (sources/"sources/test").iterdir() if p.is_dir()]) == 2


def test_invalid_identity_cannot_replace_accepted_data(sources):
    first = publish(10)
    with pytest.raises(ValueError, match="duplicate native"):
        datasets.publish_source("test", {"runners": pd.DataFrame({"id": [1, 1], "value": [20, 30]})},
            contracts={"runners": TableContract("runner", ("id",), ("id",))})
    assert json.loads((sources/"sources/test/latest.json").read_text())["version"] == first["version"]
    assert not list((sources/"sources/test").glob(".stage-*"))


def test_published_foaling_dates_use_matched_explicit_profiles(sources, monkeypatch):
    from src import database
    from src.data_processor import apply_pedigree
    from src.ingestion.racing_tv import RawStore

    pedigree = pd.DataFrame({
        "horse_id": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "horse_name": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"],
        "sire": ["Sire"] * 8, "dam": ["Dam"] * 8, "damsire": ["Damsire"] * 8,
        "foaled": ["2020-01-01", "2021-03-07", "2019-05-11", "2020-01-01", "2020-01-01",
                   "2020-01-01", "2019-06-14", "2021-01-01"],
        "fetched_at": ["2026-01-02T12:00:00", "2026-01-01T12:00:00", "2026-01-01T12:00:00",
                       "2026-01-01T12:00:00", "2026-01-01T12:00:00", "2026-01-01T12:00:00",
                       "2026-01-02T08:00:00", "2026-01-01T12:00:00"],
    })
    raw = RawStore(str(sources / "rtv_cache" / "rtv_raw.sqlite"))
    try:
        for identifier, name, dam, foaled in [
            (100, "ALPHA", "Dam (IRE)", "2020-04-15"),
            (101, "Beta", "Dam", "2021-03-07"),
            (102, "Gamma", "Dam", "2019-05-12"),
            (103, "Delta", "Dam", "2020-01-01"),
            (104, "Epsilon", "Other Dam", "2020-04-17"),
            (105, "Zeta", "Dam", "2020-04-18"),
            (106, "Zeta", "Dam", "2020-04-19"),
            (107, "Theta", "Dam", "2022-03-01"),
        ]:
            payload = {"horse": {"id": identifier, "name": name, "sire_name": "Sire",
                                 "dam_name": dam, "foaling_date": foaled, "age_days": 1234,
                                 "slug": "/horse", "country_code": {"racing": "IRE"},
                                 "display_colors": ["Bay"], "sire_country_code": "gb"}}
            if identifier == 100:
                foreign = {"id": 99, "non_runner": False, "meeting_date": "2025-12-30",
                           "track": {"country_code": {"racing": "FRA"}}}
                payload["horse"]["performances"] = [foreign, foreign,
                    {**foreign, "id": 101, "non_runner": True},
                    {**foreign, "id": 102, "meeting_date": "2026-01-02"},
                    {**foreign, "id": 103, "meeting_date": "2026-01-04"}]
            raw.put(f"/horses/{identifier}", "horse", "2026-01-01", 200, json.dumps(payload).encode())
        raw.con.execute("UPDATE responses SET fetched_at='2026-01-02T12:00:00+00:00'")
        raw.con.commit()
    finally:
        raw.close()

    monkeypatch.setattr(database, "DB_PATH", str(sources/"races.db"))
    with sqlite3.connect(database.DB_PATH) as con:
        pedigree.to_sql("horse_pedigree", con, index=False)
    results = datasets.publish_source("sl", {"results": canonical()},
        contracts={"results": TableContract("canonical runner", ("race_id", "horse_id"), ("race_id", "horse_id"))})
    previous = datasets.save_manifest({"sl": results})
    manifest = publication.capture_pedigree_dataset()
    receipt = manifest["sources"]["pedigree"]
    assert manifest["sources"]["sl"] == previous["sources"]["sl"]
    with datasets.use_manifest(manifest):
        horses = datasets.read_table("pedigree", "horses").set_index("horse_id")
        assert horses.foaled.fillna("").tolist() == ["2020-04-15", "2021-03-07", "", "", "", "", "2019-06-14", ""]
        assert horses.foaling_date_source.fillna("").tolist() == [
            "racing_tv", "racing_tv", "conflict", "", "", "", "sporting_life", ""]
        assert horses.loc["1", "rtv_horse_id"] == "100"
        assert horses.loc["1", "rtv_foaled_observed_at"]
        assert horses.loc["1", "rtv_country_of_foaling"] == "IRE"
        assert horses.loc["1", "rtv_coat_colour"] == "Bay"
        assert horses.loc["1", "rtv_sire_country"] == "gb"
        assert horses.loc["3", "foaled_sporting_life"] == "2019-05-11"
        assert horses.loc["3", "foaled_rtv"] == "2019-05-12"
        assert apply_pedigree(canonical()).foaled.tolist() == ["2020-04-15", "2021-03-07", "2020-04-15"]
        assert horses.loc["1", "rtv_profile_overseas_starts"] == 1
        assert horses.loc["1", "rtv_profile_last_overseas_start"] == "2025-12-30"
        races = pd.DataFrame({"horse_id": [1, 1, 1, 1, 4, 7, 2, 3], "race_date": ["2026-01-02"]*8,
                              "off_time": ["12:04", "12:05", "12:07", "bad time", "14:00", "14:00", "10:00", "14:00"],
                              "prediction_as_of": [None, None, "2026-01-02T11:59:00Z", None, None, None, None, None]})
        features = apply_pedigree(races)
        assert features.profile_exact_age_days.notna().tolist() == [False, True, False, False, False, False, True, False]
        assert features.loc[1, "profile_exact_age_days"] == (pd.Timestamp("2026-01-02")-pd.Timestamp("2020-04-15")).days
        assert features.loc[1, "profile_foaling_day_of_year"] == 106
        assert features.loc[1, "rtv_profile_foaled_abroad"] == 0
        assert features.loc[1, "rtv_profile_known_overseas_starts"] == 1
        assert features.loc[1, "rtv_profile_days_since_overseas_start"] == 3
        assert features.loc[4, "rtv_profile_foaled_abroad"] == 0
        assert pd.isna(features.loc[4, "rtv_profile_known_overseas_starts"])
    assert receipt["provenance"]["foaling_coverage"] == {
        "racing_tv": 2, "sporting_life": 1, "conflict": 1, "unresolved": 4, "recovered_from_january_1": 1,
    }


def test_write_failure_keeps_pointer_and_prior_observations(sources, monkeypatch):
    first = publish(10)
    def interrupted(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(datasets, "atomic_parquet", interrupted)
    with pytest.raises(OSError, match="disk full"):
        publish(20)
    assert json.loads((sources/"sources/test/latest.json").read_text())["version"] == first["version"]


def test_missing_selected_source_stays_missing(sources):
    with datasets.use_manifest(datasets.save_manifest({})):
        assert not datasets.source_path("rtv", "runs").exists()
        assert datasets.read_table("rtv", "runs").empty
    with pytest.raises(RuntimeError, match="No source dataset is selected"):
        datasets.source_path("rtv", "runs")


def test_a_declared_file_cannot_disappear_after_selection(sources):
    manifest = datasets.save_manifest({"test": publish(10)})
    with datasets.use_manifest(manifest):
        path = datasets.source_path("test", "runners")
        path.unlink()
        with pytest.raises(datasets.SourceIntegrityError, match="missing its declared file"):
            datasets.read_table("test", "runners")
        with pytest.raises(datasets.SourceIntegrityError, match="missing its declared file"):
            datasets.source_path("test", "runners")


@pytest.mark.parametrize("damage", ["undeclared", "missing", "incomplete", "invalid_ids", "row_order"])
def test_model_rows_require_a_complete_accepted_crosswalk(sources, damage):
    frame = canonical().iloc[:1]
    receipt = publication._publish("rtv", {"cards": frame},
        {"cards": TableContract("runner", ())}, canonical())
    manifest = datasets.save_manifest({"rtv": receipt})
    with datasets.use_manifest(manifest):
        path = datasets.source_path("rtv", "links_cards")
        if damage == "undeclared":
            del manifest["sources"]["rtv"]["tables"]["links_cards"]
        elif damage == "missing":
            path.unlink()
        else:
            links = pd.read_parquet(path)
            if damage == "incomplete":
                links = links.drop(columns="match_status")
            elif damage == "invalid_ids":
                links["sl_horse_id"] = pd.NA
            else:
                links["source_row"] = 3
            links.to_parquet(path, index=False)
        with pytest.raises(datasets.SourceIntegrityError, match="crosswalk|runner links|declared file"):
            datasets.model_table("rtv", "cards")


@pytest.mark.parametrize("entry", [datasets.processing_inputs, datasets.dataset_build])
def test_external_registry_survives_frame_reentry(sources, monkeypatch, entry):
    root = datasets.registry_root()
    manifest = {**datasets.save_manifest({"test": publish(10)}), "root": str(root)}
    @entry
    def prepare(df):
        return df.copy()
    with datasets.use_manifest(manifest):
        frame = prepare(pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(datasets.config, "DATA_DIR", str(sources/"different_registry"))
    monkeypatch.setattr(datasets, "capture_dataset", lambda **kw: pytest.fail("Unexpected source publication"))
    @datasets.dataset_build
    def consume(df):
        assert datasets.source_path("test", "runners").is_relative_to(root)
        return datasets.read_table("test", "runners").value.tolist()
    assert consume(frame) == [10]
    assert datasets.current_manifest() is None
    # Root-aware re-entry still performs content verification.
    (root/"test"/manifest["sources"]["test"]["version"]/"runners.parquet").write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupt"):
        consume(frame)


def test_parent_and_spawned_workers_read_the_same_version(sources):
    manifest = datasets.save_manifest({"test": publish(10)})
    publish(20)
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        assert pool.submit(_worker_read, manifest, str(sources)).result(timeout=45) == [10]


def test_corrupt_accepted_version_is_detected_on_replay(sources):
    receipt = publish(10)
    path = sources/"sources/test"/receipt["version"]/"runners.parquet"
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupt"):
        datasets.verify_source(receipt)


def test_crosswalk_preserves_native_namespace_and_rejects_ambiguity():
    reference = canonical()
    reference.loc[1, "horse_name"] = "Alpha"
    native = reference.iloc[:1].drop(columns=["race_id", "horse_id"]).assign(atr_raceid=1, atr_horse_id=1)
    links = resolve_runners(native, reference, provider="atr")
    assert links.match_status.tolist() == ["ambiguous"]
    assert links.sl_horse_id.isna().all()
    assert links.source_runner_id.tolist() == ["1"]


def test_verified_ids_resolve_namesakes_but_conflicts_never_fall_back():
    reference = canonical()
    reference.loc[1, "horse_name"] = "Alpha"
    native = reference.iloc[:1].assign(sl_race_id="r1", sl_horse_id=2)
    links = resolve_runners(native, reference, provider="atr")
    assert links.sl_horse_id.tolist() == ["2"]
    native["sl_horse_id"] = 999
    links = resolve_runners(native, reference, provider="atr")
    assert links.match_status.tolist() == ["identity_conflict"]
    assert links.sl_horse_id.isna().all()


def test_actual_capture_and_assumed_publication_are_distinguished():
    frame = canonical().iloc[:1].assign(observed_at="2025-01-05T12:00:00Z")
    timing = observation_metadata(frame, TableContract("runner", (), delay_hours=48))
    assert timing.available_at.iloc[0] == pd.Timestamp("2025-01-03T12:00Z")
    assert timing.observed_at.iloc[0] == pd.Timestamp("2025-01-05T12:00Z")
    assert timing.availability_basis.tolist() == ["assumed_event_plus_48h"]
    snapshot = observation_metadata(frame, TableContract("runner snapshot", (), role="snapshot"))
    assert snapshot.available_at.iloc[0] == pd.Timestamp("2025-01-05T12:00Z")
    absent = observation_metadata(canonical().iloc[:1], TableContract("runner", ()))
    assert absent.observed_at.isna().all()


def test_coverage_distinguishes_population_links_and_availability():
    frame = canonical().iloc[:2]
    links = resolve_runners(frame, canonical(), provider="atr")
    timing = observation_metadata(frame, TableContract("runner", (), delay_hours=48), links)
    report = coverage_report(frame, links, eligible=2, total=3, metadata=timing, now="2025-01-02T00:00Z")
    assert report["eligible_coverage"] == 1
    assert report["all_runner_coverage"] == 2/3
    assert report["usable_rows"] == 0


def test_atr_promotion_preserves_history_and_skips_incomplete_units(sources, monkeypatch):
    from src.ingestion import attheraces as atr
    from src.ingestion import atr_rebuild
    from src.ingestion.atr_tabs import COLLECTION_VERSION, PERFORMANCE_TABS
    legacy = sources/"legacy.parquet"
    old = canonical().iloc[[2]].drop(columns=["race_id", "horse_id"]).assign(atr_raceid="old", atr_horse_id="native1", atr_finishing_speed_pct=99.)
    old.to_parquet(legacy)
    old_sha = hashlib.sha256(legacy.read_bytes()).hexdigest()
    monkeypatch.setattr(atr, "ATR_CACHE_FILE", str(legacy))
    root = sources/"worker"
    con = atr_rebuild.connect(root)
    for race_id in ("r1", "incomplete"):
        con.execute("INSERT INTO jobs(race_id,race_key,race_date,track,off_time,expected,state,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (race_id, race_id, "2025-01-01", "Kempton", "12:00", "[]", "data", "2025-01-03T13:00Z"))
        row = {**canonical().iloc[0].to_dict(), "sl_race_id": race_id, "sl_horse_id": 1,
               "atr_raceid": "new", "atr_horse_id": "native1", "atr_collection_version": COLLECTION_VERSION,
               "atr_finishing_speed_pct": 105.}
        con.execute("INSERT INTO runners VALUES (?,?,?)", (race_id, "alpha", json.dumps(row)))
        for kind in PERFORMANCE_TABS if race_id == "r1" else PERFORMANCE_TABS[:1]:
            con.execute("INSERT INTO tab_coverage VALUES (?,?,?,?,?)", (race_id, kind, "data", "{}", "2025-01-03"))
    con.commit()
    con.close()
    first = publication.publish_atr(canonical(), rebuild_root=root)
    assert publication.publish_atr(canonical(), rebuild_root=root)["version"] == first["version"]
    with datasets.use_manifest(datasets.save_manifest({"atr": first})):
        accepted = datasets.read_table("atr", "sectionals", linked=True)
    assert len(accepted) == 2
    assert set(accepted.atr_raceid) == {"old", "new"}
    assert accepted.match_status.eq("matched").all()
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == old_sha
    status = json.loads((sources/"sources/atr/promotion_status.json").read_text())
    assert status["rejected_units"] == [{"race_id": "incomplete", "reason": "incomplete_tab_contract"}]
    con = atr_rebuild.connect(root)
    con.execute("UPDATE jobs SET state='pending' WHERE race_id='r1'")
    con.commit()
    con.close()
    retained = publication.publish_atr(canonical(), rebuild_root=root)
    with datasets.use_manifest(datasets.save_manifest({"atr": retained})):
        assert len(datasets.read_table("atr", "sectionals")) == 2


def test_collection_failure_does_not_change_accepted_health(sources):
    receipt = publish(10)
    health = publication.source_health({"state": "error", "error": "provider unavailable"}, receipt)
    assert health["attempt_state"] == "error"
    assert health["readiness"] == "accepted"
    assert health["accepted_version"] == receipt["version"]


def test_failed_optional_adoption_preserves_accepted_version(sources):
    first = publish(10)
    def broken():
        raise ValueError("bad candidate")
    assert publication._adopt("test", broken)["version"] == first["version"]
    status = json.loads((sources/"sources/test/publication_status.json").read_text())
    assert status["state"] == "error" and status["retained_version"] == first["version"]


def test_missing_frozen_prices_never_fall_back_to_live_cache(sources, monkeypatch):
    from src import betfair, source_prices
    monkeypatch.setattr(betfair, "load_bsp", lambda *a: pytest.fail("Read live prices"))
    with datasets.use_manifest(datasets.save_manifest({})):
        result = source_prices.attach_bsp(canonical(), cols=("bsp",))
    assert result.bsp.isna().all()


def test_atr_availability_support_and_chunk_parity(sources):
    from src.atr_features import published_atr_rolling, _published_history
    reference = pd.DataFrame({"race_id": ["a", "b", "c", "d"], "horse_id": [1]*4,
        "horse_name": ["Alpha"]*4, "race_date": ["2025-01-01", "2025-01-02", "2025-01-04", "2025-01-06"],
        "track": ["Kempton"]*4, "off_time": ["12:00"]*4, "won": [0]*4, "lengths_behind": [1.]*4})
    raw = reference.iloc[:2].copy().assign(atr_raceid=["native-a", "native-b"], atr_horse_id="native1",
        atr_finishing_speed_pct=[110., 120.], atr_in_running_comment=[None, "led throughout"])
    sl = datasets.publish_source("sl", {"results": reference},
        contracts={"results": TableContract("canonical runner", ("race_id", "horse_id"), ("race_id", "horse_id"))})
    atr = publication._publish("atr", {"sectionals": raw}, {"sectionals": TableContract("runner", (), delay_hours=48)}, reference)
    manifest = datasets.save_manifest({"sl": sl, "atr": atr})
    _published_history.cache_clear()
    with datasets.use_manifest(manifest):
        whole = published_atr_rolling(reference)
        chunks = [published_atr_rolling(reference.iloc[:2]), published_atr_rolling(reference.iloc[2:])]
    import numpy as np
    assert np.isnan(whole["atr_finishing_speed_pct_avg_3"][1])
    assert whole["atr_finishing_speed_pct_avg_3"][2] == 115.
    assert whole["atr_observation_count_5"].tolist() == [0., 0., 2., 2.]
    assert whole["atr_days_since_observation"][3] == 4.
    for column in whole:
        np.testing.assert_allclose(whole[column], np.concatenate([chunk[column] for chunk in chunks]), equal_nan=True)


def test_expected_provider_lag_does_not_block_training():
    from src.source_health import consumer_readiness
    attempts = {"sl": {"state": "ok"}, "fx": {"state": "ok"},
                "betfair": {"state": "awaiting_publication"}, "atr": {"state": "partial", "reason": "Backfill"}}
    readiness = consumer_readiness(attempts, requested_end="2025-01-10")
    assert readiness["ready"]
    betfair = next(value for value in readiness["limitations"] if value["source"] == "betfair")
    assert betfair["expected_through"] == "2025-01-04"
    attempts["sl"]["state"] = "partial"
    assert not consumer_readiness(attempts)["ready"]


def test_distinct_snapshot_captures_link_to_the_same_runner():
    reference = canonical().iloc[:1]
    snapshots = pd.concat([reference, reference], ignore_index=True).assign(
        observed_at=["2025-01-01T10:00Z", "2025-01-01T11:00Z"])
    links = resolve_runners(snapshots, reference, provider="rtv_snapshots")
    assert links.match_status.tolist() == ["matched", "matched"]
    assert links.sl_horse_id.tolist() == ["1", "1"]


def test_link_or_availability_changes_invalidate_source_fingerprint(sources):
    from src.feature_inputs import atr_cache_signature
    reference = canonical().iloc[:1]
    raw = reference.drop(columns=["race_id", "horse_id"])
    first = publication._publish("atr", {"sectionals": raw}, {"sectionals": TableContract("runner", ())}, canonical())
    with datasets.use_manifest(datasets.save_manifest({"atr": first})):
        original = atr_cache_signature("2025-02-01")
        tables = {name: datasets.read_table("atr", name) for name in first["tables"]}
    tables["observations_sectionals"]["available_at"] = pd.Timestamp("2025-01-10T00:00Z")
    contracts = {name: TableContract(**entry["contract"]) for name, entry in first["tables"].items()}
    second = datasets.publish_source("atr", tables, contracts=contracts)
    with datasets.use_manifest(datasets.save_manifest({"atr": second})):
        assert atr_cache_signature("2025-02-01") != original


def test_manifest_prices_use_persisted_ids_for_namesakes(sources):
    from src.source_prices import attach_bsp
    reference = canonical().iloc[:2].assign(horse_name="Namesake")
    prices = pd.DataFrame({"sl_race_id": ["r1", "r1"], "sl_horse_id": ["1", "2"],
                           "match_status": ["matched", "matched"], "bsp": [4., 12.]})
    result = attach_bsp(reference, prices, cols=("bsp",))
    assert result.bsp.tolist() == [4., 12.]
    assert len(result) == len(reference)


def test_frame_id_selects_inputs_for_direct_feature_or_append_calls(sources, monkeypatch):
    manifest = datasets.save_manifest({"test": publish(10)})
    frame = pd.DataFrame({"value": [1]})
    frame.attrs["dataset_manifest_id"] = manifest["id"]
    publish(20)
    monkeypatch.setattr(datasets, "capture_dataset", lambda **kwargs: pytest.fail("Republished a frozen frame"))
    @datasets.dataset_build
    def consume(processed):
        return datasets.read_table("test", "runners").value.tolist()
    assert consume(frame) == [10]
    assert datasets.current_manifest() is None


def test_explicit_unversioned_frames_use_the_committed_dataset_without_collection(sources, monkeypatch):
    monkeypatch.setattr(datasets, "capture_dataset", lambda **kwargs: pytest.fail("Captured a research frame"))
    @datasets.dataset_build
    def consume(df):
        return datasets.current_manifest()
    with pytest.raises(RuntimeError, match="No committed source dataset"):
        consume(pd.DataFrame({"value": [1]}))
    manifest = datasets.save_manifest(committed_sources())
    assert consume(pd.DataFrame({"value": [1]}))["id"] == manifest["id"]
    assert datasets.current_manifest() is None


def committed_sources():
    contract = {"values": TableContract("reference", ("id",), ("id",))}
    return {provider: datasets.publish_source(provider, {"values": pd.DataFrame({"id": [1]})}, contracts=contract)
            for provider in ("sl", "fx")}


def test_selection_reads_a_complete_dataset_not_advancing_provider_pointers(sources, monkeypatch):
    accepted = committed_sources()
    first = datasets.save_manifest(accepted)
    changed = datasets.publish_source("sl", {"values": pd.DataFrame({"id": [2]})},
        contracts={"values": TableContract("reference", ("id",), ("id",))})
    monkeypatch.setattr(datasets, "capture_dataset", lambda **kwargs: pytest.fail("Build ran source adoption"))
    assert datasets.select_dataset()["id"] == first["id"]
    second = datasets.save_manifest({**accepted, "sl": changed})
    assert datasets.select_dataset()["id"] == second["id"]


def test_adoption_lock_preserves_nonblocking_reads_of_committed_data(sources, monkeypatch):
    from src.ingestion.storage import writer_lock
    manifest = datasets.save_manifest(committed_sources())
    monkeypatch.setattr(publication, "capture_sources", lambda **kwargs: pytest.fail("Concurrent adoption ran"))
    with writer_lock(datasets.registry_root()/".adoption.lock") as acquired:
        assert acquired
        assert datasets.select_dataset()["id"] == manifest["id"]
        with pytest.raises(RuntimeError, match="already running"):
            datasets.capture_dataset()


def test_retained_observations_are_relinked_after_canonical_namesake_repair(sources):
    reference = canonical().iloc[:1]
    raw = reference.drop(columns=["race_id", "horse_id"])
    publication._publish("atr", {"sectionals": raw}, {"sectionals": TableContract("runner", ())}, reference)
    repaired = pd.concat([reference, reference.assign(horse_id=2)], ignore_index=True)
    def bad_candidate():
        raise ValueError("invalid new source generation")
    receipt = publication._adopt("atr", bad_candidate, canonical=repaired)
    with datasets.use_manifest(datasets.save_manifest({"atr": receipt})):
        kept = datasets.read_table("atr", "sectionals", linked=True)
    assert len(kept) == 1
    assert kept.match_status.tolist() == ["ambiguous"]
    assert kept.sl_horse_id.isna().all()


def test_direct_append_without_a_committed_dataset_changes_nothing(sources):
    from src.feature_store import FeatureStore
    from src.store_refresh import append_current_store
    store = FeatureStore(str(sources/"store"))
    path = sources/"store/meta.json"
    path.write_text(json.dumps({"dataset_manifest": {"id": "prior-version"}}))
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="No committed source dataset"):
        append_current_store(canonical(), store=store)
    assert path.read_bytes() == original


def test_invalid_prices_do_not_inflate_clv_coverage(sources, publish_prices):
    from src.clv import compute_clv
    names = ["Alpha", "Bravo", "Charlie", "Delta"]
    publish_prices(names, ["4", "bad", "0", "inf"], off_time="12:00")
    bets = pd.DataFrame({"race_date": ["2025-01-01"]*4, "track": ["Ascot"]*4,
        "off_time": ["12:00"]*4, "horse_name": names, "odds": [5., 2., 3., 4.]})
    result = compute_clv(bets)
    assert result["n_matched"] == 1
    assert result["coverage"] == 0.25


def test_name_only_targets_cannot_bypass_published_price_quarantine(sources):
    from src.source_prices import attach_bsp
    target = canonical().iloc[:1].drop(columns=["race_id", "horse_id"])
    prices = target.assign(off_key="12:00", horse_key="alpha", bsp=4.,
        sl_race_id=pd.NA, sl_horse_id=pd.NA, match_status="ambiguous")
    result = attach_bsp(target, prices, cols=("bsp",))
    assert result.bsp.isna().all()


def test_legacy_price_targets_recover_only_unique_consistent_canonical_ids(sources):
    from src.source_prices import attach_bsp
    reference = canonical()
    reference.loc[1, "horse_name"] = "Alpha"
    reference.loc[2, "horse_name"] = "Beta"
    receipt = datasets.publish_source("sl", {"results": reference},
        contracts={"results": TableContract("canonical runner", ("race_id", "horse_id"), ("race_id", "horse_id"))})
    prices = pd.DataFrame({"sl_race_id": ["r1", "r2"], "sl_horse_id": [1, 1],
                           "match_status": ["matched", "matched"], "bsp": [4., 7.]})
    with datasets.use_manifest(datasets.save_manifest({"sl": receipt})):
        ambiguous = reference.iloc[:1].drop(columns="horse_id")
        assert attach_bsp(ambiguous, prices, cols=("bsp",)).bsp.isna().all()
        repeated = pd.concat([reference.iloc[[2]], reference.iloc[[2]]], ignore_index=True).drop(columns="horse_id")
        assert attach_bsp(repeated, prices, cols=("bsp",)).bsp.tolist() == [7., 7.]
        conflict = repeated.iloc[:1].assign(race_id="different-known-race")
        assert attach_bsp(conflict, prices, cols=("bsp",)).bsp.isna().all()
