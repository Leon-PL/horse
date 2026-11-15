"""Regression coverage for production identity and pre-race feature reuse."""
from types import SimpleNamespace
import json

import pandas as pd
import pytest

import config
from src import live_prediction as live
from src import run_store as registry


@pytest.fixture
def registry_dirs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(registry, "RUNS_DIR", str(runs))
    monkeypatch.setattr(registry, "PRODUCTION_POINTER", str(models / "production.json"))
    monkeypatch.setattr(config, "MODELS_DIR", str(models))
    # Fake runs carry no feature signature, so promoting one is an explicit,
    # unverified override (run_store.promote refuses them by default).
    for rid in ("20260101_120000", "20260201_120000", "20260301_120000"):
        (runs / rid).mkdir()
        (runs / rid / "model.joblib").write_bytes(rid.encode())
        (runs / rid / "meta.json").write_text(json.dumps({"run_id": rid}))
    return runs


def test_production_survives_direct_delete_and_pruning(registry_dirs):
    rid = "20260101_120000"
    assert registry.promote(rid, allow_feature_mismatch=True)
    assert registry.delete_run(rid) is False
    assert registry.prune_runs(keep_latest=1) == ["20260201_120000"]
    assert registry.get_production_run_id() == rid
    assert (registry_dirs / rid / "model.joblib").is_file()


def test_delete_cannot_escape_registry(registry_dirs):
    with pytest.raises(ValueError):
        registry.delete_run("..")


def test_model_revision_changes_on_promotion_and_replacement(registry_dirs):
    registry.promote("20260101_120000", allow_feature_mismatch=True)
    first = registry.production_model_revision()
    registry.promote("20260201_120000", allow_feature_mismatch=True)
    second = registry.production_model_revision()
    assert first != second
    (registry_dirs / second[0] / "model.joblib").write_bytes(b"replacement model")
    assert registry.production_model_revision() != second


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LOOKAHEAD_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(live, "feature_store_signature", lambda store=None: "history-v1")
    return tmp_path


def test_cache_requires_metadata_version_and_history_even_without_card_signature(cache_dir, monkeypatch):
    frame = pd.DataFrame({"horse_name": ["A"]})
    date = "2026-10-12"
    live.save_lookahead_cache(date, frame, "cards-v1")
    assert live.load_lookahead_cache(date) is not None
    assert live.load_lookahead_cache(date, "other-cards") is None
    monkeypatch.setattr(live, "feature_store_signature", lambda store=None: "history-v2")
    assert live.load_lookahead_cache(date) is None
    monkeypatch.setattr(live, "feature_store_signature", lambda store=None: "history-v1")
    meta_path = cache_dir / f"{date}.json"
    meta = json.loads(meta_path.read_text())
    meta["version"] = -1
    meta_path.write_text(json.dumps(meta))
    assert not live.lookahead_cache_valid(date)
    meta_path.write_text("incomplete json")
    assert not live.lookahead_cache_valid(date)
    meta_path.unlink()
    assert live.load_lookahead_cache(date) is None


def test_live_card_changes_recompute_all_race_features(cache_dir, monkeypatch):
    from src import feature_store, featured_dataset, data_processor
    calls = []

    class Store:
        def exists(self): return True
        def meta(self): return {"fe_signature": "current"}
        def last_date(self): return pd.Timestamp("2026-10-10")
        def featurize_live(self, card):
            calls.append(card.copy())
            result = card.copy()
            result["odds_rank"] = card.groupby("race_id")["odds"].rank()
            return result

    monkeypatch.setattr(feature_store, "FeatureStore", Store)
    monkeypatch.setattr(featured_dataset, "fe_signature", lambda: "current")
    monkeypatch.setattr(data_processor, "process_data", lambda df: df)
    monkeypatch.setattr(live, "_results_db_latest", lambda: pd.Timestamp("2026-10-10"))
    cards = pd.DataFrame({"race_id": ["r1", "r1"], "horse_name": ["A", "B"], "odds": [2.0, 4.0]})
    first = live.features_for_live_cards(cards, "2026-10-11")
    assert first["odds_rank"].tolist() == [1.0, 2.0]
    live.features_for_live_cards(cards, "2026-10-11")
    assert len(calls) == 1
    changed = cards.copy()
    changed["odds"] = [6.0, 2.0]
    assert live.features_for_live_cards(changed, "2026-10-11")["odds_rank"].tolist() == [2.0, 1.0]
    added = pd.concat([changed, pd.DataFrame({"race_id": ["r1"], "horse_name": ["C"], "odds": [3.0]})], ignore_index=True)
    assert len(live.features_for_live_cards(added, "2026-10-11")) == 3
    assert len(calls) == 3
    manual = changed.copy()
    manual["race_id"] = "custom"
    live.features_for_live_cards(manual, "2026-10-11", use_cache=False)
    assert len(calls) == 4
    # Manual what-if races must not overwrite the scheduled racecard cache.
    assert len(live.load_lookahead_cache("2026-10-11", live.cards_signature(added))) == 3
    with pytest.raises(ValueError, match="after the stored results"):
        live.features_for_live_cards(cards, "2026-10-10")




def test_cache_rejects_parquet_from_another_writer(cache_dir):
    date = "2026-10-12"
    live.save_lookahead_cache(date, pd.DataFrame({"horse_name": ["A"]}), "cards-A")
    assert live.load_lookahead_cache(date, "cards-A") is not None
    # A competing cache writer published a different field before its metadata.
    pd.DataFrame({"horse_name": ["B", "C"]}).to_parquet(cache_dir / f"{date}.parquet", index=False)
    assert live.load_lookahead_cache(date, "cards-A") is None


class _CurrentStore:
    def __init__(self, last="2026-10-10", signature="current"):
        self._last, self._sig = pd.Timestamp(last), signature
    def exists(self): return True
    def meta(self): return {"fe_signature": self._sig} if self._sig else {}
    def last_date(self): return self._last
    def featurize_live(self, card): return card.copy()


@pytest.fixture
def live_env(cache_dir, monkeypatch):
    from src import featured_dataset, data_processor
    monkeypatch.setattr(featured_dataset, "fe_signature", lambda: "current")
    monkeypatch.setattr(data_processor, "process_data", lambda df: df)
    return pd.DataFrame({"race_id": ["r1"], "horse_name": ["A"], "odds": [2.0]})


def test_store_behind_the_results_database_is_refused(live_env, monkeypatch):
    """The serving app never refreshes the store: a stale one priced horses
    that ran since on old form, silently."""
    monkeypatch.setattr(live, "_results_db_latest", lambda: pd.Timestamp("2026-10-12"))
    with pytest.raises(RuntimeError, match="Refresh the store"):
        live.features_for_live_cards(live_env, "2026-10-13", store=_CurrentStore("2026-10-10"),
                                     use_cache=False)


def test_history_ending_days_before_the_card_is_refused(live_env, monkeypatch):
    monkeypatch.setattr(live, "_results_db_latest", lambda: pd.Timestamp("2026-10-01"))
    with pytest.raises(RuntimeError, match="Sync the database"):
        live.features_for_live_cards(live_env, "2026-10-13", store=_CurrentStore("2026-10-01"),
                                     use_cache=False)


def test_christmas_gap_is_tolerated_and_unsigned_store_is_not(live_env, monkeypatch):
    monkeypatch.setattr(live, "_results_db_latest", lambda: pd.Timestamp("2025-12-22"))
    out = live.features_for_live_cards(live_env, "2025-12-26", store=_CurrentStore("2025-12-22"),
                                       use_cache=False)
    assert len(out) == 1
    with pytest.raises(RuntimeError, match="no feature signature"):
        live.features_for_live_cards(live_env, "2025-12-26",
                                     store=_CurrentStore("2025-12-22", signature=None), use_cache=False)


def test_fresh_scrape_and_cache_reload_return_identical_id_types(tmp_path, monkeypatch):
    """A fresh scrape held horse_id as text, the cache reload as integers; the
    store's carried history is keyed by integers (2026-10-13: 25 of 149 model
    features lost every horse's history on a freshly scraped card)."""
    from src.ingestion import sporting_life as ds
    monkeypatch.setattr(ds, "_racecard_cache_path", lambda date_str=None: str(tmp_path / "cards.csv"))
    monkeypatch.setattr(ds, "scrape_todays_racecards", lambda **kw: pd.DataFrame(
        {"race_id": ["9001", "9001"], "horse_id": ["1235263", "1238030"], "horse_name": ["A", "B"]}))
    fresh = ds.get_scraped_racecards(date_str="2026-10-14", force_refresh=True)
    cached = ds.get_scraped_racecards(date_str="2026-10-14")
    assert fresh["horse_id"].dtype == cached["horse_id"].dtype
    assert fresh["horse_id"].tolist() == [1235263, 1238030]


def test_live_features_receive_integer_horse_ids(live_env, monkeypatch):
    seen = {}

    class Store(_CurrentStore):
        def featurize_live(self, card):
            seen["dtype"] = card["horse_id"].dtype
            return card.copy()

    monkeypatch.setattr(live, "_results_db_latest", lambda: pd.Timestamp("2026-10-12"))
    card = live_env.assign(horse_id=["1235263"])
    live.features_for_live_cards(card, "2026-10-13", store=Store("2026-10-12"), use_cache=False)
    assert seen["dtype"] == "int64"
