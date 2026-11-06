import json
import pandas as pd

from src.ingestion.sporting_life import enrich_racecard_pedigree


class Profiles:
    def __init__(self):
        self.calls = 0
    def _fetch_next_data(self, url):
        self.calls += 1
        return {"props": {"pageProps": {"profile": {
            "horse_reference": {"id": 123}, "name": "French Visitor", "age": 99,
            "current_rating": 999, "horse_lifetime_stats": [999],
            "sire": {"name": "Known Sire"}, "dam": {"name": "Known Dam"},
            "damsire": {"name": "Known Damsire"}}}}}


def cards():
    return pd.DataFrame({"horse_id": [123], "horse_name": ["French Visitor"],
                         "race_date": ["2026-10-28"], "age": [5], "official_rating": [130]})


def test_first_local_runner_receives_immutable_pedigree_before_any_result(tmp_path):
    source, scraper = cards(), Profiles()
    original = source.copy()
    out = enrich_racecard_pedigree(source, scraper=scraper, cache_dir=tmp_path, known=pd.DataFrame())
    assert out.sire.tolist() == ["Known Sire"]
    assert out.age.tolist() == [5] and out.official_rating.tolist() == [130]
    assert "horse_lifetime_stats" not in out and "current_rating" not in out
    pd.testing.assert_frame_equal(source, original)
    cached = json.loads((tmp_path / "123.json").read_text())
    assert not {"age", "current_rating", "horse_lifetime_stats", "foaled"}.intersection(cached)
    again = enrich_racecard_pedigree(source, scraper=scraper, cache_dir=tmp_path, known=pd.DataFrame())
    pd.testing.assert_frame_equal(out, again)
    assert scraper.calls == 1


def test_existing_pedigree_needs_no_network_and_preserves_integer_ids(tmp_path):
    known = pd.DataFrame({"horse_id": ["123"], "sire": ["S"], "dam": ["D"], "damsire": ["DS"]})
    scraper = Profiles()
    out = enrich_racecard_pedigree(cards(), scraper=scraper, cache_dir=tmp_path, known=known)
    assert out.sire.tolist() == ["S"] and scraper.calls == 0
    assert out.horse_id.dtype == cards().horse_id.dtype


def test_bad_identity_cache_is_not_used_for_another_horse(tmp_path):
    (tmp_path / "123.json").write_text(json.dumps({"horse_id": "999", "horse_name": "French Visitor", "sire": "Wrong Sire"}))
    scraper = Profiles()
    out = enrich_racecard_pedigree(cards(), scraper=scraper, cache_dir=tmp_path, known=pd.DataFrame())
    assert out.sire.tolist() == ["Known Sire"] and scraper.calls == 1


def test_failed_profile_lookup_stays_unknown_without_changing_card(tmp_path):
    class Unavailable:
        def _fetch_next_data(self, url):
            return None
    out = enrich_racecard_pedigree(cards(), scraper=Unavailable(), cache_dir=tmp_path, known=pd.DataFrame())
    assert out[["sire", "dam", "damsire"]].isna().all().all()
    assert list(tmp_path.iterdir()) == []


def test_parallel_lookahead_cards_can_cache_the_same_horse(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(2)
    class ConcurrentProfiles(Profiles):
        def _fetch_next_data(self, url):
            barrier.wait(timeout=10)
            return super()._fetch_next_data(url)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(enrich_racecard_pedigree, cards(), scraper=ConcurrentProfiles(),
                               cache_dir=tmp_path, known=pd.DataFrame()) for _ in range(2)]
        outputs = [f.result() for f in futures]
    pd.testing.assert_frame_equal(outputs[0], outputs[1])
    assert json.loads((tmp_path / "123.json").read_text())["sire"] == "Known Sire"
    assert not list(tmp_path.glob("*.tmp"))


def test_windows_cache_replacement_retries_a_temporary_sharing_failure(monkeypatch, tmp_path):
    from pathlib import Path
    import src.ingestion.sporting_life as scraper_module
    original = Path.replace
    calls = []
    def replace(path, target):
        if target == tmp_path / "123.json":
            calls.append(target)
            if len(calls) == 1:
                raise PermissionError("Temporary Windows sharing conflict")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(scraper_module.time, "sleep", lambda seconds: None)
    result = enrich_racecard_pedigree(cards(), scraper=Profiles(), cache_dir=tmp_path, known=pd.DataFrame())
    assert result.sire.tolist() == ["Known Sire"]
    assert len(calls) == 2 and (tmp_path / "123.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_unavailable_optional_cache_keeps_verified_parents(monkeypatch, tmp_path):
    from pathlib import Path
    import src.ingestion.sporting_life as scraper_module
    def locked(path, target):
        raise PermissionError("Cache remains locked")
    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(scraper_module.time, "sleep", lambda seconds: None)
    result = enrich_racecard_pedigree(cards(), scraper=Profiles(), cache_dir=tmp_path, known=pd.DataFrame())
    assert result.sire.tolist() == ["Known Sire"]
    assert not list(tmp_path.iterdir())
