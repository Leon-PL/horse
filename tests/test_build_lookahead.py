"""scripts/build_lookahead.py prunes stale date-stamped caches before building.

The removed picks app ran ``cleanup_stale_caches`` on start-up; the lookahead
build now owns that housekeeping. These tests use temporary cache folders only.
"""
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

import config
from scripts import build_lookahead


@pytest.fixture
def caches(tmp_path, monkeypatch):
    processed, data = tmp_path / "processed", tmp_path / "data"
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", str(processed))
    monkeypatch.setattr(config, "DATA_DIR", str(data))
    monkeypatch.setattr(config, "CACHE_TTL_DAYS", 30)
    cards = data / "racecards_cache"
    cards.mkdir(parents=True)
    old = (date.today() - timedelta(days=45)).isoformat()
    recent = (date.today() - timedelta(days=2)).isoformat()
    (cards / f"racecards_{old}.csv").write_text("old")
    (cards / f"racecards_{recent}.csv").write_text("recent")
    (cards / "notes.txt").write_text("undated entries are never touched")
    built = []

    def build(store, day):
        # A build for an old --date writes an old-dated entry; it must survive.
        (cards / f"racecards_{day}.csv").write_text("built")
        built.append(day)
        return 1

    monkeypatch.setattr(build_lookahead, "FeatureStore", lambda: SimpleNamespace(exists=lambda: True))
    monkeypatch.setattr(build_lookahead, "build_for_date", build)
    return SimpleNamespace(cards=cards, old=old, recent=recent, built=built)


def _names(folder):
    return sorted(p.name for p in folder.iterdir())


def test_stale_entries_are_removed_before_the_build(caches):
    build_lookahead.main(["--date", caches.old])
    assert caches.built == [caches.old]
    assert _names(caches.cards) == sorted(
        [f"racecards_{caches.old}.csv", f"racecards_{caches.recent}.csv", "notes.txt"])
    assert (caches.cards / f"racecards_{caches.old}.csv").read_text() == "built"


def test_no_cleanup_keeps_old_entries(caches):
    today = date.today().isoformat()
    build_lookahead.main(["--date", today, "--no-cleanup"])
    assert (caches.cards / f"racecards_{caches.old}.csv").read_text() == "old"
    assert caches.built == [today]


def test_cleanup_failure_does_not_block_the_build(caches, monkeypatch):
    def broken():
        raise PermissionError("cache folder locked")
    monkeypatch.setattr(build_lookahead, "cleanup_stale_caches", broken)
    today = date.today().isoformat()
    build_lookahead.main(["--date", today])
    assert caches.built == [today]
    assert (caches.cards / f"racecards_{caches.old}.csv").exists()
