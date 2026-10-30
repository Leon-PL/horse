"""Tests for the single build-path seam (src/featured_dataset.py).

Covers the data fingerprint and the build-vs-append decision. FeatureStore is
faked so we exercise the routing logic without a real (slow) FE build.
"""

from contextlib import nullcontext

import pandas as pd
import pytest

import config
import src.featured_dataset as fd
from src.featured_dataset import data_signature, fe_signature, get_featured_dataset


def _toy(dates, draws=None):
    # Per-row values must NOT depend on the total length, or appending later
    # rows would spuriously change the historical region's fingerprint.
    n = len(dates)
    return pd.DataFrame({
        "race_date": pd.to_datetime(dates),
        "draw": draws if draws is not None else [(i % 8) + 1 for i in range(n)],
        "won": [1 if i % 8 == 0 else 0 for i in range(n)],
        "finish_position": [(i % 8) + 1 for i in range(n)],
        "num_runners": [8] * n,
    })


class TestDataSignature:
    def test_stable_when_unchanged(self):
        df = _toy(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
        assert data_signature(df) == data_signature(df.copy())

    def test_changes_on_in_place_value_edit(self):
        # The draw backfill edits draw in place without changing row count.
        df = _toy(["2023-01-01", "2023-01-02"], draws=[1, 2])
        edited = _toy(["2023-01-01", "2023-01-02"], draws=[7, 9])
        assert data_signature(df) != data_signature(edited)

    def test_changes_on_row_removal(self):
        # The non-runner cleanup deletes rows.
        full = _toy(["2023-01-01", "2023-01-02", "2023-01-03"])
        fewer = full.iloc[:2]
        assert data_signature(full) != data_signature(fewer)

    def test_cutoff_ignores_later_rows(self):
        hist = _toy(["2023-01-01", "2023-01-02"])
        plus_new = _toy(["2023-01-01", "2023-01-02", "2023-02-01", "2023-02-02"])
        cut = pd.Timestamp("2023-01-31")
        # The historical region (<= cutoff) is unchanged by appending later rows.
        assert data_signature(hist, cutoff=cut) == data_signature(plus_new, cutoff=cut)


class _FakeStore:
    def __init__(self, *, exists, meta, last_date):
        self._exists, self._meta, self._last = exists, meta, last_date
        self.meta_path = "(fake)"
        self.calls: list[str] = []

    def _access(self):
        return nullcontext()

    def exists(self):
        return self._exists

    def meta(self):
        return self._meta

    def last_date(self):
        return self._last

    def build(self, processed):
        self.calls.append("build")
        return pd.DataFrame({"featured": [1]})

    def append(self, processed):
        self.calls.append("append")

    def load_featured(self, columns=None):
        self.calls.append("load")
        return pd.DataFrame({"featured": [2]})

    def featured_schema_names(self):
        return ["featured"]


def _no_rtv_cache(cutoff):
    return {"cutoff": str(pd.Timestamp(cutoff).date()), "sig": {"runs": None, "pre": None}}


@pytest.fixture
def patch_store(monkeypatch):
    """Install a fake store, a no-op _stamp and an absent RTV cache.

    Without the RTV stub the decision reads the real cache under data/, so
    these tests passed on a fresh checkout and failed wherever RTV data exists.
    """
    holder = {}

    def _install(*, exists, meta, last_date):
        fake = _FakeStore(exists=exists, meta=meta, last_date=last_date)
        holder["fake"] = fake
        monkeypatch.setattr(fd, "FeatureStore", lambda: fake)
        monkeypatch.setattr(fd, "_stamp", lambda *a, **k: None)
        monkeypatch.setattr(fd, "_rtv_signature", _no_rtv_cache)
        return fake

    return _install


class TestBuildVsAppend:
    def test_cold_store_builds(self, patch_store):
        fake = patch_store(exists=False, meta={}, last_date=None)
        get_featured_dataset(processed=_toy(["2023-01-01", "2023-01-02"]))
        assert fake.calls == ["build"]

    def test_rebuild_flag_builds(self, patch_store):
        fake = patch_store(exists=True, meta={}, last_date=pd.Timestamp("2023-01-02"))
        get_featured_dataset(processed=_toy(["2023-01-01", "2023-01-02"]), rebuild=True)
        assert fake.calls == ["build"]

    def test_code_change_builds(self, patch_store):
        proc = _toy(["2023-01-01", "2023-01-02"])
        last = proc["race_date"].max()
        fake = patch_store(
            exists=True, last_date=last,
            meta={"fe_signature": "STALE", "data_signature": data_signature(proc, cutoff=last)},
        )
        get_featured_dataset(processed=proc)
        assert fake.calls == ["build"]

    def test_data_change_builds(self, patch_store):
        proc = _toy(["2023-01-01", "2023-01-02"], draws=[7, 9])
        last = proc["race_date"].max()
        fake = patch_store(
            exists=True, last_date=last,
            meta={"fe_signature": fe_signature(), "data_signature": "OLD-DATA"},
        )
        get_featured_dataset(processed=proc)
        assert fake.calls == ["build"]

    def test_current_store_appends(self, patch_store):
        proc = _toy(["2023-01-01", "2023-01-02"])
        last = proc["race_date"].max()
        fake = patch_store(
            exists=True, last_date=last,
            meta={"fe_signature": fe_signature(),
                  "data_signature": data_signature(proc, cutoff=last)},
        )
        get_featured_dataset(processed=proc)
        assert fake.calls == ["append", "load"]

    def test_sectional_backfill_beneath_the_store_builds(self, patch_store, monkeypatch):
        """The ATR cache is read inside FE, outside fe/data_signature (ledger 13)."""
        proc = _toy(["2023-01-01", "2023-01-02"])
        last = proc["race_date"].max()
        meta = {"fe_signature": fe_signature(), "data_signature": data_signature(proc, cutoff=last),
                "atr_signature": {"cutoff": "2023-01-02", "sig": "before-backfill"}}
        monkeypatch.setattr(fd, "_atr_signature",
                            lambda cutoff: {"cutoff": str(cutoff), "sig": "after-backfill"})
        fake = patch_store(exists=True, last_date=last, meta=meta)
        get_featured_dataset(processed=proc)
        assert fake.calls == ["build"]

        monkeypatch.setattr(fd, "_atr_signature",
                            lambda cutoff: {"cutoff": str(cutoff), "sig": "before-backfill"})
        fake = patch_store(exists=True, last_date=last, meta=meta)
        get_featured_dataset(processed=proc)
        assert fake.calls == ["append", "load"]

    def test_rtv_cache_change_beneath_the_store_builds(self, patch_store, monkeypatch):
        """The RTV observation cache is read inside FE, outside fe/data_signature."""
        proc = _toy(["2023-01-01", "2023-01-02"])
        last = proc["race_date"].max()
        stamped = {"cutoff": "2023-01-02", "sig": {"runs": "before", "pre": None}}
        meta = {"fe_signature": fe_signature(), "data_signature": data_signature(proc, cutoff=last),
                "rtv_signature": stamped}

        fake = patch_store(exists=True, last_date=last, meta=meta)
        monkeypatch.setattr(fd, "_rtv_signature",
                            lambda cutoff: {"cutoff": "2023-01-02", "sig": {"runs": "after", "pre": None}})
        get_featured_dataset(processed=proc)
        assert fake.calls == ["build"]

        fake = patch_store(exists=True, last_date=last, meta=meta)
        monkeypatch.setattr(fd, "_rtv_signature", lambda cutoff: stamped)
        get_featured_dataset(processed=proc)
        assert fake.calls == ["append", "load"]

        # A store stamped before the RTV cache existed must rebuild once it appears.
        unstamped = {k: v for k, v in meta.items() if k != "rtv_signature"}
        fake = patch_store(exists=True, last_date=last, meta=unstamped)
        monkeypatch.setattr(fd, "_rtv_signature", lambda cutoff: stamped)
        get_featured_dataset(processed=proc)
        assert fake.calls == ["build"]


@pytest.mark.parametrize("column,value", [
    ("odds", 99.0), ("official_rating", 120.0), ("sire", "Different sire"),
    ("atr_early_pace", 55.0), ("wind_surgery", 1),
])
def test_signature_detects_all_feature_input_edits(column, value):
    df = _toy(["2023-01-01", "2023-01-02"])
    df[column] = "Original" if isinstance(value, str) else 0.0
    edited = df.copy()
    edited.loc[0, column] = value
    assert data_signature(df) != data_signature(edited)


def test_signature_detects_result_swaps_with_unchanged_totals():
    df = _toy(["2023-01-01", "2023-01-02"])
    edited = df.copy()
    edited["finish_position"] = edited["finish_position"].iloc[::-1].to_numpy()
    assert data_signature(df) != data_signature(edited)


def test_signature_ignores_future_only_one_hot_categories_and_numeric_promotion():
    hist = _toy(["2023-01-01", "2023-01-02"])
    hist["going"] = "Good"
    hist["going_Good"] = 1
    future_schema = hist.copy()
    future_schema["going_Heavy"] = 0
    future_schema["draw"] = future_schema["draw"].astype(float)
    assert data_signature(hist) == data_signature(future_schema)


@pytest.mark.parametrize("relative_path", [
    "src/data_processor.py", "src/fe_incremental.py", "src/atr_features.py",
    "src/track_config.py", "src/pedigree_backfill.py",
])
def test_signature_tracks_feature_dependencies(tmp_path, monkeypatch, relative_path):
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# first revision", encoding="utf-8")
    before = fe_signature()
    path.write_text("# corrected feature", encoding="utf-8")
    assert fe_signature() != before
