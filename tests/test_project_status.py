"""The status report must not claim the pipeline is servable when it is not.

This report is the thing a human reads before trusting a pick, so a false
"ready" is worse than no report at all.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def status():
    spec = importlib.util.spec_from_file_location(
        "project_status", ROOT / "scripts" / "project_status.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["project_status"] = module
    spec.loader.exec_module(module)
    return module


def _sections(status, **overrides):
    """A fully healthy report, with the named sections replaced."""
    healthy = {
        "database": {"state": status.OK, "detail": "fresh"},
        "feature_store": {"state": status.OK, "detail": "fresh"},
        "training_runs": {"state": status.OK, "detail": "runs"},
        "production_model": {"state": status.OK, "detail": "promoted"},
    }
    healthy.update(overrides)
    return healthy


def _collect_with(status, monkeypatch, sections):
    monkeypatch.setattr(status, "_database", lambda: sections["database"])
    monkeypatch.setattr(status, "_feature_store", lambda: sections["feature_store"])
    monkeypatch.setattr(status, "_runs", lambda: sections["training_runs"])
    monkeypatch.setattr(status, "_production", lambda: sections["production_model"])
    monkeypatch.setattr(status, "_source_refresh", lambda: {"state": status.OK, "detail": "sources current"})
    return status.collect()


def test_all_links_healthy_is_servable(status, monkeypatch):
    report = _collect_with(status, monkeypatch, _sections(status))
    assert report["servable"] is True
    assert report["blockers"] == []
    assert "ready" in status.render(report)


@pytest.mark.parametrize(
    "broken",
    ["database", "feature_store", "training_runs", "production_model"],
)
def test_any_stale_link_blocks_serving(status, monkeypatch, broken):
    sections = _sections(status, **{broken: {"state": status.BAD, "detail": "stale"}})
    report = _collect_with(status, monkeypatch, sections)
    assert report["servable"] is False, f"{broken} being stale must block"
    assert any(b.startswith(broken) for b in report["blockers"])
    assert "not ready" in status.render(report)


def test_warnings_do_not_block(status, monkeypatch):
    """A few days behind is worth flagging but should not read as unusable."""
    sections = _sections(status, database={"state": status.WARN, "detail": "3 days behind"})
    report = _collect_with(status, monkeypatch, sections)
    assert report["servable"] is True
    assert report["blockers"] == []


@pytest.mark.parametrize("probe", ["_database", "_feature_store", "_runs", "_production"])
def test_a_failing_probe_blocks_readiness_without_crashing(status, monkeypatch, probe):
    for name, section in zip(
        ["_database", "_feature_store", "_runs", "_production"], _sections(status).values(),
    ):
        monkeypatch.setattr(status, name, lambda section=section: section)

    def boom():
        raise RuntimeError("disk gone")

    monkeypatch.setattr(status, probe, boom)
    report = status.collect()
    assert report["servable"] is False
    assert any("disk gone" in message for message in report["blockers"])
    assert "not ready" in status.render(report)


def test_stale_results_are_reported_as_blocking(status):
    """Form features roll over recent runs, so months-old results are unusable."""
    section = status._database()
    behind = section.get("days_behind")
    if behind is None:
        pytest.skip("no results database available")
    expected = (
        status.OK if behind <= 2
        else status.WARN if behind <= status.FRESH_DATA_DAYS
        else status.BAD
    )
    if section.get("recent_pending_races") or section.get("recent_unverified_dates"):
        expected = status.BAD
    assert section["state"] == expected


def test_render_is_console_safe(status, monkeypatch):
    """The report is read in a Windows terminal; keep it encodable there."""
    report = _collect_with(status, monkeypatch, _sections(status))
    status.render(report).encode("cp1252")
    broken = _collect_with(
        status, monkeypatch,
        _sections(status, feature_store={"state": status.BAD, "detail": "rebuild"}),
    )
    status.render(broken).encode("cp1252")


class TestStoreFindings:
    """The FE signature alone missed three ways a store goes stale."""

    META = {"fe_signature": "fe1", "data_signature": "d1", "last_date": "2026-10-12",
            "atr_signature": {"cutoff": "2026-10-12", "sig": "a1"}}

    def test_everything_current_is_ok(self, status):
        state, notes = status.store_findings(self.META, live_fe_sig="fe1", db_latest="2026-10-12",
                                             live_data_sig="d1", atr_now="a1")
        assert state == status.OK, notes

    def test_backfill_beneath_the_store_is_stale(self, status):
        state, notes = status.store_findings(self.META, live_fe_sig="fe1", live_data_sig="d2")
        assert state == status.BAD and any("data_signature" in n for n in notes)

    def test_results_newer_than_the_store_are_stale(self, status):
        state, notes = status.store_findings(self.META, live_fe_sig="fe1", db_latest="2026-10-13",
                                             live_data_sig="d1", atr_now="a1")
        assert state == status.BAD and any("not yet in the store" in n for n in notes)

    def test_sectional_backfill_is_stale(self, status):
        state, notes = status.store_findings(self.META, live_fe_sig="fe1", live_data_sig="d1",
                                             atr_now="a2")
        assert state == status.BAD and any("ATR" in n for n in notes)

    def test_unstamped_store_blocks_readiness(self, status):
        meta = {k: v for k, v in self.META.items() if k != "atr_signature"}
        state, _ = status.store_findings(meta, live_fe_sig="fe1", live_data_sig="d1")
        assert state == status.BAD

    @pytest.mark.parametrize("cutoff", [None, "", "not-a-date", "2026-10-11", "2026-10-13"])
    def test_atr_stamp_requires_store_cutoff(self, status, cutoff):
        meta = {**self.META, "atr_signature": {"sig": "a1", "cutoff": cutoff}}
        state, notes = status.store_findings(meta, live_fe_sig="fe1", live_data_sig="d1")
        assert state == status.BAD
        assert any("sectional cache unverified" in note for note in notes)

    @pytest.mark.parametrize("missing", ["data_signature", "last_date"])
    def test_incomplete_provenance_blocks_even_quick_check(self, status, missing):
        meta = {k: v for k, v in self.META.items() if k != missing}
        state, _ = status.store_findings(meta, live_fe_sig="fe1")
        assert state == status.BAD


def test_unknown_model_generation_is_not_ready(status, monkeypatch):
    import src.run_store as run_store

    monkeypatch.setattr(run_store, "feature_generation_status", lambda: {
        "state": run_store.GENERATION_UNKNOWN,
        "run_id": "old-run", "message": "feature generation unknown",
    })
    assert status._production()["state"] == status.BAD


def test_production_reports_parameter_cutoff_from_fitting_split(status, monkeypatch):
    import src.run_store as run_store

    monkeypatch.setattr(run_store, "feature_generation_status", lambda: {
        "state": run_store.GENERATION_MATCH, "run_id": "new-artifact",
        "message": "Feature generation matches.",
    })
    monkeypatch.setattr(run_store, "load_run_meta", lambda rid: {
        "timestamp": "2026-10-25T19:30:16",
        "data_last_date": "2026-10-12",
        "training_config": {"split": {"training_last_date": "2026-07-08"}},
    })
    section = status._production()
    assert section["state"] == status.OK
    assert section["training_last_date"] == "2026-07-08"
    assert section["parameter_age_days"] == status._days_behind("2026-07-08")
    assert "fitted through 2026-07-08" in section["detail"]
    assert "2026-10-12" not in section["detail"]


@pytest.mark.parametrize("training", [None, {}, {"split": None},
    {"split": {"training_last_date": "bad-date"}}])
def test_production_never_infers_missing_fit_cutoff(status, monkeypatch, training):
    import src.run_store as run_store

    monkeypatch.setattr(run_store, "feature_generation_status", lambda: {
        "state": run_store.GENERATION_MATCH, "run_id": "legacy",
        "message": "Feature generation matches.",
    })
    monkeypatch.setattr(run_store, "load_run_meta", lambda rid: {
        "timestamp": "2026-10-25T19:30:16", "training_config": training,
    })
    section = status._production()
    assert section["training_last_date"] is None
    assert section["parameter_age_days"] is None
    assert "cutoff unknown" in section["detail"]
    assert section["state"] == status.OK  # No invented age-based readiness rule.


def test_status_command_returns_failure_when_check_fails(status, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["project_status.py"])
    report = {"checked_at": "today", "sections": {}, "servable": False,
              "blockers": ["feature_store: check failed"]}
    monkeypatch.setattr(status, "collect", lambda **kwargs: report)
    assert status.main() == 1


@pytest.mark.parametrize("problem", [None, "subset", "old_range", "partial", "running", "stale", "skipped"])
def test_source_readiness_requires_all_sources_in_a_fresh_current_report(status, tmp_path, monkeypatch, problem):
    from datetime import datetime, timedelta, timezone
    import json
    from src.ingestion.pipeline import DEFAULT_SOURCES
    monkeypatch.setattr(status.config, "DATA_DIR", str(tmp_path))
    now = datetime.now(timezone.utc)
    report = {"state": "ok", "checked_at": now.isoformat(),
              "end": (status.date.today()-timedelta(days=1)).isoformat(),
              "sources": {name: {"state": "ok"} for name in DEFAULT_SOURCES}}
    if problem == "subset":
        report["sources"] = {"sl": {"state": "ok"}}
    elif problem == "old_range":
        report["end"] = "2020-01-01"
    elif problem in {"partial", "running"}:
        report["state"] = problem
    elif problem == "stale":
        report["checked_at"] = (now-timedelta(days=2)).isoformat()
    elif problem == "skipped":
        report["sources"]["atr"]["state"] = "skipped"
    path = tmp_path/"source_refresh/status.json"
    path.parent.mkdir()
    path.write_text(json.dumps(report), encoding="utf-8")
    assert status._source_refresh()["state"] == (status.OK if problem is None else status.BAD)


def test_deep_probe_fingerprints_database_even_if_snapshot_is_stale(status, monkeypatch, selected_dataset):
    import pandas as pd
    from src import database, data_processor, feature_inputs, feature_store, featured_dataset, source_dataset

    live = pd.DataFrame({name: [None] for name in database.RESULTS_SNAPSHOT_COLUMNS})
    live = live.assign(race_id="123", horse_id="456", jockey="")
    captured = {}

    class Store:
        def exists(self):
            return True

        def meta(self):
            return TestStoreFindings.META.copy()

    def process(df=None):
        assert df is not None, "A default call would fingerprint the old snapshot"
        assert df.race_id.tolist() == [123]  # IDs are declared numeric.
        assert df.horse_id.tolist() == [456]
        assert df.jockey.isna().all()       # An empty database string is missing.
        captured["processed"] = True
        return df

    monkeypatch.setattr(feature_store, "FeatureStore", Store)
    monkeypatch.setattr(featured_dataset, "fe_signature", lambda: "fe1")
    monkeypatch.setattr(database, "load_from_database", lambda: live)
    monkeypatch.setattr(database, "get_latest_date", lambda: "2026-10-12")
    monkeypatch.setattr(data_processor, "process_data", process)
    monkeypatch.setattr(featured_dataset, "data_signature", lambda frame, cutoff: "corrected-db")
    monkeypatch.setattr(feature_inputs, "atr_cache_signature", lambda cutoff: "a1")
    monkeypatch.setattr(source_dataset, "committed_dataset", lambda: selected_dataset)
    result = status._feature_store(deep=True)
    assert captured["processed"]
    assert result["state"] == status.BAD
    assert "data_signature" in result["detail"]


def test_store_probe_without_a_committed_dataset_is_bad(status, monkeypatch):
    from src import database, feature_store, featured_dataset, source_dataset

    class Store:
        def exists(self):
            return True

        def meta(self):
            return TestStoreFindings.META.copy()

    monkeypatch.setattr(feature_store, "FeatureStore", Store)
    monkeypatch.setattr(featured_dataset, "fe_signature", lambda: "fe1")
    monkeypatch.setattr(database, "get_latest_date", lambda: "2026-10-12")
    monkeypatch.setattr(source_dataset, "committed_dataset", lambda: None)
    result = status._feature_store(deep=True)
    assert result["state"] == status.BAD
    assert "publish-sources" in result["detail"]
