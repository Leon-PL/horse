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


def test_a_failing_probe_degrades_instead_of_crashing(status, monkeypatch):
    def boom():
        raise RuntimeError("disk gone")

    monkeypatch.setattr(status, "_feature_store", boom)
    report = status.collect()
    assert report["sections"]["feature_store"]["state"] == status.WARN
    assert "disk gone" in report["sections"]["feature_store"]["detail"]
    status.render(report)  # must still render


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

    def test_unstamped_store_is_a_warning_not_a_pass(self, status):
        meta = {k: v for k, v in self.META.items() if k != "atr_signature"}
        state, _ = status.store_findings(meta, live_fe_sig="fe1", live_data_sig="d1")
        assert state == status.WARN
