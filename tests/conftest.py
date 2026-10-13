"""Shared test isolation."""

import pytest


@pytest.fixture(autouse=True, scope="session")
def _no_local_course_reports(tmp_path_factory):
    """Hide the local BHA files from every test unless it supplies its own.

    Synthetic races use real course names and 2025 dates, so
    ``process_data`` would otherwise join stalls positions and rail changes
    from ``data/processed/bha_*.parquet`` and results would depend on what
    happens to be on disk. Session scope so module-scoped fixtures that call
    ``process_data`` are covered too; tests patch the paths again to supply
    their own files.
    """
    from src import bha_reports, course_reports

    def _no_network(*_args, **_kwargs):
        raise RuntimeError("BHA network access is disabled in tests")

    missing = tmp_path_factory.mktemp("no_course_reports")
    patch = pytest.MonkeyPatch()
    patch.setattr(course_reports, "GOING_PATH", str(missing / "bha_going.parquet"))
    patch.setattr(course_reports, "RACES_PATH", str(missing / "bha_races.parquet"))
    # The sync and live picks refresh BHA reports; tests must never reach the API.
    patch.setattr(bha_reports, "refresh", _no_network)
    yield
    patch.undo()
