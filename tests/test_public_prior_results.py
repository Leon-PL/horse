import pytest
from scripts.collect_prior_horse_results import sanitise


def profile():
    return {"horse_reference": {"id": 123}, "name": "Alpha", "age": 99,
            "current_rating": 999, "stats": {"wins": 100},
            "previous_results": [{"horse_id": 123, "horse_name": "Alpha", "race_id": 5,
                "date": "2024-03-01", "position": 2, "runner_count": 7,
                "or": 150, "bha": 150, "odds": "2/1"}]}


def test_snapshot_keeps_dated_results_and_excludes_current_ratings_and_prices():
    rows = sanitise(profile(), "123", "Alpha")
    assert len(rows) == 1 and rows[0]["position"] == 2
    assert not {"or", "bha", "age", "current_rating", "stats", "odds"}.intersection(rows[0])


def test_identity_conflicts_are_rejected():
    with pytest.raises(ValueError, match="identity"):
        sanitise(profile(), "999", "Alpha")
    p = profile(); p["previous_results"][0]["horse_id"] = 999
    with pytest.raises(ValueError, match="different horse"):
        sanitise(p, "123", "Alpha")


def test_conflicting_duplicates_are_not_arbitrarily_selected():
    p = profile(); p["previous_results"].append({**p["previous_results"][0], "position": 3})
    with pytest.raises(ValueError, match="Conflicting"):
        sanitise(p, "123", "Alpha")


def test_resume_keeps_the_registered_cohort_after_local_features_change(tmp_path, monkeypatch):
    import json
    import sys
    from scripts import collect_prior_horse_results as module
    monkeypatch.setattr(module, "OUTPUT", tmp_path)
    horse = {"horse_id": "123", "horse_name": "Alpha", "first_local_date": "2025-01-01"}
    registration = {"horses": [horse], "workers": 3, "source_fields": list(module.FIELDS)}
    (tmp_path / "cohort.json").write_text(json.dumps(registration))
    monkeypatch.setattr(module, "cohort", lambda: pytest.fail("A resumed pilot must not select again"))
    monkeypatch.setattr(module, "SportingLifeScraper", lambda: object())
    monkeypatch.setattr(module, "fetch_one", lambda *args: ({**horse, "results": []}, True))
    monkeypatch.setattr(sys, "argv", ["collect_prior_horse_results.py", "--workers", "1"])
    module.main()
    assert json.loads((tmp_path / "cohort.json").read_text()) == registration
    assert json.loads((tmp_path / "coverage_report.json").read_text())["reused"] == 1
