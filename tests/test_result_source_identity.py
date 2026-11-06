"""Validate the source's own identity before accepting results or terminal states."""
import pytest

from src.ingestion.sporting_life import SportingLifeScraper, RaceNotRun

URL = "https://www.sportinglife.com/racing/results/2026-11-03/kempton/123/example"


def scraper(monkeypatch, *, identity=123, day="2026-11-03", stage="ABANDONED"):
    client = SportingLifeScraper()
    summary = {"race_summary_reference": {"id": identity}, "date": day, "race_stage": stage}
    monkeypatch.setattr(client, "_fetch_next_data", lambda url: {"props": {"pageProps": {"race": {"race_summary": summary}}}})
    return client


@pytest.mark.parametrize("stage", ["WEIGHEDIN", "ABANDONED", "RACEVOID", "DORMANT"])
def test_wrong_source_date_cannot_be_overwritten_by_the_requested_date(monkeypatch, stage):
    client = scraper(monkeypatch, day="2026-11-02", stage=stage)
    with pytest.raises(ValueError, match="another race date"):
        client.scrape_race_result(URL, "2026-11-03", raise_if_not_run=True)


@pytest.mark.parametrize("identity", [456, None, ""])
def test_wrong_or_missing_source_id_cannot_close_an_unrun_race(monkeypatch, identity):
    client = scraper(monkeypatch, identity=identity)
    with pytest.raises(ValueError, match="another race identity"):
        client.scrape_race_result(URL, "2026-11-03", raise_if_not_run=True)


def test_matching_terminal_race_still_closes(monkeypatch):
    client = scraper(monkeypatch)
    with pytest.raises(RaceNotRun, match="ABANDONED"):
        client.scrape_race_result(URL, "2026-11-03", raise_if_not_run=True)


def test_caller_date_must_also_agree_with_the_url(monkeypatch):
    client = scraper(monkeypatch, day="2026-11-02")
    with pytest.raises(ValueError, match="disagrees with its URL"):
        client.scrape_race_result(URL, "2026-11-02", raise_if_not_run=True)
