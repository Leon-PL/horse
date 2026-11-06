import json
from types import SimpleNamespace

from src.ingestion.sporting_life import SportingLifeScraper


def scraper_with(monkeypatch, payload):
    response = SimpleNamespace(text='<script id="__NEXT_DATA__">'+json.dumps(payload)+'</script>')
    scraper = SportingLifeScraper()
    monkeypatch.setattr(scraper, "_get", lambda url: response)
    return scraper


def test_profile_redirect_cannot_supply_another_horses_pedigree(monkeypatch):
    payload = {"props": {"pageProps": {"profile": {"horse_reference": {"id": 999}, "sire": {"name": "Wrong sire"}}}}}
    scraper = scraper_with(monkeypatch, payload)
    assert scraper._fetch_next_data("https://www.sportinglife.com/racing/profiles/horse/123") is None


def test_matching_profile_and_ordinary_race_responses_are_preserved(monkeypatch):
    payload = {"props": {"pageProps": {"profile": {"horse_reference": {"id": 123}}}}}
    scraper = scraper_with(monkeypatch, payload)
    assert scraper._fetch_next_data("https://www.sportinglife.com/racing/profiles/horse/123?view=form") == payload
    payload = {"props": {"pageProps": {"race": {"race_summary": {"name": "A race"}}}}}
    scraper = scraper_with(monkeypatch, payload)
    assert scraper._fetch_next_data("https://www.sportinglife.com/racing/results/2025-01-01/york/123/stakes") == payload


def test_incomplete_profile_is_unknown_rather_than_a_confirmed_identity(monkeypatch):
    scraper = scraper_with(monkeypatch, {"props": {"pageProps": {"profile": {"name": "Unknown"}}}})
    assert scraper._fetch_next_data("https://www.sportinglife.com/racing/profiles/horse/123/") is None
