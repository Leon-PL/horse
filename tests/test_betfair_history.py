"""The automatic path retains full streams and never changes accepted prices."""
import bz2
import ctypes
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.ingestion import betfair_history as history
from src.ingestion import betfair_credentials as credentials

FIRST = date(2026, 1, 1)
LAST = date(2026, 1, 31)


def remote(number):
    return f"/data/xds/historic/BASIC/123/1.{number}.bz2"


def payload(number=100, market="WIN", country="GB", day="2026-01-10"):
    definition = {"eventTypeId": "7", "eventId": "123", "eventName": "York 10th Jan",
        "countryCode": country, "marketType": market, "marketTime": day+"T14:00:00.000Z",
        "status": "CLOSED", "numberOfWinners": 1, "unparsed_source_information": [1, 2, 3],
        "runners": [{"id": 1, "name": "Alpha", "status": "WINNER", "bsp": 3.5},
                    {"id": 2, "name": "Beta", "status": "LOSER", "bsp": 5.0}]}
    messages = [{"op": "mcm", "pt": 1768032000000, "mc": [{"id": f"1.{number}",
        "marketDefinition": definition, "rc": [{"id": 1, "ltp": 4.0}, {"id": 2, "ltp": 6.0}]}]}]
    return bz2.compress("\n".join(json.dumps(m) for m in messages).encode())


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(history.config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(history.config, "PROCESSED_DATA_DIR", str(tmp_path / "processed"))
    return tmp_path


class FakeClient:
    def __init__(self, files=None, packages=None):
        self.files = files or {remote(100): payload(), remote(101): payload(101, "PLACE")}
        self.downloads = []
        self.listings = []
        self.owned = packages if packages is not None else [
            {"sport": "Horse Racing", "plan": "Basic Plan", "forDate": "2026-01-01T00:00:00"}]

    def packages(self):
        return self.owned

    def list_files(self, start, end):
        self.listings.append((start, end))
        return list(self.files)

    def download(self, path):
        self.downloads.append(path)
        return self.files[path]


def test_download_publish_and_repeat_retains_full_source_and_prices(environment):
    client = FakeClient()
    report = history.sync_history(FIRST, LAST, client=client)
    assert report["state"] == "ok" and report["files"] == {"published": 2}
    root = environment / "betfair/history"
    assert history.local_path(root, remote(100)).read_bytes() == client.files[remote(100)]
    assert b"unparsed_source_information" in bz2.decompress(history.local_path(root, remote(100)).read_bytes())
    with history.HistoryStore(root).connection() as con:
        digest = con.execute("SELECT sha256 FROM files WHERE remote=?", (remote(100),)).fetchone()[0]
    assert digest == hashlib.sha256(client.files[remote(100)]).hexdigest()
    for name in ["betfair_bsp", "betfair_bsp_early", "betfair_timing", "betfair_place_bsp"]:
        assert len(pd.read_parquet(environment / f"processed/{name}.parquet")) == 2
    win = pd.read_parquet(environment / "processed/betfair_timing.parquet")
    assert "ltp_1100" in win and "rf_after_300s" in win
    repeated = history.sync_history(FIRST, LAST, client=client)
    assert repeated["downloaded"] == 0 and len(client.downloads) == 2


def test_budgeted_queue_resumes_an_older_requested_range(environment):
    client = FakeClient()
    first = history.sync_history(FIRST, LAST, client=client, budget=1)
    assert first["state"] == "partial" and first["files"]["pending"] == 1
    # A later refresh must still validate January's queued work against its
    # original window, not reject it as a February race.
    second = history.sync_history(date(2026, 2, 1), date(2026, 2, 2), client=client)
    assert second["files"] == {"published": 2}
    assert len(client.downloads) == 2


def test_finished_download_before_crash_is_not_fetched_again(environment):
    client = FakeClient(files={remote(100): payload()})
    root = environment / "betfair/history"
    store = history.HistoryStore(root)
    store.discover([remote(100)], FIRST, LAST)
    path = history.local_path(root, remote(100))
    path.parent.mkdir(parents=True)
    path.write_bytes(payload())
    report = history.sync_history(FIRST, LAST, client=client)
    assert report["state"] == "ok" and not client.downloads


@pytest.mark.parametrize("bad", [b"<html>Sign in</html>", b"BZhbroken", bz2.compress(b"not json")])
def test_bad_downloads_remain_retryable_and_do_not_publish(environment, bad):
    result = history.sync_history(FIRST, LAST, client=FakeClient(files={remote(100): bad}))
    assert result["state"] == "partial" and result["files"] == {"retry": 1}
    assert not (environment / "processed/betfair_bsp.parquet").exists()
    assert not history.local_path(environment / "betfair/history", remote(100)).exists()


@pytest.mark.parametrize("contents,path", [
    (payload(101), remote(100)),
    (payload(country="US"), remote(100)),
    (payload(day="2026-02-01"), remote(100)),
    (payload(market="ANTEPOST_WIN"), remote(100)),
])
def test_market_identity_country_date_and_type_are_checked(contents, path):
    with pytest.raises(history.HistoryError):
        history.parse_download(contents, path, FIRST, LAST)


@pytest.mark.parametrize("path", ["../../outside.bz2", "/data/xds/historic/BASIC/123/../../outside", 
                                "https://example.test/1.1.bz2", "/data/xds/historic/BASIC/123/1.1.bz2?x=1"])
def test_remote_paths_cannot_choose_a_local_destination(tmp_path, path):
    with pytest.raises(history.HistoryError):
        history.local_path(tmp_path, path)


def test_dated_remote_layout_keeps_the_same_market_identity(tmp_path):
    dated = "/data/xds/historic/BASIC/2026/Jan/10/123/1.100.bz2"
    assert history.local_path(tmp_path, dated) == history.local_path(tmp_path, remote(100))


def test_unowned_months_do_not_download_or_try_to_purchase(environment):
    client = FakeClient(packages=[])
    result = history.sync_history(FIRST, LAST, client=client)
    assert result["state"] == "partial" and result["unavailable_months"] == ["2026-01"]
    assert not client.listings and not client.downloads


def test_existing_non_null_values_are_preserved_and_missing_fields_are_filled():
    rows = history.parse_download(payload(), remote(100), FIRST, LAST)[1]
    incoming = pd.DataFrame(rows)
    existing = incoming.iloc[:1].copy()
    existing["bsp"] = 9.0
    existing["ltp_1100"] = float("nan")
    existing["unrelated_existing_field"] = "keep"
    merged, report = history.merge_observations(existing, incoming)
    alpha = merged.loc[merged.horse_name.eq("Alpha")].iloc[0]
    assert alpha.bsp == 9.0 and alpha.ltp_1100 == 4.0
    assert alpha.unrelated_existing_field == "keep"
    assert report == {"new_rows": 1, "preserved_price_conflicts": 1}


def test_duplicate_runner_keys_block_publication():
    rows = pd.DataFrame(history.parse_download(payload(), remote(100), FIRST, LAST)[1])
    with pytest.raises(history.HistoryError, match="Ambiguous"):
        history.merge_observations(pd.DataFrame(), pd.concat([rows, rows]))


def test_publication_waits_for_other_source_writers(environment):
    client = FakeClient()
    with history.writer_lock(environment / "source_refresh/.writer.lock") as locked:
        assert locked
        result = history.sync_history(FIRST, LAST, client=client)
        assert result["state"] == "partial" and result["publication_waiting"]
        assert result["files"] == {"downloaded": 2}
    final = history.sync_history(FIRST, LAST, client=client)
    assert final["state"] == "ok" and len(client.downloads) == 2


def test_windows_saved_connection_is_encrypted_and_round_trips(environment):
    try:
        credentials.save_connection("test-login", "private-test-password", Path("certificate.crt"), Path("key.pem"))
    except RuntimeError:
        if ctypes.get_last_error() == 2:
            pytest.skip("This process has no Windows profile encryption keys; verify under the normal Windows user")
        raise
    stored = credentials.connection_path().read_text()
    assert "test-login" not in stored and "private-test-password" not in stored
    assert credentials.load_connection()["password"] == "private-test-password"


def response(status=200, value=None, content=b"", headers=None):
    return SimpleNamespace(status_code=status, json=lambda: value, content=content, headers=headers or {})


def test_api_filters_use_market_files_and_keep_win_and_place(monkeypatch):
    calls = []
    session = SimpleNamespace(request=lambda *a, **kw: calls.append((a, kw)) or response(value=[remote(100)]))
    client = history.HistoryClient("secret-session", session=session)
    assert client.list_files(FIRST, LAST) == [remote(100)]
    args, kw = calls[0]
    assert args == ("POST", history.API + "DownloadListOfFiles")
    assert kw["json"]["marketTypesCollection"] == ["WIN", "PLACE"]
    assert kw["json"]["fileTypeCollection"] == ["M"]
    assert kw["json"]["countriesCollection"] == ["GB", "IE"]
    assert kw["headers"]["ssoid"] == "secret-session" and not kw["allow_redirects"]


def test_expired_session_stops_requests_and_does_not_expose_tokens():
    calls = []
    session = SimpleNamespace(request=lambda *a, **kw: calls.append(1) or response(401))
    client = history.HistoryClient("secret-session", session=session)
    for _ in range(2):
        with pytest.raises(history.LoginRequired) as caught:
            client.packages()
        assert "secret-session" not in str(caught.value)
    assert calls == [1]


def test_certificate_login_needs_no_live_app_key(environment, monkeypatch):
    cert, key = credentials.certificate_paths()
    cert.parent.mkdir(parents=True)
    cert.write_text("test public certificate")
    key.write_text("test private key")
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None,
            json=lambda: {"loginStatus": "SUCCESS", "sessionToken": "returned-session"})
    monkeypatch.setattr(history.requests, "post", post)
    client = history.HistoryClient.certificate_login("login", "password", str(cert), str(key))
    assert client._token == "returned-session"
    assert calls[0][1]["headers"]["X-Application"] == "horse-history-downloader"
    assert "verify" not in calls[0][1] and not calls[0][1]["allow_redirects"]


def test_normal_bsp_refresh_includes_history_only_after_setup(monkeypatch):
    from src.ingestion import auxiliary
    today = date.today()
    monkeypatch.setattr(auxiliary, "_refresh_public_bsp", lambda *a: {"state": "cached"})
    monkeypatch.setattr(history, "configured", lambda: True)
    calls = []
    monkeypatch.setattr(history, "sync_history", lambda *a, **kw: calls.append((a, kw)) or {"state": "ok"})
    report = auxiliary.refresh_bsp(str(today-timedelta(days=14)), str(today-timedelta(days=1)))
    assert report["state"] == "cached" and report["history"]["state"] == "ok"
    assert calls[0][0][1] == str(today-timedelta(days=6))
    assert calls[0][1]["_source_lock_held"]
    monkeypatch.setattr(history, "configured", lambda: False)
    assert auxiliary.refresh_bsp(str(FIRST), str(LAST)) == {"state": "cached"}


def test_corrupt_cached_download_is_quarantined_and_replaced(environment):
    client = FakeClient(files={remote(100): payload()})
    store = history.HistoryStore(environment / "betfair/history")
    store.discover([remote(100)], FIRST, LAST)
    path = history.local_path(store.root, remote(100))
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupt previous download")
    result = history.sync_history(FIRST, LAST, client=client)
    assert result["state"] == "ok" and client.downloads == [remote(100)]
    assert path.read_bytes() == payload()
    assert len(list(path.parent.glob("*.invalid.*"))) == 1


def test_rate_limit_wait_is_shared_and_honours_retry_after(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(history.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(history.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    responses = [response(429, headers={"Retry-After": "17"}), response(value=[])]
    timestamps = []
    def request(*args, **kwargs):
        timestamps.append(clock[0])
        return responses.pop(0)
    client = history.HistoryClient("session", session=SimpleNamespace(request=request))
    assert client.packages() == []
    assert timestamps[1] - timestamps[0] >= 17


def test_connection_controls_render_without_an_account(environment):
    from streamlit.testing.v1 import AppTest
    cert, key = credentials.certificate_paths()
    cert.parent.mkdir(parents=True)
    cert.write_text("test public certificate")
    key.write_text("test private key")
    app = AppTest.from_string("from src.ingestion.betfair_history_ui import render_history_controls\nrender_history_controls()")
    app.run(timeout=20)
    assert not app.exception
    assert [field.label for field in app.text_input] == ["Betfair username", "Betfair password"]
