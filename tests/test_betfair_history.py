"""The automatic path retains full streams and never changes accepted prices."""
import bz2
import ctypes
from datetime import date
import hashlib
import json
import os
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


def payload(number=100, market="WIN", country="GB", day="2026-01-10", bsp=(3.5, 5.0), **definition_fields):
    definition = {"eventTypeId": "7", "eventId": "123", "eventName": "York 10th Jan",
        "countryCode": country, "marketType": market, "marketTime": day+"T14:00:00.000Z",
        "status": "CLOSED", "numberOfWinners": 1, "unparsed_source_information": [1, 2, 3],
        "runners": [{"id": 1, "name": "Alpha", "status": "WINNER", "bsp": bsp[0]},
                    {"id": 2, "name": "Beta", "status": "LOSER", "bsp": bsp[1]}], **definition_fields}
    messages = [{"op": "mcm", "pt": 1768032000000, "mc": [{"id": f"1.{number}",
        "marketDefinition": definition, "rc": [{"id": 1, "ltp": 4.0}, {"id": 2, "ltp": 6.0}]}]}]
    return bz2.compress("\n".join(json.dumps(m) for m in messages).encode())


REMOVED = [{"id": 1, "name": "Alpha", "status": "REMOVED"}, {"id": 2, "name": "Beta", "status": "REMOVED"}]


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


def test_text_nan_starting_prices_publish_as_missing_numbers(environment):
    # Betfair writes an unreconciled BSP as the text "NaN"; on 2026-11-08 one
    # such runner stopped the whole backfill at the typed Parquet write.
    client = FakeClient({remote(100): payload(bsp=(3.5, "NaN")), remote(101): payload(101, "PLACE", bsp=("Infinity", 1.8))})
    report = history.sync_history(FIRST, LAST, client=client)
    assert report["state"] == "ok" and report["files"] == {"published": 2}
    for name, column, expected in [("betfair_bsp", "bsp", [3.5, None]), ("betfair_timing", "bsp", [3.5, None]),
                                   ("betfair_place_bsp", "place_bsp", [None, 1.8])]:
        frame = pd.read_parquet(environment / f"processed/{name}.parquet").sort_values("horse_name")
        assert frame[column].dtype == float
        assert [None if pd.isna(v) else v for v in frame[column]] == expected


def test_a_match_bet_filed_as_a_win_market_is_retained_but_not_published(environment):
    # Thurles 22 Jan 2023 13:10 had "Match Betting 1" (bspMarket false) beside
    # the race's win market; the same selection IDs stopped publication.
    match = payload(102, bsp=(None, None), name="Match Betting 1", bspMarket=False,
                    runners=[{"id": 2, "name": "Beta", "status": "WINNER"}, {"id": 1, "name": "Alpha", "status": "LOSER"}])
    charity = payload(103, day="2026-01-11", bsp=(None, None), name="Charity Race", bspMarket=False)
    client = FakeClient({remote(100): payload(name="2m4f Nov Chs", bspMarket=True), remote(102): match,
                         remote(103): charity})
    report = history.sync_history(FIRST, LAST, client=client)
    assert report["state"] == "ok" and report["files"] == {"published": 3}
    assert history.local_path(environment / "betfair/history", remote(102)).read_bytes() == match
    win = pd.read_parquet(environment / "processed/betfair_bsp.parquet").sort_values(["race_date", "horse_name"])
    assert win.bf_market_id.tolist() == ["1.100", "1.100", "1.103", "1.103"]
    assert win.won.tolist() == [1, 0, 1, 0] and win.bsp.iloc[:2].tolist() == [3.5, 5.0] and win.bsp.iloc[2:].isna().all()


def test_void_markets_filed_under_another_day_or_country_are_retained_not_reviewed(environment):
    # The 2026-11-09 backfill ended with 45 such files in review: Hereford's
    # abandoned 2 Jan 2021 card filed in December, voided Chelmsford duplicates
    # with country "MO", and "Do Not Activate" placeholders.
    files = {remote(100): payload(),
             remote(104): payload(104, day="2026-02-01", runners=REMOVED),
             remote(105): payload(105, "PLACE", country="MO", runners=REMOVED),
             remote(106): payload(106, day="2025-12-28", name="Do Not Activate", bspMarket=False, runners=REMOVED)}
    report = history.sync_history(FIRST, LAST, client=FakeClient(files))
    assert report["state"] == "ok" and report["files"] == {"published": 4}
    assert history.local_path(environment / "betfair/history", remote(104)).read_bytes() == files[remote(104)]
    assert pd.read_parquet(environment / "processed/betfair_bsp.parquet").bf_market_id.tolist() == ["1.100", "1.100"]
    assert not (environment / "processed/betfair_place_bsp.parquet").exists()


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
    # Only void horse-racing WIN/PLACE markets skip the date and country checks.
    (payload(day="2026-02-01", eventTypeId="4339", runners=REMOVED), remote(100)),
    (payload(day="2026-02-01", market="ANTEPOST_WIN", runners=REMOVED), remote(100)),
    (payload(day="2026-02-01", status="OPEN", runners=REMOVED), remote(100)),
    (payload(day="2026-02-01", runners=[REMOVED[0], {"id": 2, "name": "Beta", "status": "WINNER"}]), remote(100)),
])
def test_market_identity_country_date_and_type_are_checked(contents, path):
    with pytest.raises(history.HistoryError):
        history.parse_download(contents, path, FIRST, LAST)


@pytest.mark.parametrize("path", ["../../outside.bz2", "/data/xds/historic/BASIC/123/../../outside", 
                                "https://example.test/1.1.bz2", "/data/xds/historic/BASIC/123/1.1.bz2?x=1",
                                "/xds_nfs/edp_processed/BASIC/../../outside.bz2",
                                "/xds_nfs/edp_processed/PRO/2026/Oct/31/123/1.100.bz2"])
def test_remote_paths_cannot_choose_a_local_destination(tmp_path, path):
    with pytest.raises(history.HistoryError):
        history.local_path(tmp_path, path)


@pytest.mark.parametrize("dated", [
    "/data/xds/historic/BASIC/2026/Jan/10/123/1.100.bz2",
    "/xds_nfs/edp_processed/BASIC/2026/Oct/31/123/1.100.bz2",
])
def test_dated_remote_layout_keeps_the_same_market_identity(tmp_path, dated):
    assert history.local_path(tmp_path, dated) == history.local_path(tmp_path, remote(100))


@pytest.mark.skipif(os.name != "nt", reason="Windows extended path namespace")
def test_archive_containment_accepts_mixed_windows_namespace_spellings(tmp_path, monkeypatch):
    original = Path.resolve
    def resolve(path, *args, **kwargs):
        resolved = original(path, *args, **kwargs)
        if path.name == "1.100.bz2" and not str(resolved).startswith("\\\\?\\"):
            return Path("\\\\?\\" + str(resolved))
        return resolved
    monkeypatch.setattr(Path, "resolve", resolve)
    assert history.local_path(tmp_path, remote(100)).name == "1.100.bz2"


def test_archive_containment_still_rejects_a_resolved_target_outside_its_root(tmp_path, monkeypatch):
    original = Path.resolve
    def resolve(path, *args, **kwargs):
        if path.name == "1.100.bz2":
            return original(tmp_path.parent / "outside" / path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(history.HistoryError, match="outside"):
        history.local_path(tmp_path, remote(100))


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


def test_packages_and_file_downloads_use_get_without_a_json_body():
    calls = []
    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        # The live GetMyData service rejects POST with HTTP 405.
        if method != "GET":
            return response(405)
        return response(value=[], content=b"compressed-market")
    client = history.HistoryClient("private-token", session=SimpleNamespace(request=request))
    assert client.packages() == []
    assert client.download(remote(100)) == b"compressed-market"
    assert [url for _, url, _ in calls] == [history.API + "GetMyData", history.API + "DownloadFile"]
    assert all(kwargs["json"] is None for _, _, kwargs in calls)
    assert calls[1][2]["params"] == {"filePath": remote(100)}


def test_certificate_login_needs_no_live_app_key(environment, monkeypatch):
    cert, key = credentials.certificate_paths()
    cert.parent.mkdir(parents=True)
    cert.write_text("test public certificate")
    key.write_text("test private key")
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response(value={"loginStatus": "SUCCESS", "sessionToken": "returned-session"})
    monkeypatch.setattr(history.requests, "post", post)
    client = history.HistoryClient.certificate_login("login", "password", str(cert), str(key))
    assert client._token == "returned-session"
    assert calls[0][1]["headers"]["X-Application"] == "horse-history-downloader"
    assert "verify" not in calls[0][1] and not calls[0][1]["allow_redirects"]


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


@pytest.fixture
def login_files(environment):
    cert, key = credentials.certificate_paths()
    cert.parent.mkdir(parents=True)
    cert.write_text("test public certificate")
    key.write_text("test private key")
    return cert, key


@pytest.mark.parametrize("status,action", [
    ("CERT_AUTH_REQUIRED", "same Betfair account"),
    ("INVALID_PASSWORD", "password"),
    ("EMAIL_LOGIN_NOT_ALLOWED", "username rather than your email"),
    ("CONTACT_VERIFICATION_REQUIRED", "account verification"),
])
def test_login_rejection_shows_a_safe_action_and_provider_code(login_files, monkeypatch, status, action):
    monkeypatch.setattr(history.requests, "post", lambda *a, **kw: response(value={
        "loginStatus": status, "sessionToken": "private-token", "details": "private-account-detail"}))
    with pytest.raises(history.LoginRequired) as caught:
        history.HistoryClient.certificate_login("private-login", "private-password", *map(str, login_files))
    assert status in str(caught.value) and action in str(caught.value)
    assert not any(secret in str(caught.value) for secret in
                   ("private-token", "private-login", "private-password", "private-account-detail"))


@pytest.mark.parametrize("value", [
    {"loginStatus": "private-unrecognised-status", "sessionToken": "private-token"},
    {"loginStatus": ["private-account-detail"]},
    {"loginStatus": "SUCCESS", "sessionToken": {"private-token": 1}},
    {"loginStatus": "SUCCESS"},
    ["private-response"],
])
def test_unknown_or_malformed_login_responses_are_rejected_without_echoing_them(login_files, monkeypatch, value):
    monkeypatch.setattr(history.requests, "post", lambda *a, **kw: response(value=value))
    with pytest.raises(history.LoginRequired) as caught:
        history.HistoryClient.certificate_login("private-login", "private-password", *map(str, login_files))
    assert "private-" not in str(caught.value)


@pytest.mark.parametrize("failure,expected", [
    (history.requests.exceptions.SSLError("private-network-detail"), "secure certificate connection"),
    (history.requests.Timeout("private-network-detail"), "timed out"),
    (history.requests.ConnectionError("private-network-detail"), "Could not reach"),
])
def test_login_transport_failures_remain_actionable_without_raw_exception_details(login_files, monkeypatch,
                                                                                  failure, expected):
    def post(*args, **kwargs):
        raise failure
    monkeypatch.setattr(history.requests, "post", post)
    with pytest.raises(history.LoginRequired) as caught:
        history.HistoryClient.certificate_login("private-login", "private-password", *map(str, login_files))
    assert expected in str(caught.value) and "private-" not in str(caught.value)


def test_login_http_failure_is_reported_before_parsing_the_response(login_files, monkeypatch):
    def unexpected_json():
        pytest.fail("An HTTP refusal must not be interpreted as a login response")
    monkeypatch.setattr(history.requests, "post", lambda *a, **kw:
                        SimpleNamespace(status_code=403, json=unexpected_json))
    with pytest.raises(history.LoginRequired, match="HTTP 403"):
        history.HistoryClient.certificate_login("login", "password", *map(str, login_files))


def test_non_json_login_response_is_safely_reported(login_files, monkeypatch):
    def invalid_json():
        raise ValueError("private-response")
    monkeypatch.setattr(history.requests, "post", lambda *a, **kw:
                        SimpleNamespace(status_code=200, json=invalid_json))
    with pytest.raises(history.LoginRequired) as caught:
        history.HistoryClient.certificate_login("login", "password", *map(str, login_files))
    assert "unexpected response" in str(caught.value) and "private-response" not in str(caught.value)


def test_missing_historical_months_are_distinct_from_a_bad_login(login_files, monkeypatch):
    monkeypatch.setattr(history.HistoryClient, "certificate_login", lambda *a, **kw: FakeClient(packages=[]))
    monkeypatch.setattr(history, "save_connection", lambda *a: pytest.fail("Missing access must not save credentials"))
    report = history.connect_account("private-login", "private-password", *login_files)
    assert report["state"] == "failed" and report["stage"] == "historical_access"
    assert "sign-in succeeded" in report["message"] and "My Data" in report["message"]
    retained = json.loads(credentials.connection_path().with_name("connection_status.json").read_text())
    assert retained == report
    assert "private-" not in json.dumps(retained)
    assert not credentials.connection_path().exists()


def test_historical_service_refusal_keeps_the_successful_login_distinct(login_files, monkeypatch):
    client = history.HistoryClient("private-token", session=SimpleNamespace(request=lambda *a, **kw: response(401)))
    monkeypatch.setattr(history.HistoryClient, "certificate_login", lambda *a, **kw: client)
    report = history.connect_account("private-login", "private-password", *login_files)
    assert report["stage"] == "historical_access" and report["state"] == "failed"
    assert "declined" in report["message"] and "private-" not in json.dumps(report)


@pytest.mark.parametrize("failure,action", [
    (RuntimeError("private-encryption-detail"), "Windows could not encrypt"),
    (OSError("private-filesystem-detail"), "could not save the connection"),
])
def test_local_save_failure_does_not_blame_betfair_login(login_files, monkeypatch, failure, action):
    monkeypatch.setattr(history.HistoryClient, "certificate_login", lambda *a, **kw: FakeClient())
    def fail_save(*args):
        raise failure
    monkeypatch.setattr(history, "save_connection", fail_save)
    report = history.connect_account("private-login", "private-password", *login_files)
    assert report["stage"] == "save_connection" and report["state"] == "failed"
    assert action in report["message"] and "access succeeded" in report["message"]
    assert "private-" not in json.dumps(report)


def test_successful_setup_saves_only_after_access_checks(login_files, monkeypatch):
    calls = []
    monkeypatch.setattr(history.HistoryClient, "certificate_login", lambda *a, **kw: FakeClient())
    monkeypatch.setattr(history, "save_connection", lambda *a: calls.append(a))
    report = history.connect_account("private-login", "private-password", *login_files)
    assert report["state"] == "connected" and report["stage"] == "complete"
    assert calls == [("private-login", "private-password", *login_files)]
    assert "private-" not in json.dumps(report)


def test_failed_diagnostic_write_does_not_change_a_successful_setup(login_files, monkeypatch):
    monkeypatch.setattr(history.HistoryClient, "certificate_login", lambda *a, **kw: FakeClient())
    monkeypatch.setattr(history, "save_connection", lambda *a: None)
    def no_diagnostic(*args):
        raise OSError("read-only diagnostic destination")
    monkeypatch.setattr(history, "atomic_json", no_diagnostic)
    assert history.connect_account("login", "password", *login_files)["state"] == "connected"


def test_main_refresh_uses_full_history_and_reports_the_publication_gap(monkeypatch):
    from datetime import datetime, timezone
    from src.ingestion import auxiliary
    monkeypatch.setattr(auxiliary, "_utc_now", lambda: datetime(2026, 11, 6, tzinfo=timezone.utc))
    monkeypatch.setattr(history, "configured", lambda: True)
    calls = []
    monkeypatch.setattr(history, "sync_history", lambda *a, **kw: calls.append((a, kw)) or {"state": "ok"})
    result = auxiliary.refresh_betfair("2026-10-20", "2026-11-05")
    assert calls[0][0] == ("2026-10-20", "2026-10-31")
    assert calls[0][1]["_source_lock_held"]
    assert result["state"] == "partial" and result["deferred_recent_days"] == 5


def test_only_unpublished_dates_do_not_request_history(monkeypatch):
    from datetime import datetime, timezone
    from src.ingestion import auxiliary
    monkeypatch.setattr(auxiliary, "_utc_now", lambda: datetime(2026, 11, 6, tzinfo=timezone.utc))
    monkeypatch.setattr(history, "configured", lambda: True)
    monkeypatch.setattr(history, "sync_history", lambda *a, **kw: pytest.fail("No published dates"))
    assert auxiliary.refresh_betfair("2026-11-03", "2026-11-05")["state"] == "awaiting_publication"


def test_missing_connection_is_visible_in_the_main_refresh(monkeypatch):
    from src.ingestion import auxiliary
    monkeypatch.setattr(history, "configured", lambda: False)
    assert auxiliary.refresh_betfair("2026-10-20", "2026-10-31")["state"] == "needs_login"


def test_completed_historical_listing_is_reused(environment):
    client = FakeClient(files={remote(100): payload()})
    assert history.sync_history(FIRST, LAST, client=client)["state"] == "ok"
    assert history.sync_history(FIRST, LAST, client=client)["state"] == "ok"
    assert len(client.listings) == 1 and len(client.downloads) == 1


def test_recent_listing_is_rechecked_after_its_revision_interval(environment):
    from datetime import timedelta
    store = history.HistoryStore(environment / "betfair/history")
    day = date.today()-timedelta(days=6)
    store.discover([], day, day)
    assert store.listing_known(day, day)
    with store.connection() as con:
        con.execute("UPDATE discovery SET checked_at='2020-01-01T00:00:00+00:00'")
    assert not store.listing_known(day, day)


def test_discovery_validates_remote_names_without_resolving_local_paths(environment, monkeypatch):
    store = history.HistoryStore(environment / "betfair/history")
    monkeypatch.setattr(history, "local_path", lambda *a: pytest.fail("Discovery must not inspect raw files"))
    store.discover([remote(100)], FIRST, LAST)
    assert store.counts() == {"pending": 1}
    with pytest.raises(history.HistoryError):
        store.discover(["../../outside.bz2"], FIRST, LAST)


def test_backfill_drains_multiple_batches_without_restarting_the_job(environment, monkeypatch):
    reports = iter([
        {"state": "partial", "processed": 1, "reused_local": 1, "files": {"pending": 1, "published": 1}},
        {"state": "ok", "processed": 1, "downloaded": 1, "files": {"published": 2}},
    ])
    monkeypatch.setattr(history, "sync_history", lambda *a, **kw: next(reports))
    result = history.drain_history(FIRST, LAST, budget=1)
    assert result["state"] == "ok" and result["totals"] == {"processed": 2, "downloaded": 1, "reused_local": 1}


def test_repeated_invalid_markets_eventually_require_review_instead_of_an_infinite_loop(environment):
    client = FakeClient(files={remote(100): b"invalid source"})
    for _ in range(3):
        report = history.sync_history(FIRST, LAST, client=client)
        with history.HistoryStore(environment / "betfair/history").connection() as con:
            con.execute("UPDATE files SET retry_after=NULL")
    assert report["files"] == {"review": 1} and report["state"] == "partial"


def test_checkpoint_rollback_retains_raw_files_for_an_exact_resume(environment):
    root = environment / "betfair/history"
    store = history.HistoryStore(root)
    client = FakeClient()
    store.discover(list(client.files), FIRST, LAST)
    records = []
    for remote_path, body in client.files.items():
        market, rows = history.parse_download(body, remote_path, FIRST, LAST)
        records.append({"remote": remote_path, "payload": body, "market": market, "rows": rows})
    with store.connection() as con:
        con.execute("CREATE TRIGGER interrupted_checkpoint BEFORE UPDATE ON files "
                    "WHEN NEW.remote LIKE '%1.101.bz2' AND NEW.state='downloaded' "
                    "BEGIN SELECT RAISE(ABORT,'checkpoint interrupted'); END")
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError, match="interrupted"):
        store.save_many(records)
    assert store.counts() == {"pending": 2}
    assert all(history.local_path(root, name).read_bytes() == body for name, body in client.files.items())
    with store.connection() as con:
        con.execute("DROP TRIGGER interrupted_checkpoint")
    result = history.sync_history(FIRST, LAST, client=client)
    assert result["state"] == "ok" and result["files"] == {"published": 2}
    assert result["reused_local"] == 2 and not client.downloads


def test_duplicate_checkpoint_entries_are_rejected_before_metadata_changes(environment):
    store = history.HistoryStore(environment / "betfair/history")
    store.discover([remote(100)], FIRST, LAST)
    body = payload()
    market, rows = history.parse_download(body, remote(100), FIRST, LAST)
    item = {"remote": remote(100), "payload": body, "market": market, "rows": rows}
    with pytest.raises(history.HistoryError, match="Duplicate"):
        store.save_many([item, item])
    assert store.counts() == {"pending": 1}


def test_authentication_failure_leaves_unscheduled_markets_pending(environment):
    class ExpiredClient(FakeClient):
        def download(self, path):
            self.downloads.append(path)
            raise history.LoginRequired("Reconnect to Betfair")
    client = ExpiredClient(files={remote(i): payload(i) for i in range(100, 120)})
    report = history.sync_history(FIRST, LAST, client=client, workers=1)
    assert report["state"] == "needs_login" and report["files"]["pending"] >= 18
    assert len(client.downloads) <= 2 and not report["processed"]


def test_mixed_failed_and_successful_markets_keep_accurate_checkpoint_counts(environment):
    client = FakeClient(files={remote(100): payload(), remote(101): b"invalid source"})
    result = history.sync_history(FIRST, LAST, client=client)
    assert result["state"] == "partial" and result["failed"] == 1
    assert result["files"] == {"published": 1, "retry": 1}
    assert result["processed"] == 1 and result["downloaded"] == 1


def namesake_payload(number=100, market="WIN"):
    message = json.loads(bz2.decompress(payload(number, market)))
    runners = message["mc"][0]["marketDefinition"]["runners"]
    runners[0].update(name="Sierra Nevada (USA)", id=45802885, bsp=4.61)
    runners[1].update(name="Sierra Nevada (GB)", id=620329, bsp=19.52)
    return bz2.compress(json.dumps(message).encode())


@pytest.mark.parametrize("market", ["WIN", "PLACE"])
def test_namesakes_publish_and_replay_by_exchange_id(environment, market):
    client = FakeClient(files={remote(100): namesake_payload(market=market)})
    first = history.sync_history(FIRST, LAST, client=client)
    assert first["state"] == "ok" and first["files"] == {"published": 1}
    name = "betfair_bsp" if market == "WIN" else "betfair_place_bsp"
    cache = pd.read_parquet(environment / f"processed/{name}.parquet")
    price = "bsp" if market == "WIN" else "place_bsp"
    assert cache.horse_name.tolist() == ["Sierra Nevada", "Sierra Nevada"]
    assert cache.set_index("bf_runner_id")[price].to_dict() == {45802885: 4.61, 620329: 19.52}
    replay, stats = history.merge_observations(cache, cache.iloc[::-1])
    assert len(replay) == 2 and stats == {"new_rows": 0, "preserved_price_conflicts": 0}


def test_existing_named_runner_keeps_price_and_gains_its_namesake():
    rows = pd.DataFrame(history.parse_download(namesake_payload(), remote(100), FIRST, LAST)[1])
    existing = rows.iloc[:1].assign(bsp=9., bf_runner_id=45802885., existing_note="keep").drop(columns="bf_market_id")
    merged, stats = history.merge_observations(existing, rows.iloc[::-1])
    assert len(merged) == 2 and stats == {"new_rows": 1, "preserved_price_conflicts": 1}
    by_id = merged.set_index("bf_runner_id")
    assert by_id.loc[45802885, "bsp"] == 9. and by_id.loc[45802885, "existing_note"] == "keep"
    assert by_id.loc[45802885, "bf_market_id"] == "1.100"
    assert by_id.loc[620329, "bsp"] == 19.52


def test_unique_legacy_name_gains_id_but_different_known_ids_do_not_match():
    rows = pd.DataFrame(history.parse_download(payload(), remote(100), FIRST, LAST)[1])
    legacy = rows.drop(columns="bf_runner_id").assign(bsp=9.)
    merged, stats = history.merge_observations(legacy, rows)
    assert len(merged) == 2 and merged.bf_runner_id.tolist() == [1, 2]
    assert merged.bsp.tolist() == [9., 9.] and stats["preserved_price_conflicts"] == 2
    other = rows.iloc[:1].assign(bf_runner_id=99, bsp=25.)
    merged, stats = history.merge_observations(rows.iloc[:1], other)
    assert len(merged) == 2 and stats["new_rows"] == 1
    assert merged.set_index("bf_runner_id").bsp.to_dict() == {1: 3.5, 99: 25.}


def test_old_place_stage_restores_ids_and_retains_unidentifiable_cache_row(environment):
    root = environment / "betfair/history"
    store = history.HistoryStore(root)
    body = namesake_payload(market="PLACE")
    market, rows = history.parse_download(body, remote(100), FIRST, LAST)
    assert {row["bf_runner_id"] for row in rows} == {45802885, 620329}
    old = [{k: v for k, v in row.items() if k != "bf_runner_id"} for row in rows]
    store.discover([remote(100)], FIRST, LAST)
    store.save(remote(100), body, market, old)
    destination = environment / "processed"
    destination.mkdir()
    legacy = pd.DataFrame(old[:1]).assign(place_bsp=99., legacy_note="preserve without guessing")
    legacy.to_parquet(destination / "betfair_place_bsp.parquet", index=False)
    report = history.publish(store, destination)["betfair_place_bsp.parquet"]
    assert report["restored_identity_markets"] == 1 and report["identity_review_rows"] == 1
    review = json.loads(Path(report["identity_review_path"]).read_text())
    assert review["observations"][0]["place_bsp"] == 99.
    assert review["observations"][0]["legacy_note"] == "preserve without guessing"
    assert store.counts() == {"published": 1}
    with store.connection() as con:
        updated = json.loads(con.execute("SELECT rows_json FROM files").fetchone()[0])
    assert [{k: v for k, v in row.items() if k != "bf_runner_id"} for row in updated] == old
    cache = pd.read_parquet(destination / "betfair_place_bsp.parquet")
    assert len(cache) == 2
    assert cache.set_index("bf_runner_id").place_bsp.to_dict() == {45802885: 4.61, 620329: 19.52}
    assert history.restore_place_runner_ids(store) == 0 and history.publish(store, destination) == {}


def test_place_identity_upgrade_requires_its_exact_retained_revision(environment):
    root = environment / "betfair/history"
    store = history.HistoryStore(root)
    body = namesake_payload(market="PLACE")
    market, parsed = history.parse_download(body, remote(100), FIRST, LAST)
    old = [{k: v for k, v in row.items() if k != "bf_runner_id"} for row in parsed]
    store.discover([remote(100)], FIRST, LAST)
    store.save(remote(100), body, market, old)
    path = history.local_path(root, remote(100))
    path.write_bytes(payload(market="PLACE"))
    with pytest.raises(history.HistoryError, match="revision is missing"):
        history.publish(store, environment / "processed")
    assert store.counts() == {"downloaded": 1}
    assert not (environment / "processed/betfair_place_bsp.parquet").exists()
    path.write_bytes(body)
    assert history.restore_place_runner_ids(store) == 1


def test_place_identity_upgrade_rejects_changed_staged_observations(environment):
    store = history.HistoryStore(environment / "betfair/history")
    body = namesake_payload(market="PLACE")
    market, parsed = history.parse_download(body, remote(100), FIRST, LAST)
    old = [{k: v for k, v in row.items() if k != "bf_runner_id"} for row in parsed]
    old[0]["place_bsp"] = 99.
    store.discover([remote(100)], FIRST, LAST)
    store.save(remote(100), body, market, old)
    with pytest.raises(history.HistoryError, match="observations changed"):
        history.restore_place_runner_ids(store)
    with store.connection() as con:
        assert json.loads(con.execute("SELECT rows_json FROM files").fetchone()[0]) == old


def test_already_published_legacy_place_rows_are_upgraded_by_the_normal_publisher(environment):
    store = history.HistoryStore(environment / "betfair/history")
    body = namesake_payload(market="PLACE")
    market, parsed = history.parse_download(body, remote(100), FIRST, LAST)
    old = [{k: v for k, v in row.items() if k != "bf_runner_id"} for row in parsed]
    store.discover([remote(100)], FIRST, LAST)
    store.save(remote(100), body, market, old)
    with store.connection() as con:
        con.execute("UPDATE files SET state='published'")
    outputs = history.publish(store, environment / "processed")
    assert outputs["betfair_place_bsp.parquet"]["restored_identity_markets"] == 1
    assert store.counts() == {"published": 1}
    assert history.restore_place_runner_ids(store) == 0


def test_backfill_pins_the_shared_exchange_identity_matcher(environment, monkeypatch):
    monkeypatch.setattr(history, "sync_history", lambda *a, **kw: {"state": "ok", "files": {}})
    report = history.drain_history(FIRST, LAST)
    assert {"betfair.py", "runner_identity.py", "storage.py"}.issubset(report["code_hashes"])


def test_legacy_review_must_be_saved_before_replacing_the_accepted_cache(environment, monkeypatch):
    store = history.HistoryStore(environment / "betfair/history")
    body = namesake_payload(market="PLACE")
    market, rows = history.parse_download(body, remote(100), FIRST, LAST)
    store.discover([remote(100)], FIRST, LAST)
    store.save(remote(100), body, market, rows)
    destination = environment / "processed"
    destination.mkdir()
    path = destination / "betfair_place_bsp.parquet"
    pd.DataFrame(rows[:1]).drop(columns="bf_runner_id").assign(place_bsp=99.).to_parquet(path, index=False)
    before = path.read_bytes()
    original = history.atomic_json
    def cannot_preserve_review(path, value):
        raise OSError("review storage unavailable")
    monkeypatch.setattr(history, "atomic_json", cannot_preserve_review)
    with pytest.raises(OSError, match="review storage unavailable"):
        history.publish(store, destination)
    assert path.read_bytes() == before and store.counts() == {"downloaded": 1}
    monkeypatch.setattr(history, "atomic_json", original)
    assert history.publish(store, destination)[path.name]["identity_review_rows"] == 1
    assert store.counts() == {"published": 1}


def test_legacy_alias_for_one_selection_is_consolidated_without_losing_its_observation():
    rows = pd.DataFrame(history.parse_download(namesake_payload(), remote(100), FIRST, LAST)[1])
    existing = rows.iloc[:1].assign(bsp=4.61, old_note=pd.NA)
    alias = existing.assign(horse_name="Sierra Nevada USA", bsp=4.613702474634567, old_note="retained alias")
    merged, stats = history.merge_observations(pd.concat([existing, alias], ignore_index=True), rows)
    assert len(merged) == 2 and merged.bf_runner_id.is_unique
    winner = merged.loc[merged.bf_runner_id.eq(45802885)].iloc[0]
    assert winner.bsp == 4.61 and winner.old_note == "retained alias"
    assert stats["identity_review"][0]["horse_name"] == "Sierra Nevada USA"
    assert stats["identity_review"][0]["bsp"] == 4.613702474634567


def test_place_identity_upgrade_reuses_consolidated_source_by_exact_hash(environment):
    from src.ingestion import betfair_archives as archives
    from test_betfair_archives import archive
    body = namesake_payload(market="PLACE")
    directory = environment / "betfair"
    archive(directory / "old.tar", [("BASIC/123/1.100.bz2", body)])
    archives.consolidate(directory)
    store = history.HistoryStore(directory / "history")
    market, rows = history.parse_download(body, remote(100), FIRST, LAST)
    old = [{k: v for k, v in row.items() if k != "bf_runner_id"} for row in rows]
    store.discover([remote(100)], FIRST, LAST)
    store.save(remote(100), body, market, old, retain_file=False)
    assert history.restore_place_runner_ids(store) == 1
    with store.connection() as con:
        restored, digest = con.execute("SELECT rows_json,sha256 FROM files").fetchone()
    assert {row["bf_runner_id"] for row in json.loads(restored)} == {45802885, 620329}
    assert digest == hashlib.sha256(body).hexdigest()
    assert not history.local_path(store.root, remote(100)).exists()


def test_place_identity_upgrade_does_not_commit_a_partially_verified_group(environment):
    store = history.HistoryStore(environment / "betfair/history")
    for number in (100, 101):
        body = namesake_payload(number, "PLACE")
        market, rows = history.parse_download(body, remote(number), FIRST, LAST)
        old = [{k: v for k, v in row.items() if k != "bf_runner_id"} for row in rows]
        if number == 101:
            old[0]["place_bsp"] = 99.
        store.discover([remote(number)], FIRST, LAST)
        store.save(remote(number), body, market, old)
    with pytest.raises(history.HistoryError, match="observations changed"):
        history.restore_place_runner_ids(store)
    with store.connection() as con:
        saved = con.execute("SELECT rows_json FROM files").fetchall()
    assert all("bf_runner_id" not in json.loads(body)[0] for body, in saved)
    assert store.counts() == {"downloaded": 2}
