"""Resumable downloads of entitled Betfair Historical market files.

Uses the Historical Data API, not the paid Live Exchange App Key. Full BZ2
streams are retained; existing WIN/PLACE parsers supply the derived caches.
There are deliberately no purchase or betting operations in this client.
"""
from __future__ import annotations

import argparse
import bz2
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import threading
import time

import pandas as pd
import requests

import config
from src.betfair import bf_horse_key, bf_track_key
from src.ingestion.betfair_credentials import connection_path, load_connection
from src.ingestion.storage import atomic_json, atomic_parquet, writer_lock
from src.utils import normalise_off_time_key

API = "https://historicdata.betfair.com/api/"
KEYS = ["race_date", "track_key", "off_key", "horse_key"]
WIN_COLUMNS = ["race_date", "track_key", "off_key", "horse_key", "off_time", "track", "horse_name",
               "bf_runner_id", "bsp", "ltp_preoff", "ltp_60s", "ltp_300s", "ltp_3600s", "ltp_1100",
               "ltp_last", "won", "n_runners", "market_id", "bf_market_id"]


class HistoryError(RuntimeError):
    """An actionable error without credentials or raw account responses."""


class LoginRequired(HistoryError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def configured():
    setting = os.getenv("BETFAIR_HISTORY_ENABLED", "").lower()
    if setting in {"0", "false", "no"}:
        return False
    return setting in {"1", "true", "yes"} or connection_path().exists()


class HistoryClient:
    def __init__(self, token, *, requests_per_second=5, session=None):
        if not token:
            raise LoginRequired("Connect a Betfair account before downloading historical data.")
        if not 0 < requests_per_second <= 8:
            raise ValueError("Use between 0 and 8 historical requests per second")
        self._token = token
        self._sessions = threading.local()
        self._session_override = session
        self._lock = threading.Lock()
        self._next_request = 0.0
        self._interval = 1 / requests_per_second
        self._login_failed = False

    @classmethod
    def from_environment(cls):
        token = os.getenv("BETFAIR_SESSION_TOKEN", "")
        if token:
            return cls(token)
        try:
            explicit = all(os.getenv(name) for name in ("BETFAIR_USERNAME", "BETFAIR_PASSWORD", "BETFAIR_CERT_FILE"))
            saved = load_connection() if connection_path().exists() and not explicit else {}
        except RuntimeError as error:
            raise LoginRequired(str(error)) from None
        username = os.getenv("BETFAIR_USERNAME", saved.get("username", ""))
        password = os.getenv("BETFAIR_PASSWORD", saved.get("password", ""))
        certificate = os.getenv("BETFAIR_CERT_FILE", saved.get("certificate", ""))
        key = os.getenv("BETFAIR_KEY_FILE", saved.get("key", ""))
        return cls.certificate_login(username, password, certificate, key)

    @classmethod
    def certificate_login(cls, username, password, certificate, key=""):
        if not all((username, password, certificate)):
            raise LoginRequired("Unattended historical downloads need your Betfair username, password "
                                "and a login certificate linked to your Betfair account.")
        if not Path(certificate).is_file() or (key and not Path(key).is_file()):
            raise LoginRequired("The configured Betfair login certificate/key file is missing.")
        # Certificate-login X-Application is an identifier, not a Live App Key.
        try:
            response = requests.post("https://identitysso-cert.betfair.com/api/certlogin",
                headers={"X-Application": "horse-history-downloader", "Accept": "application/json"},
                data={"username": username, "password": password},
                cert=(certificate, key) if key else certificate, timeout=(15, 45), allow_redirects=False)
            response.raise_for_status()
            value = response.json()
        except (requests.RequestException, ValueError):
            raise LoginRequired("Betfair certificate login failed; check your connection and account setup.") from None
        if not isinstance(value, dict) or value.get("loginStatus") != "SUCCESS" or not value.get("sessionToken"):
            raise LoginRequired("Betfair declined the certificate login. Check the linked certificate "
                                "and complete any account notices on the Betfair website.")
        return cls(value["sessionToken"])

    def _session(self):
        if self._session_override is not None:
            return self._session_override
        if not hasattr(self._sessions, "http"):
            self._sessions.http = requests.Session()
        return self._sessions.http

    def _request(self, operation, *, body=None, params=None, binary=False):
        for attempt in range(3):
            if self._login_failed:
                raise LoginRequired("The Betfair login session expired; reconnect to resume.")
            with self._lock:
                time.sleep(max(0, self._next_request - time.monotonic()))
                self._next_request = time.monotonic() + self._interval
            try:
                response = self._session().request("GET" if binary else "POST", API + operation,
                    headers={"ssoid": self._token, "Content-Type": "application/json"},
                    json=body if not binary else None, params=params, timeout=(15, 90), allow_redirects=False)
            except requests.RequestException:
                if attempt == 2:
                    raise HistoryError("Betfair historical download failed; saved progress can be resumed.") from None
                time.sleep(2 ** attempt)
                continue
            if response.status_code in {401, 403}:
                self._login_failed = True
                raise LoginRequired("Betfair historical access was declined or the login session expired.")
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 2:
                    raise HistoryError("Betfair is temporarily limiting historical requests; retry later.")
                try:
                    wait = max(10, min(300, float(response.headers.get("Retry-After", 10))))
                except ValueError:
                    wait = 30
                with self._lock:
                    self._next_request = max(self._next_request, time.monotonic() + wait)
                continue
            if response.status_code != 200:
                raise HistoryError(f"Betfair historical request returned HTTP {response.status_code}.")
            if binary:
                return response.content
            try:
                return response.json()
            except ValueError:
                raise HistoryError("Betfair returned an unexpected response; no source data were replaced.") from None
        raise HistoryError("Historical request did not complete")

    def packages(self):
        value = self._request("GetMyData", body={})
        if not isinstance(value, list) or not all(isinstance(p, dict) for p in value):
            raise LoginRequired("Betfair did not return a valid My Data response. Check the account login.")
        return value

    def list_files(self, start, end):
        body = {"sport": "Horse Racing", "plan": "Basic Plan",
                "fromDay": start.day, "fromMonth": start.month, "fromYear": start.year,
                "toDay": end.day, "toMonth": end.month, "toYear": end.year,
                "eventId": None, "eventName": None, "marketTypesCollection": ["WIN", "PLACE"],
                "countriesCollection": ["GB", "IE"], "fileTypeCollection": ["M"]}
        value = self._request("DownloadListOfFiles", body=body)
        if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
            raise HistoryError("Betfair did not return a valid historical file list.")
        return sorted(set(value))

    def download(self, remote):
        return self._request("DownloadFile", params={"filePath": remote}, binary=True)


def local_path(root, remote):
    # An API response supplies a server-side path, never a local destination.
    match = re.fullmatch(r"/data/xds/historic/BASIC/(?:\d{4}/(?:[A-Za-z]{3}|\d{1,2})/\d{1,2}/)?(\d+)/(1\.\d+\.bz2)", remote)
    if not match:
        raise HistoryError("Betfair returned an unrecognised market-file path.")
    path = Path(root).resolve() / "raw" / "BASIC" / match[1] / match[2]
    path.resolve().relative_to(Path(root).resolve())
    return path


def parse_download(compressed, remote, start, end):
    """Require an intact single-market stream before accepting a download."""
    from scripts.extract_betfair_bsp import parse_market
    from scripts.extract_betfair_place_bsp import parse_place_market

    try:
        stream = bz2.decompress(compressed)
        messages = [json.loads(line) for line in stream.splitlines() if line.strip()]
        changes = [mc for message in messages for mc in message.get("mc", [])]
        market_id = PurePosixPath(remote).name.removesuffix(".bz2")
        if not changes or any(mc.get("id") != market_id for mc in changes):
            raise ValueError("market identity")
        definition = next(mc["marketDefinition"] for mc in reversed(changes) if mc.get("marketDefinition"))
        market_type = definition["marketType"]
        day = date.fromisoformat(definition["marketTime"][:10])
        if (definition.get("countryCode") not in {"GB", "IE"} or market_type not in {"WIN", "PLACE"}
                or str(definition.get("eventTypeId")) != "7" or not start <= day <= end):
            raise ValueError("source filter")
        if definition.get("status") != "CLOSED":
            raise ValueError("market not settled")
        rows = (parse_market if market_type == "WIN" else parse_place_market)(stream) or []
        # A genuinely void market has no settled runners; it is still retained.
        if not rows and any(r.get("status") in {"WINNER", "LOSER"} for r in definition.get("runners", [])):
            raise ValueError("settled runners not parsed")
        for row in rows:
            row["bf_market_id"] = market_id
        return market_type, rows
    except (OSError, EOFError, ValueError, KeyError, TypeError, StopIteration, AttributeError):
        raise HistoryError("The historical file is incomplete or fails the requested race/market checks.") from None


class HistoryStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "queue.sqlite"
        with self.connection() as con:
            con.execute("CREATE TABLE IF NOT EXISTS files (remote TEXT PRIMARY KEY, state TEXT NOT NULL, "
                        "market TEXT, rows_json TEXT, sha256 TEXT, bytes INTEGER, error TEXT, updated_at TEXT, "
                        "first_date TEXT NOT NULL, last_date TEXT NOT NULL)")

    @contextmanager
    def connection(self):
        with sqlite3.connect(self.path, timeout=30) as con:
            yield con

    def discover(self, paths, start, end):
        for path in paths:
            local_path(self.root, path)
        with self.connection() as con:
            con.executemany("INSERT OR IGNORE INTO files(remote,state,first_date,last_date) VALUES (?, 'pending',?,?)",
                            [(p, str(start), str(end)) for p in paths])

    def save(self, remote, payload, market, rows):
        target = local_path(self.root, remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".partial")
        try:
            with temporary.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        with self.connection() as con:
            con.execute("UPDATE files SET state='downloaded',market=?,rows_json=?,sha256=?,bytes=?,error=NULL,updated_at=? "
                        "WHERE remote=?", (market, json.dumps(rows), hashlib.sha256(payload).hexdigest(), len(payload), now(), remote))

    def pending(self, budget):
        with self.connection() as con:
            return list(con.execute("SELECT remote,first_date,last_date FROM files "
                                    "WHERE state IN ('pending','retry') ORDER BY remote LIMIT ?", (budget,)))

    def counts(self):
        with self.connection() as con:
            return dict(con.execute("SELECT state,COUNT(*) FROM files GROUP BY state"))


def merge_observations(existing, incoming):
    """Fill absent observations; retain every previously accepted non-null value."""
    def keyed(frame):
        frame = frame.copy()
        if frame.empty:
            return frame.reindex(columns=list(dict.fromkeys([*KEYS, *frame.columns]))).set_index(KEYS)
        frame["race_date"] = pd.to_datetime(frame.race_date).dt.strftime("%Y-%m-%d")
        frame["track_key"] = frame.track.map(bf_track_key)
        frame["off_key"] = frame.off_time.map(normalise_off_time_key)
        frame["horse_key"] = frame.horse_name.map(bf_horse_key)
        if frame.duplicated(KEYS).any():
            raise HistoryError("Ambiguous historical runner keys; publication stopped for review.")
        return frame.set_index(KEYS)
    left, right = keyed(existing), keyed(incoming)
    if left.empty:
        columns = list(dict.fromkeys([*left.columns, *right.columns]))
        return right.reindex(columns=columns).reset_index(), {"new_rows": len(right), "preserved_price_conflicts": 0}
    common = left.index.intersection(right.index)
    conflicts = 0
    for column in right.columns:
        if column not in left:
            left[column] = pd.NA
        prior = left.loc[common, column]
        new = right.loc[common, column]
        both = prior.notna() & new.notna()
        if column in {"bsp", "place_bsp"}:
            conflicts += int((both & pd.to_numeric(prior, errors="coerce").sub(pd.to_numeric(new, errors="coerce")).abs().gt(1e-8)).sum())
        missing = common[prior.isna()]
        left.loc[missing, column] = right.loc[missing, column]
    additions = right.loc[~right.index.isin(left.index)]
    if not additions.empty:
        left = pd.concat([left, additions], sort=False)
    return left.reset_index(), {"new_rows": len(additions), "preserved_price_conflicts": conflicts}


def publish(store, processed):
    with store.connection() as con:
        pending = con.execute("SELECT remote,market,rows_json FROM files WHERE state='downloaded'").fetchall()
    report = {}
    outputs = {"WIN": ["betfair_bsp.parquet", "betfair_bsp_early.parquet", "betfair_timing.parquet"],
               "PLACE": ["betfair_place_bsp.parquet"]}
    for market, names in outputs.items():
        records = [(remote, json.loads(rows)) for remote, kind, rows in pending if kind == market]
        if not records:
            continue
        rows = [row for _, batch in records for row in batch]
        for name in names:
            if not rows:
                continue
            path = Path(processed) / name
            before = pd.read_parquet(path) if path.exists() else pd.DataFrame()
            incoming = pd.DataFrame(rows)
            if name in {"betfair_bsp.parquet", "betfair_bsp_early.parquet"}:
                incoming = incoming.reindex(columns=WIN_COLUMNS)
            if name == "betfair_bsp.parquet":
                incoming["bsp_source"] = "historical_api"
            result, stats = merge_observations(before, incoming)
            atomic_parquet(result, path)
            report[name] = {**stats, "rows": len(result)}
        with store.connection() as con:
            con.executemany("UPDATE files SET state='published' WHERE remote=?", [(remote,) for remote, _ in records])
    return report


def sync_history(start, end, *, client=None, budget=500, workers=4, root=None, processed=None,
                 _source_lock_held=False):
    first, last = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if first > last or last >= date.today() or not 1 <= workers <= 8 or budget < 1:
        raise ValueError("Choose a settled date range, a positive file budget and 1–8 workers")
    client = client or HistoryClient.from_environment()
    root = Path(root or Path(config.DATA_DIR) / "betfair/history")
    processed = Path(processed or config.PROCESSED_DATA_DIR)
    with writer_lock(root / ".writer.lock") as acquired:
        if not acquired:
            return {"state": "busy", "reason": "Another historical exchange download is running"}
        store = HistoryStore(root)
        report = {"state": "running", "start": str(first), "end": str(last), "checked_at": now(),
                  "downloaded": 0, "failed": 0, "unavailable_months": []}
        atomic_json(root / "status.json", report)
        try:
            packages = client.packages()
            months = {p.get("forDate", "")[:7] for p in packages
                      if p.get("sport") == "Horse Racing" and p.get("plan") == "Basic Plan"}
            cursor = first
            while cursor <= last:
                next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
                month_end = min(last, next_month - timedelta(days=1))
                if cursor.strftime("%Y-%m") in months:
                    store.discover(client.list_files(cursor, month_end), cursor, month_end)
                else:
                    report["unavailable_months"].append(cursor.strftime("%Y-%m"))
                cursor = next_month
            def collect(remote, first_date, last_date):
                path = local_path(root, remote)
                earliest, latest = date.fromisoformat(first_date), date.fromisoformat(last_date)
                if path.exists():
                    payload = path.read_bytes()
                    try:
                        market, rows = parse_download(payload, remote, earliest, latest)
                    except HistoryError:
                        # Retain a corrupt local copy for diagnosis and fetch a
                        # replacement instead of retrying that same copy forever.
                        quarantine = path.with_name(path.name+".invalid."+hashlib.sha256(payload).hexdigest()[:16])
                        path.replace(quarantine)
                        payload = client.download(remote)
                        market, rows = parse_download(payload, remote, earliest, latest)
                else:
                    payload = client.download(remote)
                    market, rows = parse_download(payload, remote, earliest, latest)
                store.save(remote, payload, market, rows)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = {pool.submit(collect, remote, start, end): remote for remote, start, end in store.pending(budget)}
                for job in as_completed(jobs):
                    remote = jobs[job]
                    try:
                        job.result()
                        report["downloaded"] += 1
                    except HistoryError as error:
                        report["failed"] += 1
                        if isinstance(error, LoginRequired):
                            report["needs_login"] = True
                        with store.connection() as con:
                            con.execute("UPDATE files SET state='retry',error=?,updated_at=? WHERE remote=?", (str(error), now(), remote))
                    report.update(checked_at=now(), files=store.counts())
                    atomic_json(root / "status.json", report)
            if _source_lock_held:
                report["outputs"] = publish(store, processed)
            else:
                with writer_lock(Path(config.DATA_DIR) / "source_refresh/.writer.lock") as can_publish:
                    if can_publish:
                        report["outputs"] = publish(store, processed)
                    else:
                        report["publication_waiting"] = True
            counts = store.counts()
            report.update(files=counts, checked_at=now())
            report["state"] = "partial" if (report["unavailable_months"] or report["failed"]
                or any(counts.get(state, 0) for state in ("pending", "retry", "downloaded"))) else "ok"
            if report.get("needs_login"):
                report["state"] = "needs_login"
        except (HistoryError, OSError, sqlite3.Error) as error:
            detail = str(error) if isinstance(error, HistoryError) else "Local historical storage could not be updated; progress is retained."
            report.update(state="needs_login" if isinstance(error, LoginRequired) else "error", error=detail, checked_at=now())
        atomic_json(root / "status.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", required=True, help="First race date")
    parser.add_argument("--end", default=(date.today() - timedelta(days=6)).isoformat(), help="Last race date (default: six days ago)")
    parser.add_argument("--max-files", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        result = sync_history(args.start, args.end, budget=args.max_files, workers=args.workers)
    except LoginRequired as error:
        result = {"state": "needs_login", "error": str(error)}
    print(json.dumps(result, indent=2))
    return 0 if result["state"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
