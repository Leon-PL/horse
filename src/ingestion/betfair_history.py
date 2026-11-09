"""Resumable downloads of entitled Betfair Historical market files.

Uses the Historical Data API, not the paid Live Exchange App Key. Full BZ2
streams are retained; existing WIN/PLACE parsers supply the derived caches.
There are deliberately no purchase or betting operations in this client.
"""
from __future__ import annotations

import argparse
import bz2
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as wait_futures
from contextlib import closing, contextmanager
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
from src.betfair import PRICE_KEYS, bf_horse_key, bf_track_key, price_runner_positions
from src.runner_identity import identity_values
from src.ingestion.betfair_credentials import connection_path, load_connection, save_connection
from src.ingestion.storage import atomic_json, atomic_parquet, writer_lock
from src.utils import normalise_off_time_key

API = "https://historicdata.betfair.com/api/"
KEYS = PRICE_KEYS
CHECKPOINT_MARKETS = 128
CHECKPOINT_INTERVAL_SECONDS = 2.0
PROGRESS_INTERVAL_SECONDS = 5.0
WIN_COLUMNS = ["race_date", "track_key", "off_key", "horse_key", "off_time", "track", "horse_name",
               "bf_runner_id", "bsp", "ltp_preoff", "ltp_60s", "ltp_300s", "ltp_3600s", "ltp_1100",
               "ltp_last", "won", "n_runners", "market_id", "bf_market_id"]


class HistoryError(RuntimeError):
    """An actionable error without credentials or raw account responses."""


class LoginRequired(HistoryError):
    pass


LOGIN_FAILURES = {
    "INVALID_USERNAME_OR_PASSWORD": "Check your Betfair username and password, then try again.",
    "INVALID_PASSWORD": "Check your Betfair password, then try again.",
    "CERT_AUTH_REQUIRED": "Betfair could not authenticate the certificate. Check that the public "
                          "client.crt from this app is uploaded to the same Betfair account.",
    "EMAIL_LOGIN_NOT_ALLOWED": "Use your Betfair username rather than your email address.",
    "CHANGE_PASSWORD_REQUIRED": "Sign in on the Betfair website and complete the required password change.",
    "ACCOUNT_PENDING_PASSWORD_CHANGE": "Complete password recovery on the Betfair website first.",
    "CONTACT_VERIFICATION_REQUIRED": "Sign in on the Betfair website and complete its account verification.",
    "INTERNATIONAL_TERMS_ACCEPTANCE_REQUIRED": "Sign in on the Betfair website and accept the updated terms.",
    "PERSONAL_MESSAGE_REQUIRED": "Sign in on the Betfair website and read the pending account message.",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def configured():
    setting = os.getenv("BETFAIR_HISTORY_ENABLED", "").lower()
    if setting in {"0", "false", "no"}:
        return False
    return setting in {"1", "true", "yes"} or connection_path().exists()


def connect_account(username, password, certificate, key=""):
    """Check each setup step and retain only a safe, credential-free outcome."""
    report = {"checked_at": now(), "state": "failed", "stage": "certificate_login"}
    try:
        client = HistoryClient.certificate_login(username, password, str(certificate), str(key) if key else "")
        report["stage"] = "historical_access"
        packages = client.packages()
        if not any(p.get("sport") == "Horse Racing" and p.get("plan") == "Basic Plan" for p in packages):
            raise HistoryError("Betfair sign-in succeeded, but no Horse Racing BASIC months were found in "
                               "My Data. Add the months you need on Betfair's Historical Data website, "
                               "then connect again.")
        report["stage"] = "save_connection"
        save_connection(username, password, certificate, key)
    except HistoryError as error:
        report["message"] = str(error)
    except RuntimeError:
        report["message"] = (
            "Betfair sign-in and historical access succeeded, but Windows could not encrypt the login. "
            "Open the training app under your usual Windows account and reconnect."
            if report["stage"] == "save_connection" else "The connection check could not finish. Try again."
        )
    except OSError:
        report["message"] = (
            "Betfair sign-in and historical access succeeded, but the app could not save the connection. "
            "Check that the app can write to its Betfair settings folder."
            if report["stage"] == "save_connection" else
            "The app could not read the local certificate or private key. Check the certificate files."
        )
    else:
        report.update(state="connected", stage="complete", message="Betfair historical downloads are connected.")
    try:
        atomic_json(connection_path().with_name("connection_status.json"), report)
    except OSError:
        # A diagnostic write must not change the result of saving the connection.
        pass
    return report


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
        except requests.exceptions.SSLError:
            raise LoginRequired("Could not establish a secure certificate connection to Betfair. "
                                "Check the local certificate/key and the computer's network settings.") from None
        except requests.Timeout:
            raise LoginRequired("Betfair certificate sign-in timed out. Try again shortly.") from None
        except requests.RequestException:
            raise LoginRequired("Could not reach Betfair's certificate sign-in service. "
                                "Check the internet connection and try again.") from None
        if response.status_code != 200:
            raise LoginRequired(f"Betfair's certificate sign-in service returned HTTP {response.status_code}. "
                                "Try signing in on the Betfair website and retry the connection.")
        try:
            value = response.json()
        except ValueError:
            raise LoginRequired("Betfair's certificate sign-in service returned an unexpected response. "
                                "Try again shortly.") from None
        if not isinstance(value, dict):
            raise LoginRequired("Betfair's certificate sign-in service returned an unexpected response.")
        status = value.get("loginStatus")
        if isinstance(status, str) and status in LOGIN_FAILURES:
            raise LoginRequired(f"{LOGIN_FAILURES[status]} (Betfair: {status})")
        if status != "SUCCESS" or not isinstance(value.get("sessionToken"), str) or not value["sessionToken"]:
            raise LoginRequired("Betfair did not approve the certificate sign-in. "
                                "Sign in on the Betfair website and check for account notices.")
        return cls(value["sessionToken"])

    def _session(self):
        if self._session_override is not None:
            return self._session_override
        if not hasattr(self._sessions, "http"):
            self._sessions.http = requests.Session()
        return self._sessions.http

    def _request(self, operation, *, body=None, params=None, binary=False):
        method = "GET" if binary or operation == "GetMyData" else "POST"
        for attempt in range(3):
            if self._login_failed:
                raise LoginRequired("The Betfair login session expired; reconnect to resume.")
            with self._lock:
                time.sleep(max(0, self._next_request - time.monotonic()))
                self._next_request = time.monotonic() + self._interval
            try:
                response = self._session().request(method, API + operation,
                    headers={"ssoid": self._token, "Content-Type": "application/json"},
                    json=body if method == "POST" else None, params=params, timeout=(15, 90), allow_redirects=False)
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
        value = self._request("GetMyData")
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


def _resolved_archive_path(path):
    resolved = str(Path(path).resolve())
    # Windows can return either spelling as files appear during parallel downloads.
    # Compare both paths in the extended namespace without skipping link resolution.
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        resolved = "\\\\?\\" + ("UNC\\" + resolved[2:] if resolved.startswith("\\\\") else resolved)
    return Path(resolved)


def _remote_path(remote):
    match = re.fullmatch(r"/(?:data/xds/historic|xds_nfs/edp_processed)/BASIC/"
                        r"(?:\d{4}/(?:[A-Za-z]{3}|\d{1,2})/\d{1,2}/)?(\d+)/(1\.\d+\.bz2)", remote)
    if not match:
        raise HistoryError("Betfair returned an unrecognised market-file path.")
    return match


def local_path(root, remote):
    # An API response supplies a server-side path, never a local destination.
    match = _remote_path(remote)
    path = Path(root).resolve() / "raw" / "BASIC" / match[1] / match[2]
    try:
        _resolved_archive_path(path).relative_to(_resolved_archive_path(root))
    except ValueError:
        raise HistoryError("The Betfair archive target resolves outside the historical-data directory.") from None
    return path


def parse_download(compressed, remote, start, end):
    """Require an intact single-market stream before accepting a download."""
    from src.ingestion.betfair_streams import parse_market, parse_place_market

    try:
        stream = bz2.decompress(compressed)
        messages = [json.loads(line) for line in stream.splitlines() if line.strip()]
        changes = [mc for message in messages for mc in message.get("mc", [])]
        market_id = PurePosixPath(remote).name.removesuffix(".bz2")
        if not changes or any(mc.get("id") != market_id for mc in changes):
            raise ValueError("market identity")
        definition = next(mc["marketDefinition"] for mc in reversed(changes) if mc.get("marketDefinition"))
        market_type = definition["marketType"]
        runners = definition.get("runners") or []
        if (market_type in {"WIN", "PLACE"} and str(definition.get("eventTypeId")) == "7"
                and definition.get("status") == "CLOSED" and runners
                and all(r.get("status") == "REMOVED" for r in runners)):
            # Every runner withdrawn: a void market (an abandoned meeting, a
            # voided duplicate, a "Do Not Activate" placeholder). Betfair files
            # many under the day it voided them or with a stray country code
            # ("MO"); with nothing to publish, retain it before those checks
            # instead of sending it to review.
            return market_type, []
        day = date.fromisoformat(definition["marketTime"][:10])
        if (definition.get("countryCode") not in {"GB", "IE"} or market_type not in {"WIN", "PLACE"}
                or str(definition.get("eventTypeId")) != "7" or not start <= day <= end):
            raise ValueError("source filter")
        if definition.get("status") != "CLOSED":
            raise ValueError("market not settled")
        if (market_type == "WIN" and definition.get("bspMarket") is False
                and str(definition.get("name", "")).casefold().startswith("match bet")):
            # Betfair files a few match bets ("Match Betting 1": two runners,
            # no BSP) as WIN markets of the race. They are not its win market,
            # and their "winner" only beat one rival; retain, publish nothing.
            # A race run without BSP (a charity race) is still its win market.
            return market_type, []
        rows =(parse_market if market_type == "WIN" else parse_place_market)(stream) or []
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
            con.execute("CREATE TABLE IF NOT EXISTS discovery (first_date TEXT,last_date TEXT,checked_at TEXT,"
                        "PRIMARY KEY(first_date,last_date))")
            columns = {row[1] for row in con.execute("PRAGMA table_info(files)")}
            for name, definition in (("attempts", "INTEGER NOT NULL DEFAULT 0"), ("retry_after", "TEXT")):
                if name not in columns:
                    con.execute(f"ALTER TABLE files ADD COLUMN {name} {definition}")
            con.execute("CREATE INDEX IF NOT EXISTS files_state ON files(state)")

    @contextmanager
    def connection(self):
        with closing(sqlite3.connect(self.path, timeout=30)) as con:
            with con:
                yield con

    def listing_known(self, start, end):
        with self.connection() as con:
            row = con.execute("SELECT MAX(checked_at) FROM discovery WHERE first_date<=? AND last_date>=?",
                              (str(start), str(end))).fetchone()
        if not row[0]:
            return False
        return (end < date.today()-timedelta(days=30)
                or datetime.fromisoformat(row[0]) > datetime.now(timezone.utc)-timedelta(hours=6))

    def discover(self, paths, start, end):
        for path in paths:
            _remote_path(path)
        with self.connection() as con:
            con.executemany("INSERT OR IGNORE INTO files(remote,state,first_date,last_date) VALUES (?, 'pending',?,?)",
                            [(p, str(start), str(end)) for p in paths])
            con.execute("INSERT OR REPLACE INTO discovery VALUES (?,?,?)", (str(start), str(end), now()))

    def save(self, remote, payload, market, rows, *, retain_file=True):
        return self.save_many([{"remote": remote, "payload": payload, "market": market,
                                "rows": rows, "retain_file": retain_file}])

    def save_many(self, records):
        """Durably checkpoint a group, retaining raw files before accepting rows."""
        if not records:
            return {}
        remotes = [item["remote"] for item in records]
        if len(set(remotes)) != len(remotes):
            raise HistoryError("Duplicate queue entries in a checkpoint; publication stopped for review.")
        encoded = []
        for item in records:
            remote, payload = item["remote"], item["payload"]
            _remote_path(remote)
            if item.get("retain_file", True):
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
            encoded.append((item["market"], json.dumps(item["rows"]), hashlib.sha256(payload).hexdigest(),
                            len(payload), now(), remote))
        with self.connection() as con:
            placeholders = ",".join("?" for _ in remotes)
            previous = dict(con.execute(f"SELECT state,COUNT(*) FROM files WHERE remote IN ({placeholders}) "
                                        "GROUP BY state", remotes))
            if sum(previous.values()) != len(records):
                raise HistoryError("Queue entries changed during a checkpoint; accepted prices were preserved.")
            con.executemany("UPDATE files SET state='downloaded',market=?,rows_json=?,sha256=?,bytes=?,"
                            "error=NULL,updated_at=? WHERE remote=?", encoded)
        return previous

    def pending(self, budget, *, local_only=False):
        from src.ingestion.betfair_archives import market_ids
        available = market_ids(self.root) if local_only else set()
        with self.connection() as con:
            rows = con.execute("SELECT remote,first_date,last_date FROM files "
                                    "WHERE state='pending' OR (state='retry' AND (retry_after IS NULL OR retry_after<=?)) "
                                    "ORDER BY first_date DESC,remote", (now(),))
            result = []
            for row in rows:
                if not local_only or PurePosixPath(row[0]).name.removesuffix(".bz2") in available:
                    result.append(row)
                    if len(result) >= budget:
                        break
            return result

    def counts(self):
        with self.connection() as con:
            return dict(con.execute("SELECT state,COUNT(*) FROM files GROUP BY state"))


def merge_observations(existing, incoming):
    """Preserve accepted values by selection ID, with unique-name fallback.

    An unidentified legacy row cannot be assigned to either namesake. Return it
    for durable review before identified observations replace it in the cache.
    """
    def keyed(frame):
        frame = frame.copy().reset_index(drop=True)
        if frame.empty:
            return frame.reindex(columns=list(dict.fromkeys([*KEYS, *frame.columns])))
        frame["race_date"] = pd.to_datetime(frame.race_date).dt.strftime("%Y-%m-%d")
        frame["track_key"] = frame.track.map(bf_track_key)
        frame["off_key"] = frame.off_time.map(normalise_off_time_key)
        frame["horse_key"] = frame.horse_name.map(bf_horse_key)
        return frame
    def review_records(frame):
        # Standard JSON preserves the complete float representation; pandas'
        # default JSON precision would round legacy price observations.
        records = frame.astype(object).where(frame.notna(), None).to_dict("records")
        return json.loads(json.dumps(records, default=str, allow_nan=False))
    left, right = keyed(existing), keyed(incoming)
    review = []
    left_ids = identity_values(left.get("bf_runner_id", pd.Series(pd.NA, index=left.index)))
    native = pd.MultiIndex.from_arrays([left[name] for name in KEYS[:3]] + [left_ids])
    duplicate_ids = left_ids.notna().to_numpy() & native.duplicated(keep=False)
    duplicates = left.loc[duplicate_ids]
    groups = {}
    for position in duplicates.index:
        groups.setdefault(native[position], []).append(position)
    removed = []
    for positions in groups.values():
        # These are observations of one proven selection, sometimes retained
        # under different country/name spellings. Keep accepted values in their
        # existing order, fill gaps, and preserve every superseded row for review.
        first, *others = positions
        review.extend(review_records(left.loc[others]))
        for column in left:
            if pd.isna(left.at[first, column]):
                values = left.loc[others, column].dropna()
                if not values.empty:
                    left.at[first, column] = values.iloc[0]
        removed.extend(others)
    left = left.drop(index=removed).reset_index(drop=True)
    known_right = right.loc[identity_values(right.get("bf_runner_id", pd.Series(pd.NA, index=right.index))).notna()]
    namesakes = pd.MultiIndex.from_frame(known_right.loc[known_right.duplicated(KEYS, keep=False), KEYS])
    unidentified = identity_values(left.get("bf_runner_id", pd.Series(pd.NA, index=left.index))).isna()
    unresolved = unidentified.to_numpy() & pd.MultiIndex.from_frame(left[KEYS]).isin(namesakes)
    review.extend(review_records(left.loc[unresolved]))
    left = left.loc[~unresolved].reset_index(drop=True)
    try:
        positions = price_runner_positions(right, left, skip_ambiguous=False)
    except ValueError:
        raise HistoryError("Ambiguous historical runner identities; publication stopped for review.") from None
    report = {"new_rows": int((positions < 0).sum()), "preserved_price_conflicts": 0}
    if review:
        report["identity_review"] = review
    if left.empty:
        columns = list(dict.fromkeys([*left.columns, *right.columns]))
        return right.reindex(columns=columns).reset_index(drop=True), report
    matched_right = right.index[positions >= 0]
    matched_left = positions[positions >= 0]
    for column in right.columns:
        if column not in left:
            left[column] = pd.NA
        prior = left.loc[matched_left, column].reset_index(drop=True)
        new = right.loc[matched_right, column].reset_index(drop=True)
        both = prior.notna() & new.notna()
        if column in {"bsp", "place_bsp"}:
            report["preserved_price_conflicts"] += int((both & pd.to_numeric(prior, errors="coerce").sub(pd.to_numeric(new, errors="coerce")).abs().gt(1e-8)).sum())
        missing = (prior.isna() & new.notna()).to_numpy()
        left.loc[matched_left[missing], column] = new.loc[missing].to_numpy()
    additions = right.loc[positions < 0]
    if not additions.empty:
        left = pd.concat([left, additions], sort=False)
    return left.reset_index(drop=True), report


def restore_place_runner_ids(store):
    """Upgrade old PLACE parses from their exact retained raw revision.

    Previously published old parses are restaged so the ordinary publisher also
    fills their cache identities; no separate manual backfill is needed.
    """
    from src.ingestion.betfair_archives import store_path
    archive = store_path(store.root)
    with store.connection() as con:
        staged = con.execute("SELECT remote,rows_json,sha256,first_date,last_date FROM files "
                             "WHERE state IN ('downloaded','published') AND market='PLACE'").fetchall()
    def replay(entry):
        remote, body, digest, first, last = entry
        old = json.loads(body)
        if all(row.get("bf_runner_id") is not None for row in old):
            return None
        path = local_path(store.root, remote)
        raw = path.read_bytes() if path.exists() else None
        if raw is None or hashlib.sha256(raw).hexdigest() != digest:
            raw = None
            if archive.exists():
                with closing(sqlite3.connect(archive.resolve().as_uri()+"?mode=ro", uri=True, timeout=30)) as con:
                    retained = con.execute("SELECT body FROM payloads WHERE sha256=? AND verified=1", (digest,)).fetchone()
                if retained and hashlib.sha256(retained[0]).hexdigest() == digest:
                    raw = retained[0]
        if raw is None:
            raise HistoryError("Retained PLACE source revision is missing; staged observations are preserved for review.")
        market, parsed = parse_download(raw, remote, date.fromisoformat(first), date.fromisoformat(last))
        if market != "PLACE" or len(old) != len(parsed) or any(
            any(key != "bf_runner_id" and new.get(key) != value for key, value in prior.items())
            or (prior.get("bf_runner_id") is not None and prior["bf_runner_id"] != new["bf_runner_id"])
            for prior, new in zip(old, parsed)):
            raise HistoryError("Replayed PLACE observations changed; existing staged values are preserved for review.")
        rows = [{**prior, "bf_runner_id": new["bf_runner_id"]} for prior, new in zip(old, parsed)]
        return {"remote": remote, "payload": raw, "market": market, "rows": rows, "retain_file": False}
    restored = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(staged), CHECKPOINT_MARKETS):
            # Bound parsed bodies and commit only after the whole group verifies.
            records = [record for record in pool.map(replay, staged[offset:offset+CHECKPOINT_MARKETS])
                       if record is not None]
            store.save_many(records)
            restored += len(records)
    return restored


def numeric_prices(frame):
    """Store starting prices as numbers; a price Betfair did not reconcile is missing.

    Stream files write an unreconciled BSP as the text "NaN" (and could write
    "Infinity"); kept as text it blocks the typed Parquet write of every row.
    """
    for column in ("bsp", "place_bsp"):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").astype(float)
            frame[column] = values.where(values.abs() != float("inf"))
    return frame


def publish(store, processed):
    restored = restore_place_runner_ids(store)
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
            incoming = numeric_prices(pd.DataFrame(rows))
            if name in {"betfair_bsp.parquet", "betfair_bsp_early.parquet"}:
                incoming = incoming.reindex(columns=WIN_COLUMNS)
            if name == "betfair_bsp.parquet":
                incoming["bsp_source"] = "historical_api"
            result, stats = merge_observations(before, incoming)
            review = stats.pop("identity_review", None)
            if review:
                digest = hashlib.sha256(json.dumps(review, sort_keys=True).encode()).hexdigest()
                review_path = store.root / "identity_review" / f"{name}.{digest}.json"
                atomic_json(review_path, {"cache": name, "reason": "Legacy ambiguous names or repeated selection observations",
                                          "observations": review})
                stats.update(identity_review_rows=len(review), identity_review_path=str(review_path))
            if market == "PLACE" and restored:
                stats["restored_identity_markets"] = restored
            atomic_parquet(result, path)
            report[name] = {**stats, "rows": len(result)}
        with store.connection() as con:
            con.executemany("UPDATE files SET state='published' WHERE remote=?", [(remote,) for remote, _ in records])
    return report


def sync_history(start, end, *, client=None, budget=500, workers=4, root=None, processed=None,
                 _source_lock_held=False, local_only=False):
    first, last = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if first > last or last >= date.today() or not 1 <= workers <= 8 or budget < 1:
        raise ValueError("Choose a settled date range, a positive file budget and 1–8 workers")
    root = Path(root or Path(config.DATA_DIR) / "betfair/history")
    processed = Path(processed or config.PROCESSED_DATA_DIR)
    with writer_lock(root / ".writer.lock") as acquired:
        if not acquired:
            return {"state": "busy", "reason": "Another historical exchange download is running"}
        client = client or HistoryClient.from_environment()
        store = HistoryStore(root)
        report = {"state": "running", "start": str(first), "end": str(last), "checked_at": now(),
                  "downloaded": 0, "reused_local": 0, "processed": 0, "failed": 0, "unavailable_months": [],
                  "pid": os.getpid(), "replaying_archives": local_only, "deferred_local": 0}
        atomic_json(root / "status.json", report)
        try:
            packages = client.packages()
            months = {p.get("forDate", "")[:7] for p in packages
                      if p.get("sport") == "Horse Racing" and p.get("plan") == "Basic Plan"}
            cursor = first
            report["stage"] = "discovering_markets"
            next_update = time.monotonic()+PROGRESS_INTERVAL_SECONDS
            while cursor <= last:
                next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
                month_end = min(last, next_month - timedelta(days=1))
                changed = False
                if cursor.strftime("%Y-%m") in months:
                    if not store.listing_known(cursor, month_end):
                        store.discover(client.list_files(cursor, month_end), cursor, month_end)
                        changed = True
                else:
                    report["unavailable_months"].append(cursor.strftime("%Y-%m"))
                report["listed_through"] = str(month_end)
                if changed or time.monotonic() >= next_update:
                    report.update(checked_at=now(), files=store.counts())
                    atomic_json(root / "status.json", report)
                    next_update = time.monotonic()+PROGRESS_INTERVAL_SECONDS
                cursor = next_month
            report["stage"] = "importing_markets"
            queue_counts = store.counts()
            collection_started = time.monotonic()
            def collect(remote, first_date, last_date):
                from src.ingestion.betfair_archives import payloads as archived_payloads
                path = local_path(root, remote)
                earliest, latest = date.fromisoformat(first_date), date.fromisoformat(last_date)
                payload = None
                origin = "reused_local"
                if path.exists():
                    payload = path.read_bytes()
                    try:
                        market, rows = parse_download(payload, remote, earliest, latest)
                    except HistoryError:
                        # Retain a corrupt local copy for diagnosis and fetch a
                        # replacement instead of retrying that same copy forever.
                        quarantine = path.with_name(path.name+".invalid."+hashlib.sha256(payload).hexdigest()[:16])
                        path.replace(quarantine)
                        payload = None
                if payload is None:
                    for _, candidate in archived_payloads(root, PurePosixPath(remote).name.removesuffix(".bz2")):
                        try:
                            market, rows = parse_download(candidate, remote, earliest, latest)
                        except HistoryError:
                            continue
                        payload = candidate
                        break
                if payload is None:
                    if local_only:
                        return {"origin": "deferred_local"}
                    payload = client.download(remote)
                    market, rows = parse_download(payload, remote, earliest, latest)
                    origin = "downloaded"
                return {"remote": remote, "payload": payload, "market": market, "rows": rows,
                        "retain_file": origin == "downloaded", "origin": origin}

            buffered = []
            next_checkpoint = time.monotonic()+CHECKPOINT_INTERVAL_SECONDS
            def checkpoint():
                nonlocal next_checkpoint
                if buffered:
                    previous = store.save_many(buffered)
                    for state, count in previous.items():
                        queue_counts[state] = queue_counts.get(state, 0)-count
                    queue_counts["downloaded"] = queue_counts.get("downloaded", 0)+len(buffered)
                    for item in buffered:
                        report[item["origin"]] += 1
                        report["processed"] += 1
                    buffered.clear()
                next_checkpoint = time.monotonic()+CHECKPOINT_INTERVAL_SECONDS

            with ThreadPoolExecutor(max_workers=workers) as pool:
                work = iter(store.pending(budget, local_only=local_only))
                jobs = {}
                stop_scheduling = False
                def submit_next():
                    if stop_scheduling:
                        return
                    item = next(work, None)
                    if item is not None:
                        remote, first_date, last_date = item
                        jobs[pool.submit(collect, remote, first_date, last_date)] = remote
                for _ in range(workers*2):
                    submit_next()
                next_update = time.monotonic()+PROGRESS_INTERVAL_SECONDS
                while jobs:
                    completed, _ = wait_futures(jobs, return_when=FIRST_COMPLETED)
                    for job in completed:
                        remote = jobs.pop(job)
                        try:
                            item = job.result()
                            if item["origin"] == "deferred_local":
                                report["deferred_local"] += 1
                            else:
                                buffered.append(item)
                        except HistoryError as error:
                            report["failed"] += 1
                            if isinstance(error, LoginRequired):
                                report["needs_login"] = True
                                stop_scheduling = True
                            with store.connection() as con:
                                attempts, previous_state = con.execute(
                                    "SELECT attempts,state FROM files WHERE remote=?", (remote,)).fetchone()
                                attempts += 1
                                state = "review" if attempts >= 3 else "retry"
                                con.execute("UPDATE files SET state=?,attempts=?,retry_after=?,error=?,updated_at=? WHERE remote=?",
                                            (state, attempts, (datetime.now(timezone.utc)+timedelta(minutes=5*attempts)).isoformat(),
                                             str(error), now(), remote))
                            queue_counts[previous_state] -= 1
                            queue_counts[state] = queue_counts.get(state, 0)+1
                        if len(buffered) >= CHECKPOINT_MARKETS or time.monotonic() >= next_checkpoint:
                            checkpoint()
                        submit_next()
                    if time.monotonic() >= next_update:
                        report.update(checked_at=now(), files={k: v for k, v in queue_counts.items() if v},
                                      import_seconds=round(time.monotonic()-collection_started, 3))
                        atomic_json(root / "status.json", report)
                        next_update = time.monotonic()+PROGRESS_INTERVAL_SECONDS
                checkpoint()
            report["import_seconds"] = round(time.monotonic()-collection_started, 3)
            report["stage"] = "publishing_prices"
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
                or any(counts.get(state, 0) for state in ("pending", "retry", "downloaded", "review"))) else "ok"
            if report.get("needs_login"):
                report["state"] = "needs_login"
        except (HistoryError, OSError, sqlite3.Error) as error:
            detail = str(error) if isinstance(error, HistoryError) else "Local historical storage could not be updated; progress is retained."
            report.update(state="needs_login" if isinstance(error, LoginRequired) else "error", error=detail, checked_at=now())
        atomic_json(root / "status.json", report)
        return report


def drain_history(start, end, *, budget=5000, workers=4):
    """Resume the canonical queue in bounded batches until caught up or stopped."""
    root = Path(config.DATA_DIR) / "betfair/history"
    with writer_lock(root / ".backfill.lock") as acquired:
        if not acquired:
            return {"state": "busy", "reason": "A Betfair historical backfill is already running"}
        sources = [Path(__file__).with_name(name) for name in
                   ("betfair_history.py", "betfair_streams.py", "betfair_archives.py", "betfair_credentials.py", "storage.py")]
        sources.extend(Path(__file__).parents[1] / name for name in ("betfair.py", "runner_identity.py"))
        def hashes():
            return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        pins = hashes()
        started = now()
        total = {"downloaded": 0, "reused_local": 0, "processed": 0}
        while True:
            if (root / "STOP").exists() or hashes() != pins:
                report = {"state": "stopped", "reason": "Stop requested or collector code changed"}
            else:
                with writer_lock(root / ".archive.lock") as archive_ready:
                    pass
                atomic_json(root / "backfill_status.json", {
                    "state": "running", "mode": "backfill", "pid": os.getpid(), "started_at": started,
                    "checked_at": now(), "totals": dict(total), "code_hashes": pins,
                    "replaying_archives": not archive_ready})
                try:
                    report = sync_history(start, end, budget=budget, workers=workers, local_only=not archive_ready)
                except HistoryError as error:
                    report = {"state": "needs_login", "error": str(error)}
            for key in total:
                total[key] += report.get(key, 0)
            progress = {**report, "mode": "backfill", "pid": os.getpid(), "started_at": started,
                        "checked_at": now(), "totals": dict(total), "code_hashes": pins}
            counts = report.get("files", {})
            no_work = not any(counts.get(state, 0) for state in ("pending", "retry", "downloaded"))
            if report["state"] == "partial" and no_work and counts.get("review"):
                progress["state"] = "review"
            atomic_json(root / "backfill_status.json", progress)
            if progress["state"] in {"ok", "error", "needs_login", "stopped", "review"}:
                return progress
            if no_work and report.get("unavailable_months"):
                return progress
            if not report.get("processed"):
                time.sleep(10 if report["state"] in {"busy", "waiting_for_archives"} or report.get("publication_waiting") else 30)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", required=True, help="First race date")
    parser.add_argument("--end", default=(date.today() - timedelta(days=6)).isoformat(), help="Last race date (default: six days ago)")
    parser.add_argument("--max-files", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--drain", action="store_true", help="Continue bounded batches until the saved queue is caught up")
    args = parser.parse_args()
    try:
        collect = drain_history if args.drain else sync_history
        result = collect(args.start, args.end, budget=args.max_files, workers=args.workers)
    except LoginRequired as error:
        result = {"state": "needs_login", "error": str(error)}
    print(json.dumps(result, indent=2))
    return 0 if result["state"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
