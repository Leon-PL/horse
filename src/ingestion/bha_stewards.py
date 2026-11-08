"""Paced, resumable BHA reports and full PDFs, outside feature building.

One API page normally supplies all fixtures on a day. Complete source bytes and
revisions are retained in SQLite; regular refreshes do not rewrite the archive.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

import config
from src.ingestion.bha_reports import normalise_report, parse_document, race_notes, PARSER_VERSION
from src.ingestion.storage import RawPayloadStore, atomic_json, read_json, writer_lock

logger = logging.getLogger(__name__)
BHA_BASE = 'https://www.britishhorseracing.com'
BHA_APP_JS = BHA_BASE + '/wp-content/themes/bha/library/js/angular/app.js?ver=1.19'
BHA_STEWARDS_API = 'https://api09.horseracing.software/bha/v1/stewards-room/reports'
_DEFAULT_HEADERS = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json, */*',
                    'Origin': BHA_BASE, 'Referer': BHA_BASE + '/racing/stewards-reports/'}
DOCUMENT_HOSTS = {'crate.horseracing.software', 'media.britishhorseracing.com',
                  'www.britishhorseracing.com'}


def now():
    return datetime.now(timezone.utc)


def default_root():
    return Path(config.DATA_DIR) / 'bha'


def code_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('bha_stewards.py', 'bha_reports.py')}


def retry_delay(value):
    try:
        delay = float(value or 0)
    except (ValueError, TypeError):
        try:
            delay = parsedate_to_datetime(value).timestamp() - now().timestamp()
        except (ValueError, TypeError, OverflowError):
            return 0
    return max(0, delay) if math.isfinite(delay) else 0


def connect(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(root / 'reports.sqlite', timeout=30)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.executescript('''
      CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS days(day TEXT PRIMARY KEY,state TEXT NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0,next_retry REAL NOT NULL DEFAULT 0,
        checked_at TEXT,total INTEGER,error TEXT);
      CREATE TABLE IF NOT EXISTS reports(report_key TEXT PRIMARY KEY,race_date TEXT NOT NULL,
        record_hash TEXT NOT NULL,row_json TEXT NOT NULL,observed_at TEXT NOT NULL,
        source_url TEXT,source_sha256 TEXT,source_kind TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS reports_day ON reports(race_date);
      CREATE TABLE IF NOT EXISTS day_reports(day TEXT,report_key TEXT,PRIMARY KEY(day,report_key));
      CREATE TABLE IF NOT EXISTS revisions(report_key TEXT,record_hash TEXT,row_json TEXT,
        observed_at TEXT,source_url TEXT,source_sha256 TEXT,source_kind TEXT,
        PRIMARY KEY(report_key,record_hash));
      CREATE TABLE IF NOT EXISTS documents(report_key TEXT PRIMARY KEY,url TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'pending',sha256 TEXT,parsed_json TEXT,observed_at TEXT,
        checked_at TEXT,next_retry REAL NOT NULL DEFAULT 0,error TEXT);
      CREATE TABLE IF NOT EXISTS document_revisions(report_key TEXT,sha256 TEXT,parsed_json TEXT,
        observed_at TEXT,PRIMARY KEY(report_key,sha256));
    ''')
    return con


def control(con, key, default=None):
    row = con.execute('SELECT value FROM control WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_control(con, key, value):
    con.execute('INSERT INTO control VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                (key, json.dumps(value, allow_nan=False)))


def _extract_bearer_token(session):
    response = session.get(BHA_APP_JS, timeout=30)
    response.raise_for_status()
    matches = re.findall(r"Authorization['\"]\]\s*=\s*['\"]Bearer\s+([^'\"]+)", response.text)
    if not matches:
        raise ValueError('BHA public frontend token not found')
    return matches[-1].strip()


class Deferred(RuntimeError):
    def __init__(self, message, retry_at):
        super().__init__(message)
        self.retry_at = retry_at


class Client:
    """One connection pool and shared request pace; cooldown survives restarts."""
    def __init__(self, con, root, *, request_delay=1.0, session=None):
        if not math.isfinite(request_delay) or request_delay < .5:
            raise ValueError('BHA request delay must be at least 0.5 seconds')
        self.con = con
        self.session = session or requests.Session()
        self.session.headers.update(_DEFAULT_HEADERS)
        self.owns_session = session is None
        self.raw = RawPayloadStore(Path(root) / 'raw.sqlite')
        self.delay, self.last = request_delay, 0.0
        self.requests = 0
        self.seconds = self.wait_seconds = 0.0
        self.token = False
        self.redirecting = False

    def defer(self, message, seconds):
        retry = now().timestamp() + seconds
        with self.con:
            set_control(self.con, 'retry_after', retry)
        raise Deferred(message, retry)

    def get(self, url, *, params=None, document=False):
        retry = control(self.con, 'retry_after', 0)
        if retry > now().timestamp():
            raise Deferred('BHA provider cooldown', retry)
        if not self.token:
            self.session.headers.pop('Authorization', None)
            try:
                token = _extract_bearer_token(self.session)
            except (requests.RequestException, ValueError) as exc:
                response = getattr(exc, 'response', None)
                delay = retry_delay(response.headers.get('Retry-After')) if response is not None else 0
                self.defer('BHA public frontend bootstrap unavailable', max(1200, delay))
            self.session.headers['Authorization'] = 'Bearer ' + token
            self.token = True
            self.last = time.monotonic()
        wait = self.delay - (time.monotonic() - self.last)
        if wait > 0:
            self.wait_seconds += wait
            time.sleep(wait)
        self.last = time.monotonic()
        started = time.monotonic()
        self.requests += 1
        try:
            # Never send the API's token to a PDF host or redirect.
            response = self.session.get(url, params=params,
                headers={'Authorization': None} if document else None,
                timeout=40, allow_redirects=False)
        except requests.RequestException as exc:
            self.seconds += time.monotonic() - started
            self.defer('BHA transport failure: ' + type(exc).__name__, 60)
        self.seconds += time.monotonic() - started
        sha = self.raw.put(response.url, response.status_code, response.content)
        if response.status_code in (301, 302, 303, 307, 308) and document:
            target = urljoin(url, response.headers.get('Location', ''))
            if not valid_document_url(target) or target == url or self.redirecting:
                raise ValueError('Unexpected BHA document redirect')
            self.redirecting = True
            try:
                return self.get(target, document=True)
            finally:
                self.redirecting = False
        if response.status_code in (401, 403, 429) or response.status_code >= 500:
            delay = retry_delay(response.headers.get('Retry-After'))
            self.defer(f'BHA HTTP {response.status_code}', max(1200 if response.status_code < 500 else 60, delay))
        return response, sha

    def close(self):
        if self.owns_session:
            self.session.close()


def valid_document_url(url):
    parsed = urlparse(url)
    return parsed.scheme == 'https' and parsed.hostname in DOCUMENT_HOSTS and not parsed.username


def store_report(con, row, *, source_url, source_sha256, source_kind='api'):
    normal = normalise_report(row)
    encoded = json.dumps(normal, sort_keys=True, ensure_ascii=False, allow_nan=False)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    key, stamp = normal['report_key'], now().isoformat()
    existing = con.execute('SELECT record_hash FROM reports WHERE report_key=?', (key,)).fetchone()
    changed = existing is None or existing[0] != digest
    if changed:
        con.execute('INSERT OR IGNORE INTO revisions VALUES (?,?,?,?,?,?,?)',
                    (key, digest, encoded, stamp, source_url, source_sha256, source_kind))
        first = con.execute('SELECT observed_at FROM revisions WHERE report_key=? AND record_hash=?',
                            (key, digest)).fetchone()[0]
        con.execute('INSERT OR REPLACE INTO reports VALUES (?,?,?,?,?,?,?,?)',
                    (key, normal['race_date'], digest, encoded, first, source_url, source_sha256, source_kind))
    url = normal['pdf_report_url']
    if url:
        refresh = changed and source_kind == 'api'
        con.execute('INSERT INTO documents(report_key,url) VALUES (?,?) ON CONFLICT(report_key) '
                    "DO UPDATE SET state=CASE WHEN url<>excluded.url OR ? THEN 'pending' ELSE state END,"
                    'next_retry=CASE WHEN url<>excluded.url OR ? THEN 0 ELSE next_retry END,url=excluded.url',
                    (key, url, refresh, refresh))
    return changed


def import_legacy(con, path):
    """Keep retained text, without treating any legacy date as fully checked."""
    path = Path(path)
    if control(con, 'legacy_import') or not path.exists():
        return
    frame = pd.read_csv(path, low_memory=False)
    rows = json.loads(frame.to_json(orient='records'))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    known = {row[0] for row in con.execute('SELECT report_key FROM reports')}
    with con:
        for row in rows:
            key = f"{str(row.get('year')).removesuffix('.0')}:{str(row.get('fixture_id')).removesuffix('.0')}"
            if key not in known:
                store_report(con, row, source_url=str(path), source_sha256=sha, source_kind='legacy_csv')
                known.add(key)
        set_control(con, 'legacy_import', {'path': str(path), 'sha256': sha, 'rows': len(rows),
                                         'imported_at': now().isoformat()})


def fetch_day(con, client, day):
    records, seen, page, expected, last_page = [], set(), 1, None, None
    while True:
        response, sha = client.get(BHA_STEWARDS_API, params={'archive': 1, 'ondate': day, 'pagenum': page})
        response.raise_for_status()
        payload = response.json()
        data = payload.get('data')
        if not isinstance(data, list) or int(payload.get('current_page', 0)) != page:
            raise ValueError('Unrecognised BHA pagination response')
        total, last = int(payload['total']), int(payload['last_page'])
        if total < 0 or not 1 <= last <= 100 or (expected is not None and (total, last) != (expected, last_page)):
            raise ValueError('BHA pagination changed while reading the day')
        expected, last_page = total, last
        for row in data:
            normal = normalise_report(row)
            if normal['race_date'] != day or normal['report_key'] in seen:
                raise ValueError('BHA date filter or fixture identity disagrees with requested day')
            seen.add(normal['report_key'])
            records.append((row, response.url, sha))
        if page >= last:
            break
        page += 1
    if len(records) != expected:
        raise ValueError('BHA day is truncated; previous reports retained')
    with con:
        changed = sum(store_report(con, row, source_url=url, source_sha256=sha) for row, url, sha in records)
        con.execute('DELETE FROM day_reports WHERE day=?', (day,))
        con.executemany('INSERT INTO day_reports VALUES (?,?)', [(day, key) for key in seen])
        con.execute("UPDATE days SET state='complete',checked_at=?,total=?,error=NULL,next_retry=0 WHERE day=?",
                    (now().isoformat(), expected, day))
    return changed


def fetch_document(con, client, row):
    if not valid_document_url(row['url']):
        raise ValueError('BHA document URL is not a permitted public report host')
    response, sha = client.get(row['url'], document=True)
    response.raise_for_status()
    parsed = parse_document(response.content)
    from src.utils import normalise_track_key
    expected = json.loads(con.execute('SELECT row_json FROM reports WHERE report_key=?', (row['report_key'],)).fetchone()[0])
    if (parsed['race_date'] != expected['race_date'] or not parsed['course']
            or normalise_track_key(parsed['course']) != normalise_track_key(expected['course'])):
        raise ValueError('BHA PDF date/course identity disagrees with the fixture')
    stamp = now().isoformat()
    with con:
        con.execute('INSERT OR IGNORE INTO document_revisions VALUES (?,?,?,?)',
                    (row['report_key'], sha, json.dumps(parsed, ensure_ascii=False), stamp))
        first = con.execute('SELECT observed_at FROM document_revisions WHERE report_key=? AND sha256=?',
                            (row['report_key'], sha)).fetchone()[0]
        con.execute("UPDATE documents SET state='complete',sha256=?,parsed_json=?,observed_at=?,"
                    "checked_at=?,next_retry=0,error=NULL WHERE report_key=?",
                    (sha, json.dumps(parsed, ensure_ascii=False), first, stamp, row['report_key']))


def read_tables(root=None):
    root = Path(root) if root is not None else default_root()
    if not (root / 'reports.sqlite').exists():
        return {}
    with closing(sqlite3.connect((root / 'reports.sqlite').resolve().as_uri() + '?mode=ro', uri=True)) as con:
        con.row_factory = sqlite3.Row
        con.execute('BEGIN')
        reports = [{**json.loads(row['row_json']), 'observed_at': row['observed_at'],
                    'available_at': row['observed_at'], 'source_url': row['source_url'],
                    'source_sha256': row['source_sha256'], 'source_kind': row['source_kind'],
                    'presence_status': ('unverified_legacy' if row['day_state'] != 'complete' else
                                        'present' if row['day_report_key'] else 'absent_from_latest_day')}
                   for row in con.execute('SELECT r.*,d.state AS day_state,m.report_key AS day_report_key '
                       'FROM reports r LEFT JOIN days d ON d.day=r.race_date LEFT JOIN day_reports m '
                       'ON m.day=r.race_date AND m.report_key=r.report_key ORDER BY r.report_key')]
        documents, races = [], []
        for row in con.execute('SELECT d.*,r.race_date FROM documents d JOIN reports r USING(report_key) '
                               'WHERE d.parsed_json IS NOT NULL ORDER BY report_key'):
            parsed = json.loads(row['parsed_json'])
            base = {'report_key': row['report_key'], 'race_date': row['race_date'],
                    'observed_at': row['observed_at'], 'available_at': row['observed_at'],
                    'source_url': row['url'], 'source_sha256': row['sha256']}
            documents.append({**base, 'text': parsed['text'], 'pages': parsed['pages'],
                              'sections_json': json.dumps(parsed['sections'], ensure_ascii=False),
                              **{key + '_text': parsed['sections'].get(key, '')
                                 for key in ('going', 'rail', 'non_runners', 'suspensions', 'stewards')},
                              'printed_publication_labels_json': json.dumps(parsed['printed_publication_labels']),
                              'replacement_character_count': parsed['replacement_character_count'],
                              'parser_version': parsed['parser_version']})
            races.extend({**base, **block} for block in race_notes(parsed['races']))
    return {name: pd.DataFrame(rows) for name, rows in
            (('reports', reports), ('documents', documents), ('race_notes', races)) if rows}


def refresh_reports(start, end, *, root=None, max_dates=31, max_documents=64, recheck=False,
                    request_delay=1.0, client_factory=Client):
    root = Path(root) if root is not None else default_root()
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or last >= date.today() or max_dates < 1 or max_documents < 1:
        raise ValueError('BHA requires a past date range and positive collection budgets')
    with writer_lock(root / '.collector.lock') as acquired:
        if not acquired:
            return {'state': 'busy', 'reason': 'BHA collector is already running'}
        with closing(connect(root)) as con:
            active_code = code_hashes()
            legacy = Path(config.DATA_DIR) / 'raw/bha_stewards_reports.csv'
            if not control(con, 'legacy_import') and legacy.exists():
                atomic_json(root / 'status.json', {'state': 'importing_legacy', 'checked_at': now().isoformat()})
                try:
                    import_legacy(con, legacy)
                except Exception as exc:
                    atomic_json(root / 'status.json', {'state': 'error', 'checked_at': now().isoformat(), 'error': str(exc)})
                    raise
            with con:
                con.executemany('INSERT OR IGNORE INTO days(day) VALUES (?)',
                                [((first+timedelta(days=i)).isoformat(),) for i in range((last-first).days+1)])
                if recheck:
                    con.execute("UPDATE days SET state='pending' WHERE day BETWEEN ? AND ?", (start, end))
            client = client_factory(con, root, request_delay=request_delay)
            changed = fetched = document_count = failed = 0
            started = time.monotonic()
            def status(state):
                pending = con.execute("SELECT count(*) FROM days WHERE day BETWEEN ? AND ? AND state<>'complete'", (start, end)).fetchone()[0]
                docs = con.execute("SELECT count(*) FROM documents d JOIN reports r USING(report_key) "
                                   "WHERE race_date BETWEEN ? AND ? AND d.state<>'complete'", (start, end)).fetchone()[0]
                body = {'state': state, 'checked_at': now().isoformat(), 'start': start, 'end': end,
                        'reports': con.execute('SELECT count(*) FROM reports').fetchone()[0],
                        'fetched_days': fetched, 'changed_reports': changed, 'fetched_documents': document_count,
                        'pending_days': pending, 'pending_documents': docs, 'failed': failed,
                        'retry_after': control(con, 'retry_after', 0), 'requests': client.requests,
                        'request_seconds': round(client.seconds, 3), 'wait_seconds': round(client.wait_seconds, 3),
                        'seconds': round(time.monotonic()-started, 3), 'parser_version': PARSER_VERSION,
                        'code_hashes': active_code}
                body['available_through'] = end if not pending and not docs else None
                body['publication_lag_days'] = 1
                atomic_json(root / 'status.json', body)
                return body
            status('running')
            try:
                if (root / 'STOP').exists():
                    return status('stopped')
                if control(con, 'retry_after', 0) > now().timestamp():
                    return status('waiting')
                recent = (date.today()-timedelta(days=7)).isoformat()
                rows = con.execute("SELECT * FROM days WHERE day BETWEEN ? AND ? AND next_retry<=? "
                    "AND (state<>'complete' OR (day>=? AND checked_at<?)) ORDER BY day DESC LIMIT ?",
                    (start, end, now().timestamp(), recent, (now()-timedelta(hours=24)).isoformat(), max_dates)).fetchall()
                for row in rows:
                    if code_hashes() != active_code:
                        return status('code_changed')
                    if (root / 'STOP').exists():
                        return status('stopped')
                    with con:
                        con.execute('UPDATE days SET attempts=attempts+1 WHERE day=?', (row['day'],))
                    try:
                        changed += fetch_day(con, client, row['day'])
                        fetched += 1
                    except Deferred:
                        raise
                    except Exception as exc:
                        failed += 1
                        with con:
                            con.execute("UPDATE days SET state='retry',next_retry=?,error=? WHERE day=?",
                                        (now().timestamp()+3600, str(exc), row['day']))
                        logger.warning('BHA day %s deferred: %s', row['day'], exc)
                    status('running')
                rows = con.execute("SELECT d.* FROM documents d JOIN reports r USING(report_key) "
                    "WHERE race_date BETWEEN ? AND ? AND d.next_retry<=? AND (d.state<>'complete' OR "
                    "(race_date>=? AND d.checked_at<?)) ORDER BY race_date DESC,d.report_key LIMIT ?",
                    (start, end, now().timestamp(), recent, (now()-timedelta(hours=24)).isoformat(), max_documents)).fetchall()
                for row in rows:
                    if code_hashes() != active_code:
                        return status('code_changed')
                    if (root / 'STOP').exists():
                        return status('stopped')
                    try:
                        fetch_document(con, client, row)
                        document_count += 1
                    except Deferred:
                        raise
                    except Exception as exc:
                        failed += 1
                        with con:
                            con.execute("UPDATE documents SET state='retry',next_retry=?,error=? WHERE report_key=?",
                                        (now().timestamp()+86400, str(exc), row['report_key']))
                        logger.warning('BHA document %s deferred: %s', row['report_key'], exc)
                    if document_count % 10 == 0:
                        status('running')
                body = status('ok' if fetched or document_count else 'cached')
                return status('partial') if body['pending_days'] or body['pending_documents'] or failed else body
            except Deferred as exc:
                body = status('waiting')
                body['reason'] = str(exc)
                atomic_json(root / 'status.json', body)
                return body
            finally:
                client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default=(date.today()-timedelta(days=14)).isoformat())
    parser.add_argument('--end', default=(date.today()-timedelta(days=1)).isoformat())
    parser.add_argument('--root', type=Path)
    parser.add_argument('--max-dates', type=int, default=31)
    parser.add_argument('--max-documents', type=int, default=64)
    parser.add_argument('--request-delay', type=float, default=1.0)
    parser.add_argument('--recheck', action='store_true')
    parser.add_argument('--drain', action='store_true', help='Drain bounded batches until complete or deferred')
    parser.add_argument('--status', action='store_true', help='Show saved progress without making requests')
    parser.add_argument('--publish', action='store_true', help='Publish accepted sources after collection finishes')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.status:
        print(json.dumps(read_json((args.root or default_root()) / 'status.json')))
        return
    initial_code = code_hashes()
    while True:
        if code_hashes() != initial_code:
            atomic_json((args.root or default_root()) / 'status.json',
                        {'state': 'code_changed', 'checked_at': now().isoformat(), 'reason': 'Review and restart the BHA collector'})
            return
        report = refresh_reports(args.start, args.end, root=args.root, max_dates=args.max_dates,
                                 max_documents=args.max_documents, request_delay=args.request_delay,
                                 recheck=args.recheck)
        args.recheck = False
        print(json.dumps(report), flush=True)
        if args.drain and report['state'] == 'waiting':
            time.sleep(min(30, max(1, report['retry_after']-now().timestamp())))
            continue
        if not args.drain or report['state'] != 'partial':
            break
        if not report['fetched_days'] and not report['fetched_documents'] and not report['failed']:
            break
    if args.publish and report['state'] not in {'stopped', 'busy', 'code_changed'}:
        from src.source_dataset import capture_dataset
        manifest = capture_dataset(include_prices=True)
        print(json.dumps({'published_dataset': manifest['id']}), flush=True)


if __name__ == '__main__':
    main()
