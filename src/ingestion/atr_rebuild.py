"""Fresh, resumable ATR reconstruction. Accepted history is never overwritten.

The old parsed cache supplies native-ID hints only. Every accepted row comes
from source HTML parsed by the current collector and checked against the
canonical field. SQLite commits each field; small daily Parquets are exports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import time
import zlib

import pandas as pd

import config
from src import database
from src.ingestion import attheraces as atr
from src.ingestion import atr_ids
from src.ingestion.storage import RawPayloadStore, atomic_json, atomic_parquet, read_json, writer_lock
from src.ingestion.atr_tabs import COLLECTION_VERSION, PERFORMANCE_TABS, TERMINAL_TAB_STATES, inventory

logger = logging.getLogger(__name__)
DEFAULT_ROOT = Path(config.DATA_DIR)/'atr_cache/rebuild_v2'
# Only source extraction and queue acceptance govern the saved raw records.
# Feature derivation has its own FE signature and is not executed by this worker.
CODE_FILES = ('src/ingestion/attheraces.py', 'src/ingestion/atr_rebuild.py',
              'src/ingestion/atr_tabs.py', 'src/ingestion/atr_jumps.py', 'src/ingestion/atr_ids.py')
STATUS_SECONDS = 10.0
EXPORT_SECONDS = 180.0
EXPORT_RACES = 50


def utc_now():
    return datetime.now(timezone.utc)


def code_hashes():
    return {name: hashlib.sha256((Path(config.BASE_DIR)/name).read_bytes()).hexdigest() for name in CODE_FILES}


def _key(day, track, off):
    # Inputs have already been batch-normalised to ISO dates. Avoid invoking
    # pandas' scalar date inference hundreds of thousands of times.
    return f'{str(day)[:10]}|{atr._normalise_track_key(track)}|{atr._normalise_off_time_key(off)}'


def connect(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(root/'rebuild.sqlite', timeout=30)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.executescript('''
        CREATE TABLE IF NOT EXISTS jobs (
          race_id TEXT PRIMARY KEY, race_key TEXT NOT NULL, race_date TEXT NOT NULL,
          track TEXT NOT NULL, off_time TEXT NOT NULL, expected TEXT NOT NULL,
          native_id TEXT, state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
          next_retry REAL NOT NULL DEFAULT 0, error TEXT, updated_at TEXT);
        CREATE INDEX IF NOT EXISTS jobs_ready ON jobs(state,next_retry,race_date);
        CREATE INDEX IF NOT EXISTS jobs_date ON jobs(race_date,off_time);
        CREATE INDEX IF NOT EXISTS jobs_native_id ON jobs(native_id) WHERE native_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS jobs_active_order ON jobs(race_date DESC,off_time DESC,native_id IS NULL)
          WHERE state IN ('pending','retry');
        CREATE INDEX IF NOT EXISTS jobs_active_native ON jobs(race_date DESC,off_time DESC,native_id IS NULL)
          WHERE state IN ('pending','retry') AND native_id IS NOT NULL;
        CREATE TABLE IF NOT EXISTS runners (
          race_id TEXT NOT NULL, horse_key TEXT NOT NULL, payload TEXT NOT NULL,
          PRIMARY KEY(race_id,horse_key));
        CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tab_coverage (
          race_id TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
          receipt TEXT, updated_at TEXT, PRIMARY KEY(race_id,kind));
        CREATE INDEX IF NOT EXISTS tabs_state ON tab_coverage(kind,state);
        CREATE TABLE IF NOT EXISTS dirty_days (race_date TEXT PRIMARY KEY);
    ''')
    atr_ids.ensure_schema(con)
    return con


def control(con, key, default=None):
    row = con.execute('SELECT value FROM control WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_control(con, key, value):
    con.execute('INSERT INTO control VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                (key, json.dumps(value, allow_nan=False)))


def audit_legacy_hints(con, cache=None):
    if control(con,'legacy_hint_audit_version',0)==1:
        return
    cache = atr.load_atr_cache() if cache is None else cache
    jobs = [dict(row) for row in con.execute('SELECT * FROM jobs')]
    if cache.empty:
        plan = {'supported':{},'changes':{},'cleared_conflicts':[]}
    else:
        plan = atr_ids.plan_legacy_hints(jobs,cache)
    path = Path(atr.ATR_CACHE_FILE)
    sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    atr_ids.apply_legacy_plan(con,plan,str(path),sha)
    set_control(con,'legacy_hint_audit_version',1)
    set_control(con,'legacy_hint_audit',{'supported':len(plan['supported']),'changes':plan['changes'],
                                      'source_sha256':sha,'checked_at':utc_now().isoformat()})


def seed_tab_coverage(con):
    for kind in PERFORMANCE_TABS:
        con.execute("INSERT OR IGNORE INTO tab_coverage(race_id,kind,state) SELECT race_id,?,'pending' FROM jobs", (kind,))


def record_tabs(con, race_id, receipts):
    job = con.execute('SELECT expected FROM jobs WHERE race_id=?',(race_id,)).fetchone()
    expected = {r['horse_key']:r['horse_name'] for r in json.loads(job[0])} if job else {}
    for receipt in receipts:
        if receipt.get('kind') not in PERFORMANCE_TABS:
            raise ValueError('Unregistered performance endpoint')
        receipt = dict(receipt)
        if receipt.get('state') == 'data' and 'signal_runner_count' not in receipt:
            receipt['state'] = 'legacy_received'
        if 'horse_names' in receipt:
            observed = {atr._normalise_horse_key(name) for name in receipt['horse_names']}
            receipt['missing_canonical_runners'] = [expected[k] for k in sorted(set(expected)-observed)]
            receipt['expected_runners'] = len(expected)
        con.execute('INSERT INTO tab_coverage VALUES (?,?,?,?,?) ON CONFLICT(race_id,kind) DO UPDATE SET '
                    'state=excluded.state,receipt=excluded.receipt,updated_at=excluded.updated_at',
                    (race_id,receipt['kind'],receipt['state'],json.dumps(receipt,allow_nan=False),utc_now().isoformat()))


def upgrade_tab_coverage(root=DEFAULT_ROOT):
    """Explicit, resumable contract upgrade; keep every saved runner unchanged.

    Invoke only after stopping and reviewing the collector. Completed three-tab
    fields become pending additions. Retained raw fragments avoid redownloading
    them. This is not a reset or an import of old parsed observations.
    """
    root = Path(root)
    with writer_lock(root/'.worker.lock') as free, writer_lock(Path(atr.ATR_CACHE_DIR)/'.collector.lock') as collector:
        if not free or not collector:
            raise RuntimeError('Stop the ATR collector before upgrading its tab contract')
        con = connect(root)
        try:
            old_hashes = control(con,'code_hashes')
            if not old_hashes:
                raise RuntimeError('No prepared queue to upgrade')
            old_rows = con.execute('SELECT count(*) FROM runners').fetchone()[0]
            by_race = {}
            for race_id, payload in con.execute('SELECT race_id,payload FROM runners'):
                row = json.loads(payload)
                entry = by_race.setdefault(race_id, {'ids':set(),'current':True,'receipts':row.get('atr_source_payloads',[])})
                entry['ids'].add(str(row['atr_raceid']))
                entry['current'] &= str(row.get('atr_collection_version')) == str(COLLECTION_VERSION)
            reopened = 0
            with con:
                seed_tab_coverage(con)
                for race_id, entry in by_race.items():
                    if len(entry['ids']) != 1:
                        raise ValueError(f'Conflicting native IDs for {race_id}')
                    native_id = next(iter(entry['ids']))
                    con.execute('UPDATE jobs SET native_id=? WHERE race_id=?', (native_id,race_id))
                    receipts = [{**r,'state':r.get('state', 'not_found' if r['status']==404 else 'data')}
                                for r in entry['receipts']]
                    record_tabs(con,race_id,receipts)
                    if not entry['current']:
                        reopened += con.execute("UPDATE jobs SET state='pending',next_retry=0,error=? "
                                                "WHERE race_id=? AND state IN ('data','data_partial','no_data')",
                                                ('Collect newly required jumps/distance tabs',race_id)).rowcount
                reopened += con.execute("UPDATE jobs SET state='pending',next_retry=0,error=? WHERE state='no_data' "
                                        "AND race_id NOT IN (SELECT race_id FROM runners)",
                                        ('Recheck absence under five-tab contract',)).rowcount
                set_control(con,'code_hashes',code_hashes())
                set_control(con,'collection_version',COLLECTION_VERSION)
            assert con.execute('SELECT count(*) FROM runners').fetchone()[0] == old_rows
            receipt = {'upgraded_at':utc_now().isoformat(),'collection_version':COLLECTION_VERSION,
                       'saved_rows_preserved':old_rows,'reopened_races':reopened,
                       'old_code_hashes':old_hashes,'code_hashes':code_hashes(),
                       'required_performance_tabs':list(PERFORMANCE_TABS)}
            manifest = read_json(root/'manifest.json')
            manifest.update(collection_version=COLLECTION_VERSION,code_hashes=code_hashes(),tab_upgrade=receipt)
            atomic_json(root/'manifest.json',manifest)
            atomic_json(root/'tab_upgrade.json',receipt)
            publish_status(con,root,'upgraded')
            return receipt
        finally:
            con.close()


def audit_navigation(con, fetcher, job, root):
    """Hourly page check; flag new tabs instead of silently ignoring them."""
    now = utc_now().timestamp()
    if control(con,'inventory_retry',0) > now:
        return
    if control(con,'discovery_retry',0) > now:
        with con:
            set_control(con,'tab_inventory',{**control(con,'tab_inventory',{}),
                        'state':'deferred','reason':'HTML discovery cooldown',
                        'retry_after':control(con,'discovery_retry',0)})
        return
    if control(con,'global_retry',0) > now:
        return
    cutoff_day = (utc_now()-pd.Timedelta(hours=atr._PUBLICATION_DELAY_HOURS)).date().isoformat()
    # A field can finish just after the shared cooldown expires. Give the next
    # queue iteration the results lookup before another tab inventory request.
    if _discovery_day(con, now, cutoff_day) is not None:
        return
    day = atr._date_to_atr(job['race_date'])
    off = atr._off_time_to_hhmm(job['off_time'])
    if atr._is_bst(job['race_date']):
        off = atr._add_one_hour(off)
    url = f"{atr.ATR_BASE}/racecard/{atr._track_to_slug(job['track'])}/{day}/{off}"
    response = fetcher.get(url)
    report = inventory(response[1]) if response and response[0] == 200 else {
        'state':'unresolved' if response and response[0] == 404 else 'blocked',
        'tabs':[],'unknown_tabs':[],'reason':'Source page unavailable'}
    report.update(source_url=url,checked_at=utc_now().isoformat())
    with con:
        set_control(con,'tab_inventory',report)
        set_control(con,'inventory_retry',now+max(3600,float(fetcher.last_response.get('retry_after_seconds') or 0)))
        if report['state']=='blocked':
            policy = _retry_policy(fetcher.last_response,0,control(con,'transient_streak',0))
            key = 'discovery_retry' if policy['scope']=='discovery' else 'global_retry'
            set_control(con,key,max(control(con,key,0),now+policy['global_delay']))
    atomic_json(Path(root)/'tab_inventory.json',report)


def prepare(root=DEFAULT_ROOT):
    root = Path(root)
    with writer_lock(root/'.prepare.lock') as acquired:
        if not acquired:
            raise RuntimeError('ATR preparation already running')
        con = connect(root)
        try:
            if con.execute('SELECT count(*) FROM jobs').fetchone()[0]:
                raise RuntimeError('Rebuild already prepared; resume it instead of resetting progress')
            with sqlite3.connect(Path(database.DB_PATH).resolve().as_uri()+'?mode=ro', uri=True) as source:
                fields = pd.read_sql_query('SELECT race_id,race_date,track,off_time,horse_name,horse_id,finish_position FROM results', source)
            fields['race_date'] = pd.to_datetime(fields.race_date,format='mixed',errors='raise').dt.strftime('%Y-%m-%d')
            old = atr.load_atr_cache()
            if not old.empty:
                old['race_date'] = pd.to_datetime(old.race_date,format='mixed',errors='raise').dt.strftime('%Y-%m-%d')
            hints = {}
            conflicts = set()
            for row in old.reindex(columns=['race_date','track','off_time','atr_raceid']).drop_duplicates().itertuples(index=False):
                key = _key(row.race_date, row.track, row.off_time)
                value = str(row.atr_raceid)
                if not re.fullmatch(r'\d+(?:\.0)?', value):
                    continue
                value = value.removesuffix('.0')
                if key in hints and hints[key] != value:
                    conflicts.add(key)
                hints[key] = value
            for key in conflicts:
                hints.pop(key, None)
            races = fields[['race_id','race_date','track','off_time']].drop_duplicates()
            excluded = races.track.map(atr._normalise_track_key).isin(atr._NO_SECTIONALS_TRACKS)
            eligible_ids = set(races.loc[~excluded, 'race_id'].astype(str))
            key_counts = races.assign(key=[_key(*r) for r in races[['race_date','track','off_time']].itertuples(index=False,name=None)]).key.value_counts()
            prepared = []
            for race_id, frame in fields.groupby('race_id', sort=False):
                if str(race_id) not in eligible_ids:
                    continue
                first = frame.iloc[0]
                day, track, off = str(first.race_date)[:10], str(first.track), str(first.off_time)
                key = _key(day, track, off)
                expected = [{'horse_key':atr._normalise_horse_key(row.horse_name),
                             'horse_name':str(row.horse_name),
                             'horse_id':None if pd.isna(row.horse_id) else str(row.horse_id),
                             'finish_position':None if pd.isna(row.finish_position) else int(row.finish_position)}
                            for row in frame.itertuples()]
                ambiguous = key_counts[key] != 1 or len({r['horse_key'] for r in expected}) != len(expected)
                prepared.append((str(race_id),key,day,track,off,json.dumps(expected),hints.get(key),
                                 'review' if ambiguous else 'pending',
                                 'Ambiguous canonical race/runner key' if ambiguous else None))
            with con:
                con.executemany('INSERT INTO jobs(race_id,race_key,race_date,track,off_time,expected,native_id,state,error) '
                                'VALUES (?,?,?,?,?,?,?,?,?)', prepared)
                set_control(con, 'code_hashes', code_hashes())
                set_control(con, 'created_at', utc_now().isoformat())
                seed_tab_coverage(con)
                set_control(con, 'collection_version', COLLECTION_VERSION)
                audit_legacy_hints(con,old)
                # The previous block was specifically on HTML ID discovery.
                previous = read_json(Path(config.DATA_DIR)/'source_refresh/atr_recovery.json')
                retry = previous.get('retry_after')
                if retry:
                    set_control(con, 'discovery_retry', datetime.fromisoformat(retry).timestamp())
            manifest = {'created_at':utc_now().isoformat(),'parser_version':atr.ATR_PARSER_VERSION,
                        'collection_version':COLLECTION_VERSION,
                        'root':str(root.resolve()), 'canonical_races':len(races), 'queued_races':len(prepared),
                        'excluded_unsupported_races':int(excluded.sum()),
                        'native_id_hints':con.execute('SELECT count(*) FROM jobs WHERE native_id IS NOT NULL').fetchone()[0],
                        'first_date':min(row[2] for row in prepared),'last_date':max(row[2] for row in prepared),
                        'old_parsed_rows_imported':0,'legacy_absence_decisions_imported':0,
                        'accepted_cache':str(Path(atr.ATR_CACHE_FILE).resolve()),'code_hashes':code_hashes()}
            atomic_json(root/'manifest.json',manifest)
            publish_status(con, root, 'prepared')
            return manifest
        finally:
            con.close()


def validate_rows(job, rows):
    """Reject unrelated runners; preserve explicitly identified tracking gaps."""
    expected = {row['horse_key']:row for row in json.loads(job['expected'])}
    incoming = [atr._normalise_horse_key(row.get('horse_name','')) for row in rows]
    missing = set(expected)-set(incoming)
    if not incoming or len(set(incoming)) != len(incoming) or set(incoming)-set(expected):
        raise ValueError('Runner field differs from canonical results: '
                         f'missing={sorted(set(expected)-set(incoming))}; extra={sorted(set(incoming)-set(expected))}')
    missing_runners = []
    coverage = 'complete'
    if missing:
        # A subset is safe only with independent source date/time evidence.
        local = atr._off_time_to_hhmm(job['off_time'])
        if atr._is_bst(job['race_date']):
            local = atr._add_one_hour(local)
        for row in rows:
            if row.get('atr_source_race_date') != job['race_date'] or row.get('atr_source_off_time_local') != f'{local[:2]}:{local[2:]}':
                raise ValueError('Incomplete field lacks independent source date/off-time proof')
        positions = {key:value.get('finish_position') for key,value in expected.items()}
        if any(positions[key] is None for key in missing):
            # Earlier prepared queues did not include outcome metadata.
            with sqlite3.connect(Path(database.DB_PATH).resolve().as_uri()+'?mode=ro',uri=True) as source:
                positions.update({atr._normalise_horse_key(name):position for name,position in source.execute(
                    'SELECT horse_name,finish_position FROM results WHERE race_id=?',(job['race_id'],))})
        missing_runners = [{'horse_name':expected[key]['horse_name'],'finish_position':positions.get(key)}
                           for key in sorted(missing)]
        coverage = 'finishers_only' if all(row['finish_position'] == 0 for row in missing_runners) else 'partial_tracking'
    for row, key in zip(rows, incoming):
        if str(row.get('atr_parser_version')) != str(atr.ATR_PARSER_VERSION):
            raise ValueError('Wrong parser version')
        speeds, labels = row.get('atr_sectional_speeds', []), row.get('atr_sectional_speed_labels', [])
        if len(speeds) != len(labels) or any('mph' in label.casefold() or not re.search(r'\bto\b',label,re.I) for label in labels):
            raise ValueError('Speed labels/units are invalid')
        splits = row.get('atr_split_times', [])
        for column in ('atr_running_totals','atr_pace_flags','atr_front_flags','atr_section_notes'):
            if column in row and len(row[column]) != len(splits):
                raise ValueError(f'Misaligned {column}')
        for metric, markers in (('atr_stride_length','atr_stride_markers'),
                                ('atr_stride_frequency','atr_stride_frequency_markers')):
            if metric in row and len(row[metric]) != len(row.get(markers, [])):
                raise ValueError(f'Misaligned {metric}')
        row.update(race_date=job['race_date'], track=job['track'], off_time=job['off_time'],
                   sl_race_id=job['race_id'], sl_horse_id=expected[key]['horse_id'],
                   atr_field_coverage=coverage, atr_field_expected_runners=len(expected),
                   atr_field_provided_runners=len(incoming), atr_field_missing_runners=missing_runners)
    return rows


def accept(con, job, outcome, rows):
    now = utc_now().isoformat()
    if outcome == atr.DATA:
        rows = validate_rows(job, rows)
        receipts = rows[0].get('atr_tab_manifest', [])
        if {r['kind'] for r in receipts if r.get('state') in TERMINAL_TAB_STATES} != set(PERFORMANCE_TABS):
            raise ValueError('Incomplete performance tab manifest')
    with con:
        if outcome == atr.DATA:
            con.executemany('INSERT OR REPLACE INTO runners VALUES (?,?,?)', [
                (job['race_id'],atr._normalise_horse_key(row['horse_name']),json.dumps(row,allow_nan=False))
                for row in rows])
            state = 'data_partial' if rows[0]['atr_field_coverage'] != 'complete' else 'data'
        elif outcome == atr.NOT_FOUND:
            state = 'no_data'
        else:
            raise ValueError('Only confirmed outcomes can be committed')
        con.execute('UPDATE jobs SET state=?,error=NULL,updated_at=? WHERE race_id=?',
                    (state,now,job['race_id']))
        if outcome == atr.DATA:
            con.execute('UPDATE jobs SET native_id=? WHERE race_id=?', (str(rows[0]['atr_raceid']),job['race_id']))
            con.execute('INSERT OR IGNORE INTO dirty_days VALUES (?)', (job['race_date'],))
            source = rows[0].get('atr_tab_manifest',[{}])[0]
            atr_ids.evidence(con,job['race_id'],str(rows[0]['atr_raceid']),'verified',
                             source.get('url'),source.get('sha256'),{'validated_runners':len(rows)})


def export_days(con, root, days):
    for day in days:
        records = con.execute('SELECT r.payload FROM runners r JOIN jobs j USING(race_id) WHERE j.race_date=?',
                              (day,)).fetchall()
        if records:
            frame = atr._encode_arrays(pd.DataFrame([json.loads(row[0]) for row in records]))
            atomic_parquet(frame, Path(root)/'daily'/f'{day}.parquet')
            with con:
                con.execute('DELETE FROM dirty_days WHERE race_date=?', (day,))


def publish_status(con, root, state, **extra):
    counts = dict(con.execute('SELECT state,count(*) FROM jobs GROUP BY state').fetchall())
    cutoff = (utc_now()-pd.Timedelta(hours=atr._PUBLICATION_DELAY_HOURS)).strftime('%Y-%m-%d %H:%M')
    deferred = con.execute("SELECT count(*) FROM jobs WHERE state IN ('pending','retry') AND "
                           "(race_date>? OR (race_date=? AND substr(off_time,1,5)>?))",
                           (cutoff[:10],cutoff[:10],cutoff[11:])).fetchone()[0]
    tabs = {kind:{} for kind in PERFORMANCE_TABS}
    for kind, tab_state, count in con.execute('SELECT kind,state,count(*) FROM tab_coverage GROUP BY kind,state'):
        tabs.setdefault(kind,{})[tab_state] = count
    result = {'state':state, 'pid':os.getpid(), 'checked_at':utc_now().isoformat(),
              'parser_version':atr.ATR_PARSER_VERSION,'races':counts,
              'collection_version':COLLECTION_VERSION, 'tabs':tabs,
              'tab_inventory':control(con,'tab_inventory',{'state':'not_yet_verified'}),
              'tab_inventory_baseline':control(con,'tab_inventory_baseline',{}),
              'total_races':sum(counts.values()), 'saved_rows':con.execute('SELECT count(*) FROM runners').fetchone()[0],
              'publication_deferred_races':deferred,
              'global_retry_after':control(con,'global_retry',0),
              'discovery_retry_after':control(con,'discovery_retry',0),
              'last_id_discovery':control(con,'last_id_discovery'),
              'native_ids':{'missing':con.execute('SELECT count(*) FROM jobs WHERE native_id IS NULL').fetchone()[0],
                            'evidence':dict(con.execute('SELECT state,count(*) FROM id_evidence GROUP BY state'))},
              'exports_pending':con.execute('SELECT count(*) FROM dirty_days').fetchone()[0],
              'accepted_cache_replaced':False, **extra}
    atomic_json(Path(root)/'status.json',result)
    return result


class ReplayFetcher:
    """Source response reuse, never reuse of old parsed telemetry."""
    def __init__(self, store, fallback=None):
        self.store = store
        self.fallback = fallback
        self.last_response = {}
        self.rejected_urls = set()
        self.reused = 0
        self.index_dir = getattr(fallback,'index_dir',Path(atr.ATR_INDEX_DIR))

    def get(self, url, **kwargs):
        body = self.store.latest(url)
        if body is not None and b'Client Challenge' in body[:4000] and hasattr(self.store,'connection'):
            # A later HTTP-200 challenge must not hide an older retained success.
            with self.store.connection() as con:
                candidates = con.execute('SELECT p.body FROM observations o JOIN payloads p USING(sha256) '
                                         'WHERE o.url=? AND o.status=200 ORDER BY o.last_seen DESC', (url,)).fetchall()
            for (compressed,) in candidates:
                candidate = zlib.decompress(compressed)
                if b'Client Challenge' not in candidate[:4000]:
                    body = candidate
                    break
        if body is None or b'Client Challenge' in body[:4000] or url in self.rejected_urls:
            if self.fallback is None:
                return None
            result = self.fallback.get(url, **kwargs)
            self.last_response = self.fallback.last_response
            return result
        self.last_response = {'url':url,'status':200,'sha256':hashlib.sha256(body).hexdigest()}
        self.reused += 1
        return 200,body.decode('utf-8')


def _raw_ids(store):
    with store.connection() as con:
        urls = con.execute("SELECT DISTINCT url FROM observations WHERE status=200 AND url LIKE '%/ajax/sectionals-razor/%'").fetchall()
    found = {}
    for (url,) in urls:
        match = re.search(r'/sectionals-razor/(\d+)/(times|tools|stride)$', url)
        if match:
            found.setdefault(match[1],set()).add(match[2])
    return {key for key, values in found.items() if len(values) == 3}


def _retry_policy(last, attempts, transient_streak, standard=3600, *, previous=None):
    """Separate genuine edge blocks from temporary service/network failures."""
    requested = max(0,float(last.get('retry_after_seconds') or 0))
    if last.get('status') == 429:
        return {'scope':'global','global_delay':max(standard,requested),
                'item_delay':max(standard,requested),'transient':False}
    if last.get('reason') == 'unrecognised_index':
        return {'scope':'discovery','global_delay':max(standard,requested),
                'item_delay':max(standard,requested),'transient':False}
    if last.get('reason') == 'client_challenge':
        return {'scope':'discovery' if '/ajax/' not in last.get('url','') else 'global',
                'global_delay':max(standard,requested),
                'item_delay':max(standard,requested),'transient':False}
    if (last.get('status') == 500 and not requested and last.get('sha256')
            and '/ajax/sectionals-razor/' in last.get('url', '') and isinstance(previous, dict)
            and all(last.get(key) == previous.get(key) for key in ('status', 'url', 'sha256'))):
        # The same field and endpoint have already returned this exact error.
        return {'scope':'item','global_delay':0,'item_delay':max(86400,standard),'transient':False}
    if last.get('status') in {500,502,503,504} or last.get('reason') == 'transport_error':
        return {'scope':'global','global_delay':max(requested,min(standard,60*2**min(transient_streak,6))),
                'item_delay':max(requested,min(standard,300*2**min(attempts,4))), 'transient':True}
    return {'scope':'global','global_delay':max(standard,requested),
            'item_delay':max(standard,requested),'transient':False}


def job_query(discovery_allowed, *, pending_only=False):
    native = '' if discovery_allowed else 'AND native_id IS NOT NULL '
    index = 'jobs_active_order' if discovery_allowed else 'jobs_active_native'
    pending = "AND state='pending' " if pending_only else ''
    return (f"SELECT * FROM jobs INDEXED BY {index} WHERE state IN ('pending','retry') AND next_retry<=? "
            "AND race_date||' '||substr(off_time,1,5)<=? "+native+pending+
            "ORDER BY race_date DESC,off_time DESC,native_id IS NULL LIMIT 1")


def pick_job(con, now, cutoff, *, discovery_allowed):
    ready = (now, cutoff)
    pending = con.execute(job_query(False, pending_only=True), ready).fetchone()
    return pending if pending is not None else con.execute(job_query(discovery_allowed), ready).fetchone()


def _discovery_day(con, now, cutoff_day, *, oldest=False):
    order = 'ASC' if oldest else 'DESC'
    day = con.execute("SELECT j.race_date FROM jobs j LEFT JOIN id_discovery d USING(race_date) "
                      "WHERE j.native_id IS NULL AND j.state IN ('pending','retry') AND j.race_date<=? "
                      "AND COALESCE(d.next_retry,0)<=? GROUP BY j.race_date ORDER BY j.race_date "+order+" LIMIT 1",
                      (cutoff_day,now)).fetchone()
    return day[0] if day else None


def discover_ids(con, client, now, cutoff, *, retry_seconds=3600):
    """One paced day lookup, then save all matches; failed discovery backs off."""
    if any(control(con,key,0)>now for key in ('global_retry','discovery_retry','id_discovery_next')):
        return None
    oldest = control(con,'id_scan_oldest',False)
    day = _discovery_day(con, now, cutoff[:10], oldest=oldest)
    if not day:
        return None
    before = con.execute('SELECT count(*) FROM jobs WHERE native_id IS NULL').fetchone()[0]
    outcome, mapping = atr._results_index(client,atr._date_to_atr(day))
    last = client.last_response
    retry = 0
    with con:
        set_control(con,'id_scan_oldest',not oldest)
        set_control(con,'id_discovery_next',utc_now().timestamp()+60)
        if outcome == atr.DATA:
            # An otherwise valid day can still omit a canonical race (for
            # example a changed advertised time). Move on to other dates;
            # do not repeatedly select the same incomplete cached index.
            remaining = con.execute("SELECT count(*) FROM jobs WHERE race_date=? AND native_id IS NULL "
                                    "AND state IN ('pending','retry')", (day,)).fetchone()[0]
            retry = utc_now().timestamp()+86400 if remaining else 0
            con.execute('UPDATE id_discovery SET next_retry=? WHERE race_date=?',(retry,day))
        else:
            retry = utc_now().timestamp()+86400
            if outcome == atr.BLOCKED:
                policy = _retry_policy(last,0,control(con,'transient_streak',0),retry_seconds)
                retry = utc_now().timestamp()+policy['item_delay']
                set_control(con,'discovery_retry' if policy['scope']=='discovery' else 'global_retry',
                            utc_now().timestamp()+policy['global_delay'])
            con.execute('INSERT INTO id_discovery(race_date,state,attempts,next_retry,source_url,error,checked_at) '
                        'VALUES (?,?,1,?,?,?,?) ON CONFLICT(race_date) DO UPDATE SET state=excluded.state,'
                        'attempts=id_discovery.attempts+1,next_retry=excluded.next_retry,error=excluded.error,'
                        'checked_at=excluded.checked_at',
                        (day,outcome,retry,last.get('url'),json.dumps(last),utc_now().isoformat()))
        report = {'race_date':day,'outcome':outcome,'source_races':len(mapping),'checked_at':utc_now().isoformat(),
                  'ids_added':before-con.execute('SELECT count(*) FROM jobs WHERE native_id IS NULL').fetchone()[0],
                  'retry_after':retry,'last_response':last}
        set_control(con,'last_id_discovery',report)
    logger.info('ATR ID discovery for %s: %s, %d IDs added', day, outcome, report['ids_added'])
    return report


def reconcile_exports(con, root):
    """Queue missing/truncated old exports once; all future dirtiness is durable."""
    if control(con,'export_queue_version',0)==1:
        return
    import pyarrow.parquet as pq
    days = con.execute('SELECT j.race_date,count(*) FROM runners r JOIN jobs j USING(race_id) GROUP BY j.race_date').fetchall()
    with con:
        for day,count in days:
            path = Path(root)/'daily'/f'{day}.parquet'
            try:
                complete = path.exists() and pq.ParquetFile(path).metadata.num_rows==count
            except (OSError, ValueError):
                complete = False
            if not complete:
                con.execute('INSERT OR IGNORE INTO dirty_days VALUES (?)',(day,))
        set_control(con,'export_queue_version',1)


def run(root=DEFAULT_ROOT, *, request_delay=3.0, max_races=None, retry_seconds=3600):
    root = Path(root)
    with writer_lock(root/'.worker.lock') as acquired, writer_lock(Path(atr.ATR_CACHE_DIR)/'.collector.lock') as collector:
        if not acquired or not collector:
            raise RuntimeError('An ATR collector is already running')
        con = connect(root)
        fetcher = atr._Fetcher(request_delay=request_delay)
        store = RawPayloadStore(fetcher.raw_store_path)
        replay = ReplayFetcher(store, fallback=fetcher)
        dirty, attempted = set(), 0
        last_status, last_export = 0.0, time.monotonic()
        status_extra = {}
        def progress(state, *, force=False, **extra):
            nonlocal last_status
            status_extra.update(extra)
            if force or time.monotonic()-last_status>=STATUS_SECONDS:
                publish_status(con,root,state,**status_extra,
                               requests_this_session=getattr(fetcher,'request_count',None),
                               retained_responses_reused=replay.reused)
                last_status = time.monotonic()
        try:
            if not control(con,'code_hashes'):
                raise RuntimeError('Run --prepare before starting the worker')
            if control(con,'collection_version') != COLLECTION_VERSION:
                raise RuntimeError('Run --upgrade-tabs after review; existing progress must not be reset')
            if code_hashes() != control(con,'code_hashes'):
                progress('code_changed',force=True,reason='Collector code changed; restart after review')
                return
            seen_indexes = set()
            def on_index(day, mapping, receipt):
                key = day,tuple(sorted(mapping.items()))
                if key not in seen_indexes:
                    atr_ids.apply_index(con,day,mapping,receipt)
                    seen_indexes.add(key)
            replay.on_index = on_index
            with con:
                audit_legacy_hints(con)
            seen_receipts = {}
            next_receipt_scan = 0.0
            reconcile_exports(con,root)
            dirty.update(row[0] for row in con.execute('SELECT race_date FROM dirty_days'))
            while not (root/'STOP').exists():
                if time.monotonic() >= next_receipt_scan:
                    for report in atr_ids.recover_cached(con,replay.index_dir,seen=seen_receipts):
                        if report.get('error') or report.get('conflicts'):
                            logger.warning('Saved ATR index needs review: %s', report)
                        if report.get('ids_added_or_corrected'):
                            logger.info('Saved ATR index for %s: %d IDs added or corrected',
                                        report['date'], report['ids_added_or_corrected'])
                    next_receipt_scan = time.monotonic()+60
                if max_races is not None and attempted >= max_races:
                    break
                if code_hashes() != control(con,'code_hashes'):
                    progress('code_changed',force=True,reason='Collector code changed; restart after review')
                    return
                now = utc_now().timestamp()
                global_retry = control(con,'global_retry',0)
                discovery_retry = control(con,'discovery_retry',0)
                cutoff = (utc_now()-pd.Timedelta(hours=atr._PUBLICATION_DELAY_HOURS)).strftime('%Y-%m-%d %H:%M')
                discover_ids(con,replay,now,cutoff,retry_seconds=retry_seconds)
                global_retry = control(con,'global_retry',0)
                discovery_retry = control(con,'discovery_retry',0)
                job = pick_job(con,now,cutoff,discovery_allowed=discovery_retry<=now) if global_retry<=now else None
                if job is None:
                    export_days(con,root,dirty)
                    dirty.clear()
                    pending = con.execute("SELECT count(*) FROM jobs WHERE state IN ('pending','retry')").fetchone()[0]
                    if not pending:
                        reviews = con.execute("SELECT count(*) FROM jobs WHERE state='review'").fetchone()[0]
                        progress('needs_review' if reviews else 'complete',force=True)
                        return
                    progress('waiting',force=True,reason='Publication window or provider cooldown')
                    time.sleep(30)
                    continue
                progress('running',current={k:job[k] for k in ('race_id','race_date','track','off_time')})
                with con:
                    con.execute('UPDATE jobs SET attempts=attempts+1 WHERE race_id=?',(job['race_id'],))
                client = replay
                before_reuse = replay.reused
                client.tab_receipts = []
                outcome, rows = atr.fetch_atr_sectionals_outcome(job['race_date'],job['track'],job['off_time'],
                    client, raceid=job['native_id'])
                used_raw = replay.reused > before_reuse
                if used_raw and outcome == atr.UNRESOLVED and client.last_response.get('reason') == 'unrecognised_tab':
                    # Recheck only the changed/invalid fragment, not the three
                    # already retained good tabs.
                    replay.rejected_urls.add(client.last_response['url'])
                    outcome, rows = atr.fetch_atr_sectionals_outcome(job['race_date'],job['track'],job['off_time'],
                        client,raceid=job['native_id'])
                if outcome == atr.NOT_FOUND and job['native_id'] is not None:
                    proof = con.execute('SELECT state FROM id_evidence WHERE race_id=?',(job['race_id'],)).fetchone()
                    if not proof or proof[0] not in ('mapped','verified','roster_supported'):
                        outcome = atr.UNRESOLVED
                        client.last_response = {**client.last_response,'reason':'unverified_identity_absence'}
                with con:
                    record_tabs(con,job['race_id'],getattr(client,'tab_receipts',[]))
                attempted += 1
                if outcome in (atr.DATA,atr.NOT_FOUND):
                    with con:
                        set_control(con,'transient_streak',0)
                    try:
                        accept(con,job,outcome,rows)
                        if outcome == atr.DATA:
                            dirty.add(job['race_date'])
                    except ValueError as error:
                        with con:
                            record_tabs(con,job['race_id'],[
                                {**r,'state':'review','reason':str(error)} if r.get('state')=='data' else r
                                for r in getattr(client,'tab_receipts',[])])
                            con.execute("UPDATE jobs SET state='review',error=?,updated_at=? WHERE race_id=?",
                                        (str(error),utc_now().isoformat(),job['race_id']))
                        logger.warning('Rejected %s: %s',job['race_id'],error)
                else:
                    last = client.last_response
                    delay = retry_seconds if outcome == atr.BLOCKED else max(retry_seconds,86400)
                    policy = None
                    retry_base = utc_now().timestamp()
                    if outcome == atr.BLOCKED:
                        try:
                            previous = json.loads(job['error'] or '{}')
                        except ValueError:
                            previous = None
                        policy = _retry_policy(last,job['attempts'],control(con,'transient_streak',0),
                                               retry_seconds,previous=previous)
                        delay = policy['item_delay']
                    invalid_hint = last.get('reason') in ('source_identity_mismatch','unverified_identity_absence')
                    if invalid_hint:
                        delay = retry_seconds
                    with con:
                        con.execute("UPDATE jobs SET state='retry',next_retry=?,error=?,updated_at=? WHERE race_id=?",
                                    (retry_base+delay,json.dumps(last) or outcome,utc_now().isoformat(),job['race_id']))
                        if invalid_hint:
                            con.execute('UPDATE jobs SET native_id=NULL WHERE race_id=?',(job['race_id'],))
                        if outcome == atr.BLOCKED and policy['scope'] != 'item':
                            set_control(con,'discovery_retry' if policy['scope']=='discovery' else 'global_retry',
                                        retry_base+policy['global_delay'])
                            if policy['transient']:
                                set_control(con,'transient_streak',control(con,'transient_streak',0)+1)
                    logger.warning('Deferred %s: %s, %s',job['race_id'],outcome,last)
                if attempted % EXPORT_RACES == 0 or time.monotonic()-last_export>=EXPORT_SECONDS:
                    export_days(con,root,dirty)
                    dirty.clear()
                    last_export = time.monotonic()
                progress('running',last_outcome=outcome,attempted_this_session=attempted,
                         last_used_retained_html=used_raw)
                if outcome in (atr.DATA,atr.NOT_FOUND):
                    audit_navigation(con,fetcher,job,root)
                    if control(con,'tab_inventory',{}).get('unknown_tabs'):
                        progress('needs_tab_review',force=True,reason='ATR advertises unhandled tabs')
                        return
                logger.info('Field %s %s %s: %s',job['race_date'],job['track'],job['off_time'],outcome)
            export_days(con,root,dirty)
            dirty.clear()
            progress('stopped' if (root/'STOP').exists() else 'batch_complete',force=True)
        except BaseException as error:
            publish_status(con,root,'error',error=str(error))
            raise
        finally:
            export_days(con,root,dirty)
            fetcher.close()
            con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--upgrade-tabs',action='store_true',help='Reviewed tab-contract upgrade; preserves saved rows')
    parser.add_argument('--request-delay',type=float,default=3.0)
    parser.add_argument('--max-races',type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.upgrade_tabs:
        print(json.dumps(upgrade_tab_coverage(args.root),indent=2))
    elif args.prepare:
        print(json.dumps(prepare(args.root),indent=2))
    else:
        run(args.root,request_delay=args.request_delay,max_races=args.max_races)


if __name__ == '__main__':
    main()
