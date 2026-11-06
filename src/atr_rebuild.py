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

import pandas as pd

import config
from src import attheraces_scraper as atr, database
from src.source_storage import RawPayloadStore, atomic_json, atomic_parquet, read_json, writer_lock

logger = logging.getLogger(__name__)
DEFAULT_ROOT = Path(config.DATA_DIR)/'atr_cache/rebuild_v2'
CODE_FILES = ('src/attheraces_scraper.py', 'src/atr_features.py', 'src/atr_rebuild.py')


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
        CREATE TABLE IF NOT EXISTS runners (
          race_id TEXT NOT NULL, horse_key TEXT NOT NULL, payload TEXT NOT NULL,
          PRIMARY KEY(race_id,horse_key));
        CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    ''')
    return con


def control(con, key, default=None):
    row = con.execute('SELECT value FROM control WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_control(con, key, value):
    con.execute('INSERT INTO control VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                (key, json.dumps(value, allow_nan=False)))


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
                # The previous block was specifically on HTML ID discovery.
                previous = read_json(Path(config.DATA_DIR)/'source_refresh/atr_recovery.json')
                retry = previous.get('retry_after')
                if retry:
                    set_control(con, 'discovery_retry', datetime.fromisoformat(retry).timestamp())
            manifest = {'created_at':utc_now().isoformat(),'parser_version':atr.ATR_PARSER_VERSION,
                        'root':str(root.resolve()), 'canonical_races':len(races), 'queued_races':len(prepared),
                        'excluded_unsupported_races':int(excluded.sum()),
                        'native_id_hints':sum(row[6] is not None for row in prepared),
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


def export_days(con, root, days):
    for day in days:
        records = con.execute('SELECT r.payload FROM runners r JOIN jobs j USING(race_id) WHERE j.race_date=?',
                              (day,)).fetchall()
        if records:
            frame = atr._encode_arrays(pd.DataFrame([json.loads(row[0]) for row in records]))
            atomic_parquet(frame, Path(root)/'daily'/f'{day}.parquet')


def publish_status(con, root, state, **extra):
    counts = dict(con.execute('SELECT state,count(*) FROM jobs GROUP BY state').fetchall())
    cutoff = (utc_now()-pd.Timedelta(hours=atr._PUBLICATION_DELAY_HOURS)).strftime('%Y-%m-%d %H:%M')
    deferred = con.execute("SELECT count(*) FROM jobs WHERE state IN ('pending','retry') AND "
                           "(race_date>? OR (race_date=? AND substr(off_time,1,5)>?))",
                           (cutoff[:10],cutoff[:10],cutoff[11:])).fetchone()[0]
    result = {'state':state, 'pid':os.getpid(), 'checked_at':utc_now().isoformat(),
              'parser_version':atr.ATR_PARSER_VERSION,'races':counts,
              'total_races':sum(counts.values()), 'saved_rows':con.execute('SELECT count(*) FROM runners').fetchone()[0],
              'publication_deferred_races':deferred,
              'global_retry_after':control(con,'global_retry',0),
              'discovery_retry_after':control(con,'discovery_retry',0),
              'accepted_cache_replaced':False, **extra}
    atomic_json(Path(root)/'status.json',result)
    return result


class ReplayFetcher:
    """Source response reuse, never reuse of old parsed telemetry."""
    def __init__(self, store):
        self.store = store
        self.last_response = {}

    def get(self, url, **kwargs):
        body = self.store.latest(url)
        if body is None or b'Client Challenge' in body[:4000]:
            return None
        self.last_response = {'url':url,'status':200,'sha256':hashlib.sha256(body).hexdigest()}
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


def _retry_policy(last, attempts, transient_streak, standard=3600):
    """Separate genuine edge blocks from temporary service/network failures."""
    requested = max(0,float(last.get('retry_after_seconds') or 0))
    if last.get('reason') == 'client_challenge':
        return {'scope':'discovery' if '/ajax/' not in last.get('url','') else 'global',
                'global_delay':max(standard,requested),
                'item_delay':max(standard,requested),'transient':False}
    if last.get('status') in {500,502,503,504} or last.get('reason') == 'transport_error':
        return {'scope':'global','global_delay':max(requested,min(standard,60*2**min(transient_streak,6))),
                'item_delay':max(requested,min(standard,300*2**min(attempts,4))), 'transient':True}
    return {'scope':'global','global_delay':max(standard,requested),
            'item_delay':max(standard,requested),'transient':False}


def run(root=DEFAULT_ROOT, *, request_delay=3.0, max_races=None, retry_seconds=3600):
    root = Path(root)
    with writer_lock(root/'.worker.lock') as acquired, writer_lock(Path(atr.ATR_CACHE_DIR)/'.collector.lock') as collector:
        if not acquired or not collector:
            raise RuntimeError('An ATR collector is already running')
        con = connect(root)
        fetcher = atr._Fetcher(request_delay=request_delay)
        store = RawPayloadStore(fetcher.raw_store_path)
        replay = ReplayFetcher(store)
        available_raw_ids = _raw_ids(store)
        dirty, attempted = set(), 0
        try:
            if not control(con,'code_hashes'):
                raise RuntimeError('Run --prepare before starting the worker')
            while not (root/'STOP').exists():
                if max_races is not None and attempted >= max_races:
                    break
                if code_hashes() != control(con,'code_hashes'):
                    publish_status(con,root,'code_changed',reason='Collector code changed; restart after review')
                    return
                now = utc_now().timestamp()
                global_retry = control(con,'global_retry',0)
                discovery_retry = control(con,'discovery_retry',0)
                cutoff = (utc_now()-pd.Timedelta(hours=atr._PUBLICATION_DELAY_HOURS)).strftime('%Y-%m-%d %H:%M')
                job = con.execute('SELECT * FROM jobs WHERE state IN (\'pending\',\'retry\') AND next_retry<=? '
                                  'AND race_date||\' \'||substr(off_time,1,5)<=? '
                                  'AND (?<=? OR native_id IS NOT NULL) '
                                  'ORDER BY race_date DESC,off_time DESC,native_id IS NULL LIMIT 1',
                                  (now,cutoff,discovery_retry,now)).fetchone() if global_retry <= now else None
                if job is None:
                    export_days(con,root,dirty)
                    dirty.clear()
                    pending = con.execute("SELECT count(*) FROM jobs WHERE state IN ('pending','retry')").fetchone()[0]
                    if not pending:
                        reviews = con.execute("SELECT count(*) FROM jobs WHERE state='review'").fetchone()[0]
                        publish_status(con,root,'needs_review' if reviews else 'complete')
                        return
                    publish_status(con,root,'waiting',reason='Publication window or provider cooldown')
                    time.sleep(30)
                    continue
                publish_status(con,root,'running',current={k:job[k] for k in ('race_id','race_date','track','off_time')})
                with con:
                    con.execute('UPDATE jobs SET attempts=attempts+1 WHERE race_id=?',(job['race_id'],))
                used_raw = job['native_id'] in available_raw_ids
                client = replay if used_raw else fetcher
                outcome, rows = atr.fetch_atr_sectionals_outcome(job['race_date'],job['track'],job['off_time'],
                    client, raceid=job['native_id'])
                if used_raw and outcome not in (atr.DATA,atr.NOT_FOUND):
                    available_raw_ids.discard(job['native_id'])
                    outcome, rows = atr.fetch_atr_sectionals_outcome(job['race_date'],job['track'],job['off_time'],
                        fetcher,raceid=job['native_id'])
                    client = fetcher
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
                            con.execute("UPDATE jobs SET state='review',error=?,updated_at=? WHERE race_id=?",
                                        (str(error),utc_now().isoformat(),job['race_id']))
                        logger.warning('Rejected %s: %s',job['race_id'],error)
                else:
                    last = client.last_response
                    delay = retry_seconds if outcome == atr.BLOCKED else max(retry_seconds,86400)
                    policy = None
                    retry_base = utc_now().timestamp()
                    if outcome == atr.BLOCKED:
                        policy = _retry_policy(last,job['attempts'],control(con,'transient_streak',0),retry_seconds)
                        delay = policy['item_delay']
                    invalid_hint = job['native_id'] is not None and last.get('reason') == 'source_identity_mismatch'
                    if invalid_hint:
                        delay = retry_seconds
                    with con:
                        con.execute("UPDATE jobs SET state='retry',next_retry=?,error=?,updated_at=? WHERE race_id=?",
                                    (retry_base+delay,json.dumps(last) or outcome,utc_now().isoformat(),job['race_id']))
                        if invalid_hint:
                            con.execute('UPDATE jobs SET native_id=NULL WHERE race_id=?',(job['race_id'],))
                        if outcome == atr.BLOCKED:
                            set_control(con,'discovery_retry' if policy['scope']=='discovery' else 'global_retry',
                                        retry_base+policy['global_delay'])
                            if policy['transient']:
                                set_control(con,'transient_streak',control(con,'transient_streak',0)+1)
                    logger.warning('Deferred %s: %s, %s',job['race_id'],outcome,last)
                if attempted % 10 == 0:
                    export_days(con,root,dirty)
                    dirty.clear()
                publish_status(con,root,'running',last_outcome=outcome,attempted_this_session=attempted,
                               last_used_retained_html=used_raw)
                logger.info('Field %s %s %s: %s',job['race_date'],job['track'],job['off_time'],outcome)
            publish_status(con,root,'stopped' if (root/'STOP').exists() else 'batch_complete')
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
    parser.add_argument('--request-delay',type=float,default=3.0)
    parser.add_argument('--max-races',type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.prepare:
        print(json.dumps(prepare(args.root),indent=2))
    else:
        run(args.root,request_delay=args.request_delay,max_races=args.max_races)


if __name__ == '__main__':
    main()
