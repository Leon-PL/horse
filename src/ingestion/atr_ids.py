"""ATR identity receipts and batched discovery; telemetry is never inferred."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from src.ingestion import attheraces as atr
from src.ingestion.storage import atomic_json, read_json


def ensure_schema(con):
    con.executescript('''
        CREATE TABLE IF NOT EXISTS id_evidence (
          race_id TEXT PRIMARY KEY, native_id TEXT, state TEXT NOT NULL,
          source_url TEXT, source_sha256 TEXT, detail TEXT, checked_at TEXT);
        CREATE TABLE IF NOT EXISTS id_discovery (
          race_date TEXT PRIMARY KEY, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
          next_retry REAL NOT NULL DEFAULT 0, mapped INTEGER NOT NULL DEFAULT 0,
          source_url TEXT, error TEXT, checked_at TEXT);
    ''')


def card_path(job):
    off = atr._off_time_to_hhmm(job['off_time'])
    if atr._is_bst(job['race_date']):
        off = atr._add_one_hour(off)
    return f"/racecard/{atr._track_to_slug(job['track'])}/{atr._date_to_atr(job['race_date'])}/{off}"


def evidence(con, race_id, native_id, state, url, sha=None, detail=None):
    con.execute('INSERT INTO id_evidence VALUES (?,?,?,?,?,?,?) ON CONFLICT(race_id) DO UPDATE SET '
                'native_id=excluded.native_id,state=excluded.state,source_url=excluded.source_url,'
                'source_sha256=excluded.source_sha256,detail=excluded.detail,checked_at=excluded.checked_at',
                (race_id,native_id,state,url,sha,json.dumps(detail or {},allow_nan=False),
                 datetime.now(timezone.utc).isoformat()))


def apply_index(con, atr_date, mapping, receipt=None):
    """Use exact dated local-time paths, preserving already verified identities."""
    day = datetime.strptime(atr_date, '%d-%B-%Y').strftime('%Y-%m-%d')
    receipt = receipt or {}
    url = receipt.get('source_url', f'{atr.ATR_BASE}/results/{atr_date}')
    sha = receipt.get('source_sha256')
    supplemental = '/replays/' in url
    if not isinstance(mapping,dict) or not mapping:
        raise ValueError('Empty or invalid ATR identity mapping')
    paths, ids = {}, defaultdict(set)
    for path, native_id in mapping.items():
        if not re.fullmatch(rf'/racecard/[^/]+/{re.escape(atr_date)}/\d{{4}}', path) or not re.fullmatch(r'\d+', str(native_id)):
            raise ValueError('Invalid dated ATR identity mapping')
        paths[path.casefold()] = str(native_id)
        ids[str(native_id)].add(path.casefold())
    jobs = con.execute('SELECT * FROM jobs WHERE race_date=?', (day,)).fetchall()
    local = defaultdict(list)
    for job in jobs:
        local[card_path(job).casefold()].append(job)
    changed, conflicts = 0, []
    with con:
        for path, matching in local.items():
            native_id = paths.get(path)
            if native_id is None:
                continue
            if len(matching) != 1 or len(ids[native_id]) != 1:
                conflicts.extend(j['race_id'] for j in matching)
                continue
            job = matching[0]
            others = con.execute('SELECT race_id,state FROM jobs WHERE native_id=? AND race_id<>?',
                                 (native_id,job['race_id'])).fetchall()
            if supplemental and (job['native_id'] is not None or others):
                if job['native_id'] != native_id or others:
                    conflicts.append(job['race_id'])
                continue
            if (job['state'] in ('data','data_partial') and job['native_id'] != native_id
                    or any(other['state'] in ('data','data_partial') for other in others)):
                conflicts.append(job['race_id'])
                continue
            for other in others:
                con.execute("UPDATE jobs SET native_id=NULL WHERE race_id=?", (other['race_id'],))
                evidence(con,other['race_id'],None,'conflict',url,sha,
                         {'previous_native_id':native_id,'mapped_to':job['race_id']})
            if job['native_id'] != native_id:
                con.execute("UPDATE jobs SET native_id=?,state=CASE WHEN state IN ('retry','no_data') THEN 'pending' ELSE state END,"
                            "next_retry=0 WHERE race_id=?", (native_id,job['race_id']))
                changed += 1
            if job['state'] not in ('data','data_partial'):
                evidence(con,job['race_id'],native_id,'mapped',url,sha,
                         {'path':card_path(job),'previous_native_id':job['native_id']})
        if '/results/' in url:
            con.execute('INSERT INTO id_discovery(race_date,state,mapped,source_url,checked_at) VALUES (?,?,?,?,?) '
                        'ON CONFLICT(race_date) DO UPDATE SET state=excluded.state,mapped=excluded.mapped,'
                        'source_url=excluded.source_url,checked_at=excluded.checked_at',
                        (day,'conflict' if conflicts else 'mapped',len(mapping),url,datetime.now(timezone.utc).isoformat()))
    return {'date':day,'ids_added_or_corrected':changed,'conflicts':conflicts,'source_races':len(mapping)}


def recover_cached(con, directory, *, seen=None):
    reports = []
    for path in sorted(Path(directory).glob('*.json')):
        try:
            stamp = path.stat().st_mtime_ns
            if seen is not None and seen.get(path) == stamp:
                continue
            receipt = read_json(path)
            day = path.stem.removesuffix('.replays')
            reports.append(apply_index(con,day,receipt['race_ids'],receipt))
            if seen is not None:
                seen[path] = stamp
        except (KeyError, OSError, ValueError, TypeError) as error:
            reports.append({'file':str(path),'error':str(error)})
    return reports


def _parse_replay_index(soup, day):
    """Pair a replay control with its own result link in the same article."""
    mapping = {}
    for article in soup.select('article.article-preview'):
        cards = set()
        for link in article.select('a[href^="/racecard/"]'):
            path = urlsplit(link['href']).path.rstrip('/')
            if re.fullmatch(rf'/racecard/[^/]+/{re.escape(day)}/\d{{4}}', path):
                cards.add(path)
        controls = article.select('a[href][data-video-raceid]')
        if not cards or not controls:
            continue
        if len(cards) != 1:
            raise ValueError('Replay article contains multiple racecard paths')
        card = cards.pop()
        _, _, course, _, off = card.split('/')
        expected = f'/replays/{day}/{course}/{off}'.casefold()
        ids = {link['data-video-raceid'] for link in controls}
        if (len(ids) != 1 or any(urlsplit(link['href']).path.rstrip('/').casefold() != expected
                                 for link in controls)):
            raise ValueError(f'Conflicting replay links for {card}')
        native = ids.pop()
        if not re.fullmatch(r'\d+', native) or (card in mapping and mapping[card] != native):
            raise ValueError(f'Conflicting replay ID for {card}')
        mapping[card] = native
    return mapping


def import_saved_results(path, directory):
    """Queue exact IDs from a saved, dated ATR results or replay page."""
    import hashlib
    from bs4 import BeautifulSoup

    raw = path.read_bytes()
    soup = BeautifulSoup(raw, 'html.parser')
    canonical = soup.select_one('link[rel="canonical"]')
    url = canonical.get('href', '') if canonical else ''
    match = re.fullmatch(r'https://(?:www\.|m\.)attheraces\.com/(results|replays)/(\d{2}-[A-Za-z]+-\d{4})/?', url)
    if not match or 'Client Challenge' in str(soup.title):
        raise ValueError(f'{path} is not a saved, dated ATR results or replay page')
    kind, day = match.groups()
    datetime.strptime(day, '%d-%B-%Y')
    if kind == 'replays':
        mapping = _parse_replay_index(soup, day)
        target = Path(directory)/f'{day}.replays.json'
    else:
        mapping = {card: native for card, native in atr._parse_results_index(str(soup)).items()
                   if re.fullmatch(rf'/racecard/[^/]+/{re.escape(day)}/\d{{4}}', card)}
        target = Path(directory)/f'{day}.json'
    folded = {card.casefold(): native for card, native in mapping.items()}
    if not mapping or len(folded) != len(mapping) or len(set(mapping.values())) != len(mapping):
        raise ValueError(f'{path} has no unambiguous dated ATR race index')
    previous = read_json(target) or {}
    if any(folded.get(card.casefold()) != str(native) for card, native in previous.get('race_ids', {}).items()):
        raise ValueError(f'{path} would remove or change existing ATR links for {day}')
    atomic_json(target, {'source_url':url, 'source_format':f'saved_{kind}_html',
                         'source_sha256':hashlib.sha256(raw).hexdigest(), 'source_file':str(path.resolve()),
                         'retrieved_at':datetime.now(timezone.utc).isoformat(), 'race_ids':mapping})
    return {'date':day, 'source_races':len(mapping), 'index':str(target)}


def plan_legacy_hints(jobs, cache):
    """Check legacy IDs against complete runner lists on the same day/course.

    Results remain lookup hints, not accepted telemetry. A subsequent source
    fetch must still pass date/time and canonical runner validation.
    """
    signatures = defaultdict(list)
    by_id = {job['race_id']:job for job in jobs}
    for job in jobs:
        runners = tuple(sorted(row['horse_key'] for row in json.loads(job['expected'])))
        signatures[(job['race_date'],atr._normalise_track_key(job['track']),runners)].append(job['race_id'])
    candidates, claimed = defaultdict(set), defaultdict(list)
    for job in jobs:
        if job['native_id']:
            claimed[job['native_id']].append(job['race_id'])
    frame = cache.reindex(columns=['race_date','track','horse_name','atr_raceid']).copy()
    frame['race_date'] = frame.race_date.astype(str).str[:10]
    frame['atr_raceid'] = frame.atr_raceid.astype(str).str.replace(r'\.0$', '',regex=True)
    frame['track'] = frame.track.map(atr._normalise_track_key)
    frame['horse_name'] = frame.horse_name.map(atr._normalise_horse_key)
    for native_id, group in frame.groupby('atr_raceid',sort=False):
        if not re.fullmatch(r'\d+',native_id) or group.race_date.nunique()!=1 or group.track.nunique()!=1:
            continue
        signature = (group.race_date.iloc[0],group.track.iloc[0],tuple(sorted(set(group.horse_name))))
        matching = signatures.get(signature,[])
        if len(matching)==1:
            candidates[matching[0]].add(native_id)
    supported = {race_id:next(iter(ids)) for race_id,ids in candidates.items() if len(ids)==1}
    supported = {race_id:native_id for race_id,native_id in supported.items()
                 if by_id[race_id]['state'] not in ('data','data_partial') or by_id[race_id]['native_id']==native_id}
    owner = defaultdict(list)
    for race_id,native_id in supported.items():
        owner[native_id].append(race_id)
    supported = {race_id:native_id for race_id,native_id in supported.items() if len(owner[native_id])==1}
    verified = defaultdict(list)
    for job in jobs:
        if job['native_id'] and job['state'] in ('data','data_partial'):
            verified[job['native_id']].append(job['race_id'])
    supported = {race_id:native_id for race_id,native_id in supported.items()
                 if not verified[native_id] or race_id in verified[native_id]}
    changes, unresolved = {}, []
    for native_id,races in claimed.items():
        owners = verified.get(native_id) or owner.get(native_id, [])
        if len(races)>1 or owners and any(race_id not in owners for race_id in races):
            for race_id in races:
                if race_id not in owners and by_id[race_id]['state'] not in ('data','data_partial'):
                    changes[race_id] = None
                    unresolved.append({'race_id':race_id,'previous_native_id':native_id,
                                       'reason':'Legacy ID attached to a different or ambiguous runner field'})
    for race_id,native_id in supported.items():
        if by_id[race_id]['state'] not in ('data','data_partial') and by_id[race_id]['native_id']!=native_id:
            changes[race_id] = native_id
    return {'supported':supported,'changes':changes,'cleared_conflicts':unresolved,
            'legacy_native_ids':int(frame.atr_raceid.nunique())}


def apply_legacy_plan(con, plan, source, sha=None):
    """Caller commits reviewed hint corrections; saved runner payloads are untouched."""
    current_rows = {row['race_id']:dict(row) for row in con.execute('SELECT race_id,native_id,state FROM jobs')}
    for race_id,native_id in plan['changes'].items():
        current = current_rows.get(race_id)
        if current is None or current['state'] in ('data','data_partial'):
            continue
        con.execute('UPDATE jobs SET native_id=? WHERE race_id=?',(native_id,race_id))
        evidence(con,race_id,native_id,'roster_supported' if native_id else 'conflict',source,sha,
                 {'previous_native_id':current['native_id'],'method':'complete runner-list comparison'})
        current['native_id'] = native_id
    for race_id,native_id in plan['supported'].items():
        current = current_rows.get(race_id)
        if current and current['native_id']==native_id and current['state'] not in ('data','data_partial'):
            evidence(con,race_id,native_id,'roster_supported',source,sha,{'method':'complete runner-list comparison'})


def main():
    import argparse
    from contextlib import closing
    import sqlite3
    import pandas as pd
    parser = argparse.ArgumentParser(description='Audit ATR native IDs or import saved results and replay pages')
    parser.add_argument('--root',type=Path,default=Path(atr.ATR_CACHE_DIR)/'rebuild_v2')
    parser.add_argument('--out',type=Path,default=Path(atr.ATR_CACHE_DIR)/'id_audit.json')
    parser.add_argument('--import-results',type=Path,nargs='+',help='Saved ATR results or replay HTML files')
    args = parser.parse_args()
    if args.import_results:
        for path in args.import_results:
            print(json.dumps(import_saved_results(path, atr.ATR_INDEX_DIR)))
        return
    with closing(sqlite3.connect((args.root/'rebuild.sqlite').resolve().as_uri()+'?mode=ro',uri=True)) as con:
        con.row_factory = sqlite3.Row
        jobs = [dict(row) for row in con.execute('SELECT * FROM jobs')]
    cache = pd.read_parquet(atr.ATR_CACHE_FILE,columns=['race_date','track','horse_name','atr_raceid'])
    report = plan_legacy_hints(jobs,cache)
    report['missing_ids'] = sum(job['native_id'] is None for job in jobs)
    atomic_json(args.out,report)
    print(json.dumps({'missing_ids':report['missing_ids'],'supported_hints':len(report['supported']),
                      'proposed_corrections':len(report['changes']),'report':str(args.out)}))


if __name__ == '__main__':
    main()
