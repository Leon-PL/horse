"""Bounded live ATR benchmark; writes only a separate research directory.

Run from the repository root. Stops immediately on the first challenge,
HTTP block or transport failure. Uses the existing collector without changing
its production cache, skip-list, parsers or retry configuration.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import attheraces_scraper as atr


class StopBenchmark(Exception):
    pass


def main():
    output = Path('data/research/atr_benchmark_2026-10-17')
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'summary.json'
    if summary_path.exists():
        raise SystemExit('Benchmark output already exists; preserve the previous run.')
    seeds = [
        {'race_date': '2025-03-11', 'track': 'Cheltenham', 'off_time': '13:20', 'expected_runners': 11},
        {'race_date': '2025-03-11', 'track': 'Wolverhampton', 'off_time': '16:30', 'expected_runners': 9},
    ]
    with sqlite3.connect('file:data/races.db?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        candidates = [dict(r) for r in db.execute('''
            SELECT race_date, track, off_time, MAX(num_runners) expected_runners,
                   COUNT(*) result_rows
            FROM results WHERE race_date BETWEEN '2025-01-01' AND '2026-09-30'
              AND track IN ('Wolverhampton','Southwell','Newcastle','Lingfield',
                            'Cheltenham','Ascot','Newmarket','Dundalk','Curragh')
              AND off_time IS NOT NULL
            GROUP BY race_date,track,off_time
            ORDER BY race_date,track,off_time
        ''')]
    # Deterministic spread over the historical interval, rather than one meeting.
    selected = [candidates[round(i*(len(candidates)-1)/47)] for i in range(48)] if len(candidates) >= 48 else candidates
    races = seeds + selected
    (output / 'planned_sample.json').write_text(json.dumps(races, indent=2), encoding='utf-8')
    started = time.perf_counter()
    attempts, requests_log = [], []
    fetcher = atr._Fetcher(request_delay=2.0)
    original_get = fetcher.get

    def monitored_get(url, *, referer=None):
        begin = time.perf_counter()
        response = original_get(url, referer=referer)
        status, body = response if response is not None else (None, '')
        entry = {'url': url, 'seconds': round(time.perf_counter()-begin, 3),
                 'status': status, 'characters': len(body),
                 'challenge_or_block': response is None or status in atr._BLOCKED_STATUSES}
        requests_log.append(entry)
        print(json.dumps({'request': entry}), flush=True)
        if entry['challenge_or_block']:
            raise StopBenchmark('Stopped on first block/challenge/transport failure')
        (output / f'response_{len(requests_log):03d}.html').write_text(body, encoding='utf-8')
        return response

    fetcher.get = monitored_get
    stop_reason = 'sample_completed'
    try:
        for index, race in enumerate(races):
            begin = time.perf_counter()
            record = dict(race)
            try:
                hhmm = atr._off_time_to_hhmm(race['off_time'])
                if index >= 2 and atr._is_bst(race['race_date']):
                    hhmm = atr._add_one_hour(hhmm)
                outcome, rows = atr._fetch_one(fetcher, atr._track_to_slug(race['track']),
                                              atr._date_to_atr(race['race_date']), hhmm)
                rows = rows or []
                record.update(outcome=outcome, returned_runners=len(rows),
                              runners_with_splits=sum(bool(r.get('atr_split_times')) for r in rows),
                              runners_with_speeds=sum(bool(r.get('atr_sectional_speeds')) for r in rows))
                record['runner_count_matches'] = len(rows) == race['expected_runners']
                record['split_rows_match'] = record['runners_with_splits'] == race['expected_runners']
                record['times_and_tools_match'] = record['split_rows_match'] and record['runners_with_speeds'] == race['expected_runners']
                (output / f'race_{index+1:03d}.json').write_text(json.dumps(rows, indent=2, default=str), encoding='utf-8')
            except StopBenchmark as exc:
                record.update(outcome='blocked', detail=str(exc))
                stop_reason = str(exc)
            record['seconds'] = round(time.perf_counter()-begin, 3)
            attempts.append(record)
            print(json.dumps({'race':record}), flush=True)
            if record['outcome'] == 'blocked':
                break
    finally:
        fetcher.close()
        elapsed = time.perf_counter()-started
        complete = sum(r.get('times_and_tools_match', False) for r in attempts)
        summary = {'planned_races':len(races), 'attempted_races':len(attempts),
                   'complete_times_and_tools_races':complete,
                   'elapsed_seconds':round(elapsed,3), 'transport':fetcher.mode,
                   'minimum_request_gap_seconds':2.0, 'stop_reason':stop_reason,
                   'complete_races_per_hour_observed':round(complete*3600/elapsed,2) if complete else None,
                   'throughput_established':complete >= 10 and stop_reason == 'sample_completed',
                   'attempts':attempts, 'requests':requests_log}
        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps({'summary':summary}), flush=True)


if __name__ == '__main__':
    main()
