"""Prior BHA observations as numeric candidates for ordinary feature selection.

Only manifest-selected, already published sources are read. Feature availability
uses the configured policy; source capture times remain unchanged for audit.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.bha_events import build_observations, prepare_canonical, METRICS, GROUPS, CONTEXT
from src.ingestion.storage import atomic_json, atomic_parquet, read_json, writer_lock
from src.source_dataset import committed_dataset, current_manifest, require_manifest, source_path, use_manifest

RECENT_GROUPS = tuple(GROUPS) + ('breathing', 'lame', 'bled', 'interference_suffered', 'unsuited_going')
FEATURES = tuple('bha_' + metric + suffix for metric in METRICS
                 for suffix in ('_last', '_rate_3', '_rate_5', '_count_5')) + tuple(
    'bha_' + metric + suffix for metric in RECENT_GROUPS
    for suffix in ('_days_since_5', '_starts_since_5')) + (
    'bha_report_count_3', 'bha_report_count_5', 'bha_named_count_5', 'bha_days_since_report',
    'bha_goingstick_last', 'bha_goingstick_mean_3', 'bha_rail_change_last',
    'bha_hung_left_x_direction', 'bha_hung_right_x_direction',
    'bha_prefers_softer_x_soft_going', 'bha_prefers_faster_x_soft_going',
    'bha_withdrawals_365d', 'bha_health_withdrawals_365d', 'bha_going_withdrawals_365d',
    'bha_days_since_withdrawal', 'bha_days_since_health_withdrawal', 'bha_days_since_going_withdrawal',
    'bha_last_withdrawal_health', 'bha_last_withdrawal_going')


def availability_policy(value=None):
    value = config.BHA_FEATURE_AVAILABILITY_POLICY if value is None else value
    if value not in {'recorded', 'race_plus_48h'}:
        raise ValueError('Unknown BHA feature availability policy')
    return value


def _available_times(events, recorded, policy):
    if policy == 'recorded':
        return recorded
    return events + int(pd.Timedelta(hours=48).value)


def source_paths():
    """The selected dataset's results and BHA tables; None where it selects none."""
    sources = require_manifest()['sources']
    paths = {}
    for provider, names in (('sl', ('results',)), ('bha', ('reports', 'documents', 'race_notes'))):
        selected = (sources.get(provider) or {}).get('tables', {})
        for name in names:
            # source_path honours a dataset stored outside the shared registry.
            paths[provider+'/'+name] = str(source_path(provider, name)) if name in selected else None
    return paths


@lru_cache(maxsize=2)
def _load_bundle(paths_tuple, parser_hash, cache_root):
    paths = dict(paths_tuple)
    if not paths.get('sl/results') or not paths.get('bha/reports'):
        return None
    canonical = pd.read_parquet(paths['sl/results'],
        columns=['race_id', 'horse_id', 'horse_name', 'race_date', 'track', 'off_time'])
    digest = hashlib.sha256(json.dumps([paths_tuple, parser_hash]).encode()).hexdigest()[:24]
    root = Path(cache_root)/digest
    if (root/'complete.json').exists():
        return (prepare_canonical(canonical), pd.read_parquet(root/'observations.parquet'),
                pd.read_parquet(root/'withdrawals.parquet'), read_json(root/'complete.json'))
    tables = {name: pd.read_parquet(path) for key, path in paths.items()
              if path and key.startswith('bha/') for name in (key.split('/')[1],)}
    prepared, observations, withdrawals, audit = build_observations(canonical, tables)
    # Concurrent helpers may compute the same deterministic observations; only
    # one writes the cache, with the completion receipt committed last.
    with writer_lock(root/'.writer.lock') as acquired:
        if acquired and not (root/'complete.json').exists():
            atomic_parquet(observations, root/'observations.parquet')
            atomic_parquet(withdrawals, root/'withdrawals.parquet')
            atomic_json(root/'complete.json', audit)
    return prepared, observations, withdrawals, audit


def load_bundle(paths=None):
    paths = source_paths() if paths is None else paths
    digest = hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('bha_events.py').read_bytes()).hexdigest()
    return _load_bundle(tuple(sorted(paths.items())), digest, str(Path(config.DATA_DIR)/'processed/bha_history'))


def _target_times(frame):
    day = pd.to_datetime(frame.race_date, format='mixed').dt.strftime('%Y-%m-%d')
    off = frame.get('off_time', pd.Series('00:00', index=frame.index)).astype(str).str[:5]
    event = pd.to_datetime(day+' '+off, utc=True, errors='raise')
    if 'prediction_as_of' in frame:
        captured = pd.to_datetime(frame.prediction_as_of, utc=True, errors='raise')
        event = event.where(event.le(captured), captured)
    return event.to_numpy(dtype='datetime64[ns]').view('i8')


def _slots(canonical, observations):
    if observations.empty:
        return np.full((len(canonical), 1), -1, dtype=np.int32), observations
    values = observations.copy()
    values['available_at'] = pd.to_datetime(values.available_at, utc=True, errors='raise')
    # Same capture can expose separate enquiry/runner-note sections. A later
    # full PDF remains a separate observation, not a future-filled earlier row.
    values = values.groupby(['run_index', 'available_at'], as_index=False, sort=False)[[*METRICS, *CONTEXT]].max()
    values = values.sort_values(['run_index', 'available_at'], kind='stable').reset_index(drop=True)
    rank = values.groupby('run_index').cumcount().to_numpy()
    slots = np.full((len(canonical), int(rank.max())+1), -1, dtype=np.int32)
    slots[values.run_index.to_numpy(int), rank] = np.arange(len(values))
    return slots, values


def compute_features(frame, canonical, observations, withdrawals=None, *, decision_times=None, policy=None):
    """Chunk/order-invariant summaries; each prior start occupies a window slot."""
    policy = availability_policy(policy)
    n = len(frame)
    output = {name: np.full(n, np.nan, dtype=np.float32) for name in FEATURES}
    for name in ('bha_report_count_3', 'bha_report_count_5', 'bha_named_count_5'):
        output[name].fill(0)
    if n == 0 or canonical.empty:
        return pd.DataFrame(output, index=frame.index)
    decisions = _target_times(frame) if decision_times is None else np.asarray(decision_times, dtype='datetime64[ns]').view('i8')
    horse = frame.get('horse_id', pd.Series('', index=frame.index)).fillna('').astype(str).to_numpy()
    event = canonical._event.to_numpy(dtype='datetime64[ns]').view('i8')
    canonical_horses = canonical.horse_id.to_numpy()
    # Canonical is sorted by horse/event. Search a structured key to avoid a
    # Python loop over every forecast row or repeatedly loading horse history.
    dtype = np.dtype([('horse', 'O'), ('event', 'i8')])
    history_keys = np.empty(len(canonical), dtype=dtype)
    history_keys['horse'], history_keys['event'] = canonical_horses, event
    target_keys = np.empty(n, dtype=dtype)
    target_keys['horse'], target_keys['event'] = horse, decisions
    end = np.searchsorted(history_keys, target_keys, side='left')
    positions = end[:, None]-np.arange(1, 6)
    safe = np.maximum(positions, 0)
    prior = (positions >= 0) & (canonical_horses[safe] == horse[:, None]) & (event[safe] < decisions[:, None])
    if 'race_id' in frame:
        prior &= canonical.race_id.to_numpy()[safe] != frame.race_id.astype(str).to_numpy()[:, None]
    slots, values = _slots(canonical, observations)
    chosen = np.full((n, 5), -1, dtype=np.int32)
    if not values.empty:
        recorded = values.available_at.to_numpy(dtype='datetime64[ns]').view('i8')
        available = _available_times(event[values.run_index.to_numpy(int)], recorded, policy)
        for j in range(slots.shape[1]):
            source = slots[safe, j]
            usable = prior & (source >= 0) & (available[np.maximum(source, 0)] <= decisions[:, None])
            chosen = np.where(usable, source, chosen)
    valid = chosen >= 0
    for count in (3, 5):
        output[f'bha_report_count_{count}'] = valid[:, :count].sum(axis=1).astype(np.float32)
    if values.empty:
        if withdrawals is not None and not withdrawals.empty:
            _withdrawal_features(output, horse, decisions, withdrawals, valid.any(axis=1), policy=policy)
        return pd.DataFrame(output, index=frame.index)
    selected = np.maximum(chosen, 0)
    collected = {}
    for metric in METRICS:
        data = np.where(valid, values[metric].to_numpy(float)[selected], np.nan)
        collected[metric] = data
        output[f'bha_{metric}_last'] = data[:, 0].astype(np.float32)
        for count in (3, 5):
            support = valid[:, :count].sum(axis=1)
            total = np.nansum(data[:, :count], axis=1)
            output[f'bha_{metric}_rate_{count}'] = np.divide(total, support,
                out=np.full(n, np.nan), where=support > 0).astype(np.float32)
        output[f'bha_{metric}_count_5'] = np.where(valid.any(axis=1), np.nansum(data, axis=1), np.nan).astype(np.float32)
    output['bha_named_count_5'] = np.nansum(collected['named_in_report'], axis=1).astype(np.float32)
    age = (decisions[:, None]-event[safe])/86_400_000_000_000
    nearest = np.where(valid, age, np.inf).min(axis=1)
    output['bha_days_since_report'] = np.where(np.isfinite(nearest), nearest, np.nan).astype(np.float32)
    for metric in RECENT_GROUPS:
        hit = collected[metric] > 0
        days = np.where(hit, age, np.inf).min(axis=1)
        starts = np.where(hit, np.arange(5)[None, :], np.inf).min(axis=1)
        output[f'bha_{metric}_days_since_5'] = np.where(np.isfinite(days), days, np.nan).astype(np.float32)
        output[f'bha_{metric}_starts_since_5'] = np.where(np.isfinite(starts), starts, np.nan).astype(np.float32)
    gst = np.where(valid, values.goingstick.to_numpy(float)[selected], np.nan)
    output['bha_goingstick_last'] = gst[:, 0].astype(np.float32)
    count = np.isfinite(gst[:, :3]).sum(axis=1)
    output['bha_goingstick_mean_3'] = np.divide(np.nansum(gst[:, :3], axis=1), count,
        out=np.full(n, np.nan), where=count > 0).astype(np.float32)
    output['bha_rail_change_last'] = np.where(valid[:, 0], values.rail_change_reported.to_numpy(float)[selected[:, 0]], np.nan).astype(np.float32)
    if 'track_direction' in frame:
        direction = pd.to_numeric(frame.track_direction, errors='coerce').to_numpy()
        for side in ('left', 'right'):
            output[f'bha_hung_{side}_x_direction'] = output[f'bha_hung_{side}_rate_5']*direction
    if 'going_Soft' in frame or 'going_Heavy' in frame:
        soft = sum(pd.to_numeric(frame.get(c, pd.Series(0, index=frame.index)), errors='coerce').to_numpy()
                   for c in ('going_Soft', 'going_Heavy'))
        for kind in ('softer', 'faster'):
            output[f'bha_prefers_{kind}_x_soft_going'] = output[f'bha_prefers_{kind}_rate_5']*soft
    if withdrawals is not None and not withdrawals.empty:
        _withdrawal_features(output, horse, decisions, withdrawals, valid.any(axis=1), policy=policy)
    return pd.DataFrame(output, index=frame.index)


def _withdrawal_features(output, horses, decisions, withdrawals, covered, *, policy):
    for name in ('bha_withdrawals_365d', 'bha_health_withdrawals_365d', 'bha_going_withdrawals_365d'):
        output[name] = np.where(covered, 0., np.nan).astype(np.float32)
    history = withdrawals.copy()
    history['event_at'] = pd.to_datetime(history.event_at, utc=True, errors='raise')
    history['available_at'] = pd.to_datetime(history.available_at, utc=True, errors='raise')
    groups = {str(k): group.sort_values('event_at') for k, group in history.groupby('horse_id')}
    for horse in set(horses) & set(groups):
        group = groups[horse]
        events = group.event_at.to_numpy(dtype='datetime64[ns]').view('i8')
        recorded = group.available_at.to_numpy(dtype='datetime64[ns]').view('i8')
        available = _available_times(events, recorded, policy)
        indices = np.flatnonzero(horses == horse)
        # Withdrawal records are sparse. Work per horse, in small blocks.
        for begin in range(0, len(indices), 2048):
            target = indices[begin:begin+2048]
            valid = (events[None, :] < decisions[target, None]) & (available[None, :] <= decisions[target, None])
            age = (decisions[target, None]-events[None, :])/86_400_000_000_000
            support = covered[target] | valid.any(axis=1)
            for prefix, flag in (('', np.ones(len(group), dtype=bool)),
                                 ('health_', group.health.to_numpy() > 0), ('going_', group.going.to_numpy() > 0)):
                selected = valid & flag[None, :]
                days = np.where(selected, age, np.inf).min(axis=1)
                output[f'bha_days_since_{prefix}withdrawal'][target] = np.where(np.isfinite(days), days, np.nan)
                output[f'bha_{prefix}withdrawals_365d'][target] = np.where(support, (selected & (age <= 365)).sum(axis=1), np.nan)
            last = np.where(valid, np.arange(len(group))[None, :], -1).max(axis=1)
            for flag in ('health', 'going'):
                output[f'bha_last_withdrawal_{flag}'][target] = np.where(last >= 0, group[flag].to_numpy()[np.maximum(last, 0)], np.nan)


def add_bha_features(frame):
    bundle = load_bundle()
    if bundle is None:
        features = pd.DataFrame(np.nan, index=frame.index, columns=FEATURES, dtype=np.float32)
        features[['bha_report_count_3', 'bha_report_count_5', 'bha_named_count_5']] = 0.
    else:
        canonical, observations, withdrawals, _ = bundle
        features = compute_features(frame, canonical, observations, withdrawals)
    return pd.concat([frame.drop(columns=[c for c in FEATURES if c in frame], errors='ignore'), features], axis=1)


def cache_signature(cutoff):
    """Only observations available through the stored history can stale it."""
    policy = availability_policy()
    paths = source_paths()
    end = pd.Timestamp(cutoff).normalize()
    end = end.tz_localize('UTC') if end.tzinfo is None else end.tz_convert('UTC')
    end += pd.Timedelta(days=1)
    output = {}
    for name in ('reports', 'documents', 'race_notes'):
        path = paths.get('bha/'+name)
        if not path:
            output[name] = None
            continue
        frame = pd.read_parquet(path)
        available = (pd.to_datetime(frame.available_at, utc=True, errors='raise') if policy == 'recorded'
                     else pd.to_datetime(frame.race_date, utc=True, errors='raise') + pd.Timedelta(hours=48))
        eligible = available.lt(end)
        frame = frame.loc[eligible]
        keys = [k for k in ('report_key', 'race_number', 'available_at') if k in frame]
        frame = frame.sort_values(keys, kind='stable')
        digest = hashlib.sha256(str([policy, list(frame.columns)]).encode())
        digest.update(pd.util.hash_pandas_object(frame.astype('string'), index=False).values.tobytes())
        output[name] = digest.hexdigest()[:16]
    return output


def main():
    import argparse
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='Materialise BHA horse observations and current feature state; no models')
    parser.add_argument('--out', type=Path, default=Path(config.DATA_DIR)/'processed/bha_candidates')
    parser.add_argument('--historical', action='store_true', help='Also write all dated historical BHA candidate rows')
    args = parser.parse_args()
    manifest = current_manifest() or committed_dataset()
    if manifest is None:
        raise RuntimeError('No committed source dataset; publish sources first (python -m src.ingestion --publish-sources).')
    with use_manifest(manifest):
        paths = source_paths()
        bundle = load_bundle(paths)
    if bundle is None:
        raise RuntimeError('No accepted BHA and canonical source versions are available')
    canonical, observations, withdrawals, audit = bundle
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_parquet(observations, args.out/'observations.parquet')
    atomic_parquet(withdrawals, args.out/'withdrawals.parquet')
    # This is a horse-state query at a recorded instant, not fabricated races.
    instant = pd.Timestamp.now(tz='UTC')
    horses = canonical[['horse_id']].drop_duplicates().reset_index(drop=True)
    state = compute_features(horses, canonical, observations, withdrawals,
                             decision_times=np.repeat(instant.to_datetime64(), len(horses)))
    atomic_parquet(pd.concat([horses, state], axis=1), args.out/'current_horse_state.parquet')
    historical = _write_historical(canonical, observations, withdrawals, args.out) if args.historical else None
    atomic_json(args.out/'coverage.json', {**audit, 'as_of': instant.isoformat(), 'features': list(FEATURES),
        'source_paths': paths,
        'code_hashes': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('bha_features.py', 'bha_events.py')},
        'state_rows': len(state), 'populated_by_feature': state.notna().sum().to_dict(),
        'availability_policy': availability_policy(), 'historical': historical,
        'availability_assumption': ('48 hours after the source event; original publication times unverified'
                                    if availability_policy() == 'race_plus_48h' else None),
        'raw_capture_times_preserved': True, 'models_trained': 0})
    print(json.dumps({**audit, 'feature_count': len(FEATURES), 'state_rows': len(state), 'out': str(args.out)}))


def _write_historical(canonical, observations, withdrawals, root):
    """Materialise real runner rows in bounded chunks without fitting a model."""
    import logging
    import os
    from uuid import uuid4
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.ingestion.storage import _replace_after_readers
    target = root/'historical_features.parquet'
    temporary = target.with_name(target.name+'.'+uuid4().hex+'.tmp')
    covered, written, writer = 0, 0, None
    try:
        for begin in range(0, len(canonical), 50_000):
            frame = canonical.iloc[begin:begin+50_000]
            features = compute_features(frame, canonical, observations, withdrawals)
            covered += int(features.bha_report_count_5.gt(0).sum())
            chunk = pd.concat([frame[['race_id', 'horse_id', 'race_date']], features], axis=1)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                metadata = {**(table.schema.metadata or {}),
                            b'bha_availability_policy': availability_policy().encode()}
                writer = pq.ParquetWriter(temporary, table.schema.with_metadata(metadata))
            writer.write_table(table)
            written += len(frame)
            logging.getLogger(__name__).info('Historical BHA features: %d/%d runner rows', written, len(canonical))
        if writer is None:
            return {'rows': 0, 'rows_with_report_history': 0}
        writer.close()
        writer = None
        assert pq.ParquetFile(temporary).metadata.num_rows == len(canonical)
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        _replace_after_readers(temporary, target)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    return {'rows': written, 'rows_with_report_history': covered, 'file': str(target)}


if __name__ == '__main__':
    main()
