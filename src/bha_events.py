"""BHA text to source-backed horse observations, with conservative attribution.

No outcomes, prices or models are used. Missing reports are not negative reports.
Canonical IDs are linked only by a unique horse within a dated course/field.
"""
from collections import defaultdict
import logging
import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from src.utils import canonical_track_key, normalise_off_time_key

logger = logging.getLogger(__name__)

PATTERNS = {
    'lame': r'\blame(?:ness)?\b', 'bled': r'\b(?:bled|bleeding|blood (?:at|from) (?:the )?nostrils)\b',
    'breathing': r'\b(?:respiratory noise|breathing (?:problem|issue|difficulty)|made a noise|wind problem)\b',
    'distress': r'\b(?:distressed|heat stress|exhausted|dehydrated)\b',
    'injury': r'\b(?:injur(?:y|ed)|wound|sore|laceration|struck into|tendon)\b',
    'lost_shoe': r'\b(?:lost|losing|shed|slipped)(?:\s+[\w-]+){0,4}\s+shoe\b',
    'vet_exam': r'\b(?:examined|examination|examining)\b',
    'vet_clear': r'\b(?:nothing to report|no abnormalities|failed to reveal any abnormalities|deemed fit to race)\b',
    'hung_left': r'\b(?:hung|hanging|jumped|jumping|edged|drifted)\s+left(?:-handed)?\b',
    'hung_right': r'\b(?:hung|hanging|jumped|jumping|edged|drifted)\s+right(?:-handed)?\b',
    'ran_freely': r'\b(?:ran|raced|running|racing|pulled)\s+(?:too\s+)?(?:freely|keenly|hard)\b',
    'slow_start': r'\b(?:slow(?:ly)? away|slow(?:ly)? (?:to|into stride)|missed the break|jumped awkwardly|stumbled leaving)\b',
    'refused': r'\b(?:refused|reluctant to (?:race|start)|refused to race)\b',
    'jump_error': r'\b(?:mistake|blunder|jumped (?:poorly|awkwardly)|jumping (?:poorly|error))\b',
    'interference_suffered': r'\b(?:was|were|being|became)(?:\s+\w+){0,3}\s+(?:hampered|checked|impeded|intimidated|bumped|brought down)\b',
    'interference_caused': r'\b(?:hampered|hampering|impeding|caused (?:considerable )?interference|causing .{0,20}interference)\b',
    'tack_problem': r'\b(?:saddle (?:slipped|slipping)|lost (?:an? |both )?stirrup|broken (?:rein|girth)|tack (?:broke|problem))\b',
    'poor_performance': r'\b(?:poor|disappointing|below-par|below par)\s+(?:run|performance)|\bnever travelling\b',
    'no_explanation': r'\b(?:no|could not provide an?) explanation\b',
    'unsuited_going': r'\b(?:unsuited|unsuitable)(?:.{0,65})\b(?:going|ground|surface)\b',
    'prefers_softer': r'\b(?:prefer|preferred|prefers|appreciate)(?:.{0,25})\b(?:softer|easier)\b',
    'prefers_faster': r'\b(?:prefer|preferred|prefers|appreciate)(?:.{0,25})\b(?:quicker|faster|firmer)\b',
    'red_hood_to_post': r'\b(?:red hood to post|red hood on the way to post)\b',
    'early_to_post': r'\b(?:go|going|went) early to post\b',
    'earplugs_removed': r'\bearplugs(?:.{0,40})\bremoved\b',
    'partially_shod': r'\bpartially shod\b',
}
TAGS = tuple(PATTERNS)
REGEX = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}
GROUPS = {
    'physical_issue': ('lame', 'bled', 'breathing', 'distress', 'injury'),
    'behaviour_issue': ('hung_left', 'hung_right', 'ran_freely', 'slow_start', 'refused'),
    'equipment_issue': ('lost_shoe', 'tack_problem'),
    'performance_explanation': ('poor_performance', 'no_explanation', 'unsuited_going'),
}
METRICS = TAGS + tuple(GROUPS) + ('named_in_report',)
CONTEXT = ('goingstick', 'rail_change_reported')


def horse_key(name):
    name = re.sub(r'\s*\([A-Za-z]{2,3}\)\s*$', '', str(name)).strip()
    return re.sub(r'[^a-z0-9]', '', name.casefold())


def prepare_canonical(frame):
    required = ['race_id', 'horse_id', 'horse_name', 'race_date', 'track', 'off_time']
    out = frame[required].copy()
    out = out.dropna(subset=['race_id', 'horse_id', 'horse_name'])
    out['race_id'], out['horse_id'] = out.race_id.astype(str), out.horse_id.astype(str)
    if out.duplicated(['race_id', 'horse_id']).any():
        raise ValueError('Duplicate canonical BHA runner identity')
    out['_day'] = pd.to_datetime(out.race_date, format='mixed').dt.strftime('%Y-%m-%d')
    out['_event'] = pd.to_datetime(out._day + ' ' + out.off_time.map(normalise_off_time_key), utc=True, errors='raise')
    out['_track'] = out.track.map(canonical_track_key)
    out['_local'] = out._event.dt.tz_convert('Europe/London').dt.strftime('%H:%M')
    out['_name'] = out.horse_name.map(horse_key)
    return out.sort_values(['horse_id', '_event', 'race_id'], kind='stable').reset_index(drop=True)


def _clock(match):
    hour, minute = map(int, match.group(1).split(':'))
    return f"{hour % 12 + (12 if match.group(2).lower() == 'pm' else 0):02d}:{minute:02d}"


def html_sections(value):
    soup = BeautifulSoup(str(value or ''), 'html.parser')
    blocks, current, parts = [], None, []
    nodes = soup.find_all(['p', 'h3', 'h4'])
    for node in nodes:
        text = node.get_text(' ', strip=True)
        heading = re.match(r'^Race\s+\d+\s*[-:]\s*(\d{1,2}:\d{2})\s*(am|pm)', text, re.I)
        if heading:
            if current is not None:
                blocks.append((current, '\n\n'.join(parts)))
            current, parts = _clock(heading), []
        elif current is not None:
            parts.append(text)
    if current is not None:
        blocks.append((current, '\n\n'.join(parts)))
    return blocks or [(None, soup.get_text(' ', strip=True))]


def _name_regex(name):
    bare = re.sub(r'\s*\([A-Za-z]{2,3}\)\s*$', '', str(name)).strip().upper()
    return re.escape(bare).replace(r'\ ', r'\s+').replace("'", "['’]")


def _positive(tag, text):
    if tag in ('vet_exam', 'vet_clear', 'no_explanation'):
        return bool(REGEX[tag].search(text))
    for match in REGEX[tag].finditer(text):
        before = text[max(0, match.start()-42):match.start()]
        if tag == 'interference_caused' and re.search(r'\b(?:was|were|being|been|became)(?:\s+\w+){0,3}\s*$', before, re.I):
            continue
        if not re.search(r'\b(?:no|not|without|denied|neither|failed to reveal)(?:\s+\w+){0,5}\s*$', before, re.I):
            return True
    return False


def attribute(text, roster, *, covered):
    """Attribute local clauses; do not assign every incident to every name."""
    by_name = defaultdict(list)
    for index, name in roster:
        by_name[horse_key(name)].append((index, name))
    unique = {key: rows[0] for key, rows in by_name.items() if len(rows) == 1 and key}
    output = {index: {tag: 0. for tag in (*TAGS, 'named_in_report')} for index, _ in unique.values()} if covered else {}
    if not unique:
        return output, []
    lookup = {_name_regex(name): index for index, name in unique.values()}
    pattern = re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(sorted(lookup, key=len, reverse=True)) + r')(?:\s*\([A-Z]{2,3}\))?(?![A-Za-z0-9])')
    identities = {horse_key(name): index for index, name in unique.values()}
    evidence = []
    for paragraph in re.split(r'\n\s*\n', str(text)):
        previous = None
        for clause in re.split(r'(?<=[.!?;])\s+', re.sub(r'\s+', ' ', paragraph).strip()):
            matches = list(pattern.finditer(clause))
            if not matches:
                if previous is not None and re.search(r'\b(?:the gelding|the mare|the filly|the colt|the horse)\b', clause, re.I):
                    segments = [(previous, clause)]
                else:
                    continue
            else:
                segments = []
                for i, match in enumerate(matches):
                    index = identities.get(horse_key(match.group()))
                    if index is None:
                        continue
                    stop = matches[i+1].start() if i+1 < len(matches) else len(clause)
                    # An unrecognised capitalised horse is still a subject
                    # boundary, not permission to assign its problem to ALPHA.
                    unknown = re.search(r'\b[A-Z][A-Z\'’\-]+(?:\s+[A-Z][A-Z\'’\-]+)*\b', clause[match.end():stop])
                    if unknown:
                        stop = match.end()+unknown.start()
                    segment = clause[match.end():stop]
                    # Verbs can precede the horse ("examined ALPHA"). Only
                    # local, explicit victim/examination language is carried.
                    prefix = clause[max(0, match.start()-65):match.start()]
                    if re.search(r'\b(?:examined|examination of)\s*$', prefix, re.I):
                        segment += ' examined'
                    if re.search(r'\b(?:hampered|hampering|impeding|interference to|brought down)\s*$', prefix, re.I):
                        segment += ' was hampered'
                    if re.search(r'\bhampered by\s*$', prefix, re.I):
                        segment += ' caused interference'
                    segments.append((index, segment))
                previous = segments[0][0] if len({i for i, _ in segments}) == 1 else None
            for index, segment in segments:
                values = output.setdefault(index, {tag: 0. for tag in (*TAGS, 'named_in_report')})
                values['named_in_report'] = 1.
                for tag in TAGS:
                    values[tag] = max(values[tag], float(_positive(tag, segment)))
                # The quoted exact clause remains reviewable outside the model.
                evidence.append({'run_index': index, 'text': clause})
            if len(segments) > 1 and re.search(r'\b(?:permission was given|^for )', clause, re.I):
                for tag in ('red_hood_to_post', 'early_to_post', 'earplugs_removed', 'partially_shod'):
                    if _positive(tag, clause):
                        for index, _ in segments:
                            output[index][tag] = 1.
    return output, evidence


def build_observations(canonical, tables):
    """Return runner observations, withdrawals and attribution/coverage counts."""
    c = prepare_canonical(canonical)
    race_ids, horse_ids = c.race_id.to_numpy(), c.horse_id.to_numpy()
    names, clocks, event_times = c.horse_name.to_numpy(), c._local.to_numpy(), c._event.array
    groups = c.groupby(['_day', '_track', '_local'], sort=False).indices
    fixtures = c.groupby(['_day', '_track'], sort=False).indices
    reports = tables.get('reports', pd.DataFrame())
    report_lookup = {str(r.report_key): r for r in reports.itertuples(index=False)}
    documents = {str(r.report_key): r for r in tables.get('documents', pd.DataFrame()).itertuples(index=False)}
    rows, withdrawals = [], []
    audit = {'unmatched_sections': 0, 'ambiguous_sections': 0, 'source_sections': 0}

    def add(report, clock, text, available, source, *, context=None):
        audit['source_sections'] += 1
        day = str(report.race_date)[:10]
        fixture_key = day, canonical_track_key(report.course)
        indices = groups.get((*fixture_key, clock)) if clock else None
        covered = indices is not None and len(set(race_ids[indices])) == 1
        if not covered:
            indices = fixtures.get(fixture_key)
        if indices is None:
            audit['unmatched_sections'] += 1
            return
        # A fixture-wide explicit empty report covers the fixture, but an
        # un-timed incident only identifies horses actually named in it.
        empty = bool(re.fullmatch(r'\s*Nothing to report[.\s]*', text, re.I))
        roster = [(int(i), names[i]) for i in indices]
        if len({horse_key(name) for _, name in roster}) != len(roster):
            audit['ambiguous_sections'] += 1
        attributed, evidence = attribute(text, roster, covered=covered or empty)
        if clock and not covered:
            # Permit a small advertised-time change only with unique horse
            # evidence. Never guess an adjacent hour or assign another field.
            target_minute = int(clock[:2])*60+int(clock[3:])
            attributed = {i: v for i, v in attributed.items()
                          if abs(int(clocks[i][:2])*60+int(clocks[i][3:])-target_minute) <= 15}
        if not attributed:
            audit['unmatched_sections'] += 1
        for i, values in attributed.items():
            for name, tags in GROUPS.items():
                values[name] = max(values[tag] for tag in tags)
            rows.append({'run_index': i, 'race_id': race_ids[i], 'horse_id': horse_ids[i],
                         'event_at': event_times[i], 'available_at': available,
                         'report_key': str(report.report_key), 'source_family': source,
                         'evidence_text': '\n'.join(dict.fromkeys(e['text'] for e in evidence if e['run_index'] == i)),
                         **values, **{k: (context or {}).get(k, np.nan) for k in CONTEXT}})

    for report_number, report in enumerate(report_lookup.values(), 1):
        if report_number % 500 == 0:
            logger.info('BHA report derivation: %d/%d reports', report_number, len(report_lookup))
        if getattr(report, 'presence_status', '') == 'absent_from_latest_day':
            continue
        available = pd.Timestamp(report.available_at)
        if available.tzinfo is None:
            raise ValueError('BHA observation availability must include timezone')
        for clock, text in html_sections(report.enquiry_html):
            if text.strip():
                add(report, clock, text, available, 'report')
        if getattr(report, 'runner_notes_html', ''):
            for clock, text in html_sections(report.runner_notes_html):
                add(report, clock, text, available, 'runner_notes')
    for note in tables.get('race_notes', pd.DataFrame()).itertuples(index=False):
        report = report_lookup.get(str(note.report_key))
        if report is None or getattr(report, 'presence_status', '') == 'absent_from_latest_day':
            continue
        doc = documents.get(str(note.report_key))
        going = getattr(doc, 'going_text', '') if doc else ''
        readings = {float(n) for n in re.findall(r'(?:GSt|GoingStick)\s*[:=]?\s*(\d+(?:\.\d+)?)', going, re.I)}
        context = {'goingstick': next(iter(readings)) if len(readings) == 1 and 0 < next(iter(readings)) <= 20 else np.nan,
                   'rail_change_reported': float(bool(re.search(r'\b(?:moved|movement|outwards?|inwards?|extra|additional)\b', getattr(doc, 'rail_text', ''), re.I)))}
        add(report, str(note.off_time_local), str(note.text), pd.Timestamp(note.available_at), 'pdf', context=context)

    # Non-runners need their own event stream: they are absent from results.
    known = c.groupby('_name').horse_id.agg(lambda x: tuple(set(x))).to_dict()
    for key, doc in documents.items():
        report = report_lookup.get(key)
        if report is None:
            continue
        text = str(getattr(doc, 'non_runners_text', ''))
        pieces = list(re.finditer(r'(\d{1,2}:\d{2})\s*(am|pm)\s+', text, re.I))
        for j, match in enumerate(pieces):
            block = text[match.end():pieces[j+1].start() if j+1 < len(pieces) else len(text)]
            split = re.split(r',?\s+trained\s+by\b', block, maxsplit=1, flags=re.I)
            if len(split) != 2:
                continue
            ids = known.get(horse_key(split[0]), ())
            if len(ids) != 1:
                continue
            local = pd.Timestamp(str(report.race_date)[:10]+' '+_clock(match)).tz_localize('Europe/London').tz_convert('UTC')
            withdrawals.append({'horse_id': ids[0], 'event_at': local, 'available_at': pd.Timestamp(doc.available_at),
                'report_key': key, 'evidence_text': block,
                'health': float(bool(re.search(r'\b(?:infect|lame|injur|sore|cough|temperature|illness|blood|abscess|swollen)\w*', split[1], re.I))),
                'going': float(bool(re.search(r'(?m)^Going\b|unsuitable (?:going|ground)', split[1], re.I)))})
    observations = pd.DataFrame(rows)
    if not observations.empty:
        # Combine sections from the same captured report, but retain later
        # document observations separately for availability-aware lookup.
        keys = ['run_index', 'race_id', 'horse_id', 'event_at', 'available_at', 'source_family', 'report_key']
        aggregate = {k: 'max' for k in (*METRICS, *CONTEXT)}
        aggregate['evidence_text'] = lambda values: '\n'.join(dict.fromkeys(v for v in values if v))
        duplicates = observations.duplicated(keys, keep=False)
        if duplicates.any():
            merged = observations.loc[duplicates].groupby(keys, as_index=False, sort=False).agg(aggregate)
            observations = pd.concat([observations.loc[~duplicates], merged], ignore_index=True)
        observations[list(METRICS)+list(CONTEXT)] = observations[list(METRICS)+list(CONTEXT)].astype(np.float32)
    audit.update(canonical_runs=len(c), runner_observations=len(observations), withdrawals=len(withdrawals))
    return c, observations, pd.DataFrame(withdrawals), audit
