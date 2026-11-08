"""Retain every BHA report field and extract full documents, without features."""
from io import BytesIO
from datetime import date
import json
import re

import pandas as pd
from bs4 import BeautifulSoup

PARSER_VERSION = 2


def race_notes(blocks):
    """The extracted blocks that open a race's own report.

    A race report starts "Race 1 - 1:25pm." with the race title on the next
    line. Revised off times in the fixture notes have the same start but never
    the full stop ("Race 1 - 12:15pm (+5) now 12:20pm", "(previously 4:22pm)",
    or a bare list of new times). Parser 1 kept them as race notes, so each
    delayed race appeared twice; documents it stored are filtered when read.
    """
    return [block for block in blocks if block['text'].startswith('.')]


def strip_html(value):
    if not value:
        return ''
    return BeautifulSoup(str(value or ''), 'html.parser').get_text(' ', strip=True)


def normalise_report(row):
    def value(*keys):
        return next((row[k] for k in keys if row.get(k) not in (None, '')), '')
    enquiry = str(value('enquiry', 'enquiryComments', 'enquiry_html'))
    notes = value('whytheyranbadly', 'whyTheyRanBadly', 'runnerNotes', 'runner_notes_html',
                  'why_they_ran_badly_html')
    split = re.split(r'</p>\s*~\s*<p[^>]*>', enquiry, maxsplit=1, flags=re.I)
    if len(split) == 2:
        enquiry, notes = split[0] + '</p>', notes or '<p>' + split[1]
    incidents = value('nonRacedayIncidents', 'nonRaceRelatedIncidents', 'non_race_incidents_html')
    comments = value('fixtureComments', 'fixture_comments')
    year = str(value('year')).removesuffix('.0')
    fixture = str(value('fixtureId', 'fixture_id')).removesuffix('.0')
    dated = value('date', 'race_date')
    try:
        day = date.fromisoformat(str(dated)[:10])
    except ValueError:
        day = pd.to_datetime(dated, format='mixed', errors='raise')
    if not year.isdigit() or not fixture.isdigit() or pd.isna(day) or not value('course'):
        raise ValueError('BHA report has invalid fixture identity, date or course')
    if str(day.year) != year:
        raise ValueError('BHA fixture year disagrees with report date')
    return {
        'report_key': f'{year}:{fixture}', 'fixture_id': fixture, 'year': int(year),
        'race_date': day.strftime('%Y-%m-%d'), 'course': str(value('course')),
        'pdf_report_url': str(value('pdfReportUrl', 'pdf_report_url')),
        'abandoned_reason': str(value('abandonedReason', 'abandoned_reason')),
        'fixture_comments': str(comments), 'fixture_comments_text': strip_html(comments),
        'enquiry_html': enquiry,
        'enquiry_text': (str(value('enquiry_text')) if row.get('enquiry_text') and 'fixtureId' not in row
                         and len(split) == 1 else strip_html(enquiry)),
        'runner_notes_html': str(notes), 'runner_notes_text': strip_html(notes),
        'non_race_incidents_html': str(incidents), 'non_race_incidents_text': strip_html(incidents),
        'data_source': str(value('dataSource', 'data_source')),
        'source_record_json': json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False),
    }


def parse_document(body):
    from pypdf import PdfReader
    if not body.startswith(b'%PDF') or len(body) > 10_000_000:
        raise ValueError('Expected a BHA report PDF of at most 10 MB')
    reader = PdfReader(BytesIO(body))
    if reader.is_encrypted or not 0 < len(reader.pages) <= 60:
        raise ValueError('Unsupported encrypted or oversized BHA report')
    pages = []
    for page in reader.pages:
        content = page.get_contents()
        if content and len(content.get_data()) > 20_000_000:
            raise ValueError('BHA PDF page exceeds extraction limit')
        pages.append(page.extract_text() or '')
    text = '\n'.join(pages)
    if len(text.strip()) < 50:
        raise ValueError('BHA PDF has no extractable text; retained for review')
    footers = re.findall(r'(?m)^Published:\s*(.+)$', text)
    course = re.search(r"(?mi)^(.+?)\s+STEWARDS['’]?\s*(?:ROOM\s*)?REPORT", text[:1000])
    dated = re.search(r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+'
                      r'(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})', text[:1000])
    document_date = (pd.to_datetime(' '.join(dated.groups()), format='%d %B %Y').strftime('%Y-%m-%d')
                     if dated else None)
    clean = re.sub(r'(?m)^(?:Page\s+\d+|Published:.*)\s*$', '', text)
    races = list(re.finditer(r'(?mi)^Race\s+(\d+)\s*-\s*(\d{1,2}:\d{2})\s*(am|pm)', clean))
    headings = list(re.finditer(r'(?m)^(Stewards:|Going:|Rail:|Non-Runners\s*$|Suspensions\s*$)', clean))
    sections = {}
    for i, match in enumerate(headings):
        stop = headings[i+1].start() if i+1 < len(headings) else len(clean)
        next_race = next((m.start() for m in races if m.start() > match.start()), len(clean))
        sections[match.group(1).strip().rstrip(':').lower().replace('-', '_')] = clean[match.end():min(stop, next_race)].strip()
    blocks = []
    for i, match in enumerate(races):
        stop = races[i+1].start() if i+1 < len(races) else len(clean)
        suspension = re.search(r'(?m)^Suspensions\s*$', clean[match.start():stop])
        if suspension:
            stop = match.start() + suspension.start()
        hour, minute = map(int, match.group(2).split(':'))
        hour = hour % 12 + (12 if match.group(3).lower() == 'pm' else 0)
        blocks.append({'race_number': int(match.group(1)), 'off_time_local': f'{hour:02d}:{minute:02d}',
                       'text': clean[match.end():stop].strip()})
    # Revised-time lines still bound sections and races, as they did in parser 1.
    notes = race_notes(blocks)
    if blocks and not notes:
        raise ValueError('BHA PDF race headings lack the expected full stop; retained for review')
    return {'parser_version': PARSER_VERSION, 'pages': len(pages), 'text': text,
            'course': course.group(1).strip() if course else None, 'race_date': document_date,
            'sections': sections, 'races': notes, 'printed_publication_labels': footers,
            'replacement_character_count': text.count('\ufffd')}
