"""Audit paired narrative question lifetimes without exporting or modeling story text."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import posixpath
import re
from statistics import median
from xml.etree import ElementTree
from zipfile import ZipFile


SOURCE = 'https://osf.io/m524v/'
PAPER = 'https://doi.org/10.1017/langcog.2025.10057'
GUIDELINES = 'https://osf.io/download/h9mxg/'
SHEET_NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
RELATION = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
WORKBOOKS = {
    'Katze': ('AA',),  # The README identifies one consolidated question list, copied for analysis.
    'Auftrag': ('AA', 'JE'),
    'Thomas': ('AA', 'JE'),
}
COLUMN = re.compile(r'([A-Z]+)[1-9][0-9]*\Z')
PREPARED_COLUMN = re.compile(r'(AA|JE)_([1-9][0-9]*)\Z')
AGE_CUTS = (4, 16, 64)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def public_source_manifest(folder):
    names = ('annotation-directory.json', 'annotation-directory-page2.json', 'rating-directory.json')
    entries = [entry for name in names for entry in json.loads((folder / name).read_text())['data']]
    manifests = {}
    for entry in entries:
        attrs = entry['attributes']
        manifests[attrs['name']] = {
            'download': entry['links']['download'],
            'sha256': attrs['extra']['hashes']['sha256'],
        }
    return manifests


def read_sheet(path):
    with ZipFile(path) as book:
        workbook = ElementTree.fromstring(book.read('xl/workbook.xml'))
        [sheet] = [node for node in workbook.findall('s:sheets/s:sheet', SHEET_NS)
                   if node.attrib['name'] == 'Textannotation']
        relationships = ElementTree.fromstring(book.read('xl/_rels/workbook.xml.rels'))
        [relative] = [rel.attrib['Target'] for rel in relationships if rel.attrib['Id'] == sheet.attrib[RELATION]]
        file_name = posixpath.join('xl', relative) if not relative.startswith('/') else relative.lstrip('/')
        strings = ElementTree.fromstring(book.read('xl/sharedStrings.xml'))
        shared = [''.join(node.text or '' for node in cell.findall('.//s:t', SHEET_NS)) for cell in strings]
        root = ElementTree.fromstring(book.read(file_name))
    rows = {}
    for row in root.findall('s:sheetData/s:row', SHEET_NS):
        values = {}
        for cell in row.findall('s:c', SHEET_NS):
            value = cell.find('s:v', SHEET_NS)
            if value is None or value.text is None:
                continue
            column = COLUMN.fullmatch(cell.attrib['r']).group(1)
            raw = shared[int(value.text)] if cell.attrib.get('t') == 's' else value.text
            values[column] = raw.strip()
        rows[int(row.attrib['r'])] = values
    return rows


def workbook_tracks(path):
    rows = read_sheet(path)
    questions = {column: int(question_id) for column, question_id in rows[1].items()
                 if column != 'A' and question_id.isdecimal()}
    if (sorted(questions.values()) != list(range(1, len(questions) + 1)) or not questions or
            any(not rows[2].get(column) for column in questions)):
        raise ValueError('Expected one labeled consecutive question per annotation column')
    text_lines = [row for number, row in rows.items()
                  if number > 2 and row.get('A') and row['A'] != 'Wichtigkeit (Wert von 1-10)']
    states = {'': 0, 'offen': 1, '?offen': 2}
    return {
        question_id: tuple(states[row.get(column, '')] for row in text_lines)
        for column, question_id in questions.items()
    }


def prepared_tracks(path):
    """Deduplicate the author's expanded indices, preserving the original question lists."""
    by_index = {}
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        parsed_columns = [(name, PREPARED_COLUMN.fullmatch(name)) for name in reader.fieldnames]
        keys = [(name, match[1], int(match[2])) for name, match in parsed_columns if match]
        if not keys:
            raise ValueError('No AA/JE annotation columns in the public prepared CSV')
        for row in reader:
            index = int(row['row_index'])
            states = tuple({'0': 0, '1': 1, '?offen': 2}[row[name]] for name, _, _ in keys)
            if by_index.setdefault(index, states) != states:
                raise ValueError('Prepared annotation columns disagree across repeated participant rows')
    if sorted(by_index) != list(range(1, len(by_index) + 1)):
        raise ValueError('Expected a full contiguous grid of expanded author indices')
    teams = defaultdict(dict)
    for position, (_, team, question_id) in enumerate(keys):
        teams[team][question_id] = tuple(by_index[index][position] for index in sorted(by_index))
    return teams


def question_bounds(states):
    compressed = tuple(value for value, _ in groupby(states))
    if not compressed or compressed[0] not in (0, 1) or compressed not in {
        (0, 1), (0, 1, 0), (0, 1, 2), (0, 1, 2, 0), (1,), (1, 0), (1, 2), (1, 2, 0),
    }:
        raise ValueError(f'Unexpected open/uncertain/closed sequence: {compressed}')
    onset = states.index(1) + 1
    uncertain = states.index(2) + 1 if 2 in states else None
    try:
        closure = states.index(0, onset) + 1
    except ValueError:
        closure = None
    return onset, uncertain, closure


def summarize(tracks, unit):
    [length] = set(map(len, tracks.values()))
    span_by_question = {question: question_bounds(state) for question, state in sorted(tracks.items())}
    openings = Counter(begin for begin, _, _ in span_by_question.values())
    closures = Counter(close for _, _, close in span_by_question.values() if close is not None)
    open_after = [sum(begin <= point and (close is None or point < close)
                      for begin, _, close in span_by_question.values()) for point in range(1, length + 1)]
    rankable = contradicted = concurrent = 0
    for point, count in closures.items():
        concurrent += count > 1
        finished = [start for start, _, close in span_by_question.values() if close == point]
        still_open = [start for start, _, close in span_by_question.values()
                      if start < point and (close is None or close > point)]
        if still_open:
            rankable += 1
            # No choice of order among simultaneous openings can satisfy LIFO in this case.
            contradicted += max(still_open) > min(finished)
    observed_durations = [close - start for start, _, close in span_by_question.values() if close is not None]
    return {
        'source_unit': unit, 'positions': length, 'questions': len(span_by_question),
        'closure_observed': len(observed_durations), 'censored_at_last_position': len(span_by_question) - len(observed_durations),
        'questions_with_an_uncertain_span': sum(stop is not None for _, stop, _ in span_by_question.values()),
        'distinct_opening_positions': len(openings), 'distinct_closing_positions': len(closures),
        'closing_positions_with_more_than_one_closure': concurrent,
        'closing_positions_with_an_older_or_newer_alternative_still_open': rankable,
        'closing_positions_inconsistent_with_strict_lifo_even_after_resolving_ties': contradicted,
        'max_simultaneously_open_after_position': max(open_after),
        'duration_from_open_to_first_confirmed_closing_median_units': median(observed_durations),
    }


def opportunities(annotation_lists):
    """Keep each question at risk from the line after opening until its close or censor."""
    for tracks in annotation_lists:
        for states in tracks.values():
            opening, _, closing = question_bounds(states)
            last = len(states) if closing is None else closing
            for point in range(opening + 1, last + 1):
                elapsed_lines = point - opening
                yield sum(elapsed_lines > cut for cut in AGE_CUTS), point == closing


def score(rows, probability_by_age):
    ties = defaultdict(Counter)
    for age, close in rows:
        ties[probability_by_age[age]][close] += 1
    closes = sum(count[True] for count in ties.values())
    if closes == 0:
        raise ValueError('A held-out annotation group needs an observed closure')
    seen = hits = 0
    nll, brier, precision = [], [], []
    for p, count in sorted(ties.items(), reverse=True):
        positive, negative = count[True], count[False]
        seen += positive + negative
        hits += positive
        nll.append(-positive * math.log(p) - negative * math.log1p(-p))
        brier.append(positive * (1 - p)**2 + negative * p**2)
        precision.append((positive / closes) * (hits / seen))
    return {
        'mean_log_loss_nats': math.fsum(nll) / len(rows),
        'brier': math.fsum(brier) / len(rows),
        'average_precision': math.fsum(precision),
        'sum_expected_closures': math.fsum(p * sum(count.values()) for p, count in ties.items()),
    }


def leave_one_story_out(stories):
    rows = {story: list(opportunities(lists.values())) for story, lists in stories.items()}
    by_story = {}
    for held, test in rows.items():
        train = [row for story, values in rows.items() if story != held for row in values]
        n_bins = len(AGE_CUTS) + 1
        by_age = [Counter() for _ in range(n_bins)]
        for age, outcome in train:
            by_age[age][outcome] += 1
        constant = (sum(v for _, v in train) + 1) / (len(train) + 2)
        hazard = [(values[True] + 1) / (sum(values.values()) + 2) for values in by_age]
        by_story[held] = {
            'held_out_at_risk_question_lines': len(test),
            'held_out_closures': sum(value for _, value in test),
            'training_at_risk_question_lines_by_interval': [sum(values.values()) for values in by_age],
            'training_closure_counts_by_interval': [values[True] for values in by_age],
            'estimated_constant_hazard': constant,
            'estimated_hazard_by_interval': hazard,
            'constant': score(test, [constant] * n_bins),
            'elapsed_line_intervals': score(test, hazard),
        }
    metrics = ('mean_log_loss_nats', 'brier', 'average_precision')
    macro = {method: {metric: math.fsum(item[method][metric] for item in by_story.values()) / len(by_story)
                     for metric in metrics} for method in ('constant', 'elapsed_line_intervals')}
    return {
        'description': 'Closure hazard conditional on an annotated question already being open. '
                       'Opening is fixed to the human annotation. Complete stories are held out.',
        'at_risk_rule': 'Begin the line after opening. Include the first blank after an open or uncertain run, '
                        'or continue through the final text line for censored questions.',
        'training_rule': 'Use a Beta(1,1) posterior mean globally or per fixed elapsed-line interval. '
                         'The two models see no stimulus or question text.',
        'elapsed_line_intervals_inclusive_upper_bounds': [*AGE_CUTS, None],
        'not_pooled': 'Unfall uses expanded author word indices, not original line units.',
        'by_held_out_story': by_story,
        'equally_weighted_story_means': macro,
    }


def run(folder, output):
    manifest = public_source_manifest(folder)
    used = {}

    def source(name):
        path = folder / name
        record = manifest[name]
        if sha256(path) != record['sha256']:
            raise ValueError(f'Downloaded {name} differs from the OSF file listing')
        used[name] = record
        return path

    workbook_annotations = {
        story: {name: workbook_tracks(source(f'PITQ_{story}_{name}_{name}.xlsx')) for name in lists}
        for story, lists in WORKBOOKS.items()
    }
    results = {story: {name: summarize(tracks, 'annotation_line') for name, tracks in lists.items()}
               for story, lists in workbook_annotations.items()}
    results['Unfall'] = {name: summarize(tracks, 'author_expanded_word_index')
                         for name, tracks in prepared_tracks(source('Unfall_suspense_data.csv')).items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'project': SOURCE,
        'paper': PAPER,
        'guidelines': GUIDELINES,
        'license_as_listed_by_osf': 'CC BY-NC-SA 4.0 for depositor-licensed material',
        'endpoint': 'Trained teams entered an opening line while reading under annotation instructions and '
                    'selected the closing line on a second reading. The workbooks record no live timestamps '
                    'or unaided reader reports.',
        'unit_limits': 'Unfall states are the authors’ deterministic expansion of line states onto their word index, '
                       'which cannot identify the within-line moment of a semantic transition.',
        'scope': 'Only numerical summaries of the released annotation columns. '
                 'No stimulus or question text is modeled or exported.',
        'inputs': used,
        'stories': results,
        'leave_one_story_out_closure_baseline': leave_one_story_out(workbook_annotations),
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-folder', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.source_folder, args.output)
