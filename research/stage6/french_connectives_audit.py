"""Reconcile the public French connective completion exports and their stimulus scope."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import file_digest
import json
from pathlib import Path
from statistics import correlation, mean
from xml.etree import ElementTree
from zipfile import ZipFile


SOURCES = {
    'data_noNA.csv': '2b05358de36e690d2c1858d831486ae7af96dcfdf0c25d36c314755b18ea4287',
    'data_noNA_large.csv': '7c7addcbbc4cba12540103a29c6d3280e68826e4d471d4ed61316879dd8e3c74',
    'Script_osf.R': '03e3bc904f152414ea7f3c4244c18a4f7149389af428527bb954b521acc685c1',
    'Cloze_task_osf.docx': '32b7b4b9bbdfc5a7ebf153c1d3ae7a3b5f5e6cdf7ab79836456b3bd922f90493',
}
SESSIONS = ('session_01', 'session_02', 'session_03')
MISSING = {'', 'NA'}


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def reconcile(long, wide):
    indexed = {}
    for row in long:
        key = (row['group'], row['order'], row['item'], row['session'])
        if (key in indexed or row['session'] not in SESSIONS
                or row['correctness'] not in {'0', '1'}):
            raise ValueError('Repeated or invalid long-format observation')
        indexed[key] = row
    counts = Counter(long_trials=len(indexed), wide_rows=len(wide))
    visited = set()
    for record in wide:
        for number, session in enumerate(SESSIONS, 1):
            key = (record['group'], record['order'], record['item'], session)
            row = indexed[key]
            choice = record[f'valeur_part_{number:02d}']
            score = record[f'correctness_part_{number:02d}']
            target = record['type_de_connecteur']
            if (key in visited or target != row['type_de_connecteur'] or score != row['correctness']
                    or (choice == target) != (score == '1')):
                raise ValueError('Wide choice or correctness disagrees with its long-format record')
            visited.add(key)
            counts['matched_trials'] += 1
            counts['missing_choices_scored_incorrect' if choice in MISSING else 'known_choices'] += 1
    counts['long_only_trials'] = len(indexed) - len(visited)
    return dict(counts)


def summarize_group(rows):
    by_reader = defaultdict(lambda: defaultdict(list))
    assessments = defaultdict(set)
    connective_session = defaultdict(list)
    ambiguous_ids = defaultdict(set)
    for row in rows:
        reader = row['order']
        session = row['session']
        outcome = int(row['correctness'])
        by_reader[reader][session].append(outcome)
        connective_session[row['type_de_connecteur'], session].append(outcome)
        ambiguous_ids[row['id']].add(reader)
        if row['art_score'] not in MISSING:
            assessments[reader].add(int(row['art_score']))
    if any(len(values) != 1 for values in assessments.values()):
        raise ValueError('A reader has more than one author-recognition score')
    complete = [sessions for sessions in by_reader.values()
                if all(len(sessions.get(session, ())) == 40 for session in SESSIONS)]
    associations = []
    for reader, values in sorted(assessments.items()):
        trials = by_reader[reader].get('session_02', ())
        if len(trials) == 40:
            associations.append((next(iter(values)), mean(trials)))
    scores, accuracies = zip(*associations, strict=True)
    return {
        'reader_order_keys': len(by_reader),
        'published_id_values_reused_by_multiple_order_keys': sum(len(keys) > 1 for keys in ambiguous_ids.values()),
        'readers_without_author_recognition_score': len(by_reader.keys() - assessments.keys()),
        'all_40_items_observed_at_all_three_sessions': len(complete),
        'trials': len(rows),
        'complete_reader_mean_accuracy': {session: mean(mean(reader[session]) for reader in complete)
                                          for session in SESSIONS},
        'complete_reader_mean_paired_accuracy_delta': {
            f'{session}_minus_session_01': mean(mean(reader[session]) - mean(reader['session_01'])
                                               for reader in complete)
            for session in SESSIONS[1:]
        },
        'session_02_reader_accuracy_vs_author_recognition': {
            'readers': len(scores),
            'pearson_r': correlation(scores, accuracies),
        },
        'connective_by_session': {
            connective: {session: {'trials': len(answers), 'correct': sum(answers), 'accuracy': mean(answers)}
                         for session in SESSIONS if (answers := connective_session[connective, session])}
            for connective in sorted({name for name, _ in connective_session})
        },
    }


def count_stimuli(path):
    namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with ZipFile(path) as document:
        root = ElementTree.fromstring(document.read('word/document.xml'))
    # Each stimulus occupies a numbered paragraph with an underscore gap between delimiters.
    paragraphs = (''.join(run.text or '' for run in row.findall('.//w:t', namespace))
                  for row in root.findall('.//w:p', namespace))
    return sum('//' in text and '_' in text for text in paragraphs)


def audit(root):
    for filename, expected in SOURCES.items():
        with (root / filename).open('rb') as stream:
            if file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'Unexpected source version: {filename}')
    long = csv_rows(root / 'data_noNA.csv')
    wide = csv_rows(root / 'data_noNA_large.csv')
    groups = sorted({row['group'] for row in long})
    return {
        'source_sha256': SOURCES,
        'stimulus_gap_paragraphs': count_stimuli(root / 'Cloze_task_osf.docx'),
        'distinct_item_numbers_across_forms': len({row['item'] for row in long}),
        'reconciliation': reconcile(long, wide),
        'groups': {group: summarize_group([row for row in long if row['group'] == group]) for group in groups},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.raw), indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
