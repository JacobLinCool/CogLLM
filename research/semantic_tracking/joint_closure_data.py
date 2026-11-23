"""Export metadata-only, interval-censored supervision from narrative annotations."""

import argparse
from collections import Counter
import json
from pathlib import Path

from research.semantic_tracking.resolution_prepare import (
    GUIDELINES,
    PAPER,
    SOURCE,
    WORKBOOKS,
    prepared_tracks,
    public_source_manifest,
    question_bounds,
    sha256,
    summarize,
    workbook_tracks,
)


def interval(story, team, number, states):
    opening, uncertain, upper = question_bounds(states)
    last = len(states)
    if uncertain is not None:
        lower = uncertain - 1
    elif upper is not None:
        lower = upper - 1
    else:
        lower = last
    observed_through = upper if upper is not None else lower
    if not 1 <= opening <= lower <= observed_through <= last or upper is not None and upper <= lower:
        raise ValueError(f'Invalid observed line bounds for {story}/{team}/{number}')
    return {
        'id': f'onea-{story}-{team}-{number:03d}',
        'story': story,
        'team': team,
        'open_line': opening,
        'lower_line': lower,
        'upper_line': upper,
        'last_line': last,
        'line_opportunities': observed_through - opening,
    }


def counts(intervals):
    finite = [row for row in intervals if row['upper_line'] is not None]
    right_censored = [row for row in intervals if row['upper_line'] is None]
    widths = Counter(row['upper_line'] - row['lower_line'] for row in finite)
    return {
        'questions': len(intervals),
        'bounded_closures': len(finite),
        'right_censored': len(right_censored),
        'censored_before_story_end_due_to_uncertainty': sum(row['lower_line'] < row['last_line'] for row in right_censored),
        'bounded_interval_width_counts': dict(sorted(widths.items())),
        'zero_line_opportunities': sum(row['line_opportunities'] == 0 for row in intervals),
        'line_opportunities': sum(row['line_opportunities'] for row in intervals),
        'confirmed_open_lines_after_opening': sum(row['lower_line'] - row['open_line'] for row in intervals),
    }


def run(source_folder, output):
    manifest = public_source_manifest(source_folder)
    used = {}

    def verified_source(name):
        path = source_folder / name
        record = manifest[name]
        if sha256(path) != record['sha256']:
            raise ValueError(f'Local {name} differs from its public OSF file hash')
        used[name] = record
        return path

    data = {}
    for story, teams in sorted(WORKBOOKS.items()):
        data[story] = []
        for team in sorted(teams):
            tracks = workbook_tracks(verified_source(f'PITQ_{story}_{team}_{team}.xlsx'))
            data[story].extend(interval(story, team, number, states) for number, states in sorted(tracks.items()))
        if len({row['last_line'] for row in data[story]}) != 1:
            raise ValueError(f'Teams use different annotation-line inventories in {story}')
    rows = [row for questions in data.values() for row in questions]
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Annotation question IDs must be unique across source lists')
    diagnostic = {
        team: summarize(tracks, 'author_expanded_word_index')
        for team, tracks in prepared_tracks(verified_source('Unfall_suspense_data.csv')).items()
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'intervals.jsonl'
    text = ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows)
    path.write_text(text, encoding='utf-8')
    result = {
        'paper': PAPER,
        'project': SOURCE,
        'guidelines': GUIDELINES,
        'source_access': 'Public OSF release. It lists CC BY-NC-SA 4.0 for depositor-licensed material.',
        'supervision': 'Numeric metadata of human-annotated question states only. The annotation line is the measurement unit.',
        'measurement': 'Trained teams entered the opening under guided annotation and selected the answer line on a second reading.',
        'interval_rule': 'One-based working-sheet text lines. An observed closure lies in (lower_line, upper_line]. '
                         'Lower is the last confirmed open line. Upper is the first blank after the open or uncertain run. '
                         'If the annotation has no following blank, censor at the last confirmed open line.',
        'line_opportunities_rule': 'Count lines after the opening through the observed upper bound or censor line. '
                                  'For a bounded multi-line interval, these include uncertain lines. '
                                  'Such opportunities are not independent binary labels for the closure event.',
        'fields': list(rows[0]),
        'line_annotation_totals': counts(rows),
        'stories': {story: counts(questions) for story, questions in data.items()},
        'folds': {held: {'training_stories': sorted(data.keys() - {held}), 'held_out_stories': [held]} for held in data},
        'unfall_diagnostic': {
            'reason_not_pooled': 'The released table expands original line states across author word indices. '
                                 'Those positions do not identify the original within-line event.',
            'teams': diagnostic,
        },
        'input_files': used,
        'intervals_sha256': sha256(path),
    }
    (output / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-folder', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(**vars(parser.parse_args()))
