"""Summarize OneStop comprehension after ordinary first reading and rereading."""

import argparse
import ast
from collections import Counter
import csv
from dataclasses import dataclass
from hashlib import file_digest
from io import TextIOWrapper
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile


SOURCES = {
    'ia_Paragraph_ordinary.csv.zip': '8883478946ee52381e7057683c9e84dc69fcea9054acc34f0c900463a6b546e9',
    'ia_Paragraph_repeated.csv.zip': '2a28d101917f70cac207dc1b6a8956146b50f93abb3f378513050af42bc4eca6',
    'session_summary.csv': '4cb5b66067bd55bd7d297a5092fc47a579e2058b30cb92f891accb2810b33dda',
}
CUTOFF_PROMPTS = {
    1: {'l3_325', 'l2_323', 'l3_327', 'l2_324'},
    2: {'l4_326', 'l5_328'},
}
FIELDS = (
    'participant_id', 'trial_index', 'question_preview', 'practice_trial', 'repeated_reading_trial',
    'article_batch', 'article_id', 'article_index', 'article_title', 'paragraph_id', 'difficulty_level', 'paragraph',
    'onestopqa_question_id', 'selected_answer', 'is_correct', 'answers_order',
)


@dataclass(frozen=True)
class Trial:
    reader: str
    batch: int
    article: int
    article_index: int
    article_title: str
    paragraph: int
    level: str
    text: str
    question: int
    answer: str
    excluded: str | None

    @property
    def material(self):
        return self.batch, self.article, self.paragraph, self.level

    @property
    def correct(self):
        return int(self.answer == 'A')


def decode(fields, repeated):
    (reader, _, preview, practice, repeat, batch, article, position, title, paragraph, level, text,
     question, answer, grade, order) = fields
    if practice == 'True':
        return None
    batch, article, position, paragraph, question = map(int, (batch, article, position, paragraph, question))
    if (preview != 'False' or repeat != str(repeated) or not title.strip() or not text.strip()
            or not 1 <= batch <= 3 or not 1 <= article <= 10 or not 1 <= paragraph <= 7
            or position not in (range(11, 13) if repeated else range(1, 11))
            or level not in {'Adv', 'Ele'} or question not in {0, 1, 2}):
        raise ValueError('Invalid ordinary reading trial metadata')
    excluded = None
    if answer not in {'A', 'B', 'C', 'D'}:
        excluded = 'missing_answer'
    elif grade != str(answer == 'A'):
        raise ValueError('Choice and correctness differ')
    elif (batch, article, paragraph) == (3, 5, 3) and reader in CUTOFF_PROMPTS.get(question, ()):
        excluded = 'cutoff_question'
    elif (batch, article, paragraph, question) == (1, 6, 2, 0) and ast.literal_eval(order)[-1] == 'D':
        excluded = 'cutoff_distractor'
    return Trial(reader, batch, article, position, title, paragraph, level, text, question, answer, excluded)


def trials(path, repeated):
    # The archives store one row per word with the same trial metadata on each row.
    # Stream the CSV and compare these fields for every row before collapsing it.
    member = path.name.removesuffix('.zip')
    records = {}
    with ZipFile(path) as archive, archive.open(member) as raw, TextIOWrapper(raw, encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError(f'Ambiguous columns in {member}')
        cols = [header.index(field) for field in FIELDS]
        current = None
        for row in reader:
            key = row[cols[0]], row[cols[1]]
            fields = tuple(row[column] for column in cols)
            if key == current:
                if fields != previous:
                    raise ValueError('Trial fields differ between adjacent word records')
                continue
            if key in records:
                raise ValueError('Trial word records are not contiguous')
            records[key] = decode(fields, repeated)
            current, previous = key, fields
    return [trial for trial in records.values() if trial is not None]


def pairs(pre, post):
    seen = {}
    for trial in pre:
        key = (trial.reader, *trial.material)
        if key in seen:
            raise ValueError('Two first encounters for the same reader and paragraph version')
        seen[key] = trial
    retained = []
    drops = Counter()
    for after in post:
        before = seen.pop((after.reader, *after.material), None)
        if before is None:
            drops['no_first_reading_trial'] += 1
            continue
        if before.text != after.text:
            drops['different_released_text'] += 1
            continue
        if (before.question == after.question or (before.article_index == 10) != (after.article_index == 11)):
            raise ValueError('Pair violates the assigned question or article-order manipulation')
        if before.excluded or after.excluded:
            drops[before.excluded or after.excluded] += 1
            continue
        retained.append((before, after))
    return retained, dict(drops)


def accuracy(rows):
    valid = [trial.correct for trial in rows if trial.excluded is None]
    return {'trials': len(rows), 'retained': len(valid), 'correct': sum(valid), 'accuracy': mean(valid),
            'excluded': dict(Counter(trial.excluded for trial in rows if trial.excluded))}


def summarize(matched):
    cells = {}
    for before, after in matched:
        for phase, trial in enumerate((before, after)):
            cell = cells.setdefault((*trial.material, trial.question), ([0, 0], [0, 0]))[phase]
            cell[0] += trial.correct
            cell[1] += 1
    common = [cell for cell in cells.values() if all(count > 0 for _, count in cell)]
    return {
        'matched_pairs': len(matched),
        'readers': len({before.reader for before, _ in matched}),
        'articles': len({(before.batch, before.article) for before, _ in matched}),
        'first_reading_accuracy': mean(before.correct for before, _ in matched),
        'repeated_reading_accuracy': mean(after.correct for _, after in matched),
        'paired_accuracy_delta': mean(after.correct - before.correct for before, after in matched),
        'different_questions_within_pair': len(matched),
        'question_by_paragraph_version_cells': len(cells),
        'cells_in_both_phases': len(common),
        'trials_in_common_cells': [sum(cell[phase][1] for cell in common) for phase in range(2)],
        'question_standardized_accuracy_delta': mean(post[0] / post[1] - pre[0] / pre[1] for pre, post in common),
    }


def load_releases(path):
    for name, expected in SOURCES.items():
        with (path / name).open('rb') as stream:
            if file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'Unexpected public source version: {name}')
    first_archive, repeated_archive, sessions_path = (path / name for name in SOURCES)
    first, repeated = trials(first_archive, False), trials(repeated_archive, True)
    with sessions_path.open(encoding='utf-8-sig', newline='') as stream:
        ordinary = {record['participant_id']: int(record['article_batch'])
                    for record in csv.DictReader(stream) if record['question_preview'] == 'False'}
    if (len(ordinary) != 180 or {trial.reader for trial in first} != ordinary.keys()
            or any(ordinary[trial.reader] != trial.batch for trial in (*first, *repeated))):
        raise ValueError('Ordinary file trial readers differ from the published session index')
    return first, repeated


def audit(path):
    first, repeated = load_releases(path)
    matched, dropped = pairs(first, repeated)
    return {
        'source_sha256': SOURCES,
        'raw_first_reading': accuracy(first),
        'raw_repeated_reading': accuracy(repeated),
        'excluded_pairs': dropped,
        'all_article_batches': summarize(matched),
        'per_batch': {batch: summarize([pair for pair in matched if pair[0].batch == batch])
                      for batch in (1, 2, 3)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.raw), indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
