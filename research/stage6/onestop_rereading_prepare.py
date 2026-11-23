"""Separate grade-free OneStop articles from development and held-out outcomes."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.stage2.records import write_jsonl

from . import onestop_rereading_audit as audit


SPLITS = {1: 'development', 2: 'development', 3: 'holdout'}
SEPARATOR = '\n\n'


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def by_instance(trials):
    instances = defaultdict(dict)
    for trial in trials:
        paragraphs = instances[trial.reader, trial.batch, trial.article]
        if trial.paragraph in paragraphs:
            raise ValueError('Repeated paragraph within one article presentation')
        paragraphs[trial.paragraph] = trial
    return instances


def expected_paragraphs(first, repeated):
    catalogue = defaultdict(set)
    for trial in (*first, *repeated):
        catalogue[trial.batch, trial.article].add(trial.paragraph)
    if set(catalogue) != {(batch, article) for batch in SPLITS for article in range(1, 11)}:
        raise ValueError('Published files do not cover all 30 articles')
    for batch in SPLITS:
        articles = [numbers for (article_batch, _), numbers in catalogue.items() if article_batch == batch]
        if sum(map(len, articles)) != 54 or any(numbers != set(range(1, len(numbers) + 1)) for numbers in articles):
            raise ValueError('Published article paragraphs do not form the expected 54-position batch')
    return catalogue


def article_source(paragraphs, expected):
    if paragraphs.keys() != expected:
        return None
    ordered = [paragraphs[number] for number in sorted(expected)]
    titles = {trial.article_title for trial in ordered}
    positions = {trial.article_index for trial in ordered}
    if len(titles) != 1 or len(positions) != 1:
        raise ValueError('An article presentation has inconsistent title or article order')
    return next(iter(titles)), tuple((trial.paragraph, trial.level, trial.text) for trial in ordered)


def article_text(source):
    title, paragraphs = source
    parts = [title]
    end = len(title)
    stops = {}
    for number, _, paragraph in paragraphs:
        end += len(SEPARATOR) + len(paragraph)
        stops[number] = end
        parts.append(paragraph)
    return SEPARATOR.join(parts), stops


def prepare_records(first, repeated):
    beginning, reread = by_instance(first), by_instance(repeated)
    expected = expected_paragraphs(first, repeated)
    complete = {}
    instance_drops = {}
    for key, paragraphs in reread.items():
        numbers = expected[key[1:]]
        first_source = article_source(beginning.get(key, {}), numbers)
        second_source = article_source(paragraphs, numbers)
        if first_source is None or second_source is None:
            instance_drops[key] = 'incomplete_article_instance'
        elif first_source != second_source:
            instance_drops[key] = 'different_released_article'
        else:
            complete[key] = article_text(first_source)

    matched, pair_drops = audit.pairs(first, repeated)
    documents = {}
    known_stops = defaultdict(set)
    outcomes = {split: [] for split in SPLITS.values()}
    documents_by_split = {split: set() for split in outcomes}
    readers_by_split = {split: set() for split in outcomes}
    article_instances_by_split = {split: set() for split in outcomes}
    source_drops = Counter()
    for before, after in matched:
        key = before.reader, before.batch, before.article
        if key not in complete:
            source_drops[instance_drops[key]] += 1
            continue
        text, stops = complete[key]
        case = sha256(text.encode('utf-8')).hexdigest()
        if documents.setdefault(case, text) != text:
            raise ValueError('Two different source documents share the same digest')
        split = SPLITS[before.batch]
        documents_by_split[split].add(case)
        readers_by_split[split].add(before.reader)
        article_instances_by_split[split].add(key)
        article = 10 * (before.batch - 1) + before.article
        stop = stops[before.paragraph]
        known_stops[case].add(stop)
        for exposure, trial in enumerate((before, after)):
            outcomes[split].append({
                'id': case, 'stop': stop, 'article': article,
                'cell': [trial.paragraph, trial.level, trial.question], 'repeat': exposure, 'correct': trial.correct,
            })

    if (documents_by_split['development'] & documents_by_split['holdout']
            or readers_by_split['development'] & readers_by_split['holdout']):
        raise ValueError('A reader or exact article text occurs in both splits')

    stimuli = [{'id': case, 'text': text, 'stops': sorted(known_stops[case])}
               for case, text in sorted(documents.items())]
    counts = {'documents': len(stimuli), 'stops': sum(len(row['stops']) for row in stimuli)}
    for split, rows in outcomes.items():
        rows.sort(key=lambda row: (row['article'], row['cell'], row['id'], row['stop'], row['repeat'], row['correct']))
        counts[split] = {
            'readers': len(readers_by_split[split]),
            'articles': sorted({row['article'] for row in rows}),
            'article_instances': len(article_instances_by_split[split]),
            'documents': len(documents_by_split[split]),
            'distinct_source_paragraph_positions': len({(row['article'], row['cell'][0]) for row in rows}),
            'stops': sum(len(known_stops[case]) for case in documents_by_split[split]),
            'observations': len(rows),
            'pairs': len(rows) // 2,
        }
    excluded = {'article_instances': dict(Counter(instance_drops.values())),
                'paired_paragraphs_at_audit': pair_drops,
                'paired_paragraphs_due_to_article_instance': dict(source_drops)}
    return stimuli, outcomes, counts, excluded


def prepare(raw, output):
    if output.exists():
        raise FileExistsError(output)
    stimuli, outcomes, counts, excluded = prepare_records(*audit.load_releases(raw))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.onestop-rereading-') as temporary:
        stage = Path(temporary) / 'assay'
        stage.mkdir()
        records = {'stimuli.jsonl': stimuli,
                   **{f'{split}.jsonl': rows for split, rows in outcomes.items()}}
        for name, rows in records.items():
            write_jsonl(stage / name, rows)
        receipt = {
            'source_sha256': audit.SOURCES,
            'prepare_sha256': digest(Path(__file__)),
            'audit_sha256': digest(Path(audit.__file__)),
            'file_sha256': {name: digest(stage / name) for name in records},
            'source_representation': 'released title then unchanged released paragraphs joined by two LF characters',
            'counts': counts,
            'exclusions': excluded,
        }
        (stage / 'assay.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n',
                                         encoding='utf-8')
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    prepare(arguments.raw, arguments.output)


if __name__ == '__main__':
    main()
