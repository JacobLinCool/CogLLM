"""Audit exact alignment of published MedReadMe jargon spans to source text."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import accumulate
import json
from pathlib import Path

REVISION = '7c52205d18f16b80e3ffccb4e9da88b1942df310'
FILES = {
    'readability.csv': '54082bb202d8916f296b37c9b304fd9f2c102920587ea6c346ea626be3d44ab1',
    'jargon.json': '426e43f8fa5b9d4fbba2c7b3fabad93f23d0ea7ccd4a45fb906e23f927ed1e1d',
}
SPLITS = {'train': 2587, 'dev': 784, 'test': 1140}
CATEGORIES = frozenset({'general-complex', 'general-medical-multisense', 'medical-jargon-google-easy',
                        'medical-jargon-google-hard', 'medical-name-entity', 'abbr-medical', 'abbr-general'})


def compact(value):
    return ''.join(value.split())


def source_spans(text, tokens, entities):
    positions = [index for index, character in enumerate(text) if not character.isspace()]
    pieces = [compact(token) for token in tokens]
    if ''.join(text[index] for index in positions) != ''.join(pieces):
        raise ValueError('Published tokens disagree with the source even after whitespace removal')
    boundaries = list(accumulate(map(len, pieces), initial=0))
    for start, end, category, terms in entities:
        start = positions[boundaries[start]]
        end = positions[boundaries[end] - 1] + 1
        if compact(text[start:end]) != compact(''.join(terms)):
            raise ValueError('Source character span differs from the annotated token span')
        yield start, end, category


def word_coverage(spans, word_offsets):
    starts = [start for start, _ in word_offsets]
    for start, end, category in spans:
        index = bisect_left(starts, start)
        previous_overlaps = index > 0 and word_offsets[index - 1][1] > start
        stop = bisect_left(starts, end)
        if previous_overlaps or (stop > index and word_offsets[stop - 1][1] > end):
            yield category, 'partial_word_boundary'
        elif index == stop:
            yield category, 'no_app_word'
        else:
            yield category, 'one_full_app_word' if stop - index == 1 else 'multiple_full_app_words'


def run(raw, output):
    from research.semantic_tracking.iirc_anchor_fit import WORD
    from research.stage3.run import save_json

    for name, expected in FILES.items():
        with (raw / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'Expected the pinned MedReadMe file: {name}')
    with (raw / 'readability.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    texts = defaultdict(list)
    for row in rows:
        text = row['Sentence']
        texts[(row['Source'], row['Side'], row['split'], compact(text))].append(text)
    labels = json.loads((raw / 'jargon.json').read_bytes())
    label_splits = Counter(row['split'] for row in labels)
    if label_splits != SPLITS:
        raise ValueError('Label split inventory changed')
    keys = [(*row['belongings'], row['split'], compact(''.join(row['tokens']))) for row in labels]
    key_counts = Counter(keys)
    counts = {split: Counter() for split in SPLITS}
    test_sources = Counter()
    all_categories = Counter()
    test_coverage = defaultdict(Counter)
    for key, row in zip(keys, labels, strict=True):
        source, _, split, _ = key
        tokens, entities = row['tokens'], row['entities']
        if not all(compact(token) for token in tokens):
            raise ValueError('Published tokens must contain at least one visible character')
        for begin, end, category, terms in entities:
            if not (0 <= begin < end <= len(tokens) and tokens[begin:end] == terms and category in CATEGORIES):
                raise ValueError('Invalid published jargon coordinates or categories')
            all_categories[category] += 1
        matches = texts.get(key, ())
        counts[split]['sentences'] += 1
        counts[split]['spans'] += len(entities)
        if len(matches) != 1 or key_counts[key] != 1:
            counts[split]['unmatched' if not matches else 'ambiguous'] += 1
            continue
        text, = matches
        spans = list(source_spans(text, tokens, entities))
        counts[split]['exact_unique_sentences'] += 1
        counts[split]['exact_spans'] += len(spans)
        if split == 'test':
            test_sources[source] += 1
            words = [match.span() for match in WORD.finditer(text)]
            for category, coverage in word_coverage(spans, words):
                test_coverage[category][coverage] += 1
    save_json(output, {
        'repository_revision': REVISION, 'files_sha256': FILES,
        'matching': 'unique source, side, split and original text after removing only whitespace',
        'split_counts': counts, 'all_annotations_by_category': all_categories,
        'aligned_test_sentences_by_source': test_sources, 'aligned_test_span_coverage_by_app_word_boundaries': test_coverage,
        'scope': 'Author span annotations, individually published sentences, no reader-by-reader response or article keys.',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
