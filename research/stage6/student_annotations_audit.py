"""Audit which published student ratings can test a new reader-background readout."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import NamedTuple

from .textcomplexityde_transfer import SOURCE_HASHES, normalized_tokens, unique_rows
from research.semantic_tracking.iirc_prepare import digest


REVISION = 'db164a0365d1398473ec7c8be4a211e03c7b9286'
STUDENT_SHA256 = '8ca024c962f11e93b09b4b954f3951ef310d2fe32cea87c311996845e891b800'
LANGUAGES = ('german', 'german_and_another_language', 'another_language')
FOLDS = 4


class GuideTrial(NamedTuple):
    reader: str
    text: str
    text_key: tuple[str, ...]
    language: str
    grade: int


def shingles(words):
    return set(zip(words, words[1:], words[2:], words[3:], words[4:]))


def partition(namespace, key):
    payload = f'cogllm-thome-reader-text-v1/{namespace}/{key}'.encode()
    return int.from_bytes(sha256(payload).digest()[:8], 'big') % FOLDS


def select_guide(student, textde):
    sources = {'source.csv': 'Sentence ID', 'ratings.csv': 'ID'}
    if digest(student) != STUDENT_SHA256 or any(digest(textde / name) != SOURCE_HASHES[name] for name in sources):
        raise ValueError('Inputs differ from their pinned author releases')
    prior = {normalized_tokens(row['Sentence']) for name, key in sources.items()
             for row in unique_rows(textde / name, key).values()}
    by_shingle = defaultdict(set)
    for old_text in prior:
        for shingle in shingles(old_text):
            by_shingle[shingle].add(old_text)
    with student.open(newline='', encoding='utf-8-sig') as source:
        rows = list(csv.DictReader(source))
    text_keys = {row['text']: normalized_tokens(row['text']) for row in rows}
    if len(rows) != 3954 or len(text_keys) != 1697 or not all(text_keys.values()):
        raise ValueError('Unexpected author annotation inventory')
    exact = prior & set(text_keys.values())
    near = set()
    for text in set(text_keys.values()) - exact:
        grams = shingles(text)
        overlap = Counter(other for gram in grams for other in by_shingle[gram])
        if any(count >= 0.8 * min(len(grams), len(shingles(other))) for other, count in overlap.items()):
            near.add(text)
    background, seen, guide = {}, set(), []
    for row in rows:
        reader, text, language = (row[field] for field in ('user_id', 'text', 'language'))
        key = text_keys[text]
        if key in exact or key in near:
            continue
        group = (language, row['self_estimation_german'])
        if ((reader, text) in seen or language not in LANGUAGES or group[1] not in {'1', '2', '3', '4'}
                or row['label'] not in {'1', '2', '3', '4', '5'} or background.setdefault(reader, group) != group):
            raise ValueError('Unmatched or duplicate reader response')
        seen.add((reader, text))
        guide.append(GuideTrial(reader, text, key, language, int(row['label'])))
    if (len(guide), len({row.text for row in guide}), len(background)) != (3160, 1143, 192):
        raise ValueError('The intersection no longer isolates the published study-guide materials')
    return guide, len(exact), len(near)


def summarize(guide, exact, near):
    background, sentence_languages, text_keys = {}, defaultdict(set), {}
    trials, grid = Counter(), Counter()
    for reader, text, key, language, _ in guide:
        background[reader] = language
        text_keys[text] = key
        sentence_languages[text].add(language)
        trials[language] += 1
        grid[partition('reader', reader), partition('sentence', ' '.join(key))] += 1
    return {
        'author_release': {'revision': REVISION, 'annotations_sha256': STUDENT_SHA256},
        'comparison_source_sha256': {name: SOURCE_HASHES[name] for name in ('source.csv', 'ratings.csv')},
        'same_text_rule': 'Unicode NFKC, casefold, letter/number words; at least 80% 5-gram containment screened',
        'exact_overlap_normalized_sentences': exact,
        'additional_near_overlap_normalized_sentences': near,
        'study_guide_observations': len(guide), 'study_guide_sentences': len(sentence_languages),
        'study_guide_normalized_sentences': len({text_keys[text] for text in sentence_languages}),
        'readers': len(background), 'pre_reading_language_readers': dict(Counter(background.values())),
        'pre_reading_language_trials': dict(trials),
        'shared_german_only_vs_no_german_sentences': sum({LANGUAGES[0], LANGUAGES[2]} <= groups
                                                       for groups in sentence_languages.values()),
        'shared_all_three_profiles_sentences': sum(set(LANGUAGES) <= groups for groups in sentence_languages.values()),
        'crossed_reader_sentence_fold_counts': [[grid[i, j] for j in range(FOLDS)] for i in range(FOLDS)],
    }


def audit(student, textde):
    return summarize(*select_guide(student, textde))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('student', 'textde', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.student, args.textde)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
