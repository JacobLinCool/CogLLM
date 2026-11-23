"""Separate released ExpMRC RACE+ questions from retrospective evidence spans."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata


COMMIT = '9827fed6466f295252f9054c56ec7b233d46e636'
SOURCE_URL = f'https://raw.githubusercontent.com/ymcui/expmrc/{COMMIT}/data/race/expmrc-race-dev.json'
SOURCE_SHA256 = '43819fee2096d40f8a209bd5bce82a47b663cb79804119c036de4655bd01f021'
LICENSE = 'CC-BY-SA-4.0'
ARTICLE_ID = re.compile(r'[0-9a-f]{8}')
WORD = re.compile(r'\w+')


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evidence_span(source, evidence):
    start = source.find(evidence)
    if start < 0 or not evidence or source.find(evidence, start + 1) != -1:
        raise ValueError('Every annotated evidence string must map to one exact source span')
    return [start, start + len(evidence)]


def load_release(source):
    if digest(source) != SOURCE_SHA256:
        raise ValueError('Source bytes differ from the pinned author release')
    payload = json.loads(source.read_bytes())
    if payload['version'] != 'expmrc-race-dev' or len(payload['data']) != 167:
        raise ValueError('Unexpected release version or article inventory')

    articles, queries, outcomes = [], [], []
    identifiers, texts = set(), set()
    for document in payload['data']:
        article_id, body = document['id'], document['article']
        if not ARTICLE_ID.fullmatch(article_id) or article_id in identifiers or not body.strip() or body in texts:
            raise ValueError(f'Missing or duplicate article identity for {article_id}')
        identifiers.add(article_id)
        texts.add(body)
        articles.append({'id': article_id, 'text': body})
        question_texts = set()
        pairs = zip(document['questions'], document['evidences'], strict=True)
        for ordinal, (question, evidence) in enumerate(pairs, 1):
            question_id = f'{article_id}:{ordinal}'
            normalized_question = ' '.join(unicodedata.normalize('NFC', question).split())
            if not normalized_question or normalized_question in question_texts or len(evidence) != 2:
                raise ValueError(f'Invalid question or reference records for {question_id}')
            question_texts.add(normalized_question)
            queries.append({'id': question_id, 'article_id': article_id, 'text': question})
            outcomes.append({'id': question_id, 'article_id': article_id,
                             'refs': [evidence_span(body, span) for span in evidence]})
    if len(queries) != 561:
        raise ValueError('Question inventory differs from the pinned public evaluation split')
    return articles, queries, outcomes


def annotation_inventory(outcomes):
    counts = Counter()
    grouped = defaultdict(list)
    for outcome in outcomes:
        first, second = outcome['refs']
        counts['identical_spans'] += first == second
        counts['identical_ends'] += first[1] == second[1]
        counts['overlapping_spans'] += max(first[0], second[0]) < min(first[1], second[1])
        bounds = (min(first[0], second[0]), max(first[1], second[1]))
        grouped[outcome['article_id']].append(bounds)
    counts['disjoint_spans'] = len(outcomes) - counts['overlapping_spans']
    for ranges in grouped.values():
        pairs = list(combinations(ranges, 2))
        counts['same_article_question_pairs'] += len(pairs)
        disjoint = sum(first[1] <= second[0] or second[1] <= first[0] for first, second in pairs)
        counts['same_article_disjoint_reference_hulls'] += disjoint
        counts['articles_with_disjoint_reference_hulls'] += bool(disjoint)
    return dict(counts)


def normalized(text):
    return unicodedata.normalize('NFKC', text).casefold()


def shingles(text):
    words = WORD.findall(normalized(text))
    return set(zip(words, words[1:], words[2:], words[3:], words[4:]))


def comparison_inventory(articles, texts):
    positions = defaultdict(set)
    sizes = {}
    normalized_bodies = {' '.join(normalized(row['text']).split()) for row in articles}
    for row in articles:
        grams = shingles(row['text'])
        sizes[row['id']] = len(grams)
        for gram in grams:
            positions[gram].add(row['id'])

    exact, near, largest_overlap, largest_count = 0, 0, 0.0, 0
    for source in texts:
        exact += ' '.join(normalized(source).split()) in normalized_bodies
        grams = shingles(source)
        overlaps = Counter(identifier for gram in grams for identifier in positions.get(gram, ()))
        for identifier, count in overlaps.items():
            # A long context that fully contains a candidate also counts as near overlap.
            fraction = count / min(len(grams), sizes[identifier])
            largest_count = max(count, largest_count)
            largest_overlap = max(fraction, largest_overlap) if count >= 20 else largest_overlap
            near += count >= 20 and fraction >= 0.8
    return {
        'reference_distinct_texts': len(texts), 'normalized_exact_reference_matches': exact,
        'near_pairs_sharing_at_least_20_distinct_5_word_shingles_and_80_percent_of_smaller': near,
        'largest_shared_shingle_count': largest_count,
        'largest_overlap_fraction_among_pairs_sharing_at_least_20_shingles': largest_overlap,
    }


def comparison_sources(articles, inputs, race_tsv):
    comparisons = {}
    for path in inputs:
        with path.open(encoding='utf-8') as stream:
            texts = {json.loads(line)['text'] for line in stream}
        comparisons[path.as_posix()] = {'sha256': digest(path), **comparison_inventory(articles, texts)}
    if race_tsv is not None:
        with race_tsv.open(encoding='utf-8', newline='') as stream:
            texts = {row['paragraph'] for row in csv.DictReader(stream, delimiter='\t') if row['source'] == 'RACE'}
        comparisons[race_tsv.as_posix()] = {'sha256': digest(race_tsv), 'subset': 'source == RACE',
                                            **comparison_inventory(articles, texts)}
    return comparisons


def prepare(source, output, compare, race_tsv):
    if output.exists():
        raise FileExistsError(f'Use a fresh output directory: {output}')
    articles, queries, outcomes = load_release(source)
    comparisons = comparison_sources(articles, compare, race_tsv)
    if any(row['normalized_exact_reference_matches']
           or row['near_pairs_sharing_at_least_20_distinct_5_word_shingles_and_80_percent_of_smaller']
           for row in comparisons.values()):
        raise ValueError('One or more articles overlap the given comparison corpus')

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.expmrc-') as temporary:
        staged = Path(temporary) / 'assay'
        staged.mkdir()
        files = {'articles.jsonl': articles, 'queries.jsonl': queries, 'outcomes.jsonl': outcomes}
        for name, rows in files.items():
            write_jsonl(staged / name, rows)
        assay = {
            'schema_version': 1, 'source_url': SOURCE_URL, 'source_sha256': SOURCE_SHA256, 'source_license': LICENSE,
            'license_reference': 'https://ymcui.com/expmrc/',
            'paper': 'https://doi.org/10.1016/j.heliyon.2022.e09290',
            'source_text_scope': 'The authors used in-house English exam passages for the RACE+ subset.',
            'endpoint': 'Two retrospective minimal supporting spans per question, selected with the answer visible.',
            'offsets': 'Zero-based, half-open Unicode code point indices in unchanged article text.',
            'article_count': len(articles), 'query_count': len(queries), 'reference_count': 2 * len(outcomes),
            'annotation_inventory': annotation_inventory(outcomes),
            'overlap_comparisons': comparisons, 'prepare_sha256': digest(Path(__file__)),
            'file_sha256': {name: digest(staged / name) for name in files},
        }
        (staged / 'assay.json').write_text(
            json.dumps(assay, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
        staged.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path, action='append', default=[])
    parser.add_argument('--compare-race-tsv', type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output, args.compare, args.compare_race_tsv)


if __name__ == '__main__':
    main()
