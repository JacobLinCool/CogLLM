"""Check IIRC linked-document evidence against the author’s pinned article release."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re
import tarfile

from research.semantic_tracking.iirc_prepare import (
    RELEASE_SHA256,
    RELEASE_URL,
    digest,
    normalized,
    read_release,
    verify_split_separation,
    visible_anchors,
)


SOURCE_URL = 'https://jamesf-incomplete-qa.s3.amazonaws.com/context_articles.tar.gz'
SOURCE_SHA256 = 'd239390a27ed3f4aa868afe126187e6d108990a7bcc2a3efb6cae5de917264c7'
HTML_TAG = re.compile(r'<[^>]*>')
JSON_KEY = re.compile(rb'"(?:\\.|[^"\\])*"')


def family(title):
    return normalized(title.replace('_', ' '))


def fingerprint(text):
    return sha256(text.encode()).hexdigest()


def source_articles(path, required_titles):
    if digest(path) != SOURCE_SHA256:
        raise ValueError('Linked-document release does not match the pinned author archive')
    requested = required_titles | {key.lower() for key in required_titles}
    articles = {}
    count = 0
    # The release has one JSON entry per line. Read only needed titles into memory.
    with tarfile.open(path, 'r|gz') as bundle:
        member = bundle.next()
        if member is None or member.name != 'context_articles.json' or not member.isfile():
            raise ValueError('Unexpected author article archive member')
        with bundle.extractfile(member) as handle:
            for line in handle:
                entry = line.strip().removesuffix(b',')
                if entry in {b'{', b'}'}:
                    continue
                matched = JSON_KEY.match(entry)
                if not matched:
                    raise ValueError(f'Invalid JSON entry for article {count + 1}')
                count += 1
                title = json.loads(matched.group())
                if title in requested:
                    text = json.loads(b'{' + entry + b'}')[title]
                    articles[title] = HTML_TAG.sub('', text)
        if bundle.next() is not None:
            raise ValueError('Unexpected additional author archive member')
    return articles, count


def lookup(title, articles):
    # Match the author's baseline util.RetrieveArticle title resolution.
    article = articles.get(title)
    return article if article is not None else articles.get(title.lower())


def exact_span(text, start, end, surface):
    return text is not None and 0 <= start < end <= len(text) and text[start:end] == surface


@dataclass(frozen=True)
class LinkedSupport:
    paragraph_id: str
    question_id: str
    seed_family: str
    linked_family: str
    linked_text_sha256: str
    start: int
    end: int
    authored_questions_for_anchor: int


def supports(rows, articles):
    stats = Counter()
    selected = []
    for paragraph in rows:
        anchors = Counter(l['target'] for l in visible_anchors(paragraph))
        authored_questions = Counter(target for question in paragraph['questions']
                                     for target in set(question['question_links']))
        seed_family = family(paragraph['title'])
        for question in paragraph['questions']:
            stats['authored_questions'] += 1
            external = []
            for context in question['context']:
                passage = context['passage']
                if passage == 'main':
                    stats['seed_excerpts'] += 1
                    stats['seed_exact'] += exact_span(paragraph['text'], *context['indices'], context['text'])
                else:
                    text = lookup(passage, articles)
                    aligned = exact_span(text, *context['indices'], context['text'])
                    stats['linked_excerpts'] += 1
                    stats['linked_document_found'] += text is not None
                    stats['linked_exact'] += aligned
                    external.append((context, text, aligned))
            links = question['question_links']
            if len(links) != 1 or anchors[links[0]] != 1:
                continue
            stats['one_visible_writer_anchor'] += 1
            answer = question['answer']
            if answer['type'] == 'none' or len(external) != 1 or external[0][0]['passage'] != links[0]:
                continue
            stats['answerable_one_linked_excerpt_matching_writer_anchor'] += 1
            ((evidence, text, aligned),) = external
            if not aligned:
                continue
            start, end = evidence['indices']
            linked_family = family(evidence['passage'])
            selected.append(LinkedSupport(paragraph['pid'], question['qid'], seed_family, linked_family,
                                          fingerprint(text), start, end, authored_questions[links[0]]))
            stats['eligible_exact_linked_evidence'] += 1
            if answer['type'] != 'span':
                continue
            stats['eligible_with_span_answer'] += 1
            answers = answer['answer_spans']
            if len(answers) != 1 or family(answers[0]['passage']) != linked_family:
                continue
            (span,) = answers
            stats['one_answer_surface_same_linked_document'] += 1
            stats['answer_surface_offset_matches_linked_document'] += exact_span(
                text, span['start'], span['end'], span['text'],
            )
            surface = span['text'].strip()
            stats['answer_surface_unique_in_gold_context'] += bool(surface) and evidence['text'].count(surface) == 1
    return selected, dict(stats)


def count_selection(rows):
    by_seed, by_document = defaultdict(list), defaultdict(list)
    for row in rows:
        by_seed[row.paragraph_id].append(row)
        by_document[row.linked_text_sha256].append(row)
    unique_identity = [row for group in by_seed.values() for row in group
                       if {other.question_id for other in group if other.linked_family == row.linked_family}
                       == {row.question_id}]
    unique_authored = [row for row in unique_identity if row.authored_questions_for_anchor == 1]
    return {
        'questions': len(rows), 'seed_article_families': len({row.seed_family for row in rows}),
        'linked_article_families': len({row.linked_family for row in rows}),
        'same_seed_distinct_linked_family_pairs': sum(
            a.linked_family != b.linked_family for group in by_seed.values() for a, b in combinations(group, 2)
        ),
        'same_seed_same_link_disjoint_support_pairs': sum(
            a.linked_family == b.linked_family and (a.end <= b.start or b.end <= a.start)
            for group in by_seed.values() for a, b in combinations(group, 2)
        ),
        'same_document_distinct_seed_disjoint_support_pairs': sum(
            a.seed_family != b.seed_family and (a.end <= b.start or b.end <= a.start)
            for group in by_document.values() for a, b in combinations(group, 2)
        ),
        'one_eligible_question_per_anchor': len(unique_identity),
        'one_question_support_end_inside_prefix_chars': {
            stop: sum(row.end <= stop for row in unique_identity) for stop in (512, 1024, 2048, 4096, 8000, 16000)
        },
        'one_authored_question_per_anchor_in_release': len(unique_authored),
        'one_authored_question_support_end_inside_prefix_chars': {
            stop: sum(row.end <= stop for row in unique_authored) for stop in (512, 1024, 2048, 4096, 8000, 16000)
        },
    }


def overlaps(train, dev):
    return {
        'linked_families': len({row.linked_family for row in train} & {row.linked_family for row in dev}),
        'linked_text_sha256': len({row.linked_text_sha256 for row in train} & {row.linked_text_sha256 for row in dev}),
    }


def run(release, context, output):
    data = read_release(release)
    source_overlap = verify_split_separation(data)
    dev_targets = {link['target'] for row in data['dev'] for link in row['links']}
    context_titles = {c['passage'] for rows in data.values() for row in rows for question in row['questions']
                      for c in question['context'] if c['passage'] != 'main'}
    articles, archive_count = source_articles(context, dev_targets | context_titles)
    train, train_stats = supports(data['train'], articles)
    dev, dev_stats = supports(data['dev'], articles)
    forbidden_titles = {family(title) for title in dev_targets} | {family(row['title']) for row in data['dev']}
    forbidden_hashes = {fingerprint(text) for title in dev_targets if (text := lookup(title, articles)) is not None}
    retained_train = [row for row in train if row.seed_family not in forbidden_titles
                      and row.linked_family not in forbidden_titles and row.linked_text_sha256 not in forbidden_hashes]
    heldout_overlap = overlaps(retained_train, dev)
    if any(heldout_overlap.values()):
        raise ValueError('Linked-document leakage remains after the fixed dev-family filter')
    result = {
        'dataset_release': {'url': RELEASE_URL, 'sha256': RELEASE_SHA256},
        'article_release': {'url': SOURCE_URL, 'sha256': SOURCE_SHA256, 'article_entries': archive_count},
        'annotation_protocol': 'https://aclanthology.org/2020.emnlp-main.86.pdf#page=4',
        'author_source_setup': 'https://github.com/jferguson144/IIRC-baseline/blob/main/setup.sh#L7-L9',
        'author_article_parser': 'https://github.com/jferguson144/IIRC-baseline/blob/main/util.py#L35-L45',
        'author_dataset_license': 'https://github.com/allenai/allennlp-website/blob/master/iirc.html',
        'source_text_terms': 'https://foundation.wikimedia.org/w/index.php?title=Policy:Terms_of_Use&oldid=98524#7._Licensing_of_Content',
        'source_overlap': source_overlap,
        'raw_annotation_alignment': {'train': train_stats, 'dev': dev_stats},
        'linked_overlap_before_filter': overlaps(train, dev),
        'linked_overlap_after_filter': heldout_overlap,
        'train_filter': 'Remove train rows whose seed title, linked title or linked text hash matches dev seed '
                        'titles or dev visible links (hashing available dev linked texts). Keep dev fixed.',
        'strict_split': {'train': count_selection(retained_train), 'dev': count_selection(dev)},
        'evidence': 'Use context.indices matched exactly against the author’s HTML-stripped document. '
                    'The answer span offsets are a separate coordinate system and were not used as evidence.',
        'scope': 'Separate answer annotators selected necessary source evidence for a writer-posed question. '
                 'Text offsets specify support location, not a time of subjective resolution.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    run(options.release, options.context, options.output)
