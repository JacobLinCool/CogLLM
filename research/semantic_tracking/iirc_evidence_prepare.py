"""Prepare IIRC seed-paragraph anchor-to-support labels separately from plain inputs."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

from research.semantic_tracking.iirc_prepare import (
    RELEASE_SHA256,
    RELEASE_URL,
    digest,
    normalized,
    read_release,
    verify_split_separation,
    visible_anchors,
)
from research.stage2.records import write_jsonl


def relate(anchor, support):
    start, end = support
    if end <= anchor['start']:
        return 'earlier'
    if start >= anchor['end']:
        return 'later'
    return 'overlap'


def source_supports(passage, question):
    spans = set()
    for item in question['context']:
        if item['passage'] != 'main':
            continue
        begin, end = item['indices']
        if (type(begin) is not int or type(end) is not int
                or not 0 <= begin < end <= len(passage['text'])
                or passage['text'][begin:end] != item['text']):
            raise ValueError(f"Seed-relative support mismatch in question {question['qid']}")
        spans.add((begin, end))
    return sorted(spans)


def observed_supports(passage, stats):
    by_title = defaultdict(list)
    for anchor in visible_anchors(passage):
        by_title[anchor['target']].append(anchor)
    article_family = normalized(passage['title'].replace('_', ' '))
    for question in passage['questions']:
        stats['authored_questions'] += 1
        support = source_supports(passage, question)
        stats['exact_seed_supports_after_deduplication'] += len(support)
        targets = question['question_links']
        if len(targets) != 1:
            stats['no_single_selected_target'] += 1
            continue
        anchors = by_title[targets[0]]
        if len(anchors) != 1:
            stats['selected_target_has_no_unique_visible_span'] += 1
            continue
        if not normalized(question['question']):
            stats['empty_question'] += 1
            continue
        if question['answer']['type'] == 'none':
            stats['answerer_reported_unanswerable'] += 1
            continue
        if not support:
            stats['answerable_without_seed_support'] += 1
            continue
        anchor = anchors[0]
        related = [(relate(anchor, span), span) for span in support]
        counts = Counter(direction for direction, _ in related)
        for side in ('earlier', 'later'):
            if counts[side]:
                stats[f'with_{side}_support_before_strict_filter'] += 1
            if counts[side] == 1:
                stats[f'with_one_{side}_support_before_strict_filter'] += 1
        disjoint = [(direction, span) for direction, span in related if direction != 'overlap']
        if not disjoint:
            stats['only_anchor_overlapping_support'] += 1
            continue
        if len(disjoint) != 1:
            stats['multiple_disjoint_supports'] += 1
            if counts['earlier'] and counts['later']:
                stats['multiple_disjoint_across_both_directions'] += 1
            continue
        ((direction, (start, end)),) = disjoint
        stats[f'eligible_{direction}'] += 1
        stats['eligible_with_other_anchor_overlapping_supports'] += bool(counts['overlap'])
        label = {'paragraph_id': passage['pid'], 'article_family': article_family, 'question_id': question['qid'],
                 'anchor': anchor, 'direction': direction, 'support_start': start, 'support_end': end}
        yield label, normalized(question['question'])


def contrast_pairs(labels, stats):
    for (first, qa), (second, qb) in combinations(labels, 2):
        side = first['direction']
        a, b = first['anchor'], second['anchor']
        if (qa == qb or side != second['direction'] or a['target'] == b['target']
                or max(a['start'], b['start']) < min(a['end'], b['end'])):
            continue
        if max(first['support_start'], second['support_start']) < min(first['support_end'], second['support_end']):
            stats[f'{side}_pairs_overlapping_supports'] += 1
            continue
        stats[f'{side}_pairs_with_distinct_supports'] += 1
        if side == 'earlier':
            start = min(a['start'], b['start'])
            if max(first['support_end'], second['support_end']) > start:
                stats[f'{side}_pairs_cross_other_anchor_boundary'] += 1
                continue
        elif min(first['support_start'], second['support_start']) < max(a['end'], b['end']):
            stats[f'{side}_pairs_cross_other_anchor_boundary'] += 1
            continue
        yield {'paragraph_id': first['paragraph_id'], 'article_family': first['article_family'],
               'direction': side, 'question_ids': [first['question_id'], second['question_id']]}


def prepare_split(passages):
    stimuli, labels, pairs = [], [], []
    stats = Counter()
    for passage in passages:
        stimuli.append({'id': passage['pid'], 'text': passage['text'], 'stops': [len(passage['text'])]})
        selected = list(observed_supports(passage, stats))
        labels.extend(label for label, _ in selected)
        pairs.extend(contrast_pairs(selected, stats))
    for side in ('earlier', 'later'):
        stats[f'{side}_label_families'] = len({row['article_family'] for row in labels if row['direction'] == side})
        stats[f'{side}_contrast_pairs'] = sum(row['direction'] == side for row in pairs)
        stats[f'{side}_contrast_families'] = len({row['article_family'] for row in pairs if row['direction'] == side})
    stats['plain_paragraphs'] = len(stimuli)
    stats['label_coverage'] = len(labels)
    exclusions = ('no_single_selected_target', 'selected_target_has_no_unique_visible_span', 'empty_question',
                  'answerer_reported_unanswerable', 'answerable_without_seed_support',
                  'only_anchor_overlapping_support', 'multiple_disjoint_supports')
    if len(labels) + sum(stats[key] for key in exclusions) != stats['authored_questions']:
        raise ValueError('Incomplete IIRC seed evidence disposition')
    return {'stimuli': stimuli, 'locked-evidence': labels, 'locked-contrast-pairs': pairs}, dict(sorted(stats.items()))


def run(archive, output, audit_path):
    data = read_release(archive)
    overlap = verify_split_separation(data)
    output.mkdir(parents=True, exist_ok=False)
    counts_by_split, hashes = {}, {}
    for split, passages in data.items():
        tables, counts_by_split[split] = prepare_split(passages)
        directory = output / split
        directory.mkdir()
        for name, records in tables.items():
            path = directory / f'{name}.jsonl'
            write_jsonl(path, records)
            hashes[str(path.relative_to(output))] = digest(path)
    audit = {
        'author_archive': {'url': RELEASE_URL, 'sha256': RELEASE_SHA256},
        'author_annotation_method': 'https://aclanthology.org/2020.emnlp-main.86.pdf#page=4',
        'author_dataset_license': 'https://github.com/allenai/allennlp-website/blob/master/iirc.html',
        'source_text_terms': 'https://foundation.wikimedia.org/w/index.php?title=Policy:Terms_of_Use&oldid=98524#7._Licensing_of_Content',
        'coordinate_system': 'Half-open Unicode character offsets into the original seed paragraphs.',
        'inputs': 'Only stimuli.jsonl is a model input. It contains the original plain paragraph.',
        'label': 'Of a single writer-selected, uniquely positioned hyperlink, identify one disjoint exact '
                 'main support selected by the answer annotator for the same question, with earlier/later '
                 'defined by character order. Other overlapping supports can coexist.',
        'pair': 'In one paragraph, two distinct questions and visible anchors, and nonoverlapping supports '
                'both before or after both anchors. Swap known assignments. Other unmarked supports are unknown.',
        'latent_reader_state': 'Roles and offsets describe annotation task conditions and text position. '
                               'The data does not time a reader noticing, maintaining, or resolving a question.',
        'source_overlap': overlap,
        'splits': counts_by_split,
        'files_sha256': hashes,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    options = parser.parse_args()
    run(options.archive, options.output, options.audit)
