"""Build an article-held-out IIRC question-to-visible-anchor assay."""

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import tarfile
import unicodedata

from research.stage2.records import write_jsonl


RELEASE_URL = 'https://jamesf-incomplete-qa.s3.amazonaws.com/iirc.tar.gz'
RELEASE_SHA256 = 'efab875854691530a3bd55809244b0656dff1e15e98e4108bae5405c63e31a9b'
SPLITS = ('train', 'dev')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def read_release(path):
    if digest(path) != RELEASE_SHA256:
        raise ValueError('IIRC input does not match the pinned author release')
    with tarfile.open(path, 'r:gz') as archive:
        data = {}
        for split in SPLITS:
            member = archive.getmember(f'iirc/{split}.json')
            if not member.isfile():
                raise ValueError(f'Expected a regular source file: {member.name}')
            with archive.extractfile(member) as stream:
                data[split] = json.load(stream)
    return data


def visible_anchors(passage):
    """Read source hyperlink offsets without inspecting question annotations."""
    anchors = []
    seen = set()
    for link in passage['links']:
        start, end = link['indices']
        target = link['target']
        if not (type(start) is type(end) is int and 0 <= start < end <= len(passage['text'])
                and passage['text'][start:end].strip() and isinstance(target, str) and target.strip()):
            raise ValueError(f"Invalid visible hyperlink at {passage['pid']}")
        reference = start, end, target
        if reference in seen:
            continue
        seen.add(reference)
        anchors.append({'anchor_id': f"{passage['pid']}:link:{len(anchors)}", 'start': start,
                        'end': end, 'target': target})
    return anchors


def observations(passage, anchors):
    by_target = defaultdict(list)
    for anchor in anchors:
        by_target[anchor['target']].append(anchor)
    questions = []
    excluded = Counter()
    for question in passage['questions']:
        targets = question['question_links']
        if not targets:
            excluded['no_selected_link'] += 1
            continue
        if len(targets) != 1:
            excluded['multiple_selected_links'] += 1
            continue
        target, = targets
        possible = by_target[target]
        if not possible:
            excluded['selected_target_has_no_exact_visible_link'] += 1
            continue
        if len(possible) != 1:
            excluded['selected_target_has_multiple_visible_links'] += 1
            continue
        if not normalized(question['question']):
            excluded['empty_question'] += 1
            continue
        questions.append({'paragraph_id': passage['pid'], 'question_id': question['qid'], 'question': question['question'],
                          'anchor_id': possible[0]['anchor_id']})
    return questions, excluded


def matching_pairs(questions, anchors):
    anchor_by_id = {row['anchor_id']: row for row in anchors}
    for left, right in combinations(questions, 2):
        a = anchor_by_id[left['anchor_id']]
        b = anchor_by_id[right['anchor_id']]
        if (a['target'] == b['target'] or max(a['start'], b['start']) < min(a['end'], b['end'])
                or normalized(left['question']) == normalized(right['question'])):
            continue
        yield {'paragraph_id': left['paragraph_id'], 'question_ids': [left['question_id'], right['question_id']]}


def prepare_split(passages):
    stimuli, inventory, labels, pairs = [], [], [], []
    excluded = Counter()
    raw_questions = 0
    for passage in passages:
        if not all(isinstance(passage[key], str) and passage[key].strip() for key in ('pid', 'title', 'text')):
            raise ValueError('IIRC passage lacks its identity, article title or text')
        stop = len(passage['text'])
        anchors = visible_anchors(passage)
        questions, failures = observations(passage, anchors)
        stimuli.append({'id': passage['pid'], 'text': passage['text'], 'stops': [stop]})
        inventory.append({'paragraph_id': passage['pid'], 'article_title': passage['title'],
                          'article_family': normalized(passage['title'].replace('_', ' ')),
                          'stop': stop, 'anchors': anchors})
        labels.extend(questions)
        pairs.extend(matching_pairs(questions, anchors))
        raw_questions += len(passage['questions'])
        excluded.update(failures)
    paired_paragraphs = {pair['paragraph_id'] for pair in pairs}
    counts = {
        'paragraphs': len(stimuli), 'article_families': len({row['article_family'] for row in inventory}),
        'visible_anchor_locations': sum(len(row['anchors']) for row in inventory),
        'authored_questions': raw_questions, 'single_anchor_questions': len(labels),
        'excluded_questions': dict(excluded), 'same_stop_pairs': len(pairs),
        'paired_articles': len({row['article_family'] for row in inventory if row['paragraph_id'] in paired_paragraphs}),
        'paired_questions': len({qid for pair in pairs for qid in pair['question_ids']}),
    }
    if raw_questions != len(labels) + sum(excluded.values()) or not pairs:
        raise ValueError('Incomplete question disposition or no distinct-anchor pairs')
    return {'stimuli': stimuli, 'visible-anchors': inventory, 'questions': labels, 'pairs': pairs}, counts


def verify_split_separation(data):
    keyed = {}
    for split, passages in data.items():
        question_ids = [question['qid'] for row in passages for question in row['questions']]
        passage_ids = [row['pid'] for row in passages]
        if len(passage_ids) != len(set(passage_ids)) or len(question_ids) != len(set(question_ids)):
            raise ValueError(f'Duplicate source IDs within {split}')
        keyed[split] = {
            'article_family': {normalized(row['title'].replace('_', ' ')) for row in passages},
            'paragraph_text': {normalized(row['text']) for row in passages},
            'paragraph_id': set(passage_ids), 'question_id': set(question_ids),
            'question_text': {normalized(question['question']) for row in passages for question in row['questions']},
        }
    overlap = {field: len(values & keyed['dev'][field]) for field, values in keyed['train'].items()}
    if any(overlap.values()):
        raise ValueError(f'Official development release overlaps training by article or question: {overlap}')
    return overlap


def run(archive, output):
    data = read_release(archive)
    overlap = verify_split_separation(data)
    prepared = {split: prepare_split(passages) for split, passages in data.items()}
    output.mkdir(parents=True, exist_ok=False)
    file_hashes = {}
    for split, (tables, _) in prepared.items():
        for name, rows in tables.items():
            destination = output / split / f'{name}.jsonl'
            destination.parent.mkdir(exist_ok=True)
            write_jsonl(destination, rows)
            file_hashes[str(destination.relative_to(output))] = digest(destination)
    audit = {
        'archive': {'url': RELEASE_URL, 'sha256': RELEASE_SHA256},
        'source_documents': {
            'annotation_method': 'https://aclanthology.org/2020.emnlp-main.86.pdf#page=3',
            'author_dataset_license': 'https://github.com/allenai/allennlp-website/blob/master/iirc.html',
            'upstream_paragraph_license': 'https://en.wikipedia.org/wiki/Wikipedia:Copyrights',
            'license_note': 'The Allen AI IIRC page lists CC BY 4.0. Retain Wikipedia article attribution and '
                            'CC BY-SA terms for source paragraphs. The bundled license.txt names DROP and does not '
                            'replace the IIRC-specific license statement or the source text terms.',
        },
        'splits': {split: counts for split, (_, counts) in prepared.items()},
        'cross_split_overlaps': overlap,
        'visible_stop': 'End of the unedited seed paragraph, in half-open Unicode character coordinates',
        'permitted_model_input': 'stimuli.jsonl alone for plain-text evaluation. visible-anchors.jsonl contains '
                                 'all hyperlink locations as shown by the source for separate link-aware diagnostics.',
        'label': 'A selected hyperlink target recorded by the question writer. Accept only one selected title '
                 'with one exactly matching visible source span. No unreported span is an absence label.',
        'contrast': 'Two different authored questions on one paragraph selecting different, disjoint links. '
                    'Swapping assignments controls anchor-to-question correspondence.',
        'answer_labels': 'The separately elicited answer and evidence context are excluded from all outputs.',
        'files_sha256': file_hashes,
    }
    (output / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(**vars(parser.parse_args()))
