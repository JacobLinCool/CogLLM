"""Prepare a paper-disjoint QASPER evidence-arrival pilot from the Ai2 v0.3 release."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tarfile
import unicodedata

from tokenizers import Tokenizer


RELEASE = 'https://qasper-dataset.s3.us-west-2.amazonaws.com/'
ARCHIVES = {
    'qasper-train-dev-v0.3.tgz': 'a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a',
    'qasper-test-and-evaluator-v0.3.tgz': '72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72',
}
MEMBERS = {
    'train': ('qasper-train-dev-v0.3.tgz', 'qasper-train-v0.3.json'),
    'dev': ('qasper-train-dev-v0.3.tgz', 'qasper-dev-v0.3.json'),
    'test': ('qasper-test-and-evaluator-v0.3.tgz', 'qasper-test-v0.3.json'),
}
TOKENIZER_SHA = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
SEED = 43
MAX_CHARS = 14_000
MAX_PARAGRAPHS = 32
MAX_TOKENS = 6_000
MAX_QUESTION_TOKENS = 192
MAX_QUESTIONS_PER_PAPER = 4
MAX_TRAIN_PAPERS = 60
HOLDOUT_LIMITS = {'test': 40, 'dev': None}


@dataclass(frozen=True)
class Question:
    identifier: str
    text: str
    evidence_position: int
    annotations: int
    tokens: int


@dataclass(frozen=True)
class Paper:
    identifier: str
    text: str
    stops: list[int]
    questions: list[Question]
    tokens: int


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sort_key(identifier: str) -> bytes:
    return hashlib.sha256(f'{SEED}:{identifier}'.encode()).digest()


def normalized(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', text).split())


def load_release(directory: Path) -> dict[str, dict]:
    for name, expected in ARCHIVES.items():
        if digest(directory / name) != expected:
            raise ValueError(f'Input archive digest differs from the recorded Ai2 v0.3 release: {name}')
    result = {}
    all_papers = set()
    for source, (archive, name) in MEMBERS.items():
        with tarfile.open(directory / archive, 'r:gz') as handle:
            member = handle.getmember(name)
            if not member.isfile():
                raise ValueError(f'Expected a regular JSON file in the archive: {name}')
            with handle.extractfile(member) as stream:
                papers = json.load(stream)
        if all_papers.intersection(papers):
            raise ValueError(f'Official QASPER split {source} overlaps an earlier split')
        all_papers.update(papers)
        result[source] = papers
    return result


def certified_stops(tokenizer: Tokenizer, text: str, stops: list[int]) -> tuple[bool, int]:
    full = tokenizer.encode(text, add_special_tokens=False)
    positions = {right: i for i, (left, right) in enumerate(full.offsets) if left < right}
    for stop in stops:
        if stop not in positions:
            return False, len(full.ids)
        prefix = tokenizer.encode(text[:stop], add_special_tokens=False)
        if full.ids[:positions[stop] + 1] != prefix.ids:
            return False, len(full.ids)
    return True, len(full.ids)


def rendered_prefix(tokenizer: Tokenizer, paper: dict, paragraphs: list[tuple[int, str, str]]) -> tuple[str, list[int]]:
    chunks = [paper['title'].strip() + '\n\n' + paper['abstract'].strip() + '\n\n']
    stops = []
    length = len(chunks[0])
    previous_section = None
    for section, heading, paragraph in paragraphs[:MAX_PARAGRAPHS]:
        chunk = (heading + '\n\n' if section != previous_section and heading else '') + paragraph + '\n\n'
        if length + len(chunk) > MAX_CHARS:
            break
        chunks.append(chunk)
        length += len(chunk)
        stops.append(length)
        previous_section = section
    # Evidence labels have not been examined. This cap can only remove trailing paragraphs.
    while stops and len(tokenizer.encode(''.join(chunks), add_special_tokens=False).ids) > MAX_TOKENS:
        chunks.pop()
        stops.pop()
    return ''.join(chunks), stops


def evidence_position(question: dict, paragraph_lookup: dict[str, list[int]]) -> tuple[int | None, str]:
    if question['paper_read'] != 'no':
        return None, 'question_writer_not_documented_as_new_to_paper'
    if not question['question'].strip():
        return None, 'empty_question'
    if not question['answers']:
        return None, 'no_answer_annotations'
    positions = set()
    for record in question['answers']:
        answer = record['answer']
        evidence = answer['evidence']
        if answer['unanswerable'] or len(evidence) != 1 or evidence[0].startswith('FLOAT SELECTED:'):
            return None, 'any_annotation_unanswerable_or_not_single_text_paragraph'
        candidates = paragraph_lookup.get(normalized(evidence[0]), [])
        if len(candidates) != 1:
            return None, 'any_evidence_not_uniquely_aligned_to_body'
        positions.add(candidates[0])
    if len(positions) != 1:
        return None, 'annotations_select_different_paragraphs'
    return positions.pop(), 'accepted'


def select_questions(questions: list[Question]) -> list[Question]:
    questions = sorted(questions, key=lambda question: sort_key(question.identifier))
    first = questions[0]
    second = next(question for question in questions if question.evidence_position != first.evidence_position)
    rest = [question for question in questions if question not in (first, second)]
    return [first, second, *rest[:MAX_QUESTIONS_PER_PAPER - 2]]


def eligible_papers(tokenizer: Tokenizer, raw_papers: dict[str, dict]) -> tuple[list[Paper], dict]:
    audit_papers = Counter()
    audit_questions = Counter()
    candidates = []
    for paper_id, paper in raw_papers.items():
        paragraphs = [(i, (section['section_name'] or '').strip(), paragraph.strip())
                      for i, section in enumerate(paper['full_text']) for paragraph in section['paragraphs']
                      if paragraph.strip()]
        lookup = defaultdict(list)
        for index, (_, _, paragraph) in enumerate(paragraphs, start=1):
            lookup[normalized(paragraph)].append(index)
        text, stops = rendered_prefix(tokenizer, paper, paragraphs)
        if len(stops) < 2:
            audit_papers['fewer_than_two_visible_body_paragraphs'] += 1
            continue
        multiplicity = Counter(normalized(item['question']).casefold() for item in paper['qas'])
        questions = []
        for item in paper['qas']:
            pos, reason = evidence_position(item, lookup)
            if pos is None:
                audit_questions[reason] += 1
                continue
            if multiplicity[normalized(item['question']).casefold()] != 1:
                audit_questions['duplicate_question_text_within_paper'] += 1
                continue
            if pos > len(stops):
                audit_questions['evidence_beyond_fixed_prefix'] += 1
                continue
            question_text = item['question'].strip()
            certified, tokens = certified_stops(tokenizer, question_text, [len(question_text)])
            if not certified or not 0 < tokens <= MAX_QUESTION_TOKENS:
                audit_questions['uncertified_or_overlong_question'] += 1
                continue
            questions.append(Question(item['question_id'], question_text, pos, len(item['answers']), tokens))
            audit_questions['eligible_before_pair_filter'] += 1
        if len({question.evidence_position for question in questions}) < 2:
            audit_papers['no_two_eligible_questions_with_distinct_positions'] += 1
            continue
        certified, tokens = certified_stops(tokenizer, text, stops)
        if not certified:
            audit_papers['uncertified_body_paragraph_boundary'] += 1
            continue
        candidates.append(Paper(paper_id, text, stops, select_questions(questions), tokens))
        audit_papers['eligible_for_paired_comparison'] += 1
    return candidates, {'official_papers': len(raw_papers), 'paper_disposition': dict(audit_papers),
                        'question_disposition_for_papers_with_two_paragraphs': dict(audit_questions)}


def write_rows(path: Path, records: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def run(raw: Path, tokenizer_path: Path, output: Path, *, holdout: str = 'test') -> None:
    if digest(tokenizer_path) != TOKENIZER_SHA:
        raise ValueError('Tokenizer digest differs from the pretrained Qwen3.5-2B revision')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    paper_limits = {'train': MAX_TRAIN_PAPERS, holdout: HOLDOUT_LIMITS[holdout]}
    unused = 'dev' if holdout == 'test' else 'test'
    selected = {}
    reports = {}
    releases = load_release(raw)
    for source, limit in paper_limits.items():
        candidates, reports[source] = eligible_papers(tokenizer, releases[source])
        selected[source] = sorted(candidates, key=lambda paper: sort_key(paper.identifier))[:limit]
        if not selected[source]:
            raise ValueError(f'No usable {source} materials for matched same-paper comparisons')

    stimuli, train, evaluation, truth = [], [], [], []
    ids = set()
    tokens = []
    metadata = {}
    for source, papers in selected.items():
        reports[source]['selected_papers'] = len(papers)
        reports[source]['selected_questions'] = sum(len(paper.questions) for paper in papers)
        reports[source]['selected_same_paper_ordered_pairs'] = sum(
            sum(a.evidence_position < b.evidence_position for a in paper.questions for b in paper.questions)
            for paper in papers
        )
        for paper in papers:
            paper_id = f'paper-{paper.identifier}'
            stimuli.append(dict(id=paper_id, text=paper.text, stops=paper.stops))
            tokens.append(paper.tokens)
            metadata[paper_id] = {'arxiv': f'https://arxiv.org/abs/{paper.identifier}',
                                  'official_split': source, 'body_paragraphs': len(paper.stops),
                                  'questions': {}}
            for question in paper.questions:
                question_id = f'question-{question.identifier}'
                if question_id in ids:
                    raise ValueError(f'Duplicate question identifier: {question_id}')
                ids.add(question_id)
                stimuli.append(dict(id=question_id, text=question.text, stops=[len(question.text)]))
                tokens.append(question.tokens)
                position = question.evidence_position
                ref = {'paper_id': paper_id, 'question_id': question_id}
                interval = {**ref, 'lower': position - 1, 'upper': position}
                metadata[paper_id]['questions'][question_id] = {'unanimous_annotations': question.annotations}
                if source == 'train':
                    train.append(interval)
                else:
                    evaluation.append(ref)
                    truth.append(interval)

    output.mkdir(parents=True, exist_ok=False)
    tables = {'stimuli.jsonl': stimuli, 'train.jsonl': train, 'eval.jsonl': evaluation, f'{holdout}-truth.jsonl': truth}
    for name, rows in tables.items():
        write_rows(output / name, rows)
    inventory = {'stimuli_sha256': digest(output / 'stimuli.jsonl'), 'texts': len(stimuli),
                 'stops': sum(len(item['stops']) for item in stimuli), 'tokens': sum(tokens),
                 'max_tokens': max(tokens), 'tokenizer_sha256': TOKENIZER_SHA, 'excluded_stops': []}
    audit = {
        'archives': {name: {'url': RELEASE + name, 'sha256': sha} for name, sha in ARCHIVES.items()},
        'official_release': 'https://huggingface.co/datasets/allenai/qasper/blob/main/qasper.py#L149',
        'public_dataset_license': 'CC BY 4.0',
        'underlying_paper_provenance': {
            'source': 'https://aclanthology.org/2021.naacl-main.365.pdf#page=9',
            'scope': 'The authors report filtering to CC-BY-* arXiv papers verified via Unpaywall and S2ORC. '
                     'Each selected paper keeps its arXiv provenance. Individual CC variants were not re-audited.',
        },
        'prefix': {'maximum_unicode_characters': MAX_CHARS, 'maximum_body_paragraphs': MAX_PARAGRAPHS,
                   'maximum_native_qwen_tokens': MAX_TOKENS,
                   'rendering': 'Title and abstract, then original source sections and body paragraphs in order. '
                                'Heading and paragraph delimiters are two newlines. Each stop includes its trailing delimiter. '
                                'Prefix cap precedes any inspection of evidence positions.'},
        'selection_rule': 'New-to-paper writer; all published answer annotations agree on one uniquely matched '
                          'visible text paragraph; at least two distinct evidence positions per selected paper. '
                          'First two questions under seeded hash order with distinct positions retained, then fill to four.',
        'seed': SEED, 'paper_limits': paper_limits, 'splits': reports,
        f'official_unused_{unused}_papers': len(releases[unused]), 'paper_metadata': metadata,
        'sha256': {name: digest(output / name) for name in tables},
    }
    for name, value in [('token-inventory.json', inventory), ('audit.json', audit)]:
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'splits': reports, 'token_inventory': inventory}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('raw', 'tokenizer', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--holdout', choices=HOLDOUT_LIMITS, default='test',
                        help='Official evaluation split. Use dev to retain all eligible fresh papers.')
    args = parser.parse_args()
    run(args.raw, args.tokenizer, args.output, holdout=args.holdout)
