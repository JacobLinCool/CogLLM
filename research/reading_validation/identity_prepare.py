"""Prepare paired QASPER evidence passages and label-free causal input geometry."""

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from tempfile import TemporaryDirectory
from pathlib import Path

from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json, write_rows
from research.semantic_tracking.qasper_prepare import ARCHIVES, evidence_position, normalized, load_release
from research.token_readout.data import TOKENIZER, TOKENIZER_SHA256


DATA = Path('.cache/cogllm/reading-validation')
PREVIOUS = Path('.cache/cogllm/token-readout/linked-prepared')
QASPER = Path('.cache/cogllm/qasper')
PROTOCOL = Path(__file__).with_name('identity-protocol.md')
RESULT = Path('research/results/reading-validation-identity')
LIMITS = {'train': 400, 'dev': 120, 'test': 200}


def order(text):
    return sha256(('CogLLM identity v1 ' + text).encode()).digest()


def to_tokens(offsets, span):
    left, right = span
    indexes = [i for i, (a, b) in enumerate(offsets) if b > left and a < right and b > a]
    if not indexes or indexes != list(range(indexes[0], indexes[-1] + 1)):
        raise ValueError('No continuous token span corresponding to source text')
    return [indexes[0], indexes[-1] + 1]


def sealed_rows(path, entries):
    with TemporaryDirectory(dir=path.parent) as temp:
        stage = Path(temp) / path.name
        write_rows(stage, entries)
        if path.exists():
            if digest(stage) != digest(path):
                raise ValueError(f'A previously sealed corpus member changed: {path.name}')
        else:
            stage.replace(path)


def article(identifier, paper, tokenizer):
    paragraphs = [p.strip() for s in paper['full_text'] for p in s['paragraphs'] if p.strip()]
    lookup = defaultdict(list)
    for i, p in enumerate(paragraphs):
        lookup[normalized(p)].append(i)
    questions = []
    sizes = {}

    def size(i):
        if i not in sizes:
            sizes[i] = len(tokenizer.encode(paragraphs[i], add_special_tokens=False).ids)
        return sizes[i]

    for question in paper['qas']:
        location, _ = evidence_position(question, lookup)
        if location is None or not 8 <= size(location) <= 192:
            continue
        text = question['question'].strip()
        if len(tokenizer.encode(text, add_special_tokens=False).ids) <= 60:
            questions.append((question['question_id'], text, location, len(question['answers'])))
    questions.sort(key=lambda q: order(q[0]))
    pair = next((pair for pair in combinations(questions, 2)
                 if pair[0][2] != pair[1][2] and normalized(pair[0][1]).casefold() != normalized(pair[1][1]).casefold()), None)
    if pair is None:
        return None
    text = ''
    questions_at, targets, annotators = {}, {}, {}
    for index, (key, question, location, count) in enumerate(pair, 1):
        text += f'Question {index}: '
        questions_at[key] = [len(text), len(text) + len(question)]
        targets[key] = location
        annotators[key] = count
        text += question + '\n'
    body = {}
    for i in sorted({j for loc in targets.values() for j in (loc - 1, loc, loc + 1)
                     if 0 <= j < len(paragraphs) and 8 <= size(j) <= 192}):
        text += '\n'
        body[i] = [len(text), len(text) + len(paragraphs[i])]
        text += paragraphs[i] + '\n'
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded.ids) > 1536:
        return None
    spans = {k: to_tokens(encoded.offsets, v) for k, v in questions_at.items()}
    stops = {k: to_tokens(encoded.offsets, v) for k, v in body.items()}
    begin = min(v[0] for v in stops.values())
    if any(e > begin for _, e in spans.values()):
        raise ValueError('Question overlaps body')
    key = 'qasper-' + identifier
    return ({'id': key, 'text': text, 'anchor_tokens': spans},
            {'id': key, 'group_id': key, 'source': 'qasper', 'risk_start': begin,
             'window_tokens': {k: stops[i] for k, i in targets.items()}, 'paragraph_tokens': list(stops.values()),
             'annotators': annotators, 'body_indices': sorted(body), 'tokens': len(encoded.ids)})


def historical_ids():
    paths = [QASPER / name / 'stimuli.jsonl' for name in ('input', 'dev-holdout', 'replay')]
    paths.append(QASPER / 'censor-v1' / 'article-metadata.jsonl')
    items = {row['id'][6:] for path in paths for row in stream_rows(path) if row['id'].startswith('paper-')}
    if not items:
        raise ValueError('Historical paper inventory absent')
    return items, {str(p): digest(p) for p in paths}


def prepare():
    RESULT.mkdir(exist_ok=True, parents=True)
    DATA.mkdir(exist_ok=True, parents=True)
    if digest(TOKENIZER) != TOKENIZER_SHA256:
        raise ValueError('Different Qwen tokenizer')
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    prior, prior_files = historical_ids()
    splits = load_release(QASPER / 'raw')
    coverage, inputs = {}, {}
    labels = defaultdict(list)
    seen_text = set()
    for split, papers in splits.items():
        eligible = previous = duplicates = 0
        for paper_id in sorted(papers, key=order):
            if split == 'test' and paper_id in prior:
                previous += 1
                continue
            item = article(paper_id, papers[paper_id], tokenizer)
            if item is None:
                continue
            eligible += 1
            row, target = item
            fingerprint = sha256(row['text'].encode()).hexdigest()
            if fingerprint in seen_text:
                duplicates += 1
                continue
            seen_text.add(fingerprint)
            if len(labels[split]) < LIMITS[split]:
                inputs[row['id']] = row
                labels[split].append(target)
        coverage[split] = {'official': len(papers), 'eligible': eligible, 'historical_papers_excluded': previous,
                           'repeated_input_text': duplicates, 'selected': len(labels[split])}
    assert all(coverage[s]['selected'] > 0 for s in LIMITS)
    receipt = read_json(PREVIOUS / 'receipt.json')
    for name in ('stimuli.jsonl', 'qwen.json', 'qwen-learn.json', 'design.json'):
        assert receipt['file_sha256'][name] == digest(PREVIOUS / name)
    original = {r['id']:r['text'] for r in stream_rows(PREVIOUS/'stimuli.jsonl')}
    geometry = read_json(PREVIOUS/'qwen.json')['rows']
    for key, row in geometry.items():
        if row['kind'] == 'frick':
            continue
        inputs[key] = {'id': key, 'text': original[key], 'anchor_tokens': row['anchor_tokens']}
    for split, packet in read_json(PREVIOUS/'qwen-learn.json').items():
        for task in ('evidence', 'linked'):
            for row in packet[task]:
                labels[split].append({'id': row['id'], 'source': 'expmrc' if task=='evidence' else 'iirc',
                                      'group_id': row['group_id'], 'risk_start': row['risk_start'],
                                      'window_tokens': row['window_tokens']})
    paths = {'stimuli': DATA/'identity-stimuli.jsonl'}
    sealed_rows(paths['stimuli'], [inputs[k] for k in sorted(inputs)])
    for split, entries in labels.items():
        paths[split] = DATA/f'identity-{split}.jsonl'
        sealed_rows(paths[split], sorted(entries, key=lambda x: x['id']))
    locked = {'protocol_sha256':digest(PROTOCOL), 'source_archives':ARCHIVES, 'source_split_coverage':coverage,
              'historical_registry_sha256':prior_files, 'historical_papers_listed':len(prior),
              'source_counts': {s:dict(Counter(row['source'] for row in rows)) for s,rows in labels.items()},
              'files':{k:digest(p) for k,p in paths.items()}, 'archive_source_receipt_sha256':digest(PREVIOUS/'receipt.json'),
              'test_papers':[r['id'] for r in labels['test']], 'one_paired_document_per_new_paper':True}
    seal_path = RESULT/'input-seal.json'
    if seal_path.exists() and read_json(seal_path) != locked:
        previous = read_json(seal_path)
        if {k:v for k,v in previous.items() if k != 'protocol_sha256'} != {k:v for k,v in locked.items() if k != 'protocol_sha256'}:
            raise ValueError('Source selection changed after sealing')
        seal_path.rename(RESULT / ('input-seal-' + previous['protocol_sha256'][:12] + '.json'))
    if not seal_path.exists():
        write_json(seal_path, locked)
    print(json.dumps(coverage, indent=2))


if __name__ == '__main__':
    prepare()
