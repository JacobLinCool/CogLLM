"""Pack locked PLABA training folds and grade-free Kerwer heading contrasts."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from research.backbone_transfer import extract
from research.backbone_transfer.prepare import plaba_rows, read_json, read_jsonl
from research.stage6 import kerwer_score as kerwer


RESEARCH = Path(__file__).resolve().parents[1]
TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'


def _index(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate source values for {key}')
    return result


def _plaba(cache):
    source = cache / 'backbone-transfer' / 'prepared'
    audit = read_json(source / 'source-audit.json')['datasets']['plaba']
    expected = plaba_rows(cache)
    names = ('stimuli', 'layout', 'outcomes')
    for name, original in zip(names, expected[:3], strict=True):
        path = source / 'plaba' / f'{name}.jsonl'
        if extract.digest(path) != audit['output_sha256'][path.name] or read_jsonl(path) != original:
            raise ValueError(f'PLABA {name} differs from the frozen, independently rebuilt source')
    stimuli, layout, raw, provenance = expected
    grades = _index(raw, 'anchor_id')
    geometry = defaultdict(list)
    alignment = defaultdict(lambda: defaultdict(list))
    for anchor in layout:
        geometry[anchor['id']].append(anchor)
        alignment[anchor['alignment_anchor_id']][anchor['variant']].append(anchor)
    ratings = []
    for row in stimuli:
        nodes = geometry[row['id']]
        common = {(node['family_id'], node['group_id'], node['fold']) for node in nodes}
        if len(common) != 1 or [node['end'] for node in nodes] != row['stops']:
            raise ValueError('PLABA sentence geometry and document-level folds disagree')
        [(family, group, fold)] = common
        ratings.append({
            'id': row['id'], 'family_id': family, 'group_id': group, 'fold': fold,
            'ratings': [
                {'anchor_id': node['anchor_id'], 'start': node['start'], 'stop': node['end'],
                 'ease': grades[node['anchor_id']]['processing_ease'],
                 'understanding': grades[node['anchor_id']]['perceived_understanding']}
                for node in nodes
            ],
        })
    pairs = []
    for key, arms in sorted(alignment.items()):
        if arms.keys() != {'orig', 'simp'}:
            continue
        identity = {(row['family_id'], row['group_id'], row['fold']) for rows in arms.values() for row in rows}
        if len(identity) != 1:
            raise ValueError('A PLABA author alignment crosses families, components or folds')
        [(family, group, fold)] = identity
        pairs.append({
            'alignment_id': key, 'family_id': family, 'group_id': group, 'fold': fold,
            'original': [[row['id'], row['end']] for row in arms['orig']],
            'plain': [[row['id'], row['end']] for row in arms['simp']],
        })
    count = sum(sum(row['ease']) for doc in ratings for row in doc['ratings'])
    grouped = {(row['group_id'], row['fold']) for row in ratings}
    if (len(stimuli), sum(len(row['ratings']) for row in ratings), count, len(pairs), len(grouped)) != (
            78, 609, 1826, 265, 25):
        raise ValueError('PLABA document, label, alignment or component counts changed')
    if len({row['family_id'] for row in ratings}) != 39 or len({group for group, _ in grouped}) != len(grouped):
        raise ValueError('PLABA source or reader component crossing the archive split')
    return stimuli, ratings, pairs, {**provenance, 'prepared_sha256': audit['output_sha256']}


def _kerwer(cache):
    source = cache / 'kerwer-2021' / 'holdout'
    for name in (kerwer.STIMULI, kerwer.DESIGN):
        if extract.digest(source / name) != kerwer.INPUTS[name]:
            raise ValueError('The grade-free Kerwer archive differs from the frozen release')
    contract_path = RESEARCH / 'results' / 'kerwer-transfer-protocol.json'
    contract = read_json(contract_path)
    alignment = contract['alignment']
    if kerwer.fingerprint(alignment) != kerwer.ALIGNMENT:
        raise ValueError('Kerwer sentence stops differ from the validated alignment')
    original = _index(read_jsonl(source / kerwer.STIMULI), 'id')
    heading_bodies = alignment['eligible']['headings']
    design = [row for row in read_jsonl(source / kerwer.DESIGN) if row['id'] in heading_bodies]
    expected_arms = {'plain_with_headings', 'plain_without_headings'}
    spans_by_family = defaultdict(dict)
    stimuli, layout = [], []
    for row in sorted(design, key=lambda item: item['id']):
        key = row['id']
        item = original[key]
        text = item['text']
        stops = [stop for _, stop, _ in alignment['served'][key]]
        body = heading_bodies[key]
        if (row['arm'] not in expected_arms or stops != sorted(set(stops))
                or any(tuple(span) not in {tuple(g[:2]) for g in alignment['served'][key]} for span in body)):
            raise ValueError('A paired Kerwer heading sentence is outside its served text stops')
        spans_by_family[row['doi']][row['arm']] = [text[start:end] for start, end in body]
        stimuli.append({'id': key, 'text': text, 'stops': stops})
        layout.append({'id': key, 'doi': row['doi'], 'arm': row['arm'],
                       'matched_stops': [stop for _, stop in body]})
    if (len(stimuli) != 24 or len(spans_by_family) != 12
            or any(arms.keys() != expected_arms or arms['plain_with_headings'] != arms['plain_without_headings']
                   for arms in spans_by_family.values())
            or Counter({arm: sum(len(row['matched_stops']) for row in layout if row['arm'] == arm)
                        for arm in expected_arms}) != {arm: 181 for arm in expected_arms}):
        raise ValueError('The 12 literal Kerwer heading-only pairs must have the same body sentences')
    return stimuli, layout, {
        'study_path': str(source), 'headings_arm_counts': {arm: 12 for arm in sorted(expected_arms)},
        'matched_sentences_per_arm': 181, 'assessment_files_copied': 0,
        'source_stimuli_sha256': kerwer.INPUTS[kerwer.STIMULI],
        'source_design_sha256': kerwer.INPUTS[kerwer.DESIGN],
        'source_ratings_sha256': kerwer.INPUTS[kerwer.OUTCOMES],
        'alignment_sha256': kerwer.ALIGNMENT, 'contract_sha256': extract.digest(contract_path),
    }


def _certify(path, tokenizer):
    docs = extract.load_documents(path, tokenizer)
    if any(doc.missing for doc in docs.values()):
        raise ValueError('A frozen sentence boundary is unavailable on the causal native-token axis')
    return {'documents': len(docs), 'stops': sum(len(doc.certified) for doc in docs.values()),
            'input_tokens': sum(len(doc.ids) for doc in docs.values()),
            'max_tokens': max(len(doc.ids) for doc in docs.values())}


def prepare(cache, tokenizer_path, output):
    from tokenizers import Tokenizer

    if output.exists():
        raise FileExistsError(output)
    if extract.digest(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError('Expected the frozen Qwen3.5 tokenizer for sentence boundary certificates')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    stimuli, ratings, pairs, plaba_source = _plaba(cache)
    kerwer_stimuli, kerwer_layout, kerwer_source = _kerwer(cache)
    payloads = {
        'plaba/stimuli.jsonl': stimuli, 'plaba/ratings.jsonl': ratings, 'plaba/pairs.jsonl': pairs,
        'kerwer/stimuli.jsonl': kerwer_stimuli, 'kerwer/layout.jsonl': kerwer_layout,
    }
    folds = {}
    for held in range(5):
        train = [row for row in ratings if row['fold'] != held]
        held_rows = [row for row in ratings if row['fold'] == held]
        train_pairs = [row for row in pairs if row['fold'] != held]
        if {doc['group_id'] for doc in train} & {doc['group_id'] for doc in held_rows}:
            raise ValueError('A reader-connected component crosses a fit and its held-out fold')
        payloads[f'plaba/folds/{held}/ratings.jsonl'] = train
        payloads[f'plaba/folds/{held}/pairs.jsonl'] = train_pairs
        folds[str(held)] = {'train_ids': [row['id'] for row in train], 'held_ids': [row['id'] for row in held_rows],
                            'train_alignments': len(train_pairs), 'held_alignments': len(pairs) - len(train_pairs)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.expository-lora-') as temporary:
        staged = Path(temporary) / 'packet'
        for name, rows in payloads.items():
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            extract.write_rows(path, rows)
        inventory = {corpus: _certify(staged / corpus / 'stimuli.jsonl', tokenizer) for corpus in ('plaba', 'kerwer')}
        extract.write_json(staged / 'source-audit.json', {
            'schema': 'cogllm-expository-lora-v1', 'tokenizer_sha256': TOKENIZER_SHA256,
            'sources': {'plaba': plaba_source, 'kerwer': kerwer_source}, 'inventory': inventory,
            'files_sha256': {name: extract.digest(staged / name) for name in sorted(payloads)},
            'folds': folds,
            'coordinates': 'Unicode code-point exclusive end stops in the literal source text.',
            'model_inputs': 'Only id, text and stops. PLABA and Kerwer grading information is separate.',
            'fold_isolation': ('Use only plaba/folds/<held>/ratings.jsonl and pairs.jsonl to fit a model for '
                               'that outer fold. The global labels and Kerwer layout belong to local scoring. '
                               'Kerwer response rows are absent from this package.'),
        })
        staged.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('.cache/cogllm'))
    parser.add_argument('--tokenizer-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    prepare(**vars(parser.parse_args()))
