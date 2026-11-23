"""Build the observation-model packet from the registered 24-source corpus."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from research.backbone_transfer.extract import certify, digest, read_json, stream_rows, write_json, write_rows
from research.union_training.observation_registry import RECONSTRUCTED, REPLACED, contracts
from research.union_training.prepare import HERE, normal_words, sealed_texts, sha


class Corpus:
    def __init__(self, packet, cache, holdout):
        receipt = read_json(packet / 'receipt.json')
        if digest(holdout) != receipt['formal_holdout_sha256']:
            raise ValueError('The registered formal holdout changed')
        for relative, expected in receipt['packet_sha256'].items():
            if digest(packet / relative) != expected:
                raise ValueError(f'Registered corpus changed: {relative}')
        for source, files in receipt['source_files_sha256'].items():
            for relative, expected in files.items():
                if digest(cache / relative) != expected:
                    raise ValueError(f'Original source changed: {source}/{relative}')
        self.receipt = receipt
        self.design = read_json(packet / 'index.json')['tasks']
        self.rows = {task['name']: list(stream_rows(packet / task['file'])) for task in self.design}
        self.documents = {}
        self.roles = {}
        self.blocks = defaultdict(list)
        blocked = sealed_texts(HERE.parents[1], read_json(holdout))
        for row in stream_rows(packet / 'stimuli.jsonl'):
            key, text = row['id'], row['text']
            if key in self.documents or key != sha(text) or sha(' '.join(normal_words(text))) in blocked:
                raise ValueError('Duplicated, altered or formally held-out stimulus')
            self.documents[key] = text
        for rows in self.rows.values():
            for row in rows:
                key = row['doc']
                group = row['split'], row['family']
                if key not in self.documents or self.roles.setdefault(key, group) != group:
                    raise ValueError('Original corpus has inconsistent document roles')

    def key(self, text):
        key = sha(text)
        if self.documents.get(key) != text:
            raise ValueError('Observation requests a stimulus outside the registered development corpus')
        return key

    def add(self, name, coordinates, outcome):
        coordinates = tuple(tuple(coord) for coord in coordinates)
        if not coordinates or not outcome or any(not math.isfinite(value) for value in outcome):
            raise ValueError(f'Missing coordinate or invalid response for {name}')
        for key, stop in coordinates:
            if type(stop) is not int or not 0 < stop <= len(self.documents[key]):
                raise ValueError(f'Out-of-range coordinate for {name}')
        role = self.roles[coordinates[0][0]]
        if any(self.roles[key] != role for key, _ in coordinates):
            raise ValueError(f'An observation crosses material groups: {name}')
        self.blocks[name].append((coordinates, list(outcome), role))


def plaba_issues(cache, corpus):
    path = cache / 'stage2/plaba'
    materials = {row['id']: row for row in stream_rows(path / 'stimuli.jsonl')}
    trials = list(stream_rows(path / 'trials.jsonl'))
    reported = []
    for trial in trials:
        field = trial['observations'].get('issue_tags')
        if trial['target_segment_id'] is not None and field and field['status'] == 'observed':
            reported.append((trial, field['value']))
    tags = sorted({tag for _, values in reported for tag in values})
    masks = Counter()
    for trial, values in reported:
        stimulus = materials[trial['stimulus_id']]
        target, = (entry for entry in stimulus['segments'] if entry['id'] == trial['target_segment_id'])
        key = corpus.key(stimulus['text'])
        coords = ((key, target['end']), (key, len(stimulus['text'])))
        index = sum(1 << tags.index(tag) for tag in set(values))
        masks[coords, index] += 1
    counts = defaultdict(lambda: [0] * (1 << len(tags)))
    for (coords, index), number in masks.items():
        counts[coords][index] += number
    for coords, values in sorted(counts.items()):
        corpus.add('plaba_issue_set', coords, values)
    return [tuple(tag for bit, tag in enumerate(tags) if index & (1 << bit)) for index in range(1 << len(tags))]


def convert(corpus, cache, registry):
    from research.union_training.paired_observations import choice_keys, read_pairs
    from research.union_training.path_observations import read_paths

    releases = {row['name']: (row['source'], row['labels']) for row in corpus.design}
    if set(releases) - REPLACED != set(registry) - RECONSTRUCTED.keys() or not REPLACED <= set(releases):
        raise ValueError('The registered endpoint list and observation contracts disagree')
    releases.update(RECONSTRUCTED)
    releases['plaba_issue_set'] = 'plaba', plaba_issues(cache, corpus)
    read_paths(cache, corpus.key, corpus.add)
    read_pairs(cache, corpus.key, corpus.add)

    clear_path = cache / 'clear-reading-ease/prepared-v1'
    clear_texts = {row['id']: row['text'] for row in stream_rows(clear_path / 'stimuli.jsonl')}
    clear = {
        corpus.key(clear_texts[row['id']]): (row['bt_easiness'], row['standard_error'])
        for row in stream_rows(clear_path / 'outcomes.jsonl')
    }
    for name in sorted(registry.keys() - RECONSTRUCTED.keys()):
        contract = registry[name]
        for row in corpus.rows[name]:
            key, stop, outcome = row['doc'], row['stop'], row['outcome']
            coords = [(key, stop)]
            if contract.coordinates == 'target_and_document_end':
                coords.append((key, len(corpus.documents[key])))
            if contract.likelihood == 'known_error_gaussian':
                point, error = clear[key]
                if not math.isclose(point, outcome[1], abs_tol=1e-10) or error < 0:
                    raise ValueError('CLEAR estimate or published standard error differs from its release')
                outcome = [point, error]
            corpus.add(name, coords, outcome)
    keys = defaultdict(dict)
    for name, coordinate, index in choice_keys(cache, corpus.key):
        if keys[name].setdefault(coordinate, index) != index:
            raise ValueError('One presented question has conflicting correct options')
    if set(corpus.blocks) != set(registry) or set(keys) != {'onestop_answer_choice', 'race_answer_choice'}:
        raise ValueError('Missing registered observations or answer identities')
    return releases, keys


def prepare(base, cache, tokenizer_path, holdout, output):
    from tokenizers import Tokenizer

    if output.exists():
        raise FileExistsError(output)
    corpus = Corpus(base, cache, holdout)
    if digest(tokenizer_path) != corpus.receipt['tokenizer_sha256']:
        raise ValueError('Observation geometry requires the registered Qwen tokenizer')
    registry = contracts()
    releases, choice_map = convert(corpus, cache, registry)
    requested = defaultdict(set)
    for records in corpus.blocks.values():
        for coords, _, _ in records:
            for key, stop in coords:
                requested[key].add(stop)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_count = 0
    encoded = []
    for key, stops in sorted(requested.items()):
        text = corpus.documents[key]
        encoding = tokenizer.encode(text, add_special_tokens=False)
        result = certify(key, text, sorted(stops), encoding, tokenizer)
        if set(result.certified) != stops:
            raise ValueError(f'Observation geometry is not an exact Qwen prefix: {key}, {result.missing}')
        token_count += len(result.ids)
        encoded.append({'id': key, 'text': text, 'stops': sorted(stops)})

    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = {}
    with TemporaryDirectory(dir=output.parent, prefix='.observations-') as temporary:
        stage = Path(temporary) / 'packet'
        stage.mkdir()
        write_rows(stage / 'stimuli.jsonl', encoded)
        tasks = []
        for name, contract in sorted(registry.items()):
            records = corpus.blocks[name]
            source, labels = releases[name]
            file = f'{name}.jsonl'
            rows = []
            for coords, outcome, (role, family) in records:
                record = {'coordinates': coords, 'role': role, 'family': family, 'outcome': outcome}
                if contract.likelihood == 'choice':
                    record['correct_index'] = choice_map[name][coords[0]]
                rows.append(record)
            counts = Counter(row['role'] for row in rows)
            if not counts[0]:
                raise ValueError(f'No training observations for {name}')
            write_rows(stage / file, rows)
            tasks.append({'name': name, 'source': source, 'labels': labels, 'file': file, **asdict(contract)})
            inventory[name] = {'source': source, 'train_rows': counts[0], 'development_rows': counts[1],
                               **asdict(contract)}
        write_json(stage / 'index.json', {'schema_version': 2, 'tasks': tasks})
        write_json(stage / 'receipt.json', {
            'base_receipt_sha256': digest(base / 'receipt.json'),
            'formal_holdout_sha256': corpus.receipt['formal_holdout_sha256'],
            'tokenizer_sha256': corpus.receipt['tokenizer_sha256'],
            'source_files_sha256': corpus.receipt['source_files_sha256'],
            'documents': len(encoded), 'input_tokens': token_count,
            'positions': sum(map(len, requested.values())),
            'endpoints': inventory,
            'packet_sha256': {path.name: digest(path) for path in sorted(stage.iterdir())},
        })
        stage.rename(output)
    print(f'{len(inventory)} observation blocks, {len(encoded)} documents, {token_count} tokens', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'tokenizer-path', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('.cache/cogllm'))
    parser.add_argument('--holdout', type=Path, default=HERE / 'holdout.json')
    prepare(**vars(parser.parse_args()))
