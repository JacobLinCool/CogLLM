"""Plan, validate, and join resumable frozen Qwen observation projections."""

import argparse
from collections import defaultdict
from itertools import islice, zip_longest
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from research.backbone_transfer.extract import (
    MAX_INPUT_TOKENS, MODELS, RELATIVE_RMS_LIMIT, checked_digest, digest, read_json, stream_rows, write_json, write_rows,
)
from research.union_training.run_observations_l4 import verify_packet


SCHEMA = 1
MAX_TOKENS = 150_000
MAX_POSITIONS = 10_000
EXTRACTION_FILES = frozenset({'states.npy', 'coordinates.jsonl', 'preflight.json', 'causality.json'})
SHARD_FIELDS = ('id', 'documents', 'positions', 'input_tokens', 'stimuli_sha256')


def verified_file(root, name, expected):
    path = root / name
    if Path(name).name != name or not path.is_file() or path.is_symlink() or digest(path) != expected:
        raise ValueError(f'Missing or altered member {name} in {root.name}')
    return path


def qwen_identity(packet, receipt, protocol_sha256):
    model = read_json(MODELS)['models']['qwen']
    if model['sha256']['tokenizer.json'] != receipt['tokenizer_sha256']:
        raise ValueError('The observation packet tokenizer differs from the registered Qwen')
    return {
        'candidate': 'qwen', 'model': model['repo'], 'revision': model['revision'],
        'hidden_width': model['hidden_width'], 'layers': model['layers'],
        'model_manifest_sha256': digest(MODELS), 'source_code_sha256': digest(MODELS.with_name('extract.py')),
        'tokenizer_sha256': receipt['tokenizer_sha256'], 'protocol_sha256': checked_digest(protocol_sha256),
        'parent_receipt_sha256': digest(packet / 'receipt.json'),
        'formal_holdout_sha256': receipt['formal_holdout_sha256'],
    }


def input_rows(stimuli):
    seen = set()
    for row in stream_rows(stimuli):
        if not isinstance(row, dict) or set(row) != {'id', 'text', 'stops'}:
            raise ValueError('A shard accepts only grade-free document rows')
        key, text, stops = (row[name] for name in ('id', 'text', 'stops'))
        if (not isinstance(text, str) or not text.strip() or not isinstance(key, str) or not key
                or key in seen or not isinstance(stops, list) or not stops
                or any(type(stop) is not int or not 0 < stop <= len(text) for stop in stops)
                or stops != sorted(set(stops))):
            raise ValueError(f'Invalid or repeated document {key!r}')
        seen.add(key)
        yield row


def partition(stimuli, tokenizer, max_tokens, max_positions):
    rows = input_rows(stimuli)
    content = []
    tokens = positions = 0
    while chunk := list(islice(rows, 256)):
        encoded = tokenizer.encode_batch([row['text'] for row in chunk], add_special_tokens=False)
        for row, encoding in zip(chunk, encoded, strict=True):
            input_tokens, stops = len(encoding.ids), len(row['stops'])
            if not 0 < input_tokens <= min(MAX_INPUT_TOKENS, max_tokens) or stops > max_positions:
                raise ValueError(f'A whole document exceeds the extraction budget: {row["id"]}')
            if content and (tokens + input_tokens > max_tokens or positions + stops > max_positions):
                yield content, tokens, positions
                content, tokens, positions = [], 0, 0
            content.append(row)
            tokens += input_tokens
            positions += stops
    if content:
        yield content, tokens, positions


def plan_shards(packet, tokenizer_path, output, protocol_sha256, *, max_tokens=MAX_TOKENS,
                max_positions=MAX_POSITIONS):
    """Create the immutable plan or verify and reuse the same existing plan."""
    from tokenizers import Tokenizer

    if type(max_tokens) is not int or type(max_positions) is not int or min(max_tokens, max_positions) <= 0:
        raise ValueError('Shard work limits must be positive integers')
    receipt = verify_packet(packet)
    identity = qwen_identity(packet, receipt, protocol_sha256)
    if digest(tokenizer_path) != identity['tokenizer_sha256']:
        raise ValueError('The local tokenizer differs from the registered Qwen')
    if output.exists():
        previous = verify_plan(packet, output, protocol_sha256)
        if (previous['max_tokens'], previous['max_positions']) != (max_tokens, max_positions):
            raise ValueError('An existing plan uses different shard limits')
        return previous
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with TemporaryDirectory(prefix='.observation-shards-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'shards'
        stage.mkdir()
        shards = []
        for number, (rows, tokens, positions) in enumerate(
                partition(packet / 'stimuli.jsonl', tokenizer, max_tokens, max_positions)):
            identifier = f'shard-{number:04d}'
            child = stage / identifier
            child.mkdir()
            stimuli = child / 'stimuli.jsonl'
            write_rows(stimuli, rows)
            entry = {'id': identifier, 'documents': len(rows), 'positions': positions,
                     'input_tokens': tokens, 'stimuli_sha256': digest(stimuli)}
            child_receipt = child / 'receipt.json'
            write_json(child_receipt, {'schema_version': SCHEMA, **identity, **entry})
            shards.append({**entry, 'receipt_sha256': digest(child_receipt)})
        manifest = {
            'schema_version': SCHEMA, **identity,
            'documents': receipt['documents'], 'positions': receipt['positions'], 'input_tokens': receipt['input_tokens'],
            'stimuli_sha256': receipt['packet_sha256']['stimuli.jsonl'], 'base_receipt_sha256': receipt['base_receipt_sha256'],
            'max_tokens': max_tokens, 'max_positions': max_positions, 'shards': shards,
        }
        write_json(stage / 'plan.json', manifest)
        verify_plan(packet, stage, protocol_sha256)
        stage.rename(output)
    return manifest


def verify_plan(packet, plan_dir, protocol_sha256=None):
    """Prove that every original document occurs exactly once, in original order."""
    plan = read_json(plan_dir / 'plan.json')
    receipt = verify_packet(packet)
    protocol = plan['protocol_sha256'] if protocol_sha256 is None else protocol_sha256
    identity = qwen_identity(packet, receipt, protocol)
    if (plan['schema_version'] != SCHEMA or any(plan[key] != value for key, value in identity.items())
            or plan['base_receipt_sha256'] != receipt['base_receipt_sha256']
            or plan['stimuli_sha256'] != receipt['packet_sha256']['stimuli.jsonl']
            or not plan['shards'] or type(plan['max_tokens']) is not int or type(plan['max_positions']) is not int
            or min(plan['max_tokens'], plan['max_positions']) <= 0):
        raise ValueError('The extraction plan does not match the registered observation packet')
    source = input_rows(packet / 'stimuli.jsonl')
    for number, entry in enumerate(plan['shards']):
        identifier = f'shard-{number:04d}'
        if entry['id'] != identifier:
            raise ValueError(f'Unexpected or repeated shard identifier: {entry["id"]}')
        child = plan_dir / identifier
        child_receipt = read_json(verified_file(child, 'receipt.json', entry['receipt_sha256']))
        expected = {'schema_version': SCHEMA, **identity, **{key: entry[key] for key in SHARD_FIELDS}}
        if child_receipt != expected or any(type(entry[k]) is not int or entry[k] <= 0
                                            for k in ('documents', 'positions', 'input_tokens')):
            raise ValueError(f'A shard has changed its identity or counts: {identifier}')
        stimuli = verified_file(child, 'stimuli.jsonl', entry['stimuli_sha256'])
        count = positions = 0
        for original, copied in zip_longest(islice(source, entry['documents']), input_rows(stimuli)):
            if original is None or copied != original:
                raise ValueError(f'A shard contains missing, changed or out-of-order documents: {identifier}')
            count += 1
            positions += len(copied['stops'])
        if (count != entry['documents'] or positions != entry['positions'] or positions > plan['max_positions']
                or entry['input_tokens'] > plan['max_tokens']):
            raise ValueError(f'Shard work totals differ: {identifier}')
    if next(source, None) is not None:
        raise ValueError('Some packet documents are absent from the extraction plan')
    for key in ('documents', 'positions', 'input_tokens'):
        if plan[key] != receipt[key] or sum(child[key] for child in plan['shards']) != receipt[key]:
            raise ValueError(f'Inconsistent extraction plan total: {key}')
    return plan


def verify_causality(value, documents, token_counts):
    if value['relative_tolerance'] != RELATIVE_RMS_LIMIT:
        raise ValueError('The independent exact-prefix tolerance changed')
    pairs = defaultdict(set)
    for check in value['checks']:
        identifier, end, variant = (check[name] for name in ('id', 'character_end', 'variant'))
        if (identifier not in documents or type(end) is not int or not 0 < end < len(documents[identifier])
                or variant not in {'truncation', 'replacement'} or variant in pairs[(identifier, end)]
                or check['passed'] is not True or type(check['prefix_tokens']) is not int
                or not 3 <= check['prefix_tokens'] < token_counts[identifier]
                or not 0 <= check['max_relative_rms'] <= RELATIVE_RMS_LIMIT
                or not math.isfinite(check['max_absolute']) or check['max_absolute'] < 0):
            raise ValueError('A shard has a missing or invalid independent exact-prefix test')
        pairs[identifier, end].add(variant)
    if (any(pair != {'truncation', 'replacement'} for pair in pairs.values())
            or len({identifier for identifier, _ in pairs}) != len(pairs)
            or len(pairs) < min(3, len(documents))):
        raise ValueError('A shard has incomplete independent exact-prefix tests')


def verify_shard(stimulus_dir, extraction_dir):
    """Verify each downloaded projection against the immutable, grade-free shard."""
    import numpy as np

    receipt = read_json(stimulus_dir / 'receipt.json')
    stimuli = verified_file(stimulus_dir, 'stimuli.jsonl', receipt['stimuli_sha256'])
    model_manifest = read_json(MODELS)
    model = model_manifest['models']['qwen']
    pinned = {
        'candidate': 'qwen', 'model': model['repo'], 'revision': model['revision'],
        'hidden_width': model['hidden_width'], 'layers': model['layers'],
        'model_manifest_sha256': digest(MODELS), 'source_code_sha256': digest(MODELS.with_name('extract.py')),
        'tokenizer_sha256': model['sha256']['tokenizer.json'],
    }
    if any(receipt[key] != value for key, value in pinned.items()):
        raise ValueError('The shard receipt does not use the pinned Qwen model and extractor')
    if set(path.name for path in extraction_dir.iterdir()) != EXTRACTION_FILES | {'seal.json'}:
        raise ValueError(f'Unexpected extraction files for {receipt["id"]}')
    seal_path = extraction_dir / 'seal.json'
    if not seal_path.is_file() or seal_path.is_symlink():
        raise ValueError('Expected a regular shard seal')
    seal = read_json(seal_path)
    expected = {key: receipt[key] for key in ('candidate', 'model', 'revision', 'hidden_width', 'layers',
                'model_manifest_sha256', 'source_code_sha256', 'tokenizer_sha256', 'protocol_sha256',
                'stimuli_sha256', 'documents')}
    expected.update({
        'certified_stops': receipt['positions'], 'states_shape': [receipt['positions'], receipt['hidden_width']],
        'gpu': 'NVIDIA L4', 'states_dtype': 'float32', 'compute_dtype': 'float32',
        'content_forward_per_document': 1, 'targets_sha256': None,
        'architecture': model['architecture'], 'model_files_sha256': model['sha256'],
        'config_adapter': model.get('config_adapter'),
        'transformers': model.get('transformers', model_manifest['transformers']),
    })
    if (receipt['schema_version'] != SCHEMA or any(seal[key] != value for key, value in expected.items())
            or type(seal['content_forward_batches']) is not int
            or not 1 <= seal['content_forward_batches'] <= receipt['documents']
            or set(seal['file_sha256']) != EXTRACTION_FILES):
        raise ValueError(f'Shard seal and immutable receipt disagree: {receipt["id"]}')
    files = {name: verified_file(extraction_dir, name, value) for name, value in seal['file_sha256'].items()}
    preflight = read_json(files['preflight.json'])
    if (any(preflight[key] != seal[key] for key in ('candidate', 'model', 'revision', 'protocol_sha256',
                                                  'stimuli_sha256', 'tokenizer_sha256', 'documents', 'certified_stops'))
            or preflight['input_tokens'] != receipt['input_tokens'] or preflight['requested_stops'] != receipt['positions']
            or preflight['unavailable_stops_by_reason'] or preflight['special_tokens'] is not False
            or preflight['target_geometry']):
        raise ValueError(f'Shard token geometry disagrees with its seal: {receipt["id"]}')
    documents, token_counts = {}, {}
    positions = 0
    for row, location, detail in zip(input_rows(stimuli), stream_rows(files['coordinates.jsonl']),
                                     preflight['details'], strict=True):
        identifier, stops = row['id'], row['stops']
        if (location != {'id': identifier, 'stops': stops}
                or detail['id'] != identifier or detail['stops'] != stops or detail['unavailable']
                or len(detail['token_indices']) != len(stops)
                or any(type(index) is not int or not 0 <= index < detail['input_tokens']
                       for index in detail['token_indices'])
                or detail['token_indices'] != sorted(set(detail['token_indices']))):
            raise ValueError(f'Shard extraction misaligns the coordinate for {identifier}')
        documents[identifier] = row['text']
        token_counts[identifier] = detail['input_tokens']
        positions += len(stops)
    if ((len(documents), positions, sum(token_counts.values()))
            != (receipt['documents'], receipt['positions'], receipt['input_tokens'])):
        raise ValueError(f'Shard coordinate totals disagree: {receipt["id"]}')
    controls = read_json(files['causality.json'])
    if len(controls['checks']) != seal['independent_causal_checks']:
        raise ValueError('Causal control counts do not match the extraction seal')
    verify_causality(controls, documents, token_counts)
    states = np.load(files['states.npy'], mmap_mode='r', allow_pickle=False)
    if (states.dtype != np.dtype('<f4') or list(states.shape) != seal['states_shape']
            or not states.flags.c_contiguous or states.offset + states.nbytes != files['states.npy'].stat().st_size
            or any(not np.isfinite(states[i:i + 512]).all() for i in range(0, len(states), 512))):
        raise ValueError(f'Unexpected array format or nonfinite values in {receipt["id"]}')
    for key in ('model_load_and_verification_s', 'forward_and_readout_s', 'peak_gpu_reserved_bytes'):
        if not math.isfinite(seal[key]) or seal[key] < 0:
            raise ValueError(f'A shard has invalid timing or memory metadata: {receipt["id"]}')
    return seal


def assemble_shards(packet, plan_dir, extracted_dir, output):
    """Join verified contiguous shard states into a memory-mapped training projection."""
    import numpy as np

    plan = verify_plan(packet, plan_dir)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shape = (plan['positions'], plan['hidden_width'])
    with TemporaryDirectory(prefix='.observation-assembly-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'projection'
        stage.mkdir()
        states_path = stage / 'states.npy'
        matrix = np.lib.format.open_memmap(states_path, mode='w+', dtype='<f4', shape=shape)
        seals, checks, offset = [], [], 0
        with (stage / 'coordinates.jsonl').open('xb') as coordinates:
            for entry in plan['shards']:
                identifier = entry['id']
                directory = extracted_dir / identifier
                seal = verify_shard(plan_dir / identifier, directory)
                fields = ('torch', 'transformers', 'cuda', 'architecture', 'config_adapter', 'model_files_sha256')
                if seals and any(seal[key] != seals[0]['seal'][key] for key in fields):
                    raise ValueError(f'Extraction environment differs for {identifier}')
                states = np.load(directory / 'states.npy', mmap_mode='r', allow_pickle=False)
                matrix[offset:offset + len(states)] = states
                offset += len(states)
                del states
                coordinates.write((directory / 'coordinates.jsonl').read_bytes())
                for control in read_json(directory / 'causality.json')['checks']:
                    checks.append({**control, 'shard': identifier})
                seals.append({'id': identifier, 'receipt_sha256': entry['receipt_sha256'],
                              'seal_sha256': digest(directory / 'seal.json'), 'seal': seal})
        if offset != shape[0]:
            raise ValueError('The assembled projection does not cover every observation position')
        matrix.flush()
        del matrix
        write_json(stage / 'causality.json', {
            'checks': checks, 'relative_tolerance': RELATIVE_RMS_LIMIT,
            'comparison': 'independent exact-prefix controls on all frozen extraction shards',
        })
        aggregated = {key: value for key, value in seals[0]['seal'].items() if key != 'file_sha256'}
        for key in ('content_forward_batches', 'model_load_and_verification_s', 'forward_and_readout_s'):
            aggregated[key] = sum(item['seal'][key] for item in seals)
        aggregated.update({
            'documents': plan['documents'], 'certified_stops': plan['positions'],
            'states_shape': list(shape), 'stimuli_sha256': plan['stimuli_sha256'],
            'independent_causal_checks': len(checks),
            'peak_gpu_reserved_bytes': max(item['seal']['peak_gpu_reserved_bytes'] for item in seals),
            'recording_plan_sha256': digest(plan_dir / 'plan.json'),
            'packet_receipt_sha256': plan['parent_receipt_sha256'],
            'formal_holdout_sha256': plan['formal_holdout_sha256'],
            'shards': [{key: value for key, value in item.items() if key != 'seal'} for item in seals],
            'file_sha256': {path.name: digest(path) for path in sorted(stage.iterdir())},
        })
        write_json(stage / 'seal.json', aggregated)
        stage.rename(output)
    return aggregated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    plan = actions.add_parser('plan')
    for option in ('packet', 'tokenizer-path', 'output'):
        plan.add_argument(f'--{option}', type=Path, required=True)
    plan.add_argument('--protocol-sha256', type=checked_digest, required=True)
    plan.add_argument('--max-tokens', type=int, default=MAX_TOKENS)
    plan.add_argument('--max-positions', type=int, default=MAX_POSITIONS)
    join = actions.add_parser('assemble')
    for option in ('packet', 'plan-dir', 'extracted-dir', 'output'):
        join.add_argument(f'--{option}', type=Path, required=True)
    config = vars(parser.parse_args())
    action = config.pop('action')
    result = (plan_shards if action == 'plan' else assemble_shards)(**config)
    print(f"{len(result['shards'])} verified shard entries for {result['documents']} documents")


if __name__ == '__main__':
    main()
