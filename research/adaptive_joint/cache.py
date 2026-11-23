"""Bind the full observation cache and the portable token packet to one backbone."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from shutil import copyfile, copyfileobj
from tempfile import TemporaryDirectory
from time import perf_counter

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json, write_rows
from research.token_readout.data import twelve_word_windows
from research.union_training.observation_shards import verify_plan, verify_shard
from research.union_training.verify_observations import verified_member


CACHE = Path('.cache/cogllm')
UNION = CACHE / 'union-training'
DEFAULT_OBSERVATIONS = UNION / 'observations-2'
DEFAULT_RAW = UNION / 'checkpointed/cogllm-resume-20261121T185154Z-8ade72'
DEFAULT_TOKEN = CACHE / 'token-readout/linked-prepared'
DEFAULT_SEED = Path('research/results/portable-token-readout-l4/linked/span-calibrated.pt')
HOLDOUT = Path(__file__).parents[1] / 'union_training/holdout.json'
TOKEN_FILES = {'stimuli.jsonl': 'stimuli.jsonl', 'layout.json': 'qwen.json', 'learning.json': 'qwen-learn.json'}
CACHE_SCHEMA = 1


def load_seed(path, revision):
    import torch

    seal = read_json(path.parent / 'seal.json')
    verified_member(path.parent, path.name, seal['files_sha256'][path.name])
    packet = torch.load(path, map_location='cpu', weights_only=True)
    if (seal['candidate'] != 'qwen' or seal['revision'] != revision
            or packet['candidate'] != 'qwen' or packet['revision'] != revision
            or packet['variant'] != 'span' or packet['relations'] is not True):
        raise ValueError('Seed must be a sealed native-attention Qwen span reader on the old cache revision')
    weight = packet['reader']['final.weight']
    if (weight.ndim != 2 or weight.shape[1] != 2048 or weight.dtype != torch.float32
            or not bool(weight.isfinite().all())):
        raise ValueError('Seed final projection has the wrong width, precision, or finiteness')
    return weight


def token_packet(source, output, tokenizer_sha):
    receipt = read_json(source / 'receipt.json')
    if receipt['tokenizer_sha256']['qwen'] != tokenizer_sha:
        raise ValueError('Token packet and observation shards use different tokenizers')
    for target, member in TOKEN_FILES.items():
        copyfile(verified_member(source, member, receipt['file_sha256'][member]), output / target)
    layout = read_json(output / 'layout.json')
    texts = {row['id']: row['text'] for row in stream_rows(output / 'stimuli.jsonl')}
    if len(texts) != receipt['documents'] or texts.keys() != layout['rows'].keys():
        raise ValueError('Token stimuli and layouts do not match the linked packet receipt')
    learn = read_json(output / 'learning.json')
    for role, assays in learn.items():
        if role not in ('train', 'dev'):
            raise ValueError('Unexpected token role in worker learning payload')
        for assay, rows in assays.items():
            for row in rows:
                value = layout['rows'][row['id']]
                if value['split'] != role or value['kind'] != assay:
                    raise ValueError('A token label crosses the training or development split')
    if set(learn) != {'train', 'dev'} or len(learn['train']['linked']) != 160:
        raise ValueError('Expected the sealed Frick/ExpMRC and 160 linked training observations')
    return texts, layout['rows']


def cross_packet_roles(observations, index, texts, layouts):
    modern_windows = defaultdict(set)
    modern_texts = defaultdict(set)
    for key, text in texts.items():
        role = layouts[key]['split']
        modern_texts[' '.join(text.casefold().split())].add(role)
        for window in twelve_word_windows(text):
            modern_windows[window].add(role)
    documents = {row['id']: row['text'] for row in stream_rows(observations / 'stimuli.jsonl')}
    overlap = {}
    for key, text in documents.items():
        roles = set(modern_texts.get(' '.join(text.casefold().split()), ()))
        for window in twelve_word_windows(text):
            roles.update(modern_windows.get(window, ()))
        if len(roles) > 1:
            raise ValueError('An observation document matches multiple token roles')
        if roles:
            overlap[key] = roles.pop()
    families = {}
    for task in index['tasks']:
        # The full packet is already verified against its original family-level split.
        for row in stream_rows(observations / task['file']):
            for key, _ in row['coordinates']:
                previous = families.setdefault(key, (row['family'], row['role']))
                if previous != (row['family'], row['role']):
                    raise ValueError('An original observation document crosses family or role')
    matched_families = {}
    for key, role in overlap.items():
        family, _ = families[key]
        if matched_families.setdefault(family, role) != role:
            raise ValueError('An observation family matches multiple token roles')
    if set(families) != set(documents):
        raise ValueError('An observation document is missing a registered family')
    expected_roles = {'train': 24, 'dev': 8, 'test': 8}
    if dict(Counter(overlap.values())) != expected_roles or any(
        not families[key][0].startswith('frick:') for key in overlap
    ):
        raise ValueError('Unexpected text overlap between the legacy observation packet and portable token data')
    return matched_families, expected_roles


def observation_labels(source, target, index, matched_families):
    changes = {}
    receipt = read_json(source / 'receipt.json')
    expected = receipt['endpoints']
    for task in index['tasks']:
        name = task['name']
        counts = Counter()

        def routed_rows():
            for row in stream_rows(source / task['file']):
                counts[f'original_{row["role"]}'] += 1
                current = matched_families.get(row['family'])
                if current == 'test':
                    counts['removed_test_overlap'] += 1
                    continue
                if current == 'dev' and row['role'] == 0:
                    row['role'] = 1
                    counts['promoted_to_development'] += 1
                if current == 'train' and row['role'] != 0:
                    raise ValueError('A joint token training stimulus contaminates observation development')
                counts[f'kept_{row["role"]}'] += 1
                yield row

        write_rows(target / task['file'], routed_rows())
        original = expected[name]
        if ((counts['original_0'], counts['original_1']) != (original['train_rows'], original['development_rows'])
                or counts['kept_0'] == 0):
            raise ValueError(f'Changed observation counts or empty training role: {name}')
        changes[name] = {
            key: counts[key] for key in ('original_0', 'original_1', 'kept_0', 'kept_1',
                                         'removed_test_overlap', 'promoted_to_development')
        }
    write_json(target / 'index.json', index)
    receipt['endpoints'] = {name: {**value, 'train_rows': changes[name]['kept_0'],
                                  'development_rows': changes[name]['kept_1']}
                            for name, value in expected.items()}
    receipt['source_stimuli_sha256'] = receipt['packet_sha256']['stimuli.jsonl']
    receipt['packet_sha256'] = {name: digest(target / name)
                                for name in ('index.json', *(task['file'] for task in index['tasks']))}
    receipt['joint_observation_view'] = 'current token heldout Frick labels excluded'
    write_json(target / 'receipt.json', receipt)
    return changes


def project_shards(shards, plan_dir, plan, output, weight, block_rows):
    import numpy as np
    import torch
    from torch.nn import functional as F

    shape = (plan['positions'], weight.shape[0])
    output_path = output / 'states.npy'
    compact = np.lib.format.open_memmap(output_path, mode='w+', dtype='<f4', shape=shape)
    verified = []
    position = 0
    with (output / 'coordinates.jsonl').open('xb') as coordinates, torch.inference_mode():
        for row in plan['shards']:
            name = row['id']
            folder = shards / name
            seal = verify_shard(plan_dir / name, folder)
            raw = np.load(folder / 'states.npy', mmap_mode='r', allow_pickle=False)
            for start in range(0, len(raw), block_rows):
                values = torch.from_numpy(raw[start:start + block_rows].copy())
                cropped = F.linear(F.layer_norm(values, (raw.shape[1],)), weight)
                if not bool(cropped.isfinite().all()):
                    raise ValueError(f'Nonfinite portable projection in {name}')
                compact[position:position + len(cropped)] = cropped.numpy()
                position += len(cropped)
            with (folder / 'coordinates.jsonl').open('rb') as handle:
                copyfileobj(handle, coordinates)
            verified.append({'id': name, 'positions': len(raw), 'seal_sha256': digest(folder / 'seal.json'),
                             'raw_state_sha256': seal['file_sha256']['states.npy']})
            print(f'{name} projected {position}/{shape[0]} positions', flush=True)
    compact.flush()
    del compact
    if position != shape[0]:
        raise ValueError('Compact states do not cover all original stops')
    return verified


def prepare(observations, raw, token, seed, output, block_rows=2048, threads=2):
    import torch

    if output.exists() or not 1 <= block_rows <= 4096 or not 1 <= threads <= 4:
        raise ValueError('Use an unused output path and a bounded CPU working set')
    load_start = perf_counter()
    torch.set_num_threads(threads)
    plan_dir = raw / 'plan'
    plan = verify_plan(observations, plan_dir)
    if plan['formal_holdout_sha256'] != digest(HOLDOUT):
        raise ValueError('Frozen observation packet uses another formal holdout list')
    weight = load_seed(seed, plan['revision'])
    index = read_json(observations / 'index.json')
    if len(index['tasks']) != 62 or len({row['source'] for row in index['tasks']}) != 24:
        raise ValueError('Expected all 62 registered observation blocks and 24 sources')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.joint-cache-', dir=output.parent) as temporary:
        root = Path(temporary) / 'inputs'
        obs_out, token_out = root / 'observations', root / 'token'
        obs_out.mkdir(parents=True)
        token_out.mkdir()
        texts, layouts = token_packet(token, token_out, plan['tokenizer_sha256'])
        matched, counts = cross_packet_roles(observations, index, texts, layouts)
        changes = observation_labels(observations, obs_out, index, matched)
        projection_start = perf_counter()
        shards = project_shards(raw / 'shards', plan_dir, plan, obs_out, weight, block_rows)
        projection_seconds = perf_counter() - projection_start
        copyfile(seed, root / 'seed.pt')
        members = {path.relative_to(root).as_posix(): digest(path) for path in sorted(root.rglob('*'))
                   if path.is_file()}
        write_json(root / 'receipt.json', {
            'schema': CACHE_SCHEMA, 'candidate': 'qwen', 'revision': plan['revision'],
            'tokenizer_sha256': plan['tokenizer_sha256'],
            'source_observation_receipt_sha256': digest(observations / 'receipt.json'),
            'source_token_receipt_sha256': digest(token / 'receipt.json'),
            'source_shard_plan_sha256': digest(plan_dir / 'plan.json'),
            'formal_holdout_sha256': plan['formal_holdout_sha256'],
            'seed_sha256': members['seed.pt'], 'documents': plan['documents'], 'positions': plan['positions'],
            'observations': len(index['tasks']), 'sources': 24, 'token_documents': len(texts),
            'legacy_train_rows': sum(row['kept_0'] for row in changes.values()),
            'legacy_dev_rows': sum(row['kept_1'] for row in changes.values()),
            'legacy_cross_packet_document_counts': counts, 'legacy_overlap_families': matched,
            'legacy_row_use': changes, 'portable_final_width': weight.shape[0],
            'feature_dtype': 'float32', 'projection': 'seed final linear applied to layer-normalized last hidden states',
            'projection_seconds': projection_seconds, 'preparation_seconds': perf_counter() - load_start,
            'raw_shards': shards, 'files_sha256': members, 'formal_labels_in_worker': False,
        })
        root.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument('--raw', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--token', type=Path, default=DEFAULT_TOKEN)
    parser.add_argument('--seed', type=Path, default=DEFAULT_SEED)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--block-rows', type=int, default=2048)
    parser.add_argument('--threads', type=int, default=2)
    prepare(**vars(parser.parse_args()))
