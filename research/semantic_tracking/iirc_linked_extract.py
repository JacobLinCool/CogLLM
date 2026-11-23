"""Extract only the certified projected stops of each IIRC linked path on an L4."""

import argparse
import fcntl
import hashlib
from itertools import chain
import json
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import numpy as np

from research.core_features import extract as core_extract
from research.core_features.extract import batches_for, causal_checks, plan, project_tokens
from research.semantic_tracking.iirc_prepare import digest
from research.stage3.probe import _jsonl
from research.stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION, encode, sample_gpu, save_json


MAX_BATCH_TOKENS = 6144
MAX_BATCH_EXAMPLES = 24
BATCHES_PER_SHARD = 48
FEATURES = 'features.npz'
WIDTH = 128
STIMULI = ('train/stimuli.jsonl', 'dev/stimuli.jsonl')


def inputs(root):
    assay = root / 'assay.json'
    metadata = json.loads(assay.read_text(encoding='utf-8'))
    hashes = {name: digest(root / name) for name in STIMULI}
    if metadata['model'] != MODEL or metadata['revision'] != REVISION:
        raise ValueError('The assay requires a different Qwen snapshot')
    if any(checksum != metadata['inputs_sha256'][name] for name, checksum in hashes.items()):
        raise ValueError('Plain linked stimuli differ from the sealed assay')
    rows = list(chain.from_iterable(_jsonl(root / name) for name in STIMULI))
    if len(rows) != sum(metadata['cases'].values()):
        raise ValueError('Plain path count does not match the sealed assay')
    return rows, {'assay': digest(assay), **hashes}, metadata['tokens']


def coordinates(documents):
    coordinates = [(identifier, stop) for identifier, _, positions in documents for stop, _ in positions]
    ids, stops = zip(*coordinates, strict=True)
    return np.asarray(ids), np.asarray(stops, dtype=np.int64)


def atomic_json(path, data):
    partial = path.with_suffix('.partial.json')
    save_json(partial, data)
    partial.replace(path)


def atomic_npz(path, fields):
    partial = path.with_suffix('.partial.npz')
    np.savez(partial, **fields)
    partial.replace(path)


def bind_plan(output, spec):
    path = output / 'plan.json'
    if path.is_file():
        if json.loads(path.read_text(encoding='utf-8')) != spec:
            raise ValueError('Cannot resume a different linked extraction plan')
    else:
        if any(member.name not in {'active.lock', 'plan.partial.json'} for member in output.iterdir()):
            raise ValueError('Extraction output already contains an unsealed plan')
        atomic_json(path, spec)
    return digest(path)


def shard_key(plan_sha, number, ids, stops):
    sha = hashlib.sha256(f'{plan_sha}:{number}'.encode())
    for identifier, stop in zip(ids, stops, strict=True):
        sha.update(f'{identifier}\t{stop}\n'.encode())
    return sha.hexdigest()


def read_shard(path, key, ids, stops):
    with np.load(path, allow_pickle=False) as source:
        if (set(source.files) != {'signature', 'ids', 'stops', 'states', 'forward_s', 'peak_reserved_bytes'}
                or source['signature'].item() != key or not np.array_equal(source['ids'], ids)
                or not np.array_equal(source['stops'], stops)):
            raise ValueError(f'Checkpoint shard differs from the sealed plan: {path.name}')
        states = source['states']
        seconds = float(source['forward_s'].item())
        peak = int(source['peak_reserved_bytes'].item())
    if (states.shape != (len(ids), WIDTH) or states.dtype != np.float32 or not np.isfinite(states).all()
            or not np.allclose(np.linalg.norm(states, axis=1), 1, atol=0.00005, rtol=0)
            or not np.isfinite(seconds) or seconds <= 0 or peak <= 0):
        raise ValueError(f'Invalid projected states or measurements in {path.name}')
    return states, seconds, peak


def write_progress(output, spec, plan_sha, shards, tokens, *, complete):
    atomic_json(output / 'progress.json', {
        'complete': complete, 'plan_sha256': plan_sha, 'shards_completed': shards, 'shards': spec['shards'],
        'tokens_completed': tokens, 'tokens': spec['tokens'],
    })


def completed(output, spec, plan_sha):
    path = output / 'extraction.json'
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if (receipt['plan_sha256'] != plan_sha or any(receipt[key] != value for key, value in spec.items())
            or receipt['features_sha256'] != digest(output / FEATURES)
            or receipt['causality_sha256'] != digest(output / 'causality.json')
            or len(receipt['per_shard']) != spec['shards']):
        raise ValueError('Completed linked extraction receipt or outputs changed')
    write_progress(output, spec, plan_sha, spec['shards'], spec['tokens'], complete=True)
    return True


def load_backbone(torch, transformers):
    model, loading = transformers.Qwen3_5ForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation='sdpa',
        use_safetensors=True, output_loading_info=True,
    )
    if any(loading.values()) or next(model.model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError(f'Pinned BF16 checkpoint did not load exactly: {loading}')
    return model.model.eval().requires_grad_(False).to(device='cuda', dtype=torch.float32)


def compute(shard_batches, language, pad, gpu_projection, torch, *, check_headroom):
    vectors = []
    for index, batch in enumerate(shard_batches):
        hidden = encode(language, [tokens for _, tokens, _ in batch], pad, torch)
        positions = [(index, position) for index, (_, _, stops) in enumerate(batch) for _, position in stops]
        sparse = hidden[[index for index, _ in positions], [stop for _, stop in positions]]
        vectors.append(project_tokens(sparse, gpu_projection, torch).cpu().numpy())
        del hidden, sparse
        if check_headroom and index == 0 and torch.cuda.mem_get_info()[0] < 2 * 1024**3:
            raise RuntimeError('Largest linked batch leaves less than 2 GiB of device headroom')
    return np.concatenate(vectors)


def extract(root, output):
    import torch
    import transformers

    from research.semantic_tracking.qasper_fit import projection

    started = perf_counter()
    rows, hashes, split_tokens = inputs(root)
    if transformers.__version__ != TRANSFORMERS_VERSION or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected pinned Transformers and one CUDA device')
    gpu = torch.cuda.get_device_properties(0)
    if gpu.name != 'NVIDIA L4' or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError(f'Expected L4 with native BF16, got {gpu.name}')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise RuntimeError('Expected the pinned fast tokenizer and a padding token')
    documents, rejected = plan(tokenizer, rows, batch_prefixes=True)
    by_id = {identifier: len(tokens) for identifier, tokens, _ in documents}
    if rejected or len(by_id) != len(rows) or any(
        sum(by_id[row['id']] for row in _jsonl(root / name)) != split_tokens[name.split('/')[0]] for name in STIMULI
    ):
        raise ValueError('Tokenizer or planned causal stops differ from the sealed linked paths')
    batches = list(batches_for(documents, MAX_BATCH_EXAMPLES, MAX_BATCH_TOKENS))
    largest = max(range(len(batches)), key=lambda i: len(batches[i]) * len(batches[i][-1][1]))
    batches.insert(0, batches.pop(largest))
    shards = [batches[i:i + BATCHES_PER_SHARD] for i in range(0, len(batches), BATCHES_PER_SHARD)]
    matrix = projection()
    spec = {
        'model': MODEL, 'revision': REVISION, 'assay_sha256': hashes['assay'], 'inputs_sha256': hashes,
        'code_sha256': {Path(path).name: digest(Path(path)) for path in (__file__, core_extract.__file__)},
        'transformers': transformers.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': gpu.name, 'dtype': 'float32', 'batches': len(batches), 'shards': len(shards),
        'batches_per_shard': BATCHES_PER_SHARD, 'max_batch_tokens': MAX_BATCH_TOKENS,
        'max_batch_examples': MAX_BATCH_EXAMPLES, 'tokens': sum(split_tokens.values()),
        'projection_sha256': hashlib.sha256(matrix.tobytes()).hexdigest(),
    }
    plan_sha = bind_plan(output, spec)
    if completed(output, spec, plan_sha):
        return
    shard_dir = output / 'shards'
    shard_dir.mkdir(exist_ok=True)
    checks_path = output / 'causality.json'
    language = None
    halt = Event()
    sampler = None
    canonical = {}
    count = 0
    for row in rows:
        canonical[row['id']] = count
        count += len(row['stops'])
    ordered_ids = np.concatenate([np.repeat(row['id'], len(row['stops'])) for row in rows])
    ordered_stops = np.concatenate([np.asarray(row['stops'], dtype=np.int64) for row in rows])
    if len(ordered_stops) != sum(len(positions) for _, _, positions in documents):
        raise ValueError('Source-order and planned-coordinate counts disagree')
    states = np.empty((len(ordered_stops), WIDTH), np.float32)
    shard_logs = []
    completed_tokens = 0
    try:
        for number, group in enumerate(shards):
            docs = tuple(chain.from_iterable(group))
            shard_ids, shard_stops = coordinates(docs)
            key = shard_key(plan_sha, number, shard_ids, shard_stops)
            path = shard_dir / f'shard-{number:04d}.npz'
            if not path.is_file():
                if language is None:
                    language = load_backbone(torch, transformers)
                    gpu_projection = torch.as_tensor(matrix, device='cuda')
                    limit = MAX_BATCH_TOKENS // 2
                    controls = [(key, tokens[:limit], [(end, pos) for end, pos in entries if pos < limit])
                                for key, tokens, entries in documents]
                    checks = causal_checks(language, controls, tokenizer.pad_token_id, torch,
                                           token_projection=gpu_projection)
                    if not all(check['passed'] and check['projected_tokens']['passed'] for check in checks):
                        raise RuntimeError('Linked causal stop checks failed')
                    atomic_json(checks_path, {'plan_sha256': plan_sha, 'checks': checks})
                    sampler = Thread(target=sample_gpu, args=(output / f'telemetry-shard-{number:04d}.jsonl', halt))
                    sampler.start()
                torch.cuda.reset_peak_memory_stats()
                start = perf_counter()
                vectors = compute(group, language, tokenizer.pad_token_id, gpu_projection, torch,
                                  check_headroom=number == 0)
                atomic_npz(path, {'signature': np.asarray(key), 'ids': shard_ids, 'stops': shard_stops,
                                  'states': vectors, 'forward_s': np.asarray(perf_counter() - start),
                                  'peak_reserved_bytes': np.asarray(torch.cuda.max_memory_reserved())})
            vectors, forward, peak = read_shard(path, key, shard_ids, shard_stops)
            cursor = 0
            for identifier, _, entries in docs:
                length = len(entries)
                target = canonical[identifier]
                if not np.array_equal(ordered_stops[target:target + length], shard_stops[cursor:cursor + length]):
                    raise ValueError(f'Checkpoint does not cover original stops for {identifier}')
                states[target:target + length] = vectors[cursor:cursor + length]
                cursor += length
            completed_tokens += sum(len(tokens) for _, tokens, _ in docs)
            shard_logs.append({'number': number, 'sha256': digest(path), 'forward_s': forward,
                               'peak_reserved_bytes': peak, 'tokens_completed': completed_tokens})
            write_progress(output, spec, plan_sha, len(shard_logs), completed_tokens, complete=False)
            print(f'Linked extraction shard {len(shard_logs)}/{len(shards)}: {completed_tokens}/{spec["tokens"]} tokens', flush=True)
    finally:
        halt.set()
        if sampler is not None:
            sampler.join()

    causal = json.loads(checks_path.read_text(encoding='utf-8'))
    if (causal['plan_sha256'] != plan_sha or len(causal['checks']) != 12 or completed_tokens != spec['tokens']
            or not all(check['passed'] and check['projected_tokens']['passed'] for check in causal['checks'])):
        raise RuntimeError('The assembled linked features do not have a complete causal verification')
    features = output / FEATURES
    atomic_npz(features, {'ids': ordered_ids, 'stops': ordered_stops, 'states': states, 'projection': matrix})
    forward = float(sum(row['forward_s'] for row in shard_logs))
    atomic_json(output / 'extraction.json', {
        **spec, 'features_sha256': digest(features), 'plan_sha256': plan_sha, 'causality_sha256': digest(checks_path),
        'input_paths': len(rows), 'certified_character_stops': len(ordered_stops),
        'feature_width': WIDTH, 'state_normalization': 'row L2 normalization after fixed float32 projection',
        'forward_and_readout_s': forward, 'total_this_invocation_s': perf_counter() - started,
        'throughput_tokens_s': completed_tokens / forward, 'per_shard': shard_logs,
    })
    write_progress(output, spec, plan_sha, len(shards), completed_tokens, complete=True)


def run(root, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'active.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'An extractor already owns {output}') from error
        extract(root, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', dest='root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(**vars(parser.parse_args()))
