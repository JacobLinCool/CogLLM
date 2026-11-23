"""Certify GUM source stops and extract the fixed app projection on a single L4."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter


SPLITS = ('train', 'dev', 'test')
FIELDS = {'id', 'text', 'stops', 'paragraphs'}
TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
CANDIDATE = '74c26ff02569b2b3bc28ed291bf15fd2f85db5b4c93a3ad7bfc34f6a5d333ea0'
MAX_TOKENS = 8192
PROJECTION_WIDTH = 128
PREFIX_BATCH = 48
FP16_TOLERANCE = 0.001


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def json_digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Document:
    identifier: str
    tokens: list[int]
    offsets: list[tuple[int, int]]
    certified: list[tuple[int, int]]
    rejected: Counter


def read_inputs(inputs):
    records, seen, hashes = [], set(), {}
    for split in SPLITS:
        name = f'{split}.jsonl'
        path = inputs / name
        hashes[name] = digest(path)
        with path.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        if not rows:
            raise ValueError(f'No GUM articles in {name}')
        for row in rows:
            if not isinstance(row, dict) or row.keys() != FIELDS:
                raise ValueError(f'Expected only grade-free GUM article fields in {name}')
            identifier, text, stops, paragraphs = (row[key] for key in ('id', 'text', 'stops', 'paragraphs'))
            if (type(identifier) is not str or not identifier or identifier in seen or type(text) is not str
                    or not text.strip() or type(stops) is not list or not stops
                    or any(type(stop) is not int or not 0 < stop <= len(text) for stop in stops)
                    or stops != sorted(set(stops)) or type(paragraphs) is not list):
                raise ValueError(f'Invalid GUM article or source stop inventory in {name}')
            previous_end = 0
            for bounds in paragraphs:
                if (type(bounds) is not list or len(bounds) != 2 or any(type(value) is not int for value in bounds)
                        or not previous_end <= bounds[0] < bounds[1] <= len(text)):
                    raise ValueError(f'Invalid half-open source paragraph coordinates for {identifier!r}')
                previous_end = bounds[1]
            records.append((split, row))
            seen.add(identifier)
    return records, hashes


def certify(row, tokenize, tokenize_prefixes):
    text, identifier, stops = row['text'], row['id'], row['stops']
    tokens, offsets = tokenize(text)
    if (not tokens or len(tokens) != len(offsets)
            or any(not 0 <= start <= end <= len(text) for start, end in offsets)):
        raise ValueError(f'Invalid full-document tokenization for {identifier!r}')
    rejected = Counter()
    if len(tokens) > MAX_TOKENS:
        rejected['document_exceeds_token_limit'] = len(stops)
        return Document(identifier, tokens, offsets, [], rejected)

    # Use the final token when several pieces share a character boundary, as in the app extractor.
    positions = {end: position for position, (start, end) in enumerate(offsets) if start < end}
    candidates = [(end, positions[end]) for end in stops if end in positions]
    if len(candidates) < len(stops):
        rejected['no_token_ends_at_stop'] = len(stops) - len(candidates)
    certified = []
    for start in range(0, len(candidates), PREFIX_BATCH):
        chunk = candidates[start:start + PREFIX_BATCH]
        prefixes = tokenize_prefixes([text[:end] for end, _ in chunk])
        for (end, position), prefix in zip(chunk, prefixes, strict=True):
            if prefix == tokens[:position + 1]:
                certified.append((end, position))
            else:
                rejected['independent_prefix_tokenization_differs'] += 1
    return Document(identifier, tokens, offsets, certified, rejected)


def inventory(inputs, tokenize, tokenize_prefixes):
    rows, hashes = read_inputs(inputs)
    documents, summaries = [], []
    for split, row in rows:
        doc = certify(row, tokenize, tokenize_prefixes)
        if len(doc.certified) + sum(doc.rejected.values()) != len(row['stops']):
            raise RuntimeError('Certified and rejected stops do not reconcile to the declared inventory')
        record = {'id': doc.identifier, 'split': split, 'input_tokens': len(doc.tokens), 'source_stops': len(row['stops']),
                  'certified_stops': len(doc.certified), 'excluded_stops_by_reason': dict(doc.rejected),
                  'certificate_sha256': json_digest([doc.tokens, doc.offsets, doc.certified])}
        summaries.append(record)
        if doc.certified:
            documents.append(doc)

    splits = {}
    for split in SPLITS:
        items = [row for row in summaries if row['split'] == split]
        reasons = Counter()
        for item in items:
            reasons.update(item['excluded_stops_by_reason'])
        splits[split] = {'input_documents': len(items), 'encoded_documents': sum(row['certified_stops'] > 0 for row in items),
                         'input_tokens': sum(row['input_tokens'] for row in items),
                         'encoded_tokens': sum(row['input_tokens'] for row in items if row['certified_stops']),
                         'max_document_tokens': max(row['input_tokens'] for row in items),
                         'source_stops': sum(row['source_stops'] for row in items),
                         'certified_stops': sum(row['certified_stops'] for row in items), 'excluded_stops_by_reason': dict(reasons)}
    result = {'schema': 1, 'tokenizer_sha256': TOKENIZER_SHA256, 'max_document_tokens': MAX_TOKENS,
              'input_sha256': hashes, 'splits': splits, 'documents': summaries,
              'input_documents': len(rows), 'encoded_documents': len(documents),
              'encoded_tokens': sum(len(doc.tokens) for doc in documents),
              'certified_stops': sum(len(doc.certified) for doc in documents)}
    return result, documents, [row for _, row in rows]


def preflight(inputs, tokenizer, output):
    from tokenizers import Tokenizer

    if digest(tokenizer) != TOKENIZER_SHA256:
        raise ValueError('Expected the frozen Qwen tokenizer.json')
    rust = Tokenizer.from_file(str(tokenizer))

    def tokenize(text):
        encoded = rust.encode(text, add_special_tokens=False)
        return encoded.ids, encoded.offsets

    def tokenize_prefixes(texts):
        return [encoded.ids for encoded in rust.encode_batch(texts, add_special_tokens=False)]

    result, _, _ = inventory(inputs, tokenize, tokenize_prefixes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['splits'], ensure_ascii=False, sort_keys=True), flush=True)


def extract(inputs, model, checkpoint, preflight_path, output):
    started = perf_counter()
    import numpy as np
    import torch
    import transformers

    from research.core_features.extract import batches_for, causal_checks, project_tokens, token_control_documents
    from research.gradio_app.checkpoint import BACKBONE, load_checkpoint
    from research.gradio_app.combined import load_anchor_head
    from research.gradio_app.cuda import CudaCausalQwen
    from research.gradio_app.causal_plaba import HIDDEN_WIDTH
    from research.stage2.records import write_jsonl
    from research.stage3.run import encode, sample_gpu, save_json

    if output.exists():
        raise FileExistsError(output)
    if model.name != BACKBONE['revision'] or transformers.__version__ != BACKBONE['transformers']:
        raise ValueError('Expected the registered app Qwen snapshot and Transformers release')
    if any(digest(model / name) != sha for name, sha in BACKBONE['snapshot_sha256'].items() if name.startswith('tokenizer')):
        raise ValueError('The runtime tokenizer differs from the registered Qwen snapshot')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError('Expected Qwen fast tokenization and a defined padding ID')

    def tokenize(text):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return encoded['input_ids'], encoded['offset_mapping']

    def tokenize_prefixes(texts):
        return tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)['input_ids']

    summary, documents, rows = inventory(inputs, tokenize, tokenize_prefixes)
    if summary != json.loads(preflight_path.read_text(encoding='utf-8')):
        raise ValueError('Pinned Transformers must reproduce every Rust preflight coordinate and token certificate')
    if any(not split['encoded_documents'] for split in summary['splits'].values()):
        raise ValueError('At least one article in each split requires a certified source stop')
    candidate = load_checkpoint(checkpoint)
    if candidate.identifier != CANDIDATE:
        raise ValueError('IIRC projection must belong to the registered Gradio checkpoint')
    projection = load_anchor_head(checkpoint / 'anchor')['projection']
    if projection.shape != (HIDDEN_WIDTH, PROJECTION_WIDTH) or projection.dtype != np.float32:
        raise ValueError('Unexpected dimensions or precision for the verified app IIRC projection')

    runtime = CudaCausalQwen(model)
    if runtime.tokenizer.pad_token_id != tokenizer.pad_token_id:
        raise RuntimeError('The serving and extraction tokenizers disagree on padding')
    matrix = torch.as_tensor(projection, device='cuda')
    packed = sorted([(doc.identifier, doc.tokens, doc.certified) for doc in documents], key=lambda item: (len(item[1]), item[0]))
    # Keep the two-sequence padding diagnostic within the production padded-token budget.
    limit = MAX_TOKENS // 2
    bounded_controls = [(key, tokens[:limit], [(end, pos) for end, pos in entries if pos < limit])
                        for key, tokens, entries in packed]
    diagnostics = token_control_documents(tokenizer, rows, bounded_controls, {doc.identifier: doc.offsets for doc in documents})
    checks = causal_checks(runtime.encoder, diagnostics, tokenizer.pad_token_id, torch, token_projection=matrix)
    if len(checks) != 12 or not all(item['passed'] and item['projected_tokens']['passed'] for item in checks):
        raise RuntimeError('Shared FP32 raw or projected-state causality controls failed')

    coordinates = [{'id': doc.identifier, 'stops': [end for end, _ in doc.certified]} for doc in documents]
    pointers = np.cumsum([0, *(len(row['stops']) for row in coordinates)], dtype=np.int64)
    slices = {row['id']: slice(pointers[index], pointers[index + 1]) for index, row in enumerate(coordinates)}
    batches = list(batches_for(packed, 8, MAX_TOKENS))
    largest = max(range(len(batches)), key=lambda index: len(batches[index]) * len(batches[index][-1][1]))
    batches.insert(0, batches.pop(largest))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.gum-extract-') as temporary:
        stage = Path(temporary) / 'output'
        stage.mkdir()
        write_jsonl(stage / 'coordinates.jsonl', coordinates)
        save_json(stage / 'causality.json', {'checks': checks, 'projected_states': 'FP32 app IIRC projection and row-wise L2'})
        vectors = np.lib.format.open_memmap(stage / 'vectors.npy', mode='w+', dtype=np.float16,
                                            shape=(summary['certified_stops'], PROJECTION_WIDTH))
        written, rounding = set(), 0.0
        torch.cuda.reset_peak_memory_stats()
        halt = Event()
        sampler = Thread(target=sample_gpu, args=(stage / 'telemetry.jsonl', halt))
        sampler.start()
        forward_started = perf_counter()
        try:
            for batch_number, batch in enumerate(batches):
                states = encode(runtime.encoder, [tokens for _, tokens, _ in batch], tokenizer.pad_token_id, torch)
                if states.dtype != torch.float32 or states.shape != (len(batch), len(batch[-1][1]), HIDDEN_WIDTH):
                    raise RuntimeError('Expected one batch of intact FP32 Qwen states')
                for index, (identifier, _, entries) in enumerate(batch):
                    positions = [position for _, position in entries]
                    projected = project_tokens(states[index, positions], matrix, torch).cpu().numpy()
                    stored = projected.astype(np.float16)
                    error = float(np.max(np.linalg.norm(stored.astype(np.float32) - projected, axis=1)))
                    if identifier in written or not np.isfinite(error) or error > FP16_TOLERANCE:
                        raise RuntimeError('Duplicate article or unsafe rounding in the delivered projection')
                    vectors[slices[identifier]] = stored
                    written.add(identifier)
                    rounding = max(rounding, error)
                del states
                if batch_number == 0 and torch.cuda.mem_get_info()[0] < 2 * 1024**3:
                    raise RuntimeError('Largest production batch left less than 2 GiB on the L4')
                if batch_number % 32 == 0:
                    print(f'Encoded {len(written)}/{len(documents)} GUM articles', flush=True)
        finally:
            forward_elapsed = perf_counter() - forward_started
            halt.set()
            sampler.join()
        if written != slices.keys():
            raise RuntimeError('Projection output does not cover every certified article')
        vectors.flush()
        del vectors
        outputs = ('coordinates.jsonl', 'vectors.npy', 'causality.json', 'telemetry.jsonl')
        save_json(stage / 'seal.json', {
            **summary, 'preflight_sha256': digest(preflight_path), 'extractor_sha256': digest(Path(__file__)),
            'model': BACKBONE['model'], 'revision': BACKBONE['revision'], 'transformers': transformers.__version__,
            'candidate_id': candidate.identifier, 'anchor_heads_sha256': digest(checkpoint / 'anchor' / 'heads.npz'),
            'projection_sha256': sha256(projection.tobytes(order='C')).hexdigest(), 'projection_width': PROJECTION_WIDTH,
            'compute_dtype': 'float32', 'storage_dtype': 'float16', 'max_storage_relative_l2': rounding,
            'vectors_shape': [summary['certified_stops'], PROJECTION_WIDTH], 'content_forward_per_document': 1,
            'content_batches': len(batches), 'max_batch_tokens': MAX_TOKENS, 'independent_causal_checks': len(checks),
            'gpu': torch.cuda.get_device_name(0), 'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
            'forward_and_readout_s': forward_elapsed, 'elapsed_s': perf_counter() - started,
            'file_sha256': {name: digest(stage / name) for name in outputs},
        })
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='mode', required=True)
    before = subcommands.add_parser('preflight')
    flight = subcommands.add_parser('extract')
    for option in ('inputs', 'tokenizer', 'output'):
        before.add_argument(f'--{option}', type=Path, required=True)
    for option in ('inputs', 'model', 'checkpoint', 'output'):
        flight.add_argument(f'--{option}', type=Path, required=True)
    flight.add_argument('--preflight', dest='preflight_path', type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop('mode')
    (preflight if command == 'preflight' else extract)(**args)
