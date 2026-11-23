"""Extract independent frozen-app vectors for sealed Beukers readout inputs."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter

from research.semantic_tracking.gum_bridge_extract import (
    CANDIDATE, FP16_TOLERANCE, MAX_TOKENS, PROJECTION_WIDTH, TOKENIZER_SHA256, certify, digest, json_digest,
)


STIMULUS_FIELDS = {'id', 'text', 'stops'}
PREFLIGHT_FIELDS = {'schema', 'stimuli_sha256', 'tokenizer_sha256', 'input_documents', 'encoded_documents',
                    'input_tokens', 'max_input_tokens', 'requested_stops', 'certified_stops', 'documents'}
DOCUMENT_FIELDS = {'id', 'input_tokens', 'stops', 'excluded_stops_by_reason', 'certificate_sha256'}
MAX_CORPUS_TOKENS = 350_000
MAX_CERTIFIED_STOPS = 150_000


def read_stimuli(path):
    with path.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    if not rows:
        raise ValueError('Expected grade-free Beukers stimuli')
    keys, rendered = set(), set()
    for row in rows:
        if not isinstance(row, dict) or row.keys() != STIMULUS_FIELDS:
            raise ValueError('Beukers inference rows may contain only id, text, and stops')
        key, text, stops = (row[name] for name in ('id', 'text', 'stops'))
        if (type(key) is not str or not key or key in keys or type(text) is not str or not text.strip()
                or text in rendered or type(stops) is not list or not stops
                or any(type(stop) is not int or not 0 < stop <= len(text) for stop in stops)
                or stops != sorted(set(stops))):
            raise ValueError('Invalid or repeated Beukers text, identity or Unicode endpoint')
        keys.add(key)
        rendered.add(text)
    return rows


def inventory(stimuli, tokenize, tokenize_prefixes):
    rows = read_stimuli(stimuli)
    documents = [certify(row, tokenize, tokenize_prefixes) for row in rows]
    details = [{'id': doc.identifier, 'input_tokens': len(doc.tokens), 'stops': [end for end, _ in doc.certified],
                'excluded_stops_by_reason': dict(doc.rejected),
                'certificate_sha256': json_digest([doc.tokens, doc.offsets, doc.certified])} for doc in documents]
    summary = {
        'schema': 1, 'stimuli_sha256': digest(stimuli), 'tokenizer_sha256': TOKENIZER_SHA256, 'input_documents': len(rows),
        'encoded_documents': sum(bool(doc.certified) for doc in documents), 'input_tokens': sum(len(doc.tokens) for doc in documents),
        'max_input_tokens': max(len(doc.tokens) for doc in documents), 'requested_stops': sum(len(row['stops']) for row in rows),
        'certified_stops': sum(len(doc.certified) for doc in documents), 'documents': details,
    }
    return summary, rows, documents


def validate_preflight(summary, rows, stimulus_sha256):
    if (summary.keys() != PREFLIGHT_FIELDS or summary['schema'] != 1 or summary['stimuli_sha256'] != stimulus_sha256
            or summary['tokenizer_sha256'] != TOKENIZER_SHA256
            or not summary['input_documents'] == summary['encoded_documents'] == len(rows)
            or not 0 < summary['input_tokens'] <= MAX_CORPUS_TOKENS
            or not 0 < summary['max_input_tokens'] <= MAX_TOKENS
            or not 0 < summary['certified_stops'] == summary['requested_stops'] <= MAX_CERTIFIED_STOPS):
        raise ValueError('The Beukers inputs exceed their budget or contain uncertified readout stops')
    for doc, row in zip(summary['documents'], rows, strict=True):
        if (doc.keys() != DOCUMENT_FIELDS or doc['id'] != row['id'] or doc['stops'] != row['stops']
                or doc['excluded_stops_by_reason']
                or type(doc['input_tokens']) is not int or not 0 < doc['input_tokens'] <= MAX_TOKENS
                or type(doc['certificate_sha256']) is not str or len(doc['certificate_sha256']) != 64
                or any(character not in '0123456789abcdef' for character in doc['certificate_sha256'])):
            raise ValueError('The exact Beukers input and every declared source stop must be certified')
    token_counts = [doc['input_tokens'] for doc in summary['documents']]
    if (summary['input_tokens'] != sum(token_counts) or summary['max_input_tokens'] != max(token_counts)
            or summary['certified_stops'] != sum(len(row['stops']) for row in rows)):
        raise ValueError('Beukers preflight scale does not reconcile with its per-text certificates')


def preflight(stimuli, tokenizer, output):
    from tokenizers import Tokenizer

    if digest(tokenizer) != TOKENIZER_SHA256:
        raise ValueError('Expected the frozen Qwen tokenizer.json')
    rust = Tokenizer.from_file(str(tokenizer))

    def tokenize(text):
        encoded = rust.encode(text, add_special_tokens=False)
        return encoded.ids, encoded.offsets

    def prefixes(texts):
        return [encoded.ids for encoded in rust.encode_batch(texts, add_special_tokens=False)]

    summary, rows, _ = inventory(stimuli, tokenize, prefixes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'documents'}, sort_keys=True), flush=True)
    validate_preflight(summary, rows, digest(stimuli))


def extract(stimuli, model, checkpoint, preflight_path, assay_sha256, protocol_sha256, output, terminal=False,
            raw_stops=False):
    started = perf_counter()
    import numpy as np
    import torch
    import transformers

    from research.core_features.extract import batches_for, causal_checks, project_tokens, token_control_documents
    from research.gradio_app.causal_plaba import HIDDEN_WIDTH
    from research.gradio_app.checkpoint import BACKBONE, load_checkpoint
    from research.gradio_app.combined import load_anchor_head
    from research.gradio_app.cuda import CudaCausalQwen
    from research.stage2.records import write_jsonl
    from research.stage3.run import encode, sample_gpu, save_json

    if output.exists():
        raise FileExistsError(output)
    if any(len(value) != 64 or any(character not in '0123456789abcdef' for character in value)
           for value in (assay_sha256, protocol_sha256)):
        raise ValueError('Expected sealed assay and protocol SHA-256 values')
    if model.name != BACKBONE['revision'] or transformers.__version__ != BACKBONE['transformers']:
        raise ValueError('Expected the registered app model snapshot and Transformers release')
    if any(digest(model / name) != sha for name, sha in BACKBONE['snapshot_sha256'].items() if name.startswith('tokenizer')):
        raise ValueError('The runtime tokenizer differs from the registered snapshot')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError('Expected the pinned fast tokenizer with its padding token')

    def tokenize(text):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return encoded['input_ids'], encoded['offset_mapping']

    def prefixes(texts):
        return tokenizer(texts, add_special_tokens=False, padding=False, truncation=False,
                         return_attention_mask=False, return_token_type_ids=False)['input_ids']

    summary, rows, documents = inventory(stimuli, tokenize, prefixes)
    if summary != json.loads(preflight_path.read_text(encoding='utf-8')):
        raise ValueError('Pinned Transformers failed to reproduce the grade-free Rust certificates')
    validate_preflight(summary, rows, digest(stimuli))
    if terminal and any(doc.certified[-1] != (len(row['text']), len(doc.tokens) - 1)
                        for row, doc in zip(rows, documents, strict=True)):
        raise ValueError('Terminal features require an exact certificate for each full rendered input')
    candidate = load_checkpoint(checkpoint)
    if candidate.identifier != CANDIDATE:
        raise ValueError('Expected the registered Gradio candidate and its IIRC projection')
    projection = load_anchor_head(checkpoint / 'anchor')['projection']
    if projection.dtype != np.float32 or projection.shape != (HIDDEN_WIDTH, PROJECTION_WIDTH):
        raise ValueError('The app IIRC projection does not have the registered FP32 dimensions')

    runtime = CudaCausalQwen(model)
    matrix = torch.as_tensor(projection, device='cuda')
    packed = sorted([(doc.identifier, doc.tokens, doc.certified) for doc in documents], key=lambda row: (len(row[1]), row[0]))
    limit = MAX_TOKENS // 2
    controls = [(key, tokens[:limit], stops) for key, tokens, stops in packed]
    diagnostics = token_control_documents(tokenizer, rows, controls, {doc.identifier: doc.offsets for doc in documents})
    checks = causal_checks(runtime.encoder, diagnostics, tokenizer.pad_token_id, torch, token_projection=matrix)
    if len(checks) != 12 or not all(row['passed'] and row['projected_tokens']['passed'] for row in checks):
        raise RuntimeError('Independent FP32 causal checks failed for raw or projected states')

    coordinates = [{'id': doc.identifier, 'stops': [end for end, _ in doc.certified]} for doc in documents]
    offsets = np.cumsum([0, *(len(row['stops']) for row in coordinates)], dtype=np.int64)
    slices = {row['id']: slice(offsets[index], offsets[index + 1]) for index, row in enumerate(coordinates)}
    terminal_rows = {row['id']: index for index, row in enumerate(coordinates)} if terminal else None
    batches = list(batches_for(packed, 32, MAX_TOKENS))
    largest = max(range(len(batches)), key=lambda i: len(batches[i]) * len(batches[i][-1][1]))
    batches.insert(0, batches.pop(largest))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.beukers-extract-') as temporary:
        stage = Path(temporary) / 'outputs'
        stage.mkdir()
        write_jsonl(stage / 'coordinates.jsonl', coordinates)
        save_json(stage / 'causality.json', {'checks': checks, 'projection': 'Frozen app IIRC FP32 L2-normalized token states'})
        shape = (summary['certified_stops'], PROJECTION_WIDTH)
        vectors = np.lib.format.open_memmap(stage / 'vectors.npy', mode='w+', dtype=np.float16, shape=shape)
        states_shape = (summary['certified_stops'], HIDDEN_WIDTH)
        stop_states = (np.lib.format.open_memmap(stage / 'states.npy', mode='w+', dtype=np.float32,
                                                shape=states_shape) if raw_stops else None)
        terminal_shape = (len(coordinates), HIDDEN_WIDTH)
        terminal_states = (np.lib.format.open_memmap(stage / 'terminal_states.npy', mode='w+', dtype=np.float32,
                                                    shape=terminal_shape) if terminal else None)
        written, maximum_error = set(), 0.0
        torch.cuda.reset_peak_memory_stats()
        halt = Event()
        sampler = Thread(target=sample_gpu, args=(stage / 'telemetry.jsonl', halt))
        sampler.start()
        before_forward = perf_counter()
        try:
            for batch_number, batch in enumerate(batches):
                states = encode(runtime.encoder, [tokens for _, tokens, _ in batch], tokenizer.pad_token_id, torch)
                if states.dtype != torch.float32 or states.shape != (len(batch), len(batch[-1][1]), HIDDEN_WIDTH):
                    raise RuntimeError('Expected one exact FP32 content pass per deduplicated Beukers input')
                locations = [(index, position) for index, (_, _, entries) in enumerate(batch) for _, position in entries]
                rows, positions = zip(*locations, strict=True)
                hidden = states[rows, positions]
                values = project_tokens(hidden, matrix, torch).cpu().numpy()
                raw_batch = hidden.cpu().numpy() if raw_stops else None
                del hidden
                if raw_batch is not None and (not np.isfinite(raw_batch).all() or not np.any(raw_batch, axis=1).all()):
                    raise RuntimeError('Invalid FP32 hidden state at a certified stop')
                stored = values.astype(np.float16)
                error = float(np.max(np.linalg.norm(stored.astype(np.float32) - values, axis=1)))
                if not np.isfinite(error) or error > FP16_TOLERANCE:
                    raise RuntimeError('Excessive FP16 storage error in a Beukers state batch')
                if terminal_states is not None:
                    ends = [len(tokens) - 1 for _, tokens, _ in batch]
                    endings = states[list(range(len(batch))), ends].cpu().numpy()
                    if not np.isfinite(endings).all() or not np.any(endings, axis=1).all():
                        raise RuntimeError('Invalid FP32 terminal hidden state')
                    terminal_states[[terminal_rows[key] for key, _, _ in batch]] = endings
                start = 0
                for key, _, entries in batch:
                    if key in written:
                        raise RuntimeError('Duplicate Beukers input forward')
                    end = start + len(entries)
                    vectors[slices[key]] = stored[start:end]
                    if stop_states is not None:
                        stop_states[slices[key]] = raw_batch[start:end]
                    start = end
                    written.add(key)
                maximum_error = max(maximum_error, error)
                del states, raw_batch
                if batch_number == 0 and torch.cuda.mem_get_info()[0] < 2 * 1024**3:
                    raise RuntimeError('Largest Beukers batch leaves less than 2 GiB of L4 headroom')
        finally:
            forward_time = perf_counter() - before_forward
            halt.set()
            sampler.join()
        if written != slices.keys():
            raise RuntimeError('Feature rows do not exactly cover the certified Beukers stimuli')
        vectors.flush()
        del vectors
        artifacts = ('coordinates.jsonl', 'vectors.npy', 'causality.json', 'telemetry.jsonl')
        state_receipt = {}
        if stop_states is not None:
            stop_states.flush()
            del stop_states
            artifacts += ('states.npy',)
            state_receipt.update(raw_stops=True, states_shape=list(states_shape), states_dtype='float32')
        if terminal_states is not None:
            terminal_states.flush()
            del terminal_states
            artifacts += ('terminal_states.npy',)
            state_receipt.update(terminal_states_shape=list(terminal_shape), terminal_state_dtype='float32')
        save_json(stage / 'seal.json', {
            **summary, 'preflight_sha256': digest(preflight_path), 'source_code_sha256': digest(Path(__file__)),
            'assay_sha256': assay_sha256, 'protocol_sha256': protocol_sha256, 'candidate_id': candidate.identifier,
            'model': BACKBONE['model'], 'revision': BACKBONE['revision'], 'transformers': transformers.__version__,
            'anchor_heads_sha256': digest(checkpoint / 'anchor' / 'heads.npz'),
            'projection_sha256': sha256(projection.tobytes(order='C')).hexdigest(), 'projection_width': PROJECTION_WIDTH,
            'compute_dtype': 'float32', 'storage_dtype': 'float16', 'vectors_shape': list(shape),
            'max_storage_relative_l2': maximum_error, 'content_forward_per_document': 1, 'content_batches': len(batches),
            'independent_causal_checks': len(checks), 'forward_and_readout_s': forward_time, 'elapsed_s': perf_counter() - started,
            'gpu': torch.cuda.get_device_name(0), 'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
            'file_sha256': {name: digest(stage / name) for name in artifacts},
            **state_receipt,
        })
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='mode', required=True)
    before = subparsers.add_parser('preflight')
    flight = subparsers.add_parser('extract')
    for option in ('stimuli', 'tokenizer', 'output'):
        before.add_argument(f'--{option}', type=Path, required=True)
    for option in ('stimuli', 'model', 'checkpoint', 'output'):
        flight.add_argument(f'--{option}', type=Path, required=True)
    flight.add_argument('--preflight', dest='preflight_path', type=Path, required=True)
    flight.add_argument('--assay-sha256', required=True)
    flight.add_argument('--protocol-sha256', required=True)
    flight.add_argument('--terminal', action='store_true')
    flight.add_argument('--raw-stops', action='store_true', help='export FP32 final-layer states at all certified stops')
    args = vars(parser.parse_args())
    command = args.pop('mode')
    (preflight if command == 'preflight' else extract)(**args)
