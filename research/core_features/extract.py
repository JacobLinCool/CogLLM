"""Read certified character stops in one frozen causal forward per text."""

import argparse
import hashlib
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

from research.stage3.probe import _jsonl
from research.stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION, difference, encode, sample_gpu, save_json


def plan(tokenizer, rows, *, token_offsets=None, batch_prefixes=False):
    documents = []
    rejected = []
    seen = set()
    for row in rows:
        key, text, stops = row['id'], row['text'], row['stops']
        if (row.keys() != {'id', 'text', 'stops'} or not isinstance(key, str) or not key or key in seen
                or not isinstance(text, str) or not text.strip() or not stops
                or sorted(set(stops)) != stops or any(type(end) is not int or not 0 < end <= len(text) for end in stops)):
            raise ValueError(f'Invalid stop inventory: {key}')
        seen.add(key)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens, offsets = encoded['input_ids'], encoded['offset_mapping']
        # Byte-split Unicode pieces can share an offset. Read after the last such token.
        positions = {end: pos for pos, (start, end) in enumerate(offsets) if start < end}
        candidates = []
        for end in stops:
            position = positions.get(end)
            if position is None:
                rejected.append({'id': key, 'stop': end, 'reason': 'no_token_ends_at_stop'})
                continue
            candidates.append((end, position))
        if batch_prefixes and candidates:
            prefixes = tokenizer([text[:end] for end, _ in candidates], add_special_tokens=False)['input_ids']
        else:
            prefixes = (tokenizer(text[:end], add_special_tokens=False)['input_ids'] for end, _ in candidates)
        accepted = []
        for (end, position), prefix in zip(candidates, prefixes, strict=True):
            if tokens[:position + 1] != prefix:
                rejected.append({'id': key, 'stop': end, 'reason': 'independent_prefix_tokenization_differs'})
                continue
            accepted.append((end, position))
        if accepted:
            documents.append((key, tokens, accepted))
            if token_offsets is not None:
                token_offsets[key] = offsets
    if not documents:
        raise ValueError('No exact causal stops')
    return sorted(documents, key=lambda row: (len(row[1]), row[0])), rejected


def project_tokens(hidden, matrix, torch):
    projected = hidden @ matrix
    norms = torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
    if not (torch.isfinite(norms) & (norms > 0)).all().item():
        raise ValueError('Expected finite nonzero projected token states')
    return projected / norms


def causal_checks(language, documents, pad_id, torch, *, token_projection=None):
    eligible = [row for row in documents if sum(p >= 2 and p + 2 < len(row[1]) for _, p in row[2]) >= 3]
    if len(eligible) < 3:
        raise ValueError('Expected three texts with annotated interior stops')
    checks = []
    control = documents[-1][1]
    for index in (len(eligible) // 4, len(eligible) // 2, 3 * len(eligible) // 4):
        key, tokens, stops = eligible[index]
        interior = [(end, p) for end, p in stops if p >= 2 and p + 2 < len(tokens)]
        end, position = interior[len(interior) // 2]
        count = position + 1
        prefix = tokens[:count]
        # Compare the recorded state with both a truncated and a changed visible sequence.
        alternate = prefix + control[:len(tokens) - count]
        if alternate == tokens:
            raise ValueError('Suffix perturbation failed to change the input')
        original = encode(language, [tokens], pad_id, torch)[0]
        changed = encode(language, [alternate], pad_id, torch)[0]
        truncated = encode(language, [prefix], pad_id, torch)[0]
        padded = encode(language, [prefix, tokens], pad_id, torch)[0, :count]
        projected_original = project_tokens(original, token_projection, torch) if token_projection is not None else None
        for name, actual, expected, maximum in (
            ('substring', truncated, original[:count], 0.002),
            ('right_padding', padded, original[:count], 0.002),
            ('suffix_replacement', changed[:count], original[:count], 0.002),
            ('changed_suffix_positive_control', changed[count:], original[count:], None),
        ):
            metrics = difference(expected, actual)
            error = metrics['max_token_relative_rms']
            passed = error <= maximum if maximum is not None else error > 0.02
            item = {'id': key, 'stop': end, 'comparison': name, 'passed': passed, **metrics}
            if projected_original is not None:
                projected_expected = projected_original[:count] if maximum is not None else projected_original[count:]
                projected_actual = project_tokens(actual, token_projection, torch)
                projected_metrics = difference(projected_expected, projected_actual)
                projected_error = projected_metrics['max_token_relative_rms']
                item['projected_tokens'] = {
                    'passed': projected_error <= maximum if maximum is not None else projected_error > 0.02,
                    **projected_metrics,
                }
            checks.append(item)
    return checks


def batches_for(documents, maximum_examples, maximum_tokens):
    """Group sorted inputs by padded token count without splitting any sequence."""
    pending = []
    for item in documents:
        length = len(item[1])
        if maximum_tokens is not None and length > maximum_tokens:
            raise ValueError(f'Input exceeds the batch token budget: {item[0]}')
        if pending and (len(pending) >= maximum_examples or (maximum_tokens is not None
                        and (len(pending) + 1) * length > maximum_tokens)):
            yield pending
            pending = []
        pending.append(item)
    if pending:
        yield pending


def token_control_documents(tokenizer, rows, documents, offsets):
    """Find certified interior prefixes for a diagnostic batch without observed interior stops."""
    eligible = [row for row in documents if len(row[1]) >= 8]
    if len(eligible) < 3:
        raise ValueError('Expected three texts long enough to check the token timeline')
    texts = {row['id']: row['text'] for row in rows}
    samples = []
    for index in (len(eligible) // 4, len(eligible) // 2, 3 * len(eligible) // 4):
        key, tokens, _ = eligible[index]
        positions = {end: pos for pos, (start, end) in enumerate(offsets[key]) if start < end < len(texts[key])}
        candidates = [(end, pos) for end, pos in positions.items() if 2 <= pos < len(tokens) - 2]
        stops = []
        for end, pos in sorted(candidates, key=lambda item: (abs(item[1] - len(tokens) / 2), item[1])):
            prefix = tokenizer(texts[key][:end], add_special_tokens=False)['input_ids']
            if tokens[:pos + 1] == prefix:
                stops.append((end, pos))
                if len(stops) == 3:
                    break
        if len(stops) != 3:
            raise ValueError(f'Cannot certify three diagnostic prefixes in {key!r}')
        samples.append((key, tokens, sorted(stops)))
    return samples


def run(stimuli, output, *, max_batch_tokens=None, max_batch_examples=8, segment_means=False,
        projected_tokens=None, projected_only=False, certified_token_controls=False):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if projected_only and (projected_tokens is None or segment_means):
        raise ValueError('Dense-only extraction needs a token projection and cannot include segment means')
    if max_batch_examples < 1:
        raise ValueError('Expected a positive maximum batch size')
    if transformers.__version__ != TRANSFORMERS_VERSION or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected pinned Transformers and one CUDA device')
    gpu = torch.cuda.get_device_properties(0)
    if gpu.name != 'NVIDIA L4' or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError(f'Expected L4 with native BF16, got {gpu.name}')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    timer = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise RuntimeError('Expected the pinned fast tokenizer and a padding token')
    rows = list(_jsonl(stimuli))
    token_offsets = {} if projected_tokens is not None or certified_token_controls else None
    documents, rejected = plan(tokenizer, rows, token_offsets=token_offsets)
    if projected_tokens is not None and (rejected or len(documents) != len(rows)):
        raise ValueError('Dense token extraction requires every declared stop to be certified')
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation='sdpa',
        use_safetensors=True, output_loading_info=True,
    )
    if any(loading.values()) or next(model.model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError(f'Pinned BF16 checkpoint did not load exactly: {loading}')
    pipeline = model.model.eval().requires_grad_(False).to(device='cuda', dtype=torch.float32)
    del model
    setup = perf_counter() - timer
    gpu_projection = None
    if projected_tokens is not None:
        _, projection = projected_tokens
        projection = np.asarray(projection, dtype=np.float32)
        if (projection.ndim != 2 or projection.shape[0] != pipeline.config.hidden_size
                or not 0 < projection.shape[1] <= projection.shape[0] or not np.isfinite(projection).all()):
            raise ValueError('Projection must match the frozen state width and have finite elements')
        gpu_projection = torch.as_tensor(projection, device='cuda')
    check_documents = (token_control_documents(tokenizer, rows, documents, token_offsets)
                       if projected_only or certified_token_controls else documents)
    if max_batch_tokens is not None:
        limit = max_batch_tokens // 2
        check_documents = [(key, tokens[:limit], [(end, pos) for end, pos in entries if pos < limit])
                           for key, tokens, entries in check_documents]
    checks = causal_checks(pipeline, check_documents, tokenizer.pad_token_id, torch, token_projection=gpu_projection)
    causality = {'checks': checks}
    if certified_token_controls:
        causality['control_stops'] = 'independently certified interior boundaries of the input text'
    save_json(output / 'causality.json', causality)
    if not all(item['passed'] and item.get('projected_tokens', {'passed': True})['passed'] for item in checks):
        raise RuntimeError('Causal stop checks failed')

    count = sum(len(row[2]) for row in documents)
    states = None if projected_only else np.empty((count, pipeline.config.hidden_size), np.float32)
    means = np.empty_like(states) if segment_means else None
    token_slices = {}
    dense = None
    if projected_tokens is not None:
        token_ids, _ = projected_tokens
        picked = [(key, tokens) for key, tokens, _ in documents if key in token_ids]
        if not picked or {key for key, _ in picked} != token_ids:
            raise ValueError('Dense token extraction needs a nonempty set of known document IDs')
        if projected_only and len(picked) != len(documents):
            raise ValueError('Dense-only extraction needs projected states for every input text')
        indptr = np.cumsum([0, *(len(tokens) for _, tokens in picked)], dtype=np.int64)
        token_slices = {key: slice(indptr[i], indptr[i + 1]) for i, (key, _) in enumerate(picked)}
        dense = {
            'ids': np.asarray([key for key, _ in picked]),
            'indptr': indptr,
            'offsets': np.concatenate([np.asarray(token_offsets[key], dtype=np.int32) for key, _ in picked]),
        }
        if not projected_only:
            dense['stop_counts'] = np.empty(count, np.int32)
        dense.update(states=np.empty((indptr[-1], projection.shape[1]), np.float32), projection=projection)
    ids, stops = [], []
    elapsed = 0.0
    packed = list(batches_for(documents, max_batch_examples, max_batch_tokens))
    if max_batch_tokens is not None:
        # Pilot the largest resident batch rather than using short questions as the memory check.
        largest = max(range(len(packed)), key=lambda i: len(packed[i]) * len(packed[i][-1][1]))
        packed.insert(0, packed.pop(largest))
    torch.cuda.reset_peak_memory_stats()
    halt = Event()
    sampler = Thread(target=sample_gpu, args=(output / 'telemetry.jsonl', halt))
    sampler.start()
    try:
        completed = 0
        for batch_number, samples in enumerate(packed):
            started = perf_counter()
            sequences = [tokens for _, tokens, _ in samples]
            hidden = encode(pipeline, sequences, tokenizer.pad_token_id, torch)
            positions = [(i, pos) for i, (_, _, entries) in enumerate(samples) for _, pos in entries]
            if states is not None:
                vectors = hidden[[i for i, _ in positions], [pos for _, pos in positions]].cpu().numpy()
                if vectors.dtype != np.float32 or not np.isfinite(vectors).all() or not np.any(vectors, axis=1).all():
                    raise ValueError('Expected finite nonzero float32 features')
                states[len(ids):len(ids) + len(vectors)] = vectors
            if dense is not None:
                if not projected_only:
                    dense['stop_counts'][len(ids):len(ids) + len(positions)] = [pos + 1 for _, pos in positions]
                for i, (key, tokens, _) in enumerate(samples):
                    if key not in token_slices:
                        continue
                    dense['states'][token_slices[key]] = project_tokens(
                        hidden[i, :len(tokens)], gpu_projection, torch).cpu().numpy()
            if means is not None:
                pooled = []
                for i, (_, _, entries) in enumerate(samples):
                    start = 0
                    for _, position in entries:
                        pooled.append(hidden[i, start:position + 1].mean(dim=0))
                        start = position + 1
                pooled = torch.stack(pooled)
                if not torch.isfinite(pooled).all().item():
                    raise ValueError('Expected finite segment means')
                means[len(ids):len(ids) + len(pooled)] = pooled.cpu().numpy()
            ids.extend(key for key, _, entries in samples for _ in entries)
            stops.extend(end for _, _, entries in samples for end, _ in entries)
            elapsed += perf_counter() - started
            completed += len(samples)
            if batch_number % 32 == 0:
                print(f'Encoded {completed}/{len(documents)} texts', flush=True)
            if batch_number == 0 and torch.cuda.mem_get_info()[0] < 2 * 1024**3:
                raise RuntimeError('L4 pilot left less than 2 GiB device headroom')
            del hidden
    finally:
        halt.set()
        sampler.join()
    if len(ids) != count:
        raise RuntimeError('Incomplete causal features')
    if states is not None:
        arrays = {'ids': np.asarray(ids), 'stops': np.asarray(stops), 'states': states}
        if means is not None:
            arrays['segment_means'] = means
        np.savez(output / 'features.npz', **arrays)
    tokens = sum(len(row[1]) for row in documents)
    manifest = {
        'model': MODEL, 'revision': REVISION, 'transformers': transformers.__version__,
        'gpu': gpu.name, 'cuda': torch.version.cuda, 'torch': torch.__version__, 'dtype': 'float32',
        'input_sha256': hashlib.sha256(stimuli.read_bytes()).hexdigest(),
        'input_texts': len(rows), 'encoded_texts': len(documents), 'unique_certified_stops': count,
        'excluded_stops': rejected, 'tokens': tokens, 'forward_and_readout_s': elapsed,
        'throughput_tokens_s': tokens / elapsed, 'load_tokenize_setup_s': setup,
        'approx_total_s': perf_counter() - timer, 'batches': len(packed),
        'max_batch_tokens': max_batch_tokens, 'segment_means': segment_means,
        'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
    }
    if projected_only:
        manifest['max_batch_examples'] = max_batch_examples
        manifest['padded_tokens'] = sum(len(batch) * len(batch[-1][1]) for batch in packed)
    if dense is not None:
        dense_path = output / 'token-features.npz'
        np.savez(dense_path, **dense)
        with dense_path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest['projected_tokens'] = {
            'file': dense_path.name, 'sha256': digest, 'documents': len(token_slices),
            'tokens': len(dense['states']), 'components': dense['states'].shape[1],
            'normalization': 'float32 row-wise L2 after fixed projection',
            'offset_units': 'half-open Unicode character indices in the original rendered stimulus',
        }
        if not projected_only:
            manifest['projected_tokens']['stop_counts'] = (
                'aligned with features.npz rows, exclusive upper token indices per document'
            )
    save_json(output / 'extraction.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stimuli', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--certified-token-controls', action='store_true',
                        help='Test causal invariants at separate, independently certified interior token stops')
    args = parser.parse_args()
    run(args.stimuli, args.output, certified_token_controls=args.certified_token_controls)
