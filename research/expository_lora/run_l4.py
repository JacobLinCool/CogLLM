"""Fit the matched single-forward expository readouts on one NVIDIA L4."""

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as functional
import transformers
from tokenizers import Tokenizer

from research.backbone_transfer.extract import (
    batches,
    causal_controls,
    checked_digest,
    digest,
    forward,
    load_documents,
    load_language_model,
    read_json,
    stream_rows,
    verified_model,
    write_json,
    write_rows,
)
from research.expository_lora.fit import ALPHA, RANK, LoRANotConverged, fit_fold, project


PROBABILITY_LIMIT = 5e-5


def coordinate_tables(documents, spans):
    starts = {row['id']: row['starts'] for row in stream_rows(spans)}
    if starts.keys() != documents.keys() or any(doc.missing for doc in documents.values()):
        raise ValueError('The coordinate files disagree or contain an unstable prefix')
    by_document, offsets, lengths = {}, {}, []
    for key, doc in documents.items():
        begin = len(lengths)
        for start, stop in zip(starts[key], doc.certified, strict=True):
            if not 0 <= start < stop <= len(doc.text):
                raise ValueError('A sentence length feature has invalid source bounds')
            sentence = doc.text[start:stop]
            lengths.append((len(sentence), len(sentence.split())))
            offsets[key, stop] = len(lengths) - 1
        by_document[key] = slice(begin, len(lengths))
    return by_document, offsets, np.log1p(np.asarray(lengths, dtype=np.float64))


def training_inputs(directory, offsets, train_ids):
    ratings = np.zeros((len(offsets), 2, 5))
    train = np.zeros(len(offsets), dtype=bool)
    grade_documents = set()
    for doc in stream_rows(directory / 'ratings.jsonl'):
        key = doc['id']
        if key in grade_documents or key not in train_ids:
            raise ValueError('A duplicate or held-out document entered the fit')
        grade_documents.add(key)
        for row in doc['ratings']:
            index = offsets[key, row['stop']]
            values = np.asarray([row['ease'], row['understanding']], dtype=np.float64)
            if train[index] or values.shape != (2, 5) or not np.all(values.sum(axis=1) > 0):
                raise ValueError('An invalid or duplicate PLABA rating entered the fit')
            ratings[index] = values / values.sum(axis=1, keepdims=True)
            train[index] = True
    if grade_documents != train_ids or not np.array_equal(train, [key in train_ids for key, _ in offsets]):
        raise ValueError('The sealed PLABA training set does not match the ratings')
    weights = np.zeros(len(offsets))
    alignments = 0
    for row in stream_rows(directory / 'pairs.jsonl'):
        for arm, sign in (('original', -1), ('plain', 1)):
            indices = [offsets[key, stop] for key, stop in row[arm]]
            if not indices or len(indices) != len(set(indices)) or not train[indices].all():
                raise ValueError('A rewrite alignment is empty, duplicated or crosses folds')
            weights[indices] += sign / len(indices)
        alignments += 1
    if not alignments:
        raise ValueError('Paired rewrite supervision is required')
    return ratings, train, weights[train] / alignments, alignments


def row_pick(batch, tensor):
    indices, positions = zip(*[(i, token) for i, doc in enumerate(batch) for token in doc.certified.values()], strict=True)
    return tensor[list(indices), list(positions)].cpu().numpy()


def frozen_cache(encoder, documents, pad_id, slices):
    placement = encoder.layers[-1].mlp.down_proj
    count = sum(len(doc.certified) for doc in documents.values())
    selected = {
        'pre_norm': np.empty((count, placement.out_features), dtype=np.float32),
        'down_input': np.empty((count, placement.in_features), dtype=np.float32),
    }

    def tap(name, subset):
        def capture(_module, args):
            values = row_pick(subset, args[0])
            start = 0
            for doc in subset:
                end = start + len(doc.certified)
                selected[name][slices[doc.identifier]] = values[start:end]
                start = end
        return capture

    for batch in batches(documents):
        with ExitStack() as stack:
            for name, layer in (('pre_norm', encoder.norm), ('down_input', placement)):
                hook = layer.register_forward_pre_hook(tap(name, batch))
                stack.callback(hook.remove)
            forward(encoder, [doc.ids for doc in batch], pad_id, torch)
    if not all(np.isfinite(array).all() for array in selected.values()):
        raise ArithmeticError('Non-finite registered sentence activations')
    return selected


@contextmanager
def adapter(encoder, fitted):
    if fitted is None:
        yield
        return
    a = torch.as_tensor(fitted['lora_a'], dtype=torch.float32, device='cuda')
    b = torch.as_tensor(fitted['lora_b'], dtype=torch.float32, device='cuda')

    def update(_module, args, base):
        return base + functional.linear(functional.linear(args[0], a), b) * (ALPHA / RANK)

    hook = encoder.layers[-1].mlp.down_proj.register_forward_hook(update)
    try:
        yield
    finally:
        hook.remove()


def live_predictions(encoder, documents, pad_id, lengths, slices, fitted, use_lora):
    probabilities = {}
    head = fitted['lora_head' if use_lora else 'frozen_head']
    with adapter(encoder, fitted if use_lora else None):
        for batch in batches(documents):
            hidden = forward(encoder, [doc.ids for doc in batch], pad_id, torch)
            state = row_pick(batch, hidden)
            sentence_lengths = np.concatenate([lengths[slices[doc.identifier]] for doc in batch])
            log_p = project(state, sentence_lengths, head, fitted['length_mean'], fitted['length_scale'])
            start = 0
            for doc in batch:
                end = start + len(doc.certified)
                probabilities[doc.identifier] = log_p[start:end]
                start = end
    ordered = np.concatenate([probabilities[key] for key in documents])
    return probabilities, ordered


@contextmanager
def gpu_monitor(output):
    fields = 'timestamp,name,memory.total,memory.used,utilization.gpu,power.draw'
    with output.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(['nvidia-smi', f'--query-gpu={fields}', '--format=csv', '-l', '1'], stdout=stream)
        try:
            yield
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)


def experiment(model, stimuli, output, protocol_sha256):
    verified_model(model, 'qwen', tokenizer_only=True)
    tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
    documents = load_documents(stimuli, tokenizer)
    assay = read_json(stimuli.with_name('assay.json'))
    plaba_ids, kerwer_ids = (set(assay['corpus'][key]) for key in ('plaba', 'kerwer'))
    if plaba_ids & kerwer_ids or plaba_ids | kerwer_ids != documents.keys():
        raise ValueError('Corpus identities contradict the sealed experiment')
    folds = {int(fold): set(keys) for fold, keys in assay['held_ids'].items()}
    assigned = [key for keys in folds.values() for key in keys]
    if set(folds) != set(range(5)) or set(assigned) != plaba_ids or len(assigned) != len(plaba_ids):
        raise ValueError('PLABA requires exactly five disjoint registered outer folds')
    lengths_by_doc, offsets, lengths = coordinate_tables(documents, stimuli.with_name('spans.jsonl'))
    started = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    encoder, spec = load_language_model(model, 'qwen', torch, transformers)
    cache_start = perf_counter()
    pad = spec['pad_id']
    cached = frozen_cache(encoder, documents, pad, lengths_by_doc)
    cache_elapsed = perf_counter() - cache_start
    norm = encoder.norm.weight.detach().cpu().numpy()
    norm_eps = encoder.norm.eps
    print(f'Frozen activation pass finished in {cache_elapsed:.2f}s', flush=True)

    convergence, forecasts, checkpoints, blocked = {}, [], {}, []
    controls = {'frozen': causal_controls(encoder, documents, tokenizer, pad, torch)}
    equivalence = []
    for fold, prediction_ids in [*sorted(folds.items(), reverse=True), ('full', kerwer_ids)]:
        fit_start = perf_counter()
        ratings, train, contrast, alignments = training_inputs(
            stimuli.parent / 'labels' / str(fold), offsets, plaba_ids - prediction_ids)
        subset = {key: doc for key, doc in documents.items() if key in prediction_ids}
        predict = np.asarray([key in prediction_ids for key, _ in offsets])

        rejected = []
        solvers = ('guarded', 'joint') if fold == 4 else ('joint', 'guarded')
        for solver in solvers:
            def report(event):
                print(json.dumps({'fold': fold, 'solver': solver, **event}, allow_nan=False), flush=True)

            try:
                fitted = fit_fold(cached['pre_norm'], cached['down_input'], norm, norm_eps, lengths, ratings, train,
                                  contrast, predict=predict, device='cuda', progress=report, polishing=solver)
            except LoRANotConverged as error:
                rejected.append({'solver': solver, 'reason': str(error)})
                report({'event': 'solver_not_converged', 'reason': str(error)})
            else:
                break
        else:
            blocked.append({'fold': fold, 'rejected_lora_solvers': rejected})
            continue
        convergence[str(fold)] = {
            'training_sentences': int(train.sum()), 'training_pairs': alignments,
            'seconds': perf_counter() - fit_start, 'solver': solver, 'rejected_lora_solvers': rejected,
            **fitted['diagnostics'],
        }
        print(f"Fold {fold} converged in {convergence[str(fold)]['seconds']:.2f}s", flush=True)
        for key in ('length_mean', 'length_scale', 'frozen_head', 'lora_head', 'lora_a', 'lora_b'):
            checkpoints[f'{fold}_{key}'] = fitted[key]
        for method in ('frozen', 'lora'):
            predictions, ordered = live_predictions(encoder, subset, pad, lengths, lengths_by_doc, fitted, method == 'lora')
            probability_error = float(np.max(np.abs(np.exp(ordered) - np.exp(fitted[method + '_log_probabilities']))))
            if not np.isfinite(probability_error) or probability_error > PROBABILITY_LIMIT:
                raise ArithmeticError(f'{fold}/{method} cached versus live prediction error: {probability_error}')
            equivalence.append({'fold': fold, 'model': method, 'max_absolute_probability_error': probability_error})
            for key, doc in subset.items():
                forecasts.append({
                    'model': method, 'dataset': 'kerwer' if fold == 'full' else 'plaba',
                    'fold': None if fold == 'full' else fold, 'id': key, 'stops': list(doc.certified),
                    'probabilities': np.exp(predictions[key]).tolist(),
                })
        if fold == 'full':
            with adapter(encoder, fitted):
                controls['lora'] = causal_controls(encoder, documents, tokenizer, pad, torch)

    forecasts_path = output / 'predictions.jsonl'
    write_rows(forecasts_path, forecasts)
    write_json(output / 'convergence.json', convergence)
    write_json(output / 'causality.json', {
        **controls, 'inference_forwards_per_document_per_model': 1,
        'cached_live_equivalence': equivalence, 'probability_tolerance': PROBABILITY_LIMIT,
    })
    np.savez_compressed(output / 'checkpoints.npz', **checkpoints)
    duration = perf_counter() - started
    write_json(output / 'seal.json', {
        'protocol_sha256': checked_digest(protocol_sha256), 'assay_sha256': assay['assay_sha256'],
        'stimulus_file_sha256': assay['stimulus_file_sha256'], 'predictions_sha256': digest(forecasts_path),
        'file_sha256': {path.name: digest(path) for path in output.iterdir() if path.name != 'gpu.csv'},
        'fit_converged': not blocked, 'blocked_fit_scopes': blocked,
        'qwen_revision': spec['revision'], 'model': spec['repo'],
        'model_files_sha256': spec['sha256'], 'adapter_placement': 'layers[-1].mlp.down_proj',
        'rank': RANK, 'alpha': ALPHA, 'documents': len(documents), 'stops': len(offsets),
        'input_tokens': sum(len(doc.ids) for doc in documents.values()),
        'gpu': torch.cuda.get_device_name(), 'gpu_bytes': torch.cuda.get_device_properties(0).total_memory,
        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__,
        'seconds': {'model_load': cache_start - started, 'cache': cache_elapsed, 'experiment': duration},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=('qwen',), required=True)
    for name in ('model', 'stimuli', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['USE_HUB_KERNELS'] = 'NO'
    args.output.mkdir()
    with gpu_monitor(args.output / 'gpu.csv'):
        experiment(args.model, args.stimuli, args.output, args.protocol_sha256)


if __name__ == '__main__':
    main()
