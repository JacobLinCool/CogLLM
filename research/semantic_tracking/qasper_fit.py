"""Fit positional and text-conditioned causal evidence hazards on QASPER."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.core_features.fit import canonical_stops, write_rows
from research.stage3.probe import _jsonl
from research.stage3.run import save_json
from .joint_model import interval_nll


WIDTH = 128
PROJECTION_SEED = 43
PENALTY = 0.012
STEPS = 120
BASE = 'position'
BODY = 'body'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unit(values):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        raise ValueError('Expected finite nonzero frozen embeddings')
    return values / norms


def projection():
    rng = np.random.default_rng(PROJECTION_SEED)
    signs = rng.integers(0, 2, size=(2048, WIDTH), dtype=np.int8)
    return (2 * signs.astype(np.float32) - 1) / np.sqrt(np.float32(WIDTH))


def load_states(path, documents):
    expected = {(identifier, end) for identifier, row in documents.items() for end in row['stops']}
    with np.load(path, allow_pickle=False) as data:
        keys = list(zip(data['ids'].tolist(), data['stops'].tolist(), strict=True))
        if set(data.files) != {'ids', 'stops', 'states', 'segment_means'} or len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Readout inventory must match every sealed paragraph and question')
        return (dict(zip(keys, unit(data['states']), strict=True)),
                dict(zip(keys, unit(data['segment_means']), strict=True)))


def assemble(rows, documents, states, means, matrix):
    structural = {}
    questions = {}
    position_features, body_features = [], []
    for row in rows:
        paper, question = row['paper_id'], row['question_id']
        if paper not in structural:
            stops = np.asarray(documents[paper]['stops'])
            paragraph_index = np.arange(1, len(stops) + 1)
            geometry = np.column_stack((np.log1p(paragraph_index), np.log1p(stops),
                                        np.log1p(np.diff(stops, prepend=0))))
            full = np.stack([states[paper, end] for end in stops])
            local = np.stack([means[paper, end] for end in stops])
            structural[paper] = geometry, full, local, unit(full @ matrix), unit(local @ matrix)
        if question not in questions:
            only, = documents[question]['stops']
            q = means[question, only]
            questions[question] = q, unit(q @ matrix)
        geometry, full, local, embedded_full, embedded_local = structural[paper]
        q, embedded_question = questions[question]
        # Question information enters only through interactions with visible paper states.
        body = np.column_stack((embedded_full, embedded_local,
                                embedded_full * embedded_question * np.sqrt(WIDTH),
                                embedded_local * embedded_question * np.sqrt(WIDTH),
                                full @ q, local @ q))
        position_features.append(geometry)
        body_features.append(body)
    return position_features, body_features


def pack(geometries, extras, center, scale):
    length = max(map(len, geometries))
    extra = extras[0].shape[1] if extras is not None else 0
    features = np.zeros((len(geometries), length, 1 + len(center) + extra), dtype=np.float32)
    for i, geometry in enumerate(geometries):
        features[i, :len(geometry), 0] = 1
        features[i, :len(geometry), 1:1 + len(center)] = (geometry - center) / scale / np.sqrt(len(center))
        if extras is not None:
            features[i, :len(geometry), 1 + len(center):] = extras[i]
    return features


def fit(features, bounds, device):
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    lower, upper = (torch.as_tensor(bounds[:, i], device=device) for i in (0, 1))
    weights = torch.zeros(x.shape[-1], dtype=x.dtype, device=device, requires_grad=True)
    with torch.no_grad():
        weights[0] = -float(np.log(bounds[:, 1].mean() - 1)) if bounds[:, 1].mean() > 1 else 0
    optimizer = torch.optim.LBFGS([weights], max_iter=STEPS, line_search_fn='strong_wolfe',
                                  tolerance_grad=1e-7, tolerance_change=1e-9)

    def closure():
        optimizer.zero_grad()
        loss = interval_nll(x @ weights, lower, upper).mean() + PENALTY * weights[1:].square().sum() / 2
        loss.backward()
        return loss

    optimizer.step(closure)
    cost = closure()
    if not torch.isfinite(cost).item():
        raise ArithmeticError('Nonfinite interval hazard fit')
    return weights.detach().cpu().numpy(), {'objective': cost.item(), 'gradient_max': weights.grad.abs().max().item()}


def validate(train, evaluation, documents):
    common = {'paper_id', 'question_id'}
    if not train or not evaluation or any(row.keys() != common | {'lower', 'upper'} for row in train) or any(
        row.keys() != common for row in evaluation
    ):
        raise ValueError('Only the training rows may carry evidence intervals')
    training_papers = {row['paper_id'] for row in train}
    heldout_papers = {row['paper_id'] for row in evaluation}
    if training_papers & heldout_papers or len({row['question_id'] for row in train + evaluation}) != len(train + evaluation):
        raise ValueError('Question and paper identities must be disjoint across splits')
    for row in train + evaluation:
        p, q = row['paper_id'], row['question_id']
        if p == q or len(documents[q]['stops']) != 1:
            raise ValueError('A research question needs one independent frozen encoding')
        if 'lower' in row and not (0 <= row['lower'] and row['upper'] == row['lower'] + 1 <= len(documents[p]['stops'])):
            raise ValueError('The initial assay expects a single annotated body paragraph')
    return training_papers, heldout_papers


def run(stimuli, features, train, evaluation, output):
    started = perf_counter()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
        raise RuntimeError('Expected one NVIDIA L4 for fitting')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda')
    output.mkdir(parents=True, exist_ok=False)
    documents = canonical_stops(stimuli)
    train_rows, eval_rows = list(_jsonl(train)), list(_jsonl(evaluation))
    training_papers, heldout_papers = validate(train_rows, eval_rows, documents)
    extraction = json.loads(features.with_name('extraction.json').read_text())
    if extraction['input_sha256'] != digest(stimuli):
        raise ValueError('Frozen states were extracted from a different stimulus file')
    states, means = load_states(features, documents)
    matrix = projection()
    tr_pos, tr_body = assemble(train_rows, documents, states, means, matrix)
    ev_pos, ev_body = assemble(eval_rows, documents, states, means, matrix)
    geometry = np.concatenate([rows[:row['upper']] for row, rows in zip(train_rows, tr_pos, strict=True)])
    center, scale = geometry.mean(0), np.maximum(0.1, geometry.std(0))
    bounds = np.asarray([[row['lower'], row['upper']] for row in train_rows], dtype=np.int64)
    result = [{**row, 'stops': documents[row['paper_id']]['stops']} for row in eval_rows]
    audit, checkpoints = {}, {}
    for name, first, second in ((BASE, None, None), (BODY, tr_body, ev_body)):
        x = pack(tr_pos, first, center, scale)
        y = pack(ev_pos, second, center, scale)
        weights, audit[name] = fit(x, bounds, device)
        checkpoints[name] = weights
        logits = y.astype(np.float64) @ weights.astype(np.float64)
        hazards = np.exp(-np.logaddexp(0, -logits))
        for row, estimates in zip(result, hazards, strict=True):
            row[name] = estimates[:len(row['stops'])].tolist()
        print(f'Fitted QASPER {name}', flush=True)
    predictions = output / 'predictions.jsonl'
    write_rows(predictions, result)
    np.savez(output / 'heads.npz', **checkpoints, projection=matrix, position_mean=center, position_scale=scale)
    save_json(output / 'fit-manifest.json', {
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'predictions_sha256': digest(predictions),
        'train_questions': len(train_rows), 'heldout_questions': len(eval_rows),
        'train_papers': len(training_papers), 'heldout_papers': len(heldout_papers),
        'methods': audit, 'fit_wall_s': perf_counter() - started, 'max_lbfgs_steps': STEPS, 'penalty': PENALTY,
        'projection_seed': PROJECTION_SEED, 'position_features': ['body paragraph index', 'visible characters', 'segment characters'],
        'question_features': 'fixed pooled question states interact only with the visible paper prefix',
        'feature_file_sha256': digest(features), 'source_sha256': {
            'stimuli': digest(stimuli), 'train': digest(train), 'eval': digest(evaluation),
        },
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('stimuli', 'features', 'train', 'evaluation', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    run(**vars(parser.parse_args()))
