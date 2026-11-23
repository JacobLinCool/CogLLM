"""Decode authored stimulus conditions from original Beukers frozen states."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from research.semantic_tracking.beukers_background import CHECKPOINT, digest, read_rows, read_vectors, vectors_for
from research.semantic_tracking.beukers_prepare import OPTIONS, PREFIX_NODES, SENTENCES, text_layout


SALT = 'cogllm-beukers-state-v1:'
RIDGE = 0.01
SCALES = ('cafe_cue', 'current_event', 'current_event_depth_1', 'current_event_depth_2', 'candidate_event')
TRIAL_FIELDS = {'id', 'article_id', 'left_id', 'right_id'}


def rendered_label(sentence, name, labels):
    matches = [label for label in labels if sentence == SENTENCES[label].format(name=name)]
    if len(matches) != 1:
        raise ValueError('Each observed stimulus sentence must map uniquely to an authored node')
    return matches[0]


def examples_from_original(maps, layouts, indices, states):
    cases = {condition: {} for condition in SCALES}
    if (any(row.keys() != TRIAL_FIELDS for row in maps) or len({row['id'] for row in maps}) != len(maps)
            or len({key for row in maps for key in (row['article_id'], row['left_id'], row['right_id'])}) != len(layouts)):
        raise ValueError('The diagnostic requires exactly the unique original fit stimuli and mapping rows')
    _, suffix = SENTENCES['BEGIN'].split('{name}')
    triples = sorted({(row['article_id'], row['left_id'], row['right_id']) for row in maps})
    cache = {}

    def vectors(key):
        if key not in cache:
            cache[key] = vectors_for(indices, states, key)
        return cache[key]

    for article, left, right in triples:
        text = layouts[article]['text']
        sentences = text.split('\n')
        if len(sentences) not in (3, 4) or not sentences[0].endswith(suffix) or ' ' in sentences[0].removesuffix(suffix):
            raise ValueError('Unexpected fictional-name or visible-prefix format')
        name = sentences[0].removesuffix(suffix)
        if not name:
            raise ValueError('Missing fictional name')
        depth = len(sentences) - 2
        cafe = rendered_label(sentences[1], name, ('deep', 'jungle'))
        event = rendered_label(sentences[-1], name, PREFIX_NODES[depth])
        if depth == 2:
            rendered_label(sentences[2], name, PREFIX_NODES[1])
        if layouts[article] != text_layout(sentences) or len(vectors(article)) != len(sentences):
            raise ValueError('Article certificate layout must be the original authored sentence sequence')
        # The first text ID in sorted order deterministically represents each identical two-sentence prefix.
        cases['cafe_cue'].setdefault((name, cafe), (name, cafe, vectors(article)[1]))
        cases['current_event'].setdefault(article, (name, event, vectors(article)[-1]))
        cases[f'current_event_depth_{depth}'].setdefault(article, (name, event, vectors(article)[-1]))
        candidates = set()
        for key in (left, right):
            option = layouts[key]['text']
            node = rendered_label(option, name, OPTIONS[depth])
            if layouts[key] != text_layout([option]) or len(vectors(key)) != 1:
                raise ValueError('Candidate text must come from its independently encoded one-sentence layout')
            cases['candidate_event'].setdefault(key, (name, node, vectors(key)[0]))
            candidates.add(node)
        if candidates != OPTIONS[depth]:
            raise ValueError('Both authored candidates must be present in the original mapping')
    return {condition: list(rows.values()) for condition, rows in cases.items()}


def accuracy(predicted, labels, names):
    by_label = {}
    for index, label in enumerate(names):
        mask = labels == index
        correct = int(np.count_nonzero(predicted[mask] == index))
        count = int(np.count_nonzero(mask))
        by_label[label] = {'items': count, 'correct': correct, 'accuracy': correct / count}
    return {'accuracy': float(np.mean(predicted == labels)),
            'balanced_accuracy': float(np.mean([row['accuracy'] for row in by_label.values()])), 'by_label': by_label}


def test_probes(examples, holdout):
    fitted = [(label, state) for name, label, state in examples if name not in holdout]
    tested = [(label, state) for name, label, state in examples if name in holdout]
    labels = sorted({label for label, _ in fitted})
    if len(labels) < 2 or set(labels) != {label for label, _ in tested}:
        raise ValueError('The locked name split must cover all authored classes in fit and test')
    indices = {label: index for index, label in enumerate(labels)}
    x_train = np.stack([state for _, state in fitted])
    x_test = np.stack([state for _, state in tested])
    y_train = np.array([indices[label] for label, _ in fitted])
    y_test = np.array([indices[label] for label, _ in tested])
    center, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale < 1e-8] = 1
    x_train, x_test = (x_train - center) / scale, (x_test - center) / scale
    means = np.stack([x_train[y_train == k].mean(axis=0) for k in range(len(labels))])
    centers = np.argmin(np.sum((x_test[:, None] - means)**2, axis=-1), axis=1)
    baseline = np.full(len(y_test), np.argmax(np.bincount(y_train, minlength=len(labels))))
    width = x_train.shape[1]
    target = np.eye(len(labels))[y_train]

    def loss(parameter):
        section = parameter.reshape(width + 1, len(labels))
        weight, bias = section[:-1], section[-1]
        logits = x_train @ weight + bias
        normalized = logits - logsumexp(logits, axis=-1, keepdims=True)
        delta = (np.exp(normalized) - target) / len(y_train)
        value = -np.mean(normalized[np.arange(len(y_train)), y_train]) + RIDGE / 2 * np.sum(weight**2)
        gradient = np.vstack((x_train.T @ delta + RIDGE * weight, np.sum(delta, axis=0)))
        return value, gradient.ravel()

    result = minimize(loss, np.zeros((width + 1) * len(labels)), method='L-BFGS-B', jac=True,
                      options={'maxiter': 250, 'gtol': 1e-8, 'ftol': 1e-12})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f'Multinomial probe failed: {result.message}')
    parameters = result.x.reshape(width + 1, len(labels))
    predicted = np.argmax(x_test @ parameters[:-1] + parameters[-1], axis=1)
    return {'fit_examples': len(fitted), 'test_examples': len(tested), 'probe_iterations': result.nit,
            'majority': accuracy(baseline, y_test, labels), 'centroid': accuracy(centers, y_test, labels),
            'multinomial_probe': accuracy(predicted, y_test, labels)}


def run(assay, extraction, output):
    assay_file = assay / 'assay.json'
    receipt = json.loads(assay_file.read_text(encoding='utf-8'))
    seal_file = extraction / 'seal.json'
    seal = json.loads(seal_file.read_text(encoding='utf-8'))
    fit_map = assay / 'fit.jsonl'
    stimuli = assay / 'model_input/stimuli.jsonl'
    locked = {name: digest(path) for name, path in (('fit.jsonl', fit_map), ('model_input/stimuli.jsonl', stimuli))}
    if (any(receipt['file_sha256'][name] != value for name, value in locked.items())
            or seal['assay_sha256'] != digest(assay_file) or seal['stimuli_sha256'] != locked['model_input/stimuli.jsonl']
            or seal['candidate_id'] != CHECKPOINT or seal['projection_width'] != 128
            or any(seal['file_sha256'][name] != digest(extraction / name) for name in ('coordinates.jsonl', 'vectors.npy'))):
        raise ValueError('Sealed original mapping, layout, and frozen app projection must all match')
    maps = read_rows(fit_map)
    wanted = {row[key] for row in maps for key in ('article_id', 'left_id', 'right_id')}
    layouts = {row['id']: row for row in read_rows(stimuli) if row['id'] in wanted}
    indices, states = read_vectors(extraction)
    if layouts.keys() != wanted or not wanted <= indices.keys():
        raise ValueError('Original study layouts are missing from the sealed feature block')
    examples = examples_from_original(maps, layouts, indices, states)
    names = sorted({name for example in examples.values() for name, _, _ in example},
                   key=lambda name: sha256(f'{SALT}{name}'.encode()).digest())
    holdout = set(names[:(len(names) + 3) // 4])
    payload = {
        'scope': 'Post-hoc decoding of authored stimulus conditions; only layouts referenced by original fit rows were used',
        'split': {'unit': 'fictional name', 'salt': SALT, 'fit_names': len(names) - len(holdout), 'held_out_names': len(holdout),
                  'heldout_name_sha256': sha256(json.dumps(sorted(holdout), ensure_ascii=False).encode()).hexdigest()},
        'probes': {'standardization': 'Original fit names only, then applied to unseen names', 'multinomial_l2': RIDGE},
        'results': {condition: test_probes(rows, holdout) for condition, rows in examples.items()},
        'input_sha256': {**locked, 'assay_sha256': digest(assay_file), 'extraction_sha256': digest(seal_file),
                         'coordinates_sha256': seal['file_sha256']['coordinates.jsonl'], 'vectors_sha256': seal['file_sha256']['vectors.npy'],
                         'diagnostic_code_sha256': digest(Path(__file__))},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('assay', 'extraction', 'output'):
        parser.add_argument(f'--{flag}', required=True, type=Path)
    run(**vars(parser.parse_args()))
