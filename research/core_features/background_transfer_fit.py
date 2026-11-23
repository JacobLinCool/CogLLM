"""Fit and export the five preregistered CWI transfer readouts from sealed inputs."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from research.core_features.background_transfer import KEYS, MODEL_INPUTS, PROTOCOL, VOTE_KEYS
from research.core_features.fit import (
    CWI_METRICS, PENALTY, STEPS, apply_scaler, canonical_stops, cwi_corpus, cwi_design, cwi_features,
    cwi_targets, fit_scaler, read_features, sigmoid, train_score,
)
from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3.probe import _jsonl
from research.stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION, save_json


HIDDEN_WIDTH = 2048
LEXICAL_WIDTH = 7
WEIGHTS_NAME = 'heads.npz'
LEXICON_NAME = 'lexicon.json'
FORECAST_NAME = 'cwi-predictions.jsonl'
SCORER_SEAL = 'fit-manifest.json'
SPLITS = {'train': 'wikinews_train', 'dev': 'wikinews_dev', 'test': 'wikipedia_test'}


def read_inputs(root):
    assay = root / 'assay.json'
    manifest = json.loads(assay.read_text(encoding='utf-8'))
    if (manifest['version'] != 1 or manifest['model'] != MODEL or manifest['revision'] != REVISION
            or manifest['protocol_sha256'] != digest(PROTOCOL) or manifest['forecast_methods'] != list(CWI_METRICS)
            or set(manifest['inputs_sha256']) != set(MODEL_INPUTS)):
        raise ValueError('Unexpected sealed CWI transfer assay schema or protocol')
    for name in MODEL_INPUTS:
        if digest(root / name) != manifest['inputs_sha256'][name]:
            raise ValueError(f'CWI transfer input differs from its seal: {name}')

    documents = canonical_stops(root / 'stimuli.jsonl')
    train = list(_jsonl(root / 'cwi-train.jsonl'))
    evaluation = list(_jsonl(root / 'cwi-eval.jsonl'))
    if (any(row.keys() != {*KEYS, *VOTE_KEYS} or row['split'] != 'train' for row in train)
            or any(row.keys() != set(KEYS) or row['split'] not in {'dev', 'test'} for row in evaluation)):
        raise ValueError('Unexpected training or label-free evaluation input fields')
    case_roles, group_roles, keys = {}, {}, set()
    planned = defaultdict(set)
    for row in train + evaluation:
        case, group, start, stop, role = (row[key] for key in KEYS)
        if (case_roles.setdefault(case, role) != role or group_roles.setdefault(group, role) != role
                or (case, start, stop) in keys or case not in documents
                or type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(documents[case]['text'])):
            raise ValueError('CWI candidate span or fit/validation/test role conflicts')
        keys.add((case, start, stop))
        planned[case].add(stop)
    if planned.keys() != documents.keys() or any(sorted(planned[case]) != row['stops'] for case, row in documents.items()):
        raise ValueError('The candidate coordinates do not match the prepared document stops')
    for role, name in SPLITS.items():
        selected = [row for row in train + evaluation if row['split'] == role]
        counts = {'candidates': len(selected), 'documents': len({row['id'] for row in selected}),
                  'material_groups': len({row['group_id'] for row in selected})}
        if not selected or counts != manifest['counts'][name]:
            raise ValueError(f'Unexpected CWI transfer counts for {role}')
    return manifest, digest(assay), documents, train, evaluation


def read_extraction(folder, manifest, documents):
    extraction_path = folder / 'extraction.json'
    features_path = folder / 'features.npz'
    extraction = json.loads(extraction_path.read_text(encoding='utf-8'))
    if (extraction['model'] != MODEL or extraction['revision'] != REVISION
            or extraction['transformers'] != TRANSFORMERS_VERSION or extraction['dtype'] != 'float32'
            or extraction['gpu'] != 'NVIDIA L4' or extraction['segment_means'] is not False
            or extraction['input_texts'] != len(documents)
            or extraction['input_sha256'] != manifest['inputs_sha256']['stimuli.jsonl']):
        raise ValueError('Expected the pinned L4 FP32 extraction of the sealed transfer documents')
    with np.load(features_path, allow_pickle=False) as archive:
        if set(archive.files) != {'ids', 'stops', 'states'}:
            raise ValueError('Expected sparse causal stop states')
    states = read_features(features_path)
    planned = {(identifier, stop) for identifier, item in documents.items() for stop in item['stops']}
    rejected = {(row['id'], row['stop']) for row in extraction['excluded_stops']}
    if (len(rejected) != len(extraction['excluded_stops']) or rejected - planned or states.keys() != planned - rejected
            or len(states) != extraction['unique_certified_stops']
            or any(row.shape != (HIDDEN_WIDTH,) or row.dtype != np.float32 for row in states.values())):
        raise ValueError('Certified CWI state coordinates or dimensions differ from the sealed plan')
    return states, digest(extraction_path), digest(features_path)


def fit_readouts(documents, states, train, device):
    corpus = cwi_corpus(documents, train)
    eligible, base, hidden = cwi_features(documents, states, train, corpus)
    center, scale = fit_scaler(base)
    lexical = apply_scaler(base, center, scale)
    values, totals = cwi_targets(eligible)
    weights, statistics = {}, {}
    for method in CWI_METRICS:
        design = cwi_design(method, lexical, hidden)
        weights[method], statistics[method] = train_score(design.reshape(-1, design.shape[-1]),
                                                         values.ravel(), totals.ravel(), device)
        del design
        print(f'Fitted CWI {method}', flush=True)
    return weights, center, scale, corpus, statistics, len(eligible)


def load_parameters(heads_path, lexicon_path):
    empty_lexical = np.empty((0, LEXICAL_WIDTH))
    empty_hidden = np.empty((0, HIDDEN_WIDTH))
    shapes = {name: (cwi_design(name, empty_lexical, empty_hidden).shape[-1],) for name in CWI_METRICS}
    shapes.update(lexical_center=(LEXICAL_WIDTH,), lexical_scale=(LEXICAL_WIDTH,))
    with np.load(heads_path, allow_pickle=False) as archive:
        if set(archive.files) != set(shapes):
            raise ValueError('Unexpected CWI checkpoint parameters')
        weights = {key: archive[key] for key in shapes}
    if (any(array.shape != shapes[key] or array.dtype != np.float64 or not np.isfinite(array).all()
            for key, array in weights.items()) or np.any(weights['lexical_scale'] < 0.1)):
        raise ValueError('Invalid CWI checkpoint dimensions, coefficients, or scales')
    corpus = json.loads(lexicon_path.read_text(encoding='utf-8'))
    if (not isinstance(corpus, dict) or not corpus
            or any(not key or key != key.casefold() or type(count) is not int or count < 1
                   for key, count in corpus.items())):
        raise ValueError('Invalid CWI fit lexicon')
    return weights, Counter(corpus)


def forecast(documents, states, evaluation, heads_path, lexicon_path):
    weights, corpus = load_parameters(heads_path, lexicon_path)
    eligible, base, hidden = cwi_features(documents, states, evaluation, corpus)
    lexical = apply_scaler(base, weights['lexical_center'], weights['lexical_scale'])
    rows = [{key: row[key] for key in ('id', 'start', 'stop')} for row in eligible]
    for method in CWI_METRICS:
        probabilities = sigmoid(cwi_design(method, lexical, hidden) @ weights[method])
        if not np.all(np.isfinite(probabilities) & (0 < probabilities) & (probabilities < 1)):
            raise ValueError(f'Expected two finite interior CWI probabilities from {method}')
        for row, pair in zip(rows, probabilities, strict=True):
            row[method] = pair.tolist()
    return rows, dict(Counter(row['split'] for row in eligible))


def fit_transfer(inputs, features, output, device):
    if output.exists():
        raise FileExistsError(output)
    manifest, assay_hash, documents, train, evaluation = read_inputs(inputs)
    states, extraction_hash, feature_hash = read_extraction(features, manifest, documents)
    weights, center, scale, corpus, statistics, n_fit = fit_readouts(documents, states, train, device)

    output.mkdir(parents=True, exist_ok=False)
    heads_path, lexicon_path = output / WEIGHTS_NAME, output / LEXICON_NAME
    np.savez_compressed(heads_path, **weights, lexical_center=center, lexical_scale=scale)
    save_json(lexicon_path, dict(sorted(corpus.items())))
    estimates, counts = forecast(documents, states, evaluation, heads_path, lexicon_path)
    predictions = output / FORECAST_NAME
    write_jsonl(predictions, estimates)
    save_json(output / SCORER_SEAL, {
        'version': 1, 'model': MODEL, 'revision': REVISION, 'status': 'experimental_before_transfer_scoring',
        'assay_sha256': assay_hash, 'protocol_sha256': manifest['protocol_sha256'],
        'extraction_sha256': extraction_hash, 'features_sha256': feature_hash,
        'heads_sha256': digest(heads_path), 'lexicon_sha256': digest(lexicon_path), 'predictions_sha256': digest(predictions),
        'inputs_sha256': manifest['inputs_sha256'], 'methods': statistics, 'penalty': PENALTY, 'max_lbfgs_iterations': STEPS,
        'fit_candidates': n_fit, 'fit_excluded_candidates': len(train) - n_fit, 'forecast_candidates': counts,
        'annotator_order': ['native', 'nonnative'], 'hidden_width': HIDDEN_WIDTH, 'hidden_normalization': 'row L2',
        'lexical_features_in_order': ['log1p_span_characters', 'log1p_span_words', 'log1p_start_character',
                                      'log1p_lowest_fit_word_count', 'mean_log1p_fit_word_count',
                                      'uppercase_span', 'span_contains_digit'],
        'lexical_scaling': '(x - training_center) / training_scale / sqrt(7), scale floor 0.1',
        'lexicon': 'Case-folded lexical counts over unique fit documents; missing words have zero count.',
        'candidate_coordinates': 'Unicode character offsets, half-open spans at certified causal stops.',
        'forecasts_from_reloaded_checkpoint': True,
    })


def run(inputs, features, output):
    import torch

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
        raise RuntimeError('Expected one NVIDIA L4 for the preregistered fit')
    torch.set_num_threads(2)
    fit_transfer(inputs, features, output, torch.device('cuda'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'features', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    run(**vars(parser.parse_args()))
