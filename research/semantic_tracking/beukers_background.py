"""Fit and score the Beukers reader-curriculum forecast on frozen Qwen states."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


PROTOCOL = Path(__file__).with_name('beukers-background.md')
CHECKPOINT = '74c26ff02569b2b3bc28ed291bf15fd2f85db5b4c93a3ad7bfc34f6a5d333ea0'
METHODS = ('text', 'additive', 'interaction', 'temperature', 'structure')
GROUPS = ('interleaved', 'blocked')
PENALTY = 0.01
WIDTH = 128
DRAW_COUNT = 10_000


@dataclass(frozen=True)
class Trials:
    qwen: np.ndarray
    choices: np.ndarray
    groups: np.ndarray
    structure: tuple[tuple[str, ...], ...]
    order: np.ndarray
    people: tuple[str, ...]
    strata: tuple[tuple[int, str, str], ...]
    node21_right: np.ndarray
    inventory: dict


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def read_vectors(extraction):
    indices, start = {}, 0
    for row in read_rows(extraction / 'coordinates.jsonl'):
        key, stops = row['id'], row['stops']
        if key in indices or not stops or stops != sorted(set(stops)):
            raise ValueError(f'Duplicate or invalid extraction coordinates for {key}')
        indices[key] = slice(start, start + len(stops))
        start += len(stops)
    states = np.load(extraction / 'vectors.npy', mmap_mode='r', allow_pickle=False)
    if states.shape != (start, WIDTH) or states.dtype not in ('float16', 'float32'):
        raise ValueError('Qwen vector shape or dtype differs from the extraction coordinates')
    return indices, states


def vectors_for(indices, states, key):
    selected = states[indices[key]].astype(np.float64)
    norm = np.linalg.norm(selected, axis=1, keepdims=True)
    if not np.all(np.isfinite(norm) & (np.abs(norm - 1) < 0.002)):
        raise ValueError(f'Invalid unit vectors for {key}')
    return selected / norm


def feature_vector(key, indices, states):
    article, left, right = (vectors_for(indices, states, key_part) for key_part in key)
    if len(article) not in (3, 4) or len(left) != 1 or len(right) != 1:
        raise ValueError('Each trial needs a café cue, current prefix and two independent single-stop options')
    delta = right[0] - left[0]
    early = article[1] * delta
    late = article[-1] * delta
    return np.concatenate((delta * np.sqrt(WIDTH), early * WIDTH, late * WIDTH, [early.sum(), late.sum()]))


def load_trials(assay, split, indices, states):
    maps = read_rows(assay / f'{split}.jsonl')
    truth = read_rows(assay / f'{split}-truth.jsonl')
    labels = {row['id']: row for row in truth}
    if len(labels) != len(truth) or len(maps) != len({row['id'] for row in maps}) or set(labels) != {row['id'] for row in maps}:
        raise ValueError(f'Mismatched immutable trial and response IDs in {split}')
    examples = []
    by_key = defaultdict(set)
    for row in maps:
        label = labels[row['id']]
        key = row['article_id'], row['left_id'], row['right_id']
        group = label['curriculum']
        if group not in GROUPS or label['depth'] not in (1, 2) or label['cafe'] not in ('deep', 'jungle'):
            raise ValueError('Unexpected source condition')
        left, right = label['left_node'], label['right_node']
        if left == right or label['response_node'] not in (left, right) or key[1] == key[2]:
            raise ValueError('The two alternatives and chosen event must be consistently coded')
        if label['depth'] == 1 and label['current_node'] == 'NODE11' and {left, right} != {'NODE21', 'NODE22'}:
            raise ValueError('The fixed NODE11 contrast needs the two registered alternatives')
        by_key[key].add(group)
        examples.append((key, label))
    matched = {key for key, groups in by_key.items() if groups == set(GROUPS)}
    samples = [(key, row) for key, row in examples if key in matched]
    if not samples:
        raise ValueError(f'No same-name, same-order trials in {split}')
    features = {key: feature_vector(key, indices, states) for key in sorted(matched)}
    ordered = [tuple(sorted((row['left_node'], row['right_node']))) for _, row in samples]
    assignment = np.array([GROUPS.index(row['curriculum']) for _, row in samples])
    people = tuple(row['reader_id'] for _, row in samples)
    if len(set(np.array(people)[assignment == 0]) & set(np.array(people)[assignment == 1])):
        raise ValueError('A reader is registered in both curricula')
    inventory = {'exact_keys': len(matched), 'available_trials': len(maps), 'retained_trials': len(samples),
                 'groups': {name: {'trials': int(np.count_nonzero(assignment == group)),
                                   'readers': len(set(np.array(people)[assignment == group]))}
                            for group, name in enumerate(GROUPS)}}
    return Trials(
        np.stack([features[key] for key, _ in samples]),
        np.array([row['response_node'] == row['right_node'] for _, row in samples], dtype=np.float64), assignment,
        tuple((row['family'], *options) for (_, row), options in zip(samples, ordered, strict=True)),
        np.array([1.0 if right == row['right_node'] else -1.0 for (_, right), (_, row) in zip(ordered, samples, strict=True)]),
        people, tuple((row['depth'], row['current_node'], row['cafe']) for _, row in samples),
        np.array([row['right_node'] == 'NODE21' for _, row in samples]), inventory,
    )


def structure_design(trials, basis):
    column = {tuple(key): index for index, key in enumerate(basis)}
    design = np.zeros((len(trials.choices), len(column)))
    for row, key in enumerate(trials.structure):
        if key not in column:
            raise ValueError(f'No training example for structural cell {key}')
        design[row, column[key]] = trials.order[row] * np.sqrt(len(column))
    return design


def designs(trials, basis):
    intercept = np.ones(len(trials.choices))
    group = 2 * trials.groups - 1
    text = trials.qwen
    structure = structure_design(trials, basis)
    return {'text': np.column_stack((intercept, text)),
            'additive': np.column_stack((intercept, text, group)),
            'interaction': np.column_stack((intercept, text, group, group[:, None] * text)),
            'structure': np.column_stack((intercept, structure, group, group[:, None] * structure))}


def reader_weights(trials):
    by_group = []
    for group in range(len(GROUPS)):
        readers = defaultdict(list)
        for index in np.flatnonzero(trials.groups == group):
            readers[trials.people[index]].append(index)
        by_group.append(readers)
    result = np.zeros(len(trials.choices))
    for readers in by_group:
        for rows in readers.values():
            result[rows] = 1 / (len(GROUPS) * len(readers) * len(rows))
    if not np.isclose(result.sum(), 1.0) or np.any(result <= 0):
        raise ValueError('Expected equal group weight and every reader and trial represented')
    return result


def fit_logit(design, labels, weights):
    def loss(parameters):
        logits = design @ parameters
        regularized = parameters.copy()
        regularized[0] = 0
        objective = weights @ (np.logaddexp(0, logits) - labels * logits) + PENALTY / 2 * (regularized @ regularized)
        gradient = design.T @ (weights * (expit(logits) - labels)) + PENALTY * regularized
        return objective, gradient

    result = minimize(loss, np.zeros(design.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f'Logit fit failed: {result.message}')
    return result.x


def calibration_design(trials, text_design, coefficients):
    logit = text_design @ coefficients
    group = 2 * trials.groups - 1
    return np.column_stack((np.ones(len(trials.choices)), logit, group, logit * group))


def fit(trials):
    basis = sorted(set(trials.structure))
    matrix = designs(trials, basis)
    weights = reader_weights(trials)
    params = {key: fit_logit(design, trials.choices, weights) for key, design in matrix.items()}
    temperature = calibration_design(trials, matrix['text'], params['text'])
    params['temperature'] = fit_logit(temperature, trials.choices, weights)
    return {'basis': basis, 'coefficients': {name: values.tolist() for name, values in params.items()}}


def predict(trials, fitted):
    params = fitted['coefficients']
    if params.keys() != set(METHODS):
        raise ValueError('Fit receipt must identify exactly the registered conditions')
    matrices = designs(trials, fitted['basis'])
    matrices['temperature'] = calibration_design(trials, matrices['text'], params['text'])
    scores = np.column_stack([matrices[name] @ params[name] for name in METHODS])
    if not np.all(np.isfinite(scores)):
        raise ValueError('Nonfinite held-out logit')
    return scores


def ci(values, tail):
    return np.quantile(values, [tail, 1 - tail]).tolist()


def compare(trials, logits):
    probabilities = expit(logits)
    nll = np.logaddexp(0, logits) - trials.choices[:, None] * logits
    # Orientation to the event identity makes the target invariant to screen side.
    rate = np.column_stack((trials.choices, probabilities))
    rate = np.where(trials.node21_right[:, None], rate, 1.0 - rate)
    selected = {cafe: np.array([state == (1, 'NODE11', cafe) for state in trials.strata])
                for cafe in ('deep', 'jungle')}
    means, effect_means, nll_draws, effect_draws, coverage = [], [], [], [], {}
    rng = np.random.default_rng(868431)
    names = np.array(trials.people)
    for group, group_name in enumerate(GROUPS):
        subset = np.flatnonzero(trials.groups == group)
        records, effects = [], []
        for person in np.unique(names[subset]):
            rows = subset[names[subset] == person]
            records.append(nll[rows].mean(axis=0))
            deep, jungle = (rows[selected[cafe][rows]] for cafe in ('deep', 'jungle'))
            if len(deep) and len(jungle):
                effects.append(rate[deep].mean(axis=0) - rate[jungle].mean(axis=0))
        if len(records) < 2 or len(effects) < 2:
            raise ValueError(f'Too few independent readers with exact context coverage for {group_name}')
        coverage[group_name] = {'readers_nll': len(records), 'readers_with_both_cafes': len(effects)}
        means.append(np.mean(records, axis=0))
        effect_means.append(np.mean(effects, axis=0))
        draw = rng.integers(0, len(records), size=(DRAW_COUNT, len(records)))
        nll_draws.append(np.asarray(records)[draw].mean(axis=1))
        draw = rng.integers(0, len(effects), size=(DRAW_COUNT, len(effects)))
        effect_draws.append(np.asarray(effects)[draw].mean(axis=1))
    mean = np.mean(means, axis=0)
    draws = np.mean(nll_draws, axis=0)
    effect_means.append(effect_means[1] - effect_means[0])
    effect_draws.append(effect_draws[1] - effect_draws[0])
    interaction_index = METHODS.index('interaction')
    deltas = {}
    for index, name in enumerate(METHODS):
        if name != 'interaction':
            deltas[name] = {'nll_nats': float(mean[interaction_index] - mean[index]),
                            'ci95': ci(draws[:, interaction_index] - draws[:, index], 0.025)}
    endpoints = {}
    for group_name, effect, uncertainty in zip((*GROUPS, 'blocked_minus_interleaved'), effect_means, effect_draws, strict=True):
        endpoints[group_name] = {
            name: {'probability_difference': float(value), 'ci95': ci(uncertainty[:, index], 0.025),
                   'difference_from_human': float(value - effect[0]),
                   'difference_from_human_ci90': ci(uncertainty[:, index] - uncertainty[:, 0], 0.05)}
            for index, (name, value) in enumerate(zip(('human', *METHODS), effect, strict=True))
        }
    gate = {'nll_vs_text': deltas['text']['ci95'][1] < 0,
            'nll_vs_additive': deltas['additive']['ci95'][1] < 0,
            'interaction_direction': endpoints['blocked_minus_interleaved']['interaction']['ci95'][0] > 0}
    for name, tolerance in (('interleaved', 0.15), ('blocked', 0.15), ('blocked_minus_interleaved', 0.20)):
        lower, upper = endpoints[name]['interaction']['difference_from_human_ci90']
        gate[f'{name}_magnitude'] = bool(-tolerance <= lower <= upper <= tolerance)
    return {'nll_nats': dict(zip(METHODS, mean.tolist(), strict=True)), 'interaction_minus_baseline': deltas,
            'cafe_effects': endpoints, 'reader_coverage': coverage, 'gates': gate, 'passes': all(gate.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('fit', 'score'))
    for flag in ('assay', 'extraction', 'fit', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=flag != 'output')
    args = parser.parse_args()
    source = args.extraction / 'seal.json'
    seal = json.loads(source.read_text(encoding='utf-8'))
    provenance = {'assay_sha256': digest(args.assay / 'assay.json'), 'fit_truth_sha256': digest(args.assay / 'fit-truth.jsonl'),
                  'protocol_sha256': digest(PROTOCOL), 'scorer_sha256': digest(Path(__file__)),
                  'extraction_sha256': digest(source), 'fit_mapping_sha256': digest(args.assay / 'fit.jsonl'),
                  'test_mapping_sha256': digest(args.assay / 'test.jsonl')}
    if (seal['candidate_id'] != CHECKPOINT or seal['assay_sha256'] != provenance['assay_sha256']
            or seal['protocol_sha256'] != provenance['protocol_sha256'] or
            any(seal['file_sha256'][name] != digest(args.extraction / name) for name in ('coordinates.jsonl', 'vectors.npy'))):
        raise ValueError('Expected the sealed candidate extraction for this assay and registered protocol')
    indices, states = read_vectors(args.extraction)
    if args.command == 'fit':
        trials = load_trials(args.assay, 'fit', indices, states)
        destination = args.fit
        result = {**provenance, 'fit_counts': trials.inventory, 'params': fit(trials)}
    else:
        if args.output is None:
            parser.error('score requires --output')
        receipt = json.loads(args.fit.read_text(encoding='utf-8'))
        truth = json.loads((args.assay / 'test-truth-seal.json').read_text(encoding='utf-8'))
        if (any(receipt[key] != value for key, value in provenance.items()) or truth['fit_sha256'] != digest(args.fit)
                or truth['assay_sha256'] != provenance['assay_sha256']
                or truth['test_truth_sha256'] != digest(args.assay / 'test-truth.jsonl')):
            raise ValueError('Holdout labels must match the assay and have been opened after the fit seal')
        trials = load_trials(args.assay, 'test', indices, states)
        destination = args.output
        result = {'fit_sha256': digest(args.fit), 'test_truth_sha256': truth['test_truth_sha256'],
                  'test_counts': trials.inventory, **compare(trials, predict(trials, receipt['params']))}
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
