"""Retrospectively test a choice head driven by decoded text states instead of oracle tags."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.semantic_tracking import beukers_background as main
from research.semantic_tracking.beukers_state_diagnostic import OPTIONS, examples_from_original


def centroid(examples):
    labels = sorted({label for _, label, _ in examples})
    states = np.stack([state for _, _, state in examples])
    center, scale = states.mean(axis=0), states.std(axis=0)
    scale[scale < 1e-8] = 1.0
    clusters = np.stack([np.mean([state for _, actual, state in examples if actual == label], axis=0)
                         for label in labels])
    return labels, (clusters - center) / scale, center, scale


def tag(model, state):
    labels, clusters, center, scale = model
    distances = np.sum(np.square(clusters - (state - center) / scale), axis=1)
    return labels[int(np.argmin(distances))]


def one_key(row):
    return row['article_id'], row['left_id'], row['right_id']


def project(assay, extraction, fit):
    seal = json.loads((extraction / 'seal.json').read_text(encoding='utf-8'))
    if not (seal['candidate_id'] == main.CHECKPOINT and seal['assay_sha256'] == fit['assay_sha256'] == main.digest(assay / 'assay.json')
            and fit['extraction_sha256'] == main.digest(extraction / 'seal.json')
            and fit['fit_mapping_sha256'] == main.digest(assay / 'fit.jsonl')
            and fit['test_mapping_sha256'] == main.digest(assay / 'test.jsonl')
            and seal['stimuli_sha256'] == main.digest(assay / 'model_input/stimuli.jsonl')
            and all(main.digest(extraction / name) == seal['file_sha256'][name] for name in ('coordinates.jsonl', 'vectors.npy'))):
        raise ValueError('The factorized replay requires the original inputs, fit and sealed state projections')
    indices, states = main.read_vectors(extraction)
    original = main.read_rows(assay / 'fit.jsonl')
    original_text = {value for row in original for value in one_key(row)}
    layouts = {row['id']: row for row in main.read_rows(assay / 'model_input/stimuli.jsonl') if row['id'] in original_text}
    examples = examples_from_original(original, layouts, indices, states)
    cafe = centroid(examples['cafe_cue'])
    current = {d: centroid(examples[f'current_event_depth_{d}']) for d in (1, 2)}
    options = {depth: centroid([row for row in examples['candidate_event'] if row[1] in nodes])
               for depth, nodes in OPTIONS.items()}
    basis = {tuple(entry): i for i, entry in enumerate(fit['params']['basis'])}
    coefficients = np.array(fit['params']['coefficients']['structure'])
    predictions = {}
    for key in sorted({one_key(row) for row in main.read_rows(assay / 'test.jsonl')}):
        story, left, right = (main.vectors_for(indices, states, value) for value in key)
        depth = len(story) - 2
        nodes = [tag(current[1], story[2])]
        if depth == 2:
            nodes.append(tag(current[2], story[3]))
        family = '/'.join((tag(cafe, story[1]), *nodes))
        left_node, right_node = tag(options[depth], left[0]), tag(options[depth], right[0])
        cell = (family, *sorted((left_node, right_node)))
        if cell not in basis:
            predictions[key] = {'family': family, 'left_node': left_node, 'right_node': right_node, 'logits': None}
            continue
        indicator = np.zeros(len(basis))
        indicator[basis[cell]] = (1 if left_node < right_node else -1) * np.sqrt(len(basis))
        logits = [float(np.concatenate(([1], indicator, [group], group * indicator)) @ coefficients) for group in (-1, 1)]
        predictions[key] = {'family': family, 'left_node': left_node, 'right_node': right_node, 'logits': logits}
    return predictions


def score(assay, extraction, fit, predictions):
    indices, states = main.read_vectors(extraction)
    rows = main.read_rows(assay / 'test.jsonl')
    truth = {row['id']: row for row in main.read_rows(assay / 'test-truth.jsonl')}
    groups = {}
    for row in rows:
        groups.setdefault(one_key(row), set()).add(truth[row['id']]['curriculum'])
    matched = [(one_key(row), truth[row['id']]) for row in rows if groups[one_key(row)] == set(main.GROUPS)]
    trials = main.load_trials(assay, 'test', indices, states)
    oracle = main.predict(trials, fit['params'])
    if len(matched) != len(trials.choices):
        raise ValueError('Expected identical matched rows before applying the original scoring')
    logits, hits = [], []
    for index, (key, row) in enumerate(matched):
        pred = predictions[key]
        if pred['logits'] is None:
            raise ValueError('Factorized candidate abstained on the matched holdout; common-support scoring is required')
        group = main.GROUPS.index(row['curriculum'])
        if trials.people[index] != row['reader_id'] or trials.groups[index] != group:
            raise ValueError('Replay order does not match the original scorer')
        logits.append(pred['logits'][group])
        hits.append(all(pred[name] == row[name] for name in ('family', 'left_node', 'right_node')))
    structure_index = main.METHODS.index('structure')
    oracle_difference = np.max(np.abs(oracle[:, structure_index] - logits))
    oracle[:, structure_index] = logits
    result = main.compare(trials, oracle)
    return {'matched_trials': len(matched), 'decoded_all_roles_correct': int(np.count_nonzero(hits)),
            'abstained_layouts_all': sum(row['logits'] is None for row in predictions.values()),
            'oracle_max_logit_abs_difference': float(oracle_difference),
            'factorized_nll_nats': result['nll_nats']['structure'],
            'original_interaction_minus_factorized': result['interaction_minus_baseline']['structure'],
            'cafe_effects': {name: values['structure'] for name, values in result['cafe_effects'].items()}}


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def run(args):
    fit = json.loads(args.fit.read_text(encoding='utf-8'))
    forecasts = project(args.assay, args.extraction, fit)
    write(args.predictions, {','.join(key): value for key, value in forecasts.items()})
    truth = json.loads((args.assay / 'test-truth-seal.json').read_text(encoding='utf-8'))
    if truth['test_truth_sha256'] != main.digest(args.assay / 'test-truth.jsonl') or truth['fit_sha256'] != main.digest(args.fit):
        raise ValueError('Expected original replicated truth and original sealed behavioral fit')
    write(args.output, {'scope': 'Retrospective use of the previously scored replication; original behavioral fit and cue decoders',
                        'predictions_sha256': main.digest(args.predictions), 'fit_sha256': main.digest(args.fit),
                        'test_truth_sha256': truth['test_truth_sha256'], 'replay_sha256': main.digest(Path(__file__)),
                        **score(args.assay, args.extraction, fit, forecasts)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('assay', 'extraction', 'fit', 'predictions', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    run(parser.parse_args())
