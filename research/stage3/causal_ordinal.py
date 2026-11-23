"""Fit and evaluate distributions for the two PLABA sentence rating scales."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import log_softmax

from research.core_features.readback_score import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, paired_material_interval
from research.stage2.records import write_jsonl

from .causal_export import RESULTS, WIDTH, sha256, verified_inputs
from .probe import TARGETS, _jsonl, _length_standardization, _load_plaba
from .run import MODEL, REVISION, save_json


LEVELS = np.arange(1, 6)
PENALTY = 1.0
METHODS = ('pooled', 'length', 'length_plus_variant', 'qwen_plus_length', 'qwen_and_rewrite_effect')
EFFECT_NLL_MARGIN = 0.01


def distributions(ratings, identifiers):
    rows = [np.asarray(ratings[key]) for key in identifiers]
    return np.stack([(row[..., None] == LEVELS).mean(axis=0) for row in rows])


def fit_softmax(design, observed, contrast_weights=None, initial=None):
    effect_target = None if contrast_weights is None else contrast_weights @ (observed @ LEVELS)

    def loss(flat):
        weights = flat.reshape(design.shape[1], len(LEVELS))
        log_p = log_softmax(design @ weights, axis=-1)
        error = float(-np.sum(observed * log_p) + (PENALTY / 2) * np.sum(weights[1:] ** 2))
        probabilities = np.exp(log_p)
        residual = probabilities - observed
        if contrast_weights is not None:
            expectations = probabilities @ LEVELS
            effect_error = contrast_weights @ expectations - effect_target
            error += len(design) * effect_error ** 2 / 2
            residual += (len(design) * effect_error * contrast_weights)[:, None] * probabilities * (LEVELS - expectations[:, None])
        gradient = design.T @ residual
        gradient[1:] += PENALTY * weights[1:]
        return error, gradient.ravel()

    start = np.zeros(design.shape[1] * len(LEVELS)) if initial is None else initial.flatten()
    solution = minimize(loss, start, method='L-BFGS-B', jac=True,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-7})
    if not solution.success or not np.isfinite(solution.x).all():
        raise ArithmeticError(f'PLABA probability fit failed: {solution.message}')
    return solution.x.reshape(design.shape[1], len(LEVELS))


def reader_components(stimuli, source, families):
    family_values, family_index = np.unique(families, return_inverse=True)
    mapping = {name: i for i, name in enumerate(family_values)}
    readers = {}
    pairs = set()
    for trial in _jsonl(source):
        reader = readers.setdefault(trial['participant_id'], len(readers))
        family = stimuli[trial['stimulus_id']]['family_id']
        pairs.add((mapping[family], len(mapping) + reader))
    left, right = zip(*sorted(pairs), strict=True)
    width = len(mapping) + len(readers)
    graph = coo_matrix((np.ones(len(pairs)), (left, right)), shape=(width, width)).tocsr()
    count, group = connected_components(graph, directed=False)
    return count, [{'group_id': int(group[i])} for i in family_index]


def design_matrices(features, lengths, variant, train):
    center, scale = _length_standardization(lengths[train])
    lengths = (lengths - center) / scale / np.sqrt(lengths.shape[1])
    intercept = np.ones(len(features))
    return {'length': np.column_stack((intercept, lengths)),
            'length_plus_variant': np.column_stack((intercept, lengths, variant)),
            'qwen_plus_length': np.column_stack((intercept, features, lengths))}, center, scale


def metrics(log_p, observed):
    probability = np.exp(log_p)
    predicted_cdf = probability.cumsum(axis=-1)[..., :-1]
    actual_cdf = observed.cumsum(axis=-1)[..., :-1]
    rps = (predicted_cdf ** 2 - 2 * predicted_cdf * actual_cdf + actual_cdf).mean(axis=-1)
    return {'nll_nats': -np.sum(observed * log_p, axis=-1), 'ranked_probability_score': rps}


def rewrite_pairs(alignment, ids, keys, stimuli):
    audit = json.loads((RESULTS / 'stage2-audit.json').read_text(encoding='utf-8'))['plaba']
    alignment_sha256 = sha256(alignment)
    if alignment_sha256 != audit['source']['files']['sentences.csv']['sha256']:
        raise ValueError('PLABA alignment differs from the pinned author data')
    with alignment.open(encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    by_id = {row['sentence_uid']: row for row in rows}
    if len(by_id) != len(rows) or by_id.keys() != set(ids):
        raise ValueError('Expected one source alignment for each PLABA sentence')
    anchors = defaultdict(lambda: defaultdict(list))
    for i, (sentence, key) in enumerate(zip(ids, keys, strict=True)):
        row, stimulus = by_id[sentence], stimuli[key]
        variant = stimulus['metadata']['variant']
        if (row['doc_id'], row['base_text_id'], row['variant']) != (key, stimulus['family_id'], variant):
            raise ValueError(f'Invalid PLABA rewrite alignment for {sentence}')
        anchors[(row['base_text_id'], int(row['mapped_orig_sentence_index']))][variant].append(i)
    pairs = [(by_variant['orig'], by_variant['simp']) for by_variant in anchors.values()
             if set(by_variant) == {'orig', 'simp'}]
    expected_metrics = audit['metrics']
    if len(pairs) != expected_metrics['paired_sentence_alignments']:
        raise ValueError('The PLABA rewrite contrast denominator changed')
    return pairs, alignment_sha256, expected_metrics['paired_mean_understanding_difference_simplified_minus_original']


def fit_contrast(pairs, train):
    weights = np.zeros(len(train))
    included = 0
    for original, simplified in pairs:
        sides = train[original + simplified]
        if not np.any(sides):
            continue
        if not np.all(sides):
            raise ValueError('Paired rewrite alignment crossed a fit/holdout boundary')
        weights[original] -= 1 / len(original)
        weights[simplified] += 1 / len(simplified)
        included += 1
    if not included:
        raise ValueError('No paired rewrite contrasts in the training fold')
    return weights[train] / included


def paired_effect_improvement(candidate_error, reference_error, rows):
    groups, indices = np.unique([row['group_id'] for row in rows], return_inverse=True)
    candidates = np.bincount(indices, weights=candidate_error)
    references = np.bincount(indices, weights=reference_error)
    counts = np.bincount(indices)
    sample = np.random.default_rng(BOOTSTRAP_SEED).integers(len(groups), size=(BOOTSTRAP_DRAWS, len(groups)))
    score = (np.abs(candidates[sample].sum(axis=1)) - np.abs(references[sample].sum(axis=1))) / counts[sample].sum(axis=1)
    return {'difference': float(abs(candidate_error.mean()) - abs(reference_error.mean())),
            'descriptive_paired_material_95_ci': np.quantile(score, [0.025, 0.975]).tolist(), 'groups': len(groups)}


def rewrite_effect(pairs, alignment_sha256, expected_understanding, observed, forecasts, bootstrap_rows):
    grouped = [bootstrap_rows[orig[0]] for orig, _ in pairs]

    def contrast(values):
        return np.array([values[simple].mean(axis=0) - values[orig].mean(axis=0) for orig, simple in pairs])

    reference = contrast(observed @ LEVELS)
    predicted = {method: contrast(np.exp(log_p) @ LEVELS) for method, log_p in forecasts.items()}
    if not np.isclose(reference[:, TARGETS.index('perceived_understanding')].mean(),
                      expected_understanding, rtol=0, atol=1e-12):
        raise ValueError('Original rewrite contrast does not reproduce the publisher-linked audit')
    base_error = predicted['qwen_plus_length'] - reference
    calibrated_error = predicted['qwen_and_rewrite_effect'] - reference
    return {'aligned_anchors': len(pairs), 'sentences_csv_sha256': alignment_sha256,
            'targets': {name: {
                'observed_simplified_minus_original': float(reference[:, i].mean()),
                'held_out_difference': {method: float(values[:, i].mean()) for method, values in predicted.items()},
                'qwen_effect_minus_observed': paired_material_interval(base_error[:, i], grouped),
                'calibrated_effect_minus_observed': paired_material_interval(calibrated_error[:, i], grouped),
                'calibrated_minus_base_absolute_amplitude_error': paired_effect_improvement(
                    calibrated_error[:, i], base_error[:, i], grouped),
                'qwen_mean_absolute_alignment_error': float(np.abs(base_error[:, i]).mean()),
            } for i, name in enumerate(TARGETS)}}


def run(features, data_dir, stage3_payload, alignment, output):
    if output.exists():
        raise FileExistsError(output)
    _, provenance = verified_inputs(features, data_dir, stage3_payload)
    stimuli, sentences, ratings, fold_of, participants = _load_plaba(data_dir)
    ids = list(sentences)
    keys = [sentences[key][0] for key in ids]
    families = [stimuli[key]['family_id'] for key in keys]
    folds = np.array([fold_of[name] for name in families])
    length = np.log1p([(len(text), len(text.split())) for _, text in sentences.values()])
    variant = np.array([stimuli[key]['metadata']['variant'] == 'orig' for key in keys])
    groups, bootstrap_rows = reader_components(stimuli, data_dir / 'trials.jsonl', families)
    split = json.loads((data_dir / 'splits.json').read_text(encoding='utf-8'))
    if groups != split['group_count']:
        raise ValueError('PLABA reader-connected bootstrap groups differ from the pinned development folds')

    with np.load(features, allow_pickle=False) as archive:
        feature_ids, raw = archive['ids'].tolist(), archive['features']
    index = {key: i for i, key in enumerate(feature_ids)}
    if len(index) != len(feature_ids) or index.keys() != sentences.keys() or raw.shape != (len(ids), 2 * WIDTH):
        raise ValueError('Expected exactly the certified PLABA sentence rows and feature blocks')
    frozen = raw[[index[key] for key in ids], :WIDTH].astype(np.float64)
    frozen /= np.linalg.norm(frozen, axis=1, keepdims=True)
    if not np.isfinite(frozen).all():
        raise ValueError('Invalid causal PLABA feature norms')

    observed = distributions(ratings, ids)
    pairs, alignment_sha256, expected_understanding = rewrite_pairs(alignment, ids, keys, stimuli)
    forecasts = {method: np.empty_like(observed) for method in METHODS}
    for fold in range(5):
        test = folds == fold
        train = ~test
        matrices, _, _ = design_matrices(frozen, length, variant, train)
        contrast_weights = fit_contrast(pairs, train)
        forecasts['pooled'][test] = np.log((observed[train].sum(axis=0) + 0.5) / (train.sum() + 2.5))
        for method, matrix in matrices.items():
            for target in range(len(TARGETS)):
                weights = fit_softmax(matrix[train], observed[train, target])
                forecasts[method][test, target] = log_softmax(matrix[test] @ weights, axis=-1)
                if method == 'qwen_plus_length':
                    calibrated = fit_softmax(matrix[train], observed[train, target], contrast_weights, weights)
                    forecasts['qwen_and_rewrite_effect'][test, target] = log_softmax(matrix[test] @ calibrated, axis=-1)
    rewrite = rewrite_effect(pairs, alignment_sha256, expected_understanding, observed, forecasts, bootstrap_rows)

    scores = {method: metrics(log_p, observed) for method, log_p in forecasts.items()}
    result = {}
    for target, name in enumerate(TARGETS):
        candidate = scores['qwen_plus_length']
        calibrated = {measure: paired_material_interval((scores['qwen_and_rewrite_effect'][measure] - loss)[:, target], bootstrap_rows)
                      for measure, loss in candidate.items()}
        amplitude = rewrite['targets'][name]['calibrated_minus_base_absolute_amplitude_error']
        result[name] = {
            'mean_score': {method: {measure: float(value[:, target].mean()) for measure, value in values.items()}
                           for method, values in scores.items()},
            'paired_qwen_minus_controls': {
                method: {measure: paired_material_interval((value - scores[method][measure])[:, target], bootstrap_rows)
                         for measure, value in candidate.items()} for method in METHODS[:3]
            },
            'paired_calibrated_minus_qwen': calibrated,
            'development_alignment_gate': {
                'amplitude_error_decreased': amplitude['descriptive_paired_material_95_ci'][1] < 0,
                'nll_increase_within_margin': calibrated['nll_nats']['descriptive_paired_material_95_ci'][1] < EFFECT_NLL_MARGIN,
            },
            'rating_bin_frequency': observed[:, target].mean(axis=0).tolist(),
            'predicted_bin_frequency': np.exp(forecasts['qwen_plus_length'][:, target]).mean(axis=0).tolist(),
        }

    matrices, center, scale = design_matrices(frozen, length, variant, slice(None))
    matrix = matrices['qwen_plus_length']
    weights = np.stack([fit_softmax(matrix, observed[:, target]) for target in range(len(TARGETS))])
    all_contrast_weights = fit_contrast(pairs, np.ones(len(folds), dtype=bool))
    calibrated = np.stack([fit_softmax(matrix, observed[:, target], all_contrast_weights, weights[target])
                           for target in range(len(TARGETS))])
    manifest = {
        'model': MODEL, 'revision': REVISION, 'data': provenance, 'scope': 'Development cross-validation',
        'target_order': list(TARGETS), 'ratings': LEVELS.tolist(), 'sentences': len(ids), 'readers': len(participants),
        'reader_connected_groups': groups, 'folds': np.bincount(folds).tolist(),
        'rating_pairs': sum(map(len, ratings.values())), 'loss': 'sum of equally weighted sentence rating cross entropies',
        'l2_nonintercept': PENALTY, 'all_data_head_not_scored_out_of_fold': True,
        'rewrite_effect_penalty': 'half the squared mean training rewrite effect error, times the number of training sentences',
        'development_nll_noninferiority_margin_nats': EFFECT_NLL_MARGIN,
        'results': result, 'paired_simplified_rewrite_effect': rewrite,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.plaba-distributions-') as temporary:
        bundle = Path(temporary) / 'checkpoint'
        bundle.mkdir()
        predictions = bundle / 'oof-predictions.jsonl'
        head = bundle / 'head.npz'
        write_jsonl(predictions, ({'sentence_id': key, 'fold': int(folds[i]), 'rating_count': len(ratings[key]),
                                   'probability': {method: np.exp(log_p[i]).tolist() for method, log_p in forecasts.items()}}
                                  for i, key in enumerate(ids)))
        np.savez(head, weights=weights, effect_calibrated_weights=calibrated, length_mean=center, length_scale=scale,
                 ratings=LEVELS, targets=np.asarray(TARGETS))
        save_json(bundle / 'manifest.json', {**manifest, 'head_sha256': sha256(head), 'oof_predictions_sha256': sha256(predictions)})
        bundle.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'data_dir', 'stage3_payload', 'alignment', 'output'):
        parser.add_argument(f'--{name.replace("_", "-")}', type=Path, required=True)
    run(**vars(parser.parse_args()))
