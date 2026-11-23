"""Cross-fit retrieval of Frick's earlier manipulated sentence from causal article states."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from research.semantic_tracking import frick_consistency_score as source
from research.semantic_tracking import frick_reference_features as features


PROTOCOL = Path(__file__).with_name('frick_reference_protocol.md')
PROTOCOL_SHA256 = 'b6cf0dc912fafdbf8a5b84dad68feca5a23eab63a75744bcc7bec0313e57546a'
RIDGE = 0.05
PSEUDOCOUNT = 0.5
SEED = 735972
DRAWS = 10_000
SCOPES = ('lag_7_to_10', 'all_earlier')
COLUMNS = {
    'position': len(features.AGE_FEATURES),
    'text': len(features.AGE_FEATURES) + len(features.TEXT_FEATURES),
    'qwen': len(features.FEATURES),
}
METHODS = ('ordinal_counts', *COLUMNS)


@dataclass(frozen=True)
class Fit:
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    def log_probabilities(self, matrix):
        design = (matrix[:, :len(self.center)] - self.center) / self.scale
        logits = design @ self.coefficients
        return logits - logsumexp(logits)

    def receipt(self):
        return {key: value.tolist() for key, value in vars(self).items()}


def candidates(scope, t):
    if scope == SCOPES[0]:
        return np.arange(max(0, t - 10), max(0, t - 6))
    if scope == SCOPES[1]:
        return np.arange(t)
    raise ValueError(f'Unknown candidate scope {scope}')


def rank_objective(coefficients, designs, gold):
    losses = []
    gradient = np.zeros_like(coefficients)
    for matrix, index in zip(designs, gold, strict=True):
        logits = matrix @ coefficients
        partition = logsumexp(logits)
        losses.append(partition - logits[index])
        gradient += np.exp(logits - partition) @ matrix - matrix[index]
    return (np.mean(losses) + RIDGE / 2 * (coefficients @ coefficients),
            gradient / len(losses) + RIDGE * coefficients)


def fit_ranker(matrices, gold, width):
    sliced = [matrix[:, :width] for matrix in matrices]
    center = np.mean([matrix.mean(axis=0) for matrix in sliced], axis=0)
    variance = np.mean([np.square(matrix - center).mean(axis=0) for matrix in sliced], axis=0)
    scale = np.sqrt(variance)
    scale[scale <= 1e-12] = 1.0
    designs = [(matrix - center) / scale for matrix in sliced]
    result = minimize(rank_objective, np.zeros(width), args=(designs, gold), jac=True, method='L-BFGS-B',
                      options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-10})
    if not result.success or not np.isfinite(result.x).all():
        raise ArithmeticError(f'Frick source ranking failed to converge: {result.message}')
    return Fit(center, scale, result.x)


def train(fit_keys, labels, matrices, scope):
    designs, indices, locations = [], [], []
    for key in fit_keys:
        t, j = labels[key]
        earlier = candidates(scope, t)
        match = np.flatnonzero(earlier == j)
        if len(match) != 1:
            raise ValueError('Frick training source is outside the candidate set')
        designs.append(matrices[key][t][earlier])
        indices.append(int(match[0]))
        locations.append(j)
    return {name: fit_ranker(designs, indices, width) for name, width in COLUMNS.items()}, Counter(locations)


def forecast(scope, t, matrix, fits, counts):
    earlier = candidates(scope, t)
    if not len(earlier):
        return None
    weights = np.log([counts[int(j)] + PSEUDOCOUNT for j in earlier])
    scores = {'ordinal_counts': weights - logsumexp(weights)}
    scores.update({name: fitted.log_probabilities(matrix[earlier]) for name, fitted in fits.items()})
    return {
        'prior_sentence_indices': earlier.tolist(),
        'log_probabilities': {name: values.tolist() for name, values in scores.items()},
    }


def crossfit(families, training_targets, matrices, texts, states, stimuli):
    receipts, forecasts = [], []
    feature_error = score_error = 0.0
    prefix_checks = 0
    for story in sorted(families):
        heldout = tuple(sorted(families[story].values()))
        fitting = tuple(sorted(key for other, pair in families.items() if other != story for key in pair.values()))
        training = {key: training_targets[key] for key in fitting}
        fits = {scope: train(fitting, training, matrices, scope) for scope in SCOPES}
        receipts.append({'held_out_story': story, 'held_out_versions': heldout, 'training_versions': fitting,
                         'fits': {scope: {'ordinal_source_counts': dict(counts),
                                          **{name: fit.receipt() for name, fit in fitted.items()}}
                                  for scope, (fitted, counts) in fits.items()}})
        for key in heldout:
            for t, matrix in enumerate(matrices[key]):
                if not t:
                    continue
                shorter = features.build_features(texts[key][:t + 1], states[key][:t + 1])[-1]
                feature_error = max(feature_error, float(np.max(np.abs(matrix - shorter))))
                by_scope = {}
                for scope, (fitted, counts) in fits.items():
                    full = forecast(scope, t, matrix, fitted, counts)
                    if full is None:
                        continue
                    prefix = forecast(scope, t, shorter, fitted, counts)
                    full_scores = full['log_probabilities']
                    short_scores = prefix['log_probabilities']
                    score_error = max(score_error, *(float(np.max(np.abs(np.subtract(full_scores[name],
                                                                                    short_scores[name]))))
                                                     for name in METHODS))
                    prefix_checks += 1
                    by_scope[scope] = full
                forecasts.append({'id': key, 'sentence_index': t, 'character_stop': stimuli[key]['stops'][t],
                                  'scopes': by_scope})
    if feature_error > 1e-12 or score_error > 1e-11 or len(forecasts) != sum(len(rows) - 1 for rows in texts.values()):
        raise ValueError('Frick forecasts differ on truncated prefixes or miss a visible stop')
    return forecasts, receipts, {
        'prefix_feature_max_error': feature_error, 'prefix_log_probability_max_error': score_error,
        'prefix_scope_checks': prefix_checks, 'forecast_stops': len(forecasts),
        'archive_forward_calls_per_story': 1, 'additional_backbone_forward_calls': 0,
    }


def retrieval_metrics(logp, gold):
    logs = np.asarray(logp)
    if not np.isfinite(logs).all() or abs(float(logsumexp(logs))) > 1e-12:
        raise ValueError('Each source forecast must be a normalized finite log distribution')
    same = np.abs(logs - logs[gold]) <= 1e-12
    above = int(np.count_nonzero((logs > logs[gold]) & ~same))
    tied = int(np.count_nonzero(same))
    return (-logs[gold], float(above == 0) / tied, (1 / np.arange(above + 1, above + tied + 1)).mean())


def summarize(records, rows, resamples):
    means = records.mean(axis=(0, 1))
    metrics = ('nll', 'top1', 'mrr')
    table = {name: dict(zip(metrics, mean.tolist(), strict=True)) for name, mean in zip(METHODS, means, strict=True)}
    per_condition = {condition: {
        name: dict(zip(metrics, values.tolist(), strict=True))
        for name, values in zip(METHODS, records[:, i].mean(axis=0), strict=True)
    } for i, condition in enumerate(source.CONDITIONS)}
    gains = {}
    for i, method in enumerate(METHODS[:-1]):
        difference = (records[:, :, i, 0] - records[:, :, -1, 0]).mean(axis=1)
        bounds = np.quantile(difference[resamples].mean(axis=1), [0.025, 0.975]).tolist()
        gains[method] = {'mean': float(difference.mean()), 'story_bootstrap_ci95': bounds, 'pass': bounds[0] > 0.0}
    return {'methods': table, 'by_condition': per_condition, 'qwen_nll_gain': gains,
            'pass': all(item['pass'] for item in gains.values()), 'per_story': rows}


def evaluate(forecasts, metadata, families):
    indexed = {(row['id'], row['sentence_index']): row for row in forecasts}
    if len(indexed) != len(forecasts):
        raise ValueError('Duplicate Frick forecasts')
    story_order = sorted(families)
    draws = np.random.default_rng(SEED).integers(0, len(story_order), (DRAWS, len(story_order)))
    results = {}
    for scope in SCOPES:
        all_records = []
        authored = []
        for story in story_order:
            entries, item = [], {}
            for condition in source.CONDITIONS:
                key = families[story][condition]
                t, j = (metadata[key][field] for field in ('target_index', 'changed_index'))
                target = indexed[key, t]['scopes'][scope]
                earlier = target['prior_sentence_indices']
                gold = earlier.index(j)
                if earlier != candidates(scope, t).tolist() or tuple(sorted(target['log_probabilities'])) != tuple(sorted(METHODS)):
                    raise ValueError('Sealed candidate support differs from the scoring protocol')
                entry = [retrieval_metrics(target['log_probabilities'][name], gold) for name in METHODS]
                entries.append(entry)
                item[condition] = {'source_probability': {name: float(np.exp(-metric[0])) for name, metric in
                                                          zip(METHODS, entry, strict=True)},
                                   'top1': {name: metric[1] for name, metric in zip(METHODS, entry, strict=True)}}
            all_records.append(entries)
            authored.append({'story': story, 'source_sentence': j + 1, 'target_sentence': t + 1,
                             'candidate_count': len(earlier), 'conditions': item})
        results[scope] = summarize(np.asarray(all_records), authored, draws)
    return results


def run(assay, extraction, packet, output):
    if output.exists():
        raise FileExistsError(output)
    if source.digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('The Frick source-localization protocol changed')
    _, stimuli, metadata, families, texts = source.load_layout(assay)
    states, provenance = source.load_states(assay, extraction, packet, source.PROTOCOL, stimuli, metadata, families)
    matrices = {key: features.build_features(texts[key], states[key]) for key in sorted(texts)}
    training_targets = {key: (item['target_index'], item['changed_index']) for key, item in metadata.items()}
    forecasts, receipts, verification = crossfit(families, training_targets, matrices, texts, states, stimuli)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.frick-reference-') as temporary:
        stage = Path(temporary) / 'scored'
        stage.mkdir()
        forecast_path = stage / 'forecasts.jsonl'
        with forecast_path.open('x', encoding='utf-8') as stream:
            for forecast_row in forecasts:
                stream.write(json.dumps(forecast_row, sort_keys=True, allow_nan=False) + '\n')
        source.write_json(stage / 'fits.json', {'features': features.FEATURES, 'ridge': RIDGE,
                                                'ordinal_pseudocount': PSEUDOCOUNT, 'folds': receipts})
        source.write_json(stage / 'verification.json', verification)
        seal = stage / 'forecast-seal.json'
        source.write_json(seal, {'provenance': provenance, 'protocol_sha256': PROTOCOL_SHA256,
                                'scorer_sha256': source.digest(Path(__file__)),
                                'features_sha256': source.digest(Path(features.__file__)),
                                'file_sha256': {name: source.digest(stage / name) for name in
                                                ('forecasts.jsonl', 'fits.json', 'verification.json')}})
        results = evaluate(source.read_rows(forecast_path), metadata, families)
        source.write_json(stage / 'results.json', {
            'stories': len(families), 'versions': len(texts), 'scopes': results,
            'candidate_pair_counts_at_target': {scope: 2 * sum(row['candidate_count'] for row in result['per_story'])
                                                for scope, result in results.items()},
            'retrospective': True, 'bootstrap_draws': DRAWS, 'forecast_seal_sha256': source.digest(seal),
        })
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assay', 'extraction', 'packet', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
