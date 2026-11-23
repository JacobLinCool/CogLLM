"""Evaluate the post-hoc QASPER positive-only clock control on the sealed test papers."""

import argparse
import json
from pathlib import Path

from research.semantic_tracking import evidence_tail_calibrate as primary

import numpy as np
from scipy.optimize import minimize


def positive_clock(cases, layouts):
    positives = [(case, [stop for stop in case.bins if stop is not None]) for case in cases]
    positives = [(case, stops) for case, stops in positives if stops]
    if not positives:
        raise ValueError('Expected at least one development question with observed evidence arrival')
    left, right, ratio, weights = [], [], [], []
    for case, stops in positives:
        layout = layouts[case.identifier]
        left.extend(layout.log_left[stops])
        right.extend(layout.log_right[stops])
        ratio.extend(layout.log_ratio[stops])
        weights.extend([1 / (len(positives) * len(stops))] * len(stops))
    left, right, ratio, weights = map(np.asarray, (left, right, ratio, weights))
    fit = minimize(lambda estimate: -np.sum(weights * primary.log_interval_mass(estimate, left, right, ratio)),
                   (0.0, 0.0), method='L-BFGS-B', bounds=primary.FIT_BOUNDS[:2],
                   options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f'The positive-only clock did not converge: {fit.message}')
    return {'estimate': fit.x.tolist(), 'a': float(fit.x[0]), 'b': float(np.exp(fit.x[1])),
            'train_nll_nats': float(fit.fun), 'iterations': fit.nit, 'questions': len(positives), 'references': len(left),
            'papers': len({case.article for case, _ in positives})}


def compare(cases, paths, subset):
    methods = ('mixed_qasper_clock', 'positive_only_qasper_clock')
    nll = {name: [] for name in methods}
    brier = {name: [] for name in methods}
    terminal = {name: [] for name in methods}
    groups, references = [], 0
    for case in cases:
        stops = [stop for stop in case.bins if subset == 'all' or (stop is None) == (subset == 'null')]
        if not stops:
            continue
        groups.append(case.article)
        references += len(stops)
        for name in methods:
            path = paths[case.identifier][name]
            nll[name].append(float(np.mean([-path.log_tail if stop is None else -path.log_event[stop] for stop in stops])))
            expected = np.zeros((len(stops), len(path.cdf)))
            for index, stop in enumerate(stops):
                if stop is not None:
                    expected[index, stop:] = 1
            brier[name].append(float(np.mean((path.cdf - expected)**2)))
            terminal[name].append(float(np.exp(path.log_tail)))
    difference = np.asarray(nll[methods[0]]) - np.asarray(nll[methods[1]])
    return {'papers': len(set(groups)), 'questions': len(groups), 'references': references,
            **{name: {method: float(np.mean(values)) for method, values in metric.items()}
               for name, metric in (('nll_nats', nll), ('cumulative_brier', brier), ('mean_terminal_residual', terminal))},
            'mixed_minus_positive_nll_nats': primary.paired_interval(difference, groups)}


def run(assay_root, predictions_dir, fit, scores, output):
    protocol, heads = primary.DEFAULT_PROTOCOL, primary.DEFAULT_HEADS
    plan, hashes, articles, queries, schedules, _ = primary.forecast_inputs(assay_root, predictions_dir, protocol, heads)
    fit_hash = primary.digest(fit)
    sealed_fit = json.loads(fit.read_text(encoding='utf-8'))
    sealed_scores = json.loads(scores.read_text(encoding='utf-8'))
    if (any(sealed_fit['source_hashes'].get(name) != digest for name, digest in hashes.items())
            or sealed_scores['input_sha256']['fit_receipt_sha256'] != fit_hash
            or any(sealed_scores['input_sha256'].get(name) != digest for name, digest in sealed_fit['source_hashes'].items())):
        raise ValueError('The comparison requires the original sealed fit, scores, and forecasts')
    partitions, sha = primary.split_inventory(assay_root, plan, articles)
    if sha != sealed_fit['source_hashes']['article_metadata_sha256']:
        raise ValueError('Official material split differs from the sealed fit')
    forecasts = primary.indexed_rows(predictions_dir / 'predictions.jsonl', {'id', 'article_id', 'stops', 'methods'})
    layouts = {identifier: primary.PathLayout.from_hazards(row['methods']['token_rank']) for identifier, row in forecasts.items()}
    training, _, train, training_ids, train_sha = primary.read_references(
        assay_root / 'development.jsonl', plan, articles, queries, schedules,
    )
    if train != partitions['train'] or train_sha != sealed_fit['source_hashes']['training_sha256']:
        raise ValueError('Development references differ from the sealed official training inventory')
    control = positive_clock(training, layouts)
    heldout, _, test, test_ids, test_sha = primary.read_references(
        assay_root / 'holdout.jsonl', plan, articles, queries, schedules,
    )
    if (test != partitions['test'] or train & test or training_ids & test_ids or training_ids | test_ids != queries.keys()
            or test_sha != sealed_scores['input_sha256']['holdout_sha256']):
        raise ValueError('Test references differ from the original sealed independent holdout')
    clocks = {'mixed_qasper_clock': sealed_fit['fitted_methods']['token_rank']['clock']['estimate'],
              'positive_only_qasper_clock': control['estimate']}
    paths = {case.identifier: {name: primary.probability_path(estimate, layouts[case.identifier])
                              for name, estimate in clocks.items()} for case in heldout}
    metrics = {subset: compare(heldout, paths, subset) for subset in ('all', 'event', 'null')}
    for subset, result in metrics.items():
        original = sealed_scores['methods']['token_rank'][subset]
        if any(result[metric]['mixed_qasper_clock'] != original[metric]['clock']
               for metric in ('nll_nats', 'cumulative_brier', 'mean_terminal_residual')):
            raise ValueError('Mixed clock metrics failed to reproduce the original sealed scores')
    primary.write_new(output, {
        'scope': 'Post-hoc QASPER comparison holding domain, forecast, clock family, and official test papers fixed',
        'training_weighting': 'Equal questions with an eligible training reference, equal eligible references within question',
        'mixed_train': sealed_fit['development'], 'positive_only_train': control, 'mixed_estimate': clocks['mixed_qasper_clock'],
        'holdout': metrics, 'bootstrap': {'draws': primary.DRAWS, 'seed': primary.SEED, 'unit': 'official test paper'},
        'input_sha256': {**hashes, 'fit_receipt_sha256': fit_hash, 'original_scores_sha256': primary.digest(scores),
                         'holdout_sha256': test_sha, 'training_sha256': train_sha, 'control_code_sha256': primary.digest(Path(__file__))},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('assay-root', 'predictions-dir', 'fit', 'scores', 'output'):
        parser.add_argument(f'--{flag}', required=True, type=Path)
    run(**vars(parser.parse_args()))
