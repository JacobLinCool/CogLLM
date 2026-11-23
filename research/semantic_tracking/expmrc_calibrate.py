"""Cross-fit the locked ExpMRC arrival calibration on a single CPU thread."""

import argparse
from hashlib import sha256
from itertools import combinations
import json
from math import fsum, log
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

for thread_variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[thread_variable] = '1'

import numpy as np
from scipy.optimize import minimize

from expmrc_score import (
    Arrival, DRAWS, PRIOR_MATERIAL_OVERLAP, SEED, TIE_ATOL, digest, forecast_inputs, indexed_rows,
    nll_at_references, paired_interval, score_endpoints,
)


PROTOCOL_SHA256 = '82ce7b16267045ab1763ba7d55b14ec2d59f19fd3c8f8e2201bb9ae83e4e32a3'
CALIBRATED = 'calibrated_token_rank'
BASELINES = ('token_rank', 'question_only')
METHODS = (CALIBRATED, *BASELINES)


def fold_assignment(articles):
    order = sorted(articles, key=lambda article: sha256(f'expmrc-calibration-v1:{article}'.encode()).digest())
    return {article: rank % 5 for rank, article in enumerate(order)}


def log_intervals(intensity):
    upper = np.log(intensity)
    lower = np.concatenate(([-np.inf], upper[:-1]))
    return lower, upper


def transform(estimate, lower, upper):
    a, log_b = estimate
    b = np.exp(log_b)
    lower_intensity = np.exp(a + b * lower)
    upper_intensity = np.exp(a + b * upper)
    increment = upper_intensity * -np.expm1(b * (lower - upper))
    hazard = -np.expm1(-increment)
    return upper_intensity, hazard, lower_intensity - np.log(hazard)


def average(values):
    return fsum(values) / len(values)


def fit_other_articles(fold, assignments, references, bounds_by_query):
    training = [case for article, cases in sorted(references.items()) if assignments[article] != fold
                for case in sorted(cases, key=lambda item: item.identifier)]
    reference_bounds = np.array([[bound[list(case.bins)] for bound in bounds_by_query[case.identifier]]
                                 for case in training])
    lower, upper = reference_bounds.transpose(1, 0, 2)
    result = minimize(lambda estimate: float(transform(estimate, lower, upper)[2].mean()), x0=(0, 0),
                      method='L-BFGS-B', bounds=((-12, 12), (log(0.25), log(4))),
                      options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f'Calibration did not converge in fold {fold}: {result.message}')
    return result.x, result.fun, len(training), result.nit


def calibrated_path(estimate, intervals):
    intensity, hazard, nll = transform(estimate, *intervals)
    cdf = -np.expm1(-intensity)
    survival = np.exp(-intensity[-1])
    if (np.any(np.diff(intensity) <= 0) or np.any((hazard <= 0) | (hazard > 1))
            or np.any(np.diff(cdf) < 0) or np.any((cdf < 0) | (cdf > 1))
            or not np.isfinite(nll).all() or not np.isclose(np.exp(-nll).sum() + survival, 1, atol=1e-12, rtol=0)):
        raise RuntimeError('Transformed intervals and residual survival do not define a proper distribution')
    if not np.allclose(np.cumsum(np.exp(-nll)), cdf, atol=1e-12, rtol=0):
        raise RuntimeError('The cumulative interval masses disagree with the transformed CDF')
    return intensity, Arrival(nll.tolist(), cdf.tolist())


def paired_results(references, paths, raw_intensity, calibrated_intensity, archived):
    rank_counts = {name: dict(wins=0, ties=0, losses=0) for name in ('token_rank', CALIBRATED)}
    exact_changes = tolerance_changes = 0
    for cases in references.values():
        for early, late in combinations(sorted(cases, key=lambda case: (case.ends[1], case.identifier)), 2):
            stop = early.bins[1]
            if early.ends[1] > late.first_start or stop >= late.bins[0]:
                continue
            left, right = early.identifier, late.identifier
            raw_order = np.sign(raw_intensity[left][stop] - raw_intensity[right][stop])
            new_order = np.sign(calibrated_intensity[left][stop] - calibrated_intensity[right][stop])
            exact_changes += int(raw_order != new_order)
            outcomes = []
            for name, counts in rank_counts.items():
                delta = paths[left][name].cdf[stop] - paths[right][name].cdf[stop]
                outcome = 'ties' if abs(delta) <= TIE_ATOL else ('wins' if delta > 0 else 'losses')
                counts[outcome] += 1
                outcomes.append(outcome)
            tolerance_changes += outcomes[0] != outcomes[1]
    pairs = sum(rank_counts['token_rank'].values())
    if pairs != archived['pairs'] or rank_counts['token_rank'] != archived['same_stop_wins_ties_losses']['token_rank']:
        raise RuntimeError('Raw pair outcomes disagree with the original evaluation')
    return {'pairs': pairs, 'outcomes': rank_counts, 'exact_intensity_order_changes': exact_changes,
            'tolerance_scored_cdf_order_changes': tolerance_changes}


def question_results(references, paths, archived):
    nll, brier, first = ({method: [] for method in METHODS} for _ in range(3))
    articles, observed_first = [], []
    for article, cases in sorted(references.items()):
        for case in sorted(cases, key=lambda item: item.identifier):
            articles.append(article)
            observed_first.append(average([int(end == 0) for end in case.bins]))
            for method in METHODS:
                path = paths[case.identifier][method]
                nll[method].append(nll_at_references(path, case.bins))
                brier[method].append(average([(cdf - int(stop >= end)) ** 2
                                              for end in case.bins for stop, cdf in enumerate(path.cdf)]))
                first[method].append(path.cdf[0])
    results = {'first_stop': {'mean_predicted': {name: average(values) for name, values in first.items()},
                              'observed_reference_fraction': average(observed_first)}}
    for metric, values in (('interval_nll_nats', nll), ('cumulative_brier', brier)):
        means = {name: average(rows) for name, rows in values.items()}
        if any(means[method] != archived[metric]['means'][method] for method in BASELINES):
            raise RuntimeError(f'Raw comparison for {metric} differs from the original evaluation')
        results[metric] = {'means': means, 'calibrated_minus_raw': {
            method: paired_interval([candidate - raw for candidate, raw in
                                     zip(values[CALIBRATED], values[method], strict=True)], articles)
            for method in BASELINES}}
    return results


def run(assay_root, forecast_dir, original_protocol, calibration_protocol, heads, output):
    if digest(calibration_protocol) != PROTOCOL_SHA256:
        raise RuntimeError('Calibration protocol differs from the locked analysis specification')
    plan, hashes, articles, queries, schedules, paths = forecast_inputs(assay_root, forecast_dir, original_protocol, heads)
    assignments = fold_assignment(articles.keys() - PRIOR_MATERIAL_OVERLAP)
    assignment_sha = sha256(json.dumps(assignments, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    outcomes = assay_root / 'outcomes.jsonl'
    references = {article: cases for article, cases in score_endpoints(outcomes, plan, articles, queries, schedules).items()
                  if article in assignments}
    if references.keys() != assignments.keys() or (len(references), sum(map(len, references.values()))) != (165, 555):
        raise RuntimeError('The locked calibration inventory has changed')
    forecast_rows = indexed_rows(forecast_dir / 'predictions.jsonl', {'id', 'article_id', 'stops', 'methods'})
    raw_intensity = {identifier: np.cumsum(-np.log1p(-np.asarray(row['methods']['token_rank'], dtype=float)))
                     for identifier, row in forecast_rows.items() if row['article_id'] in assignments}
    interval_bounds = {identifier: log_intervals(intensity) for identifier, intensity in raw_intensity.items()}
    folds, calibrated_intensity = [], {}
    for fold in range(5):
        estimate, objective, train_count, iterations = fit_other_articles(fold, assignments, references, interval_bounds)
        held = [article for article in assignments if assignments[article] == fold]
        held_count = sum(len(references[article]) for article in held)
        for article in held:
            for case in references[article]:
                identifier = case.identifier
                intensity, path = calibrated_path(estimate, interval_bounds[identifier])
                calibrated_intensity[identifier] = intensity
                paths[identifier][CALIBRATED] = path
        folds.append({'fold': fold, 'test_articles': len(held), 'test_questions': held_count, 'fit_questions': train_count,
                      'a': float(estimate[0]), 'b': float(np.exp(estimate[1])),
                      'fit_interval_nll_nats': float(objective), 'fit_iterations': iterations})
    if calibrated_intensity.keys() != raw_intensity.keys():
        raise RuntimeError('The calibrated predictions do not exactly cover the locked primary sample')

    original_scores = forecast_dir / 'scores.json'
    archived = json.loads(original_scores.read_text(encoding='utf-8'))
    result = {
        'scope': 'Retrospective article-grouped out-of-fold calibration of frozen evidence-arrival forecasts',
        'articles': len(assignments), 'questions': len(calibrated_intensity), 'references': 2 * len(calibrated_intensity),
        'folds': folds, 'fold_assignment_sha256': assignment_sha,
        'metrics': question_results(references, paths, archived['per_question']),
        'same_stop': paired_results(references, paths, raw_intensity, calibrated_intensity, archived['same_article']),
        'valid_probability_laws': len(calibrated_intensity),
        'bootstrap': {'draws': DRAWS, 'seed': SEED, 'unit': 'article', 'conditional_on_fixed_fold_predictions': True},
        'input_sha256': {**hashes, 'outcomes_sha256': digest(outcomes), 'original_scores_sha256': digest(original_scores),
                         'calibration_protocol_sha256': PROTOCOL_SHA256, 'calibrator_sha256': digest(Path(__file__))},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('assay-root', 'forecast-dir', 'original-protocol', 'calibration-protocol', 'heads', 'output'):
        parser.add_argument(f'--{option}', type=Path, required=True)
    start = perf_counter()
    run(**vars(parser.parse_args()))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mib = rss / (1024 ** (2 if sys.platform == 'darwin' else 1))
    print(f'Calibration complete in {perf_counter() - start:.2f} s, max process RSS {mib:.1f} MiB')
