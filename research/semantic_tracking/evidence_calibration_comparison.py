"""Give each frozen evidence readout the same ExpMRC calibration and freeze transfer weights."""

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

from expmrc_calibrate import average, calibrated_path, fit_other_articles, fold_assignment, log_intervals
import numpy as np

from expmrc_score import (
    DRAWS, METHODS, PRIOR_MATERIAL_OVERLAP, SEED, digest, forecast_inputs, indexed_rows, nll_at_references,
    paired_interval, score_endpoints,
)


PROTOCOL = Path(__file__).with_name('evidence-calibration-comparison.md')
BASE_FORECAST_PROTOCOL = Path(__file__).with_name('expmrc-transfer.md')
COMPARISONS = ('question_only', 'pooled_rank', 'position')


def available_cases(assay, forecast, heads):
    plan, hashes, articles, queries, schedule, paths = forecast_inputs(assay, forecast, BASE_FORECAST_PROTOCOL, heads)
    outcomes = assay / 'outcomes.jsonl'
    references = score_endpoints(outcomes, plan, articles, queries, schedule)
    references = {article: cases for article, cases in sorted(references.items()) if article not in PRIOR_MATERIAL_OVERLAP}
    cases = [(article, case) for article, rows in references.items() for case in sorted(rows, key=lambda row: row.identifier)]
    if (len(references), len(cases), sum(len(case.bins) for _, case in cases)) != (165, 555, 1110):
        raise ValueError('Unexpected ExpMRC reference counts for symmetric calibration')
    rows = indexed_rows(forecast / 'predictions.jsonl', {'id', 'article_id', 'stops', 'methods'})
    hashes.update(outcomes_sha256=digest(outcomes), comparison_protocol_sha256=digest(PROTOCOL),
                  comparison_code_sha256=digest(Path(__file__)))
    return references, cases, rows, paths, hashes


def statistics(path, case):
    return {
        'interval_nll_nats': nll_at_references(path, case.bins),
        'cumulative_brier': average([(cdf - int(stop >= end)) ** 2
                                     for end in case.bins for stop, cdf in enumerate(path.cdf)]),
        'first_stop_probability': path.cdf[0],
    }


def same_stop(references, original, transformed):
    outcomes, changes = {'win': 0, 'loss': 0, 'tie': 0}, 0
    for rows in references.values():
        for early, late in combinations(sorted(rows, key=lambda case: (case.ends[1], case.identifier)), 2):
            stop = early.bins[1]
            if early.ends[1] > late.first_start or stop >= late.bins[0]:
                continue
            a, b = early.identifier, late.identifier
            raw_order = np.sign(original[a][stop] - original[b][stop])
            adjusted_order = np.sign(transformed[a][stop] - transformed[b][stop])
            outcomes['win' if raw_order > 0 else 'loss' if raw_order < 0 else 'tie'] += 1
            changes += int(raw_order != adjusted_order)
    if sum(outcomes.values()) != 698 or changes:
        raise ValueError('A monotonic calibration changed a registered within-article ranking')
    return {'raw_intensity_order': outcomes, 'calibrated_intensity_order_changes': changes}


def parameters(estimate, objective, count, iterations):
    return {'a': float(estimate[0]), 'b': float(np.exp(estimate[1])), 'training_nll_nats': float(objective),
            'fit_questions': count, 'iterations': iterations}


def run(assay, forecast, heads, output):
    references, cases, rows, original, hashes = available_cases(assay, forecast, heads)
    assignments = fold_assignment(references)
    summary, adjusted_metrics, transfer_weights = {}, {}, {}
    groups = [article for article, _ in cases]
    for method in METHODS:
        raw_intensity = {case.identifier: np.cumsum(-np.log1p(-np.asarray(rows[case.identifier]['methods'][method])))
                         for _, case in cases}
        bounds = {identifier: log_intervals(intensity) for identifier, intensity in raw_intensity.items()}
        fold_results = [fit_other_articles(fold, assignments, references, bounds) for fold in range(5)]
        calibrated = {case.identifier: calibrated_path(fold_results[assignments[article]][0], bounds[case.identifier])
                      for article, case in cases}
        calibrated_stats = [statistics(calibrated[case.identifier][1], case) for _, case in cases]
        original_stats = [statistics(original[case.identifier][method], case) for _, case in cases]
        adjusted_metrics[method] = {metric: [row[metric] for row in calibrated_stats] for metric in calibrated_stats[0]}
        summary[method] = {
            'folds': [parameters(*entry) for entry in fold_results],
            'means': {metric: {'raw': average([row[metric] for row in original_stats]), 'calibrated': average(values)}
                      for metric, values in adjusted_metrics[method].items()},
            'same_stop': same_stop(references, raw_intensity, {key: result[0] for key, result in calibrated.items()}),
        }
        transfer_weights[method] = parameters(*fit_other_articles(None, assignments, references, bounds))

    comparisons = {}
    for baseline in COMPARISONS:
        comparisons[baseline] = {
            metric: paired_interval([token - other for token, other in zip(adjusted_metrics['token_rank'][metric],
                                                                          adjusted_metrics[baseline][metric],
                                                                          strict=True)], groups)
            for metric in ('interval_nll_nats', 'cumulative_brier')
        }
    artifact = {
        'articles': len(references), 'questions': len(cases), 'references': 2 * len(cases),
        'scope': 'retrospective article-grouped out-of-fold, with identical calibration models in every arm',
        'input_sha256': hashes,
        'fold_assignment_sha256': sha256(json.dumps(assignments, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'methods': summary, 'calibrated_token_minus_calibrated': comparisons,
        'resampling': {'draws': DRAWS, 'seed': SEED, 'group': 'article'},
        'frozen_exp_mrc_transfer_weights': {'use': 'new wholly held-out corpora only', 'methods': transfer_weights},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assay', 'forecast', 'heads', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    start = perf_counter()
    run(**vars(parser.parse_args()))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mib = rss / (1024 ** (2 if sys.platform == 'darwin' else 1))
    print(f'Calibration completed in {perf_counter() - start:.2f} s, max process RSS {mib:.1f} MiB')
