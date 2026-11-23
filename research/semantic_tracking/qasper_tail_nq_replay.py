"""Apply frozen QASPER tail parameters to already scored Natural Questions forecasts."""

import argparse
import json
from math import exp, fsum
from pathlib import Path

from research.semantic_tracking.evidence_tail_calibrate import PathLayout, probability_path
from research.semantic_tracking.expmrc_score import METHODS, digest, indexed_rows, paired_interval

from nq_evidence_score import read_forecasts, read_sealed_outcomes, references_by_article


def mean(values):
    return fsum(values) / len(values)


def replay(assay, forecasts_dir, outcomes, fit, qasper_scores, output):
    qasper = json.loads(qasper_scores.read_text(encoding='utf-8'))
    trained = json.loads(fit.read_text(encoding='utf-8'))
    plan, inputs, articles, queries, stops, _ = read_forecasts(assay, forecasts_dir)
    if (qasper['input_sha256']['fit_receipt_sha256'] != digest(fit)
            or trained['source_hashes']['heads_sha256'] != qasper['input_sha256']['heads_sha256']
            or trained['source_hashes']['heads_sha256'] != inputs['heads_sha256']):
        raise ValueError('The frozen tail calibrators do not match the scored QASPER fit and NQ readouts')
    selected, observations = read_sealed_outcomes(outcomes, plan, queries, inputs)
    references, counts = references_by_article(assay, selected, observations, articles, queries, stops)
    paths = indexed_rows(forecasts_dir / 'predictions.jsonl', {'id', 'article_id', 'stops', 'methods'})
    cases = [case for rows in references.values() for case in rows]
    comparisons = {}
    for method in METHODS:
        metrics = {}
        for model in ('clock', 'clock_plateau'):
            estimate = trained['fitted_methods'][method][model]['estimate']
            by_scope = {scope: [] for scope in ('all', 'event', 'null')}
            brier, tail = [], []
            for case in cases:
                path = probability_path(estimate, PathLayout.from_hazards(paths[case.identifier]['methods'][method]))
                nll, errors = [], []
                for box in case.references:
                    position = None if box is None else box.stop
                    value = -float(path.log_tail if box is None else path.log_event[position])
                    nll.append(value)
                    by_scope['null' if box is None else 'event'].append(value)
                    errors.append(mean([(float(p) - int(position is not None and index >= position)) ** 2
                                        for index, p in enumerate(path.cdf)]))
                by_scope['all'].append(mean(nll))
                brier.append(mean(errors))
                tail.append(exp(path.log_tail))
            metrics[model] = {'all_by_question': by_scope.pop('all'), 'nll_by_reference': by_scope,
                              'brier': mean(brier), 'mean_terminal_survival': mean(tail)}
        diff = [higher - lower for higher, lower in zip(metrics['clock_plateau']['all_by_question'],
                                                        metrics['clock']['all_by_question'], strict=True)]
        comparisons[method] = {'plateau_minus_clock_nll': paired_interval(diff, [case.family for case in cases])}
        for model, result in metrics.items():
            overall = mean(result.pop('all_by_question'))
            per_ref = {name: {'reference_count': len(values), 'nll_nats': mean(values) if values else None}
                       for name, values in result.pop('nll_by_reference').items()}
            comparisons[method][model] = {'mean_question_nll_nats': overall, 'by_reference_type': per_ref, **result}
    result = {'scope': 'Previously scored Natural Questions material; apply parameters fitted on QASPER only.',
              'inventory': counts, 'methods': comparisons, 'input_sha256': {**inputs, 'qasper_fit_sha256': digest(fit),
                                                                           'qasper_scores_sha256': digest(qasper_scores)}}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assay', 'forecasts-dir', 'outcomes', 'fit', 'qasper-scores', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    replay(**vars(parser.parse_args()))
