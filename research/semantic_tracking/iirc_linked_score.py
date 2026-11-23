"""Score held-out evidence-arrival forecasts after verifying their seal."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.core_features.readback_score import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, paired_material_interval
from research.semantic_tracking.iirc_linked_pack import BRIER_TOLERANCE, METHODS, NLL_MINIMUM_GAIN
from research.semantic_tracking.iirc_prepare import digest
from research.semantic_tracking.joint_score import interval_nll, probability_logits, rows_by_key
from research.stage3.run import save_json


def score_case(forecast, outcome, linked_stops):
    lower, upper = outcome['lower'], outcome['upper']
    if type(lower) is not int or not 0 <= lower <= linked_stops:
        raise ValueError('Invalid lower evidence bound')
    if (upper is None and lower != linked_stops
            or upper is not None and (type(upper) is not int or upper != lower + 1 or upper > linked_stops)):
        raise ValueError('Invalid observed evidence interval or censoring point')
    if set(forecast) != {'case_id', 'methods'} or set(forecast['methods']) != set(METHODS):
        raise ValueError('Forecast methods differ from the sealed linked-evidence assay')
    observed = np.zeros(linked_stops, dtype=bool)
    if upper is not None:
        observed[upper - 1:] = True

    result = {}
    for method, hazards in forecast['methods'].items():
        logits = probability_logits(hazards, (linked_stops,))
        cdf = -np.expm1(-np.logaddexp(0, logits).cumsum())
        error = cdf - observed
        masses = np.diff(np.r_[0.0, cdf, 1.0])
        target = upper - 1 if upper is not None else linked_stops
        tied = np.isclose(masses, masses.max(), rtol=0, atol=1e-12)
        result[method] = {
            'interval_nll_nats': interval_nll(logits, lower, upper),
            'cumulative_brier': float((error ** 2).mean()),
            'terminal_calibration_in_large': float(error[-1]),
            'modal_exact_bin_or_censor': float(tied[target] / tied.sum()),
        }
    return result


def summarize(cases, outcomes):
    groups = [{'group_id': outcome['article_cluster']} for outcome in outcomes]
    methods = {method: {metric: np.array([case[method][metric] for case in cases])
                        for metric in cases[0][method]} for method in METHODS}
    metrics = {method: {metric: float(values.mean()) for metric, values in scores.items()}
               for method, scores in methods.items()}
    contrasts = {}
    for baseline in ('geometry', 'path_text'):
        contrasts[baseline] = {metric: paired_material_interval(values - methods[baseline][metric], groups)
                               for metric, values in methods['anchor_conditioned'].items()}

    def upper(baseline, metric):
        return contrasts[baseline][metric]['descriptive_paired_material_95_ci'][1]

    gates = {
        'anchor_conditioned_improves_interval_nll_vs_geometry': upper('geometry', 'interval_nll_nats') <= -NLL_MINIMUM_GAIN,
        'anchor_conditioned_cumulative_brier_noninferior_to_geometry': upper('geometry', 'cumulative_brier') <= BRIER_TOLERANCE,
        'explicit_anchor_improves_interval_nll_vs_path_text': upper('path_text', 'interval_nll_nats') < 0,
    }
    return {
        'metrics': metrics, 'paired_anchor_conditioned_minus_baseline': contrasts,
        'conditions': gates, 'all_conditions_passed': all(gates.values()),
    }


def run(assay, stimuli, outcomes, swaps, forecasts, seal, output):
    provenance = json.loads(assay.read_text(encoding='utf-8'))
    receipt = json.loads(seal.read_text(encoding='utf-8'))
    if (receipt['predictions_sha256'] != digest(forecasts) or receipt['assay_sha256'] != digest(assay)
            or receipt['model'] != provenance['model'] or receipt['revision'] != provenance['revision']):
        raise ValueError('Forecasts, assay, or model disagree with the pre-scoring seal')
    if (digest(stimuli) != provenance['inputs_sha256']['dev/stimuli.jsonl']
            or digest(outcomes) != provenance['held_out_not_in_archive']['dev/locked-labels.jsonl']
            or digest(swaps) != provenance['held_out_not_in_archive']['dev/locked-matched-swaps.jsonl']):
        raise ValueError('Held-out inputs or outcomes differ from the packaged assay')

    planned = rows_by_key(stimuli, lambda row: row['id'])
    predicted = rows_by_key(forecasts, lambda row: row['case_id'])
    truth = rows_by_key(outcomes, lambda row: row['case_id'])
    if planned.keys() != predicted.keys() or planned.keys() != truth.keys() or len(planned) != provenance['cases']['dev']:
        raise ValueError('Held-out case IDs or counts disagree across sources')
    cases = []
    ordered_outcomes = []
    for case_id, stimulus in planned.items():
        row = truth[case_id]
        stops = stimulus['stops']
        cases.append(score_case(predicted[case_id], row, len(stops) - 1))
        ordered_outcomes.append(row)
    observed = sum(row['upper'] is not None for row in ordered_outcomes)
    with swaps.open(encoding='utf-8') as handle:
        pairs = sum(bool(line.strip()) for line in handle)
    result = {
        'scope': 'Linked retrospective text-support arrival given a supplied IIRC writer anchor.',
        'observed_intervals': observed, 'right_censored': len(cases) - observed,
        'strict_same_document_swaps': pairs,
        'identity_swap_available': pairs > 0,
        'bootstrap': {'seed': BOOTSTRAP_SEED, 'draws': BOOTSTRAP_DRAWS, 'unit': 'connected article component'},
        'assay_sha256': digest(assay), 'forecasts_sha256': digest(forecasts), 'seal_sha256': digest(seal),
        **summarize(cases, ordered_outcomes),
    }
    save_json(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assay', 'stimuli', 'outcomes', 'swaps', 'forecasts', 'seal', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
