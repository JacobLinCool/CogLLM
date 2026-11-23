"""Score the frozen QASPER factorial candidate and conditional controls."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from research.core_features.readback_score import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, digest, paired_material_interval
from research.stage3.run import save_json

from .joint_score import interval_nll, rows_by_key
from .qasper_score import TIE_ATOL, row_key, validate_inputs


METHODS = ('position', 'question_only', 'pooled_interval', 'pooled_rank', 'token_interval', 'token_rank')
CANDIDATE = 'token_rank'
BASELINE = 'pooled_interval'
MAX_BRIER_INCREASE = 0.01
MAX_CALIBRATION_BIAS = 0.05
CONTRASTS = (
    ('pooled_rank', BASELINE), ('token_interval', BASELINE), ('token_rank', 'token_interval'),
    ('token_rank', 'pooled_rank'), (CANDIDATE, BASELINE), (CANDIDATE, 'position'), (CANDIDATE, 'question_only'),
)


def arrival_probabilities(logits):
    return -np.expm1(-np.logaddexp(0.0, logits).cumsum())


def paired(values, groups):
    return paired_material_interval(np.asarray(values), groups)


def compare(values, groups):
    values = {method: np.asarray(values[method]) for method in METHODS}
    differences = {f'{left}_minus_{right}': paired(values[left] - values[right], groups)
                   for left, right in CONTRASTS}
    differences['token_by_rank_interaction'] = paired(
        (values['token_rank'] - values['token_interval']) - (values['pooled_rank'] - values[BASELINE]), groups,
    )
    return {'means': {method: float(values[method].mean()) for method in METHODS}, 'contrasts': differences}


def question_metrics(by_paper):
    nll = {name: [] for name in METHODS}
    brier = {name: [] for name in METHODS}
    bias = {name: [] for name in METHODS}
    groups = []
    for paper, questions in by_paper.items():
        for truth, stops, logits in questions:
            groups.append({'group_id': paper})
            observed = np.arange(1, len(stops) + 1) >= truth['upper']
            for name in METHODS:
                nll[name].append(interval_nll(logits[name], truth['lower'], truth['upper']))
                error = arrival_probabilities(logits[name]) - observed
                brier[name].append(float(np.mean(error ** 2)))
                bias[name].append(float(np.mean(error)))
    return {
        'questions': len(groups),
        'interval_nll_nats': compare(nll, groups),
        'cumulative_brier': compare(brier, groups),
        'cumulative_calibration_in_large': {name: paired(values, groups) for name, values in bias.items()},
    }


def pair_metrics(by_paper):
    matched_nll = {name: [] for name in METHODS}
    swapped_nll = {name: [] for name in METHODS}
    accuracy = {name: [] for name in METHODS}
    groups = []
    tied_evidence = 0
    for paper, questions in by_paper.items():
        ordered = sorted(questions, key=lambda item: (item[0]['upper'], item[0]['question_id']))
        for (early, _, early_logits), (late, _, late_logits) in combinations(ordered, 2):
            cutoff = early['upper']
            if cutoff > late['lower']:
                tied_evidence += 1
                continue
            groups.append({'group_id': paper})
            for name in METHODS:
                a, b = early_logits[name], late_logits[name]
                matched_nll[name].append(0.5 * (interval_nll(a, early['lower'], cutoff)
                                                + interval_nll(b, late['lower'], late['upper'])))
                swapped_nll[name].append(0.5 * (interval_nll(b, early['lower'], cutoff)
                                                + interval_nll(a, late['lower'], late['upper'])))
                difference = arrival_probabilities(a)[cutoff - 1] - arrival_probabilities(b)[cutoff - 1]
                accuracy[name].append(0.5 if abs(difference) <= TIE_ATOL else float(difference > 0))
    return {
        'papers': len({row['group_id'] for row in groups}),
        'pairs': len(groups),
        'pairs_without_strict_evidence_order': tied_evidence,
        'interval_nll_nats_per_direction': compare(matched_nll, groups),
        'matched_minus_swapped_interval_nll_nats': {
            name: paired(np.asarray(matched_nll[name]) - swapped_nll[name], groups) for name in METHODS
        },
        'same_stop_accuracy': compare(accuracy, groups),
        'same_stop_wins_ties_losses': {
            method: [values.count(score) for score in (1.0, 0.5, 0.0)] for method, values in accuracy.items()
        },
    }


def acceptance(questions, pairs):
    def bounds(result):
        return result['descriptive_paired_material_95_ci']

    ranking = pairs['same_stop_accuracy']['contrasts']
    brier = questions['cumulative_brier']['contrasts'][f'{CANDIDATE}_minus_{BASELINE}']
    bias = bounds(questions['cumulative_calibration_in_large'][CANDIDATE])
    conditions = {
        'same_stop_superior_to_position': bounds(ranking[f'{CANDIDATE}_minus_position'])[0] > 0,
        'same_stop_superior_to_question_only': bounds(ranking[f'{CANDIDATE}_minus_question_only'])[0] > 0,
        'correct_question_lower_nll_than_swap': bounds(pairs['matched_minus_swapped_interval_nll_nats'][CANDIDATE])[1] < 0,
        'brier_noninferior_to_pooled_interval': bounds(brier)[1] <= MAX_BRIER_INCREASE,
        'calibration_bias_within_tolerance': -MAX_CALIBRATION_BIAS <= bias[0] <= bias[1] <= MAX_CALIBRATION_BIAS,
    }
    return {
        'candidate': CANDIDATE, 'brier_reference': BASELINE, 'max_brier_increase': MAX_BRIER_INCREASE,
        'max_absolute_calibration_bias': MAX_CALIBRATION_BIAS, 'all_conditions_passed': all(conditions.values()),
        'conditions': conditions,
    }


def run(outcomes, predictions, seal, stimuli, output):
    prediction_hash = digest(predictions)
    if json.loads(seal.read_text(encoding='utf-8'))['predictions_sha256'] != prediction_hash:
        raise ValueError('Forecasts differ from the pre-scoring seal')
    by_paper = validate_inputs(rows_by_key(outcomes, row_key), rows_by_key(predictions, row_key), METHODS)
    stimulus_rows = rows_by_key(stimuli, lambda row: row['id'])
    for paper, questions in by_paper.items():
        if questions[0][1] != stimulus_rows[paper]['stops']:
            raise ValueError(f'Forecast paragraph stops differ from the prepared article {paper!r}')
    questions = question_metrics(by_paper)
    pairs = pair_metrics(by_paper)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {
        'scope': 'Arrival of retrospectively marked supporting text for an externally supplied question.',
        'per_question': questions,
        'same_paper': pairs,
        'acceptance': acceptance(questions, pairs),
        'bootstrap': {'draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED, 'unit': 'held-out paper'},
        'input_sha256': {'outcomes': digest(outcomes), 'predictions': prediction_hash,
                         'seal': digest(seal), 'stimuli': digest(stimuli), 'scorer': digest(Path(__file__))},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('outcomes', 'predictions', 'seal', 'stimuli', 'output'):
        parser.add_argument(f'--{field}', type=Path, required=True)
    run(**vars(parser.parse_args()))
