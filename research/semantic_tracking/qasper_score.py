"""Score sealed evidence-arrival forecasts on held-out QASPER papers."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from research.core_features.readback_score import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, digest, paired_material_interval
from research.stage3.run import save_json

from .joint_score import interval_nll, probability_logits, rows_by_key


METHODS = ('position', 'body')
TRUTH_FIELDS = {'paper_id', 'question_id', 'lower', 'upper'}
TIE_ATOL = 1e-12


def row_key(row):
    paper, question = row['paper_id'], row['question_id']
    if not all(isinstance(value, str) and value for value in (paper, question)):
        raise ValueError('Expected nonempty paper and question identifiers')
    return paper, question


def validate_inputs(outcomes, predictions, methods=METHODS):
    by_paper = defaultdict(list)
    if outcomes.keys() != predictions.keys():
        raise ValueError('Forecasts must cover exactly the planned held-out question inventory')
    forecast_fields = {'paper_id', 'question_id', 'stops', *methods}
    for key in sorted(outcomes):
        truth, prediction = outcomes[key], predictions[key]
        if truth.keys() != TRUTH_FIELDS or prediction.keys() != forecast_fields:
            raise ValueError(f'Unexpected holdout or forecast fields for {key!r}')
        stops = prediction['stops']
        lower, upper = truth['lower'], truth['upper']
        if (not isinstance(stops, list) or not stops or any(type(stop) is not int or stop < 1 for stop in stops)
                or stops != sorted(set(stops))):
            raise ValueError(f'Expected ordered positive paragraph stops for {key!r}')
        if (type(lower) is not int or type(upper) is not int or not 0 <= lower < upper <= len(stops)
                or upper != lower + 1):
            raise ValueError(f'Expected one uniquely aligned evidence paragraph for {key!r}')
        logits = {name: probability_logits(prediction[name], (len(stops),)) for name in methods}
        by_paper[key[0]].append((truth, stops, logits))

    for paper, questions in by_paper.items():
        _, first_stops, first = questions[0]
        for _, stops, logits in questions[1:]:
            if stops != first_stops:
                raise ValueError(f'Swaps require the same visible paragraph trajectory in {paper!r}')
            if not np.allclose(logits['position'], first['position'], rtol=0.0, atol=TIE_ATOL):
                raise ValueError(f'Position baseline unexpectedly depends on question identity in {paper!r}')
    return by_paper


def contrasts(losses, groups):
    pairs = (('body', 'position'), ('body', 'swapped'))
    return {f'{left}_minus_{right}': paired_material_interval(
        np.asarray(losses[left]) - np.asarray(losses[right]), groups,
    ) for left, right in pairs if right in losses}


def interval_scores(by_paper):
    losses = {name: [] for name in METHODS}
    groups = []
    for paper, questions in by_paper.items():
        for truth, _, logits in questions:
            groups.append({'group_id': paper})
            for name in METHODS:
                losses[name].append(interval_nll(logits[name], truth['lower'], truth['upper']))
    return {
        'questions': len(groups),
        'papers': len(by_paper),
        'mean_interval_nll_nats': {name: float(np.mean(values)) for name, values in losses.items()},
        'paired_interval_nll_differences': contrasts(losses, groups),
    }


def arrival_at_cutoff(logits, cutoff):
    """Return local hazard and total evidence-arrival probability by a body paragraph."""
    log_survival = -np.logaddexp(0.0, logits[:cutoff]).sum()
    hazard = np.exp(-np.logaddexp(0.0, -logits[cutoff - 1]))
    return hazard, -np.expm1(log_survival)


def agreement_summary(pairs, groups):
    probabilities = np.asarray(pairs)
    correct, swapped = probabilities.T
    delta = correct - swapped
    win, loss = delta > TIE_ATOL, delta < -TIE_ATOL
    agreement = np.where(win, 1.0, np.where(loss, 0.0, 0.5))
    return {
        'correct_question_probability_mean': float(correct.mean()),
        'swapped_question_probability_mean': float(swapped.mean()),
        'position_only_agreement': 0.5,
        'body_agreement': float(agreement.mean()),
        'wins_ties_losses': {'wins': int(win.sum()), 'ties': int((~win & ~loss).sum()), 'losses': int(loss.sum())},
        'body_agreement_minus_position': paired_material_interval(agreement - 0.5, groups),
        'body_correct_minus_swapped_probability': paired_material_interval(delta, groups),
    }


def same_paper_scores(by_paper):
    losses = {name: [] for name in (*METHODS, 'swapped')}
    comparisons = {'hazard_at_cutoff': [], 'cumulative_arrival_by_cutoff': []}
    groups = []
    matched_questions = set()
    tied_or_overlapping = 0
    for paper, questions in by_paper.items():
        ordered = sorted(questions, key=lambda record: (record[0]['upper'], record[0]['question_id']))
        for (early, _, early_logits), (late, _, late_logits) in combinations(ordered, 2):
            cutoff = early['upper']
            if cutoff > late['lower']:
                tied_or_overlapping += 1
                continue
            groups.append({'group_id': paper})
            matched_questions.update((row_key(early), row_key(late)))
            # Each pair contributes one mean over the two directions, for every arm.
            for name in METHODS:
                losses[name].append(0.5 * (interval_nll(early_logits[name], early['lower'], cutoff)
                                           + interval_nll(late_logits[name], late['lower'], late['upper'])))
            losses['swapped'].append(0.5 * (interval_nll(late_logits['body'], early['lower'], cutoff)
                                            + interval_nll(early_logits['body'], late['lower'], late['upper'])))
            early_hazard, early_cumulative = arrival_at_cutoff(early_logits['body'], cutoff)
            late_hazard, late_cumulative = arrival_at_cutoff(late_logits['body'], cutoff)
            comparisons['hazard_at_cutoff'].append((early_hazard, late_hazard))
            comparisons['cumulative_arrival_by_cutoff'].append((early_cumulative, late_cumulative))

    if len({row['group_id'] for row in groups}) < 2:
        raise ValueError('Expected eligible same-prefix question swaps in at least two held-out papers')
    return {
        'pairs': len(groups),
        'papers': len({row['group_id'] for row in groups}),
        'unique_questions': len(matched_questions),
        'directional_swaps': 2 * len(groups),
        'pairs_without_strict_evidence_order': tied_or_overlapping,
        'mean_interval_nll_nats_per_question_direction': {name: float(np.mean(values))
                                                         for name, values in losses.items()},
        'paired_interval_nll_differences': contrasts(losses, groups),
        'same_cutoff_ranking': {name: agreement_summary(pairs, groups) for name, pairs in comparisons.items()},
    }


def run(outcomes, predictions, seal, output):
    metadata = json.loads(seal.read_text(encoding='utf-8'))
    prediction_hash = digest(predictions)
    if metadata['predictions_sha256'] != prediction_hash:
        raise ValueError('Forecasts differ from the pre-scoring seal')
    by_paper = validate_inputs(rows_by_key(outcomes, row_key), rows_by_key(predictions, row_key))
    result = {
        'scope': ('Annotated textual evidence arrival for a supplied question. At matched cutoffs the selected '
                  'evidence paragraph for the comparison question has not arrived. The outcome measures the '
                  'finished evidence paragraph, without a real-time observation of readers noticing an answer.'),
        'all_questions': interval_scores(by_paper),
        'same_paper_swaps': same_paper_scores(by_paper),
        'bootstrap': {'draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED, 'unit': 'held-out paper'},
        'input_sha256': {'outcomes': digest(outcomes), 'predictions': prediction_hash, 'seal': digest(seal)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('outcomes', 'predictions', 'seal', 'output'):
        parser.add_argument(f'--{field}', type=Path, required=True)
    run(**vars(parser.parse_args()))
