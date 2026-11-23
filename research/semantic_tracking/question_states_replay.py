"""Check the structured likelihood on sealed retrospective question intervals."""

import argparse
from collections import defaultdict
from hashlib import file_digest
import json
from math import exp, fsum, isfinite, log, log1p
from pathlib import Path
from statistics import mean

import torch

from research.semantic_tracking.question_states import (
    STATE_COUNT, TRACE_STATES, first_hit_log_masses, trace_nll, transition_log_probs,
)


METHODS = ('independent', 'shared_two', 'shared_three')
PREDICTIONS = 'closure-predictions.jsonl'
INTERVALS = 'closure-intervals.jsonl'
SCORES = 'joint-scores.json'
UNRESOLVED = TRACE_STATES.index('unresolved')
RESOLVED = TRACE_STATES.index('resolved')
# All replay comparisons use the published per-line scores in float64.
TOLERANCE = 1e-11


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        yield from map(json.loads, stream)


def records(source):
    prior = json.loads((source / SCORES).read_bytes())
    checks = {
        source / PREDICTIONS: prior['input_sha256']['forecasts'][PREDICTIONS],
        source / INTERVALS: prior['input_sha256']['closure'],
        source / 'joint-manifest.json': prior['input_sha256']['seal'],
    }
    if any(digest(path) != expected for path, expected in checks.items()):
        raise ValueError('Question interval replay requires the exact sealed source artifacts')
    examples, ids = [], set()
    for forecast, observed in zip(rows(source / PREDICTIONS), rows(source / INTERVALS), strict=True):
        if (set(forecast) != {'id', 'story', 'open_line', 'last_line', *METHODS}
                or any(forecast[key] != observed[key] for key in ('id', 'story', 'open_line', 'last_line'))
                or observed['id'] in ids):
            raise ValueError('Invalid or repeated story and question interval identity')
        ids.add(observed['id'])
        opening = observed['open_line']
        lower = observed['lower_line'] - opening
        upper = observed['upper_line'] - opening if observed['upper_line'] is not None else None
        length = observed['last_line'] - opening
        if not 0 <= lower <= length or (upper is not None and not lower < upper <= length):
            raise ValueError('Expected a valid line-aligned observation interval after the question opened')
        for method in METHODS:
            probabilities = forecast[method]
            if (len(probabilities) != length
                    or not all(type(value) in (float, int) and isfinite(value) and 0 < value < 1
                               for value in probabilities)):
                raise ValueError('Expected finite interior first-resolution hazards at subsequent lines')
            examples.append((observed['story'], method, lower, upper, probabilities))
    if len(ids) != prior['closure']['question_count']:
        raise ValueError('Expected all sealed retrospective annotation questions')
    return examples, prior['closure']


def direct_nll(probabilities, lower, upper):
    if upper is None:
        return -fsum(log1p(-value) for value in probabilities[:lower])
    survival = 0.0
    masses = []
    for i, value in enumerate(probabilities[:upper]):
        if i >= lower:
            masses.append(survival + log(value))
        survival += log1p(-value)
    largest = max(masses)
    return -(largest + log(fsum(exp(value - largest) for value in masses)))


def score_batch(batch):
    length = max(len(probabilities) for *_, probabilities in batch)
    shape = (len(batch), length)
    logits = torch.full((*shape, STATE_COUNT, STATE_COUNT), -torch.inf, dtype=torch.float64)
    logits.diagonal(dim1=-2, dim2=-1).fill_(0)
    initial = torch.full((len(batch), STATE_COUNT), -torch.inf, dtype=torch.float64)
    initial[:, UNRESOLVED] = 0
    emissions = torch.full((*shape, STATE_COUNT), -torch.inf, dtype=torch.float64)
    visible = torch.zeros(shape, dtype=torch.bool)
    observed = torch.zeros_like(visible)
    for index, (_, _, lower, upper, probabilities) in enumerate(batch):
        count = len(probabilities)
        if count:
            prob = torch.tensor(probabilities, dtype=torch.float64)
            logits[index, :count, UNRESOLVED, RESOLVED] = torch.logit(prob)
            visible[index, :count] = True
        if lower:
            observed[index, lower - 1] = True
            emissions[index, lower - 1, UNRESOLVED] = 0
        if upper is not None:
            observed[index, upper - 1] = True
            emissions[index, upper - 1, RESOLVED] = 0
    transition = transition_log_probs(logits, visible)
    likelihood = trace_nll(transition, initial, emissions, observed)
    masses = first_hit_log_masses(transition, initial, target=RESOLVED)
    interval_scores = torch.stack([
        -masses[i, lower:upper].logsumexp(dim=-1)
        for i, (_, _, lower, upper, _) in enumerate(batch)
    ])
    mass_error = masses.logsumexp(dim=-1).abs().max().item()
    return likelihood.tolist(), interval_scores.tolist(), mass_error


def replay(source):
    examples, prior = records(source)
    by_story = defaultdict(list)
    max_path_error = max_interval_error = max_mass_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(examples), 24):
            batch = examples[start:start + 24]
            path, interval, mass_error = score_batch(batch)
            max_mass_error = max(max_mass_error, mass_error)
            for (story, method, lower, upper, probabilities), path_nll, interval_nll in zip(
                batch, path, interval, strict=True,
            ):
                reference = direct_nll(probabilities, lower, upper)
                max_path_error = max(max_path_error, abs(path_nll - reference))
                max_interval_error = max(max_interval_error, abs(interval_nll - reference))
                by_story[story, method].append(path_nll)
    estimates = {}
    for story, cell in prior['per_story'].items():
        scores = {method: mean(by_story[story, method]) for method in METHODS}
        if any(len(by_story[story, method]) != cell['questions'] for method in METHODS):
            raise ValueError('Cohort count changed in the structured replay')
        estimates[story] = scores
    maximum_summary_delta = max(
        abs(value - prior['per_story'][story]['interval_nll_nats_per_question'][method])
        for story, scores in estimates.items() for method, value in scores.items()
    )
    if max(max_path_error, max_interval_error, max_mass_error, maximum_summary_delta) > TOLERANCE:
        raise RuntimeError('The structured layer does not reproduce the published first-closure likelihood')
    return {
        'source_hashes': {name: digest(source / name) for name in (PREDICTIONS, INTERVALS, SCORES)},
        'retrospective_annotation_questions': prior['question_count'],
        'scored_paths': len(examples),
        'bounded_intervals': sum(cell['observed_closures'] for cell in prior['per_story'].values()),
        'right_censored_intervals': sum(cell['right_censored'] for cell in prior['per_story'].values()),
        'multiline_intervals': sum(cell['confirmed_intervals_wider_than_one_line'] for cell in prior['per_story'].values()),
        'max_abs_nll_delta_from_direct_interval_math': max_interval_error,
        'max_abs_nll_delta_from_state_observation_math': max_path_error,
        'max_abs_first_hit_total_log_mass_error': max_mass_error,
        'max_abs_story_average_delta_from_published': maximum_summary_delta,
        'tolerance': TOLERANCE,
        'story_interval_nll': estimates,
        'scope': ('The existing line-only, first-closure hazards are used as a restricted case. '
                  'No new observations, Qwen forwards, weights, or token-level resolution claims.'),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = replay(args.source)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False, sort_keys=True)
        stream.write('\n')
