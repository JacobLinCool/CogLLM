"""Score matched semantic checkpoints against the existing held-out observations."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from research.core_features.readback_score import average_precision, digest, paired_material_interval
from research.stage3.probe import _jsonl
from research.stage3.run import save_json


METHODS = ('independent', 'shared_two', 'shared_three')
CONTRASTS = (('shared_two', 'independent'), ('shared_three', 'independent'), ('shared_three', 'shared_two'))
FORECAST_NAMES = ('onset-predictions.jsonl', 'source-predictions.jsonl', 'closure-predictions.jsonl')


def probability_logits(probs, shape):
    array = np.asarray(probs, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array) & (0 < array) & (array < 1)):
        raise ValueError('Expected finite probabilities strictly between zero and one')
    return np.log(array) - np.log1p(-array)


def interval_nll(logits, definitely_open, first_closed):
    """Score T in (L, U] or T > L, indexing opportunities after opening from zero."""
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError('Expected one finite closure logit per opportunity')
    if not 0 <= definitely_open <= len(array):
        raise ValueError('Last definitely open position is outside forecast opportunities')
    if first_closed is not None and not definitely_open < first_closed <= len(array):
        raise ValueError('Observed closure interval must follow opening')
    log_survival_at_line = -np.logaddexp(0.0, array)
    before = -float(log_survival_at_line[:definitely_open].sum())
    if first_closed is None:
        return before
    # Summing the first-hit probabilities in the interval stays finite even
    # when all interval hazards are too small for subtraction from one.
    scope = slice(definitely_open, first_closed)
    through_previous = np.r_[0.0, log_survival_at_line[scope][:-1].cumsum()]
    log_event = -np.logaddexp(0.0, -array[scope])
    return before - float(np.logaddexp.reduce(through_previous + log_event))


def event_losses(logits, labels):
    """Return finite scores without clipping small or large probabilities."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if logits.shape != labels.shape or logits.ndim != 1 or not np.all(np.isfinite(logits)):
        raise ValueError('Expected one finite event logit per binary observation')
    probability = np.exp(-np.logaddexp(0.0, -logits))
    return {
        'nll_nats': np.logaddexp(0.0, np.where(labels, -logits, logits)),
        'brier': (probability - labels) ** 2,
    }


def source_losses(logits, marked):
    """Compute the annotated-span mass and expected word rank with ties."""
    logits = np.asarray(logits, dtype=np.float64)
    marked = np.asarray(marked, dtype=bool)
    if (logits.shape != marked.shape or logits.ndim != 1 or not marked.any()
            or not np.all(np.isfinite(logits))):
        raise ValueError('Expected finite pointer logits for each candidate and an observed target')
    log_normalizer = np.logaddexp.reduce(logits)
    probability = np.exp(logits - log_normalizer)
    top = np.isclose(probability, probability.max(), rtol=0.0, atol=1e-12)
    best_gold = probability[marked].max()
    tied = np.isclose(probability, best_gold, rtol=0.0, atol=1e-12)
    first = int(np.sum(probability > best_gold + 1e-12)) + 1
    correct_in_tie = int(np.sum(marked & tied))
    tie_size = int(tied.sum())
    # The first correct element of a randomly ordered tie is at offset j
    # with probability C(n-j, k-1)/C(n, k).
    reciprocal = math.fsum(
        math.comb(tie_size - j, correct_in_tie - 1) / math.comb(tie_size, correct_in_tie) / (first + j - 1)
        for j in range(1, tie_size - correct_in_tie + 2)
    )
    return {
        'source_span_nll': float(log_normalizer - np.logaddexp.reduce(logits[marked])),
        'top_word_in_source': float((marked & top).sum() / top.sum()),
        'reciprocal_rank': reciprocal,
    }


def rows_by_key(path, key):
    indexed = {}
    for row in _jsonl(path):
        value = key(row)
        if value in indexed:
            raise ValueError(f'Duplicate row {value!r} in {path}')
        indexed[value] = row
    if not indexed:
        raise ValueError(f'Expected nonempty observations in {path}')
    return indexed


def event_summary(labels, rows, forecasts, contrasts):
    if not 0 < np.sum(labels) < len(labels):
        raise ValueError('Both event classes are required on the held-out visits')
    values = {name: event_losses(logits, labels) for name, logits in forecasts.items()}
    result = {'visits': len(rows), 'events': int(np.sum(labels)), 'metrics': {}, 'paired': {}}
    for name, metrics in values.items():
        result['metrics'][name] = {field: float(value.mean()) for field, value in metrics.items()}
        result['metrics'][name]['average_precision'] = average_precision(np.asarray(forecasts[name]), labels)
    for new, old in contrasts:
        result['paired'][f'{new}_minus_{old}'] = {
            metric: paired_material_interval(values[new][metric] - values[old][metric], rows)
            for metric in values[new]
        }
    return result


def source_summary(rows, forecasts, contrasts):
    scores = defaultdict(list)
    for row in rows:
        candidate = np.asarray(row['source_candidates'])
        marked = (row['source_start'] < candidate) & (candidate <= row['source_stop'])
        for name, logits in forecasts[row['row_id']].items():
            scores[name].append(source_losses(logits, marked))
    losses = {name: {key: np.asarray([value[key] for value in records]) for key in records[0]}
              for name, records in scores.items()}
    return {
        'annotations': len(rows),
        'families': len({row['group_id'] for row in rows}),
        'metrics': {name: {key: float(value.mean()) for key, value in metrics.items()}
                    for name, metrics in losses.items()},
        'paired': {f'{new}_minus_{old}': {
            key: paired_material_interval(losses[new][key] - losses[old][key], rows) for key in losses[new]
        } for new, old in contrasts},
    }


def onset_score(outcomes, predictions):
    rows = list(_jsonl(outcomes))
    if not rows or len({(row['trial_id'], row['stop']) for row in rows}) != len(rows):
        raise ValueError('Expected distinct held-out trial visits')
    if any(row.get('split') != 'test' or row.get('initial_risk') is not True or type(row.get('readback')) is not bool
           for row in rows):
        raise ValueError('Expected observed binary movement at previously forward-only stops')

    def position(row):
        return row['id'], row['stop']

    forecasts = rows_by_key(predictions, position)
    if forecasts.keys() != {position(row) for row in rows}:
        raise ValueError('Onset forecast positions differ from the planned held-out observations')
    if any(row.keys() != {'id', 'stop', *METHODS} for row in forecasts.values()):
        raise ValueError('Unexpected fields in the onset forecast')
    summary = {}
    subgroups = {
        'single_word': [row for row in rows if row['region_words'] == 1],
        'all': rows,
        'single_word_critical_verb': [row for row in rows if row['region_words'] == 1 and row['critical']],
    }
    for key, selected in subgroups.items():
        labels = np.asarray([row['readback'] for row in selected], dtype=bool)
        by_method = {name: probability_logits([forecasts[position(row)][name] for row in selected], labels.shape)
                     for name in METHODS}
        summary[key] = event_summary(labels, selected, by_method, CONTRASTS)
    return {'heldout_families': len({row['group_id'] for row in rows}), 'anchors': len(forecasts), **summary}


def score_source(outcomes, predictions):
    gold = rows_by_key(outcomes, lambda row: row['row_id'])
    predictions = rows_by_key(predictions, lambda row: row['row_id'])
    if gold.keys() != predictions.keys():
        raise ValueError('Source forecasts differ from the planned held-out annotation rows')
    fields = ('row_id', 'id', 'group_id', 'stop', 'source_candidates')
    logits = {}
    for key, prediction in predictions.items():
        truth = gold[key]
        if (truth.get('split') != 'test' or prediction.keys() != {*fields, *METHODS}
                or any(truth[field] != prediction[field] for field in fields)):
            raise ValueError(f'Changed or invalid source candidate set {key!r}')
        candidates = prediction['source_candidates']
        if candidates != sorted(set(candidates)) or any(not 0 < p < truth['stop'] for p in candidates):
            raise ValueError('Source candidates must be distinct words before the marked stop')
        for name in METHODS:
            probability = np.asarray(prediction[name], dtype=np.float64)
            if (probability.shape != (len(candidates),) or not np.all(np.isfinite(probability) & (probability > 0))
                    or not math.isclose(float(probability.sum()), 1.0, abs_tol=2e-6, rel_tol=0.0)):
                raise ValueError(f'Expected normalized {name!r} source distribution')
            logits.setdefault(key, {})[name] = np.log(probability)
    return source_summary(list(gold.values()), logits, CONTRASTS)


def closure_score(outcomes, predictions):
    gold = rows_by_key(outcomes, lambda row: row['id'])
    forecasts = rows_by_key(predictions, lambda row: row['id'])
    if gold.keys() != forecasts.keys():
        raise ValueError('Closure forecasts differ from the planned held-out question trajectories')
    by_story = defaultdict(list)
    closed_by_story = defaultdict(int)
    broader_intervals = defaultdict(int)
    zero_followup = defaultdict(int)
    coordinates = ('story', 'open_line', 'last_line')
    for key in sorted(gold):
        row, prediction = gold[key], forecasts[key]
        if prediction.keys() != {'id', *coordinates, *METHODS} or any(row[c] != prediction[c] for c in coordinates):
            raise ValueError(f'Forecast reading coordinates changed for {key!r}')
        if prediction['shared_two'] != prediction['independent']:
            raise ValueError('Two-task closure control must reuse the independent closure forecast')
        opening, last = row['open_line'], row['last_line']
        lower, upper = row['lower_line'], row['upper_line']
        if not all(type(line) is int for line in (opening, last, lower)) or not 1 <= opening <= lower <= last:
            raise ValueError(f'Invalid opening or lower closure line for {key!r}')
        if upper is not None and (type(upper) is not int or not lower < upper <= last):
            raise ValueError(f'Invalid upper closure line for {key!r}')
        shape = (last - opening,)
        by_story[row['story']].append([
            interval_nll(probability_logits(prediction[name], shape), lower - opening,
                         None if upper is None else upper - opening)
            for name in METHODS
        ])
        closed_by_story[row['story']] += upper is not None
        broader_intervals[row['story']] += upper is not None and upper - lower > 1
        zero_followup[row['story']] += lower == opening and upper is None
    if len(by_story) != 3:
        raise ValueError('Expected three separate line-annotated story holdouts')
    per_story = {}
    for story, values in sorted(by_story.items()):
        summary = np.mean(values, axis=0)
        per_story[story] = {'questions': len(values), 'observed_closures': closed_by_story[story],
                            'right_censored': len(values) - closed_by_story[story],
                            'confirmed_intervals_wider_than_one_line': broader_intervals[story],
                            'no_observed_post_opening_line': zero_followup[story],
                            'interval_nll_nats_per_question': dict(zip(METHODS, summary.tolist(), strict=True)),
                            'shared_three_minus_independent': float(summary[2] - summary[0])}
    macro = {name: float(np.mean([row['interval_nll_nats_per_question'][name] for row in per_story.values()]))
             for name in METHODS}
    return {
        'outcome': ('Retrospective closure interval or right censoring, conditional on the annotated opening line. '
                    'Coordinates and age only, with no narrative content.'),
        'question_count': len(gold),
        'story_count': len(per_story),
        'per_story': per_story,
        'macro_story_interval_nll_nats_per_question': macro,
        'macro_shared_three_minus_independent': macro['shared_three'] - macro['independent'],
    }


def run(onset, source, closure, forecasts, seal, output):
    metadata = json.loads(seal.read_text(encoding='utf-8'))
    paths = [forecasts / name for name in FORECAST_NAMES]
    hashes = {path.name: digest(path) for path in paths}
    if metadata['forecast_sha256'] != hashes:
        raise ValueError('A forecast differs from the sealed holdout predictions')
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {
        'scope': ('Identical held-out observations across training conditions. Onset measures backward actions at '
                  'reached word or region positions. Source conditions on marked contradiction stops. Closure '
                  'conditions on retrospective annotated start and uses age and position, with no narrative text.'),
        'primary_endpoints': {'onset': 'single_word.brier', 'source': 'source_span_nll',
                              'closure': 'macro_story_interval_nll_nats_per_question'},
        'onset': onset_score(onset, paths[0]),
        'source': score_source(source, paths[1]),
        'closure': closure_score(closure, paths[2]),
        'input_sha256': {'forecasts': hashes, 'onset': digest(onset), 'source': digest(source),
                         'closure': digest(closure), 'seal': digest(seal)},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('onset', 'source', 'closure', 'forecasts', 'seal', 'output'):
        parser.add_argument(f'--{field}', type=Path, required=True)
    run(**vars(parser.parse_args()))
