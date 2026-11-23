"""Score a sealed, independent first-readback assay and its paired context effect."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import os
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from research.semantic_tracking.context_readback_forecast import (
    METHODS, PREDICTION_FIELDS, digest, read_stimuli,
)
from research.semantic_tracking.context_readback_prepare import SOURCE_SHA256, verified_forecasts
from research.semantic_tracking.first_readback_score import DRAWS, average_precision, interval


CALIBRATION_MARGIN = 0.01
CONTEXT_EFFECT_MARGIN = 0.01
SEED = 20261122
CONTEXTS = {'oneref': 1, 'tworef': 2}
DISAMBIGUATIONS = ('comp', 'rc', 'unamb')
PHASES = ('critical', 'spill', 'other')
TARGET_FIELDS = {'id', 'stop', 'family', 'context', 'disamb', 'region_words', 'phase'}
TRUTH_FIELDS = TARGET_FIELDS | {'positive', 'negative'}
EVENTS_SHA256 = SOURCE_SHA256['fixations.Rda']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_index(path, fields):
    index = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if (not isinstance(row, dict) or row.keys() != fields or type(row['id']) is not str or not row['id']
                    or type(row['stop']) is not int or row['stop'] < 1):
                raise ValueError(f'Unexpected first-readback fields in {path.name}')
            key = row['id'], row['stop']
            if key in index:
                raise ValueError(f'Duplicate display stop in {path.name}')
            index[key] = row
    if not index:
        raise ValueError(f'No display stops in {path.name}')
    return index


def verify_targets(targets, articles):
    variants = set()
    rc_critical = defaultdict(set)
    for article in articles:
        identifier, stops, text = article['id'], article['stops'], article['text']
        rows = [targets[identifier, stop] for stop in stops[1:]]
        first = rows[0]
        family, context, disamb = (first[field] for field in ('family', 'context', 'disamb'))
        variant = family, context, disamb
        if (type(family) not in (str, int) or family == '' or context not in CONTEXTS
                or disamb not in DISAMBIGUATIONS or variant in variants):
            raise ValueError('Expected one independent material version for each registered condition')
        variants.add(variant)
        phase_order = [row['phase'] for row in rows]
        if (phase_order.count('critical') != 1 or phase_order.count('spill') != 1
                or phase_order.index('critical') + 1 != phase_order.index('spill')):
            raise ValueError('The registered critical stop must be followed by one spillover stop')
        for row, previous in zip(rows, stops[:-1], strict=True):
            if (variant != (row['family'], row['context'], row['disamb']) or type(row['family']) is not type(family)
                    or row['phase'] not in PHASES or type(row['region_words']) is not int
                    or row['region_words'] != len(text[previous:row['stop']].split())):
                raise ValueError('Target condition, display phase or region width disagrees with the registered material')
            if row['phase'] == 'critical' and disamb == 'rc':
                rc_critical[family].add(text[previous:row['stop']])
    families = {variant[0] for variant in variants}
    if variants != set(product(families, CONTEXTS, DISAMBIGUATIONS)):
        raise ValueError('Each registered material family must cover all six versions')
    if rc_critical.keys() != families or any(len(displays) != 1 for displays in rc_critical.values()):
        raise ValueError('One- and two-referent relative clauses must share the same critical display within families')


def verify_forecasts(assay_root, predictions_dir):
    plan = read_json(assay_root / 'assay.json')
    articles = read_stimuli(assay_root / 'stimuli.jsonl', plan['file_sha256']['stimuli.jsonl'])
    targets = read_index(assay_root / 'targets.jsonl', TARGET_FIELDS)
    expected = {(row['id'], stop) for row in articles for stop in row['stops'][1:]}
    if expected != targets.keys():
        raise ValueError('The local targets must cover every designed region end')
    _, seal_sha256 = verified_forecasts(assay_root, predictions_dir, articles, targets)
    seal = predictions_dir / 'seal.json'
    receipt = read_json(seal)
    hashes = {name: value for name, value in receipt.items() if name.endswith('_sha256')}
    hashes.update(forecast_seal_sha256=seal_sha256, scorer_sha256=digest(Path(__file__)))
    forecasts = read_index(predictions_dir / 'predictions.jsonl', PREDICTION_FIELDS)
    verify_targets(targets, articles)
    geometry = {
        'stimulus_variants': len(articles), 'families': len({row['family'] for row in targets.values()}),
        'original_stops': sum(len(row['stops']) for row in articles), 'targets': len(targets),
        'targets_by_phase': dict(Counter(row['phase'] for row in targets.values())),
    }
    if any(plan['geometry'][field] != count for field, count in geometry.items()):
        raise ValueError('Exact stimulus, condition and display-stop counts differ from the assay inventory')
    return plan, targets, forecasts, hashes


def verify_truth(path, targets, hashes):
    seal = path.with_name('truth-seal.json')
    receipt = read_json(seal)
    expected = {
        'assay_sha256': hashes['assay_sha256'], 'forecast_seal_sha256': hashes['forecast_seal_sha256'],
        'events_sha256': EVENTS_SHA256,
        'source_code_sha256': digest(Path(__file__).with_name('context_readback_prepare.py')),
    }
    if any(receipt.get(name) != value for name, value in expected.items()):
        raise ValueError('The truth receipt does not bind the archived events, replay code and verified forecast bundle')
    truth_sha = digest(path)
    if receipt.get('truth_sha256') != truth_sha:
        raise ValueError('The decoded truth differs from the post-forecast truth seal')
    outcomes = read_index(path, TRUTH_FIELDS)
    for key, row in outcomes.items():
        if (key not in targets or any(type(row[field]) is not type(value) or row[field] != value
                                     for field, value in targets[key].items())
                or any(type(row[field]) is not int or row[field] < 0 for field in ('positive', 'negative'))
                or row['positive'] + row['negative'] == 0):
            raise ValueError('Observed outcomes do not align with the sealed local design or contain invalid counts')
    hashes.update(truth_sha256=truth_sha, truth_seal_sha256=digest(seal), events_sha256=EVENTS_SHA256,
                  replay_code_sha256=expected['source_code_sha256'])
    return outcomes


def family_draws(count):
    rng = np.random.default_rng(SEED)
    return rng.multinomial(count, np.full(count, 1 / count), size=DRAWS)


def inside_margin(bounds, margin):
    return -margin < bounds[0] <= bounds[1] < margin


def score_population(families, positive, negative, probabilities, *, qwen, position):
    """Score individual opportunities, preserving their material clusters for uncertainty."""
    groups, group = np.unique(families, return_inverse=True)
    positive, negative = np.asarray(positive), np.asarray(negative)
    total = positive + negative
    trials = int(total.sum())
    if len(groups) < 2 or not 0 < positive.sum() < trials or np.any(total <= 0):
        raise ValueError('Population scoring requires observed trials, both outcomes and multiple material families')
    draws = family_draws(len(groups))
    bootstrap_trials = draws @ np.bincount(group, weights=total)
    scores, brier_replicates, gap_replicates = {}, {}, {}
    for name, values in probabilities.items():
        p = np.asarray(values, dtype=float)
        trial_probability = total * p
        quantities = {
            'brier': negative * p**2 + positive * (1 - p)**2,
            'nll_nats': -negative * np.log1p(-p) - positive * np.log(p),
            'mean_probability': trial_probability,
            'calibration_gap': trial_probability - positive,
        }
        by_family = np.column_stack([np.bincount(group, weights=value) for value in quantities.values()])
        bootstrap = (draws @ by_family) / bootstrap_trials[:, None]
        brier_replicates[name], gap_replicates[name] = bootstrap[:, 0], bootstrap[:, 3]
        bootstrap_ap = np.concatenate([average_precision(p, positive, total, draws[start:start + 128, group])
                                       for start in range(0, len(draws), 128)])
        scores[name] = {metric: float(values.sum() / trials) for metric, values in quantities.items()}
        scores[name]['average_precision'] = float(average_precision(p, positive, total, np.ones((1, len(p))))[0])
        scores[name]['article_family_ci95'] = {
            **{metric: interval(bootstrap[:, column]) for column, metric in enumerate(quantities)},
            'average_precision': interval(bootstrap_ap),
        }
    brier_bounds = interval(brier_replicates[qwen] - brier_replicates[position])
    calibration_bounds = interval(gap_replicates[qwen], 0.90)
    return {
        'families': len(groups), 'observed_stops': len(total), 'trials': trials,
        'positive': int(positive.sum()), 'negative': int(negative.sum()), 'event_rate': float(positive.sum() / trials),
        'methods': scores,
        'qwen_minus_position_brier': {'difference': scores[qwen]['brier'] - scores[position]['brier'],
                                     'paired_family_ci95': brier_bounds},
        'qwen_calibration_gap': {'difference': scores[qwen]['calibration_gap'], 'family_ci90': calibration_bounds},
        'gates': {'brier_superiority': brier_bounds[1] < 0,
                  'calibration_equivalence': inside_margin(calibration_bounds, CALIBRATION_MARGIN)},
    }


def context_effect(families, disambiguation, referent_count, positive, negative, probabilities, *, strata, qwen):
    """Estimate two minus one referent with equal weight to complete families and RC cells."""
    readouts = tuple(probabilities)
    totals = defaultdict(int)
    weighted = defaultdict(lambda: np.zeros(1 + len(readouts)))
    observed_families = set()
    for index, (family, reading, referents, yes, no) in enumerate(zip(
            families, disambiguation, referent_count, positive, negative, strict=True)):
        if reading not in strata or yes + no <= 0:
            continue
        if referents not in (1, 2):
            raise ValueError('A context cell must have one or two referents')
        cell = family, reading, referents
        total = int(yes) + int(no)
        totals[cell] += total
        weighted[cell] += [yes, *(total * probabilities[name][index] for name in readouts)]
        observed_families.add(family)

    registered_cells = tuple(product(strata, (1, 2)))
    complete = [family for family in sorted(observed_families)
                if all((family, reading, referents) in totals for reading, referents in registered_cells)]
    summary = {
        'disambiguation_strata': list(strata), 'families_with_observations': len(observed_families),
        'complete_families': len(complete), 'incomplete_families': len(observed_families) - len(complete),
        'included_trials': sum(totals[family, reading, referents] for family in complete
                               for reading, referents in registered_cells),
    }
    if len(complete) < 2:
        return {**summary, 'effects': None, 'gates': {'human_direction_confirmed': False,
                                                     'qwen_direction': False, 'qwen_human_equivalence': False}}

    rates = np.stack([weighted[family, reading, referents] / totals[family, reading, referents]
                      for family in complete for reading, referents in registered_cells])
    rates = rates.reshape(len(complete), len(strata), 2, 1 + len(readouts))
    differences = (rates[:, :, 1] - rates[:, :, 0]).mean(axis=1)
    replicates = (family_draws(len(complete)) @ differences) / len(complete)
    human_bounds = interval(replicates[:, 0])
    effects = {'human': {'two_minus_one': float(differences[:, 0].mean()), 'paired_family_ci95': human_bounds}}
    for column, method in enumerate(readouts, start=1):
        effects[method] = {
            'two_minus_one': float(differences[:, column].mean()),
            'paired_family_ci95': interval(replicates[:, column]),
            'model_minus_human': float((differences[:, column] - differences[:, 0]).mean()),
            'model_minus_human_paired_family_ci90': interval(replicates[:, column] - replicates[:, 0], 0.90),
        }
    return {
        **summary, 'effects': effects,
        'gates': {'human_direction_confirmed': human_bounds[1] < 0, 'qwen_direction': effects[qwen]['two_minus_one'] < 0,
                  'qwen_human_equivalence': inside_margin(effects[qwen]['model_minus_human_paired_family_ci90'],
                                                          CONTEXT_EFFECT_MARGIN)},
    }


def subset_arrays(rows, forecasts):
    families = [row['family'] for row in rows]
    positives = [row['positive'] for row in rows]
    negatives = [row['negative'] for row in rows]
    probabilities = {name: [forecasts[row['id'], row['stop']]['methods'][name] for row in rows] for name in METHODS}
    return families, positives, negatives, probabilities


def score(assay_root, predictions_dir, truth, output):
    plan, targets, forecasts, hashes = verify_forecasts(assay_root, predictions_dir)
    outcomes = verify_truth(truth, targets, hashes)
    all_rows = [row for _, row in sorted(outcomes.items())]
    by_scope = {
        'all': all_rows,
        'critical': [row for row in all_rows if row['phase'] == 'critical'],
        'spill': [row for row in all_rows if row['phase'] == 'spill'],
        'single_word': [row for row in all_rows if row['region_words'] == 1],
    }
    population = {scope: score_population(*subset_arrays(by_scope[scope], forecasts), qwen='qwen', position='position')
                  for scope in ('all', 'single_word')}
    effects = {}
    for scope, rows in by_scope.items():
        families, positives, negatives, probabilities = subset_arrays(rows, forecasts)
        disambiguations = [row['disamb'] for row in rows]
        referents = [CONTEXTS[row['context']] for row in rows]
        effects[scope] = {
            disamb: context_effect(families, disambiguations, referents, positives, negatives, probabilities,
                                   strata=(disamb,), qwen='qwen')
            for disamb in DISAMBIGUATIONS
        }
    gates = {**population['all']['gates'], **effects['critical']['rc']['gates']}
    result = {
        'study': 'Paape and Vasishth (2026)', 'source_article': plan['article'], 'population_unit': 'observed trial opportunity',
        'context_effect_unit': 'complete original family, equal weight across families',
        'context_cell_weight': 'observed trial opportunity within each family-by-context-by-disambiguation cell',
        'context_effect_direction': 'tworef minus oneref', 'designed_stops': len(targets),
        'designed_families': len({row['family'] for row in targets.values()}), 'unobserved_stops': len(targets) - len(outcomes),
        'population': population, 'context_effects': effects,
        'primary': {'population_scope': 'all', 'effect_scope': 'critical', 'effect_disambiguation': 'rc',
                    'calibration_absolute_margin': CALIBRATION_MARGIN, 'context_effect_absolute_margin': CONTEXT_EFFECT_MARGIN,
                    'gates': gates, 'passed_all': all(gates.values())},
        'bootstrap': {'draws': DRAWS, 'seed': SEED, 'unit': 'original material family',
                      'fixed': ['readers', 'forecasts'], 'interval': 'linear percentile',
                      'within_effect_same_resamples_for_model_and_human': True, 'no_positive_replicate_ap': 0},
        'input_sha256': hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('assay-root', 'predictions-dir', 'truth', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    score(**vars(parser.parse_args()))
