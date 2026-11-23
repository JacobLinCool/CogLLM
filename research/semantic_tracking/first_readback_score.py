"""Score sealed first-readback forecasts against the two fixed human holdouts."""

import argparse
import json
import os
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from research.semantic_tracking.first_readback_transfer import (
    METHODS, MODES, STUDIES, TARGET_FIELDS, TRAIN_FIELDS, digest, key,
)


ASSAY_SHA256 = 'd8dca9e9bfd7977d38939aede6cd0d8e51db34bc5e4e04cde93c85adfd9e4bbd'
PROTOCOL_SHA256 = '667518a52592e50a42e0e2560319fc68b7a696d5d92b4de2e61fd9bd88652b66'
CANDIDATE = '74c26ff02569b2b3bc28ed291bf15fd2f85db5b4c93a3ad7bfc34f6a5d333ea0'
MODEL_REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
MODEL_NAME = 'Qwen/Qwen3.5-2B'
READOUTS = tuple(f'{mode}_{method}' for mode in MODES for method in METHODS)
SEED = 20261121
DRAWS = 10_000
CALIBRATION_MARGIN = 0.01


def read_index(path, fields):
    index = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if (not isinstance(row, dict) or row.keys() != fields
                    or any(type(row[name]) is not str or not row[name] for name in ('id', 'group_id'))
                    or any(type(row[name]) is not int or row[name] < 1 for name in ('stop', 'region_words'))
                    or row['study'] not in STUDIES or key(row) in index):
                raise ValueError(f'Unexpected fields or duplicate display stop in {path.name}')
            index[key(row)] = row
    if not index:
        raise ValueError(f'No rows in {path.name}')
    return index


def verify_forecasts(assay_root, predictions_dir):
    code = Path(__file__).resolve()
    assay = assay_root / 'assay.json'
    protocol = code.with_name('first-readback-transfer.md')
    if digest(assay) != ASSAY_SHA256 or digest(protocol) != PROTOCOL_SHA256:
        raise ValueError('The first-readback assay or protocol differs from the locked analysis')
    plan = json.loads(assay.read_text(encoding='utf-8'))
    predictions = predictions_dir / 'predictions.jsonl'
    seal = predictions_dir / 'seal.json'
    receipt = json.loads(seal.read_text(encoding='utf-8'))
    files = {
        'stimuli_sha256': assay_root / 'stimuli.jsonl', 'targets_sha256': assay_root / 'targets.jsonl',
        'train_sha256': assay_root / 'train.jsonl', 'predictions_sha256': predictions,
        'source_code_sha256': code.with_name('first_readback_transfer.py'),
        'extraction_sha256': predictions_dir / 'extraction.json', 'heads_sha256': predictions_dir / 'heads.npz',
    }
    hashes = {'assay_sha256': ASSAY_SHA256, 'protocol_sha256': PROTOCOL_SHA256,
              **{field: digest(path) for field, path in files.items()}}
    if (any(receipt.get(field) != sha for field, sha in hashes.items())
            or receipt.get('candidate_id') != CANDIDATE or receipt.get('model_revision') != MODEL_REVISION):
        raise ValueError('The forecast bundle disagrees with the sealed assay, code, or model')
    if any(hashes[f'{name}_sha256'] != plan['file_sha256'][f'{name}.jsonl']
           for name in ('stimuli', 'targets', 'train')):
        raise ValueError('First-readback inputs disagree with the locked assay')
    extraction = json.loads(files['extraction_sha256'].read_text(encoding='utf-8'))
    if ((extraction.get('model'), extraction.get('revision'), extraction.get('dtype'))
            != (MODEL_NAME, MODEL_REVISION, 'float32') or extraction.get('input_sha256') != hashes['stimuli_sha256']):
        raise ValueError('Extraction does not match the registered FP32 backbone and stimuli')

    targets = read_index(files['targets_sha256'], TARGET_FIELDS)
    by_family, by_article = {}, {}
    for row in targets.values():
        study, split, family = row['study'], row['split'], row['group_id']
        if (split not in ('train', 'test') or by_family.setdefault(family, (study, split)) != (study, split)
                or by_article.setdefault(row['id'], (study, split, family)) != (study, split, family)):
            raise ValueError('First-readback material families or variants cross an assay boundary')
    for study in STUDIES:
        for split in ('train', 'test'):
            counts = plan['counts'][study][split]
            designed = [row for row in targets.values() if (row['study'], row['split']) == (study, split)]
            if (len(designed) != counts['designed_stops'] or len({row['id'] for row in designed}) != counts['stimuli']
                    or len({row['group_id'] for row in designed}) != counts['families']):
                raise ValueError('First-readback design differs from its registered inventory')
    heldout = {position: row for position, row in targets.items() if row['split'] == 'test'}
    forecasts = read_index(predictions, TARGET_FIELDS | {'methods'})
    registered_training = sum(plan['counts'][study]['train']['observed_stops'] for study in STUDIES)
    if (forecasts.keys() != heldout.keys() or receipt.get('heldout_stops') != len(heldout)
            or receipt.get('training_stops') != registered_training):
        raise ValueError('Forecasts must include every designed held-out stop and certify the observed training count')
    for position, row in forecasts.items():
        probabilities = row['methods']
        if (any(row[field] != value for field, value in heldout[position].items())
                or not isinstance(probabilities, dict) or probabilities.keys() != set(READOUTS)
                or any(type(p) not in (int, float) or not 0 < p < 1 for p in probabilities.values())):
            raise ValueError('Forecast metadata or probabilities disagree with the registered stops and readouts')

    hashes.update(seal_sha256=digest(seal), scorer_sha256=digest(code))
    return plan, heldout, forecasts, hashes


def verify_truth(assay_root, plan, heldout):
    path = assay_root / 'test.jsonl'
    sha = digest(path)
    if sha != plan['file_sha256'][path.name]:
        raise ValueError('The held-out reference file differs from the locked first-readback assay')
    truth = read_index(path, TRAIN_FIELDS)
    for position, row in truth.items():
        if (position not in heldout or any(row[field] != heldout[position][field] for field in TARGET_FIELDS - {'split'})
                or any(type(row[label]) is not int or row[label] < 0 for label in ('positive', 'negative'))
                or row['positive'] + row['negative'] < 1):
            raise ValueError('A held-out observation has invalid metadata or outcomes')
    for study in STUDIES:
        registered = plan['counts'][study]['test']
        rows = [row for row in truth.values() if row['study'] == study]
        for single in (False, True):
            subset = [row for row in rows if row['region_words'] == 1] if single else rows
            prefix = 'single_word_' if single else ''
            if (len(subset) != registered[prefix + 'observed_stops']
                    or any(sum(row[label] for row in subset) != registered[prefix + label]
                           for label in ('positive', 'negative'))):
                raise ValueError('Observed stop and response counts disagree with the registered holdout')
    return truth, sha


def average_precision(p, positive, total, multiplicities):
    """Trial-level AP with all trials at a tied probability entering together."""
    order = np.argsort(-p, kind='stable')
    sorted_scores = p[order]
    thresholds = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    weighted = multiplicities[:, order]
    positives = np.add.reduceat(weighted * positive[order], thresholds, axis=1)
    totals = np.add.reduceat(weighted * total[order], thresholds, axis=1)
    true_positive = np.cumsum(positives, axis=1)
    observed = np.cumsum(totals, axis=1)
    precision = np.divide(true_positive, observed, out=np.zeros_like(observed, dtype=np.float64), where=observed > 0)
    mass = true_positive[:, -1]
    return np.divide((precision * positives).sum(axis=1), mass, out=np.zeros(len(mass)), where=mass > 0)


def interval(replicates, mass=0.95):
    tail = (1 - mass) / 2
    return np.quantile(replicates, [tail, 1 - tail]).tolist()


def score_subset(rows, forecasts, groups, draws):
    group = np.searchsorted(groups, [row['group_id'] for row in rows])
    positive = np.array([row['positive'] for row in rows])
    negative = np.array([row['negative'] for row in rows])
    total = positive + negative
    trials = int(total.sum())
    totals_by_family = np.bincount(group, weights=total, minlength=len(groups))
    bootstrap_trials = draws @ totals_by_family
    if np.any(bootstrap_trials <= 0):
        raise ValueError('A registered family bootstrap replicate has no trials for this subset')
    if not 0 < positive.sum() < trials:
        raise ValueError('Both readback outcomes must be present in each registered comparison')

    scores, brier_replicates, gap_replicates = {}, {}, {}
    for readout in READOUTS:
        p = np.array([forecasts[key(row)]['methods'][readout] for row in rows])
        brier = negative * p**2 + positive * (1 - p)**2
        nll = -negative * np.log1p(-p) - positive * np.log(p)
        predicted = total * p
        gap = predicted - positive
        quantities = {'brier': brier, 'nll_nats': nll, 'mean_probability': predicted, 'calibration_gap': gap}
        by_family = np.column_stack([np.bincount(group, weights=value, minlength=len(groups))
                                     for value in quantities.values()])
        bootstrap = (draws @ by_family) / bootstrap_trials[:, None]
        brier_replicates[readout], gap_replicates[readout] = bootstrap[:, 0], bootstrap[:, 3]
        # Batches bound memory without changing the paired material resamples.
        bootstrap_ap = np.concatenate([average_precision(p, positive, total, draws[start:start + 128, group])
                                       for start in range(0, len(draws), 128)])
        scores[readout] = {name: float(array.sum() / trials) for name, array in quantities.items()}
        scores[readout]['average_precision'] = float(average_precision(p, positive, total, np.ones((1, len(p))))[0])
        scores[readout]['article_family_ci95'] = {
            **{name: interval(bootstrap[:, column]) for column, name in enumerate(quantities)},
            'average_precision': interval(bootstrap_ap),
        }
    text, position = 'pooled_qwen', 'pooled_position'
    brier_interval = interval(brier_replicates[text] - brier_replicates[position])
    gap_interval = interval(gap_replicates[text], 0.90)
    return {
        'observed_stops': len(rows), 'families_with_observations': len(set(group)), 'trials': trials,
        'positive': int(positive.sum()), 'negative': int(negative.sum()), 'event_rate': float(positive.sum() / trials),
        'methods': scores,
        'pooled_qwen_minus_pooled_position_brier': {'difference': scores[text]['brier'] - scores[position]['brier'],
                                                  'paired_article_family_ci95': brier_interval},
        'pooled_qwen_calibration_gap': {'difference': scores[text]['calibration_gap'], 'article_family_ci90': gap_interval},
    }


def score(assay_root, predictions_dir, output):
    plan, heldout, forecasts, hashes = verify_forecasts(assay_root, predictions_dir)
    truth, hashes['test_sha256'] = verify_truth(assay_root, plan, heldout)
    rng = np.random.default_rng(SEED)
    results, gates = {}, {}
    for study in STUDIES:
        groups = sorted({row['group_id'] for row in heldout.values() if row['study'] == study})
        draws = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=DRAWS)
        rows = [row for _, row in sorted(truth.items()) if row['study'] == study]
        all_regions = score_subset(rows, forecasts, groups, draws)
        single_word = score_subset([row for row in rows if row['region_words'] == 1], forecasts, groups, draws)
        results[study] = {'design_families': len(groups), 'all': all_regions, 'single_word': single_word}
        brier = all_regions['pooled_qwen_minus_pooled_position_brier']['paired_article_family_ci95']
        calibration = all_regions['pooled_qwen_calibration_gap']['article_family_ci90']
        gates[study] = {'brier_superiority': brier[1] < 0,
                        'calibration_equivalence': -CALIBRATION_MARGIN < calibration[0] <= calibration[1] < CALIBRATION_MARGIN}
    payload = {
        'endpoint': plan['endpoint'], 'scoring_unit': 'one observed trial opportunity',
        'bootstrap': {'draws': DRAWS, 'seed': SEED, 'strata': 'study', 'unit': 'original material family',
                      'fixed': ['readers', 'forecasts'], 'interval': 'linear percentile',
                      'shared_resamples_within_study': True, 'no_positive_replicate_ap': 0},
        'studies': results,
        'primary': {'population': 'all observed held-out regions in each study', 'training_mode': 'pooled',
                    'calibration_absolute_margin': CALIBRATION_MARGIN, 'gates': gates,
                    'passed_both_studies': all(all(study.values()) for study in gates.values())},
        'input_sha256': hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('assay-root', 'predictions-dir', 'output'):
        parser.add_argument(f'--{option}', required=True, type=Path)
    score(**vars(parser.parse_args()))
