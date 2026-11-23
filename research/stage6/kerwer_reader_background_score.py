"""Score the frozen Kerwer reader-background forecasts after verifying their lineage."""

import argparse
import math
import os
from pathlib import Path

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np

from . import kerwer_reader_background_prepare as prepare
from . import kerwer_score as kerwer


PREPARED_RECEIPT_SHA256 = '09001439f37c8ba91ce0d508ac9e10a4c8678dca43850b9e010d0df35604ffb6'
METHODS = ('L', 'LP', 'T', 'TP', 'TPI', 'TP_blank', 'TPI_blank')
METRICS = ('rps', 'nll', 'mae', 'cumulative_ece')
CONTRASTS = {'reader_background': ('T', 'TP'), 'joint_vs_length_arm_and_reader': ('LP', 'TP'),
             'interaction': ('TP', 'TPI'), 'blank_tp': ('T', 'TP_blank'), 'blank_tpi': ('T', 'TPI_blank')}
BIN_EDGES = np.array([0.2, 0.4, 0.6, 0.8])
GRADES = np.arange(1, 9)
DRAWS, SEED, BATCH = 10_000, 20261121, 48
MARGIN = 0.01


def verified_predictions(prepared, staged, fit):
    path = prepared / 'preparation-receipt.json'
    prepared_sha = kerwer.digest(path)
    receipt = kerwer.read_json(path)
    if (prepared_sha != PREPARED_RECEIPT_SHA256
            or receipt['protocol_sha256'] != kerwer.digest(prepare.PROTOCOL)
            or receipt['source']['candidate_id'] != kerwer.CANDIDATE_ID
            or receipt['source']['holdout_files_sha256'][kerwer.OUTCOMES] != kerwer.INPUTS[kerwer.OUTCOMES]
            or receipt['fit_and_score'] != prepare.specification()
            or any(kerwer.digest(prepared / name) != value for name, value in receipt['prepared_sha256'].items())):
        raise ValueError('The prepared Kerwer receipt, frozen protocol, or local ledgers changed')
    staging_path = staged / 'staging-receipt.json'
    staging = kerwer.read_json(staging_path)
    test_map = staged / 'test-map.jsonl'
    expected = {'prepared_receipt_sha256': prepared_sha, 'protocol_sha256': receipt['protocol_sha256'],
                'candidate_id': kerwer.CANDIDATE_ID, 'packet_sha256': kerwer.digest(staged / 'fit-packet.json'),
                'test_map_sha256': kerwer.digest(test_map), 'original_outcomes_sha256': kerwer.INPUTS[kerwer.OUTCOMES],
                'trial_count': receipt['counts']['trials']}
    predictions = fit / 'predictions.jsonl'
    seal_path = fit / 'fit-seal.json'
    seal = kerwer.read_json(seal_path)
    fit_expected = {key: expected[key] for key in ('candidate_id', 'protocol_sha256', 'packet_sha256', 'trial_count')}
    fit_expected['predictions_sha256'] = kerwer.digest(predictions)
    if (any(staging[key] != value for key, value in expected.items())
            or any(seal[key] != value for key, value in fit_expected.items())):
        raise ValueError('The staging or fit seal does not bind the exact packet, map and predictions')

    articles = {row['id']: row for row in kerwer.rows(prepared / 'local/articles.jsonl')}
    trials = sorted(kerwer.rows(prepared / 'local/trial-keys.jsonl'), key=lambda row: (row['reader'], row['id']))
    if (len(trials) != expected['trial_count']
            or prepare.validate_folds(trials, articles) != receipt['fold_design']['cells']):
        raise ValueError('The crossed reader and DOI holdout differs from the prepared fold receipt')
    mapped = list(kerwer.rows(test_map))
    if mapped != [{'trial': f'{index:04d}', **row} for index, row in enumerate(trials)]:
        raise ValueError('The local test map does not preserve the exact frozen reader, article and folds')
    rows = {}
    identifiers = {row['trial'] for row in mapped}
    for row in kerwer.rows(predictions):
        key = row['trial']
        if row.keys() != {'trial', *METHODS} or key not in identifiers or key in rows:
            raise ValueError('Unexpected method, duplicate trial or invalid anonymous forecast ID')
        estimates = [row[method] for method in METHODS]
        if any(not isinstance(p, list) or len(p) != len(GRADES)
               or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > 1 for value in p)
               or not math.isclose(math.fsum(p), 1, rel_tol=0, abs_tol=1e-12) for p in estimates):
            raise ValueError(f'Invalid eight-grade probability distribution at {key}')
        rows[key] = estimates
    if rows.keys() != identifiers:
        raise ValueError('Expected one sealed prediction for every eligible held-out trial')
    matrices = np.asarray([rows[row['trial']] for row in mapped], dtype=np.float64)
    blank = {}
    independent = [METHODS.index(name) for name in ('L', 'T', 'TP_blank', 'TPI_blank')]
    for row, prediction in zip(mapped, matrices, strict=True):
        key = row['reader_fold'], row['doi_fold'], row['id']
        values = prediction[independent]
        if not np.allclose(blank.setdefault(key, values), values, rtol=0, atol=1e-12):
            raise ValueError('Text-only and blank forecasts must match for the same article in the same test cell')
    provenance = {'prepared_receipt_sha256': prepared_sha, 'protocol_sha256': receipt['protocol_sha256'],
                  'staging_receipt_sha256': kerwer.digest(staging_path), 'packet_sha256': expected['packet_sha256'],
                  'test_map_sha256': expected['test_map_sha256'], 'fit_seal_sha256': kerwer.digest(seal_path),
                  'predictions_sha256': fit_expected['predictions_sha256'], 'candidate_id': kerwer.CANDIDATE_ID,
                  'outcomes_sha256': expected['original_outcomes_sha256']}
    return mapped, articles, matrices, provenance


def read_grades(outcomes, mapped, expected):
    if kerwer.digest(outcomes) != expected:
        raise ValueError('The original Kerwer outcome file changed')
    keys = {(row['reader'], row['id']) for row in mapped}
    grades = {}
    for row in kerwer.rows(outcomes):
        key = row['reader'], row['id']
        grade = row['comprehensibility']
        if key not in keys or key in grades or type(grade) is not int or not 1 <= grade <= 8:
            raise ValueError('The frozen outcome trial or ordinal comprehensibility grade is invalid')
        grades[key] = grade
    if grades.keys() != keys:
        raise ValueError('The frozen outcome ledger does not cover exactly the sealed predictions')
    return np.asarray([grades[row['reader'], row['id']] for row in mapped])


def components(probabilities, grades):
    cumulative = np.cumsum(probabilities, axis=-1)[..., :-1]
    targets = grades[:, None, None] <= GRADES[None, None, :-1]
    residual = cumulative - targets
    rps = np.mean(residual**2, axis=-1)
    chosen = np.take_along_axis(probabilities, grades[:, None, None] - 1, axis=-1)[..., 0]
    nll = np.zeros_like(chosen)
    np.log(chosen, out=nll, where=chosen > 0)
    mae = np.abs(probabilities @ GRADES - grades[:, None])
    errors = np.stack((rps, -nll, mae), axis=-1)
    return errors, chosen == 0, residual, np.searchsorted(BIN_EDGES, cumulative, side='right')


def summaries(weights, trials):
    errors, zeros, residual, bins = trials
    totals = weights.sum(axis=1)
    if not np.all(totals > 0):
        raise ValueError('A crossed resample has no observed reader-DOI cells')
    methods, cuts, bin_count = residual.shape[1], residual.shape[2], len(BIN_EDGES) + 1
    values = np.empty((len(weights), methods, len(METRICS)), dtype=np.float64)
    sample = (weights @ errors.reshape(len(errors), -1)).reshape(len(weights), methods, -1)
    values[..., :3] = sample / totals[:, None, None]
    values[..., 1][weights @ zeros > 0] = np.inf
    # One bin index per trial and fixed cutoff. Aggregate weighted residuals before taking absolute values.
    offsets = np.arange(len(weights))[:, None] * bin_count
    for method in range(methods):
        error = np.zeros(len(weights))
        for cut in range(cuts):
            indexes = offsets + bins[:, method, cut]
            weighted = weights * residual[:, method, cut]
            bin_errors = np.bincount(indexes.ravel(), weights=weighted.ravel(), minlength=len(weights) * bin_count)
            error += np.abs(bin_errors.reshape(len(weights), bin_count)).sum(axis=1)
        values[:, method, 3] = error / (totals * cuts)
    return values


def crossed_bootstrap(trials, reader_index, doi_index, reader_count, doi_count):
    rng = np.random.default_rng(SEED)
    reader_weights = rng.multinomial(reader_count, np.full(reader_count, 1 / reader_count), size=DRAWS).astype(np.int16)
    doi_weights = rng.multinomial(doi_count, np.full(doi_count, 1 / doi_count), size=DRAWS).astype(np.int16)
    samples = []
    for start in range(0, DRAWS, BATCH):
        block = slice(start, start + BATCH)
        weights = (reader_weights[block, reader_index] * doi_weights[block, doi_index]).astype(np.float64)
        samples.append(summaries(weights, trials))
    return np.concatenate(samples)


def as_interval(point, samples):
    return {'estimate': float(point) if math.isfinite(point) else None,
            'ci95': np.quantile(samples, [0.025, 0.975]).tolist() if np.isfinite(samples).all() else None}


def point_metrics(values):
    return {method: {metric: float(point) if math.isfinite(point) else None for metric, point in zip(METRICS, row, strict=True)}
            for method, row in zip(METHODS, values, strict=True)}


def contrast(reference, candidate, values):
    first, second = (METHODS.index(method) for method in (reference, candidate))
    return (values[..., first, 0] - values[..., second, 0], values[..., second, 3] - values[..., first, 3])


def compare(point, draws):
    comparisons = {}
    for name, (reference, candidate) in CONTRASTS.items():
        gain, ece = contrast(reference, candidate, point)
        sampled_gain, sampled_ece = contrast(reference, candidate, draws)
        calibration = as_interval(ece, sampled_ece)
        test = {'reference': reference, 'candidate': candidate, 'calibration_increase': calibration}
        ece_ok = calibration['ci95'][1] < MARGIN
        if name.startswith('blank_'):
            increase = as_interval(-gain, -sampled_gain)
            rps_ok = increase['ci95'][1] < MARGIN
            test.update(rps_increase=increase, rps_noninferiority_passes=rps_ok,
                        calibration_noninferiority_passes=ece_ok, passed=rps_ok and ece_ok)
        else:
            improvement = as_interval(gain, sampled_gain)
            rps_ok = improvement['ci95'][0] > 0
            calibrated = name != 'joint_vs_length_arm_and_reader'
            test.update(rps_reduction=improvement, improvement_passes=rps_ok,
                        calibration_noninferiority_passes=ece_ok if calibrated else None,
                        passed=rps_ok and (ece_ok or not calibrated))
        comparisons[name] = test
    return comparisons


def evaluate(mapped, articles, predictions, grades):
    reader_ids = sorted({row['reader'] for row in mapped})
    doi_ids = sorted({articles[row['id']]['doi'] for row in mapped})
    reader_positions = {key: index for index, key in enumerate(reader_ids)}
    doi_positions = {key: index for index, key in enumerate(doi_ids)}
    reader_index = np.asarray([reader_positions[row['reader']] for row in mapped])
    doi_index = np.asarray([doi_positions[articles[row['id']]['doi']] for row in mapped])
    trials = components(predictions, grades)
    point = summaries(np.ones((1, len(mapped))), trials)[0]
    draws = crossed_bootstrap(trials, reader_index, doi_index, len(reader_ids), len(doi_ids))
    methods = {method: {metric: as_interval(point[m, k], draws[:, m, k]) for k, metric in enumerate(METRICS)}
               for m, method in enumerate(METHODS)}
    for index, method in enumerate(METHODS):
        methods[method]['zero_probability_gold_trials'] = int(np.count_nonzero(trials[1][:, index]))
    omitted = []
    for index, doi in enumerate(doi_ids):
        kept = doi_index != index
        values = summaries(kept[None, :].astype(np.float64), trials)[0]
        contrasts = {}
        for name, (reference, candidate) in CONTRASTS.items():
            gain, ece = contrast(reference, candidate, values)
            difference = {'rps_increase': -float(gain)} if name.startswith('blank_') else {'rps_reduction': float(gain)}
            contrasts[name] = {**difference, 'calibration_increase': float(ece)}
        omitted.append({'excluded_doi': doi, 'remaining_trials': int(kept.sum()), 'methods': point_metrics(values),
                        'comparisons': contrasts})
    return {'counts': {'trials': len(mapped), 'readers': len(reader_ids), 'doi_families': len(doi_ids)},
            'methods': methods, 'comparisons': compare(point, draws), 'leave_one_doi_out': omitted,
            'resampling': {'draws': DRAWS, 'seed': SEED, 'unit_counts': ['reader', 'doi'], 'refit': False},
            'numeric_runtime': {'numpy': np.__version__},
            'nll_policy': 'A null NLL point denotes infinite loss from assigned zero gold probability; an interval is null when a draw has infinite loss.'}


def score(prepared, staged, fit, outcomes, output):
    if output.exists():
        raise FileExistsError(output)
    mapped, articles, predictions, provenance = verified_predictions(prepared, staged, fit)
    grades = read_grades(outcomes, mapped, provenance['outcomes_sha256'])
    report = evaluate(mapped, articles, predictions, grades)
    output.parent.mkdir(parents=True, exist_ok=True)
    kerwer.write_json(output, {'status': 'scored_frozen_kerwer_reader_background_oof', 'provenance': provenance,
                               'scoring_code_sha256': kerwer.digest(Path(__file__)), **report})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'staged', 'fit', 'outcomes', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    score(**vars(parser.parse_args()))
