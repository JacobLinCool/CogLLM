"""Stage Kerwer sufficient statistics locally and fit the sealed crossed readouts."""

import argparse
from collections import defaultdict
from hashlib import file_digest
from importlib.metadata import version
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

if __name__ == '__main__':
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'

import numpy as np


METHODS = {'L': (0, 1, 2, 3), 'LP': (0, 1, 2, 3, 5), 'T': (0, 1, 2, 3, 4),
           'TP': (0, 1, 2, 3, 4, 5), 'TPI': (0, 1, 2, 3, 4, 5, 6)}
ARMS = ('plain_no_headings', 'plain_with_headings')
RUNTIME = {'numpy': '2.4.6', 'scipy': '1.16.3'}


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        yield from map(json.loads, stream)


def write_json(path, data):
    path.write_text(json.dumps(data, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def fit_scale(values):
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    return center, np.where(scale > 0, scale, 1.0)


def article_raw(row):
    return np.array([np.log1p(row['body_characters']), np.log1p(row['body_words']), row['served_understanding']])


def build_cell(train, test, articles, readers, outcomes, fold):
    fit_articles = sorted({row['id'] for row in train})
    fit_readers = sorted({row['reader'] for row in train})
    article_mean, article_scale = fit_scale(np.stack([article_raw(articles[key]) for key in fit_articles]))
    reader_mean, reader_scale = fit_scale(np.array([readers[key]['english_e_pa'] for key in fit_readers]))
    scaled = {}
    for key in {row['id'] for row in train + test}:
        length_chars, length_words, served = (article_raw(articles[key]) - article_mean) / article_scale
        arm = articles[key]['arm']
        scaled[key] = (float(length_chars), float(length_words), *(int(arm == value) for value in ARMS), float(served))

    def features(row):
        epa = (readers[row['reader']]['english_e_pa'] - reader_mean) / reader_scale
        return (*scaled[row['id']], float(epa))

    groups = defaultdict(lambda: [0] * 8)
    for row in train:
        rating = outcomes[row['reader'], row['id']]
        groups[features(row)][rating - 1] += 1
    totals = np.sum(list(groups.values()), axis=0)
    if not np.all(totals > 0) or int(totals.sum()) != len(train):
        raise ValueError('A training cell is missing an ordinal category or a fitting observation')
    return {**fold, 'train': [{'x': x, 'counts': counts} for x, counts in sorted(groups.items())],
            'test': [{'trial': row['trial'], 'x': features(row)} for row in test],
            'blank_epa_z': ((np.array([readers[key]['english_e_pa'] for key in fit_readers]) - reader_mean) /
                            reader_scale).tolist(),
            'scaling': {'article_center': article_mean.tolist(), 'article_scale': article_scale.tolist(),
                        'epa_center': float(reader_mean), 'epa_scale': float(reader_scale)}}


def stage_inputs(prepared, outcomes_path, output):
    from research.stage2.records import write_jsonl
    from research.stage6 import kerwer_reader_background_prepare as prep

    if output.exists():
        raise FileExistsError(output)
    prepared_path = prepared / 'preparation-receipt.json'
    receipt = json.loads(prepared_path.read_bytes())
    reference = prep.RESULTS / 'kerwer-reader-background-preparation.json'
    if (digest(prepared_path) != digest(reference) or receipt['fit_and_score'] != prep.specification()
            or receipt['protocol_sha256'] != digest(prep.PROTOCOL)
            or any(digest(prepared / name) != expected for name, expected in receipt['prepared_sha256'].items())):
        raise ValueError('Kerwer prepared receipt, protocol or files differ from the frozen release')
    expected_outcomes = receipt['source']['holdout_files_sha256'][prep.kerwer.OUTCOMES]
    if digest(outcomes_path) != expected_outcomes:
        raise ValueError('Kerwer outcomes differ from the original locked release')

    articles = {row['id']: row for row in rows(prepared / 'local/articles.jsonl')}
    readers = {row['reader']: row for row in rows(prepared / 'local/readers.jsonl')}
    trials = [dict(row, trial=f'{index:04d}') for index, row in enumerate(rows(prepared / 'local/trial-keys.jsonl'))]
    if prep.validate_folds(trials, articles) != receipt['fold_design']['cells']:
        raise ValueError('Kerwer split membership differs from its prefrozen receipt')
    outcomes = {}
    for row in rows(outcomes_path):
        trial = row['reader'], row['id']
        value = row['comprehensibility']
        if trial in outcomes or type(value) is not int or not 1 <= value <= 8:
            raise ValueError('Invalid or duplicate Kerwer comprehension rating')
        outcomes[trial] = value
    if set(outcomes) != {(row['reader'], row['id']) for row in trials} or len(trials) != receipt['counts']['trials']:
        raise ValueError('Kerwer outcomes do not match the exact crossed trial inventory')

    cells = []
    for fold in receipt['fold_design']['cells']:
        r, d = fold['reader_fold'], fold['doi_fold']
        train = [row for row in trials if row['reader_fold'] != r and row['doi_fold'] != d]
        test = [row for row in trials if row['reader_fold'] == r and row['doi_fold'] == d]
        cells.append(build_cell(train, test, articles, readers, outcomes, fold))
    metadata = {'candidate_id': receipt['source']['candidate_id'], 'protocol_sha256': receipt['protocol_sha256'],
                'trial_count': len(trials), 'prepared_receipt_sha256': digest(prepared_path)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.kerwer-background-fit-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'staged'
        stage.mkdir(mode=0o700)
        packet = stage / 'fit-packet.json'
        test_map = stage / 'test-map.jsonl'
        write_json(packet, {**metadata, 'cells': cells})
        write_jsonl(test_map, trials)
        write_json(stage / 'staging-receipt.json', {**metadata, 'packet_sha256': digest(packet),
                   'test_map_sha256': digest(test_map), 'original_outcomes_sha256': expected_outcomes})
        stage.rename(output)


def design(rows, method):
    values = np.column_stack((rows, rows[:, 4] * rows[:, 5]))
    return values[:, METHODS[method]]


def fit_cell(cell, fit, predict):
    raw = np.asarray([row['x'] for row in cell['train']], dtype=np.float64)
    counts = np.asarray([row['counts'] for row in cell['train']])
    test = np.asarray([row['x'] for row in cell['test']], dtype=np.float64)
    blank = np.asarray(cell['blank_epa_z'], dtype=np.float64)
    trials = [row['trial'] for row in cell['test']]
    if (raw.shape != (len(counts), 6) or test.shape != (cell['test_trials'], 6) or counts.shape[1:] != (8,)
            or blank.shape != (cell['fit_readers'],) or len(set(trials)) != len(trials)
            or counts.dtype.kind not in 'iu' or np.any(counts < 0) or not np.all(counts.sum(axis=0) > 0)
            or not np.all(counts.sum(axis=1) > 0) or counts.sum() != cell['fit_trials']
            or not all(np.isfinite(array).all() for array in (raw, test, blank))):
        raise ValueError('Kerwer cell differs from its prespecified sufficient statistics')
    heads, forecasts = {}, {}
    marginal = np.repeat(test, len(blank), axis=0)
    marginal[:, 5] = np.tile(blank, len(test))
    for method in METHODS:
        thresholds, slopes, diagnostics = fit(design(raw, method), counts)
        heads[method] = {'thresholds': thresholds.tolist(), 'slopes': slopes.tolist(), 'optimizer': diagnostics}
        forecasts[method] = predict(design(test, method), thresholds, slopes)
        if method in {'TP', 'TPI'}:
            p = predict(design(marginal, method), thresholds, slopes).reshape(len(test), len(blank), 8)
            forecasts[f'{method}_blank'] = p.mean(axis=1)
    return heads, [{'trial': trial, **{method: values[index].tolist() for method, values in forecasts.items()}}
                   for index, trial in enumerate(trials)]


def run(packet, checksum, output):
    from research.stage6 import kerwer_proportional_odds as kernel

    if output.exists():
        raise FileExistsError(output)
    if digest(packet) != checksum or {name: version(name) for name in RUNTIME} != RUNTIME:
        raise ValueError('Kerwer sufficient statistics or numerical runtime differ from the frozen edition')
    data = json.loads(packet.read_bytes())
    expected = [(reader, doi) for reader in range(5) for doi in range(4)]
    if [(row['reader_fold'], row['doi_fold']) for row in data['cells']] != expected:
        raise ValueError('Expected exactly the 20 ordered Kerwer crossed folds')
    predictions, heads = [], []
    for cell in data['cells']:
        parameters, forecasts = fit_cell(cell, kernel.fit, kernel.predict)
        heads.append({'reader_fold': cell['reader_fold'], 'doi_fold': cell['doi_fold'],
                      'scaling': cell['scaling'], 'methods': parameters})
        predictions.extend(forecasts)
    predictions.sort(key=lambda row: row['trial'])
    if [row['trial'] for row in predictions] != [f'{index:04d}' for index in range(data['trial_count'])]:
        raise ValueError('Kerwer out-of-fold forecasts do not cover each sealed trial exactly once')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.kerwer-background-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'fit'
        stage.mkdir()
        forecasts, parameters = stage / 'predictions.jsonl', stage / 'heads.json'
        with forecasts.open('w', encoding='utf-8') as stream:
            for row in predictions:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        write_json(parameters, heads)
        write_json(stage / 'fit-seal.json', {key: data[key] for key in ('candidate_id', 'protocol_sha256', 'trial_count')} |
                   {'packet_sha256': checksum, 'predictions_sha256': digest(forecasts),
                    'heads_sha256': digest(parameters), 'runtime': RUNTIME,
                    'fit_code_sha256': digest(Path(__file__)), 'kernel_code_sha256': digest(Path(kernel.__file__))})
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    stage = commands.add_parser('stage-inputs')
    for name in ('prepared', 'outcomes', 'output'):
        stage.add_argument(f'--{name}', type=Path, required=True)
    fit = commands.add_parser('fit')
    fit.add_argument('--packet', type=Path, required=True)
    fit.add_argument('--checksum', required=True)
    fit.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'stage-inputs':
        stage_inputs(args.prepared, args.outcomes, args.output)
    else:
        run(args.packet, args.checksum, args.output)


if __name__ == '__main__':
    main()
