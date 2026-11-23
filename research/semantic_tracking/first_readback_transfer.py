"""Fit article-only first-regression readouts on one frozen Qwen backbone."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


STUDIES = ('english', 'german')
MODES = ('within', 'pooled', 'cross_study')
METHODS = ('prior', 'position', 'qwen')
TARGET_FIELDS = {'id', 'stop', 'study', 'group_id', 'split', 'region_words'}
TRAIN_FIELDS = TARGET_FIELDS - {'split'} | {'negative', 'positive'}


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read_rows(path, expected_sha, fields):
    if digest(path) != expected_sha:
        raise ValueError(f'First-read stimuli or metadata changed: {path.name}')
    with path.open(encoding='utf-8') as stream:
        rows = list(map(json.loads, stream))
    if not rows or any(set(row) != fields for row in rows):
        raise ValueError(f'Unexpected fields in first-read input: {path.name}')
    return rows


def key(row):
    return row['id'], row['stop']


def load_inputs(stimuli, targets, training, hashes):
    docs = read_rows(stimuli, hashes['stimuli_sha256'], {'id', 'text', 'stops'})
    targets = read_rows(targets, hashes['targets_sha256'], TARGET_FIELDS)
    training = read_rows(training, hashes['train_sha256'], TRAIN_FIELDS)
    articles = {row['id']: row for row in docs}
    targets_by_key = {key(row): row for row in targets}
    if (len(articles) != len(docs) or len(targets_by_key) != len(targets)
            or len({key(row) for row in training}) != len(training)
            or not {key(row) for row in training} <= {key(row) for row in targets if row['split'] == 'train'}):
        raise ValueError('Observed training keys must belong to the registered first-read training stops')
    for row in training:
        target = targets_by_key[key(row)]
        if (any(target[field] != row[field] for field in TRAIN_FIELDS - {'negative', 'positive'})
                or any(type(row[field]) is not int or row[field] < 0 for field in ('negative', 'positive'))
                or not row['negative'] + row['positive']):
            raise ValueError('Invalid aggregated first-read responses')
    by_article = {}
    for row in targets:
        article = articles[row['id']]
        identity = row['study'], row['split'], row['group_id']
        if (row['study'] not in STUDIES or row['split'] not in ('train', 'test')
                or by_article.setdefault(row['id'], identity) != identity
                or row['stop'] not in article['stops'] or type(row['region_words']) is not int
                or row['stop'] <= article['stops'][0] or row['region_words'] <= 0):
            raise ValueError('First-read stop has invalid text or split metadata')
    training_groups = {(study, group) for study, split, group in by_article.values() if split == 'train'}
    evaluation_groups = {(study, group) for study, split, group in by_article.values() if split == 'test'}
    if training_groups & evaluation_groups or articles.keys() != by_article.keys():
        raise ValueError('An article family crosses the first-read train/test boundary')
    return articles, targets, training


def geometry(article, end):
    from math import log1p

    words = article['text'][:end].split()
    return [log1p(end), log1p(len(words)), log1p(len(words[-1]))]


def fit_and_forecast(articles, targets, training, features, output, device):
    import numpy as np

    from research.core_features.fit import apply_scaler, fit_scaler, read_features, sigmoid, train_score, write_rows

    states = read_features(features)
    if states.keys() != {(article['id'], stop) for article in articles.values() for stop in article['stops']}:
        raise ValueError('Qwen feature inventory disagrees with the locked first-read stimuli')
    evaluation = [row for row in targets if row['split'] == 'test']
    predictions = [{**row, 'methods': {}} for row in evaluation]
    position = np.asarray([geometry(articles[row['id']], row['stop']) for row in training])
    target_geometry = np.asarray([geometry(articles[row['id']], row['stop']) for row in evaluation])
    text_states = np.stack([states[key(row)] for row in training])
    target_states = np.stack([states[key(row)] for row in evaluation])
    study = np.asarray([row['study'] for row in training])
    eval_studies = np.asarray([row['study'] for row in evaluation])
    positives = np.asarray([row['positive'] for row in training], dtype=np.float64)
    totals = positives + np.asarray([row['negative'] for row in training])
    weights, diagnostics = {}, {}

    # Each single-study fit serves both evaluation arms without a second fit.
    for fitted_on in (*STUDIES, 'pooled'):
        selection = study == fitted_on if fitted_on != 'pooled' else np.ones(len(study), dtype=bool)
        mass = totals[selection].copy()
        if fitted_on == 'pooled':
            mass /= np.where(study == STUDIES[0], totals[study == STUDIES[0]].sum(),
                             totals[study == STUDIES[1]].sum())
        y = positives[selection] / totals[selection]
        center, scale = fit_scaler(position[selection])
        weights[f'{fitted_on}_center'], weights[f'{fitted_on}_scale'] = center, scale
        before = np.column_stack((np.ones(len(y)), apply_scaler(position[selection], center, scale)))
        after = np.column_stack((np.ones(len(evaluation)), apply_scaler(target_geometry, center, scale)))
        inputs = ((before[:, :1], after[:, :1]), (before, after),
                  (np.column_stack((before, text_states[selection])), np.column_stack((after, target_states))))
        for method, (fit_design, eval_design) in zip(METHODS, inputs, strict=True):
            coefficients, diag = train_score(fit_design, y, mass, device)
            weights[f'{fitted_on}_{method}'] = coefficients
            diagnostics[f'{fitted_on}_{method}'] = diag
            probabilities = sigmoid(eval_design @ coefficients)
            if not np.all(np.isfinite(probabilities) & (0 < probabilities) & (probabilities < 1)):
                raise ValueError('Expected finite first-read probabilities')
            modes = (['pooled'] * len(evaluation) if fitted_on == 'pooled'
                     else np.where(eval_studies == fitted_on, 'within', 'cross_study'))
            for row, mode, probability in zip(predictions, modes, probabilities, strict=True):
                row['methods'][f'{mode}_{method}'] = float(probability)
    for row in predictions:
        if row['methods'].keys() != {f'{mode}_{method}' for mode in MODES for method in METHODS}:
            raise ValueError('First-read forecasts lack one of the registered comparisons')
    np.savez(output / 'heads.npz', **weights)
    write_rows(output / 'predictions.jsonl', predictions)
    return diagnostics


def run(args):
    import torch

    from research.core_features.extract import run as extract
    from research.gradio_app.causal_plaba import QWEN_FILES
    from research.gradio_app.checkpoint import load_checkpoint
    from research.gradio_app.plaba import BASE_REVISION

    hashes = {f'{name}_sha256': getattr(args, f'{name}_sha256') for name in ('stimuli', 'targets', 'train')}
    articles, targets, training = load_inputs(args.stimuli, args.targets, args.train, hashes)
    checkpoint = load_checkpoint(args.checkpoint)
    if args.model.name != BASE_REVISION or any(digest(args.model / name) != sha for name, sha in QWEN_FILES.items()):
        raise ValueError('Expected the certified candidate Qwen snapshot')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix='.first-readback-output-') as temporary:
        staging = Path(temporary)
        output = staging / 'return'
        output.mkdir()
        extraction = staging / 'features'
        extract(args.stimuli, extraction, max_batch_tokens=6144)
        fit_stats = fit_and_forecast(articles, targets, training, extraction / 'features.npz', output,
                                     torch.device('cuda'))
        for name in ('extraction.json', 'causality.json', 'telemetry.jsonl'):
            shutil.copyfile(extraction / name, output / name)
        seal = {**hashes, 'fit': fit_stats, 'candidate_id': checkpoint.identifier, 'model_revision': BASE_REVISION,
                'assay_sha256': args.assay_sha256, 'protocol_sha256': args.protocol_sha256,
                'source_code_sha256': digest(Path(__file__)), 'extraction_sha256': digest(output / 'extraction.json'),
                'heads_sha256': digest(output / 'heads.npz'), 'predictions_sha256': digest(output / 'predictions.jsonl'),
                'training_stops': len(training), 'heldout_stops': sum(row['split'] == 'test' for row in targets)}
        (output / 'seal.json').write_text(json.dumps(seal, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        output.rename(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('stimuli', 'targets', 'train', 'model', 'checkpoint', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    for flag in ('stimuli', 'targets', 'train', 'assay', 'protocol'):
        parser.add_argument(f'--{flag}-sha256', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
