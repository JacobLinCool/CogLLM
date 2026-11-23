"""Apply the sealed pooled first-readback heads to independent English stimuli."""

import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from research.semantic_tracking.first_readback_transfer import METHODS, digest, geometry, read_rows


HEADS_SHA256 = '029c8503fb4110f8c853478809674b68c077c078c56ce62662ad98b03387d432'
CANDIDATE = '74c26ff02569b2b3bc28ed291bf15fd2f85db5b4c93a3ad7bfc34f6a5d333ea0'
STIMULUS_FIELDS = {'id', 'text', 'stops'}
PREDICTION_FIELDS = {'id', 'stop', 'methods'}


def read_stimuli(path, expected_sha):
    rows = read_rows(path, expected_sha, STIMULUS_FIELDS)
    seen = set()
    for row in rows:
        identifier, text, stops = row['id'], row['text'], row['stops']
        if (type(identifier) is not str or not identifier or identifier in seen or type(text) is not str
                or type(stops) is not list or len(stops) < 2 or any(type(stop) is not int for stop in stops)
                or stops != sorted(set(stops)) or not 0 < stops[0] < stops[-1] == len(text)):
            raise ValueError('Invalid independent readback stimulus boundary or identifier')
        seen.add(identifier)
    return rows


def forecast(articles, features_path, heads_path, destination):
    import numpy as np

    from research.core_features.fit import apply_scaler, read_features, sigmoid, write_rows

    if digest(heads_path) != HEADS_SHA256:
        raise ValueError('Expected the immutable pooled readback head archive')
    shapes = {'center': (3,), 'scale': (3,), 'prior': (1,), 'position': (4,), 'qwen': (2052,)}
    with np.load(heads_path, allow_pickle=False) as archived:
        weights = {name: archived[f'pooled_{name}'] for name in shapes}
    if (any(value.shape != shapes[name] or value.dtype != np.float64 or not np.isfinite(value).all()
            for name, value in weights.items()) or np.any(weights['scale'] <= 0)):
        raise ValueError('The archived pooled readout is incompatible with the registered state width')
    states = read_features(features_path)
    if states.keys() != {(row['id'], stop) for row in articles for stop in row['stops']}:
        raise ValueError('Not all original stop boundaries were encoded')

    rows = [(row, stop) for row in articles for stop in row['stops'][1:]]
    positions = np.asarray([geometry(row, stop) for row, stop in rows])
    normalized = apply_scaler(positions, weights['center'], weights['scale'])
    design = np.column_stack((np.ones(len(rows)), normalized))
    vectors = np.stack([states[row['id'], stop] for row, stop in rows])
    inputs = (design[:, :1], design, np.column_stack((design, vectors)))
    probabilities = np.column_stack([sigmoid(design @ weights[method])
                                     for method, design in zip(METHODS, inputs, strict=True)])
    if not np.all(np.isfinite(probabilities) & (0 < probabilities) & (probabilities < 1)):
        raise ValueError('Frozen readback probabilities must be finite and interior')
    write_rows(destination, [{'id': row['id'], 'stop': stop,
                              'methods': {name: float(p) for name, p in zip(METHODS, values, strict=True)}}
                             for (row, stop), values in zip(rows, probabilities, strict=True)])
    return len(rows)


def run(args):
    from research.core_features.extract import run as extract
    from research.gradio_app.checkpoint import BACKBONE, load_checkpoint

    articles = read_stimuli(args.stimuli, args.stimuli_sha256)
    checkpoint = load_checkpoint(args.checkpoint)
    if (checkpoint.identifier != CANDIDATE or digest(args.heads) != HEADS_SHA256
            or args.model.name != BACKBONE['revision']
            or any(digest(args.model / name) != expected for name, expected in BACKBONE['snapshot_sha256'].items())):
        raise ValueError('Expected the certified Gradio candidate and frozen prior-experiment heads')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix='.context-readback-output-') as temporary:
        staging = Path(temporary)
        features = staging / 'features'
        extract(args.stimuli, features, max_batch_tokens=6144)
        result = staging / 'result'
        result.mkdir()
        predictions = result / 'predictions.jsonl'
        stops = forecast(articles, features / 'features.npz', args.heads, predictions)
        for name in ('extraction.json', 'causality.json', 'telemetry.jsonl'):
            shutil.copyfile(features / name, result / name)
        seal = {f'{name}_sha256': getattr(args, f'{name}_sha256') for name in ('stimuli', 'targets', 'assay', 'protocol')}
        seal.update(heads_sha256=HEADS_SHA256, predictions_sha256=digest(predictions),
                    extraction_sha256=digest(result / 'extraction.json'), causality_sha256=digest(result / 'causality.json'),
                    source_code_sha256=digest(Path(__file__)), candidate_id=checkpoint.identifier,
                    model_revision=BACKBONE['revision'], documents=len(articles), forecast_stops=stops)
        (result / 'seal.json').write_text(json.dumps(seal, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        result.rename(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'checkpoint', 'heads', 'stimuli', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    for name in ('stimuli', 'targets', 'assay', 'protocol'):
        parser.add_argument(f'--{name}-sha256', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
