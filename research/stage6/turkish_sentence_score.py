"""Evaluate the exact Gradio reading-ease score on matched Turkish sentence preferences."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.gradio_app.causal_plaba import CausalPlabaReader, digest
from research.gradio_app.checkpoint import DEFAULT_CHECKPOINT, load_checkpoint
from research.gradio_app.plaba import sentence_spans
from research.semantic_tracking.gum_bridge_rank import read_rows
from research.stage6.external_projection_prepare import PROTOCOL, STIMULI, read


SURFACE_COLUMNS = ('char_count', 'syllable_count')
EASE_COLUMN = len(SURFACE_COLUMNS) + 2
METHODS = {'order': (), 'surface': tuple(range(EASE_COLUMN)), 'app': (EASE_COLUMN,),
           'surface_app': tuple(range(EASE_COLUMN + 1))}
PENALTY = 0.01


@dataclass(frozen=True)
class Pairs:
    features: np.ndarray
    observed: np.ndarray
    reader: list[str]
    component: list[str]
    quality: np.ndarray
    pair: list[tuple[str, str]]
    available: int


def evidence(packet, extraction, source):
    assay_path = packet / 'assay.json'
    assay = read(assay_path)
    source_manifest = read(source / 'assay.json')
    receipt_path = extraction / 'seal.json'
    projection = read(receipt_path)
    checkpoint = load_checkpoint(DEFAULT_CHECKPOINT)
    files = projection['file_sha256']
    if (projection['candidate_id'] != checkpoint.identifier or digest(assay_path) != projection['assay_sha256']
            or not digest(PROTOCOL) == projection['protocol_sha256'] == assay['protocol_sha256']
            or digest(source / 'assay.json') != assay['turkish_assay_sha256']
            or digest(source / STIMULI) != source_manifest['file_sha256'][STIMULI]
            or not digest(packet / STIMULI) == projection['stimuli_sha256'] == assay['file_sha256'][STIMULI]
            or any(digest(extraction / name) != files[name] for name in ('coordinates.jsonl', 'terminal_states.npy'))):
        raise ValueError('The Turkish source, candidate and sealed terminal vectors must match the frozen protocol')
    return assay, source_manifest, projection, checkpoint


def score_sentences(source, extraction, assay, projection, checkpoint):
    stimuli = read_rows(source / STIMULI)
    coords = read_rows(extraction / 'coordinates.jsonl')
    indices = {row['id']: index for index, row in enumerate(coords)}
    if len(indices) != len(coords) or [row['id'] for row in stimuli] != assay['partitions']['turkish']:
        raise ValueError('Unexpected sentence or coordinate inventory')
    terminal = np.load(extraction / 'terminal_states.npy', allow_pickle=False, mmap_mode='r')
    if terminal.shape != (len(coords), 2048) or list(terminal.shape) != projection['terminal_states_shape']:
        raise ValueError('Unexpected full sentence-end state dimensions')
    keep, spans = [], {}
    segmenter = checkpoint.segmenter()
    for row in stimuli:
        text = row['text']
        index = indices[row['id']]
        if coords[index]['stops'] != row['stops'] or row['stops'] != [len(text)]:
            raise ValueError(f'Expected a certified full terminal stop for {row["id"]}')
        bounds = sentence_spans(text, segmenter)
        if bounds == [(0, len(text))]:
            keep.append(row)
        else:
            spans[row['id']] = len(bounds)
    reader = CausalPlabaReader(checkpoint.directory, backbone=object(), checkpoint=checkpoint.directory / 'sentence',
                               segmenter=segmenter)
    scores = reader.score(terminal[[indices[row['id']] for row in keep]], [row['text'] for row in keep])
    return {row['id']: float(ease) for row, (ease, _) in zip(keep, scores, strict=True)}, spans


def pairs(source, release, split, ease):
    names = ('sentence-metadata.jsonl', f'{split}.jsonl', f'{split}-truth.jsonl')
    if any(digest(source / name) != release['file_sha256'][name] for name in names):
        raise ValueError(f'{split} mapping or human labels differ from the sealed Turkish source')
    descriptions = read_rows(source / names[0])
    metadata = {row['id']: row for row in descriptions}
    display_length = {row['id']: (len(row['text']), len(row['text'].split())) for row in read_rows(source / STIMULI)}
    observed = read_rows(source / names[2])
    truth = {row['id']: row for row in observed}
    mapping = read_rows(source / names[1])
    if len(descriptions) != len(metadata) or len(observed) != len(truth) or len(mapping) != len(truth):
        raise ValueError('Duplicate Turkish material or judgment IDs')
    if len({row['id'] for row in mapping}) != len(truth) or truth.keys() != {row['id'] for row in mapping}:
        raise ValueError('The displayed sentence pairs and human observations must match exactly')
    values, choices, reader, component, quality, pairs_used = [], [], [], [], [], []
    for entry in mapping:
        response = truth[entry['id']]
        a, b = entry['earlier_id'], entry['later_id']
        left, right = metadata[a], metadata[b]
        if (a == b or left['component_id'] != right['component_id'] or left['word_count'] != right['word_count']
                or response['earlier_easier'] not in (True, False) or response['quality_retained'] not in (True, False)):
            raise ValueError('Expected an ordered exact equal-word pair and a binary source response')
        if a not in ease or b not in ease:
            continue
        form = [np.log1p(left[column]) - np.log1p(right[column]) for column in SURFACE_COLUMNS]
        form.extend(np.log1p(display_length[a]) - np.log1p(display_length[b]))
        values.append([*form, ease[a] - ease[b]])
        choices.append(response['earlier_easier'])
        reader.append(response['reader_id'])
        component.append(left['component_id'])
        quality.append(response['quality_retained'])
        pairs_used.append(tuple(sorted((a, b))))
    if not values:
        raise ValueError(f'No App-segmentable strict equal-word pairs in {split}')
    return Pairs(np.asarray(values), np.asarray(choices, dtype=np.float64), reader, component,
                 np.asarray(quality, dtype=bool), pairs_used, len(mapping))


def design(values, columns):
    return np.column_stack((np.ones(len(values)), values[:, columns]))


def fit(data):
    scale = np.sqrt(np.mean(np.square(data.features), axis=0))
    scale[scale < 1e-12] = 1.0
    features = data.features / scale
    coefficients = {}
    for name, columns in METHODS.items():
        matrix = design(features, columns)

        def objective(beta):
            logits = matrix @ beta
            penalized = beta.copy()
            penalized[0] = 0.0
            nll = np.mean(np.logaddexp(0, logits) - data.observed * logits)
            gradient = matrix.T @ (expit(logits) - data.observed) / len(logits)
            return nll + PENALTY / 2 * (penalized @ penalized), gradient + PENALTY * penalized

        bounds = [(None, None), *((0, None) if column == EASE_COLUMN else (None, None) for column in columns)]
        result = minimize(objective, np.zeros(matrix.shape[1]), jac=True, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9})
        if not result.success or not np.all(np.isfinite(result.x)):
            raise ValueError(f'Turkish preference calibration failed for {name}: {result.message}')
        coefficients[name] = result.x.tolist()
    return {'rms_scale': scale.tolist(), 'coefficients': coefficients}


def cross_cluster_ci(delta, reader, component):
    indices = [np.unique(values, return_inverse=True)[1] for values in (reader, component)]
    rng = np.random.default_rng(320481)
    simulations = []
    for _ in range(40):
        weights = np.ones((250, len(delta)), dtype=np.int32)
        for index in indices:
            clusters = int(np.max(index)) + 1
            counts = rng.multinomial(clusters, np.full(clusters, 1.0 / clusters), size=len(weights))
            weights *= counts[:, index]
        denominator = weights.sum(axis=1)
        nonempty = denominator > 0
        simulations.extend((weights[nonempty] @ delta / denominator[nonempty]).tolist())
    if len(simulations) < 9900:
        raise ValueError('Insufficient nonempty crossed-cluster resamples')
    return np.quantile(simulations, [0.025, 0.975]).tolist()


def counts(data):
    return {'available': data.available, 'scored': len(data.observed), 'unique_pairs': len(set(data.pair)),
            'readers': len(set(data.reader)), 'components': len(set(data.component))}


def score(data, params):
    scaled = data.features / params['rms_scale']
    profiles, losses = {}, {}
    for name, columns in METHODS.items():
        logits = design(scaled, columns) @ params['coefficients'][name]
        predictions = expit(logits)
        losses[name] = np.logaddexp(0, logits) - data.observed * logits
        brier = np.square(predictions - data.observed)
        profiles[name] = {'nll_nats': float(losses[name].mean()), 'brier': float(brier.mean()),
                          'quality_retained_nll_nats': float(losses[name][data.quality].mean()),
                          'quality_retained_brier': float(brier[data.quality].mean())}
    delta = losses['surface_app'] - losses['surface']
    ci = cross_cluster_ci(delta, data.reader, data.component)
    n = counts(data)
    return {'all': n, 'quality_retained': int(data.quality.sum()), 'metrics': profiles,
            'combined_minus_surface_nll': float(delta.mean()), 'combined_minus_surface_ci95': ci,
            'passes': bool(n['components'] >= 6 and n['unique_pairs'] >= 100 and ci[-1] < 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('fit', 'score'))
    for option in ('packet', 'extraction', 'source', 'fit', 'output'):
        parser.add_argument(f'--{option}', required=option != 'output', type=Path)
    args = parser.parse_args()
    assay, release, projection, checkpoint = evidence(args.packet, args.extraction, args.source)
    ease, unavailable = score_sentences(args.source, args.extraction, assay, projection, checkpoint)
    provenance = {'source_sha256': assay['turkish_assay_sha256'], 'packet_sha256': projection['assay_sha256'],
                  'projection_sha256': digest(args.extraction / 'seal.json'), 'candidate_id': checkpoint.identifier,
                  'protocol_sha256': digest(PROTOCOL), 'scorer_sha256': digest(Path(__file__)),
                  'fit_truth_sha256': release['file_sha256']['fit-truth.jsonl']}
    if args.command == 'fit':
        samples = pairs(args.source, release, 'fit', ease)
        result = {'provenance': provenance, 'fit_counts': counts(samples), 'parameters': fit(samples)}
        destination = args.fit
    else:
        if args.output is None:
            parser.error('score requires --output')
        sealed = read(args.fit)
        if sealed['provenance'] != provenance:
            raise ValueError('The calibration fit does not match the fixed evaluation data and code')
        samples = pairs(args.source, release, 'test', ease)
        result = {'fit_sha256': digest(args.fit), 'source_sha256': provenance['source_sha256'],
                  'test_truth_sha256': release['file_sha256']['test-truth.jsonl'],
                  'sentences_app_segmentable': len(ease), 'sentences_excluded_by_app_segmenter': len(unavailable),
                  **score(samples, sealed['parameters'])}
        destination = args.output
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
