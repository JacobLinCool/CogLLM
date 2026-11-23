"""Freeze text-only Frick forecasts on Currie materials, then score the sealed forecasts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from research.semantic_tracking import beukers_extract as extraction
from research.semantic_tracking import frick_consistency_score as frick
from research.semantic_tracking import frick_pair_score as readout
from research.semantic_tracking import larrc_consistency_evaluate as evaluation


PROTOCOL = Path(__file__).with_name('larrc-consistency-protocol.md')
FRICK_ARCHIVE = Path(__file__).resolve().parents[1] / 'results' / 'frick-pair'
FRICK_SEAL_SHA256 = '11639d39ffcab1cd413fc47eb451b6cca038d515c336951b204dae66cc030d08'
DOCUMENTS = 48
SENTENCES = 8
MODELS = (*readout.METHODS, 'constant')
BASE_RATE = 754 / 1800
SOURCE_SHA256 = '89480b61bf51031b9fafe1d79e01b40197928f7bc082b5f66b27424b2b88c3be'


def digest_match(path, expected):
    actual = frick.digest(path)
    if actual != expected:
        raise ValueError(f'SHA-256 mismatch for {path}')
    return actual


def read_frick_heads(archive):
    seal = archive / 'forecast-seal.json'
    digest_match(seal, FRICK_SEAL_SHA256)
    manifest = frick.read_json(seal)
    fits_path, directions_path = archive / 'fits.json', archive / 'directions.npz'
    for path in (fits_path, directions_path):
        digest_match(path, manifest['file_sha256'][path.name])
    fits = frick.read_json(fits_path)
    if fits['features'] != list(readout.FEATURES) or fits['methods'] != list(readout.METHODS):
        raise ValueError('The archived scores use unexpected features or training arms')
    folds = {}
    for fitted in fits['folds']:
        if fitted['task'] == 'first':
            key, method = fitted['direction_key'], fitted['method']
            methods = folds.setdefault(key, {})
            if fitted['train_pairs'] != 19 or method in methods:
                raise ValueError('Expected each 19-story first-event training fold once')
            methods[method] = fitted
    with np.load(directions_path, allow_pickle=False) as stored:
        directions = {key: stored[key] for key in stored.files}
    if len(folds) != 20 or folds.keys() != directions.keys() or any(
        set(fitted) != set(readout.METHODS) for fitted in folds.values()
    ):
        raise ValueError('The archived first-event heads must cover all twenty folds and three arms')
    for axis in directions.values():
        if axis.shape != (frick.WIDTH,) or not np.isclose(np.linalg.norm(axis), 1, atol=1e-12, rtol=0):
            raise ValueError('The sealed axes must be finite unit vectors in the archived state space')
    return [(directions[key], folds[key]) for key in sorted(folds)], manifest['provenance']


def packet_inputs(packet):
    stimuli = packet / 'stimuli.jsonl'
    preflight_path = packet / 'extractor-preflight.json'
    lock = frick.read_json(packet / 'preflight.json')
    stimulus_hash = frick.digest(stimuli)
    rows = extraction.read_stimuli(stimuli)
    preflight = frick.read_json(preflight_path)
    extraction.validate_preflight(preflight, rows, stimulus_hash)
    stops = {row['id']: row['stops'] for row in rows}
    expected = {
        'schema': 1, 'candidate_id': extraction.CANDIDATE, 'stimuli_sha256': stimulus_hash,
        'tokenizer_sha256': extraction.TOKENIZER_SHA256, 'documents': stops,
        'extractor_preflight_sha256': frick.digest(preflight_path), 'protocol_sha256': frick.digest(PROTOCOL),
        'frick_forecast_seal_sha256': FRICK_SEAL_SHA256, 'source_audit_sha256': SOURCE_SHA256,
    }
    if (lock != expected or len(rows) != DOCUMENTS or any(
        len(row['stops']) != SENTENCES or row['stops'][-1] != len(row['text'])
        or row['id'] != sha256(row['text'].encode()).hexdigest() for row in rows
    )):
        raise ValueError('The Currie packet does not match the fixed endpoint and text inventory')
    return rows, lock, preflight


def load_states(packet, states_dir, rows, lock, preflight, frick_provenance):
    seal_path = states_dir / 'seal.json'
    seal = frick.read_json(seal_path)
    expected_shape = (DOCUMENTS * SENTENCES, frick.WIDTH)
    scope = {
        'assay_sha256': frick.digest(packet / 'preflight.json'), 'protocol_sha256': lock['protocol_sha256'],
        'preflight_sha256': lock['extractor_preflight_sha256'],
        'source_code_sha256': frick.digest(Path(extraction.__file__)),
        'content_forward_per_document': 1, 'raw_stops': True, 'states_dtype': 'float32',
        'states_shape': list(expected_shape), 'independent_causal_checks': 12,
        **{key: frick_provenance[key] for key in ('model', 'revision', 'transformers', 'candidate_id',
                                               'tokenizer_sha256', 'anchor_heads_sha256', 'projection_sha256')},
    }
    if any(seal.get(key) != value for key, value in (preflight | scope).items()) or 'L4' not in seal['gpu']:
        raise ValueError('The extraction must match the exact L4 packet and archived Qwen candidate')
    for name in ('coordinates.jsonl', 'causality.json', 'states.npy'):
        digest_match(states_dir / name, seal['file_sha256'][name])
    coordinates = frick.read_rows(states_dir / 'coordinates.jsonl')
    if coordinates != [{'id': row['id'], 'stops': row['stops']} for row in rows]:
        raise ValueError('Every raw-state slice must match the ordered certified source coordinates')
    checks = frick.read_json(states_dir / 'causality.json')['checks']
    if len(checks) != 12 or not all(row['passed'] and row['projected_tokens']['passed'] for row in checks):
        raise ValueError('The independent L4 causality checks did not pass')
    states = np.load(states_dir / 'states.npy', mmap_mode='r', allow_pickle=False)
    if states.shape != expected_shape or states.dtype != np.float32 or not np.isfinite(states).all():
        raise ValueError('L4 output must contain every source sentence stop as a finite FP32 vector')
    return states.reshape((DOCUMENTS, SENTENCES, frick.WIDTH)), frick.digest(seal_path)


def article_sentences(row):
    text, stops = row['text'], row['stops']
    starts = [0, *(stop + 2 for stop in stops[:-1])]
    sentences = [text[start:stop] for start, stop in zip(starts, stops, strict=True)]
    if any(not sentence or sentence != sentence.strip() for sentence in sentences) or '\n\n'.join(sentences) != text:
        raise ValueError('Article sentences do not reproduce the exact grade-free rendering')
    return sentences


def forecast_article(row, states, folds):
    lexical = readout.sentence_forms(article_sentences(row))
    unit_states = readout.normalize_states(states)
    cumulative = {method: np.zeros(SENTENCES - 1, dtype=np.float64) for method in readout.METHODS}
    for axis, fits in folds:
        design = np.column_stack((lexical, readout.project_story(unit_states, axis)))
        for method in readout.METHODS:
            logits = frick.predict_logits(design, fits[method])
            log_survival = -np.cumsum(np.logaddexp(0, logits))
            cumulative[method] -= np.expm1(log_survival)
    curves = {method: (trajectory / len(folds)).tolist() for method, trajectory in cumulative.items()}
    if any(not 0 < values[0] <= values[-1] < 1 or np.any(np.diff(values) < 0) for values in curves.values()):
        raise ValueError('Accumulating first-event probability must yield a finite increasing probability')
    return {'id': row['id'], 'stops': row['stops'][1:], 'predicted_cumulative_report': curves,
            'methods': {**{method: values[-1] for method, values in curves.items()}, 'constant': BASE_RATE}}


def forecast(packet, states_dir, archive, output):
    if output.exists():
        raise FileExistsError(output)
    rows, lock, preflight = packet_inputs(packet)
    folds, provenance = read_frick_heads(archive)
    states, state_hash = load_states(packet, states_dir, rows, lock, preflight, provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.larrc-forecast-') as work:
        provisional = Path(work) / 'forecasts'
        provisional.mkdir()
        forecasts = provisional / 'forecasts.jsonl'
        with forecasts.open('x', encoding='utf-8') as stream:
            for row, vectors in zip(rows, states, strict=True):
                stream.write(json.dumps(forecast_article(row, vectors, folds), sort_keys=True, allow_nan=False) + '\n')
        frick.write_json(provisional / 'forecast-seal.json', {
            'forecasts_sha256': frick.digest(forecasts), 'packet_sha256': frick.digest(packet / 'preflight.json'),
            'state_extraction_sha256': state_hash, 'frick_forecast_seal_sha256': FRICK_SEAL_SHA256,
            'protocol_sha256': lock['protocol_sha256'], 'source_audit_sha256': SOURCE_SHA256,
            'scorer_sha256': frick.digest(Path(__file__)), 'scored_documents': len(rows),
        })
        provisional.rename(output)


def score(packet, forecasts, prepared, output):
    seal_path = forecasts / 'forecast-seal.json'
    seal = frick.read_json(seal_path)
    rows, lock, _ = packet_inputs(packet)
    expected = {
        'packet_sha256': frick.digest(packet / 'preflight.json'), 'source_audit_sha256': SOURCE_SHA256,
        'protocol_sha256': lock['protocol_sha256'], 'scorer_sha256': frick.digest(Path(__file__)),
        'frick_forecast_seal_sha256': FRICK_SEAL_SHA256, 'scored_documents': DOCUMENTS,
    }
    if any(seal.get(key) != value for key, value in expected.items()):
        raise ValueError('Cannot score forecasts from another model, source or protocol')
    forecast_path = forecasts / 'forecasts.jsonl'
    digest_match(forecast_path, seal['forecasts_sha256'])
    predictions = frick.read_rows(forecast_path)
    if any(
        source['id'] != row['id'] or source['stops'][1:] != row['stops']
        or set(row['methods']) != set(MODELS) or row['methods']['constant'] != BASE_RATE
        for source, row in zip(rows, predictions, strict=True)
    ):
        raise ValueError('The archived predictions do not match every certified text')

    audit_path = prepared / 'source-audit.json'
    digest_match(audit_path, SOURCE_SHA256)
    audit = frick.read_json(audit_path)
    for name in ('stimuli.jsonl', 'layout.jsonl', 'outcomes.jsonl'):
        digest_match(prepared / name, audit['output_sha256'][name])
    layout = frick.read_rows(prepared / 'layout.jsonl')
    outcomes = frick.read_rows(prepared / 'outcomes.jsonl')
    if (len(outcomes) != 4248 or len({row['reader'] for row in outcomes}) != 179
            or len({row['family'] for row in layout}) != 24
            or sum(row['same_sentence_inventory'] for row in layout) != 40):
        raise ValueError('Ground-truth outcomes do not cover the complete registered study slice')
    reduced = [{'id': row['id'], 'methods': row['methods']} for row in predictions]
    report = evaluation.evaluate(layout, outcomes, reduced)
    frick.write_json(output, {
        **report, 'forecast_seal_sha256': frick.digest(seal_path),
        'evaluation_sha256': frick.digest(Path(evaluation.__file__)), 'inputs_sha256': audit['output_sha256'],
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    infer = commands.add_parser('forecast')
    assess = commands.add_parser('score')
    for command in (infer, assess):
        command.add_argument('--packet', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    infer.add_argument('--states-dir', type=Path, required=True)
    infer.add_argument('--archive', type=Path, default=FRICK_ARCHIVE)
    assess.add_argument('--forecasts', type=Path, required=True)
    assess.add_argument('--prepared', type=Path, required=True)
    args = vars(parser.parse_args())
    (forecast if args.pop('command') == 'forecast' else score)(**args)
