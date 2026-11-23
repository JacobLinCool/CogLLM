"""Evaluate frozen binary Sense readouts without fitting on Blott 2023 responses."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit

from research.gradio_app.checkpoint import DEFAULT_CHECKPOINT, load_checkpoint
from research.gradio_app.ordinal import LEVELS, OrdinalReadout
from research.semantic_tracking import blott_format_calibration as calibration
from research.semantic_tracking import frick_consistency_score as records


DRAWS, SEED, BATCH = 10_000, 735972, 128
ARMS = ('Amb', 'UA')
RANKERS = ('understanding', 'ease', 'shorter_characters', 'shorter_words')
CALIBRATION_SHA256 = '0915fa87ec729286bb3f0eaf85a3a077eab2c26aeeebea6fc148717b468f6244'
SOURCE_AUDIT_SHA256 = 'a5ca8d67c87ed617d73de168b2b95e52356c57c69b9310590af20a5e37e58849'


def load_forecasts(prepared, packet, extraction, checkpoint_path, calibration_path):
    if (records.digest(calibration.PROTOCOL) != calibration.PROTOCOL_SHA256
            or records.digest(calibration_path) != CALIBRATION_SHA256):
        raise ValueError('Blott transfer protocol or frozen calibration has changed')
    checkpoint = load_checkpoint(checkpoint_path)
    ordinal = OrdinalReadout(checkpoint_path / 'ordinal')
    fitted = records.read_json(calibration_path)
    expected = {'candidate_id': checkpoint.identifier, 'protocol_sha256': calibration.PROTOCOL_SHA256,
                'features': list(calibration.FEATURES), 'training_responses': 2256,
                'scorer_sha256': records.digest(Path(calibration.__file__))}
    if any(fitted[key] != value for key, value in expected.items()):
        raise ValueError('Calibration does not match the frozen Blott candidate and protocol')

    input_file = packet / 'stimuli.jsonl'
    original = prepared / 'stimuli.jsonl'
    receipt = records.read_json(packet / 'preflight.json')
    seal = records.read_json(extraction / 'seal.json')
    scope = {'candidate_id': checkpoint.identifier, 'protocol_sha256': calibration.PROTOCOL_SHA256,
             'stimuli_sha256': records.digest(input_file),
             'preflight_sha256': records.digest(packet / 'extractor-preflight.json'),
             'assay_sha256': records.digest(packet / 'preflight.json'),
             'content_forward_per_document': 1, 'states_dtype': 'float32', 'raw_stops': True}
    if (receipt['original_stimuli_sha256'] != records.digest(original)
            or any(seal.get(key) != value for key, value in scope.items())
            or any(records.digest(extraction / name) != seal['file_sha256'][name] for name in
                   ('states.npy', 'coordinates.jsonl', 'causality.json'))):
        raise ValueError('The Blott model export differs from the grade-free packet or frozen checkpoint')
    checks = records.read_json(extraction / 'causality.json')['checks']
    if len(checks) != 12 or not all(row['passed'] is True and row['projected_tokens']['passed'] is True for row in checks):
        raise ValueError('Blott requires all registered extractor causality checks')
    documents = records.indexed(records.read_rows(input_file), 'Blott packet')
    coordinates = records.read_rows(extraction / 'coordinates.jsonl')
    if (len(coordinates) != len(documents) or {item['id']: item['stops'] for item in coordinates} != receipt['documents']
            or receipt['documents'] != {key: item['stops'] for key, item in documents.items()}):
        raise ValueError('The export does not cover each certified source stop exactly')
    states = np.load(extraction / 'states.npy', mmap_mode='r', allow_pickle=False)
    if states.dtype != np.float32 or list(states.shape) != seal['states_shape']:
        raise ValueError('Expected the exact archived FP32 stop states')
    forecasts = []
    offset = 0
    for doc in coordinates:
        key, stops = doc['id'], doc['stops']
        text = documents[key]['text']
        prefixes = [text[:stop] for stop in stops]
        distributions = ordinal.score(states[offset:offset + len(stops)], prefixes)
        offset += len(stops)
        for stop, prefix, distribution in zip(stops, prefixes, distributions, strict=True):
            understanding = float(np.dot(LEVELS, distribution.understanding))
            ease = float(np.dot(LEVELS, distribution.ease))
            standardized = (calibration.inputs(prefix, understanding) - fitted['center']) / fitted['scale']
            design = np.r_[1., standardized]
            logits = {name: float(design[:width + 1] @ fitted['methods'][name]['coefficients'])
                      for name, width in calibration.METHODS.items()}
            forecasts.append({'id': key, 'stop': stop, 'understanding': understanding, 'ease': ease,
                              'shorter_characters': -len(prefix), 'shorter_words': -len(prefix.split()),
                              'logits': logits})
    if offset != len(states):
        raise ValueError('A raw state is unbound to the exact grade-free prefix')
    return forecasts


def evaluate_batches(positive, present, logits, scores):
    rng = np.random.default_rng(SEED)
    n_readers, n_families = present.shape
    log_loss_negative = np.logaddexp(0, logits)
    effects = np.diff(expit(logits), axis=1)[:, 0]
    orders = np.sign(np.diff(scores, axis=1)[:, 0])

    def batch(reader_weights, family_weights):
        denominator = reader_weights @ present
        sampled = family_weights > 0
        if np.any(sampled & (denominator < 1)):
            raise ValueError('A sampled family has no complete written-response pairs')
        rates = np.einsum('br,rfa->bfa', reader_weights, positive) / np.maximum(denominator[..., None], 1)
        human = rates[:, :, 1] - rates[:, :, 0]
        loss = (log_loss_negative[None, ...] - rates[..., None] * logits[None, ...]).mean(axis=2)
        rank = (1 + np.sign(human[..., None]) * orders[None, ...]) / 2
        weights = family_weights / family_weights.sum(axis=1, keepdims=True)
        return np.column_stack(((loss[..., 0] * weights).sum(axis=1), (loss[..., 1] * weights).sum(axis=1),
                                ((loss[..., 0] - loss[..., 1]) * weights).sum(axis=1),
                                (human * weights).sum(axis=1), weights @ effects,
                                ((effects[None, :, 1] - human) * weights).sum(axis=1),
                                np.einsum('bf,bfm->bm', weights, rank)))

    point = batch(np.ones((1, n_readers)), np.ones((1, n_families)))[0]
    draws = []
    for start in range(0, DRAWS, BATCH):
        size = min(BATCH, DRAWS - start)
        reader_weights = rng.multinomial(n_readers, np.full(n_readers, 1 / n_readers), size)
        family_weights = rng.multinomial(n_families, np.full(n_families, 1 / n_families), size)
        draws.append(batch(reader_weights, family_weights))
    lower, upper = np.quantile(np.concatenate(draws), [0.025, 0.975], axis=0)
    names = ('surface_nll', 'qwen_nll', 'qwen_nll_gain', 'human_effect', 'surface_effect', 'qwen_effect',
             'qwen_minus_human_effect', *(name + '_rank_agreement' for name in RANKERS))
    summary = {name: {'estimate': float(value), 'ci95': [float(low), float(high)]}
               for name, value, low, high in zip(names, point, lower, upper, strict=True)}
    gates = {'nll_gain': lower[2] > 0., 'effect_direction': min(lower[3], lower[5]) > 0.,
             'effect_equivalence': lower[6] > -0.05 and upper[6] < 0.05}
    return summary, {name: bool(value) for name, value in gates.items()}


def evaluate(prepared, forecasts):
    audit = records.read_json(prepared / 'source-audit.json')
    for name in ('stimuli.jsonl', 'layout.jsonl', 'outcomes.jsonl'):
        if records.digest(prepared / name) != audit['prepared_sha256'][name]:
            raise ValueError(f'Sanitized Blott source artifact checksum differs for {name}')
    layout = records.indexed(records.read_rows(prepared / 'layout.jsonl'), 'Blott layout')
    candidates = {(row['id'], row['stop']): row for row in forecasts}
    if len(candidates) != len(forecasts) or {key for key, _ in candidates} != layout.keys():
        raise ValueError('The Blott readout does not cover the retained published designs')
    trial = {}
    for row in records.read_rows(prepared / 'outcomes.jsonl'):
        item = layout[row['id']]
        key = (row['reader_id'], item['family'], item['ambiguity'])
        if key in trial or type(row['sense']) is not bool or not row['rt_ms'] > 250:
            raise ValueError('Expected one eligible binary response per written reader, family and version')
        trial[key] = row['sense']
    pairs = sorted({(reader, family) for reader, family, _ in trial
                    if all((reader, family, arm) in trial for arm in ARMS)})
    readers = {key: i for i, key in enumerate(sorted({reader for reader, _ in pairs}))}
    families = {key: i for i, key in enumerate(sorted({family for _, family in pairs}))}
    if not pairs or len(families) != len(layout) // len(ARMS):
        raise ValueError('Every frozen pair requires matched written responses after exclusions')
    present = np.zeros((len(readers), len(families)), dtype=np.int16)
    positive = np.zeros((*present.shape, len(ARMS)), dtype=np.int16)
    for reader, family in pairs:
        index = readers[reader], families[family]
        present[index] = 1
        positive[index] = [trial[reader, family, arm] for arm in ARMS]
    by_family = {(row['family'], row['ambiguity']): row for row in layout.values()}
    ordered = [[by_family[family, arm] for arm in ARMS] for family in families]
    final = [[candidates[row['id'], row['sentence_stop']] for row in group] for group in ordered]
    logits = np.asarray([[[row['logits'][method] for method in calibration.METHODS] for row in group] for group in final])
    scores = np.asarray([[[row[method] for method in RANKERS] for row in group] for group in final])
    summary, gates = evaluate_batches(positive, present, logits, scores)

    cue_changes = []
    for group in ordered:
        stops = [[(row['id'], row['before_cue_stop']), (row['id'], row['cue_span'][1])] for row in group]
        if all(stop in candidates for path in stops for stop in path):
            cue = np.asarray([[candidates[stop]['understanding'] for stop in path] for path in stops])
            cue_changes.append(float(np.diff(np.diff(cue, axis=1), axis=0)[0, 0]))
    return {'readers': len(readers), 'families': len(families), 'complete_pairs': len(pairs),
            'eligible_outcomes': len(trial), 'matched_outcomes': int(present.sum()) * 2,
            'summary': summary, 'gates': gates, 'pass': all(gates.values()),
            'cue_model_update': {'families': len(cue_changes), 'mean_ua_minus_amb_post_minus_pre':
                                 float(np.mean(cue_changes)) if cue_changes else None}}


def run(prepared, packet, extraction, checkpoint, calibration_file, output):
    if output.exists():
        raise FileExistsError(output)
    predictions = load_forecasts(prepared, packet, extraction, checkpoint, calibration_file)
    source_audit = prepared / 'source-audit.json'
    source_audit_sha = records.digest(source_audit)
    if source_audit_sha != SOURCE_AUDIT_SHA256:
        raise ValueError('The Blott evaluation requires the verified trial cohort')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.blott-format-') as temporary:
        stage = Path(temporary) / 'scored'
        stage.mkdir()
        path = stage / 'forecasts.json'
        records.write_json(path, predictions)
        seal = stage / 'forecast-seal.json'
        records.write_json(seal, {'forecast_sha256': records.digest(path),
                                  'protocol_sha256': calibration.PROTOCOL_SHA256,
                                  'extraction_sha256': records.digest(extraction / 'seal.json'),
                                  'calibration_sha256': records.digest(calibration_file),
                                  'source_audit_sha256': source_audit_sha,
                                  'scorer_sha256': records.digest(Path(__file__))})
        results = evaluate(prepared, records.read_json(path))
        if records.digest(source_audit) != source_audit_sha:
            raise ValueError('The retained cohort changed while scoring the sealed Blott forecasts')
        records.write_json(stage / 'results.json', {**results, 'forecast_seal_sha256': records.digest(seal)})
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('prepared', 'packet', 'extraction', 'calibration-file', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    run(**vars(parser.parse_args()))
