"""Prepare and seal an English SPR transfer assay for the classic SAP pairs."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import math
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote

from research.backbone_transfer.extract import MODELS, digest, read_json, stream_rows, write_json, write_rows
from research.semantic_tracking import context_readback_prepare as paape


PROTOCOL = Path(__file__).with_name('formal-sap-spr.md')
HOLDOUT = Path(__file__).with_name('holdout.json')
PAAPE_RT_SHA256 = '71c0704683c7ce30a9eae606a6ced566790055f7ff85898996daf0b1bb873de4'
CONSTRUCTIONS = ('MVRR', 'NPS', 'NPZ')
CONDITIONS = {f'{kind}_{condition}' for kind in CONSTRUCTIONS for condition in ('AMB', 'UAMB')}
REGIONS = (0, 1, 2)
PAAPE_REGIONS = frozenset({'crit', 'spill', 'final'})
SAP_POSITION_EXCLUSION = (14, 'NPZ', 2)


def check_digest(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Source snapshot changed: {path.name}')


def sap_files(source):
    family = next(row for row in read_json(HOLDOUT)['families'] if row['id'] == 'sap-garden-path')
    pins = {Path(row['path']).name: row['sha256'] for row in family['source_pins'] if 'path' in row}
    material = source / 'huang-material.csv'
    release = source / 'huang-ClassicGardenPathSet.csv'
    for path in (material, release):
        check_digest(path, pins[path.name])
    return material, release


def sap_materials(path):
    materials = {}
    with path.open(encoding='utf-8-sig', newline='') as file:
        for row in csv.reader(file):
            if len(row) != 7:
                raise ValueError('Unexpected SAP materials format')
            condition, sentence, *_, item = map(str.strip, row)
            sentence = sentence.replace("\\'", "'")
            if condition not in CONDITIONS:
                continue
            key = int(item), condition
            if materials.setdefault(key, sentence) != sentence:
                raise ValueError(f'SAP lists disagree about wording for {key}')
    if Counter(condition for _, condition in materials) != dict.fromkeys(CONDITIONS, 24):
        raise ValueError('Expected 24 materials per classic SAP condition')
    return materials


def projected_csv(path, columns):
    """Expose only explicitly named fields from the author's mixed file."""
    with path.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        if len(set(header)) != len(header):
            raise ValueError('Repeated SAP column')
        indices = tuple(header.index(column) for column in columns)
        for row in reader:
            if len(row) != len(header):
                raise ValueError('Ragged SAP CSV')
            yield tuple(row[index] for index in indices)


def sap_geometry(materials, release):
    positions = {}
    columns = ('item', 'Type', 'Sentence', 'ROI', 'WordPosition', 'CriticalPosition', 'EachWord')
    for row in projected_csv(release, columns):
        item, condition, sentence, roi, position, critical, word = row
        condition, roi = condition.strip(), roi.strip()
        if condition not in CONDITIONS or roi not in {'0', '1', '2'} or sentence == 'NA':
            continue
        key = int(item), condition
        position, critical, roi = int(position), int(critical), int(roi)
        if unquote(sentence.strip()) != materials[key] or position - critical != roi:
            raise ValueError(f'SAP author coordinates disagree with materials for {key}')
        point = position, unquote(word.strip())
        if positions.setdefault((*key, roi), point) != point:
            raise ValueError(f'Nonunique SAP location for {key}, ROI {roi}')
    if set(positions) != {(item, condition, roi) for item, condition in materials for roi in REGIONS}:
        raise ValueError('Incomplete SAP classic-pair location map')
    stimuli, targets = [], []
    for (item, condition), text in sorted(materials.items()):
        stimulus_id = f'sap-{item}-{digest_text(text)}'
        words = tuple(re.finditer(r'\S+', text))
        stops = []
        for roi in REGIONS:
            position, word = positions[item, condition, roi]
            if not 1 <= position <= len(words) or words[position - 1][0] != word:
                raise ValueError(f'SAP word fails exact display alignment for {item}, {condition}, ROI {roi}')
            stop = words[position - 1].end()
            stops.append(stop)
            targets.append({'id': stimulus_id, 'stop': stop, 'item': item, 'condition': condition,
                            'roi': roi, 'words': 1, 'word': word})
        stimuli.append({'id': stimulus_id, 'text': text, 'stops': stops})
    paired = {(p['item'], p['condition'], p['roi']): p['word'] for p in targets}
    mismatched = {(p['item'], p['condition'].split('_')[0], p['roi']) for p in targets
                  if p['condition'].endswith('_AMB')
                  and p['word'] != paired.get((p['item'], p['condition'].replace('_AMB', '_UAMB'), p['roi']))}
    if mismatched != {SAP_POSITION_EXCLUSION}:
        raise ValueError('SAP has an unexpected same-word alignment exception')
    return stimuli, targets


def digest_text(text):
    return sha256(text.encode()).hexdigest()[:16]


def paape_geometry(source):
    material = source / 'alt_stim.txt'
    check_digest(material, paape.SOURCE_SHA256[material.name])
    regions = paape.read_regions(material)
    designs, _ = paape.design(regions)
    stimuli, targets = [], []
    for (item, condition), stimulus in sorted(designs.items()):
        context, disambiguation = paape.CONDITIONS[condition]
        displays = regions[item, condition]
        critical = paape.CRITICAL_REGIONS[item]
        positions = {'crit': critical, 'spill': critical + 1, 'final': len(displays)}
        for region, position in positions.items():
            targets.append({'id': stimulus['id'], 'stop': stimulus['stops'][position - 1],
                            'item': item, 'region': region, 'position': position,
                            'context': context, 'disambiguation': disambiguation,
                            'words': len(displays[position - 1].split())})
        stops = sorted({stimulus['stops'][p - 1] for p in positions.values()})
        stimuli.append({**stimulus, 'stops': stops})
    return stimuli, targets


def prepare(paape_source, sap_source, output):
    material, release = sap_files(sap_source)
    sap_stimuli, sap_targets = sap_geometry(sap_materials(material), release)
    dev_stimuli, dev_targets = paape_geometry(paape_source)
    stimuli = sorted(dev_stimuli + sap_stimuli, key=lambda row: row['id'])
    output.mkdir()
    write_rows(output / 'stimuli.jsonl', stimuli)
    write_rows(output / 'paape-targets.jsonl', dev_targets)
    write_rows(output / 'sap-targets.jsonl', sap_targets)
    stats = {'development_documents': len(dev_stimuli), 'sap_documents': len(sap_stimuli),
             'stops': sum(len(row['stops']) for row in stimuli)}
    write_json(output / 'design.json', {
        **stats, 'protocol_sha256': digest(PROTOCOL), 'code_sha256': digest(Path(__file__)),
        'source_sha256': {path.name: digest(path) for path in (material, release, paape_source / 'alt_stim.txt')},
        'files_sha256': {path.name: digest(path) for path in sorted(output.iterdir())},
        'formal_columns_loaded': ['item', 'Type', 'Sentence', 'ROI', 'WordPosition', 'CriticalPosition', 'EachWord'],
    })
    return stats


def verified_design(root):
    design = read_json(root / 'design.json')
    check_digest(PROTOCOL, design['protocol_sha256'])
    for name in ('stimuli.jsonl', 'paape-targets.jsonl', 'sap-targets.jsonl'):
        check_digest(root / name, design['files_sha256'][name])
    return design


def processing_states(projection, design, heads):
    import numpy as np

    seal = read_json(projection / 'seal.json')
    model = read_json(MODELS)['models']['qwen']
    expected = {
        'model': model['repo'], 'revision': model['revision'], 'hidden_width': model['hidden_width'],
        'tokenizer_sha256': model['sha256']['tokenizer.json'], 'content_forward_per_document': 1,
        'gpu': 'NVIDIA L4', 'protocol_sha256': design['protocol_sha256'],
        'documents': design['development_documents'] + design['sap_documents'],
        'certified_stops': design['stops'], 'stimuli_sha256': design['files_sha256']['stimuli.jsonl'],
    }
    if any(seal.get(key) != value for key, value in expected.items()):
        raise ValueError('Projection does not match the fixed single-forward assay')
    for name in ('states.npy', 'coordinates.jsonl', 'causality.json'):
        check_digest(projection / name, seal['file_sha256'][name])
    coordinates = [(row['id'], stop) for row in stream_rows(projection / 'coordinates.jsonl') for stop in row['stops']]
    if len(set(coordinates)) != design['stops'] or len(coordinates) != design['stops']:
        raise ValueError('Projection positions are missing or repeated')
    states = np.load(projection / 'states.npy', allow_pickle=False).astype(np.float32, copy=False)
    with np.load(heads, allow_pickle=False) as parameters:
        weight = parameters['shared.processing.weight']
        bias = parameters['shared.processing.bias']
    if states.shape != (len(coordinates), model['hidden_width']) or weight.shape != (len(bias), states.shape[1]):
        raise ValueError('Incompatible projection or joint processing head')
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    hidden = states / np.maximum(norms, 1e-12) * states.shape[1] ** 0.5
    values = np.tanh(hidden @ weight.T + bias)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite processing values')
    return dict(zip(coordinates, values, strict=True))


def recording_key(row):
    return row['item'], row['context'], row['disambiguation'], row['region']


def paape_observations(path, rows):
    import rdata

    check_digest(path, PAAPE_RT_SHA256)
    by_position = {recording_key(row): row for row in rows}
    logs = defaultdict(list)
    reader_positions = set()
    table = rdata.read_rda(path)['meas3']
    columns = ['subj', 'item', 'context', 'disamb', 'wr', 'roi', 'FPRT']
    for reader, item, context, disambiguation, region, position, duration in table[columns].itertuples(index=False):
        key = int(item), str(context), str(disambiguation), str(region)
        if key not in by_position or int(position) != by_position[key]['position'] or (reader, key) in reader_positions:
            raise ValueError('Unexpected or duplicate Paape reader position')
        reader_positions.add((reader, key))
        if 150 < duration < 5000:
            logs[key].append(math.log(duration))
    if len(logs) != 628 or sum(map(len, logs.values())) != 31_933:
        raise ValueError('Eligible Paape reading observations differ from the audited source')
    return [({**by_position[key]}, math.fsum(values) / len(values)) for key, values in sorted(logs.items())]


def features(row, states):
    return [*states[row['id'], row['stop']], math.log(row['words'])]


def forecast(assay, projection, heads, paape_source, output):
    import numpy as np

    design = verified_design(assay)
    targets = list(stream_rows(assay / 'sap-targets.jsonl'))
    dev_targets = list(stream_rows(assay / 'paape-targets.jsonl'))
    states = processing_states(projection, design, heads)
    if states.keys() != {(row['id'], row['stop']) for row in targets + dev_targets}:
        raise ValueError('Projection does not cover the verified development and SAP targets')
    observations = paape_observations(paape_source / 'data_brms.Rda', dev_targets)
    x = np.array([features(row, states) for row, _ in observations], dtype=np.float64)
    y = np.array([value for _, value in observations], dtype=np.float64)
    means = x.mean(axis=0)
    centered = x - means
    slopes = np.linalg.solve(centered.T @ centered / len(y) + 0.01 * np.eye(x.shape[1]),
                             centered.T @ (y - y.mean()) / len(y))
    intercept = y.mean() - means @ slopes
    predictions = [{'id': row['id'], 'stop': row['stop'],
                    'forecast': float(intercept + np.dot(features(row, states), slopes))} for row in targets]
    if any(not math.isfinite(row['forecast']) for row in predictions) or output.exists():
        raise ValueError('Expected finite predictions and a fresh forecast directory')
    with TemporaryDirectory(dir=output.parent, prefix='.sap-spr-forecasts-') as temporary:
        stage = Path(temporary) / 'forecast'
        stage.mkdir()
        write_rows(stage / 'predictions.jsonl', predictions)
        write_json(stage / 'seal.json', {
            'protocol_sha256': digest(PROTOCOL), 'code_sha256': digest(Path(__file__)),
            'design_sha256': digest(assay / 'design.json'), 'predictions_sha256': digest(stage / 'predictions.jsonl'),
            'projection_seal_sha256': digest(projection / 'seal.json'), 'joint_heads_sha256': digest(heads),
            'calibration_source_sha256': PAAPE_RT_SHA256, 'calibration_positions': len(observations),
            'slopes': slopes.tolist(), 'intercept': float(intercept), 'prediction_count': len(predictions),
            'formal_participant_outcomes_used': False, 'scale': 'development log-duration',
        })
        stage.rename(output)


def sap_observations(path, targets):
    seen = set()
    values = defaultdict(list)
    for reader, item, condition, roi, duration in projected_csv(path, ('MD5', 'item', 'Type', 'ROI', 'RT')):
        condition, roi = condition.strip(), roi.strip()
        if condition not in CONDITIONS or roi not in {'0', '1', '2'} or duration == 'NA':
            continue
        point = int(item), condition, int(roi)
        duration = float(duration)
        if not math.isfinite(duration) or not 0 < duration <= 7000:
            continue
        if point not in targets or not reader or reader == 'NA' or (reader, point) in seen:
            raise ValueError('Unexpected or duplicate SAP observation')
        seen.add((reader, point))
        values[point].append(math.log(duration))
    if values.keys() != targets:
        raise ValueError('Missing human observations on at least one verified SAP position')
    return {key: math.fsum(group) / len(group) for key, group in values.items()}, len(seen)


def score(assay, forecast_dir, sap_source, output):
    import numpy as np

    if output.exists():
        raise FileExistsError('Use a fresh SAP results path')
    verified_design(assay)
    seal_path = forecast_dir / 'seal.json'
    seal = read_json(seal_path)
    check_digest(assay / 'design.json', seal['design_sha256'])
    check_digest(PROTOCOL, seal['protocol_sha256'])
    check_digest(Path(__file__), seal['code_sha256'])
    predictions_file = forecast_dir / 'predictions.jsonl'
    check_digest(predictions_file, seal['predictions_sha256'])
    targets = list(stream_rows(assay / 'sap-targets.jsonl'))
    forecasts = {(row['id'], row['stop']): row['forecast'] for row in stream_rows(predictions_file)}
    if (seal['formal_participant_outcomes_used'] is not False or seal['prediction_count'] != len(targets)
            or forecasts.keys() != {(row['id'], row['stop']) for row in targets}
            or not all(map(math.isfinite, forecasts.values()))):
        raise ValueError('Forecast seal does not cover the complete assay')
    predicted = {(row['item'], row['condition'], row['roi']): forecasts[row['id'], row['stop']] for row in targets}
    _, release = sap_files(sap_source)
    empirical, n = sap_observations(release, predicted.keys())
    rng = np.random.default_rng(1729)
    results = []
    for construction in CONSTRUCTIONS:
        for roi in REGIONS:
            items = sorted(item for item, condition, position in empirical
                           if condition == f'{construction}_AMB' and position == roi
                           and (item, construction, roi) != SAP_POSITION_EXCLUSION)
            ambiguous, control = f'{construction}_AMB', f'{construction}_UAMB'
            actual = np.array([empirical[item, ambiguous, roi] - empirical[item, control, roi] for item in items])
            model = np.array([predicted[item, ambiguous, roi] - predicted[item, control, roi] for item in items])
            interval = np.quantile(rng.choice(actual, size=(2000, len(actual)), replace=True).mean(axis=1), [.025, .975])
            mean = float(model.mean())
            direction = 'indeterminate' if interval[0] <= 0 <= interval[1] else (
                'agrees' if np.sign(mean) == np.sign(actual.mean()) else 'disagrees')
            results.append({'construction': construction, 'roi': roi, 'paired_items': len(items),
                            'human_log_duration_difference': float(actual.mean()), 'human_ci95': interval.tolist(),
                            'forecast_difference': mean, 'direction': direction})
    write_json(output, {'forecast_seal_sha256': digest(seal_path), 'receipt_sha256': digest(assay / 'design.json'),
                        'sap_response_sha256': digest(release), 'eligible_reading_times': n, 'results': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    geometry = commands.add_parser('prepare')
    for name in ('paape-source', 'sap-source', 'output'):
        geometry.add_argument(f'--{name}', type=Path, required=True)
    prediction = commands.add_parser('forecast')
    for name in ('assay', 'projection', 'heads', 'paape-source', 'output'):
        prediction.add_argument(f'--{name}', type=Path, required=True)
    evaluation = commands.add_parser('score')
    for name in ('assay', 'forecast-dir', 'sap-source', 'output'):
        evaluation.add_argument(f'--{name}', type=Path, required=True)
    arguments = vars(parser.parse_args())
    command = arguments.pop('command')
    if command == 'prepare':
        print(prepare(**arguments))
    elif command == 'forecast':
        forecast(**arguments)
    else:
        score(**arguments)


if __name__ == '__main__':
    main()
