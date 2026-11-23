"""Verify both IIRC holdouts with the exact head fitted on 3,240 paths."""

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit

from research.semantic_tracking import iirc_linked_colab_worker as worker
from research.semantic_tracking import iirc_linked_identity as identity
from research.semantic_tracking import iirc_linked_score as linked
from research.semantic_tracking.iirc_linked_extract import read_shard, shard_key
from research.semantic_tracking.iirc_linked_fit import (
    GEOMETRY_WIDTH, WIDTH, Timeline, design, expected_projection, risk_rows,
)
from research.semantic_tracking.iirc_linked_pack import METHODS, validate_split
from research.stage2.records import write_jsonl
from research.stage3.probe import _jsonl


PROTOCOL = identity.PROTOCOL.with_name('iirc-linked-shared-head.md')
OFFICIAL = {'train': 3354, 'dev': 452}
DIGEST = worker.digest
read = worker.read_json


def head_receipt(prepared, fit, extraction):
    assay_path = prepared / 'assay.json'
    assay = read(assay_path)
    seal = read(fit / 'fit-manifest.json')
    source = read(extraction)
    model = (worker.MODEL, worker.REVISION)
    if (assay['counts'] != identity.EXPECTED or assay['protocol_sha256'] != DIGEST(identity.PROTOCOL)
            or assay['code_sha256'] != DIGEST(Path(identity.__file__))
            or seal['assay_sha256'] != DIGEST(assay_path) or source['assay_sha256'] != seal['assay_sha256']
            or (seal['model'], seal['revision']) != model or (source['model'], source['revision']) != model
            or (assay['model'], assay['revision']) != model or seal['inputs_sha256'] != assay['inputs_sha256']
            or seal['fit_paths'] != identity.EXPECTED['fit_paths']
            or seal['heldout_forecast_paths'] != identity.EXPECTED['holdout_paths']
            or (seal['features_sha256'], seal['extraction_sha256']) != (source['features_sha256'], DIGEST(extraction))
            or seal['heads_sha256'] != DIGEST(fit / 'heads.npz')
            or seal['predictions_sha256'] != DIGEST(fit / 'predictions.jsonl')):
        raise ValueError('Expected the fixed matched-assay head, forecasts and causal feature receipt')
    return assay, seal, source


def official_timelines(original, assay, matched, receipt):
    if (assay['cases'] != OFFICIAL or assay['preparation_audit_sha256'] != matched['original_audit_sha256']
            or (assay['model'], assay['revision']) != (worker.MODEL, worker.REVISION)):
        raise ValueError('Expected the original IIRC preparation and frozen model')
    for name in ('dev/stimuli.jsonl', 'dev/queries.jsonl'):
        if DIGEST(original / name) != assay['inputs_sha256'][name] or (
                receipt['data_members_sha256'][name] != assay['inputs_sha256'][name]):
            raise ValueError(f'Official dev input changed: {name}')
    source_hashes = {split: assay['inputs_sha256'][f'{split}/stimuli.jsonl'] for split in ('train', 'dev')}
    if matched['original_stimuli_sha256'] != source_hashes:
        raise ValueError('The two holdouts have different source train/dev partitions')
    validate_split(original, 'dev', OFFICIAL['dev'])
    return [Timeline(row['id'], tuple(row['stops']), query['linked_start']) for row, query in zip(
        _jsonl(original / 'dev/stimuli.jsonl'), _jsonl(original / 'dev/queries.jsonl'), strict=True)]


def replay_states(timelines, checkpoints, receipt):
    stops = [(item.case, stop) for item in timelines for stop in item.stops]
    coordinates = {coordinate: index for index, coordinate in enumerate(stops)}
    if len(coordinates) != len(stops):
        raise ValueError('Duplicate held-out coordinates')
    result = np.empty((len(coordinates), WIDTH), dtype=np.float32)
    missing = set(coordinates)
    for member in receipt['shards']:
        path = checkpoints / worker.shard_name(member['number'])
        worker.check_shard(path, member)
        with np.load(path, allow_pickle=False) as arrays:
            ids, stops = arrays['ids'], arrays['stops']
        key = shard_key(receipt['plan_sha256'], member['number'], ids, stops)
        states, _, _ = read_shard(path, key, ids, stops)
        for index, coordinate in enumerate(zip(ids.tolist(), stops.tolist(), strict=True)):
            if coordinate in coordinates:
                missing.remove(coordinate)
                result[coordinates[coordinate]] = states[index]
    if missing:
        raise ValueError('The held-out readouts are missing promised stop states')
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result


def forecasts(timelines, states, heads):
    rows = risk_rows(timelines, 0)
    probabilities = {}
    with np.load(heads, allow_pickle=False) as params:
        if (set(params.files) != {*METHODS, 'projection', 'geometry_center', 'geometry_scale'}
                or not np.array_equal(params['projection'], expected_projection())):
            raise ValueError('Unexpected archived head parameters or sign projection')
        center, scale = params['geometry_center'], params['geometry_scale']
        if (center.shape != (GEOMETRY_WIDTH,) or scale.shape != center.shape or not np.isfinite(center).all()
                or not np.all(np.isfinite(scale) & (scale >= 0.1))):
            raise ValueError('Use the original fit-only geometry normalization')
        geometry = (rows.geometry - center) / (scale * np.sqrt(GEOMETRY_WIDTH))
        for index, method in enumerate(METHODS):
            weights = params[method]
            if weights.shape != (1 + GEOMETRY_WIDTH + index * WIDTH,) or not np.isfinite(weights).all():
                raise ValueError(f'Unexpected shape or value in {method} weights')
            hazards = expit(design(rows, states, geometry, method) @ weights)
            if not np.all(np.isfinite(hazards) & (hazards > 0) & (hazards < 1)):
                raise ValueError(f'{method} produced invalid hazards')
            probabilities[method] = np.split(hazards, rows.case_ends[:-1])
    for index, item in enumerate(timelines):
        yield {'case_id': item.case, 'methods': {name: probabilities[name][index].tolist() for name in METHODS}}


def check_matched_replay(replayed, archived):
    maximum_difference = 0.0
    for row, original in zip(replayed, _jsonl(archived), strict=True):
        if row['case_id'] != original['case_id'] or row['methods'].keys() != original['methods'].keys():
            raise ValueError('Identity predictions differ from replayed case order or methods')
        for name, hazards in row['methods'].items():
            before = np.asarray(original['methods'][name])
            if before.shape != (len(hazards),) or not np.isfinite(before).all():
                raise ValueError(f'Invalid archived {name} predictions')
            maximum_difference = max(maximum_difference, float(np.max(np.abs(before - hazards))))
    if maximum_difference > 1e-12:
        raise ValueError('The archived head does not reproduce the identity predictions')
    return maximum_difference


def forecast(prepared, fit, extraction, original, official_assay, job, output):
    if output.exists():
        raise FileExistsError(output)
    matched, head, source = head_receipt(prepared, fit, extraction)
    official = read(official_assay)
    job_receipt = read(job / 'manifest.json')
    directory = job / 'checkpoints'
    receipt_path = directory / 'receipt.json'
    checkpoint = read(receipt_path)
    if (job_receipt['phase'] not in ('finished', 'failed') or job_receipt.get('release_verified') is not True
            or len(checkpoint['shards']) != checkpoint['shards_total'] or checkpoint['run_id'] != job.name
            or job_receipt['run_id'] != checkpoint['run_id'] or DIGEST(receipt_path) != source['source_checkpoint_sha256']
            or job_receipt['data_members_sha256']['assay.json'] != DIGEST(official_assay)):
        raise ValueError('Both applications must reuse the complete sealed extraction from the released L4')
    worker.check_proofs(directory, checkpoint, DIGEST(official_assay), job_receipt['payload_members_sha256'],
                        job_receipt['data_members_sha256'])
    if read(directory / 'plan.json')['projection_sha256'] != sha256(expected_projection().tobytes()).hexdigest():
        raise ValueError('The L4 plan does not use the fitted sign projection')
    timelines = official_timelines(original, official, matched, job_receipt)
    original_ids = {item.case for item in timelines}
    for name in ('train/stimuli.jsonl', 'dev/stimuli.jsonl', 'dev/queries.jsonl'):
        if DIGEST(prepared / name) != matched['inputs_sha256'][name]:
            raise ValueError('Identity split stimuli changed since the head was fitted')
    for split in ('train', 'dev'):
        if any(row['id'] in original_ids for row in _jsonl(prepared / split / 'stimuli.jsonl')):
            raise ValueError('The identity fit or holdout paths overlap official dev')
    validate_split(prepared, 'dev', identity.EXPECTED['holdout_paths'])
    matched_timelines = [Timeline(row['id'], tuple(row['stops']), query['linked_start']) for row, query in zip(
        _jsonl(prepared / 'dev/stimuli.jsonl'), _jsonl(prepared / 'dev/queries.jsonl'), strict=True)]
    states = replay_states(timelines + matched_timelines, directory, checkpoint)
    end = sum(len(item.stops) for item in timelines)
    heads = fit / 'heads.npz'
    replay_error = check_matched_replay(forecasts(matched_timelines, states[end:], heads), fit / 'predictions.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.shared-head-') as temporary:
        stage = Path(temporary) / 'forecast'
        stage.mkdir()
        predictions = stage / 'predictions.jsonl'
        write_jsonl(predictions, forecasts(timelines, states[:end], heads))
        worker.write_json(stage / 'seal.json', {
            'model': head['model'], 'revision': head['revision'], 'heads_sha256': head['heads_sha256'],
            'fit_seal_sha256': DIGEST(fit / 'fit-manifest.json'), 'assay_sha256': DIGEST(official_assay),
            'predictions_sha256': DIGEST(predictions), 'source_checkpoint_sha256': DIGEST(receipt_path),
            'protocol_sha256': DIGEST(PROTOCOL), 'code_sha256': DIGEST(Path(__file__)), 'run_id': job.name,
            'matched_replay_max_abs_probability_difference': replay_error,
        })
        stage.rename(output)


def score(prepared, fit, extraction, original, official_assay, predictions, output):
    _, head, source = head_receipt(prepared, fit, extraction)
    seal_path = predictions / 'seal.json'
    seal = read(seal_path)
    expected = {'heads_sha256': head['heads_sha256'], 'fit_seal_sha256': DIGEST(fit / 'fit-manifest.json'),
                'source_checkpoint_sha256': source['source_checkpoint_sha256'], 'protocol_sha256': DIGEST(PROTOCOL),
                'code_sha256': DIGEST(Path(__file__))}
    replay_error = seal['matched_replay_max_abs_probability_difference']
    if any(seal[name] != digest for name, digest in expected.items()) or not 0 <= replay_error <= 1e-12:
        raise ValueError('The official forecasts and paired holdout use different head, protocol or extraction bytes')
    with TemporaryDirectory() as temporary:
        stage = Path(temporary)
        linked.run(official_assay, original / 'dev/stimuli.jsonl', original / 'dev/locked-labels.jsonl',
                   original / 'dev/locked-matched-swaps.jsonl', predictions / 'predictions.jsonl', seal_path,
                   stage / 'official.json')
        identity.score(prepared, fit, stage / 'identity.json')
        official, matched = read(stage / 'official.json'), read(stage / 'identity.json')
    report = {'scope': official['scope'], 'run_id': seal['run_id'], 'fit_paths': head['fit_paths'], **expected,
              'matched_replay_max_abs_probability_difference': replay_error,
              'official_forecast_seal_sha256': DIGEST(seal_path), 'official_dev': official, 'matched': matched,
              'both_pass': official['all_conditions_passed'] and matched['acceptance']}
    worker.write_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    common = ('prepared', 'fit', 'extraction', 'original', 'official_assay')
    for name, function, source in (('forecast', forecast, 'job'), ('score', score, 'predictions')):
        action = actions.add_parser(name)
        action.set_defaults(function=function)
        for field in (*common, source, 'output'):
            action.add_argument(f'--{field.replace("_", "-")}', type=Path, required=True)
    kwargs = vars(parser.parse_args())
    function = kwargs.pop('function')
    kwargs.pop('action')
    function(**kwargs)


if __name__ == '__main__':
    main()
