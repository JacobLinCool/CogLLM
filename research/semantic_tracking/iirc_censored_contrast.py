"""Diagnose IIRC completion ranking with strictly censored training contrasts."""

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
import json
from pathlib import Path
import re
import resource
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.core_features.readback_score import paired_material_interval
from research.semantic_tracking.iirc_linked_fit import (
    FUNCTION_TOLERANCE, GEOMETRY_WIDTH, GRADIENT_TOLERANCE, MAX_ITERATIONS,
    PENALTY, WIDTH, design, expected_projection, read_inputs, risk_rows,
)
from research.semantic_tracking.iirc_prepare import digest
from research.semantic_tracking.joint_score import interval_nll, probability_logits
from research.stage2.records import write_jsonl
from research.stage3.probe import _jsonl
from research.stage3.run import save_json


PROTOCOL = Path(__file__).resolve().parents[1] / 'iirc-censored-contrast.md'
TRAIN_RELEASE_SHA256 = '6dbbcc64306d9410a0f75b9f4613c2ad37c9032f4ec36811bb7d2f8eeec1f5b3'
TRAIN_CASE = re.compile(r'^\{"case_id": ("[^"]+"),')
METHODS = ('locked', 'no_rank', 'censored_rank')
RANK_WEIGHT = 0.5
EXPECTED = {'training_pairs': 15, 'training_groups': 13, 'evaluation_pairs': 37, 'evaluation_groups': 33}


@dataclass(frozen=True)
class Contrast:
    early: str
    late: str
    cutoff: int
    group: str


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fit_source_rows(path, fit_targets):
    if digest(path) != TRAIN_RELEASE_SHA256:
        raise ValueError('Use the pinned original train labels for pairing')
    selected = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            match = TRAIN_CASE.match(line)
            if match is None:
                raise ValueError('Expected first-field case IDs in the prepared train JSONL')
            case = json.loads(match[1])
            if case not in fit_targets:
                continue
            row = json.loads(line)
            if case in selected or (row['lower'], row['upper']) != fit_targets[case]:
                raise ValueError('Selected original train row disagrees with the split')
            selected[case] = row
    if selected.keys() != fit_targets.keys():
        raise ValueError('Original train rows omit planned fit paths')
    return selected


def prepare(prepared, source_labels, output):
    _, assay_sha, timelines, targets = read_inputs(prepared)
    labels = fit_source_rows(source_labels, targets)
    body = {row['id']: row['text'] for row in _jsonl(prepared / 'train/stimuli.jsonl')}
    schedules = {item.case: tuple(stop - item.linked_start for stop in item.stops[1:])
                 for item in timelines['train']}
    families = defaultdict(list)
    for item in timelines['train']:
        row = labels[item.case]
        signature = (row['linked_text_sha256'], body[item.case][item.linked_start:], schedules[item.case])
        families[signature].append(row)
    pairs = []
    for rows in families.values():
        for a, b in combinations(rows, 2):
            if (a['upper'] is None) == (b['upper'] is None):
                continue
            early, late = (a, b) if a['upper'] is not None else (b, a)
            if (early['seed_family'] == late['seed_family'] or early['upper'] > late['lower']
                    or early['evidence_end'] > late['evidence_start']):
                continue
            if early['article_cluster'] != late['article_cluster']:
                raise ValueError('Identical linked documents must occur in the same article component')
            pairs.append(Contrast(early['case_id'], late['case_id'], early['upper'], early['article_cluster']))
    pairs.sort(key=lambda row: (row.group, row.early, row.late))
    if len(pairs) != EXPECTED['training_pairs'] or len({pair.group for pair in pairs}) != EXPECTED['training_groups']:
        raise ValueError('Training contrast inventory differs from the protocol')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {'assay_sha256': assay_sha, 'source_labels_sha256': TRAIN_RELEASE_SHA256,
                       'code_sha256': digest(Path(__file__)), 'protocol_sha256': digest(PROTOCOL),
                       'pairs': [vars(pair) for pair in pairs]})


def cutoffs(paths, rows, targets, pairs):
    starts = dict(zip((path.case for path in paths), (0, *rows.case_ends[:-1]), strict=True))
    result = []
    for pair in pairs:
        early, late, cutoff = pair.early, pair.late, pair.cutoff
        if targets[early] != (cutoff - 1, cutoff) or targets[late][1] is not None or targets[late][0] < cutoff:
            raise ValueError('Contrast cutoff is outside the two observed risk sets')
        result.append((slice(starts[early], starts[early] + cutoff), slice(starts[late], starts[late] + cutoff)))
    return result


def rank_loss(logits, slices, gradient=None, multiplier=0.0):
    nll = 0.0
    for early, late in slices:
        totals = [np.logaddexp(0, logits[segment]).sum() for segment in (early, late)]
        odds = [total + np.log(-np.expm1(-total)) for total in totals]
        margin = odds[1] - odds[0]
        nll += np.logaddexp(0, margin)
        if gradient is not None:
            scale = multiplier * expit(margin) / len(slices)
            for segment, total, direction in zip((early, late), totals, (-1, 1), strict=True):
                gradient[segment] += direction * scale * expit(logits[segment]) / -np.expm1(-total)
    return float(nll / len(slices))


def fit_head(matrix, risk, pairs, rank_weight):
    events = risk.arrivals.sum()
    cases = len(risk.case_ends)
    signed = np.where(risk.arrivals, -1.0, 1.0)
    initial = np.zeros(matrix.shape[1], dtype=np.float64)
    initial[0] = np.log(events / (len(signed) - events))

    def objective(weights):
        logits = matrix @ weights
        margins = signed * logits
        intervals = np.logaddexp(0, margins).sum() / cases
        residual = signed * expit(margins) / cases
        ranking = rank_loss(logits, pairs, residual, rank_weight) if rank_weight else 0.0
        gradient = matrix.T @ residual
        gradient[1:] += PENALTY * weights[1:]
        total = intervals + rank_weight * ranking + 0.5 * PENALTY * (weights[1:] @ weights[1:])
        return float(total), gradient

    optimum = minimize(objective, initial, jac=True, method='L-BFGS-B',
                       options={'maxiter': MAX_ITERATIONS, 'ftol': FUNCTION_TOLERANCE,
                                'gtol': GRADIENT_TOLERANCE})
    if not optimum.success or not np.isfinite(optimum.x).all() or not np.isfinite(optimum.fun):
        raise ValueError(f'Scipy fit failed: {optimum.message}')
    logits = matrix @ optimum.x
    return optimum.x, {'iterations': int(optimum.nit), 'objective': float(optimum.fun),
                       'train_interval_nll': float(np.logaddexp(0, signed * logits).sum() / cases),
                       'train_pair_rank_nll': rank_loss(logits, pairs)}


def read_risk_states(features, timelines, train, dev):
    indices = np.unique(np.r_[train.stop_rows, train.anchor_rows, dev.stop_rows, dev.anchor_rows])
    paths = [item for split in ('train', 'dev') for item in timelines[split]]
    lengths = [len(path.stops) for path in paths]
    expected_rows = sum(lengths)
    with np.load(features, allow_pickle=False) as package:
        if (set(package.files) != {'ids', 'stops', 'states', 'projection'}
                or not np.array_equal(package['ids'], np.repeat([item.case for item in paths], lengths))
                or not np.array_equal(package['stops'], np.concatenate([item.stops for item in paths]))
                or not np.array_equal(package['projection'], expected_projection())):
            raise ValueError('Cache does not match the frozen causal paths and projection')
    kept = np.empty((len(indices), WIDTH), dtype=np.float32)
    with ZipFile(features) as archive, archive.open('states.npy') as stream:
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ValueError('Expected the frozen NPY header format')
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        if shape != (expected_rows, WIDTH) or fortran or dtype != np.dtype('<f4'):
            raise ValueError('Unexpected cached state dimensions, layout, or dtype')
        for start in range(0, expected_rows, 4096):
            end = min(start + 4096, expected_rows)
            width_bytes = WIDTH * dtype.itemsize
            contents = stream.read((end - start) * width_bytes)
            if len(contents) != (end - start) * width_bytes:
                raise ValueError('Truncated cached state vectors')
            block = np.frombuffer(contents, dtype=dtype).reshape(-1, WIDTH)
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            if not np.all(np.isfinite(norms) & (norms > 0)):
                raise ValueError('Nonfinite or zero-norm frozen state')
            left, right = np.searchsorted(indices, [start, end])
            local = indices[left:right] - start
            kept[left:right] = block[local] / norms[local]
        if stream.read(1):
            raise ValueError('Unexpected trailing state data')

    def remap(rows):
        return replace(rows, stop_rows=np.searchsorted(indices, rows.stop_rows),
                       anchor_rows=np.searchsorted(indices, rows.anchor_rows))

    return kept, remap(train), remap(dev)


def forecast(prepared, features, extraction, training_pairs, locked, output):
    started = perf_counter()
    if output.exists():
        raise FileExistsError(output)
    manifest, assay, timelines, targets = read_inputs(prepared)
    extract = read_json(extraction)
    training = read_json(training_pairs)
    published = read_json(locked / 'fit-manifest.json')
    if (extract['assay_sha256'] != assay or extract['features_sha256'] != digest(features)
            or (extract['model'], extract['revision']) != (manifest['model'], manifest['revision'])
            or published['features_sha256'] != extract['features_sha256'] or published['assay_sha256'] != assay
            or published['heads_sha256'] != digest(locked / 'heads.npz')
            or published['predictions_sha256'] != digest(locked / 'predictions.jsonl')
            or published['penalty'] != PENALTY
            or training['assay_sha256'] != assay or training['source_labels_sha256'] != TRAIN_RELEASE_SHA256
            or training['code_sha256'] != digest(Path(__file__)) or training['protocol_sha256'] != digest(PROTOCOL)):
        raise ValueError('Frozen inputs or the published reference differ from the experiment')
    pairs = [Contrast(**row) for row in training['pairs']]
    if len(pairs) != EXPECTED['training_pairs'] or len({pair.group for pair in pairs}) != EXPECTED['training_groups']:
        raise ValueError('Unexpected sealed training contrasts')
    train = risk_rows(timelines['train'], 0, targets)
    dev = risk_rows(timelines['dev'], sum(len(path.stops) for path in timelines['train']))
    states, train, dev = read_risk_states(features, timelines, train, dev)
    pair_rows = cutoffs(timelines['train'], train, targets, pairs)
    center = train.geometry.mean(axis=0)
    scale = np.maximum(train.geometry.std(axis=0), 0.1)
    denominator = scale * np.sqrt(GEOMETRY_WIDTH)
    matrix = design(train, states, (train.geometry - center) / denominator, 'anchor_conditioned')
    with np.load(locked / 'heads.npz', allow_pickle=False) as reference:
        if (not np.array_equal(center, reference['geometry_center'])
                or not np.array_equal(scale, reference['geometry_scale'])):
            raise ValueError('Published geometry preprocessing changed')
        weights = {'locked': reference['anchor_conditioned']}
    diagnostics = {}
    for name, weight in (('no_rank', 0.0), ('censored_rank', RANK_WEIGHT)):
        weights[name], diagnostics[name] = fit_head(matrix, train, pair_rows, weight)
    diagnostics['locked'] = {'train_pair_rank_nll': rank_loss(matrix @ weights['locked'], pair_rows)}
    del matrix
    matrix = design(dev, states, (dev.geometry - center) / denominator, 'anchor_conditioned')
    hazards = {name: np.split(expit(matrix @ coefficients), dev.case_ends[:-1])
               for name, coefficients in weights.items()}
    references = {row['case_id']: row['methods']['anchor_conditioned']
                  for row in _jsonl(locked / 'predictions.jsonl')}
    replay_error = max(float(np.max(np.abs(references[path.case] - fitted)))
                       for path, fitted in zip(timelines['dev'], hazards['locked'], strict=True))
    if replay_error > 1e-12:
        raise ValueError('Published predictions differ from direct replay')
    no_rank_error = max(float(np.max(np.abs(p - q)))
                        for p, q in zip(hazards['locked'], hazards['no_rank'], strict=True))
    if no_rank_error > 1e-5:
        raise ValueError('Zero-weight refit differs substantially from the published reference')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.censored-forecast-') as tmp:
        stage = Path(tmp) / 'forecast'
        stage.mkdir()
        prediction = stage / 'predictions.jsonl'
        learned = stage / 'heads.npz'
        np.savez_compressed(learned, **weights, geometry_center=center, geometry_scale=scale)
        write_jsonl(prediction, ({'case_id': path.case, 'methods': {name: hazards[name][i].tolist() for name in METHODS}}
                                 for i, path in enumerate(timelines['dev'])))
        save_json(stage / 'seal.json', {
            'assay_sha256': assay, 'features_sha256': extract['features_sha256'], 'training_pairs_sha256': digest(training_pairs),
            'protocol_sha256': digest(PROTOCOL), 'code_sha256': digest(Path(__file__)), 'heads_sha256': digest(learned),
            'predictions_sha256': digest(prediction), 'reference_sha256': digest(locked / 'fit-manifest.json'),
            'fit_paths': len(timelines['train']), 'forecast_paths': len(timelines['dev']), 'rank_weight': RANK_WEIGHT,
            'ridge_penalty': PENALTY, 'diagnostics': diagnostics, 'published_replay_max_hazard_error': replay_error,
            'published_vs_zero_rank_max_hazard_error': no_rank_error, 'elapsed_seconds': perf_counter() - started,
            'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == 'darwin' else 2**10),
        })
        stage.rename(output)


def score(prepared, forecasts, output):
    if output.exists():
        raise FileExistsError(output)
    manifest, assay, timelines, _ = read_inputs(prepared)
    seal_path = forecasts / 'seal.json'
    seal = read_json(seal_path)
    prediction = forecasts / 'predictions.jsonl'
    if (seal['assay_sha256'] != assay or seal['protocol_sha256'] != digest(PROTOCOL)
            or seal['code_sha256'] != digest(Path(__file__)) or seal['predictions_sha256'] != digest(prediction)
            or seal['heads_sha256'] != digest(forecasts / 'heads.npz')):
        raise ValueError('Forecasts must be sealed before reading held-out outcomes')
    for name, revision in manifest['held_out_not_in_archive'].items():
        if digest(prepared / name) != revision:
            raise ValueError('Published holdout material changed')
    labels = {row['case_id']: row for row in _jsonl(prepared / 'dev/locked-labels.jsonl')}
    pairs = list(_jsonl(prepared / 'dev/locked-matched-swaps.jsonl'))
    records = {row['case_id']: row['methods'] for row in _jsonl(prediction)}
    stops = {row.case: len(row.stops) - 1 for row in timelines['dev']}
    if (labels.keys() != stops.keys() or records.keys() != stops.keys() or len(pairs) != EXPECTED['evaluation_pairs']
            or len({pair['article_cluster'] for pair in pairs}) != EXPECTED['evaluation_groups']
            or any(row.keys() != set(METHODS) for row in records.values())):
        raise ValueError('Unexpected held-out pair inventory')
    groups = [{'group_id': pair['article_cluster']} for pair in pairs]
    summaries, values = {}, {}
    for method in METHODS:
        logits = {case: probability_logits(records[case][method], (count,)) for case, count in stops.items()}
        margins, original_scores, wins = [], [], []
        for pair in pairs:
            left, right = sorted(pair['case_ids'], key=lambda case: labels[case]['upper'])
            earlier = labels[left]['upper']
            h0, h1 = [np.logaddexp(0.0, logits[case][:earlier]).sum() for case in (left, right)]
            wins.append(1.0 if h0 > h1 else 0.5 if h0 == h1 else 0.0)
            own, swaps = [], []
            for case, other in ((left, right), (right, left)):
                own.append(interval_nll(logits[case], labels[case]['lower'], labels[case]['upper']))
                swaps.append(interval_nll(logits[case], labels[other]['lower'], labels[other]['upper']))
            original_scores.append(own)
            margins.append(np.subtract(swaps, own))
        values[method] = {'same_stop_accuracy': np.asarray(wins), 'swapped_minus_own_nll_nats': np.mean(margins, axis=1),
                          'own_nll_nats': np.mean(original_scores, axis=1)}
        summaries[method] = {name: paired_material_interval(value, groups) for name, value in values[method].items()}
        summaries[method]['both_paths_favor_own_interval'] = int(np.count_nonzero(np.all(np.asarray(margins) > 0, axis=1)))
        summaries[method]['same_stop_correct'] = float(sum(wins))
    differences = {name: paired_material_interval(values['censored_rank'][name] - values['no_rank'][name], groups)
                   for name in values['no_rank']}
    save_json(output, {'retrospective': True, 'counts': EXPECTED, 'fit_seal_sha256': digest(seal_path),
                       'fit_seal': {key: seal[key] for key in (
                           'features_sha256', 'predictions_sha256', 'diagnostics', 'published_replay_max_hazard_error',
                           'published_vs_zero_rank_max_hazard_error', 'elapsed_seconds', 'peak_rss_mib')},
                       'model': manifest['model'], 'revision': manifest['revision'],
                       'holdout': summaries, 'censored_rank_minus_no_rank': differences})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for name, function, options in (
        ('prepare', prepare, ('prepared', 'source_labels', 'output')),
        ('forecast', forecast, ('prepared', 'features', 'extraction', 'training_pairs', 'locked', 'output')),
        ('score', score, ('prepared', 'forecasts', 'output')),
    ):
        command = actions.add_parser(name)
        command.set_defaults(function=function)
        for option in options:
            command.add_argument(f'--{option.replace("_", "-")}', type=Path, required=True)
    args = vars(parser.parse_args())
    function = args.pop('function')
    args.pop('action')
    function(**args)


if __name__ == '__main__':
    main()
