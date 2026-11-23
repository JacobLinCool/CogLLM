"""Fit and seal the linked IIRC evidence hazards without opening held-out labels."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

from research.semantic_tracking.iirc_linked_pack import METHODS, PUBLIC_MEMBERS, fit_intervals, validate_split
from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3.probe import _jsonl
from research.stage3.run import save_json


WIDTH = 128
GEOMETRY_WIDTH = 3
PROJECTION_SEED = 43
PENALTY = 0.012
MAX_ITERATIONS = 250
FUNCTION_TOLERANCE = 1e-10
GRADIENT_TOLERANCE = 1e-7
FIT_INTERVALS = 'train/intervals.jsonl'


@dataclass(frozen=True)
class Timeline:
    case: str
    stops: tuple[int, ...]
    linked_start: int


@dataclass(frozen=True)
class RiskRows:
    geometry: np.ndarray
    stop_rows: np.ndarray
    anchor_rows: np.ndarray
    case_ends: np.ndarray
    arrivals: np.ndarray


def read_inputs(root):
    assay = root / 'assay.json'
    manifest = json.loads(assay.read_text(encoding='utf-8'))
    paths = (*PUBLIC_MEMBERS, FIT_INTERVALS)
    if (manifest['schema_version'] != 1 or manifest['method_fields'] != list(METHODS)
            or set(manifest['inputs_sha256']) != set(paths)):
        raise ValueError('Unknown linked-evidence assay format')
    for name in paths:
        if digest(root / name) != manifest['inputs_sha256'][name]:
            raise ValueError(f'Fit input disagrees with the linked-evidence assay: {name}')

    counts, timelines = {}, {}
    for split in ('train', 'dev'):
        counts[split] = validate_split(root, split, manifest['cases'][split])
        if not counts[split]:
            raise ValueError(f'No linked-evidence paths in {split}')
        stimuli = _jsonl(root / split / 'stimuli.jsonl')
        queries = _jsonl(root / split / 'queries.jsonl')
        timelines[split] = [Timeline(item['id'], tuple(item['stops']), query['linked_start'])
                            for item, query in zip(stimuli, queries, strict=True)]
    if counts['train'].keys() & counts['dev'].keys():
        raise ValueError('Fit and held-out input cases overlap')
    targets = {item['case_id']: (item['lower'], item['upper'])
               for item in fit_intervals(root / FIT_INTERVALS, counts['train'])}
    return manifest, digest(assay), timelines, targets


def expected_projection():
    rng = np.random.default_rng(PROJECTION_SEED)
    signs = rng.integers(0, 2, size=(2048, WIDTH), dtype=np.int8)
    return (2 * signs.astype(np.float32) - 1) / np.sqrt(np.float32(WIDTH))


def read_features(path, timelines):
    cases = [item for split in ('train', 'dev') for item in timelines[split]]
    counts = np.array([len(item.stops) for item in cases])
    expected_ids = np.repeat([item.case for item in cases], counts)
    expected_stops = np.concatenate([item.stops for item in cases])
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'ids', 'stops', 'states', 'projection'}:
            raise ValueError('Expected a linked-evidence extraction archive')
        ids, stops = archive['ids'], archive['stops']
        if (ids.dtype.kind != 'U' or stops.dtype.kind not in 'iu'
                or not np.array_equal(ids, expected_ids) or not np.array_equal(stops, expected_stops)):
            raise ValueError('Readouts must match the sealed train/dev case and stop order')
        states, projection = archive['states'], archive['projection']
    if projection.dtype != np.float32 or not np.array_equal(projection, expected_projection()):
        raise ValueError('Readouts must use the pinned linked-evidence sign projection')
    if states.shape != (len(expected_stops), WIDTH) or states.dtype != np.float32:
        raise ValueError('Unexpected projected state shape or dtype')
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        raise ValueError('Expected a finite nonzero projected state at each promised stop')
    states /= norms
    return states, projection


def risk_rows(timelines, offset, targets=None):
    geometry, linked_rows, anchor_rows, risk_counts, arrivals = [], [], [], [], []
    for item in timelines:
        n_stops = len(item.stops) - 1
        lower, upper = (n_stops, None) if targets is None else targets[item.case]
        n_risk = upper if upper is not None else lower
        stops = np.asarray(item.stops[1:n_risk + 1], dtype=np.int64)
        geometry.append(np.column_stack((np.log1p(np.arange(1, n_risk + 1)),
                                         np.log1p(stops - item.linked_start),
                                         np.log1p(np.diff(stops, prepend=item.linked_start)))))
        linked_rows.append(offset + np.arange(1, n_risk + 1))
        anchor_rows.append(np.full(n_risk, offset))
        event = np.zeros(n_risk, dtype=bool)
        if upper is not None:
            event[-1] = True
        arrivals.append(event)
        offset += len(item.stops)
        risk_counts.append(n_risk)
    return RiskRows(np.concatenate(geometry), np.concatenate(linked_rows), np.concatenate(anchor_rows),
                    np.cumsum(risk_counts), np.concatenate(arrivals))


def design(rows, states, scaled_geometry, method):
    dimensions = 1 + GEOMETRY_WIDTH + METHODS.index(method) * WIDTH
    data = np.empty((len(rows.stop_rows), dimensions), dtype=np.float64)
    data[:, 0] = 1
    data[:, 1:1 + GEOMETRY_WIDTH] = scaled_geometry
    if method != 'geometry':
        stop_states = states[rows.stop_rows]
        data[:, 1 + GEOMETRY_WIDTH:1 + GEOMETRY_WIDTH + WIDTH] = stop_states
    if method == 'anchor_conditioned':
        interactions = states[rows.anchor_rows]
        interactions *= stop_states
        interactions *= np.sqrt(np.float32(WIDTH))
        data[:, 1 + GEOMETRY_WIDTH + WIDTH:] = interactions
    return data


def fit(data, rows):
    signed = np.where(rows.arrivals, -1.0, 1.0)
    n_cases = len(rows.case_ends)
    n_events = int(rows.arrivals.sum())
    if not 0 < n_events < len(signed):
        raise ValueError('Training requires both a positive arrival and a survival opportunity')
    initial = np.zeros(data.shape[1])
    initial[0] = np.log(n_events / (len(signed) - n_events))

    def objective(weights):
        # (L, L+1] multiplies the previous survival factors by one hazard.
        # A right-censored path contributes survival factors through its cutoff.
        margins = signed * (data @ weights)
        nll = np.logaddexp(0.0, margins).sum() / n_cases
        loss = nll + 0.5 * PENALTY * (weights[1:] @ weights[1:])
        grad = (data.T @ (signed * expit(margins))) / n_cases
        grad[1:] += PENALTY * weights[1:]
        return loss, grad

    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      options={'maxiter': MAX_ITERATIONS, 'ftol': FUNCTION_TOLERANCE,
                               'gtol': GRADIENT_TOLERANCE})
    if not result.success or not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f'Linked-evidence logistic fit failed: {result.message}')
    return result.x, {'iterations': int(result.nit), 'penalized_objective': float(result.fun)}


def run(inputs, features, extraction, output):
    if output.exists():
        raise FileExistsError(output)
    manifest, assay_hash, timelines, targets = read_inputs(inputs)
    feature_hash = digest(features)
    receipt = json.loads(extraction.read_text(encoding='utf-8'))
    expected_extraction_inputs = {'assay': assay_hash}
    for name in ('train/stimuli.jsonl', 'dev/stimuli.jsonl'):
        expected_extraction_inputs[name] = manifest['inputs_sha256'][name]
    if (receipt['features_sha256'] != feature_hash or receipt['assay_sha256'] != assay_hash
            or receipt['model'] != manifest['model'] or receipt['revision'] != manifest['revision']
            or receipt['inputs_sha256'] != expected_extraction_inputs):
        raise ValueError('Extraction receipt disagrees with the sealed assay, model, or feature archive')
    states, projection = read_features(features, timelines)
    train = risk_rows(timelines['train'], 0, targets)
    dev = risk_rows(timelines['dev'], sum(len(item.stops) for item in timelines['train']))
    center = train.geometry.mean(axis=0)
    scale = np.maximum(train.geometry.std(axis=0), 0.1)
    denominator = scale * np.sqrt(GEOMETRY_WIDTH)
    train_geometry = (train.geometry - center) / denominator
    dev_geometry = (dev.geometry - center) / denominator

    weights, fitting, forecasts = {}, {}, {}
    for method in METHODS:
        train_data = design(train, states, train_geometry, method)
        weights[method], fitting[method] = fit(train_data, train)
        del train_data
        hazards = expit(design(dev, states, dev_geometry, method) @ weights[method])
        if not np.all(np.isfinite(hazards) & (0 < hazards) & (hazards < 1)):
            raise RuntimeError(f'{method} produced hazards outside the strict probability range')
        forecasts[method] = np.split(hazards, dev.case_ends[:-1])

    output.mkdir(parents=True, exist_ok=False)
    heads = output / 'heads.npz'
    np.savez_compressed(heads, **weights, projection=projection, geometry_center=center, geometry_scale=scale)
    predictions = output / 'predictions.jsonl'
    write_jsonl(predictions, ({'case_id': item.case, 'methods': {method: forecasts[method][index].tolist()
                                                             for method in METHODS}}
                              for index, item in enumerate(timelines['dev'])))
    save_json(output / 'fit-manifest.json', {
        'schema_version': 1, 'model': manifest['model'], 'revision': manifest['revision'],
        'assay_sha256': assay_hash, 'features_sha256': feature_hash, 'extraction_sha256': digest(extraction),
        'heads_sha256': digest(heads), 'predictions_sha256': digest(predictions),
        'inputs_sha256': manifest['inputs_sha256'], 'fit_paths': len(train.case_ends),
        'heldout_forecast_paths': len(dev.case_ends), 'fit_opportunity_rows': len(train.arrivals),
        'fit_arrivals': int(train.arrivals.sum()),
        'fit_right_censored': len(train.case_ends) - int(train.arrivals.sum()),
        'estimator': 'Discrete conditional arrival hazards, fitted with case-mean interval NLL and L2.',
        'geometry': ['log1p(linked stop number)', 'log1p(visible linked characters)',
                     'log1p(newly visible linked characters)'],
        'geometry_scaling': 'Center and standard deviation over fit risk rows, with 0.1 floor; divide by sqrt(3).',
        'path_text': 'Unit stop state of the joined seed and linked text, appended to geometry.',
        'anchor_conditioned': 'Add the elementwise unit anchor-state and stop-state product times sqrt(128).',
        'penalty': PENALTY, 'intercept_penalized': False, 'fit_dtype': 'float64',
        'optimizer': {'name': 'L-BFGS-B', 'max_iterations': MAX_ITERATIONS,
                      'function_tolerance': FUNCTION_TOLERANCE, 'gradient_tolerance': GRADIENT_TOLERANCE},
        'numpy': np.__version__, 'scipy': scipy.__version__, 'fits': fitting,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--extraction', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(**vars(parser.parse_args()))
