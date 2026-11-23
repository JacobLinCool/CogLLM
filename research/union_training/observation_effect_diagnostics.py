"""Replay registered development contrasts from observation-model predictions."""

import argparse
import math
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json


def verified_packet(packet):
    receipt = read_json(packet / 'receipt.json')
    for name, expected in receipt['packet_sha256'].items():
        if digest(packet / name) != expected:
            raise ValueError(f'Registered diagnostic input changed: {name}')
    entries = read_json(packet / 'index.json')['tasks']
    index = {entry['name']: entry for entry in entries}
    if len(entries) != len(index):
        raise ValueError('Repeated task name in diagnostic inputs')
    return index, receipt


def statistic(values, task, effect):
    if task['kind'] == 'gaussian' and effect['metric'] == 'mean':
        return float(values[1])
    probabilities = np.asarray(values, dtype=np.float64) / sum(values)
    return categorical_statistic(probabilities, effect)


def categorical_statistic(probabilities, effect):
    if effect['metric'] == 'category':
        category = effect['category']
        if type(category) is not int or not 0 <= category < len(probabilities):
            raise ValueError('Invalid category for a paired contrast')
        return float(probabilities[category])
    if effect['metric'] == 'mean':
        # Use the original contrast's zero-based scores. Cross-condition coefficients sum to zero.
        return float(np.dot(probabilities, np.arange(len(probabilities))))
    raise ValueError(f'Unsupported registered contrast metric: {effect["metric"]}')


class Development:
    def __init__(self, packet, index, predictions):
        self.packet = packet
        self.index = index
        self.predictions = predictions
        self.loaded = {}
        self.points = {}
        self.trajectories = None

    def load(self, task, field):
        cache_key = task, field
        if cache_key not in self.loaded:
            path = self.packet / self.index[task]['file']
            rows = [row for row in stream_rows(path) if row['role'] == 1]
            indices = np.asarray([i for i, row in enumerate(stream_rows(path)) if row['role'] == 1])
            prefix = f'{task}.development.'
            selected, forecast = self.predictions[prefix + 'rows'], self.predictions[prefix + field]
            if (selected.dtype.kind not in 'iu' or not np.array_equal(selected, indices)
                    or forecast.ndim not in {1, 2} or forecast.shape[0] != len(rows)
                    or not np.isfinite(forecast).all()):
                raise ValueError(f'Missing or inconsistent development forecasts for {task}')
            if field == 'probabilities' and (
                forecast.ndim != 2 or np.any(forecast < 0) or np.any(forecast > 1)
                or not np.allclose(forecast.sum(axis=1), 1, atol=1e-6, rtol=1e-6)
            ):
                raise ValueError(f'Invalid forecast distribution for {task}')
            self.loaded[cache_key] = list(zip(rows, forecast, strict=True))
        return self.loaded[cache_key]

    def point(self, name, field, coordinate):
        key = name, field
        if key not in self.points:
            points = {tuple(row['coordinates'][0]): (row, value) for row, value in self.load(name, field)}
            if len(points) != len(self.loaded[key]):
                raise ValueError(f'Multiple development forecasts for one point in {name}')
            self.points[key] = points
        return self.points[key][coordinate]

    def first_event(self, coordinate):
        if self.trajectories is None:
            self.trajectories = {}
            for row, probabilities in self.load('frick_first_report', 'probabilities'):
                coordinates = row['coordinates']
                counts = row['outcome']
                if (len(counts) != len(coordinates) + 1
                        or len(probabilities) < len(counts)
                        or np.any(probabilities[len(coordinates):-1] != 0)):
                    raise ValueError('Frick trajectory does not cover its first-report distribution')
                count, prior = sum(counts), 0
                earlier = 0.0
                for point, observed, probability in zip(coordinates, counts[:-1], probabilities[:len(coordinates)],
                                                        strict=True):
                    key = tuple(point)
                    if key in self.trajectories:
                        raise ValueError('Repeated development coordinate for Frick first-report paths')
                    self.trajectories[key] = observed, prior, count, float(probability), earlier
                    prior += observed
                    earlier += float(probability)
        return self.trajectories[coordinate]

    def measurement(self, task, coordinate, effect):
        name = task['name']
        if name in {'onestop_answer_accuracy', 'race_answer_accuracy'}:
            choice = name.replace('_accuracy', '_choice')
            row, correct = self.point(choice, 'correct_probability', coordinate)
            observed = row['outcome'][row['correct_index']] / sum(row['outcome'])
            if effect['metric'] != 'category' or effect['category'] not in (0, 1) or not 0 <= correct <= 1:
                raise ValueError(f'Invalid correct-answer forecast or contrast for {name}')
            return (observed, float(correct)) if effect['category'] == 1 else (1 - observed, 1 - float(correct))
        if name == 'ambistory_given_sense_plausibility':
            row, probabilities = self.point('ambistory_joint', 'probabilities', coordinate)
            labels = self.index['ambistory_joint']['labels']
            old_labels = task['labels']
            if labels != [f'{rating}|{flag}' for rating in old_labels for flag in (0, 1)]:
                raise ValueError('AmbiStory joint outcome changed its rating order')
            counts = np.asarray(row['outcome']).reshape((-1, 2)).sum(axis=1)
            rating = probabilities.reshape((-1, 2)).sum(axis=1)
            return statistic(counts, task, effect), categorical_statistic(rating, effect)
        if name in {'frick_target_first_report', 'frick_sentence_inconsistent'}:
            if effect['metric'] != 'category' or effect['category'] != 1:
                raise ValueError(f'Unexpected Frick contrast estimand for {name}')
            first, prior, count, probability, earlier = self.first_event(coordinate)
            if name == 'frick_target_first_report':
                return first / count, probability
            after, after_probability = self.point('frick_after_first_report', 'probabilities', coordinate)
            if prior != sum(after['outcome']) or not 0 <= after_probability[1] <= 1:
                raise ValueError('Frick subsequent-report counts and documented risk set differ')
            return (first + after['outcome'][1]) / count, probability + earlier * float(after_probability[1])
        if name not in self.index:
            return None
        entry = self.index[name]
        if entry['labels'] != task['labels'] or entry['source'] != task['source']:
            raise ValueError(f'Changed measurement labels or source for {name}')
        if task['kind'] == 'gaussian':
            row, forecast = self.point(name, 'mean', coordinate)
            return statistic(row['outcome'], task, effect), float(forecast)
        row, forecast = self.point(name, 'probabilities', coordinate)
        return statistic(row['outcome'], task, effect), categorical_statistic(forecast, effect)


def replay(base, packet, predictions, output):
    old_index, old_receipt = verified_packet(base)
    new_index, new_receipt = verified_packet(packet)
    old_digest = digest(base / 'receipt.json')
    if (new_receipt['base_receipt_sha256'] != old_digest
            or new_receipt['formal_holdout_sha256'] != old_receipt['formal_holdout_sha256']):
        raise ValueError('The development packets do not share the same registered split')

    effects = []
    with np.load(predictions, allow_pickle=False) as forecasts:
        development = Development(packet, new_index, forecasts)
        for task in old_index.values():
            if not task['dev_effects']:
                continue
            rows = list(stream_rows(base / task['file']))
            for effect in task['dev_effects']:
                positions, coefficients = effect['indices'], effect['coefficients']
                if (not positions or len(set(positions)) != len(positions) or len(positions) != len(coefficients)
                        or not all(type(index) is int and 0 <= index < len(rows) for index in positions)
                        or not all(math.isfinite(value) for value in coefficients)
                        or not math.isclose(sum(coefficients), 0.0, abs_tol=1e-10)
                        or not math.isfinite(effect['scale']) or effect['scale'] <= 0
                        or type(effect['matched_pairs']) is not int or effect['matched_pairs'] < 1):
                    raise ValueError(f'Invalid registered contrast geometry for {task["name"]}')
                matched = [rows[index] for index in positions]
                if any(row['split'] != 1 for row in matched):
                    raise ValueError('A diagnostic crossed the development material split')
                observed = np.dot(coefficients, [statistic(row['outcome'], task, effect) for row in matched])
                if not math.isclose(observed, effect['target'], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError('The published development contrast did not replay from its source rows')
                result = {
                    'task': task['name'], 'label': effect['label'], 'matched_pairs': effect['matched_pairs'],
                    'coordinate_count': len(positions), 'human_difference': float(observed),
                    'scale': effect['scale'],
                }
                values = [development.measurement(task, (row['doc'], row['stop']), effect) for row in matched]
                if any(value is None for value in values):
                    result.update(status='unsupported', reason='No compatible development measurement')
                else:
                    replayed, model = coefficients @ np.asarray(values, dtype=np.float64)
                    if not math.isclose(observed, replayed, rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError(f'Converted observations changed the contrast for {task["name"]}')
                    error = float(model - observed)
                    result.update(status='replayed', model_difference=float(model), error=error,
                                  standardized_absolute_error=abs(error) / effect['scale'])
                effects.append(result)
    scores = [row for row in effects if row['status'] == 'replayed']
    report = {
        'schema_version': 1, 'purpose': 'post-training development diagnostics',
        'base_receipt_sha256': old_digest, 'packet_receipt_sha256': digest(packet / 'receipt.json'),
        'predictions_sha256': digest(predictions),
        'registered_contrasts': len(effects), 'replayed_contrasts': len(scores),
        'unsupported_contrasts': len(effects) - len(scores),
        'mean_standardized_absolute_error': (
            float(np.mean([row['standardized_absolute_error'] for row in scores])) if scores else None
        ),
        'effects': effects,
    }
    write_json(output, report)
    print(f'Development contrasts replayed: {len(scores)}/{len(effects)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'packet', 'predictions', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    replay(**vars(parser.parse_args()))
