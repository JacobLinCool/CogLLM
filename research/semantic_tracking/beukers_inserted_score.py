"""Evaluate presealed Beukers curriculum transfer on exact shared text layouts."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from research.semantic_tracking.beukers_background import digest, read_rows


PROTOCOL = Path(__file__).with_name('beukers-inserted-protocol.md')
METHODS = ('initial_block', 'empty_context', 'text')
COHORTS = ('original', 'replication')
ARMS = ('early', 'middle', 'late')
SIDES = ('left', 'right')
MARGIN = 0.10
DRAWS = 10_000


@dataclass(frozen=True)
class Trial:
    cohort: str
    arm: str
    reader: str
    layout: tuple[str, str, str]
    values: np.ndarray


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def indexed(path):
    items = read_rows(path)
    by_id = {row['id']: row for row in items}
    if not items or len(items) != len(by_id):
        raise ValueError(f'Empty or duplicate trial IDs in {path.name}')
    return by_id


def load_trials(assay, forecast):
    audit_path = assay / 'assay.json'
    audit = read(audit_path)
    forecasts_path = forecast / 'forecasts.jsonl'
    forecast_seal_path = forecast / 'forecast-seal.json'
    forecast_seal = read(forecast_seal_path)
    truth_path = assay / 'test-truth.jsonl'
    truth_seal_path = assay / 'test-truth-seal.json'
    truth_seal = read(truth_seal_path)
    mapping = assay / 'test.jsonl'
    if (forecast_seal['assay_sha256'] != digest(audit_path) or truth_seal['assay_sha256'] != digest(audit_path)
            or truth_seal['forecast_seal_sha256'] != digest(forecast_seal_path)
            or forecast_seal['protocol_sha256'] != digest(PROTOCOL)
            or not forecast_seal['test_sha256'] == audit['file_sha256']['test.jsonl'] == digest(mapping)
            or forecast_seal['forecasts_sha256'] != digest(forecasts_path)
            or truth_seal['test_truth_sha256'] != digest(truth_path)):
        raise ValueError('The geometry, forecast, protocol and human response seals do not match')
    groups = indexed(mapping)
    forecasts = indexed(forecasts_path)
    truth = indexed(truth_path)
    if groups.keys() != forecasts.keys() or groups.keys() != truth.keys():
        raise ValueError('Every sealed reading layout needs exactly one forecast and one observed choice')
    trials, metadata = [], {}
    for key, source in groups.items():
        cohort, arm, reader, story, depth = (source[name] for name in ('cohort', 'arm', 'reader_id', 'story', 'depth'))
        layout = source['article_id'], source['left_id'], source['right_id']
        observed, expected = (truth[key][field] for field in ('response_side', 'correct_side'))
        pred = forecasts[key]['methods']
        if (cohort not in COHORTS or arm not in ARMS or not 160 <= story < 200 or depth not in (1, 2)
                or layout[1] == layout[2] or observed not in SIDES or expected not in SIDES or pred.keys() != set(METHODS)):
            raise ValueError(f'Unsupported regime, side or forecast scope for trial {key}')
        probability = np.array([pred[method] for method in METHODS], dtype=np.float64)
        if not np.all(np.isfinite(probability) & (0 < probability) & (probability < 1)):
            raise ValueError(f'Invalid choice probability for {key}')
        loss = -np.log(probability) if observed == 'right' else -np.log1p(-probability)
        correct = probability if expected == 'right' else 1 - probability
        values = np.concatenate(([float(observed == expected)], correct, loss))
        trials.append(Trial(cohort, arm, reader, layout, values))

        # Identical visible text/opponents must share the correct side and both grade-free controls.
        common = metadata.setdefault(layout, (expected, probability[1:], {}))
        regime = arm == 'early'
        if (expected != common[0] or not np.allclose(common[1], probability[1:], atol=1e-12, rtol=0)
                or not np.isclose(common[2].setdefault(regime, probability[0]), probability[0], atol=1e-12, rtol=0)):
            raise ValueError('A repeated exact layout changed its correct side or grade-free forecaster')
    return trials, {'assay_sha256': digest(audit_path), 'forecast_seal_sha256': digest(forecast_seal_path),
                    'truth_seal_sha256': digest(truth_seal_path), 'protocol_sha256': digest(PROTOCOL),
                    'scorer_sha256': digest(Path(__file__))}


def paired_sample(arms, control):
    early, other = arms['early'], arms[control]
    common = {trial.layout for trial in early} & {trial.layout for trial in other}
    if not common:
        raise ValueError(f'No exact shared layout for early and {control}')
    return {name: [trial for trial in arms[name] if trial.layout in common] for name in ('early', control)}, common


def weights_and_values(rows, reader_index, layout_index):
    count = Counter(row.layout for row in rows)
    values = np.stack([row.values for row in rows])
    frozen = np.array([1 / count[trial.layout] for trial in rows], dtype=np.float64)
    reader = np.array([reader_index[trial.arm, trial.reader] for trial in rows])
    layout = np.array([layout_index[trial.layout] for trial in rows])
    unique_layout, index = np.unique(layout, return_index=True)
    return values, frozen, reader, layout, unique_layout, values[index, 1:4]


def uncertainty(pairs):
    rng = np.random.default_rng(893681)
    # A common draw follows any early reader or layout used in both contrasts.
    rows = [row for groups in pairs.values() for arm in groups.values() for row in arm]
    reader_index = {key: i for i, key in enumerate(sorted({(row.arm, row.reader) for row in rows}))}
    layout_index = {key: i for i, key in enumerate(sorted({row.layout for row in rows}))}
    data = {control: {arm: weights_and_values(rows, reader_index, layout_index) for arm, rows in groups.items()}
            for control, groups in pairs.items()}
    point = {control: {arm: frozen @ values / frozen.sum() for arm, (values, frozen, *_) in camps.items()}
             for control, camps in data.items()}
    simulations = {control: {arm: [] for arm in groups} for control, groups in pairs.items()}
    for _ in range(DRAWS // 200):
        reader = rng.exponential(size=(200, len(reader_index)))
        layout = rng.exponential(size=(200, len(layout_index)))
        for control, camps in data.items():
            for arm, (values, frozen, readers, layouts, unique_layout, conditional) in camps.items():
                weighted = frozen * reader[:, readers] * layout[:, layouts]
                estimate = (weighted @ values) / weighted.sum(axis=1, keepdims=True)
                matched_weights = layout[:, unique_layout]
                estimate[:, 1:4] = (matched_weights @ conditional) / matched_weights.sum(axis=1, keepdims=True)
                simulations[control][arm].append(estimate)
    draws = {control: {arm: np.concatenate(blocks) for arm, blocks in group.items()}
             for control, group in simulations.items()}
    return point, draws


def interval(values, tail=0.025):
    return np.quantile(values, [tail, 1 - tail]).tolist()


def result(point, draws, groups, common, control):
    early, other = point['early'], point[control]
    early_draw, other_draw = draws['early'], draws[control]
    effect = early[:4] - other[:4]
    effect_draw = early_draw[:, :4] - other_draw[:, :4]
    nll = (early[4:] + other[4:]) / 2
    nll_draw = (early_draw[:, 4:] + other_draw[:, 4:]) / 2
    if not (np.allclose(effect[2:], 0, atol=1e-12, rtol=0)
            and np.allclose(effect_draw[:, 2:], 0, atol=1e-12, rtol=0)):
        raise AssertionError('Shared-layout weighting must cancel both grade-free controls exactly')
    names = ('human', *METHODS)
    correctness = {arm: dict(zip(names, point[arm][:4].tolist(), strict=True)) for arm in groups}
    effects = {name: {'difference': float(effect[index]), 'ci95': interval(effect_draw[:, index])}
               for index, name in enumerate(names)}
    human_difference = float(effect[1] - effect[0])
    human_ci90 = interval(effect_draw[:, 1] - effect_draw[:, 0], 0.05)
    losses = {METHODS[index]: {'difference_nats': float(nll[0] - nll[index]),
                              'ci95': interval(nll_draw[:, 0] - nll_draw[:, index])}
              for index in (1, 2)}
    coverage = {arm: {'trials': len(rows), 'readers': len({row.reader for row in rows})} for arm, rows in groups.items()}
    gates = {'readers': all(row['readers'] >= 10 for row in coverage.values()),
             'human_positive': bool(effects['human']['ci95'][0] > 0),
             'initial_block_positive': bool(effects['initial_block']['ci95'][0] > 0),
             'equivalent_within_10pp': bool(-MARGIN <= human_ci90[0] <= human_ci90[1] <= MARGIN),
             'nll_vs_empty_context': bool(losses['empty_context']['ci95'][1] < 0)}
    return {'shared_layouts': len(common), 'coverage': coverage, 'expected_correctness': correctness,
            'early_minus_control': effects, 'model_minus_human_effect': {'difference': human_difference, 'ci90': human_ci90},
            'nll_nats': dict(zip(METHODS, nll.tolist(), strict=True)), 'initial_block_minus_nll': losses,
            'gates': gates, 'passes': all(gates.values())}


def evaluate(assay, forecast):
    trials, provenance = load_trials(assay, forecast)
    studies = {}
    for cohort in COHORTS:
        arms = {arm: [trial for trial in trials if trial.cohort == cohort and trial.arm == arm] for arm in ARMS}
        if len({reader for group in arms.values() for reader in {row.reader for row in group}}) != sum(
                len({row.reader for row in arm}) for arm in arms.values()):
            raise ValueError('Reader clusters must be scoped to the source cohort and arm')
        compared = {control: paired_sample(arms, control) for control in ('middle', 'late')}
        point, draws = uncertainty({name: groups for name, (groups, _) in compared.items()})
        studies[cohort] = {name: result(point[name], draws[name], groups, common, name)
                           for name, (groups, common) in compared.items()}
    return {'provenance': provenance, 'conditions': studies,
            'passes': all(comparison['passes'] for cohort in studies.values() for comparison in cohort.values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assay', type=Path, required=True)
    parser.add_argument('--forecast', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    outcome = evaluate(args.assay, args.forecast)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(outcome, indent=2, sort_keys=True, allow_nan=False) + '\n')
