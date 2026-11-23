"""Fit and score the frozen Qwen pilot on separate human observation channels."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.semantic_tracking import load_curve_control_score as control_score
from research.semantic_tracking.load_curve_colab import PROTOCOL
from research.semantic_tracking.load_curve_readout import TaskData, fit, per_observation_nll, predict_logits


SEEDS = (21, 47, 101)
MODES = {
    'independent': (('issue', 'evidence', 'sense'), False),
    'joint': (('issue', 'evidence', 'sense'), True),
    'joint_semantic': (('issue', 'evidence'), True),
}
TASK_NAMES = {'scarecrow_contradiction_coverage': 'issue', 'blott_sentence_sense': 'sense'}
OPEN_EXCLUSION_TOKENS = 5


@dataclass(frozen=True)
class TokenDocument:
    text: str
    ids: tuple
    offsets: np.ndarray
    states: np.ndarray

    @property
    def ends(self):
        return self.offsets[:, 1]

    def exact_stop(self, end):
        position = int(np.searchsorted(self.ends, end, side='right')) - 1
        return position if position >= 0 and self.ends[position] == end else None


def projection(job, protocol=PROTOCOL):
    manifest = read_json(job / 'manifest.json')
    archive = job / 'extraction'
    seal = read_json(archive / 'seal.json')
    assertions = (
        manifest['phase'] == 'finished', manifest['release_verified'], manifest['verified_outputs'],
        seal['gpu'] == 'NVIDIA L4', seal['candidate'] == 'qwen', seal['content_forward_per_document'] == 1,
        seal['revision'] == manifest['revision'], seal['protocol_sha256'] == digest(protocol),
        seal['stimuli_sha256'] == digest(job / 'stimuli.jsonl'),
    )
    if not all(assertions):
        raise ValueError('An owned, completed, released L4 extraction with the locked protocol is required')
    if any(digest(archive / name) != expect for name, expect in seal['file_sha256'].items()):
        raise ValueError('The dense projection or coordinate archive changed')
    source = {row['id']: row['text'] for row in stream_rows(job / 'stimuli.jsonl')}
    states = np.load(archive / 'states.npy', mmap_mode='r', allow_pickle=False)
    if list(states.shape) != seal['shape'] or states.dtype != np.float32:
        raise ValueError('The dense projection dimensions differ from the seal')
    documents, current = {}, 0
    for row in stream_rows(archive / 'coordinates.jsonl'):
        key = row['id']
        cuts = tuple(row['token_ids'])
        stop = current + len(cuts)
        if key in documents or len(cuts) != len(row['offsets']):
            raise ValueError('Unexpected duplicate or incomplete token coordinate')
        documents[key] = TokenDocument(source[key], cuts, np.asarray(row['offsets']), states[current:stop])
        current = stop
    if current != len(states) or documents.keys() != source.keys():
        raise ValueError('The native token axis and corpus do not describe the same documents')
    return documents, seal


def read_checked(root, receipt_name, key, name):
    receipt = read_json(root / receipt_name)
    path = root / name
    if digest(path) != receipt[key][name]:
        raise ValueError(f'Unexpected human label hash for {name}')
    return list(stream_rows(path))


def pair_features(anchor, tokens, distance):
    scale = np.linalg.norm(anchor, axis=-1) * np.linalg.norm(tokens, axis=-1)
    cosine = (anchor * tokens).sum(axis=-1) / np.maximum(scale, 1e-8)
    age = np.log1p(np.asarray(distance)) / 5
    return np.column_stack((cosine, age))


def observation_sets(documents, event_rows, layouts, evidence_rows, split):
    binary = defaultdict(list)
    misses = Counter()
    for row in event_rows:
        if row['split'] != split:
            continue
        key = row['id']
        task = TASK_NAMES[row['task']]
        doc = documents[key]
        for obs in row['observations']:
            token = doc.exact_stop(obs['stop'])
            if token is None:
                misses[task] += 1
                continue
            counts = obs['counts']
            binary[task].append((doc.states[token], (counts[0], sum(counts[1:])), row['group_id'], key))
    data, records = {}, {}
    for task, rows in binary.items():
        data[task] = TaskData(
            np.stack([row[0] for row in rows]).astype(np.float64), np.zeros((len(rows), 2)),
            'binary', np.asarray([row[1] for row in rows], dtype=np.int64),
        )
        records[task] = [{'group': row[2], 'id': row[3]} for row in rows]

    tokens, detail, ranges, labels = [], [], [], []
    flattened_start = 0
    for row in evidence_rows:
        origin = layouts[row['article_id']]
        if origin['split'] != split:
            continue
        proxy = row['anchor_proxy']
        if proxy is None:
            misses['evidence_no_prior_proxy'] += 1
            continue
        doc = documents[row['article_id']]
        anchor = doc.exact_stop(proxy['span'][1])
        last = doc.exact_stop(row['first_reference_end'])
        first = int(np.searchsorted(doc.ends, row['first_marked_start'], side='right'))
        if anchor is None or last is None or not anchor < first <= last:
            misses['evidence_no_exact_geometry'] += 1
            continue
        start = anchor + 1
        vectors = doc.states[start:last + 1].astype(np.float64)
        tokens.append(vectors)
        detail.append(pair_features(doc.states[anchor].astype(np.float64), vectors,
                                    np.arange(start, last + 1) - anchor))
        end = flattened_start + len(vectors)
        ranges.append((flattened_start, flattened_start + first - start, end))
        labels.append({'group': origin['group_id'], 'id': row['id'], 'article_id': row['article_id']})
        flattened_start = end
    if not ranges or set(data) != {'issue', 'sense'}:
        raise ValueError(f'Missing a verified {split} endpoint: {misses}')
    data['evidence'] = TaskData(np.concatenate(tokens), np.concatenate(detail), 'interval',
                                np.asarray(ranges, dtype=np.int64))
    records['evidence'] = labels
    if any(len({row['group'] for row in rows}) < 3 for rows in records.values()):
        raise ValueError(f'Too few independent {split} groups for an endpoint')
    return data, records, dict(misses)


def ensemble_readout(models, task, x, extra):
    return np.stack([predict_logits(model, task, x, extra) for model in models])


def stash_models(directory, fitted):
    arrays, metadata = {}, {}
    for mode, models in fitted.items():
        metadata[mode] = {}
        for seed, model in zip(SEEDS, models, strict=True):
            metadata[mode][str(seed)] = {'shared': model['shared'], 'fit_info': model['fit_info']}
            for group in ('adapters', 'heads'):
                for name, value in model[group].items():
                    arrays[f'{mode}/{seed}/{group}/{name}'] = value
    np.savez_compressed(directory / 'weights.npz', **arrays)
    write_json(directory / 'fitting.json', metadata)
    with np.load(directory / 'weights.npz', allow_pickle=False) as reloaded:
        if reloaded.files != list(arrays) or any(not np.array_equal(reloaded[key], arrays[key]) for key in arrays):
            raise RuntimeError('Readout weights did not round-trip losslessly')


def trajectory(models, doc, prior=None):
    vectors = doc.states.astype(np.float64)
    length = len(vectors)
    anchor, current = np.triu_indices(length, k=1)
    if prior is not None:
        open_logits = np.full((1, length), prior['issue'])
        close_logits = np.full((1, len(current)), prior['evidence'])
    else:
        open_logits = ensemble_readout(models, 'issue', vectors, np.zeros((length, 2)))
        pairs = pair_features(vectors[anchor], vectors[current], current - anchor)
        close_logits = ensemble_readout(models, 'evidence', vectors[current], pairs)
    masses, openings = [], []
    for openings_logit, closes_logit in zip(open_logits, close_logits, strict=True):
        p_open = expit(openings_logit)
        no_open = np.concatenate(([0.0], np.cumsum(-np.logaddexp(0, openings_logit))))
        i = np.arange(length)
        opening = p_open * np.exp(no_open[i] - no_open[np.maximum(0, i - OPEN_EXCLUSION_TOKENS)])
        log_step = np.zeros((length, length))
        log_step[anchor, current] = -np.logaddexp(0, closes_logit)
        remaining = np.triu(opening[:, None] * np.exp(np.cumsum(log_step, axis=1)))
        masses.append(remaining)
        openings.append(opening)
    live = np.mean(masses, axis=0)
    opening = np.mean(openings, axis=0)
    load = live.sum(axis=0)
    closing = np.maximum(0.0, np.r_[0.0, load[:-1]] + opening - load)
    return live, {'token_ids': doc.ids, 'offsets': doc.offsets.tolist(),
                  'load': load.tolist(), 'opening': opening.tolist(), 'closing': closing.tolist()}


def cluster_interval(delta, group, weight):
    _, inverse = np.unique(group, return_inverse=True)
    masses = np.bincount(inverse, weights=weight)
    values = np.bincount(inverse, weights=weight * delta)
    generator = np.random.default_rng(35842)
    draws = []
    for _ in range(40):
        indices = generator.integers(0, len(masses), size=(250, len(masses)))
        draws.extend(values[indices].sum(axis=1) / masses[indices].sum(axis=1))
    return {'mean': float(np.average(delta, weights=weight)),
            'ci95': np.quantile(draws, (.025, .975)).tolist(), 'groups': len(masses)}


def constant_readout(task):
    if task.kind == 'binary':
        negative, positive = task.y.sum(axis=0) + .5
        return float(np.log(positive) - np.log(negative))

    def objective(value):
        return per_observation_nll(task, np.full(len(task.x), value)).mean()

    fitted = minimize_scalar(objective, bounds=(-20, 10), method='bounded')
    if not fitted.success:
        raise RuntimeError('The constant first-hit hazard did not converge')
    return float(fitted.x)


def score_human(prior, held, records, fitted):
    result = {}
    for task, dataset in held.items():
        weight = (np.ones(len(records[task])) if dataset.kind == 'interval'
                  else dataset.y.sum(axis=-1))
        groups = [row['group'] for row in records[task]]
        point = {}
        for mode, models in fitted.items():
            if task not in MODES[mode][0]:
                continue
            scores = ensemble_readout(models, task, dataset.x, dataset.extra)
            point[mode] = np.mean([per_observation_nll(dataset, logits) for logits in scores], axis=0)
        point['constant_train_rate'] = per_observation_nll(dataset, np.full(len(dataset.x), prior[task]))
        result[task] = {
            'observed_rows': len(weight), 'observation_weight': int(weight.sum()), 'groups': len(set(groups)),
            'nll_by_mode': {key: float(np.average(value, weights=weight)) for key, value in point.items()},
            'minus_independent': {key: cluster_interval(value - point['independent'], groups, weight)
                                  for key, value in point.items() if key.startswith('joint')},
        }
    return result


def compare_control_effects(analysis):
    methods = analysis['methods']
    comparisons = {}
    for candidate, reference in (('joint', 'independent'), ('joint_semantic', 'independent'),
                                 ('joint', 'constant_train_rate'), ('independent', 'constant_train_rate'),
                                 ('joint_semantic', 'constant_train_rate')):
        name = f'{candidate}_minus_{reference}'
        comparisons[name] = {}
        for channel in ('cohort', 'load'):
            left = methods[candidate]['channels'][channel]
            right = methods[reference]['channels'][channel]
            left_by_family = {row['family_id']: row for row in left['families']}
            right_by_family = {row['family_id']: row for row in right['families']}
            if left_by_family.keys() != right_by_family.keys() or left['effects'].keys() != right['effects'].keys():
                raise ValueError('Curve effects must use the same matched families and outcomes')
            paired = {}
            for effect in left['effects']:
                group = [key for key in sorted(left_by_family)
                         if left_by_family[key][effect] is not None and right_by_family[key][effect] is not None]
                if len(group) >= 2:
                    delta = np.array([left_by_family[key][effect] - right_by_family[key][effect] for key in group])
                    paired[effect] = cluster_interval(delta, group, np.ones(len(group)))
            comparisons[name][channel] = paired
    return comparisons


def report_inventory(data, records):
    return {task: {'observed_rows': len(records[task]), 'fit_rows': len(values.x),
                   'groups': len({row['group'] for row in records[task]}),
                   'observation_weight': (int(values.y.sum()) if values.kind == 'binary' else len(values.y))}
            for task, values in data.items()}


def run(job, events, evidence, controls, output):
    if output.exists():
        raise FileExistsError(output)
    docs, seal = projection(job)
    event_rows = read_checked(events, 'source-audit.json', 'files_sha256', 'labels.jsonl')
    layouts_array = read_checked(evidence, 'assay.json', 'file_sha256', 'layout.jsonl')
    layouts = {row['id']: row for row in layouts_array}
    evidence_rows = read_checked(evidence, 'assay.json', 'file_sha256', 'labels.jsonl')
    control_rows = list(stream_rows(controls))
    control_keys = {row['id'] for row in control_rows}
    expected = {row['id'] for row in event_rows} | set(layouts) | control_keys
    if (expected != docs.keys() or len(expected) != len(event_rows) + len(layouts_array) + len(control_rows)):
        raise ValueError('Every sealed document must have exactly one data role')
    all_splits = defaultdict(set)
    for row in (*event_rows, *layouts_array):
        all_splits[row['group_id']].add(row['split'])
    if any(splits not in ({'train'}, {'test'}) for splits in all_splits.values()):
        raise ValueError('At least one source family crosses the holdout boundary')

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.load-curve-fit-') as folder:
        stage = Path(folder) / 'completed'
        stage.mkdir()
        train, train_records, train_missing = observation_sets(docs, event_rows, layouts, evidence_rows, 'train')
        fitted = {}
        for mode, (tasks, shared) in MODES.items():
            fitted[mode] = []
            for seed in SEEDS:
                model = fit({task: train[task] for task in tasks}, seed, shared)
                fitted[mode].append(model)
                print(f'fitted {mode} seed {seed}', flush=True)
        stash_models(stage, fitted)
        prior = {task: constant_readout(values) for task, values in train.items()}
        held, held_records, held_missing = observation_sets(docs, event_rows, layouts, evidence_rows, 'test')
        human = score_human(prior, held, held_records, fitted)

        traces = {}
        for mode, models in (*fitted.items(), ('constant_train_rate', None)):
            traces[mode] = {}
            for case in control_rows:
                key = case['id']
                doc = docs[key]
                live, trace = trajectory(models, doc, prior=prior if models is None else None)
                stop = doc.exact_stop(case['offsets']['common_prefix_end'])
                if stop is None:
                    raise ValueError('A control cohort boundary is not an exact native-token endpoint')
                trace['cohort'] = live[:stop + 1].sum(axis=0).tolist()
                traces[mode][key] = trace
        write_json(stage / 'traces.json', traces)
        analysis = control_score.evaluate(controls, traces, {key: docs[key].text for key in control_keys})
        write_json(stage / 'controls.json', analysis)
        write_json(stage / 'curve-comparisons.json', compare_control_effects(analysis))
        write_json(stage / 'human.json', {
            'train': report_inventory(train, train_records), 'heldout': human,
            'excluded_observations': {'train': train_missing, 'test': held_missing},
            'seeds': list(SEEDS), 'constant_train_logits': prior,
            'nll_unit': 'nats per observed vote or evidence interval; average across seeds',
        })
        write_json(stage / 'receipt.json', {
            'protocol_sha256': digest(PROTOCOL), 'projection_seal_sha256': digest(job / 'extraction' / 'seal.json'),
            'fit_source_sha256': digest(Path(__file__)),
            'source_audit_sha256': {path.name: digest(path) for path in
                                    (events / 'source-audit.json', evidence / 'assay.json', controls)},
            'files_sha256': {path.name: digest(path) for path in sorted(stage.iterdir())},
            'readout_device': 'CPU', 'dense_projection': seal,
        })
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('job', 'events', 'evidence', 'controls', 'output'):
        parser.add_argument('--' + field, required=True, type=Path)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
