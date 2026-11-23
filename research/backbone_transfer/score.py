"""Re-fit PLABA and CompLex readouts on identical eligible human observations."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import log_softmax

from research.backbone_transfer.worker import digest
from research.semantic_tracking.frick_consistency_score import indexed, read_json, read_rows, write_json
from research.stage3.causal_ordinal import LEVELS, TARGETS, design_matrices, fit_softmax, metrics
from research.stage5.complex_score import TIE_TOLERANCE


PROTOCOL = Path(__file__).with_name('protocol.md')
REGISTRY = Path(__file__).with_name('models.json')
ALIASES = ('qwen', 'lfm3', 'gemma4')
DRAWS = 10_000
SEED = 43
SOURCE_AUDIT_SHA256 = '82a35093e87c31a2b15d650c87ff660f74fbd6a26ccfee434c1a3c80b23209e6'


def pair_changes(values, groups):
    values = np.asarray(values, dtype=np.float64)
    _, index = np.unique(groups, return_inverse=True)
    counts = np.bincount(index)
    totals = np.bincount(index, weights=values)
    if len(counts) < 2:
        raise ValueError('A material interval needs two or more independent groups')
    rng = np.random.default_rng(SEED)
    samples = np.empty(DRAWS)
    for start in range(0, DRAWS, 256):
        end = min(start + 256, DRAWS)
        drawn = rng.integers(len(counts), size=(end - start, len(counts)))
        samples[start:end] = totals[drawn].sum(axis=1) / counts[drawn].sum(axis=1)
    return {'estimate': float(values.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist(), 'groups': len(counts)}


def declarations(packing):
    stimuli = packing / 'stimuli.jsonl'
    targets = packing / 'targets.jsonl'
    inventory = read_json(packing / 'inventory.json')
    if (inventory['protocol_sha256'] != digest(PROTOCOL) or inventory['stimuli_sha256'] != digest(stimuli)
            or inventory['targets_sha256'] != digest(targets)):
        raise ValueError('The frozen multi-backbone input packet changed')
    return indexed(read_rows(stimuli), 'combined grade-free stimuli'), {
        (row['id'], row['start'], row['end']) for row in read_rows(targets)
    }


def open_projection(folder, model, corpus, target_keys, packing):
    receipt = read_json(folder / 'seal.json')
    registry = read_json(REGISTRY)
    spec = registry['models'][model]
    expected = {'protocol_sha256': digest(PROTOCOL), 'stimuli_sha256': digest(packing / 'stimuli.jsonl'),
                'targets_sha256': digest(packing / 'targets.jsonl'), 'model_files_sha256': spec['sha256'],
                'candidate': model, 'model': spec['repo'], 'revision': spec['revision'],
                'transformers': spec.get('transformers', registry['transformers']),
                'content_forward_per_document': 1, 'compute_dtype': 'float32', 'gpu': 'NVIDIA L4'}
    names = {'states.npy', 'coordinates.jsonl', 'preflight.json', 'causality.json'}
    if (any(receipt.get(key) != value for key, value in expected.items())
            or receipt['file_sha256'].keys() != names
            or any(digest(folder / name) != value for name, value in receipt['file_sha256'].items())):
        raise ValueError(f'Extraction does not belong to this protocol and data: {model}')
    checks = read_json(folder / 'causality.json')['checks']
    if not checks or not all(row['passed'] for row in checks):
        raise ValueError(f'Lookahead or runtime-causing cross-input interference detected: {model}')
    coordinates = indexed(read_rows(folder / 'coordinates.jsonl'), f'{model} coordinates')
    if list(coordinates) != list(corpus):
        raise ValueError(f'Extraction changed document identities or order: {model}')
    rows = {}
    offset = 0
    for key, row in coordinates.items():
        stops = row['stops']
        if stops != sorted(set(stops)) or not set(stops) <= set(corpus[key]['stops']):
            raise ValueError(f'Unexpected feature stop for {model}: {key}')
        rows.update({(key, end): offset + i for i, end in enumerate(stops)})
        offset += len(stops)
    features = np.load(folder / 'states.npy', mmap_mode='r', allow_pickle=False)
    if features.dtype != np.float32 or features.shape != (offset, spec['hidden_width']):
        raise ValueError(f'Extraction lacks the sealed FP32 state matrix: {model}')
    preflight = read_json(folder / 'preflight.json')
    targets = {(row['id'], row['start'], row['end']): row['eligible'] for row in preflight['target_geometry']}
    if (targets.keys() != target_keys or len(targets) != len(preflight['target_geometry'])
            or any(type(value) is not bool for value in targets.values())):
        raise ValueError(f'Extraction altered the official CompLex geometry request: {model}')
    return rows, features, targets


def common_cases(directory, extracted, dataset):
    layout = read_rows(directory / 'layout.jsonl')
    outcomes = {row['anchor_id']: row for row in read_rows(directory / 'outcomes.jsonl')}
    if len(outcomes) != len(layout) or set(outcomes) != {row['anchor_id'] for row in layout}:
        raise ValueError(f'{dataset} layouts and original human annotation keys differ')

    def visible(key, row):
        coordinates, _, targets = extracted[key]
        exists = (row['id'], row['end']) in coordinates
        return exists and (dataset != 'complex' or targets[(row['id'], row['start'], row['end'])])

    groups = ('fold' if dataset == 'plaba' else 'split')
    eligibility = {key: Counter(str(row[groups]) for row in layout if visible(key, row)) for key in ALIASES}
    common = [row for row in layout if all(visible(key, row) for key in ALIASES)]
    if len(common) == 0:
        raise ValueError('No certified token coordinates are shared across the three backbones')
    return common, outcomes, {'original': dict(Counter(str(row[groups]) for row in layout)),
                              'by_model': {key: dict(counts) for key, counts in eligibility.items()},
                              'common': dict(Counter(str(row[groups]) for row in common)),
                              'excluded_from_common': [row['anchor_id'] for row in layout
                                                       if any(not visible(key, row) for key in ALIASES)]}


def unit_features(extracted, rows):
    index, vectors, _ = extracted
    features = vectors[[index[(row['id'], row['end'])] for row in rows]].astype(np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('A retained human observation has an invalid frozen model state')
    features /= norms
    return features


def changes_plaba(rows, observed, predictions, full_sizes):
    by_pair = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        if row['alignment_anchor_id'] is not None:
            by_pair[row['alignment_anchor_id']][row['variant']].append(i)
    pairs = [(values['orig'], values['simp']) for key, values in by_pair.items()
             if values.keys() == {'orig', 'simp'} and {name: len(v) for name, v in values.items()} == full_sizes[key]]
    groups = [rows[original[0]]['group_id'] for original, _ in pairs]

    def contrast(ratings):
        return np.array([ratings[simple].mean(axis=0) - ratings[original].mean(axis=0) for original, simple in pairs])

    actual = contrast(observed @ LEVELS)
    estimates = {key: contrast(np.exp(probs) @ LEVELS) for key, probs in predictions.items()}
    return {'aligned_anchors': len(pairs), 'scales': {
        name: {'human': float(actual[:, i].mean()), 'model': {
            key: {'effect': float(value[:, i].mean()), 'model_minus_human': pair_changes((value - actual)[:, i], groups)}
            for key, value in estimates.items()}}
        for i, name in enumerate(TARGETS)
    }}


def plaba(prepared, corpus, extracted, output):
    rows, outcomes, coverage = common_cases(prepared, extracted, 'plaba')
    groups = [row['group_id'] for row in rows]
    folds = np.array([row['fold'] for row in rows])
    expected_folds = set(coverage['original'])
    if {str(value) for value in folds} != expected_folds or len(expected_folds) != 5:
        raise ValueError('Expected all five original grouped PLABA folds in the common model subset')
    raw = np.array([[outcomes[row['anchor_id']][target] for target in TARGETS] for row in rows], dtype=float)
    if raw.shape != (len(rows), len(TARGETS), 5) or np.any(raw < 0) or np.any(raw.sum(axis=-1) <= 0):
        raise ValueError('Invalid original two-scale PLABA grade counts')
    observed = raw / raw.sum(axis=-1, keepdims=True)
    spans = [corpus[row['id']]['text'][row['start']:row['end']] for row in rows]
    lengths = np.log1p([(len(span), len(span.split())) for span in spans])
    variants = np.array([row['variant'] == 'orig' for row in rows])
    predictions, weights = {}, {}
    for method in ('length', 'length_plus_variant', *ALIASES):
        features = unit_features(extracted[method], rows) if method in ALIASES else np.empty((len(rows), 0))
        estimates = np.empty_like(observed)
        method_weights = []
        for fold in sorted(set(folds)):
            test = folds == fold
            matrices, center, scale = design_matrices(features, lengths, variants, ~test)
            design = matrices['qwen_plus_length' if method in ALIASES else method]
            fits = []
            for col in range(len(TARGETS)):
                fit = fit_softmax(design[~test], observed[~test, col])
                fits.append(fit.tolist())
                estimates[test, col] = log_softmax(design[test] @ fit, axis=-1)
            method_weights.append({'fold': int(fold), 'length_center': center.tolist(), 'length_scale': scale.tolist(), 'head': fits})
        predictions[method] = estimates
        weights[method] = method_weights
    losses = {method: metrics(prediction, observed) for method, prediction in predictions.items()}
    results = {'cohort': coverage, 'rated_pairs': int(raw[:, 0].sum()), 'scales': {}}
    for i, name in enumerate(TARGETS):
        results['scales'][name] = {}
        for measure in next(iter(losses.values())):
            values = {key: loss[measure][:, i] for key, loss in losses.items()}
            results['scales'][name][measure] = {'score': {key: float(value.mean()) for key, value in values.items()},
                                               'models_minus_length_and_variant': {key: pair_changes(values[key] - values['length_plus_variant'], groups) for key in ALIASES},
                                               'alternatives_minus_qwen': {key: pair_changes(values[key] - values['qwen'], groups) for key in ALIASES[1:]}}
    full = defaultdict(Counter)
    for row in read_rows(prepared / 'layout.jsonl'):
        if row['alignment_anchor_id'] is not None:
            full[row['alignment_anchor_id']][row['variant']] += 1
    results['rewrite_effect'] = changes_plaba(rows, observed, predictions, full)
    output.mkdir()
    write_json(output / 'metrics.json', results)
    write_json(output / 'weights.json', weights)
    np.savez_compressed(output / 'forecasts.npz', anchor_id=[row['anchor_id'] for row in rows],
                        **{key: np.exp(value) for key, value in predictions.items()})


def ridge(features, target, penalty):
    center = features.mean(axis=0)
    shifted = features - center
    mean = float(target.mean())
    gram = shifted.T @ shifted
    gram.flat[::gram.shape[0] + 1] += penalty
    rhs = shifted.T @ (target - mean)
    weights = cho_solve(cho_factor(gram, overwrite_a=True), rhs)
    intercept = mean - center @ weights
    if not np.isfinite(weights).all() or not np.isfinite(intercept):
        raise ArithmeticError('The CompLex ridge solution is not finite')
    return weights, float(intercept)


def lexical_surface(rows, corpus, train):
    target = [corpus[row['id']]['text'][row['start']:row['end']] for row in rows]
    normalized = [' '.join(term.casefold().split()) for term in target]
    frequencies = Counter(term for term, learn in zip(normalized, train, strict=True) if learn)
    features = np.asarray([[np.log1p(len(term)), np.log1p(len(term.split())), np.log1p(row['start']),
                            np.log1p(len(corpus[row['id']]['text'][:row['start']].split())),
                            np.log1p(frequencies[form]), term[:1].isupper(), any(char.isdigit() for char in term)]
                           for row, term, form in zip(rows, target, normalized, strict=True)])
    center, scale = features[train].mean(axis=0), features[train].std(axis=0)
    scale = np.maximum(scale, .1)
    return (features - center) / scale / np.sqrt(features.shape[1]), target, center, scale


def same_target_pairs(rows, targets, values):
    by_target = defaultdict(list)
    for i, (row, target) in enumerate(zip(rows, targets, strict=True)):
        by_target[(row['kind'], target)].append(i)
    pairs = [(a, b) for indices in by_target.values() for a, b in combinations(indices, 2)
             if rows[a]['group_id'] != rows[b]['group_id'] and values[a] != values[b]]
    if not pairs:
        raise ValueError('No held-out same-target pairs have different sentences and human ratings')
    left, right = np.asarray(pairs).T
    return left, right, np.sign(values[left] - values[right])


def complex_scores(prepared, corpus, extracted, output):
    rows, outcomes, coverage = common_cases(prepared, extracted, 'complex')
    splits = np.array([row['split'] for row in rows])
    train, test = splits == 'train', splits == 'test'
    groups = [row['group_id'] for row, held in zip(rows, test, strict=True) if held]
    human = np.array([outcomes[row['anchor_id']]['value'] for row in rows])
    if not np.all((0 <= human) & (human <= 1)) or not train.any() or not test.any():
        raise ValueError('Invalid original CompLex 0–1 data or split coverage')
    surface, targets, center, scale = lexical_surface(rows, corpus, train)
    test_rows = [row for row, held in zip(rows, test, strict=True) if held]
    test_targets = [target for target, held in zip(targets, test, strict=True) if held]
    left, right, direction = same_target_pairs(test_rows, test_targets, human[test])
    penalty = train.sum() / 8667
    fits, forecasts, absolute = {}, {}, {}
    concordance = {}
    for method in ('surface', *ALIASES):
        features = surface if method == 'surface' else unit_features(extracted[method], rows)
        coef, intercept = ridge(features[train], human[train], penalty)
        estimates = np.clip(features @ coef + intercept, 0., 1.)
        forecasts[method] = estimates
        absolute[method] = np.abs(estimates[test] - human[test])
        fits[method] = {'weights': coef.tolist(), 'intercept': intercept}
        differences = estimates[test][left] - estimates[test][right]
        agreement = np.where(np.abs(differences) <= TIE_TOLERANCE, 0.5, differences * direction > 0)
        concordance[method] = {'pairs': len(left), 'concordance': float(agreement.mean())}
    output.mkdir()
    write_json(output / 'metrics.json', {'cohort': coverage, 'mae': {key: float(loss.mean()) for key, loss in absolute.items()},
                                        'models_minus_surface': {key: pair_changes(absolute[key] - absolute['surface'], groups) for key in ALIASES},
                                        'alternatives_minus_qwen': {key: pair_changes(absolute[key] - absolute['qwen'], groups) for key in ALIASES[1:]},
                                        'same_raw_target_different_sentences': concordance})
    write_json(output / 'weights.json', {'fits': fits, 'lambda': float(penalty), 'surface_center': center.tolist(), 'surface_scale': scale.tolist()})
    np.savez_compressed(output / 'forecasts.npz', anchor_id=[row['anchor_id'] for row in rows], **forecasts)


def run(prepared, packing, extraction, output):
    folders = dict((key, Path(path)) for key, path in (item.split('=', 1) for item in extraction))
    if len(extraction) != len(ALIASES) or folders.keys() != set(ALIASES):
        raise ValueError('Pass exactly one verified extraction directory for each of qwen, lfm3 and gemma4')
    audit = prepared / 'source-audit.json'
    if digest(audit) != SOURCE_AUDIT_SHA256:
        raise ValueError('PLABA and CompLex original study alignment audit changed')
    for dataset, study in read_json(audit)['datasets'].items():
        if any(digest(prepared / dataset / name) != signature for name, signature in study['output_sha256'].items()):
            raise ValueError(f'The fixed source human observations or folds changed: {dataset}')
    corpus, targets = declarations(packing)
    extracted = {key: open_projection(folders[key], key, corpus, targets, packing) for key in ALIASES}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.transfer-score-') as directory:
        table = Path(directory) / 'result'
        table.mkdir()
        plaba(prepared / 'plaba', corpus, extracted, table / 'plaba')
        complex_scores(prepared / 'complex', corpus, extracted, table / 'complex')
        write_json(table / 'receipt.json', {'protocol_sha256': digest(PROTOCOL), 'scorer_sha256': digest(Path(__file__)),
                                            'backbone_registry_sha256': digest(REGISTRY),
                                            'study_source_audit_sha256': digest(audit),
                                            'input_sha256': {key: digest(folders[key] / 'seal.json') for key in ALIASES},
                                            'files_sha256': {str(path.relative_to(table)): digest(path)
                                                             for path in sorted(table.rglob('*')) if path.is_file()}})
        table.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'packing', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--extraction', action='append', required=True, metavar='ALIAS=DIR')
    run(**vars(parser.parse_args()))
