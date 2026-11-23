"""Fit and evaluate causal word-level contradiction coverage and severity."""

import argparse
from collections import Counter, deque
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.core_features.readback_score import average_precision
from research.semantic_tracking import beukers_extract as extractor
from research.semantic_tracking import frick_consistency_score as records
from research.semantic_tracking.source_prepare import WORD


PROTOCOL = Path(__file__).with_name('scarecrow_detection_protocol.md')
SOURCE = PROTOCOL.parents[1] / 'results/scarecrow-detection-source-audit.json'
RIDGE = 0.003
SEED = 43421
DRAWS = 10_000
SURFACE_WIDTH = 10
METHODS = ('constant', 'surface', 'qwen')


def projection_inputs(work, extraction):
    packet = work / 'packet'
    audit = records.read_json(SOURCE)
    if (any(records.digest(packet / name) != value for name, value in audit['packet_sha256'].items())
            or any(records.digest(work / 'prepared' / (stem + '.jsonl')) != value
                   for stem, value in audit['prepared_sha256'].items())):
        raise ValueError('An audited SCARECROW input or local annotation crosswalk changed')
    corpus = records.indexed(extractor.read_stimuli(packet / 'stimuli.jsonl'), 'SCARECROW stimuli')
    layout = records.indexed(records.read_rows(work / 'prepared/layout.jsonl'), 'SCARECROW word geometry')
    preflight = records.read_json(packet / 'preflight.json')
    seal = records.read_json(extraction / 'seal.json')
    expected = {'candidate_id': extractor.CANDIDATE, 'tokenizer_sha256': extractor.TOKENIZER_SHA256,
                'stimuli_sha256': records.digest(packet / 'stimuli.jsonl'), 'gpu': 'NVIDIA L4',
                'assay_sha256': records.digest(packet / 'preflight.json'), 'content_forward_per_document': 1,
                'preflight_sha256': records.digest(packet / 'extractor-preflight.json'), 'storage_dtype': 'float16',
                'protocol_sha256': records.digest(PROTOCOL), 'independent_causal_checks': 12}
    if (any(seal.get(key) != value for key, value in expected.items())
            or any(records.digest(extraction / name) != digest for name, digest in seal['file_sha256'].items())):
        raise ValueError('Frozen-archive seal disagrees with the exact SCARECROW procedure or data')
    if corpus.keys() != layout.keys() or corpus.keys() != preflight['documents'].keys():
        raise ValueError('The SCARECROW word inventory does not match the inference packet')
    coordinates = records.read_rows(extraction / 'coordinates.jsonl')
    expected_coordinates = [{'id': key, 'stops': row['stops']} for key, row in corpus.items()]
    checks = records.read_json(extraction / 'causality.json')['checks']
    if (coordinates != expected_coordinates or len(checks) != 12
            or not all(item['passed'] and item['projected_tokens']['passed'] for item in checks)):
        raise ValueError('SCARECROW positional or causal controls did not pass')
    vectors = np.load(extraction / 'vectors.npy', allow_pickle=False)
    expected_shape = (sum(len(row['stops']) for row in corpus.values()), extractor.PROJECTION_WIDTH)
    if (vectors.dtype != np.float16 or vectors.shape != expected_shape
            or list(vectors.shape) != seal['vectors_shape'] or not np.isfinite(vectors).all()):
        raise ValueError('Projection array does not contain one FP16 vector per sealed word stop')
    return corpus, layout, vectors


def word_features(text, candidates):
    candidate_set = {tuple(span) for span in candidates}
    features = []
    last = {}
    history = deque(maxlen=32)
    previous = ''
    for i, match in enumerate(WORD.finditer(text)):
        word = match.group()
        form = word.casefold()
        if match.span() in candidate_set:
            features.append([np.log1p(len(word)), np.log1p(i), form in last,
                             np.log1p(i - last.get(form, i)), word[:1].isupper(), any(ch.isdigit() for ch in word),
                             any(not ch.isalnum() for ch in word), previous[:1].isupper(),
                             bool(previous) and not previous[-1].isalnum(), len(set(history)) / max(1, len(history))])
        last[form] = i
        history.append(form)
        previous = word
    if len(features) != len(candidates):
        raise ValueError('A retained SCARECROW text range is not a lexical word span')
    return np.asarray(features, dtype=np.float32)


def feature_matrix(corpus, layout, vectors):
    width = vectors.shape[1]
    matrix = np.empty((len(vectors), SURFACE_WIDTH + 2 * width), dtype=np.float32)
    matrix[:, SURFACE_WIDTH:SURFACE_WIDTH + width] = vectors
    split = np.empty(len(vectors), dtype=bool)
    spans_by_doc, offset = {}, 0
    for key, content in corpus.items():
        row = layout[key]
        words = row['words']
        stops = [right for _, right in words]
        word_stops = set(stops)
        if (row['split'] not in ('train', 'test') or stops != sorted(word_stops) or not stops
                or not word_stops <= set(content['stops']) or type(row['prior_pointer_group']) is not bool):
            raise ValueError('Expected one documented train/test group and exact stops per lexical word')
        length = len(content['stops'])
        relative = np.array([i for i, stop in enumerate(content['stops']) if stop in word_stops])
        indices = relative + offset
        matrix[indices, :SURFACE_WIDTH] = word_features(content['text'], words)
        difference = matrix[indices, SURFACE_WIDTH:SURFACE_WIDTH + width]
        matrix[indices[1:], -width:] = np.diff(difference, axis=0)
        matrix[indices[0], -width:] = 0
        split[offset:offset + length] = False
        split[indices] = row['split'] == 'train'
        spans_by_doc[key] = indices
        offset += length
    family_sides = {(row['group_id'], row['split']) for row in layout.values()}
    if len({family for family, _ in family_sides}) != len(family_sides):
        raise ValueError('A SCARECROW provenance family crosses fit and test')
    indices = np.concatenate(list(spans_by_doc.values()))
    active = np.zeros(len(vectors), dtype=bool)
    active[indices] = True
    matrix[~active] = 0
    training = matrix[split]
    center = training.mean(axis=0, dtype=np.float64).astype(np.float32)
    variance = np.var(training, axis=0, dtype=np.float64)
    del training
    scale = np.sqrt(variance)
    for block in (slice(SURFACE_WIDTH, SURFACE_WIDTH + width), slice(SURFACE_WIDTH + width, None)):
        scale[block] = np.sqrt(variance[block].sum())
    scale = np.where(scale > 1e-9, scale, 1.).astype(np.float32)
    matrix -= center
    matrix /= scale
    return matrix, split, spans_by_doc, center, scale


def fit_binary(features, positive, total):
    design = features[total > 0]
    if not 0 < positive.sum() < total.sum():
        raise ValueError('Each contradiction branch requires both observed classes')
    observed = positive[total > 0].astype(np.float64)
    trials = total[total > 0].astype(np.float64)
    n = trials.sum()

    def objective(parameters):
        coef = parameters[1:]
        loss = RIDGE / 2 * (coef @ coef)
        gradient = np.r_[0., RIDGE * coef]
        for start in range(0, len(design), 8192):
            rows = slice(start, start + 8192)
            block = design[rows].astype(np.float64)
            logits = parameters[0] + block @ coef
            loss += (trials[rows] @ np.logaddexp(0., logits) - observed[rows] @ logits) / n
            residual = (trials[rows] * expit(logits) - observed[rows]) / n
            gradient[0] += residual.sum()
            gradient[1:] += block.T @ residual
        return float(loss), gradient

    intercept = np.log(observed.sum() / (n - observed.sum()))
    initial = np.r_[intercept, np.zeros(design.shape[1])]
    fitted = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      options={'maxiter': 260, 'gtol': 1e-8, 'ftol': 1e-13})
    loss, gradient = objective(fitted.x)
    if not np.isfinite(loss) or np.max(np.abs(gradient)) > 1e-6:
        raise ArithmeticError(f'Word-level contradiction head failed to converge: {fitted.message}')
    return fitted.x, {'total': int(n), 'positive': int(observed.sum()), 'iterations': int(fitted.nit),
                      'objective': loss, 'maximum_gradient': float(np.max(np.abs(gradient)))}


def predict(fit, features):
    logits = fit[None, :, 0] + features @ fit[:, 1:].T
    p, severe, high = expit(logits).T
    probabilities = np.column_stack((1 - p, p * (1 - severe), p * severe * (1 - high), p * severe * high))
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0) or np.any(probabilities <= 0.):
        raise ArithmeticError('Non-normalized retrospective severity distributions')
    return probabilities


def counts_row(values):
    positive, total, severity = values['positive'], values['total'], values['severity_counts']
    if (type(total) is not int or type(positive) is not int or len(severity) != 3
            or any(type(value) is not int or value < 0 for value in severity)
            or positive != sum(severity) or not 0 <= positive <= total):
        raise ValueError('Contradiction votes and severity must share the same observed annotation denominator')
    return [total - positive, *severity]


def observed_counts(work, spans_by_doc, length):
    outcomes = records.indexed(records.read_rows(work / 'prepared/outcomes.jsonl'), 'SCARECROW local votes')
    if outcomes.keys() != spans_by_doc.keys():
        raise ValueError('The SCARECROW word and known-annotation inventories differ')
    counts = np.zeros((length, 4), dtype=np.int64)
    for key, indices in spans_by_doc.items():
        rows = outcomes[key]['counts']
        if len(rows) != len(indices):
            raise ValueError('Expected exactly one outcome count per retained lexical position')
        counts[indices] = [counts_row(row) for row in rows]
    return counts


def fit_arms(matrix, counts, split):
    training = counts[split]
    total = training.sum(axis=1)
    risk = [total, training[:, 1:].sum(axis=1), training[:, 2:].sum(axis=1)]
    hits = [risk[1], risk[2], training[:, 3]]
    fits = {}
    for method, width in zip(METHODS, (0, SURFACE_WIDTH, matrix.shape[1]), strict=True):
        features = matrix[split, :width]
        coefficients, diagnostics = zip(*(fit_binary(features, y, n) for y, n in zip(hits, risk, strict=True)), strict=True)
        fits[method] = {'columns': width, 'coefficients': [row.tolist() for row in coefficients], 'training': diagnostics}
    return fits


METRICS = ('within_document_ap', 'joint_nll', 'mark_positive_severity_nll', 'coverage_bias', 'severity_bias', 'zero_document_mark_rate')


def document_metrics(forecast, counts):
    eligible = counts.sum(axis=1) > 0
    probabilities, labels = forecast[eligible], counts[eligible]
    observed = labels.sum(axis=1)
    marks = labels[:, 1:].sum(axis=1)
    mark_probability = 1. - probabilities[:, 0]
    conditionals = probabilities[:, 1:] / probabilities[:, 1:].sum(axis=1, keepdims=True)
    severity = np.arange(1, 4)
    valid_ap = bool((marks > 0).any() and (marks == 0).any())
    ap = average_precision(mark_probability, marks > 0) if valid_ap else 0.
    return (np.array([ap, -np.sum(labels * np.log(probabilities)), -np.sum(labels[:, 1:] * np.log(conditionals)),
                      observed @ mark_probability - marks.sum(), marks @ (conditionals @ severity) - (labels[:, 1:] @ severity).sum(),
                      (observed @ mark_probability) if not marks.any() else 0.]),
            np.array([valid_ap, observed.sum(), marks.sum(), observed.sum(), marks.sum(), observed.sum() if not marks.any() else 0.]))


def interval_rows(document, layout):
    groups = sorted({layout[key]['group_id'] for key in document})
    group_index = {group: i for i, group in enumerate(groups)}
    totals = np.zeros((len(groups), len(METHODS), len(METRICS)), dtype=np.float64)
    denominators = np.zeros_like(totals)
    for key, row in document.items():
        i = group_index[layout[key]['group_id']]
        for j, method in enumerate(METHODS):
            numerator, denominator = row[method]
            totals[i, j] += numerator
            denominators[i, j] += denominator
    rng = np.random.default_rng(SEED)
    weights = np.vstack((np.ones(len(groups)), rng.multinomial(len(groups), np.ones(len(groups)) / len(groups), DRAWS)))
    samples = np.einsum('bg,gam->bam', weights, totals, optimize=True)
    counts = np.einsum('bg,gam->bam', weights, denominators, optimize=True)
    estimates = np.full_like(samples, np.nan)
    np.divide(samples, counts, out=estimates, where=counts > 0)

    def interval(values):
        usable = values[1:][np.isfinite(values[1:])]
        if not len(usable) or not np.isfinite(values[0]):
            return None
        return {'estimate': float(values[0]), 'ci90': np.quantile(usable, [.05, .95]).tolist(),
                'ci95': np.quantile(usable, [.025, .975]).tolist(), 'bootstrap_with_observations': len(usable)}

    return {'groups': len(groups), 'documents': len(document),
            'units': dict(zip(METRICS, counts[0, 0].astype(int).tolist(), strict=True)),
            'methods': {method: {metric: interval(estimates[:, j, k]) for k, metric in enumerate(METRICS)}
                        for j, method in enumerate(METHODS)},
            'qwen_minus_surface': {metric: interval(estimates[:, 2, k] - estimates[:, 1, k])
                                   for k, metric in enumerate(METRICS)}}


def write_scores(stage, work, extraction):
    corpus, layout, vectors = projection_inputs(work, extraction)
    matrix, split, spans, center, scale = feature_matrix(corpus, layout, vectors)
    counts = observed_counts(work, spans, len(matrix))
    fits = fit_arms(matrix, counts, split)
    test_spans = {key: idx for key, idx in spans.items() if layout[key]['split'] == 'test'}
    doc_metrics = {key: {} for key in test_spans}
    forecast = stage / 'forecasts.jsonl'
    with forecast.open('x', encoding='utf-8') as handle:
        for key, ix in test_spans.items():
            joint = {method: predict(np.asarray(fit['coefficients']), matrix[ix, :fit['columns']])
                     for method, fit in fits.items()}
            result = {'id': key, 'words': layout[key]['words'], 'probabilities': {key: arr.tolist() for key, arr in joint.items()}}
            handle.write(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
            doc_metrics[key] = {method: document_metrics(prob, counts[ix]) for method, prob in joint.items()}
    metrics = {name: interval_rows(selected, layout) for name, selected in (
        ('all', doc_metrics),
        ('previous_pointer_groups', {key: row for key, row in doc_metrics.items() if layout[key]['prior_pointer_group']}),
        ('new_groups', {key: row for key, row in doc_metrics.items() if not layout[key]['prior_pointer_group']}),
    ) if selected}
    delta = metrics['all']['qwen_minus_surface']
    qwen = metrics['all']['methods']['qwen']
    gates = {'within_document_ap': delta['within_document_ap']['ci95'][0] > 0,
             'joint_nll': delta['joint_nll']['ci95'][1] < 0,
             'mark_positive_severity_nll': delta['mark_positive_severity_nll']['ci95'][1] < 0}
    for metric, limit in (('coverage_bias', .01), ('severity_bias', .15)):
        low, high = qwen[metric]['ci90']
        gates[metric] = low >= -limit and high <= limit
    records.write_json(stage / 'metrics.json', {'scoring': metrics, 'gates': gates, 'seed': SEED, 'draws': DRAWS,
                                                 'scored_documents': len(test_spans), 'train_documents': len(spans) - len(test_spans),
                                                 'train_counts': counts[split].sum(axis=0).tolist(), 'test_counts': counts[~split].sum(axis=0).tolist()})
    records.write_json(stage / 'parameters.json', {'fits': fits, 'standardization_center': center.tolist(),
                                                    'standardization_scale': scale.tolist(), 'ridge': RIDGE})
    records.write_json(stage / 'receipt.json', {'protocol_sha256': records.digest(PROTOCOL),
                                                'scorer_sha256': records.digest(Path(__file__)),
                                                'source_audit_sha256': records.digest(SOURCE),
                                                'extraction_seal_sha256': records.digest(extraction / 'seal.json'),
                                                'prepared_sha256': {name: records.digest(work / 'prepared' / name)
                                                                    for name in ('stimuli.jsonl', 'layout.jsonl', 'outcomes.jsonl')},
                                                'outputs_sha256': {item.name: records.digest(item) for item in sorted(stage.iterdir())}})


def score(work, extraction, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent, prefix='.scarecrow-score-') as temp:
        stage = Path(temp) / 'results'
        stage.mkdir()
        write_scores(stage, work, extraction)
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('work', 'extraction', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    score(**vars(parser.parse_args()))
