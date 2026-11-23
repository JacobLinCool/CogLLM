"""Fit and evaluate matched listwise ContractNLI evidence rerankers on CPU."""

from collections import Counter, defaultdict
from itertools import product

import numpy as np
from scipy.optimize import minimize

from research.semantic_tracking.contractnli_score import (
    DRAWS,
    NO_EVIDENCE,
    POSITIVE_LABELS,
    SEED,
    average_precision,
    estimate,
    evidence_order,
    family_means,
    is_identifier,
    is_integer,
    is_number,
)


FEATURES = ('bm25_log1p', 'qwen_raw_logit', 'qwen_centered_logit')
BASELINES = {'bm25': 0, 'token_rank': 1, 'centered_token_rank': 2}
LEARNED = {'qwen_only': (1, 2), 'fused_raw': (0, 1), 'fused_centered': (0, 2)}
METHODS = (*BASELINES, *LEARNED)
PRIMARY = 'fused_centered'
RIDGE = 0.05


def index_cases(cases):
    records, documents = {}, {}
    for row in cases:
        identifier, query, family = (row[field] for field in ('id', 'query_id', 'group_id'))
        if any(not is_identifier(value) for value in (identifier, query, family)):
            raise ValueError('Every candidate row needs a document, query and family identifier')
        key = identifier, query
        indices = tuple(row['candidate_indices'])
        if (key in records or any(not is_integer(i) or i < 0 for i in indices)
                or any(a >= b for a, b in zip(indices, indices[1:]))):
            raise ValueError('Each candidate row requires a unique document–claim pair and ordered source indices')
        document = {'group_id': family, 'candidate_indices': indices}
        if documents.setdefault(identifier, document) != document:
            raise ValueError(f'Claims disagree on the candidate pool or family for {identifier}')
        records[key] = row
    if not records:
        raise ValueError('Expected a nonempty case inventory')
    return records, documents


def index_gold(gold, records, documents):
    truth, official_ids = {}, {}
    for row in gold:
        identifier, query, family, official = (row[field] for field in ('id', 'query_id', 'group_id', 'official_id'))
        if any(not is_identifier(value) for value in (identifier, query, family)) or not is_integer(official) or official < 0:
            raise ValueError('Gold rows require exact document, query, family and official source identifiers')
        key = identifier, query
        if (key not in records or key in truth or documents[identifier]['group_id'] != family
                or official_ids.setdefault(identifier, official) != official):
            raise ValueError('Gold and candidate records disagree on claim identity or document membership')
        label = row['label']
        evidence = row['evidence_indices']
        if (label not in (*POSITIVE_LABELS, NO_EVIDENCE)
                or any(not is_integer(i) or i < 0 for i in evidence)
                or len(evidence) != len(set(evidence)) or bool(evidence) != (label in POSITIVE_LABELS)):
            raise ValueError(f'Malformed evidence set or official label for {identifier}/{query}')
        truth[key] = label, frozenset(evidence)
    if truth.keys() != records.keys() or len(set(official_ids.values())) != len(documents):
        raise ValueError('Gold rows must exactly cover each candidate record and each official document')
    return truth


def feature_matrices(records):
    matrices = {}
    for key, row in records.items():
        values = row['features']
        if len(values) != len(row['candidate_indices']) or any(
            len(vector) != len(FEATURES) or any(not is_number(x) for x in vector) or vector[0] < 0 for vector in values
        ):
            raise ValueError(f'Expected finite candidate features and nonnegative log1p BM25 scores for {key}')
        matrices[key] = np.asarray(values, dtype=np.float64).reshape(len(values), len(FEATURES))
    return matrices


def training_data(matrices, documents, truth):
    comparisons, selected, by_document = [], [], defaultdict(list)
    positions = {key: {span: i for i, span in enumerate(info['candidate_indices'])} for key, info in documents.items()}
    for (identifier, query), (label, evidence) in sorted(truth.items()):
        indices = positions[identifier]
        if label == NO_EVIDENCE or not evidence <= indices.keys():
            continue
        position = len(comparisons)
        comparisons.append(matrices[identifier, query])
        selected.append([indices[i] for i in sorted(evidence)])
        by_document[identifier].append(position)
    if not comparisons:
        raise ValueError('Listwise fitting requires positive claims with fully aligned evidence')

    group_sizes = Counter(documents[identifier]['group_id'] for identifier in by_document)
    weights = np.empty(len(comparisons), dtype=np.float64)
    for identifier, rows in by_document.items():
        group = documents[identifier]['group_id']
        weights[rows] = 1.0 / (len(group_sizes) * group_sizes[group] * len(rows))

    lengths = np.asarray([len(values) for values in comparisons], dtype=np.int64)
    matrix = np.concatenate(comparisons)
    starts = np.r_[0, lengths.cumsum()[:-1]]
    candidate_weights = np.repeat(weights / lengths, lengths)
    means = np.average(matrix, axis=0, weights=candidate_weights)
    centered = matrix - means
    scales = np.sqrt(np.average(centered ** 2, axis=0, weights=candidate_weights))
    constant = np.max(matrix, axis=0) == np.min(matrix, axis=0)
    scales[constant] = 1.0
    if not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError('Training candidate scales must be finite and positive')
    standardized = centered / scales
    positive_features = np.stack([standardized[start + indices].mean(axis=0)
                                  for start, indices in zip(starts, map(np.asarray, selected), strict=True)])
    normalization = {'means': means.tolist(), 'scales': scales.tolist(),
                     'weighting': 'eligible training family, document, claim and candidate each equal at its level'}
    inventory = {'families': len(group_sizes), 'documents': len(by_document), 'claims': len(comparisons),
                 'candidates': len(standardized)}
    return standardized, positive_features, starts, lengths, weights, normalization, inventory


def listwise_loss(coefficients, features, positive_features, starts, lengths, weights):
    """Weighted uniform-target cross-entropy and its analytic ridge gradient."""
    scores = features @ coefficients
    peaks = np.maximum.reduceat(scores, starts)
    masses = np.exp(scores - np.repeat(peaks, lengths))
    sums = np.add.reduceat(masses, starts)
    loss = weights @ (peaks + np.log(sums) - positive_features @ coefficients)
    marginal = masses * np.repeat(weights / sums, lengths)
    gradient = features.T @ marginal - weights @ positive_features
    return float(loss + 0.5 * RIDGE * (coefficients @ coefficients)), gradient + RIDGE * coefficients


def fit(cases, gold):
    """Fit solely from the supplied training cases and their training annotations."""
    records, documents = index_cases(cases)
    truth = index_gold(gold, records, documents)
    matrices = feature_matrices(records)
    features, positives, starts, lengths, weights, normalization, inventory = training_data(matrices, documents, truth)
    methods = {}
    for method, columns in LEARNED.items():
        result = minimize(listwise_loss, np.zeros(len(columns)),
                          args=(features[:, columns], positives[:, columns], starts, lengths, weights),
                          method='L-BFGS-B', jac=True,
                          options={'maxiter': 250, 'ftol': 1e-13, 'gtol': 1e-9})
        if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError(f'Listwise training failed for {method}: {result.message}')
        methods[method] = {'columns': list(columns), 'coefficients': result.x.tolist(),
                           'objective_nats': float(result.fun), 'iterations': int(result.nit)}
    return {'feature_order': list(FEATURES), 'ridge': RIDGE, 'standardization': normalization,
            'training_inventory': inventory, 'methods': methods}


def forecast(cases, fitted):
    """Apply the training-only artifact to grade-free candidate rows."""
    records, _ = index_cases(cases)
    matrices = feature_matrices(records)
    if (tuple(fitted['feature_order']) != FEATURES or fitted['ridge'] != RIDGE
            or set(fitted['methods']) != set(LEARNED)):
        raise ValueError('Fitted methods do not match the fixed ContractNLI readouts')
    means, scales = (fitted['standardization'][field] for field in ('means', 'scales'))
    if (len(means) != len(FEATURES) or len(scales) != len(FEATURES) or any(not is_number(x) for x in means)
            or any(not is_number(x) or x <= 0 for x in scales)):
        raise ValueError('Expected shared finite training means and positive scales')
    learned = {}
    for method, columns in LEARNED.items():
        settings = fitted['methods'][method]
        vector = settings['coefficients']
        if (any(not is_integer(i) for i in settings['columns']) or tuple(settings['columns']) != columns or len(vector) != len(columns)
                or any(not is_number(x) for x in vector)):
            raise ValueError(f'Unexpected feature columns or coefficients for {method}')
        learned[method] = np.asarray(vector, dtype=np.float64)

    predictions = []
    for (identifier, query), row in sorted(records.items()):
        values = matrices[identifier, query]
        standardized = (values - means) / scales
        scores = {method: values[:, column].tolist() for method, column in BASELINES.items()}
        for method, columns in LEARNED.items():
            result = standardized[:, columns] @ learned[method]
            if not np.isfinite(result).all():
                raise ValueError(f'Nonfinite frozen scores for {identifier}/{query}')
            scores[method] = result.tolist()
        predictions.append({'id': identifier, 'query_id': query, 'group_id': row['group_id'],
                            'candidate_indices': [int(i) for i in row['candidate_indices']], 'methods': scores})
    return predictions


def metric_summary(by_document, documents, families, draws, count, unit):
    present, values = family_means(by_document, documents, families)
    if not values.size:
        return {unit: count, 'documents': 0, 'families': 0, 'valid_bootstrap_draws': 0,
                'methods': {key: {'estimate': None, 'ci95': None} for key in METHODS},
                'fused_centered_minus': {key: {'estimate': None, 'ci95': None} for key in METHODS if key != PRIMARY}}
    weighted = draws[:, present]
    denominator = weighted.sum(axis=1)
    valid = denominator > 0
    replicates = (weighted[valid] @ values) / denominator[valid, None]
    point = values.mean(axis=0)
    center = METHODS.index(PRIMARY)
    return {unit: count, 'documents': len(by_document), 'families': int(present.sum()),
            'valid_bootstrap_draws': int(valid.sum()),
            'methods': {name: estimate(point[i], replicates[:, i]) for i, name in enumerate(METHODS)},
            'fused_centered_minus': {
                name: estimate(point[center] - point[i], replicates[:, center] - replicates[:, i])
                for i, name in enumerate(METHODS) if name != PRIMARY
            }}


def evaluate(forecasts, gold):
    """Score a frozen forecast inventory on complete, aligned official evidence sets."""
    rows, documents = index_cases(forecasts)
    truth = index_gold(gold, rows, documents)
    ap_scores = defaultdict(list)
    ap_by_label = {label: defaultdict(list) for label in POSITIVE_LABELS}
    labels = {label: Counter(annotated_claims=0, claims_with_unaligned_evidence=0, scored_claims=0)
              for label in POSITIVE_LABELS}
    paired = defaultdict(lambda: {label: [] for label in POSITIVE_LABELS})
    pair_coverage = Counter(annotated_entailment_contradiction_pairs=0, overlapping_pairs=0, disjoint_pairs=0,
                            disjoint_pairs_with_unaligned_evidence=0, scored_pairs=0)
    pair_scores = defaultdict(list)
    direction_counts = {method: Counter(wins=0, ties=0, losses=0) for method in METHODS}
    positions = {key: {span: j for j, span in enumerate(info['candidate_indices'])} for key, info in documents.items()}
    for (identifier, query), (label, evidence) in sorted(truth.items()):
        row = rows[identifier, query]
        scores = row['methods']
        indices = positions[identifier]
        if set(scores) != set(METHODS) or any(len(values) != len(indices) or any(not is_number(x) for x in values)
                                             for values in scores.values()):
            raise ValueError(f'Incomplete or nonfinite frozen scores for {identifier}/{query}')
        if label == NO_EVIDENCE:
            continue
        labels[label]['annotated_claims'] += 1
        selected = tuple(indices[i] for i in sorted(evidence) if i in indices)
        matrix = np.asarray([scores[m] for m in METHODS], dtype=np.float64)
        paired[identifier][label].append((evidence, selected, matrix))
        if len(selected) != len(evidence):
            labels[label]['claims_with_unaligned_evidence'] += 1
            continue
        labels[label]['scored_claims'] += 1
        positive = np.zeros(len(indices), dtype=np.bool_)
        positive[list(selected)] = True
        values = [average_precision(candidate_scores, positive) for candidate_scores in matrix]
        ap_scores[identifier].append(values)
        ap_by_label[label][identifier].append(values)

    for identifier, claims in sorted(paired.items()):
        for (evidence_e, e, e_scores), (evidence_c, c, c_scores) in product(*(claims[label] for label in POSITIVE_LABELS)):
            pair_coverage['annotated_entailment_contradiction_pairs'] += 1
            if evidence_e & evidence_c:
                pair_coverage['overlapping_pairs'] += 1
                continue
            pair_coverage['disjoint_pairs'] += 1
            if len(e) != len(evidence_e) or len(c) != len(evidence_c):
                pair_coverage['disjoint_pairs_with_unaligned_evidence'] += 1
                continue
            pair_coverage['scored_pairs'] += 1
            e_direction, c_direction = evidence_order(e_scores, e, c), evidence_order(c_scores, c, e)
            pair_scores[identifier].append(0.5 * (e_direction + c_direction))
            for method, a, b in zip(METHODS, e_direction, c_direction, strict=True):
                direction_counts[method].update({'wins': int(a == 1) + int(b == 1),
                                                 'ties': int(a == .5) + int(b == .5),
                                                 'losses': int(a == 0) + int(b == 0)})

    families = sorted({document['group_id'] for document in documents.values()})
    rng = np.random.default_rng(SEED)
    draws = rng.multinomial(len(families), np.full(len(families), 1 / len(families)), size=DRAWS)
    covered = sum(coverage['scored_claims'] for coverage in labels.values())
    ap = metric_summary(ap_scores, documents, families, draws, covered, 'claims')
    ap_strata = {label: metric_summary(rows, documents, families, draws, labels[label]['scored_claims'], 'claims')
                 for label, rows in ap_by_label.items()}
    pair = metric_summary(pair_scores, documents, families, draws, pair_coverage['scored_pairs'], 'pairs')
    gates = {}
    for name, report, comparator in (('average_precision', ap, 'bm25'), ('average_precision', ap, 'qwen_only'),
                                     ('claim_pair_ordering', pair, 'bm25')):
        bounds = report['fused_centered_minus'][comparator]['ci95']
        gates[f'{name}_fused_centered_over_{comparator}'] = bounds is not None and bounds[0] > 0
    return {'scope': {'documents': len(documents), 'families': len(families), 'claims': len(truth)},
            'average_precision': {'all': ap, 'by_label': ap_strata, 'coverage': labels},
            'claim_pair_ordering': {**pair, 'coverage': pair_coverage, 'raw_direction_counts': direction_counts,
                                    'directions': 2 * pair_coverage['scored_pairs']},
            'bootstrap': {'draws': DRAWS, 'seed': SEED, 'sampling_unit': 'document_family', 'interval': 'paired_percentile'},
            'gates': gates, 'pass': all(gates.values())}
