"""Score certified ContractNLI evidence rankings from sealed forecasts."""

from collections import Counter, defaultdict
from itertools import product
from math import isfinite
from numbers import Integral, Real

import numpy as np


FORECAST_METHODS = ('position', 'question_only', 'pooled_rank', 'token_rank')
METHODS = (*FORECAST_METHODS, 'lexical')
POSITIVE_LABELS = ('Entailment', 'Contradiction')
NO_EVIDENCE = 'NotMentioned'
DRAWS = 10_000
SEED = 735972


def is_integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def is_number(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and isfinite(value)


def is_identifier(value):
    return isinstance(value, str) and bool(value.strip())


def read_layout(layout, maps):
    documents, candidate_positions, certified = {}, {}, {}
    official_ids = set()
    for row in layout:
        identifier = row['id']
        official, group, source, spans = (row[field] for field in ('official_id', 'group_id', 'document_type', 'spans'))
        if (not is_identifier(identifier) or identifier in documents or not is_integer(official) or official < 0
                or official in official_ids or not is_identifier(group) or not is_identifier(source)):
            raise ValueError('Layout rows must identify distinct documents in known families')
        previous_end = 0
        if not spans:
            raise ValueError(f'Missing source candidate spans for {identifier}')
        for span in spans:
            if len(span) != 2 or any(not is_integer(value) for value in span):
                raise ValueError(f'Source spans require integer start and end coordinates for {identifier}')
            start, end = span
            if not previous_end <= start < end:
                raise ValueError(f'Source spans must be ordered and disjoint for {identifier}')
            previous_end = end
        if identifier not in maps or len(maps[identifier]) != len(spans):
            raise ValueError(f'Missing or unaligned candidate map for {identifier}')
        previous_stop = -1
        mapped = []
        for candidate, stop in enumerate(maps[identifier]):
            if stop is None:
                continue
            if not is_integer(stop) or stop <= previous_stop:
                raise ValueError(f'Certified candidates need distinct ordered stop indices for {identifier}')
            previous_stop = stop
            mapped.append((candidate, int(stop)))
        documents[identifier] = row
        official_ids.add(official)
        certified[identifier] = mapped
        candidate_positions[identifier] = {index: position for position, (index, _) in enumerate(mapped)}
    if not documents or documents.keys() != maps.keys():
        raise ValueError('Expected one candidate map per layout document')
    return documents, certified, candidate_positions


def read_gold(gold, documents):
    claims = {}
    for row in gold:
        identifier, query, label = (row[key] for key in ('id', 'query_id', 'label'))
        if not is_identifier(identifier) or not is_identifier(query):
            raise ValueError('Gold rows require document and query identifiers')
        document = documents.get(identifier)
        key = (identifier, query)
        if (document is None or key in claims or not is_integer(row['official_id'])
                or any(row[field] != document[field] for field in ('official_id', 'group_id'))
                or label not in (*POSITIVE_LABELS, NO_EVIDENCE)):
            raise ValueError('Gold rows must uniquely identify a known document and claim with its official label')
        indices = row['evidence_indices']
        if (any(not is_integer(i) or not 0 <= i < len(document['spans']) for i in indices)
                or len(indices) != len(set(indices)) or bool(indices) != (label in POSITIVE_LABELS)):
            raise ValueError(f'Unexpected evidence indices for {identifier}/{query}')
        claims[key] = label, tuple(sorted(indices))
    if {identifier for identifier, _ in claims} != documents.keys():
        raise ValueError('Gold claim inventory does not cover the layout documents')
    return claims


def read_scores(forecasts, documents, certified, claims, lexical):
    scores, document_stops = {}, {}
    if lexical.keys() != claims.keys():
        raise ValueError('Lexical score inventory must match the gold claim inventory')
    for row in forecasts:
        identifier, query, stops, methods = (row[field] for field in ('id', 'query_id', 'stops', 'methods'))
        if not is_identifier(identifier) or not is_identifier(query):
            raise ValueError('Forecast rows require document and query identifiers')
        key = (identifier, query)
        if key not in claims or key in scores or set(methods) != set(FORECAST_METHODS):
            raise ValueError('Forecasts must provide all frozen methods once per known document and claim')
        if (not stops or any(not is_integer(i) or i <= 0 for i in stops)
                or any(left >= right for left, right in zip(stops, stops[1:]))):
            raise ValueError(f'Forecast stops must be strictly increasing positive integers for {identifier}/{query}')
        if document_stops.setdefault(identifier, tuple(stops)) != tuple(stops):
            raise ValueError(f'Claim forecasts disagree on certified stops for {identifier}')
        spans = documents[identifier]['spans']
        mapped = certified[identifier]
        if any(stop >= len(stops) or not spans[index][0] < stops[stop] <= spans[index][1]
               for index, stop in mapped):
            raise ValueError(f'Candidate map points outside a certified source span for {identifier}')
        if any(len(values) != len(stops) or any(not is_number(value) or not 0 <= value <= 1 for value in values)
               for values in methods.values()):
            raise ValueError(f'Invalid hazard values or stop alignment for {identifier}/{query}')
        lexical_scores = lexical[key]
        if len(lexical_scores) != len(spans) or any(not is_number(lexical_scores[i]) for i, _ in mapped):
            raise ValueError(f'Expected a finite lexical score at every certified candidate for {identifier}/{query}')
        scores[key] = np.asarray(
            [[methods[method][stop] for _, stop in mapped] for method in FORECAST_METHODS]
            + [[lexical_scores[i] for i, _ in mapped]], dtype=np.float64,
        )
    if scores.keys() != claims.keys():
        raise ValueError('Forecast and gold claim inventories differ')
    return scores


def average_precision(scores, positive):
    """Group exact score ties before updating precision and recall."""
    order = np.argsort(-scores, kind='stable')
    ranked = scores[order]
    ties = np.r_[0, np.flatnonzero(ranked[1:] != ranked[:-1]) + 1]
    true_positive = np.add.reduceat(positive[order], ties)
    rank = np.r_[ties[1:], len(order)]
    return float(np.dot(np.cumsum(true_positive) / rank, true_positive) / true_positive.sum())


def family_means(by_document, documents, families):
    by_family = defaultdict(list)
    for identifier in sorted(by_document):
        by_family[documents[identifier]['group_id']].append(np.mean(by_document[identifier], axis=0))
    have_values = np.array([name in by_family for name in families], dtype=np.bool_)
    means = np.asarray([np.mean(by_family[name], axis=0) for name in families if name in by_family], dtype=np.float64)
    return have_values, means


def metric_summary(by_document, documents, families, family_draws, count, unit):
    present, values = family_means(by_document, documents, families)
    result = {unit: count, 'documents': len(by_document), 'families': int(present.sum())}
    methods = {}
    comparisons = {}
    if values.size:
        weights = family_draws[:, present]
        denominator = weights.sum(axis=1)
        valid = denominator > 0
        draws = (weights[valid] @ values) / denominator[valid, None]
        point = values.mean(axis=0)
        token = METHODS.index('token_rank')
        for index, method in enumerate(METHODS):
            methods[method] = estimate(point[index], draws[:, index])
            if method != 'token_rank':
                comparisons[method] = estimate(point[token] - point[index], draws[:, token] - draws[:, index])
    else:
        valid = np.zeros(len(family_draws), dtype=np.bool_)
        methods = {method: {'estimate': None, 'ci95': None} for method in METHODS}
        comparisons = {method: {'estimate': None, 'ci95': None} for method in METHODS if method != 'token_rank'}
    return {**result, 'valid_bootstrap_draws': int(valid.sum()), 'methods': methods, 'token_rank_minus': comparisons}


def estimate(point, draws):
    return {'estimate': float(point), 'ci95': np.quantile(draws, (0.025, 0.975)).tolist() if len(draws) else None}


def evidence_order(own_scores, own, other):
    """Compare two evidence-set means under one unchanged claim."""
    ours, theirs = (own_scores[:, index].mean(axis=1) for index in (own, other))
    return (ours > theirs).astype(np.float64) + 0.5 * (ours == theirs)


def evaluate(layout, gold, forecasts, maps, lexical):
    """Aggregate matched evidence rankings by claim, document and document family."""
    documents, certified, candidate_positions = read_layout(layout, maps)
    claims = read_gold(gold, documents)
    scores = read_scores(forecasts, documents, certified, claims, lexical)
    by_label = {label: defaultdict(list) for label in POSITIVE_LABELS}
    ap_documents = defaultdict(list)
    pair_documents = defaultdict(list)
    ap_fields = ('annotated_claims', 'annotated_evidence_spans', 'certified_evidence_spans',
                 'claims_with_unaligned_evidence', 'scored_claims', 'scored_candidate_spans',
                 'claims_without_certified_distractors')
    pair_fields = ('annotated_entailment_contradiction_pairs', 'overlapping_pairs', 'disjoint_pairs',
                   'disjoint_pairs_with_unaligned_evidence', 'scored_pairs')
    ap_coverage = {label: Counter(dict.fromkeys(ap_fields, 0)) for label in POSITIVE_LABELS}
    pair_coverage = Counter(dict.fromkeys(pair_fields, 0))
    pair_directions = {method: Counter(wins=0, ties=0, losses=0) for method in METHODS}
    document_claims = defaultdict(lambda: {label: [] for label in POSITIVE_LABELS})

    for (identifier, query), (label, evidence) in sorted(claims.items()):
        if label == NO_EVIDENCE:
            continue
        positions = candidate_positions[identifier]
        aligned = tuple(positions[i] for i in evidence if i in positions)
        coverage = ap_coverage[label]
        coverage['annotated_claims'] += 1
        coverage['annotated_evidence_spans'] += len(evidence)
        coverage['certified_evidence_spans'] += len(aligned)
        document_claims[identifier][label].append((query, set(evidence), aligned))
        if len(aligned) != len(evidence):
            coverage['claims_with_unaligned_evidence'] += 1
            continue
        coverage['scored_claims'] += 1
        coverage['scored_candidate_spans'] += len(positions)
        coverage['claims_without_certified_distractors'] += len(aligned) == len(positions)
        positive = np.zeros(len(positions), dtype=np.bool_)
        positive[list(aligned)] = True
        metrics = [average_precision(values, positive) for values in scores[identifier, query]]
        by_label[label][identifier].append(metrics)
        ap_documents[identifier].append(metrics)

    for identifier, labeled in sorted(document_claims.items()):
        for (e_query, e_source, e), (c_query, c_source, c) in product(*(labeled[label] for label in POSITIVE_LABELS)):
            pair_coverage['annotated_entailment_contradiction_pairs'] += 1
            if e_source & c_source:
                pair_coverage['overlapping_pairs'] += 1
                continue
            pair_coverage['disjoint_pairs'] += 1
            if len(e) != len(e_source) or len(c) != len(c_source):
                pair_coverage['disjoint_pairs_with_unaligned_evidence'] += 1
                continue
            pair_coverage['scored_pairs'] += 1
            first = evidence_order(scores[identifier, e_query], e, c)
            second = evidence_order(scores[identifier, c_query], c, e)
            pair_documents[identifier].append((first + second) / 2)
            for method, a, b in zip(METHODS, first, second, strict=True):
                pair_directions[method].update({
                    'wins': int(a == 1) + int(b == 1), 'ties': int(a == 0.5) + int(b == 0.5),
                    'losses': int(a == 0) + int(b == 0),
                })

    families = sorted({doc['group_id'] for doc in documents.values()})
    rng = np.random.default_rng(SEED)
    family_draws = rng.multinomial(len(families), np.full(len(families), 1 / len(families)), size=DRAWS)
    ap_count = sum(count['scored_claims'] for count in ap_coverage.values())
    ap = metric_summary(ap_documents, documents, families, family_draws, ap_count, 'claims')
    by_label_scores = {
        label: metric_summary(by_label[label], documents, families, family_draws, ap_coverage[label]['scored_claims'], 'claims')
        for label in POSITIVE_LABELS
    }
    pair = metric_summary(pair_documents, documents, families, family_draws, pair_coverage['scored_pairs'], 'pairs')
    gates = {}
    for name, report in (('average_precision', ap), ('claim_pair_ordering', pair)):
        for comparator in ('lexical', 'pooled_rank'):
            interval = report['token_rank_minus'][comparator]['ci95']
            gates[f'{name}_token_rank_over_{comparator}'] = interval is not None and interval[0] > 0
    claims_by_label = Counter(label for label, _ in claims.values())
    return {
        'scope': {
            'documents': len(documents), 'families': len(families), 'claims': len(claims),
            'claims_by_label': {label: claims_by_label[label] for label in (*POSITIVE_LABELS, NO_EVIDENCE)},
            'source_candidate_spans': sum(len(doc['spans']) for doc in documents.values()),
            'certified_candidate_spans': sum(map(len, certified.values())),
        },
        'average_precision': {'all': ap, 'by_label': by_label_scores, 'coverage': ap_coverage},
        'claim_pair_ordering': {
            **pair, 'coverage': pair_coverage, 'raw_direction_counts': pair_directions,
            'directions': 2 * pair_coverage['scored_pairs'],
        },
        'bootstrap': {'draws': DRAWS, 'seed': SEED, 'sampling_unit': 'document_family', 'interval': 'paired_percentile'},
        'gates': gates, 'pass': all(gates.values()),
    }
