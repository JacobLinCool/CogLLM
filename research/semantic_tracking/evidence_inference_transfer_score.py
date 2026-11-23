"""Evaluate sealed Evidence Inference rankings against exact source references."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from research.semantic_tracking.contractnli_score import (
    DRAWS,
    SEED,
    estimate,
    evidence_order,
    is_identifier,
    is_integer,
    is_number,
)


METHODS = ('bm25_causal', 'bm25_unit', 'token_rank', 'centered_token_rank',
           'qwen_only', 'fused_raw', 'fused_centered')
PRIMARY = 'fused_centered'


@dataclass(frozen=True)
class Article:
    text: str
    spans: tuple[tuple[int, int], ...]
    ends: tuple[int, ...]
    candidate_indices: tuple[int, ...]
    queries: tuple[str, ...]


def interval_bounds(value, text_length, identity):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or any(not is_integer(i) for i in value)
            or not 0 <= value[0] < value[1] <= text_length):
        raise ValueError(f'Expected a nonempty half-open source interval for {identity}')
    return int(value[0]), int(value[1])


def index_articles(articles, layout):
    texts, documents = {}, {}
    for row in articles:
        identifier, text = row['id'], row['text']
        if not is_identifier(identifier) or identifier in texts or not isinstance(text, str) or not text.strip():
            raise ValueError('Articles need unique identifiers and nonempty original source text')
        texts[identifier] = text
    for row in layout:
        identifier = row['id']
        if not is_identifier(identifier) or identifier not in texts or identifier in documents:
            raise ValueError('Expected a single sentence layout for each known article')
        text = texts[identifier]
        spans = tuple(interval_bounds(s, len(text), identifier) for s in row['spans'])
        stops = row['candidate_stops']
        queries = tuple(row['query_ids'])
        if (not spans or len(stops) != len(spans)
                or any(right > next_left for (_, right), (next_left, _) in zip(spans, spans[1:]))):
            raise ValueError(f'Sentence spans must be ordered, disjoint and aligned to the candidate map for {identifier}')
        if not queries or any(not is_identifier(query) for query in queries) or len(set(queries)) != len(queries):
            raise ValueError(f'Expected distinct query identifiers in the layout for {identifier}')
        selected = []
        previous = -1
        for index, stop in enumerate(stops):
            if stop is None:
                continue
            left, right = spans[index]
            if not is_integer(stop) or stop <= previous or not text[left:right].strip():
                raise ValueError(f'Certified sentence indices must be ordered and refer to visible source for {identifier}')
            previous = stop
            selected.append(index)
        documents[identifier] = Article(text, spans, tuple(end for _, end in spans), tuple(selected), queries)
    if not documents or documents.keys() != texts.keys():
        raise ValueError('Sentence layout must cover every supplied article')
    return documents


def index_predictions(predictions, documents, expected):
    scores = {}
    for row in predictions:
        identifier, query = row['id'], row['query_id']
        if not is_identifier(identifier) or not is_identifier(query):
            raise ValueError('Predictions need article and query identifiers')
        key = identifier, query
        if key not in expected or key in scores:
            raise ValueError('Expected exactly one forecast for each registered article and query')
        indices, methods = tuple(row['candidate_indices']), row['methods']
        if (any(not is_integer(i) for i in indices) or indices != documents[identifier].candidate_indices
                or set(methods) != set(METHODS)
                or any(len(values) != len(indices) or any(not is_number(x) for x in values) for values in methods.values())):
            raise ValueError(f'Unaligned or nonfinite candidate scores for {identifier}/{query}')
        scores[key] = np.asarray([methods[m] for m in METHODS], dtype=np.float64)
    if scores.keys() != expected:
        raise ValueError('Predictions must cover every registered article and query')
    return scores


def index_references(outcomes, documents, expected):
    references = {}
    for row in outcomes:
        identifier, query = row['article_id'], row['id']
        if not is_identifier(identifier) or not is_identifier(query):
            raise ValueError('References need article and query identifiers')
        key = identifier, query
        if key not in expected or key in references:
            raise ValueError('Expected one source reference list per registered article and query')
        references[key] = tuple(interval_bounds(ref, len(documents[identifier].text), f'{identifier}/{query}')
                                for ref in row['references'])
    if references.keys() != expected:
        raise ValueError('Reference lists must cover every registered article and query')
    return references


def reference_candidates(article, certified, reference):
    """Return all intersecting certified sentences if they cover every visible reference character."""
    start, end = reference
    text = article.text
    if not text[start:end].strip():
        return (), 'whitespace_only'
    selected = []
    cursor = start
    first = bisect_right(article.ends, start)
    for index in range(first, len(article.spans)):
        left, right = article.spans[index]
        if left >= end:
            break
        begin, stop = max(start, left), min(end, right)
        if index not in certified or not text[begin:stop].strip():
            continue
        if text[cursor:begin].strip():
            return (), 'uncovered_non_whitespace'
        selected.append(index)
        cursor = stop
    if not selected or text[cursor:end].strip():
        return (), 'uncovered_non_whitespace'
    return tuple(selected), None


def source_order_average_precision(scores, positive):
    """Compute AP with ties ordered by original candidate position."""
    ordered = positive[np.argsort(-scores, kind='stable')]
    ranks = np.flatnonzero(ordered) + 1
    return float(np.mean(np.arange(1, len(ranks) + 1) / ranks))


def metric_summary(by_article, counts):
    eligible = sorted(by_article)
    if not eligible:
        missing = {m: {'estimate': None, 'ci95': None} for m in METHODS}
        return {**counts, 'articles': 0, 'valid_bootstrap_draws': 0, 'methods': missing,
                'fused_centered_minus': {m: value for m, value in missing.items() if m != PRIMARY}}
    values = np.asarray([np.mean(by_article[key], axis=0) for key in eligible])
    size = len(eligible)
    rng = np.random.default_rng(SEED)
    draws = rng.multinomial(size, np.full(size, 1 / size), size=DRAWS)
    replicas = draws @ values / size
    point = values.mean(axis=0)
    center = METHODS.index(PRIMARY)
    return {**counts, 'articles': size, 'valid_bootstrap_draws': DRAWS,
            'methods': {m: estimate(point[i], replicas[:, i]) for i, m in enumerate(METHODS)},
            'fused_centered_minus': {
                m: estimate(point[center] - point[i], replicas[:, center] - replicas[:, i])
                for i, m in enumerate(METHODS) if m != PRIMARY
            }}


def evaluate(articles, layout, outcomes, predictions):
    """Compute article-macro reference AP and within-article evidence-set ordering."""
    documents = index_articles(articles, layout)
    expected = {(key, query) for key, document in documents.items() for query in document.queries}
    scores = index_predictions(predictions, documents, expected)
    references = index_references(outcomes, documents, expected)
    ap_by_article, pairs_by_article = {}, {}
    references_coverage = Counter(reference_occurrences=0, repeated_reference_occurrences=0,
                                  mapped_reference_occurrences=0, unalignable_reference_occurrences=0,
                                  whitespace_only=0, uncovered_non_whitespace=0)
    query_coverage = Counter(queries=0, queries_without_references=0, queries_with_any_mapped_reference=0,
                             fully_mapped_queries=0, partially_mapped_queries=0, queries_with_no_mapped_reference=0)
    pair_coverage = Counter(within_article_query_pairs=0, pairs_without_two_fully_mapped_queries=0,
                            fully_mapped_pairs=0, overlapping_candidate_unions=0, scored_disjoint_pairs=0)
    directions = {method: Counter(wins=0, ties=0, losses=0) for method in METHODS}

    for identifier, document in sorted(documents.items()):
        positions = {i: j for j, i in enumerate(document.candidate_indices)}
        cache, full_unions = {}, {}
        query_ap, pair_scores = [], []
        for query in sorted(document.queries):
            refs = references[identifier, query]
            query_coverage['queries'] += 1
            if not refs:
                query_coverage['queries_without_references'] += 1
                continue
            references_coverage['reference_occurrences'] += len(refs)
            references_coverage['repeated_reference_occurrences'] += len(refs) - len(set(refs))
            values, union = [], set()
            for ref in refs:
                if ref not in cache:
                    cache[ref] = reference_candidates(document, positions, ref)
                selected, reason = cache[ref]
                if reason is not None:
                    references_coverage['unalignable_reference_occurrences'] += 1
                    references_coverage[reason] += 1
                    continue
                references_coverage['mapped_reference_occurrences'] += 1
                aligned = [positions[i] for i in selected]
                positive = np.zeros(len(positions), dtype=np.bool_)
                positive[aligned] = True
                values.append([source_order_average_precision(scores_row, positive)
                               for scores_row in scores[identifier, query]])
                union.update(aligned)
            if not values:
                query_coverage['queries_with_no_mapped_reference'] += 1
                continue
            query_coverage['queries_with_any_mapped_reference'] += 1
            query_ap.append(np.mean(values, axis=0))
            if len(values) == len(refs):
                query_coverage['fully_mapped_queries'] += 1
                full_unions[query] = union
            else:
                query_coverage['partially_mapped_queries'] += 1
        if query_ap:
            ap_by_article[identifier] = query_ap

        for left, right in combinations(sorted(document.queries), 2):
            pair_coverage['within_article_query_pairs'] += 1
            if left not in full_unions or right not in full_unions:
                pair_coverage['pairs_without_two_fully_mapped_queries'] += 1
                continue
            pair_coverage['fully_mapped_pairs'] += 1
            a, b = full_unions[left], full_unions[right]
            if a & b:
                pair_coverage['overlapping_candidate_unions'] += 1
                continue
            pair_coverage['scored_disjoint_pairs'] += 1
            a, b = tuple(sorted(a)), tuple(sorted(b))
            first = evidence_order(scores[identifier, left], a, b)
            second = evidence_order(scores[identifier, right], b, a)
            pair_scores.append((first + second) / 2)
            for method, own, swap in zip(METHODS, first, second, strict=True):
                directions[method].update({
                    'wins': int(own == 1) + int(swap == 1), 'ties': int(own == .5) + int(swap == .5),
                    'losses': int(own == 0) + int(swap == 0),
                })
        if pair_scores:
            pairs_by_article[identifier] = pair_scores

    ap = metric_summary(ap_by_article,
                        {'mapped_reference_occurrences': references_coverage['mapped_reference_occurrences'],
                         'queries': query_coverage['queries_with_any_mapped_reference']})
    pair = metric_summary(pairs_by_article, {'pairs': pair_coverage['scored_disjoint_pairs']})
    gates = {}
    for endpoint, report, method in (('average_precision', ap, 'bm25_causal'), ('average_precision', ap, 'bm25_unit'),
                                     ('claim_pair_ordering', pair, 'bm25_unit')):
        interval = report['fused_centered_minus'][method]['ci95']
        gates[f'{endpoint}_fused_centered_over_{method}'] = interval is not None and interval[0] > 0
    return {'scope': {'articles': len(documents), 'queries': len(expected),
                      'sentence_candidates': sum(len(doc.spans) for doc in documents.values()),
                      'certified_sentence_candidates': sum(len(doc.candidate_indices) for doc in documents.values())},
            'coverage': {'references': references_coverage, 'queries': query_coverage, 'pairs': pair_coverage},
            'average_precision': ap,
            'claim_pair_ordering': {**pair, 'raw_direction_counts': directions,
                                    'directions': 2 * pair_coverage['scored_disjoint_pairs']},
            'bootstrap': {'draws': DRAWS, 'seed': SEED, 'sampling_unit': 'article',
                          'population': 'eligible_articles_per_endpoint', 'interval': 'paired_percentile'},
            'gates': gates, 'pass': all(gates.values())}


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('articles', 'layout', 'outcomes', 'predictions', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    predictions = read_jsonl(args.predictions)
    report = evaluate(read_jsonl(args.articles), read_jsonl(args.layout), read_jsonl(args.outcomes), predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
