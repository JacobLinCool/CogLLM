"""Score sealed ExpMRC reference-arrival forecasts without loading a model."""

import argparse
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from hashlib import file_digest
from itertools import combinations
import json
from math import expm1, floor, fsum, log, log1p
from pathlib import Path
from random import Random


METHODS = ('position', 'question_only', 'pooled_rank', 'token_rank')
CANDIDATE = 'token_rank'
REFERENCES = METHODS[:-1]
SEED = 20261121
DRAWS = 10_000
TIE_ATOL = 1e-12
# Registered partial article matches with RACE before any ExpMRC forecast.
PRIOR_MATERIAL_OVERLAP = {'0c4f08f3', '1b332837'}


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def indexed_rows(path, fields):
    records = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict) or row.keys() != fields:
                raise ValueError(f'Unexpected fields in {path.name}')
            identifier = row['id']
            if not isinstance(identifier, str) or not identifier or identifier in records:
                raise ValueError(f'Invalid or duplicate identifier in {path.name}')
            records[identifier] = row
    if not records:
        raise ValueError(f'No records in {path.name}')
    return records


@dataclass(frozen=True)
class Arrival:
    interval_nll: list[float]
    cdf: list[float]

    @classmethod
    def from_hazards(cls, hazards, size):
        if (not isinstance(hazards, list) or len(hazards) != size
                or any(type(h) not in (int, float) or not 0 < h < 1 for h in hazards)):
            raise ValueError('Expected one finite interior hazard per stop')
        log_survival = 0.0
        nll, cdf = [], []
        for hazard in hazards:
            nll.append(-log(hazard) - log_survival)
            log_survival += log1p(-hazard)
            cdf.append(-expm1(log_survival))
        return cls(nll, cdf)


@dataclass(frozen=True)
class ReferenceEndpoints:
    identifier: str
    first_start: int
    ends: tuple[int, int]
    bins: tuple[int, int]


def forecast_inputs(root, forecast_dir, protocol, heads):
    assay = root / 'assay.json'
    seal = forecast_dir / 'seal.json'
    predictions = forecast_dir / 'predictions.jsonl'
    articles, queries = (root / name for name in ('articles.jsonl', 'queries.jsonl'))
    plan = json.loads(assay.read_text(encoding='utf-8'))
    receipt = json.loads(seal.read_text(encoding='utf-8'))
    hashes = {
        'assay_sha256': digest(assay), 'protocol_sha256': digest(protocol), 'heads_sha256': digest(heads),
        'predictions_sha256': digest(predictions), 'articles_sha256': digest(articles),
        'queries_sha256': digest(queries), 'seal_sha256': digest(seal), 'scorer_sha256': digest(Path(__file__)),
    }
    bound = ('assay_sha256', 'protocol_sha256', 'heads_sha256', 'predictions_sha256',
             'articles_sha256', 'queries_sha256')
    if any(receipt.get(field) != hashes[field] for field in bound):
        raise ValueError('ExpMRC forecasts or input artifacts differ from the pre-scoring seal')
    if (plan['file_sha256']['articles.jsonl'] != hashes['articles_sha256']
            or plan['file_sha256']['queries.jsonl'] != hashes['queries_sha256']):
        raise ValueError('ExpMRC article or question inventory differs from the prepared assay')

    articles = indexed_rows(articles, {'id', 'text'})
    queries = indexed_rows(queries, {'id', 'article_id', 'text'})
    forecasts = indexed_rows(predictions, {'id', 'article_id', 'stops', 'methods'})
    if (len(articles) != plan['article_count'] or len(queries) != plan['query_count']
            or forecasts.keys() != queries.keys()):
        raise ValueError('Forecasts must cover every planned ExpMRC question exactly once')
    if any(not isinstance(row['text'], str) or not row['text'].strip()
           for records in (articles, queries) for row in records.values()):
        raise ValueError('Expected a nonempty source article and question text')

    schedules, paths, position = {}, {}, {}
    for identifier, row in forecasts.items():
        article, stops = row['article_id'], row['stops']
        if article != queries[identifier]['article_id'] or article not in articles:
            raise ValueError(f'Article and question identities disagree for {identifier!r}')
        if (not isinstance(stops, list) or not stops or any(type(s) is not int or s < 1 for s in stops)
                or sorted(set(stops)) != stops or stops[-1] != len(articles[article]['text'])
                or schedules.setdefault(article, stops) != stops):
            raise ValueError(f'Expected the complete shared source stop schedule for {article!r}')
        methods = row['methods']
        if not isinstance(methods, dict) or methods.keys() != set(METHODS):
            raise ValueError(f'Unexpected readout methods for {identifier!r}')
        paths[identifier] = {name: Arrival.from_hazards(methods[name], len(stops)) for name in METHODS}
        locked_position = position.setdefault(article, methods['position'])
        if any(abs(a - b) > TIE_ATOL for a, b in zip(methods['position'], locked_position, strict=True)):
            raise ValueError(f'Position baseline depends on question identity for {article!r}')
    if schedules.keys() != articles.keys():
        raise ValueError('Every article must have a complete trajectory')
    return plan, hashes, articles, queries, schedules, paths


def score_endpoints(outcome_file, plan, articles, queries, schedules):
    if digest(outcome_file) != plan['file_sha256']['outcomes.jsonl']:
        raise ValueError('ExpMRC references differ from the sealed held-out outcomes')
    outcomes = indexed_rows(outcome_file, {'id', 'article_id', 'refs'})
    if outcomes.keys() != queries.keys() or plan['reference_count'] != 2 * len(outcomes):
        raise ValueError('The held-out outcome inventory does not match the questions and reference count')
    by_article = defaultdict(list)
    for identifier, outcome in outcomes.items():
        article, refs = outcome['article_id'], outcome['refs']
        if article != queries[identifier]['article_id'] or not isinstance(refs, list) or len(refs) != 2:
            raise ValueError(f'Expected two reference spans for {identifier!r} in its original article')
        length = len(articles[article]['text'])
        if any(not isinstance(ref, list) or len(ref) != 2 or any(type(index) is not int for index in ref)
               or not 0 <= ref[0] < ref[1] <= length for ref in refs):
            raise ValueError(f'Invalid half-open Unicode reference span for {identifier!r}')
        ends = tuple(sorted(end for _, end in refs))
        bins = tuple(bisect_left(schedules[article], end) for end in ends)
        by_article[article].append(ReferenceEndpoints(identifier, min(start for start, _ in refs), ends, bins))
    return by_article


def percentile(sorted_values, probability):
    point = probability * (len(sorted_values) - 1)
    lower = floor(point)
    weight = point - lower
    return (1 - weight) * sorted_values[lower] + weight * sorted_values[min(lower + 1, len(sorted_values) - 1)]


def paired_interval(values, articles):
    clusters = defaultdict(list)
    for value, article in zip(values, articles, strict=True):
        clusters[article].append(value)
    if not clusters:
        return {'difference': None, 'article_cluster_ci95': None, 'articles': 0}
    totals = [(fsum(cluster), len(cluster)) for _, cluster in sorted(clusters.items())]
    interval = None
    if len(totals) >= 2:
        rng = Random(SEED)
        replications = []
        for _ in range(DRAWS):
            effects, counts = zip(*rng.choices(totals, k=len(totals)), strict=True)
            replications.append(fsum(effects) / sum(counts))
        replications.sort()
        interval = [percentile(replications, level) for level in (0.025, 0.975)]
    return {'difference': fsum(values) / len(values), 'article_cluster_ci95': interval, 'articles': len(totals)}


def compare(values, articles):
    return {
        'means': {name: fsum(items) / len(items) if items else None for name, items in values.items()},
        'contrasts': {
            f'{CANDIDATE}_minus_{baseline}': paired_interval(
                [a - b for a, b in zip(values[CANDIDATE], values[baseline], strict=True)], articles,
            ) for baseline in REFERENCES
        },
    }


def nll_at_references(path, endpoints):
    return fsum(path.interval_nll[stop] for stop in endpoints) / len(endpoints)


def question_metrics(by_article, paths):
    nll, brier = ({name: [] for name in METHODS} for _ in range(2))
    articles = []
    identical_ends, identical_bins = 0, 0
    for article, cases in sorted(by_article.items()):
        for case in sorted(cases, key=lambda item: item.identifier):
            articles.append(article)
            identical_ends += case.ends[0] == case.ends[1]
            identical_bins += case.bins[0] == case.bins[1]
            for name, path in paths[case.identifier].items():
                nll[name].append(nll_at_references(path, case.bins))
                brier[name].append(fsum((cdf - int(stop >= end)) ** 2
                                        for end in case.bins for stop, cdf in enumerate(path.cdf))
                                   / (len(case.bins) * len(path.cdf)))
    return {'questions': len(articles), 'articles': len(set(articles)),
            'reference_end_agreement': {'same_character_end': identical_ends,
                                        'same_character_end_fraction': identical_ends / len(articles),
                                        'same_first_covered_stop': identical_bins,
                                        'same_first_covered_stop_fraction': identical_bins / len(articles)},
            'interval_nll_nats': compare(nll, articles), 'cumulative_brier': compare(brier, articles)}


def pair_metrics(by_article, paths):
    accuracy, matched, swapped = ({name: [] for name in METHODS} for _ in range(3))
    articles, overlap, same_stop = [], 0, 0
    for article, cases in sorted(by_article.items()):
        ordered = sorted(cases, key=lambda case: (case.ends[1], case.identifier))
        for early, late in combinations(ordered, 2):
            if early.ends[1] > late.first_start:
                overlap += 1
                continue
            cutoff = early.bins[1]
            if cutoff >= late.bins[0]:
                same_stop += 1
                continue
            articles.append(article)
            for name in METHODS:
                left, right = paths[early.identifier][name], paths[late.identifier][name]
                difference = left.cdf[cutoff] - right.cdf[cutoff]
                accuracy[name].append(0.5 if abs(difference) <= TIE_ATOL else float(difference > 0))
                matched[name].append((nll_at_references(left, early.bins) + nll_at_references(right, late.bins)) / 2)
                swapped[name].append((nll_at_references(right, early.bins) + nll_at_references(left, late.bins)) / 2)
    return {
        'articles': len(set(articles)), 'pairs': len(articles),
        'excluded_pairs_reference_windows_not_strictly_ordered': overlap,
        'excluded_pairs_no_separating_stop': same_stop,
        'same_stop_accuracy': compare(accuracy, articles),
        'same_stop_wins_ties_losses': {
            name: {label: accuracy[name].count(value) for label, value in (('wins', 1), ('ties', 0.5), ('losses', 0))}
            for name in METHODS
        },
        'matched_interval_nll_nats_per_question': {
            name: fsum(values) / len(values) if values else None for name, values in matched.items()
        },
        'swapped_interval_nll_nats_per_question': {
            name: fsum(values) / len(values) if values else None for name, values in swapped.items()
        },
        'matched_minus_swapped_interval_nll_nats': {
            name: paired_interval([a - b for a, b in zip(matched[name], swapped[name], strict=True)], articles)
            for name in METHODS
        },
    }


def acceptance(pairs):
    rank = pairs['same_stop_accuracy']['contrasts']
    position = rank[f'{CANDIDATE}_minus_position']['article_cluster_ci95']
    question = rank[f'{CANDIDATE}_minus_question_only']['article_cluster_ci95']
    swap = pairs['matched_minus_swapped_interval_nll_nats'][CANDIDATE]['article_cluster_ci95']
    conditions = {
        'same_stop_superior_to_position': position is not None and position[0] > 0,
        'same_stop_superior_to_question_only': question is not None and question[0] > 0,
        'matched_question_lower_nll_than_swap': swap is not None and swap[1] < 0,
    }
    return {'conditions': conditions, 'all_conditions_passed': all(conditions.values())}


def run(assay_root, forecast_dir, protocol, heads, output):
    plan, hashes, articles, queries, schedules, paths = forecast_inputs(assay_root, forecast_dir, protocol, heads)
    outcome_path = assay_root / 'outcomes.jsonl'
    by_article = score_endpoints(outcome_path, plan, articles, queries, schedules)
    excluded = sorted(by_article.keys() & PRIOR_MATERIAL_OVERLAP)
    primary = {article: cases for article, cases in by_article.items() if article not in PRIOR_MATERIAL_OVERLAP}
    if not primary:
        raise ValueError('The predeclared independent ExpMRC evaluation has no articles')
    pairs = pair_metrics(primary, paths)
    result = {
        'scope': 'Arrival of retrospectively highlighted supporting text for externally supplied questions.',
        'reference_rule': 'Average the two original annotator targets with equal weight, retaining repeated targets.',
        'pair_nll_rule': 'Mean NLL per question across two swap directions, with two reference targets per question.',
        'bootstrap': {'draws': DRAWS, 'seed': SEED, 'unit': 'held-out article', 'mean': 'equal questions or pairs'},
        'excluded_article_ids': excluded,
        'inventory': {'validated_articles': len(by_article), 'validated_questions': len(queries),
                      'validated_references': 2 * len(queries),
                      'excluded_questions': sum(len(by_article[article]) for article in excluded)},
        'per_question': question_metrics(primary, paths),
        'same_article': pairs,
        'acceptance': acceptance(pairs),
        'all_articles_sensitivity': {'per_question': question_metrics(by_article, paths),
                                     'same_article': pair_metrics(by_article, paths)},
        'input_sha256': {**hashes, 'outcomes_sha256': plan['file_sha256']['outcomes.jsonl']},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('assay-root', 'forecast-dir', 'protocol', 'heads', 'output'):
        parser.add_argument(f'--{option}', type=Path, required=True)
    run(**vars(parser.parse_args()))
