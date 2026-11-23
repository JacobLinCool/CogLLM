"""Score sealed Natural Questions evidence-arrival forecasts on one CPU thread."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
from math import exp, expm1, fsum, isclose, isfinite, log, log1p
from pathlib import Path

from expmrc_forecast import HEADS_SHA256
from expmrc_score import Arrival, DRAWS, METHODS, SEED, digest, indexed_rows, paired_interval
from nq_source_positions import TextView


CALIBRATION = Path(__file__).resolve().parents[1] / 'results' / 'evidence-calibration-comparison.json'
CALIBRATION_SHA256 = 'efab56f7379a00b6b468a82c5a0ac8ad34aa8fe9d1189f434e995bd27bf1f944'
PROTOCOL = Path(__file__).with_name('nq-evidence-transfer.md')
REFERENCE_COUNT = 5
LONG_FIELDS = {'start_token', 'end_token', 'start_byte', 'end_byte', 'candidate_index'}
MAP_FIELDS = {
    'id', 'article_id', 'row_index', 'source_oldid', 'document_title', 'title_family',
    'source_positions', 'character_spans', 'byte_spans', 'candidate_spans',
}
MODES = ('raw', 'calibrated')


@dataclass(frozen=True)
class PathForecast:
    arrival: Arrival
    censored_nll: float
    log_intensity: list[float]


@dataclass(frozen=True)
class Box:
    source_start: int
    source_end: int
    start: int
    stop: int
    source_tag: str


@dataclass(frozen=True)
class Case:
    identifier: str
    family: str
    references: tuple[Box | None, ...]

    @property
    def boxes(self):
        return tuple(ref for ref in self.references if ref is not None)


def frozen_calibrators():
    artifact = json.loads(CALIBRATION.read_text(encoding='utf-8'))
    weights = artifact['frozen_exp_mrc_transfer_weights']['methods']
    if set(weights) != set(METHODS):
        raise ValueError('Expected one frozen ExpMRC calibration for each readout')
    parameters = {method: (row['a'], row['b']) for method, row in weights.items()}
    if any(not isfinite(a) or not isfinite(b) or b <= 0 for a, b in parameters.values()):
        raise ValueError('Expected finite ExpMRC calibration with strictly positive shape')
    return parameters


def forecast_paths(hazards, size, parameters):
    raw = Arrival.from_hazards(hazards, size)
    a, b = parameters
    raw_intensity = adjusted_previous = 0.0
    raw_log_intensity, adjusted_log_intensity, adjusted_nll, adjusted_cdf = [], [], [], []
    for hazard in hazards:
        raw_increment = -log1p(-hazard)
        log_ratio = log1p(raw_increment / raw_intensity) if raw_intensity else float('inf')
        raw_intensity += raw_increment
        raw_log_intensity.append(log(raw_intensity))
        adjusted_log_intensity.append(a + b * raw_log_intensity[-1])
        adjusted = exp(adjusted_log_intensity[-1])
        increment = adjusted * -expm1(-b * log_ratio)
        if increment <= 0 or not isfinite(increment):
            raise ValueError('Frozen calibration failed to produce increasing cumulative intensity')
        adjusted_nll.append(adjusted_previous - log(-expm1(-increment)))
        adjusted_cdf.append(-expm1(-adjusted))
        adjusted_previous = adjusted
    mass = fsum(exp(-value) for value in adjusted_nll) + exp(-adjusted_previous)
    if not isclose(mass, 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Calibrated intervals and residual survival do not sum to one')
    return {
        'raw': PathForecast(raw, raw_intensity, raw_log_intensity),
        'calibrated': PathForecast(Arrival(adjusted_nll, adjusted_cdf), adjusted_previous, adjusted_log_intensity),
    }


def read_forecasts(root, predictions_dir):
    assay = root / 'assay.json'
    predictions = predictions_dir / 'predictions.jsonl'
    seal = predictions_dir / 'seal.json'
    article_file, query_file, maps = (root / name for name in ('articles.jsonl', 'queries.jsonl', 'span-maps.jsonl'))
    files = {
        'assay_sha256': assay, 'articles_sha256': article_file, 'queries_sha256': query_file,
        'predictions_sha256': predictions, 'seal_sha256': seal, 'span_maps_sha256': maps,
        'calibration_sha256': CALIBRATION, 'scorer_sha256': Path(__file__), 'protocol_sha256': PROTOCOL,
    }
    hashes = {key: digest(path) for key, path in files.items()}
    receipt = json.loads(seal.read_text(encoding='utf-8'))
    plan = json.loads(assay.read_text(encoding='utf-8'))
    for key in ('assay_sha256', 'articles_sha256', 'queries_sha256', 'predictions_sha256', 'protocol_sha256'):
        if receipt.get(key) != hashes[key]:
            raise ValueError(f'Forecast seal differs from its input: {key}')
    for key in ('articles_sha256', 'queries_sha256', 'span_maps_sha256'):
        if plan['file_sha256'].get(files[key].name) != hashes[key]:
            raise ValueError(f'Sealed assay differs from its input: {files[key].name}')
    if hashes['calibration_sha256'] != CALIBRATION_SHA256 or receipt.get('heads_sha256') != HEADS_SHA256:
        raise ValueError('Frozen readout or calibration differs from the registered transfer artifacts')
    hashes['heads_sha256'] = HEADS_SHA256

    articles = indexed_rows(article_file, {'id', 'text'})
    queries = indexed_rows(query_file, {'id', 'article_id', 'text'})
    forecasts = indexed_rows(predictions, {'id', 'article_id', 'stops', 'methods'})
    if (len(articles) != plan['article_count'] or len(queries) != plan['query_count']
            or forecasts.keys() != queries.keys()):
        raise ValueError('Forecasts do not cover every planned Natural Questions record')
    if any(not isinstance(row['text'], str) or not row['text'].strip()
           for records in (articles, queries) for row in records.values()):
        raise ValueError('Expected nonempty Natural Questions text')

    calibrators = frozen_calibrators()
    schedules, positions, paths = {}, {}, {}
    for identifier, row in forecasts.items():
        article, stops = row['article_id'], row['stops']
        if article != queries[identifier]['article_id'] or article not in articles:
            raise ValueError(f'Article and question identities disagree for {identifier!r}')
        if (not isinstance(stops, list) or not stops or any(type(stop) is not int or stop <= 0 for stop in stops)
                or sorted(set(stops)) != stops or stops[-1] != len(articles[article]['text'])
                or schedules.setdefault(article, stops) != stops):
            raise ValueError(f'Invalid certified source stop schedule for {article!r}')
        methods = row['methods']
        if not isinstance(methods, dict) or methods.keys() != set(METHODS):
            raise ValueError(f'Unexpected readout methods for {identifier!r}')
        paths[identifier] = {method: forecast_paths(methods[method], len(stops), calibrators[method])
                             for method in METHODS}
        locked_position = positions.setdefault(article, methods['position'])
        if methods['position'] != locked_position:
            raise ValueError(f'Position baseline depends on question identity for {article!r}')
    if schedules.keys() != articles.keys():
        raise ValueError('Expected a complete forecast for every article')
    return plan, hashes, articles, queries, schedules, paths


def reference_box(reference, view, candidates, stops):
    if (not isinstance(reference, dict) or reference.keys() != LONG_FIELDS
            or any(type(value) is not int for value in reference.values())):
        raise ValueError('malformed_long_answer')
    if all(value == -1 for value in reference.values()):
        return None
    index = reference['candidate_index']
    if not 0 <= index < len(candidates):
        raise ValueError('invalid_candidate_index')
    candidate = candidates[index]
    coordinates = ('start_token', 'end_token', 'start_byte', 'end_byte')
    if any(reference[key] != candidate[key] for key in coordinates):
        raise ValueError('candidate_coordinates_disagree')
    try:
        span = view.support(*(reference[key] for key in coordinates))
    except ValueError as error:
        raise ValueError('invalid_source_coordinates') from error
    if span is None:
        raise ValueError('no_visible_long_answer')
    if tuple(candidate['character_span'] or ()) != span:
        raise ValueError('candidate_text_map_disagrees')
    start, end = span
    if not 0 <= start < end <= len(view.text):
        raise ValueError('mapped_span_outside_forecast_text')
    stop = bisect_left(stops, end)
    if stop == len(stops):
        raise ValueError('no_certified_complete_stop')
    return Box(reference['start_token'], reference['end_token'], start, stop, candidate['source_tag'])


def annotation_refs(row):
    annotations = row['annotations']
    if not isinstance(annotations, dict):
        raise ValueError('malformed_annotation_set')
    ids, refs = annotations.get('id'), annotations.get('long_answer')
    if not isinstance(ids, list) or not isinstance(refs, list) or len(ids) != len(refs) or len(refs) != REFERENCE_COUNT:
        raise ValueError('expected_five_annotations')
    if any(not isinstance(identifier, str) or not identifier for identifier in ids) or len(set(ids)) != len(ids):
        raise ValueError('invalid_annotation_identifiers')
    return refs


def read_sealed_outcomes(path, plan, queries, hashes):
    manifest_file = path.with_name('manifest.json')
    hashes['raw_manifest_sha256'] = digest(manifest_file)
    if hashes['raw_manifest_sha256'] != plan['raw_manifest_sha256']:
        raise ValueError('Raw Natural Questions manifest differs from the sealed assay receipt')
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    hashes['outcomes_sha256'] = digest(path)
    if hashes['outcomes_sha256'] != manifest['file_sha256']['outcomes.jsonl']:
        raise ValueError('Annotation escrow differs from the sealed raw manifest')
    selected = manifest['selection']['rows']
    selected_by_id = {row['id']: row for row in selected}
    outcomes = indexed_rows(path, {'id', 'annotations'})
    if len(selected_by_id) != len(selected) or outcomes.keys() != selected_by_id.keys() or not queries.keys() <= outcomes.keys():
        raise ValueError('Annotation escrow and sealed question inventories disagree')
    return selected_by_id, outcomes


def references_by_article(root, selected, outcomes, articles, queries, schedules):
    by_article = defaultdict(list)
    inventory, exclusion_reasons, source_tags = Counter(), Counter(), Counter()
    seen, family_by_article = set(), {}
    with (root / 'span-maps.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict) or row.keys() != MAP_FIELDS:
                raise ValueError('Unexpected span-map fields')
            identifier, article, family = row['id'], row['article_id'], row['title_family']
            if (identifier in seen or identifier not in queries or article != queries[identifier]['article_id']
                    or row['row_index'] != selected[identifier]['row_index']
                    or row['source_oldid'] != selected[identifier]['revision']
                    or article != f'nq-wiki-{row["source_oldid"]}'
                    or not isinstance(family, str) or not family or family_by_article.setdefault(article, family) != family):
                raise ValueError(f'Span-map identity or article family disagrees for {identifier!r}')
            seen.add(identifier)
            view = TextView(articles[article]['text'], tuple(row['source_positions']),
                            tuple(map(tuple, row['character_spans'])), tuple(map(tuple, row['byte_spans'])))
            try:
                source_refs = annotation_refs(outcomes[identifier])
            except ValueError as error:
                exclusion_reasons[str(error)] += 1
                inventory['excluded_questions'] += 1
                continue
            refs = []
            for reference in source_refs:
                try:
                    refs.append(reference_box(reference, view, row['candidate_spans'], schedules[article]))
                except ValueError as error:
                    exclusion_reasons[str(error)] += 1
                    inventory['invalid_long_answers'] += 1
            if len(refs) != REFERENCE_COUNT:
                inventory['excluded_questions'] += 1
                continue
            case = Case(identifier, family, tuple(refs))
            by_article[article].append(case)
            source_tags.update(box.source_tag for box in case.boxes)
            inventory['scored_questions'] += 1
            inventory['observed_long_answers'] += len(case.boxes)
            inventory['censored_long_answers'] += REFERENCE_COUNT - len(case.boxes)
    if seen != queries.keys():
        raise ValueError('Expected one span map for every sealed query')
    if not by_article:
        raise ValueError(f'No questions have five valid annotated references: {dict(exclusion_reasons)}')
    return by_article, {
        'validated_questions': len(seen), 'scored_questions': inventory['scored_questions'],
        'excluded_questions': inventory['excluded_questions'], 'invalid_long_answers': inventory['invalid_long_answers'],
        'observed_long_answers': inventory['observed_long_answers'], 'censored_long_answers': inventory['censored_long_answers'],
        'non_null_source_tags': dict(source_tags), 'exclusion_reasons': dict(exclusion_reasons),
    }


def mean(values):
    values = list(values)
    return fsum(values) / len(values) if values else None


def comparison(values, families):
    return {
        'means': {mode: {method: mean(scores) for method, scores in arms.items()} for mode, arms in values.items()},
        'calibrated_token_minus': {
            baseline: paired_interval(
                [left - right for left, right in zip(
                    values['calibrated']['token_rank'], values['calibrated'][baseline], strict=True)],
                families,
            )
            for baseline in ('question_only', 'position', 'pooled_rank')
        },
    }


def empty_scores():
    return {mode: {method: [] for method in METHODS} for mode in MODES}


def question_metrics(by_article, schedules, paths):
    nll, brier, first = empty_scores(), empty_scores(), empty_scores()
    layout_nll = defaultdict(empty_scores)
    families, observed_first = [], []
    for article, cases in sorted(by_article.items()):
        for case in sorted(cases, key=lambda row: row.identifier):
            stop_count = len(schedules[article])
            endpoints = tuple(ref.stop if ref is not None else stop_count for ref in case.references)
            families.append(case.family)
            observed_first.append(endpoints.count(0) / REFERENCE_COUNT)
            for method, arms in paths[case.identifier].items():
                for mode, path in arms.items():
                    arrival = path.arrival
                    reference_losses = [arrival.interval_nll[ref.stop] if ref is not None else path.censored_nll
                                        for ref in case.references]
                    nll[mode][method].append(mean(reference_losses))
                    brier[mode][method].append(mean((cdf - int(stop >= end)) ** 2 for end in endpoints
                                                    for stop, cdf in enumerate(arrival.cdf)))
                    first[mode][method].append(arrival.cdf[0])
                    for ref, loss in zip(case.references, reference_losses, strict=True):
                        if ref is not None:
                            layout_nll[ref.source_tag][mode][method].append(loss)
    observed = mean(observed_first)
    return {
        'interval_nll_nats': comparison(nll, families),
        'cumulative_brier': comparison(brier, families),
        'interval_nll_nats_by_non_null_source_tag': {
            tag: {
                'references': len(arms['raw']['position']),
                'means': {mode: {method: mean(scores) for method, scores in methods.items()}
                          for mode, methods in arms.items()},
            } for tag, arms in sorted(layout_nll.items())
        },
        'first_stop': {
            'observed_reference_fraction': observed,
            'mean_predicted': {mode: {method: mean(scores) for method, scores in arms.items()}
                               for mode, arms in first.items()},
            'mean_prediction_minus_observed': {mode: {method: mean(scores) - observed for method, scores in arms.items()}
                                               for mode, arms in first.items()},
        },
    }


def pair_metrics(by_article, schedules, paths):
    rank, families, pair_articles = empty_scores(), [], set()
    rejected = Counter()
    for article, cases in sorted(by_article.items()):
        for left, right in combinations(sorted(cases, key=lambda row: row.identifier), 2):
            if min(len(left.boxes), len(right.boxes)) < 2:
                rejected['fewer_than_two_non_null_per_question'] += 1
                continue
            early, late = sorted((left, right), key=lambda row: max(box.source_end for box in row.boxes))
            if max(box.source_end for box in early.boxes) > min(box.source_start for box in late.boxes):
                rejected['overlapping_source_windows'] += 1
                continue
            stop = max(box.stop for box in early.boxes)
            if schedules[article][stop] > min(box.start for box in late.boxes):
                rejected['no_certified_stop_before_late_start'] += 1
                continue
            families.append(early.family)
            pair_articles.add(article)
            for mode, arms in rank.items():
                for method, scores in arms.items():
                    left_signal = paths[early.identifier][method][mode].log_intensity[stop]
                    right_signal = paths[late.identifier][method][mode].log_intensity[stop]
                    scores.append(0.5 if left_signal == right_signal else float(left_signal > right_signal))
    return {
        'pairs': len(families), 'articles': len(pair_articles), 'article_families': len(set(families)),
        'excluded_pair_reasons': dict(rejected),
        'same_stop_accuracy': comparison(rank, families),
        'wins_ties_losses': {
            mode: {method: {label: scores.count(value) for label, value in (('wins', 1), ('ties', 0.5), ('losses', 0))}
                   for method, scores in arms.items()}
            for mode, arms in rank.items()
        },
    }


def run(assay_root, predictions_dir, outcomes, output):
    if output.exists():
        raise FileExistsError(output)
    plan, hashes, articles, queries, schedules, paths = read_forecasts(assay_root, predictions_dir)
    selected, outcome_rows = read_sealed_outcomes(outcomes, plan, queries, hashes)
    by_article, inventory = references_by_article(assay_root, selected, outcome_rows, articles, queries, schedules)
    questions = question_metrics(by_article, schedules, paths)
    pairs = pair_metrics(by_article, schedules, paths)
    primary_ci = questions['interval_nll_nats']['calibrated_token_minus']['question_only']['article_cluster_ci95']
    pair_ci = pairs['same_stop_accuracy']['calibrated_token_minus']['question_only']['article_cluster_ci95']
    result = {
        'scope': 'Arrival of the complete retrospectively annotated long-answer box for an externally supplied question.',
        'reference_rule': 'Five equally weighted original judgments per scored question. Null is right-censored after the final stop.',
        'pair_rule': 'Identical article revision and certified stop. Each question has at least two non-null boxes. All earlier boxes end before or at the beginning of every later box in source coordinates. No later box has begun at the stop. Rank log cumulative intensity with half credit for ties.',
        'bootstrap': {'draws': DRAWS, 'seed': SEED, 'unit': 'normalized Wikipedia title family', 'mean': 'equal questions or pairs'},
        'input_sha256': hashes,
        'inventory': {**inventory, 'articles': len(by_article), 'unforecasted_outcomes': len(outcome_rows) - len(queries),
                      'article_families': len({case.family for cases in by_article.values() for case in cases})},
        'per_question': questions,
        'same_article': pairs,
        'primary_superiority': primary_ci is not None and primary_ci[1] < 0,
        'secondary_same_stop_superiority': pair_ci is not None and pair_ci[0] > 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assay-root', 'predictions-dir', 'outcomes', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    run(**vars(parser.parse_args()))
