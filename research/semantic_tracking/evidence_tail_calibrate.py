"""Fit and score monotone evidence-arrival calibrators on a single CPU thread."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
from math import log
import os
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from scipy.optimize import minimize

from research.semantic_tracking.expmrc_forecast import HEADS_SHA256
from research.semantic_tracking.expmrc_score import DRAWS, METHODS, SEED, TIE_ATOL, digest, forecast_inputs, indexed_rows, paired_interval


CALIBRATION_SHA256 = 'efab56f7379a00b6b468a82c5a0ac8ad34aa8fe9d1189f434e995bd27bf1f944'
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATION = ROOT / 'results' / 'evidence-calibration-comparison.json'
DEFAULT_HEADS = ROOT / 'results' / 'qasper-factorial-run' / 'heads.npz'
DEFAULT_PROTOCOL = ROOT / 'semantic_tracking' / 'qasper-evidence-tail.md'
MODES = ('raw', 'expmrc_frozen', 'clock', 'clock_plateau')
FIT_BOUNDS = ((-12, 12), (log(0.25), log(4)), (-12, 12))
TRUTH_FIELDS = {'id', 'article_id', 'refs', 'annotation_count', 'excluded'}
ARTICLE_METADATA_FIELDS = {'id', 'source_paper_id', 'official_split', 'title', 'title_family', 'arxiv',
                           'body_paragraphs', 'text_characters', 'input_tokens_with_questions'}
LOG_MAX = log(np.finfo(np.float64).max) - 2


@dataclass(frozen=True)
class PathLayout:
    log_left: np.ndarray
    log_right: np.ndarray
    log_ratio: np.ndarray

    @classmethod
    def from_hazards(cls, hazards):
        p = np.asarray(hazards, dtype=np.float64)
        if p.ndim != 1 or len(p) == 0 or not np.all((p > 0) & (p < 1)):
            raise ValueError('Expected a nonempty path of finite interior hazards')
        increments = -np.log1p(-p)
        intensity = np.cumsum(increments)
        if not np.all(np.isfinite(intensity)) or not np.all(np.diff(intensity) > 0):
            raise ValueError('Raw cumulative intensity must increase strictly and stay finite')
        right = np.log(intensity)
        left = np.concatenate(([-np.inf], right[:-1]))
        ratio = np.concatenate(([-np.inf], -np.log1p(increments[1:] / intensity[:-1])))
        return cls(left, right, ratio)


@dataclass(frozen=True)
class ReferenceCase:
    identifier: str
    article: str
    bins: tuple[int | None, ...]
    ends: tuple[int | None, ...]


@dataclass(frozen=True)
class ProbabilityPath:
    ordering: np.ndarray
    cdf: np.ndarray
    log_event: np.ndarray
    log_tail: float


def mixture_weights(estimate):
    if len(estimate) == 2:
        return -np.inf, 0.0
    logit = estimate[2]
    return -np.logaddexp(0.0, -logit), -np.logaddexp(0.0, logit)


def transformed_log_intensity(estimate, log_intensity):
    estimate = np.asarray(estimate, dtype=np.float64)
    if estimate.shape not in ((2,), (3,)) or not np.isfinite(estimate).all():
        raise ValueError('Expected finite clock parameters and optional plateau logit')
    a, log_b = estimate[:2]
    if abs(log_b) >= LOG_MAX:
        raise ValueError('Clock shape is outside the representable positive range')
    b = np.exp(log_b)
    with np.errstate(over='ignore', invalid='ignore'):
        result = a + b * log_intensity
    if np.any(result > LOG_MAX) or np.any(np.isnan(result)) or np.any(np.isneginf(result) & np.isfinite(log_intensity)):
        raise ValueError('Transformed cumulative intensity exceeds the finite range')
    return result


def log_interval_mass(estimate, left, right, ratio):
    left_intensity = np.exp(transformed_log_intensity(estimate, left))
    log_right = transformed_log_intensity(estimate, right)
    b = np.exp(estimate[1])
    log_increment = log_right + np.log(-np.expm1(b * ratio))
    # For tiny increments the log mass equals the log increment to floating precision.
    # Keeping this branch in log space also handles masses below the representable probability range.
    stable = np.maximum(log_increment, -36)
    elapsed = np.where(log_increment < -36, log_increment, np.log(-np.expm1(-np.exp(stable))))
    return mixture_weights(estimate)[1] - left_intensity + elapsed


def log_terminal_mass(estimate, log_right):
    log_pi, log_survivor_weight = mixture_weights(estimate)
    return np.logaddexp(log_pi, log_survivor_weight - np.exp(transformed_log_intensity(estimate, log_right)))


def probability_path(estimate, layout):
    ordering = transformed_log_intensity(estimate, layout.log_right)
    log_weight = mixture_weights(estimate)[1]
    cdf = np.exp(log_weight) * -np.expm1(-np.exp(ordering))
    event = log_interval_mass(estimate, layout.log_left, layout.log_right, layout.log_ratio)
    tail = float(log_terminal_mass(estimate, layout.log_right[-1]))
    cumulative = np.cumsum(np.exp(event))
    if (not np.all(np.isfinite(event)) or not np.isfinite(tail) or not np.all((cdf >= 0) & (cdf <= 1))
            or np.any(np.diff(cdf) < 0) or not np.allclose(cumulative, cdf, rtol=0, atol=1e-12)
            or not np.isclose(cumulative[-1] + np.exp(tail), 1, rtol=0, atol=1e-12)):
        raise ValueError('Intervals and terminal survivor mass must form one proper arrival distribution')
    return ProbabilityPath(ordering, cdf, event, tail)


def training_data(cases, layouts):
    rows = [[], [], [], [], [], []]
    for case in cases:
        layout = layouts[case.identifier]
        weight = 1 / (len(cases) * len(case.bins))
        for stop in case.bins:
            if stop is None:
                rows[4].append(layout.log_right[-1])
                rows[5].append(weight)
            else:
                rows[0].append(layout.log_left[stop])
                rows[1].append(layout.log_right[stop])
                rows[2].append(layout.log_ratio[stop])
                rows[3].append(weight)
    if not rows[0] or not rows[4]:
        raise ValueError('Both observed evidence and explicit null training references are required')
    return tuple(map(np.asarray, rows))


def fit_calibrators(cases, layouts):
    left, right, ratio, event_weights, terminal, null_weights = training_data(cases, layouts)

    def objective(estimate):
        events = log_interval_mass(estimate, left, right, ratio)
        nulls = log_terminal_mass(estimate, terminal)
        return float(-np.sum(event_weights * events) - np.sum(null_weights * nulls))

    def fit(initial):
        return minimize(objective, initial, method='L-BFGS-B', bounds=FIT_BOUNDS[:len(initial)],
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})

    clock = fit((0.0, 0.0))
    if not clock.success or not np.isfinite(clock.fun):
        raise RuntimeError(f'The train-only clock fit failed: {clock.message}')
    starts = [(*clock.x, value) for value in (-10.0, -2.0, 0.0)]
    mixture_fits = [fit(start) for start in starts]
    accepted = [result for result in mixture_fits if result.success and np.isfinite(result.fun)]
    if not accepted:
        raise RuntimeError('No deterministic train-only plateau start converged')
    plateau = min(accepted, key=lambda result: result.fun)
    if plateau.fun > clock.fun + 1e-5:
        raise RuntimeError('The fitted plateau objective failed to recover the near-zero plateau boundary')
    return {name: {'estimate': result.x.tolist(), 'train_nll_nats': float(result.fun), 'iterations': result.nit,
                   'a': float(result.x[0]), 'b': float(np.exp(result.x[1])),
                   'pi': float(np.exp(mixture_weights(result.x)[0]))}
            for name, result in (('clock', clock), ('clock_plateau', plateau))}


def read_references(path, plan, articles, queries, schedules):
    sha = digest(path)
    if sha != plan['file_sha256'][path.name]:
        raise ValueError(f'{path.name} differs from its sealed QASPER annotation file')
    entries = indexed_rows(path, TRUTH_FIELDS)
    cases, parts, exclusions = [], set(), Counter()
    for identifier, row in sorted(entries.items()):
        article = row['article_id']
        refs, missing = row['refs'], row['excluded']
        if (identifier not in queries or article != queries[identifier]['article_id'] or not isinstance(refs, list)
                or not isinstance(missing, dict) or any(type(count) is not int or count <= 0 for count in missing.values())
                or type(row['annotation_count']) is not int or row['annotation_count'] != len(refs) + sum(missing.values())):
            raise ValueError(f'A question has inconsistent article or reference counts in {path.name}')
        parts.add(article)
        exclusions.update(missing)
        length = len(articles[article]['text'])
        ends = []
        for ref in refs:
            if not isinstance(ref, dict) or ref.keys() != {'end'}:
                raise ValueError('Expected a sole evidence end or explicit null per reference')
            end = ref['end']
            if end is not None and (type(end) is not int or not 0 < end <= length):
                raise ValueError('Evidence end is outside its original Unicode article')
            ends.append(end)
        if ends:
            bins = tuple(None if end is None else bisect_left(schedules[article], end) for end in ends)
            cases.append(ReferenceCase(identifier, article, bins, tuple(ends)))
    if not cases:
        raise ValueError(f'No eligible questions in {path.name}')
    return cases, {'questions': len(entries), 'scored_questions': len(cases), 'papers': len(parts),
                   'unusable_references_by_reason': dict(exclusions)}, parts, entries.keys(), sha


def frozen_parameters(path):
    if digest(path) != CALIBRATION_SHA256:
        raise ValueError('The transferred ExpMRC calibration differs from its frozen artifact')
    calibrated = json.loads(path.read_text(encoding='utf-8'))['frozen_exp_mrc_transfer_weights']['methods']
    if calibrated.keys() != set(METHODS):
        raise ValueError('The ExpMRC artifact must cover each frozen readout exactly once')
    return {method: (row['a'], log(row['b'])) for method, row in calibrated.items()}


def split_inventory(root, plan, articles):
    path = root / 'article-metadata.jsonl'
    sha = digest(path)
    if sha != plan['file_sha256'][path.name]:
        raise ValueError('The official split metadata differs from the sealed assay')
    metadata = indexed_rows(path, ARTICLE_METADATA_FIELDS)
    if (metadata.keys() != articles.keys() or len({row['source_paper_id'] for row in metadata.values()}) != len(articles)
            or len({row['title_family'] for row in metadata.values()}) != len(articles)
            or set(row['official_split'] for row in metadata.values()) != {'train', 'test'}
            or any(row['text_characters'] != len(articles[key]['text']) for key, row in metadata.items())):
        raise ValueError('Each sealed article must have unique source/title identity and one matching official split')
    return {split: {identifier for identifier, row in metadata.items() if row['official_split'] == split}
            for split in ('train', 'test')}, sha


def subset_scores(cases, paths, subset):
    nll = {mode: [] for mode in MODES}
    brier = {mode: [] for mode in MODES}
    terminal = {mode: [] for mode in MODES}
    groups, references = [], 0
    for case in cases:
        stops = [stop for stop in case.bins if subset == 'all' or (stop is None) == (subset == 'null')]
        if not stops:
            continue
        groups.append(case.article)
        references += len(stops)
        for mode, path in paths[case.identifier].items():
            failures = [-path.log_tail if stop is None else -path.log_event[stop] for stop in stops]
            nll[mode].append(float(np.mean(failures)))
            expected = np.zeros((len(stops), len(path.cdf)))
            for index, stop in enumerate(stops):
                if stop is not None:
                    expected[index, stop:] = 1
            brier[mode].append(float(np.mean((path.cdf - expected)**2)))
            terminal[mode].append(float(np.exp(path.log_tail)))
    scores = {name: {mode: float(np.mean(values)) if values else None for mode, values in metric.items()}
              for name, metric in (('nll_nats', nll), ('cumulative_brier', brier), ('mean_terminal_residual', terminal))}
    if groups:
        differences = [candidate - clock for candidate, clock in zip(nll['clock_plateau'], nll['clock'], strict=True)]
        scores['plateau_minus_clock_nll_nats'] = paired_interval(differences, groups)
    return {'questions': len(groups), 'papers': len(set(groups)), 'references': references, **scores}


def verify_orders(cases, paths):
    by_paper = defaultdict(list)
    for case in cases:
        by_paper[case.article].append(case)
    order_changes, cdf_changes, registered_changes = (dict.fromkeys(MODES[1:], 0) for _ in range(3))
    total_stops = registered_pairs = 0
    for paper in sorted(by_paper):
        for left, right in combinations(by_paper[paper], 2):
            p, q = paths[left.identifier], paths[right.identifier]
            raw_order = np.sign(p['raw'].ordering - q['raw'].ordering)
            raw_delta = p['raw'].cdf - q['raw'].cdf
            raw_probability_order = np.where(np.abs(raw_delta) <= TIE_ATOL, 0.0, np.sign(raw_delta))
            total_stops += len(raw_order)
            cutoff = None
            if None not in left.bins and None not in right.bins:
                if max(left.bins) < min(right.bins):
                    cutoff = max(left.bins)
                elif max(right.bins) < min(left.bins):
                    cutoff = max(right.bins)
            registered_pairs += cutoff is not None
            for mode in MODES[1:]:
                order = np.sign(p[mode].ordering - q[mode].ordering)
                delta = p[mode].cdf - q[mode].cdf
                probability_order = np.where(np.abs(delta) <= TIE_ATOL, 0.0, np.sign(delta))
                order_changes[mode] += int(np.count_nonzero(order != raw_order))
                cdf_changes[mode] += int(np.count_nonzero(probability_order != raw_probability_order))
                registered_changes[mode] += int(cutoff is not None and order[cutoff] != raw_order[cutoff])
    if any(order_changes.values()):
        raise RuntimeError('A calibration changed the same-stop ordering of finite arrival scores')
    return {'all_question_pair_stops': total_stops, 'all_strictly_ordered_event_pairs': registered_pairs,
            'intensity_order_changes': order_changes, 'strict_pair_intensity_order_changes': registered_changes,
            'cdf_tolerance_order_changes': cdf_changes, 'cdf_comparison_tolerance': TIE_ATOL}


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def run(assay_root, predictions_dir, output, fit_receipt, protocol, heads, expmrc_calibration):
    if output.exists():
        raise FileExistsError(output)
    fit_receipt = fit_receipt or output.with_name(f'{output.stem}-fit.json')
    plan, hashes, articles, queries, schedules, paths = forecast_inputs(assay_root, predictions_dir, protocol, heads)
    del paths
    seal = json.loads((predictions_dir / 'seal.json').read_text(encoding='utf-8'))
    extraction_sha = digest(predictions_dir / 'extraction.json')
    if (hashes['heads_sha256'] != HEADS_SHA256 or seal.get('extraction_sha256') != extraction_sha
            or seal.get('source_code_sha256') != digest(Path(__file__).with_name('expmrc_forecast.py'))):
        raise ValueError('The sealed model head, extraction, or shared forecast source does not match the frozen assay')
    article_splits, metadata_sha = split_inventory(assay_root, plan, articles)
    forecasts = indexed_rows(predictions_dir / 'predictions.jsonl', {'id', 'article_id', 'stops', 'methods'})
    prior = frozen_parameters(expmrc_calibration)
    train, train_summary, train_papers, train_ids, train_sha = read_references(
        assay_root / 'development.jsonl', plan, articles, queries, schedules,
    )
    if train_papers != article_splits['train'] or train_ids != {key for key, row in queries.items()
                                                               if row['article_id'] in train_papers}:
        raise ValueError('Fit outcomes must cover exactly the sealed official training questions')
    hashes.update(training_sha256=train_sha, frozen_expmrc_calibration_sha256=CALIBRATION_SHA256,
                  article_metadata_sha256=metadata_sha, extraction_sha256=extraction_sha, calibrator_sha256=digest(Path(__file__)))
    layouts = {method: {identifier: PathLayout.from_hazards(row['methods'][method])
                        for identifier, row in forecasts.items()} for method in METHODS}
    receipt = {'source_hashes': hashes, 'development': train_summary, 'weighting': 'equal questions, equal within-question references',
               'bounds': FIT_BOUNDS, 'fitted_methods': {method: fit_calibrators(train, layouts[method]) for method in METHODS}}
    write_new(fit_receipt, receipt)
    fit_sha = digest(fit_receipt)

    holdout, test_summary, test_papers, test_ids, test_sha = read_references(
        assay_root / 'holdout.jsonl', plan, articles, queries, schedules,
    )
    if (test_papers != article_splits['test'] or train_ids & test_ids or train_papers & test_papers
            or train_ids | test_ids != queries.keys()):
        raise ValueError('Train and held-out QASPER articles and questions must be disjoint and cover the sealed forecast')

    metrics = {}
    for method in METHODS:
        parameters = {'raw': (0.0, 0.0), 'expmrc_frozen': prior[method],
                      **{name: row['estimate'] for name, row in receipt['fitted_methods'][method].items()}}
        transformed = {case.identifier: {name: probability_path(estimate, layouts[method][case.identifier])
                                         for name, estimate in parameters.items()} for case in holdout}
        metrics[method] = {subset: subset_scores(holdout, transformed, subset) for subset in ('all', 'event', 'null')}
        metrics[method]['same_stop_order'] = verify_orders(holdout, transformed)
    primary = metrics['token_rank']['all']['plateau_minus_clock_nll_nats']
    bounds = primary['article_cluster_ci95']
    write_new(output, {
        'scope': 'Externally supplied question and complete annotated evidence arrival, including explicit null votes',
        'holdout': test_summary, 'methods': metrics, 'primary': {
            'contrast': 'token_rank clock_plateau minus clock NLL in nats', **primary,
            'upper_ci95_below_zero': bounds is not None and bounds[1] < 0},
        'bootstrap': {'draws': DRAWS, 'seed': SEED, 'group': 'paper', 'weighting': 'equal scored questions'},
        'input_sha256': {**hashes, 'holdout_sha256': test_sha, 'fit_receipt_sha256': fit_sha},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('assay-root', 'predictions-dir', 'output'):
        parser.add_argument(f'--{flag}', required=True, type=Path)
    parser.add_argument('--fit-receipt', type=Path)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--heads', type=Path, default=DEFAULT_HEADS)
    parser.add_argument('--expmrc-calibration', type=Path, default=DEFAULT_CALIBRATION)
    run(**vars(parser.parse_args()))
