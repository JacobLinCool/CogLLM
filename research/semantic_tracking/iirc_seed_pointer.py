"""Forecast an earlier IIRC support span from an observed in-paragraph anchor."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from research.core_features.readback_score import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, paired_material_interval
from research.stage2.records import write_jsonl
from research.stage3.probe import _jsonl
from research.stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION, save_json
from .iirc_anchor_fit import WORD
from .iirc_prepare import digest


MAX_WORDS = 28
TOP_SPANS = 5
IOU_THRESHOLD = 0.5
RIDGE = 0.005
MAX_ITERATIONS = 60
FORM_FEATURES = (
    'log_words', 'log_characters', 'log_gap', 'relative_end', 'relative_end_squared',
    'initial_uppercase', 'uppercase_word_share', 'digit_share', 'punctuation_share',
    'anchor_word_share', 'anchor_word_count_ratio', 'touches_anchor',
)
METHODS = ('form', 'qwen')
QUERY_FIELDS = {'paragraph_id', 'question_id', 'anchor_start', 'anchor_end'}
COHORT_FIELDS = QUERY_FIELDS | {'article_family', 'support_start', 'support_end'}


@dataclass(frozen=True)
class Candidates:
    words: np.ndarray
    endpoints: np.ndarray
    form: np.ndarray
    contextual: np.ndarray
    anchor_product: np.ndarray

    def bounds(self):
        return np.column_stack((self.words[self.endpoints[:, 0], 0], self.words[self.endpoints[:, 1], 1]))


def prepare(prepared, audit, output):
    """Freeze split ownership and strip all outcome fields from held-out queries."""
    verified = json.loads(audit.read_text(encoding='utf-8'))
    cohorts, paragraphs = {}, {}
    for split in ('train', 'dev'):
        files = {name: prepared / split / f'{name}.jsonl' for name in ('stimuli', 'locked-evidence')}
        for name, path in files.items():
            if digest(path) != verified['files_sha256'][f'{split}/{name}.jsonl']:
                raise ValueError(f'{split}/{name} differs from the audited source')
        labels = []
        for row in _jsonl(files['locked-evidence']):
            if row['direction'] != 'earlier':
                continue
            labels.append({
                'paragraph_id': row['paragraph_id'], 'question_id': row['question_id'],
                'article_family': row['article_family'], 'anchor_start': row['anchor']['start'],
                'anchor_end': row['anchor']['end'], 'support_start': row['support_start'],
                'support_end': row['support_end'],
            })
        ids = {row['paragraph_id'] for row in labels}
        paragraphs[split] = [row for row in _jsonl(files['stimuli']) if row['id'] in ids]
        if len(labels) != verified['splits'][split]['eligible_earlier'] or len(paragraphs[split]) != len(ids):
            raise ValueError('The earlier support cohort does not match the audited inventory')
        cohorts[split] = labels
    for key in ('article_family', 'paragraph_id', 'question_id'):
        if {row[key] for row in cohorts['train']} & {row[key] for row in cohorts['dev']}:
            raise ValueError(f'Train and dev overlap by {key}')
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / 'stimuli.jsonl', paragraphs['train'] + paragraphs['dev'])
    write_jsonl(output / 'train.jsonl', cohorts['train'])
    write_jsonl(output / 'queries.jsonl', [{key: value for key, value in row.items() if key in QUERY_FIELDS}
                                           for row in cohorts['dev']])
    write_jsonl(output / 'locked-test.jsonl', cohorts['dev'])
    save_json(output / 'intake.json', {
        'prepared_audit_sha256': digest(audit),
        'retained': {split: {'questions': len(rows), 'paragraphs': len(paragraphs[split]),
                             'article_families': len({row['article_family'] for row in rows})}
                     for split, rows in cohorts.items()},
        'files': {path.name: digest(path) for path in sorted(output.glob('*.jsonl'))},
    })


def row_queries(path, texts, *, training):
    records = list(_jsonl(path))
    required = COHORT_FIELDS if training else QUERY_FIELDS
    if not records or len({row['question_id'] for row in records}) != len(records):
        raise ValueError('Expected unique question queries')
    for row in records:
        lo, hi = row['anchor_start'], row['anchor_end']
        text = texts[row['paragraph_id']]
        if row.keys() != required or any(type(pos) is not int for pos in (lo, hi)) or not 0 < lo < hi <= len(text):
            raise ValueError('Invalid observed anchor or forbidden outcome in prediction inputs')
        if training:
            begin, end = row['support_start'], row['support_end']
            if any(type(pos) is not int for pos in (begin, end)) or not 0 <= begin < end <= lo:
                raise ValueError('A training support must lie entirely before its observed anchor')
    return records


def build_candidates(text, offsets, projected, lo, hi, anchor_token):
    """Rank only lexical intervals wholly before the known anchor start."""
    words = np.asarray([match.span() for match in WORD.finditer(text[:lo])], dtype=np.int64).reshape(-1, 2)
    offsets = np.asarray(offsets)
    kept = np.flatnonzero((offsets[:, 0] < offsets[:, 1]) & (offsets[:, 1] <= lo))
    if not len(words) or not len(kept):
        return None
    starts, ends = offsets[kept].T
    if np.any(np.diff(starts) < 0):
        raise ValueError('Expected source-ordered offsets')
    last = np.searchsorted(starts, words[:, 1], side='left') - 1
    valid = (last >= 0) & (ends[np.maximum(last, 0)] > words[:, 0])
    words = words[valid]
    if not len(words):
        return None
    word_tokens = kept[last[valid]]
    if np.any(word_tokens >= anchor_token) or np.any(offsets[word_tokens, 1] > lo):
        raise ValueError('The prior pointer attempted to use states beyond the anchor start')

    widths = np.minimum(MAX_WORDS, len(words) - np.arange(len(words)))
    left = np.repeat(np.arange(len(words)), widths)
    right = left + np.arange(len(left)) - np.repeat(np.cumsum(widths) - widths, widths)
    begin, end = words[left, 0], words[right, 1]
    word_width, char_width = right - left + 1, end - begin
    gap = lo - end
    relative_end = end / lo
    caps = np.fromiter((text[start].isupper() for start in words[:, 0]), dtype=np.int64)
    uppercase = np.r_[0, caps.cumsum()]
    digit = np.r_[0, np.fromiter(map(str.isdigit, text[:lo]), dtype=np.int64).cumsum()]
    punctuation = np.r_[0, np.fromiter((not (c.isalnum() or c.isspace()) for c in text[:lo]), dtype=np.int64).cumsum()]
    anchor_words = {match[0].casefold() for match in WORD.finditer(text[lo:hi])}
    matches = np.r_[0, np.fromiter((text[a:b].casefold() in anchor_words for a, b in words), dtype=np.int64).cumsum()]
    overlap = matches[right + 1] - matches[left]
    form = np.column_stack((
        np.log1p(word_width) / np.log1p(MAX_WORDS), np.log1p(char_width) / 6, np.log1p(gap) / 7,
        relative_end, relative_end ** 2, caps[left], (uppercase[right + 1] - uppercase[left]) / word_width,
        (digit[end] - digit[begin]) / char_width, (punctuation[end] - punctuation[begin]) / char_width,
        overlap / word_width, overlap / max(len(anchor_words), 1), gap == 0,
    ))
    contextual = projected[word_tokens].astype(np.float64)
    # Scale a product of two unit-vector coordinates to the contextual feature scale.
    product = contextual * projected[anchor_token] * np.sqrt(projected.shape[1])
    return Candidates(words, np.column_stack((left, right)), form, contextual, product)


def support_iou(bounds, begin, end):
    shared = np.maximum(0, np.minimum(bounds[:, 1], end) - np.maximum(bounds[:, 0], begin))
    return shared / (bounds[:, 1] - bounds[:, 0] + end - begin - shared)


def certify(query, encoded, tokenizer):
    end = query['anchor_end']
    prefix = tokenizer.encode(encoded.text[:end], add_special_tokens=False)
    count = len(prefix.ids)
    if (not count or tuple(prefix.ids) != encoded.tokens[:count]
            or tuple(prefix.offsets) != encoded.offsets[:count]
            or max(pos for _, pos in encoded.offsets[:count]) != end):
        return None
    return count - 1


def project(states, projection):
    values = states @ projection
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms) | (norms <= 0)):
        raise ValueError('Invalid projected causal state')
    return values / norms


def pack_training(examples):
    forms, contexts, products, endpoints, starts, supervision = [], [], [], [], [], []
    word_count, candidate_count = 0, 0
    for item, positive in examples:
        starts.append(candidate_count)
        forms.append(item.form)
        contexts.append(item.contextual)
        products.append(item.anchor_product)
        endpoints.append(item.endpoints + word_count)
        supervision.append(positive)
        word_count += len(item.words)
        candidate_count += len(item.endpoints)
    if not starts:
        raise ValueError('No certified training examples with an overlapping candidate span')
    return (np.concatenate(forms), np.column_stack((np.concatenate(contexts), np.concatenate(products))),
            np.concatenate(endpoints), np.asarray(starts), np.concatenate(supervision))


def fit(packed, *, use_context):
    """Optimize conditional log mass of overlapping spans, equally per question."""
    from scipy.optimize import minimize

    forms, word_features, endpoints, starts, positive = packed
    widths = np.diff(np.r_[starts, len(forms)])
    left, right = endpoints.T
    form_width, word_width = forms.shape[1], word_features.shape[1]
    n = len(starts)
    coefficient_count = form_width + (2 * word_width if use_context else 0)

    def objective(weights):
        logits = forms @ weights[:form_width]
        if use_context:
            ends = word_features @ weights[form_width:].reshape(2, word_width).T
            logits += ends[left, 0] + ends[right, 1]
        maxima = np.maximum.reduceat(logits, starts)
        means = np.exp(logits - np.repeat(maxima, widths))
        marginals = np.where(positive, means, 0.)
        total = np.add.reduceat(means, starts)
        mass = np.add.reduceat(marginals, starts)
        loss = (np.log(total) - np.log(mass)).mean() + 0.5 * RIDGE * (weights @ weights)
        residual = (means / np.repeat(total, widths) - marginals / np.repeat(mass, widths)) / n
        gradient = forms.T @ residual
        if use_context:
            words = len(word_features)
            left_gradient = np.bincount(left, weights=residual, minlength=words) @ word_features
            right_gradient = np.bincount(right, weights=residual, minlength=words) @ word_features
            gradient = np.concatenate((gradient, left_gradient, right_gradient))
        return loss, gradient + RIDGE * weights

    result = minimize(objective, np.zeros(coefficient_count), jac=True, method='L-BFGS-B',
                      options={'maxiter': MAX_ITERATIONS, 'ftol': 1e-6, 'gtol': 1e-5})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or not (result.success or result.status == 1):
        raise RuntimeError(f'Support ranker optimization failed: {result.message}')
    return result.x, {'loss': float(result.fun), 'iterations': int(result.nit),
                      'converged': bool(result.success), 'max_abs_gradient': float(np.abs(result.jac).max())}


def rank(item, weights, *, use_context):
    logits = item.form @ weights[:len(FORM_FEATURES)]
    left, right = item.endpoints.T
    if use_context:
        context = np.column_stack((item.contextual, item.anchor_product))
        endpoints = context @ weights[len(FORM_FEATURES):].reshape(2, context.shape[1]).T
        logits += endpoints[left, 0] + endpoints[right, 1]
    mass = np.exp(logits - logits.max())
    probabilities = mass / mass.sum()
    bounds = item.bounds()
    selected = []
    for index in np.argsort(-logits, kind='stable'):
        begin, end = map(int, bounds[index])
        if any(begin < span['end'] and span['start'] < end for span in selected):
            continue
        selected.append({'start': begin, 'end': end, 'score': float(probabilities[index])})
        if len(selected) == TOP_SPANS:
            break
    return selected


def load_projection(anchors):
    seal = json.loads((anchors / 'fit-manifest.json').read_text(encoding='utf-8'))
    path = anchors / 'heads.npz'
    if seal['revision'] != REVISION or seal['heads_sha256'] != digest(path):
        raise ValueError('Existing IIRC projection checkpoint is not the frozen reference')
    with np.load(path, allow_pickle=False) as data:
        projection = data['projection']
    if projection.shape != (2048, 128) or projection.dtype != np.float32 or not np.isfinite(projection).all():
        raise ValueError('Expected the certified 2,048 × 128 projection')
    return projection


def verify_suffix(backbone, encoded, projected, query, candidates, projection):
    stop = query['anchor_end']
    anchor_token = certify(query, encoded, backbone.tokenizer)
    truncated = backbone.encode(encoded.text[:stop])
    if truncated.tokens != encoded.tokens[:anchor_token + 1] or truncated.offsets != encoded.offsets[:anchor_token + 1]:
        raise ValueError('Certified prefix failed independent causal replay')
    replay = project(truncated.states, projection)
    comparison = build_candidates(truncated.text, truncated.offsets, replay, query['anchor_start'], stop, anchor_token)
    if not np.array_equal(candidates.bounds(), comparison.bounds()) or not np.array_equal(candidates.form, comparison.form):
        raise RuntimeError('Earlier candidate form changed after suffix removal')
    delta = float(np.abs(projected[:anchor_token + 1] - replay).max())
    if delta > 0.003:
        raise RuntimeError('Earlier projected states changed after suffix removal')
    return {'question_id': query['question_id'], 'prefix_tokens': anchor_token + 1,
            'max_projected_absolute_difference': delta}


def forecast(model, anchors, stimuli, train, queries, output):
    from transformers import AutoTokenizer, __version__ as transformers_version
    from research.gradio_app.causal_plaba import CausalQwen

    if transformers_version != TRANSFORMERS_VERSION:
        raise ValueError(f'Expected the pinned Transformers {TRANSFORMERS_VERSION} tokenizer wrapper')
    source = list(_jsonl(stimuli))
    texts = {row['id']: row['text'] for row in source}
    if len(texts) != len(source) or any(row.keys() != {'id', 'text', 'stops'} for row in source):
        raise ValueError('Expected unique, plain paragraph inputs')
    training = row_queries(train, texts, training=True)
    held_out = row_queries(queries, texts, training=False)
    for key in ('question_id', 'paragraph_id'):
        if {row[key] for row in training} & {row[key] for row in held_out}:
            raise ValueError(f'Train and held-out overlap by {key}')
    projection = load_projection(anchors)
    clock = perf_counter()
    backbone = CausalQwen(model)
    production = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
    if not production.is_fast:
        raise ValueError('Expected the production tokenizer to provide native offset alignment')
    backbone.tokenizer = production.backend_tokenizer
    training_report = defaultdict(int)
    held_out_report = defaultdict(int)
    suffix_checks, examples, proposals = [], [], []
    chunk_forwards, total_tokens = 0, 0
    for labels, is_training, report in ((training, True, training_report), (held_out, False, held_out_report)):
        by_paragraph = defaultdict(list)
        for row in labels:
            by_paragraph[row['paragraph_id']].append(row)
        for index, (identifier, rows) in enumerate(by_paragraph.items(), 1):
            encoded = backbone.encode(texts[identifier])
            chunk_forwards += 1
            total_tokens += len(encoded.tokens)
            projected = project(encoded.states, projection)
            for row in rows:
                lo, hi = row['anchor_start'], row['anchor_end']
                stop = certify(row, encoded, backbone.tokenizer)
                item = build_candidates(encoded.text, encoded.offsets, projected, lo, hi, stop) if stop is not None else None
                excluded = ('uncertified_anchor_end' if stop is None else 'no_earlier_candidates') if item is None else None
                if not is_training:
                    proposals.append((row, item, excluded))
                    report[excluded or 'scored'] += 1
                    continue
                if excluded:
                    report[excluded] += 1
                    continue
                if len(suffix_checks) < 3 and hi < len(encoded.text):
                    suffix_checks.append(verify_suffix(backbone, encoded, projected, row, item, projection))
                positive = support_iou(item.bounds(), row['support_start'], row['support_end']) >= IOU_THRESHOLD
                if not positive.any():
                    report['no_compatible_training_span'] += 1
                    continue
                report['scored'] += 1
                examples.append((item, positive))
            if index % 100 == 0:
                print('train' if is_training else 'held_out', index, len(by_paragraph), flush=True)
    if len(suffix_checks) != 3:
        raise RuntimeError('Expected three independent truncated-prefix positive controls')
    packed = pack_training(examples)
    treatments = {name: fit(packed, use_context=(name == 'qwen')) for name in METHODS}
    forecasts = []
    for row, item, excluded in proposals:
        forecasts.append({
            **row, 'unscored_reason': excluded, 'candidate_words': item.words.tolist() if item is not None else [],
            'methods': {name: rank(item, treatment[0], use_context=(name == 'qwen')) if item is not None else []
                        for name, treatment in treatments.items()},
        })

    output.mkdir(parents=True, exist_ok=False)
    prediction_path, heads_path = output / 'predictions.jsonl', output / 'heads.npz'
    write_jsonl(prediction_path, forecasts)
    np.savez(heads_path, form=treatments['form'][0], qwen=treatments['qwen'][0], projection=projection)
    save_json(output / 'manifest.json', {
        'model': MODEL, 'revision': REVISION, 'dtype': 'float32', 'device': str(backbone.mx.default_device()),
        'tokenizer_wrapper': transformers_version, 'ordinary_paragraph_forwards': chunk_forwards,
        'ordinary_nonpadding_tokens': total_tokens, 'separate_truncated_replays': suffix_checks,
        'elapsed_s': perf_counter() - clock,
        'source_sha256': {path.name: digest(path) for path in (stimuli, train, queries)},
        'source_projection_sha256': digest(anchors / 'heads.npz'),
        'predictions_sha256': digest(prediction_path), 'heads_sha256': digest(heads_path),
        'training_disposition': dict(training_report), 'held_out_disposition': dict(held_out_report),
        'treatments': {name: result[1] for name, result in treatments.items()},
        'preregistered': {'max_lexical_runs_per_span': MAX_WORDS, 'top_spans': TOP_SPANS, 'min_iou': IOU_THRESHOLD,
                          'ridge': RIDGE, 'optimizer_max_iterations': MAX_ITERATIONS,
                          'form_features': list(FORM_FEATURES), 'unit_weight': 'authored question',
                          'primary': 'Top-one exact-character IoU≥0.5, qwen minus form; include unscored cases as misses',
                          'superiority_gate': 'Lower 95% paired article bootstrap endpoint above zero'},
    })


def score(holdout, results, output):
    manifest = json.loads((results / 'manifest.json').read_text(encoding='utf-8'))
    path = results / 'predictions.jsonl'
    if digest(path) != manifest['predictions_sha256'] or digest(results / 'heads.npz') != manifest['heads_sha256']:
        raise ValueError('The prediction/readout package does not match its original seal')
    predictions = {row['question_id']: row for row in _jsonl(path)}
    truth = list(_jsonl(holdout))
    if len(predictions) != len(truth) or predictions.keys() != {row['question_id'] for row in truth}:
        raise ValueError('The sealed predictions do not cover all held-out questions')
    entries = defaultdict(list)
    groups = []
    candidate_coverage = []
    for gold in truth:
        query = predictions[gold['question_id']]
        if any(query[field] != gold[field] for field in QUERY_FIELDS):
            raise ValueError('Sealed observed anchor differs from gold query identity')
        lo = gold['anchor_start']
        begin, end = gold['support_start'], gold['support_end']
        if not 0 <= begin < end <= lo:
            raise ValueError('A held-out support must lie before the anchor start')
        words = np.asarray(query['candidate_words'], dtype=np.int64).reshape(-1, 2)
        match = False
        if len(words):
            widths = np.minimum(MAX_WORDS, len(words) - np.arange(len(words)))
            left = np.repeat(np.arange(len(words)), widths)
            right = left + np.arange(len(left)) - np.repeat(np.cumsum(widths) - widths, widths)
            candidate_bounds = np.column_stack((words[left, 0], words[right, 1]))
            match = bool(np.any(support_iou(candidate_bounds, begin, end) >= IOU_THRESHOLD))
        candidate_coverage.append(float(match))
        groups.append({'group_id': gold['article_family']})
        for name in METHODS:
            predicted = query['methods'][name]
            spans = np.asarray([(row['start'], row['end']) for row in predicted], dtype=np.int64).reshape(-1, 2)
            if len(spans) > TOP_SPANS or np.any(spans[:, 0] < 0) or np.any(spans[:, 0] >= spans[:, 1]) or np.any(spans[:, 1] > lo):
                raise ValueError('A prediction is outside the earlier candidate horizon')
            overlap = support_iou(spans, begin, end)
            top = float(overlap[0]) if len(overlap) else 0.
            entries[name].append((float(top >= IOU_THRESHOLD), top, float(np.any(overlap >= IOU_THRESHOLD))))
    methods = {name: np.asarray(values) for name, values in entries.items()}
    contrast = methods['qwen'] - methods['form']
    endpoints = ('top1_hit_at_half_iou', 'top1_mean_iou', 'top5_hit_at_half_iou')
    differences = {endpoint: paired_material_interval(contrast[:, index], groups)
                   for index, endpoint in enumerate(endpoints)}
    primary = differences[endpoints[0]]
    save_json(output, {
        'population': 'Official IIRC dev earlier support labels, including uncertified anchors as misses',
        'question_count': len(truth), 'article_families': len({row['article_family'] for row in truth}),
        'candidate_coverage_at_half_iou': float(np.mean(candidate_coverage)),
        'methods': {name: dict(zip(endpoints, values.mean(axis=0).tolist(), strict=True)) for name, values in methods.items()},
        'qwen_minus_form': differences, 'article_cluster_bootstrap': {'seed': BOOTSTRAP_SEED, 'replicates': BOOTSTRAP_DRAWS},
        'passed_preregistered_superiority': primary['descriptive_paired_material_95_ci'][0] > 0,
        'sealed_predictions_sha256': digest(path), 'locked_eval_sha256': digest(holdout),
        'measurement_scope': 'Answer-worker annotated support occurring before an observed anchor, no reader timeline',
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    for name, parameters in (
        ('prepare', ('prepared', 'audit', 'output')),
        ('forecast', ('model', 'anchors', 'stimuli', 'train', 'queries', 'output')),
        ('score', ('holdout', 'results', 'output')),
    ):
        sub = subcommands.add_parser(name)
        for parameter in parameters:
            sub.add_argument(f'--{parameter}', type=Path, required=True)
    options = vars(parser.parse_args())
    {'prepare': prepare, 'forecast': forecast, 'score': score}[options.pop('command')](**options)


if __name__ == '__main__':
    main()
